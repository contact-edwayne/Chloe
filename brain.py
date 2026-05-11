"""Chloe's brain — three-layer memory (facts/wiki/episodic).

Drop this file at C:\\Chloe\\brain.py. Import from jarvis.py.

Pattern: Karpathy's wiki-as-memory + ChatGPT-style flat user facts + episodic log.
Schema lives in C:\\Chloe\\brain\\CHLOE_BRAIN.md and co-evolves with use.

Routing rule (enforced via the llm_call mode parameter):
  - heavy: ingest, lint, page updates, extraction. Use Groq (or strongest model).
  - light: query, fact_add. Local Ollama is fine.
"""

import re
import json
import datetime
from pathlib import Path
from typing import Callable, Optional

# llm_call signature: (prompt: str, mode: 'heavy' | 'light') -> str
LlmCall = Callable[[str, str], str]


class Brain:
    """Memory operations bound to a single brain root directory."""

    def __init__(self, root, llm_call: LlmCall):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Brain root not found: {self.root}")
        self.llm = llm_call
        self.facts_dir = self.root / 'facts'
        self.wiki_dir = self.root / 'wiki'
        self.episodic_dir = self.root / 'episodic'
        self.raw_dir = self.root / 'raw'
        self.overviews_dir = self.root / 'overviews'
        self.schema_path = self.root / 'CHLOE_BRAIN.md'
        for d in (self.facts_dir, self.wiki_dir, self.episodic_dir, self.raw_dir,
                  self.overviews_dir, self.wiki_dir / 'entities',
                  self.wiki_dir / 'concepts', self.wiki_dir / 'sources'):
            d.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # File ops (path-safe, scoped to brain root)
    # ========================================================================

    def _safe_path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"Path escapes brain root: {rel}")
        return p

    def read(self, rel: str) -> str:
        return self._safe_path(rel).read_text(encoding='utf-8')

    def exists(self, rel: str) -> bool:
        return self._safe_path(rel).exists()

    def write(self, rel: str, content: str) -> None:
        p = self._safe_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')

    def edit(self, rel: str, old: str, new: str) -> None:
        p = self._safe_path(rel)
        text = p.read_text(encoding='utf-8')
        if old not in text:
            raise ValueError(f"Edit target not found in {rel}")
        p.write_text(text.replace(old, new, 1), encoding='utf-8')

    def list_dir(self, rel: str = '') -> list:
        p = self._safe_path(rel)
        return sorted(str(c.relative_to(self.root)) for c in p.iterdir())

    def grep(self, pattern: str, glob: str = '**/*.md') -> list:
        regex = re.compile(pattern)
        results = []
        for p in self.root.glob(glob):
            try:
                for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
                    if regex.search(line):
                        results.append((str(p.relative_to(self.root)), i, line))
            except (UnicodeDecodeError, OSError):
                continue
        return results

    # ========================================================================
    # Boot context (load into system prompt at conversation start)
    # ========================================================================

    def boot_context(self) -> str:
        """Schema + all facts, ready to inject into Chloe's system prompt."""
        parts = []
        if self.schema_path.exists():
            parts.append('=== Chloe Brain Schema ===')
            parts.append(self.schema_path.read_text(encoding='utf-8'))
        if self.facts_dir.exists():
            facts = []
            for fact_file in sorted(self.facts_dir.glob('*.md')):
                if fact_file.name == 'MEMORY.md':
                    continue
                facts.append(f'--- {fact_file.stem} ---')
                facts.append(fact_file.read_text(encoding='utf-8'))
            if facts:
                parts.append('\n=== User Facts ===')
                parts.extend(facts)
        return '\n'.join(parts)

    def facts_only(self) -> str:
        """Just the user facts, no schema. Use this for lighter context loads."""
        if not self.facts_dir.exists():
            return ''
        out = []
        for fact_file in sorted(self.facts_dir.glob('*.md')):
            if fact_file.name == 'MEMORY.md':
                continue
            out.append(fact_file.read_text(encoding='utf-8'))
        return '\n\n'.join(out)

    # ========================================================================
    # Operations
    # ========================================================================

    def ingest(self, source_path, dry_run: bool = False) -> dict:
        """Ingest a raw source into the wiki.

        Returns dict with keys: slug, tldr, entities_touched, concepts_touched.
        """
        src = Path(source_path)
        if not src.is_absolute():
            src = self.raw_dir / src
        if not src.exists():
            # Auto-append .md if the bare slug was passed
            if src.suffix != '.md':
                alt = src.with_suffix('.md')
                if alt.exists():
                    src = alt
            if not src.exists():
                raise FileNotFoundError(f"Source not found: {src}")

        source_title = src.stem.replace('_', ' ').replace('-', ' ').title()
        source_text = src.read_text(encoding='utf-8', errors='replace')[:50000]

        # 1. Extract structure from source
        extraction = self._extract(source_title, source_text)

        # 1b. Dry-run short-circuit — compute would-create/would-update
        # without writing anything. Catches pollution before it lands.
        if dry_run:
            entities_status = []
            for entity in extraction.get('entities', [])[:10]:
                entity_slug = self._slug(entity)
                status = 'UPDATE' if self.exists(f'wiki/entities/{entity_slug}.md') else 'CREATE'
                entities_status.append((entity_slug, status))
            concepts_status = []
            for concept in extraction.get('concepts', [])[:10]:
                concept_slug = self._slug(concept)
                status = 'UPDATE' if self.exists(f'wiki/concepts/{concept_slug}.md') else 'CREATE'
                concepts_status.append((concept_slug, status))
            return {
                'dry_run': True,
                'slug': self._slug(src.stem),
                'tldr': extraction.get('tldr', ''),
                'key_points': extraction.get('key_points', []),
                'entities_status': entities_status,
                'concepts_status': concepts_status,
            }

        # 2. Source page
        slug = self._slug(src.stem)
        self.write(f'wiki/sources/{slug}.md', self._render_source_page(slug, source_title, extraction))

        # 3. Entity pages (cap at 10 to bound cost)
        entities_touched = []
        for entity in extraction.get('entities', [])[:10]:
            entity_slug = self._slug(entity)
            rel = f'wiki/entities/{entity_slug}.md'
            existing = self.read(rel) if self.exists(rel) else None
            new_content = self._update_page('entity', entity, existing, extraction, slug, source_title, source_text)
            new_content = self._validate_and_clean_page(new_content)
            if new_content is None:
                continue
            self.write(rel, new_content)
            entities_touched.append(entity_slug)

        # 4. Concept pages
        concepts_touched = []
        for concept in extraction.get('concepts', [])[:10]:
            concept_slug = self._slug(concept)
            rel = f'wiki/concepts/{concept_slug}.md'
            existing = self.read(rel) if self.exists(rel) else None
            new_content = self._update_page('concept', concept, existing, extraction, slug, source_title, source_text)
            new_content = self._validate_and_clean_page(new_content)
            if new_content is None:
                continue
            self.write(rel, new_content)
            concepts_touched.append(concept_slug)

        # 5. Index + log
        self._update_index(slug, source_title, entities_touched, concepts_touched)
        self._log('ingest', source_title)

        return {
            'slug': slug,
            'tldr': extraction.get('tldr', ''),
            'entities_touched': entities_touched,
            'concepts_touched': concepts_touched,
        }

    def query(self, question: str) -> str:
        """Answer a question against the wiki."""
        try:
            index = self.read('wiki/index.md')
        except FileNotFoundError:
            return "Wiki index missing — nothing to query yet."

        # Step 1: deterministic keyword match. For any question word that
        # appears as a part of a page slug (underscore-split), include that
        # page. This bypasses Ollama selection's reliability problems for
        # the common case where the question names the topic directly.
        pages = self._keyword_select(question, index)

        # Step 2: only ask Ollama to select if keyword match found nothing.
        if not pages:
            selection_prompt = (
                f"You're answering a question against Chloe's wiki.\n\n"
                f"Question: {question}\n\n"
                f"Wiki index:\n---\n{index}\n---\n\n"
                f'Return a JSON array of up to 5 page paths (relative to wiki/, e.g. '
                f'"entities/foo.md") most likely to answer this question. '
                f'Return ONLY the JSON array.'
            )
            try:
                pages = self._json_call(selection_prompt, 'light')
                if not isinstance(pages, list):
                    pages = []
            except Exception:
                pages = []

        contexts = []
        for p in pages[:5]:
            try:
                contexts.append(f'## {p}\n\n{self.read(f"wiki/{p}")}')
            except (FileNotFoundError, ValueError):
                continue

        if not contexts:
            self._log('query', question[:80] + ' (no matching pages)')
            return "Wiki had no pages relevant to that question."

        # Answer step routes to HEAVY (Groq). Light models routinely ignore
        # the "use only wiki content" constraint and pattern-complete from
        # training data, producing confident hallucinations even when the
        # right page is in context. Heavy models follow constraints reliably.
        # Selection above stays light because picking paths from a markdown
        # index is a much simpler task that local models handle fine.
        answer_prompt = (
            f"Answer the question using STRICTLY the wiki content below. "
            f"Do NOT use prior knowledge, training data, or general world knowledge. "
            f"If the wiki content is silent on the question or any part of it, "
            f"say so explicitly — do not fill gaps from outside knowledge. "
            f"Cite pages used as [[page_name]].\n\n"
            f"Question: {question}\n\n"
            f"Wiki content:\n---\n{chr(10).join(contexts)}\n---"
        )
        answer = self.llm(answer_prompt, 'heavy')
        self._log('query', question[:80])
        return answer

    def lint(self) -> dict:
        """Check wiki health: orphans (mechanical) + contradictions (LLM)."""
        page_files = list(self.wiki_dir.rglob('*.md'))
        page_names = {p.stem for p in page_files}
        inbound = {name: 0 for name in page_names}
        link_re = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        for p in page_files:
            text = p.read_text(encoding='utf-8', errors='replace')
            for match in link_re.findall(text):
                target = match.strip()
                if target in inbound:
                    inbound[target] += 1
        orphans = [
            name for name, count in inbound.items()
            if count == 0 and name not in {'index', 'log', 'gaps'}
        ]

        contradictions = []
        if len(page_files) >= 5:
            summaries = []
            for p in page_files[:30]:
                text = p.read_text(encoding='utf-8', errors='replace')[:1500]
                summaries.append(f"### {p.stem}\n{text}")
            scan_prompt = (
                f"Scan these wiki pages for contradictions, stale claims, "
                f"or factual conflicts.\n\n{chr(10).join(summaries)}\n\n"
                f'Return JSON: {{"contradictions": ["one-sentence description", ...]}}. '
                f"Empty list if none. Return ONLY the JSON object."
            )
            try:
                result = self._json_call(scan_prompt, 'heavy')
                if isinstance(result, dict):
                    contradictions = result.get('contradictions', [])
            except Exception:
                contradictions = []

        gaps_path = self.wiki_dir / 'gaps.md'
        if orphans or contradictions:
            entry = f"\n## Lint pass {self._now()}\n\n"
            if orphans:
                entry += "**Orphans:** " + ', '.join(f'[[{o}]]' for o in orphans) + '\n\n'
            if contradictions:
                entry += "**Contradictions:**\n" + '\n'.join(f'- {c}' for c in contradictions) + '\n'
            existing = gaps_path.read_text(encoding='utf-8') if gaps_path.exists() else "# Gaps & Issues\n"
            gaps_path.write_text(existing + entry, encoding='utf-8')

        n_issues = len(orphans) + len(contradictions)
        self._log('lint', f'{n_issues} issues ({len(orphans)} orphans, {len(contradictions)} contradictions)')

        return {
            'orphans': orphans,
            'contradictions': contradictions,
            'pages_scanned': len(page_files),
        }

    def fact_add(self, name: str, body: str, description: str = '') -> str:
        """Add or update a user fact. Returns the slug used."""
        slug = self._slug(name)
        path = f'facts/{slug}.md'
        frontmatter = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description or name}\n"
            f"type: facts\n"
            f"---\n\n"
        )
        self.write(path, frontmatter + body.strip() + '\n')
        self._update_facts_index(slug, name, description or name)
        return slug

    def fact_extract_and_add(self, statement: str) -> Optional[str]:
        """Light-mode helper: extract a single fact from natural language and store it."""
        prompt = (
            f"Edward said: \"{statement}\"\n\n"
            f"If this statement contains a durable fact about Edward worth remembering "
            f"(preference, identity, role, recurring context), return JSON: "
            f'{{"name": "<short topic>", "description": "<one-line description>", "body": "<the fact in 1-3 sentences>"}}.\n'
            f'If it does NOT contain a durable fact (just a passing comment), return: {{"skip": true}}.\n'
            f"Return ONLY the JSON object."
        )
        try:
            result = self._json_call(prompt, 'light')
            if not isinstance(result, dict) or result.get('skip'):
                return None
            return self.fact_add(result['name'], result['body'], result.get('description', ''))
        except Exception:
            return None

    def add_page(self, page_type: str, slug: str, body: str) -> dict:
        """Manually add or augment a wiki entity or concept page.

        Bypasses extraction — Edward provides the slug + body directly.
        Used for /add command when ingest extraction missed something.

        If the page already exists, appends a new dated section rather than
        overwriting. Updates index and logs the operation.
        """
        if page_type not in ('entity', 'concept'):
            raise ValueError(f"page_type must be 'entity' or 'concept', got {page_type!r}")
        if not body or not body.strip():
            raise ValueError("body cannot be empty")

        slug = self._slug(slug)
        if not slug:
            raise ValueError("slug normalized to empty string")

        subdir = 'entities' if page_type == 'entity' else 'concepts'
        rel = f'wiki/{subdir}/{slug}.md'
        today = self._today()
        title = slug.replace('_', ' ').title()

        if self.exists(rel):
            existing = self.read(rel)
            new_section = f"\n## Added {today}\n\n{body.strip()}\n"
            # Try to bump the 'updated' date in YAML frontmatter
            updated_re = re.compile(r'^(updated:\s*)(\S+)', re.MULTILINE)
            existing = updated_re.sub(rf'\g<1>{today}', existing, count=1)
            self.write(rel, existing.rstrip() + '\n' + new_section)
            action = 'updated'
        else:
            frontmatter = (
                f"---\n"
                f"title: {title}\n"
                f"type: {page_type}\n"
                f"created: {today}\n"
                f"updated: {today}\n"
                f"tags: [manual]\n"
                f"sources: [manual_add]\n"
                f"---\n\n"
            )
            self.write(rel, frontmatter + f"## Description\n\n{body.strip()}\n")
            action = 'created'

        # Update wiki/index.md
        index_path = self.wiki_dir / 'index.md'
        if index_path.exists():
            index_text = index_path.read_text(encoding='utf-8')
            line = f"- [[{slug}]]"
            if line not in index_text:
                section = '## Entities' if page_type == 'entity' else '## Concepts'
                index_text = self._insert_under_section(index_text, section, line)
                index_path.write_text(index_text, encoding='utf-8')

        self._log('add', f'{page_type}/{slug}')

        return {'action': action, 'slug': slug, 'type': page_type, 'path': rel}

    def audio_overview_script(self, source_slugs=None) -> dict:
        """Generate a two-voice conversational script summarizing wiki sources.

        Inspired by NotebookLM's audio overview feature. Speaker A is a
        curious host; Speaker B is the expert. Aims for 6-9 minute episodes
        with 35-50 speaker turns.

        v2 corpus: includes the RAW source files (not just wiki summaries) so
        the model has actual quotable detail to work with. Without raw, the
        model summarizes summaries and the script comes out generic.

        v2 prompt: hard requirements on naming specific things, preserving
        source contrasts, speaker pushback, and minimum length.

        Returns: dict with path, source_count, exchanges, estimated_minutes, text.
        """
        sources_dir = self.wiki_dir / 'sources'
        if not sources_dir.exists():
            raise ValueError("No sources/ directory — ingest something first")

        if source_slugs:
            source_files = []
            for slug in source_slugs:
                slug = self._slug(slug)
                p = sources_dir / f'{slug}.md'
                if p.exists():
                    source_files.append(p)
            if not source_files:
                raise ValueError(f"No matching sources found for: {source_slugs}")
        else:
            source_files = sorted(sources_dir.glob('*.md'))
            if not source_files:
                raise ValueError("No sources to summarize. Ingest something first.")

        # ── Corpus building ────────────────────────────────────────────────
        # Order: raw sources first (richest), then wiki summaries (key points,
        # tags, links), then linked entity/concept pages (cross-context).
        corpus_parts = []
        raw_included = 0
        for sf in source_files:
            slug = sf.stem
            raw_candidates = [
                self.raw_dir / f'{slug}.md',
                self.raw_dir / f'{slug}.txt',
                self.raw_dir / slug,  # extension-less fallback
            ]
            raw_file = next((p for p in raw_candidates if p.is_file()), None)
            if raw_file:
                raw_text = raw_file.read_text(encoding='utf-8', errors='replace')[:30000]
                corpus_parts.append(f"=== RAW SOURCE: {slug} ===\n\n{raw_text}")
                raw_included += 1
            # Wiki summary always included — has key points + entity/concept links
            wiki_text = sf.read_text(encoding='utf-8', errors='replace')
            corpus_parts.append(f"=== WIKI SUMMARY: {slug} ===\n\n{wiki_text}")

        # Linked entity/concept pages
        linked_slugs = set()
        link_re = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        for sf in source_files:
            for slug in link_re.findall(sf.read_text(encoding='utf-8', errors='replace')):
                linked_slugs.add(slug.strip())
        for slug in list(linked_slugs)[:20]:
            for subdir in ('entities', 'concepts'):
                p = self.wiki_dir / subdir / f'{slug}.md'
                if p.exists():
                    content = p.read_text(encoding='utf-8', errors='replace')[:1200]
                    corpus_parts.append(f"=== {subdir.upper().rstrip('S')}: {slug} ===\n\n{content}")
                    break

        corpus = '\n\n'.join(corpus_parts)
        if len(corpus) > 50000:
            corpus = corpus[:50000] + '\n\n[corpus truncated]'

        # ── Prompt ─────────────────────────────────────────────────────────
        prompt = (
            f"You are scripting a 6-9 minute audio podcast episode between two AI hosts "
            f"discussing source material. Think NotebookLM podcast style: genuinely curious, "
            f"intellectually substantive, conversational. NOT a lecture. NOT alternating "
            f"monologues. NOT generic acknowledgments.\n\n"
            f"=== HOSTS ===\n\n"
            f"SPEAKER_A — the curious host. Pushes back, asks 'wait, but...', expresses "
            f"surprise, makes the listener feel smart. Says things like 'OK so let me see "
            f"if I'm following', 'huh, that's not what I expected', 'so what's the actual "
            f"difference between X and Y?'. Energetic, engaged, sometimes skeptical.\n\n"
            f"SPEAKER_B — the expert. Explains in plain language, draws specific "
            f"connections, reaches for concrete examples. Quotes specific terms and "
            f"details from the source. Thoughtful, precise, occasionally enthusiastic "
            f"when something clicks.\n\n"
            f"=== HARD REQUIREMENTS ===\n\n"
            f"1. NAME SPECIFIC THINGS from the source. Tools, people, concepts, file "
            f"names. If the source mentions RAG, qmd, NotebookLM, ChatGPT, Obsidian, "
            f"Marp, Dataview, Karpathy, index.md, log.md — these MUST appear by name "
            f"in the dialogue. Generic 'the wiki approach' is not enough.\n\n"
            f"2. PRESERVE CONTRASTS. When the source contrasts itself with another "
            f"approach (e.g., compounding wikis vs. RAG), MAKE THIS EXPLICIT. Speaker A "
            f"might literally say 'wait, isn\'t that just RAG?' and Speaker B explains "
            f"the distinction. Do NOT conflate the contrasted approach with what's being "
            f"advocated.\n\n"
            f"3. SPEAKER_A PUSHES BACK AT LEAST 3 TIMES. With phrases like 'but doesn\'t "
            f"that...', 'wait, how is that different from...', or 'that sounds great in "
            f"theory, but...'. Real conversation has friction.\n\n"
            f"4. AT LEAST 35 SPEAKER TURNS, ideally 40-50. Each turn 1-3 sentences. "
            f"Less than 30 turns is a failure.\n\n"
            f"5. CONCRETE > ABSTRACT. Pull specific lines, examples, file paths, tool "
            f"names from the source. Avoid generic 'allowing for accumulation and "
            f"synthesis'.\n\n"
            f"=== STRUCTURE ===\n\n"
            f"  Open with a hook — what makes this interesting? (1-2 turns)\n"
            f"  Establish the central tension or idea, ideally vs. a contrasted "
            f"alternative (4-6 turns)\n"
            f"  Walk through the architecture concretely (8-12 turns)\n"
            f"  Get into specific operations, tools, or implementation details "
            f"(8-12 turns)\n"
            f"  Discuss implications, limits, or open questions (4-6 turns)\n"
            f"  Close with what's actually novel or counterintuitive (1-2 turns)\n\n"
            f"=== OUTPUT FORMAT ===\n\n"
            f"Just dialogue. No headers, no markdown decoration around speaker labels.\n"
            f"Exactly this format:\n"
            f"  SPEAKER_A: ...\n"
            f"  SPEAKER_B: ...\n"
            f"  SPEAKER_A: ...\n\n"
            f"=== SOURCE MATERIAL ({raw_included} raw / {len(source_files)} summaries) ===\n\n"
            f"{corpus}"
        )

        script = self.llm(prompt, 'heavy')
        if not script or not script.strip():
            raise RuntimeError("LLM returned empty script — check Groq quota / connection")

        exchanges = sum(1 for line in script.splitlines()
                        if line.strip().startswith(('SPEAKER_A:', 'SPEAKER_B:')))

        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if source_slugs and len(source_slugs) == 1:
            base = self._slug(source_slugs[0])
        elif source_slugs:
            base = 'multi'
        else:
            base = 'all'
        filename = f'{base}_{ts}.md'
        out_path = self.overviews_dir / filename

        header = (
            f"---\n"
            f"type: audio_overview_script\n"
            f"created: {self._today()}\n"
            f"sources: {[sf.stem for sf in source_files]}\n"
            f"raw_included: {raw_included}\n"
            f"exchanges: {exchanges}\n"
            f"---\n\n"
            f"# Audio Overview Script\n\n"
            f"Generated from {len(source_files)} source(s) "
            f"({raw_included} raw + {len(source_files)} summaries): "
            f"{', '.join(sf.stem for sf in source_files)}\n\n"
            f"---\n\n"
        )
        out_path.write_text(header + script.strip() + '\n', encoding='utf-8')

        estimated_seconds = exchanges * 12 / 2.5
        estimated_minutes = round(estimated_seconds / 60, 1)
        self._log('overview', f'{base} ({exchanges} exchanges, ~{estimated_minutes}min, raw={raw_included})')

        return {
            'path': str(out_path),
            'source_count': len(source_files),
            'raw_included': raw_included,
            'exchanges': exchanges,
            'estimated_minutes': estimated_minutes,
            'text': script,
        }


    def episodic_append(self, summary: str) -> None:
        """Append a 1-3 sentence summary to today's episodic file."""
        today = self._today()
        path = self.episodic_dir / f'{today}.md'
        ts = datetime.datetime.now().strftime('%H:%M')
        new_block = f"## [{ts}]\n\n{summary.strip()}\n\n"
        if path.exists():
            path.write_text(path.read_text(encoding='utf-8') + new_block, encoding='utf-8')
        else:
            path.write_text(f"# {today}\n\n{new_block}", encoding='utf-8')
            self._update_sessions_index(today, summary[:80])

    # ========================================================================
    # Internal helpers
    # ========================================================================

    _STOPWORDS = frozenset(
        "a an and are as at be but by do does did for from has have he her "
        "him his how i in is it its me my of on or so that the this those "
        "to was were what when where which who whom whose why will with would "
        "you your".split()
    )

    def _keyword_select(self, question: str, index_text: str) -> list:
        """Match question keywords against page slugs from the index.

        For each non-stopword in the question, check if it appears as one of
        the underscore-separated parts of a page slug. Returns up to 5 page
        paths ready to read. Deterministic, fast, and far more reliable than
        light-mode Ollama for "what is X" / "who is Y" questions where X
        appears in the slug.
        """
        q_words = {
            w for w in re.findall(r'\w+', question.lower())
            if w not in self._STOPWORDS and len(w) > 1
        }
        if not q_words:
            return []

        link_re = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        candidates = []
        current_cat = None
        for line in index_text.splitlines():
            stripped = line.strip()
            if stripped.startswith('## '):
                current_cat = stripped[3:].strip().lower()
                continue
            if current_cat not in ('entities', 'concepts', 'sources'):
                continue
            for slug in link_re.findall(line):
                slug = slug.strip()
                slug_parts = set(slug.lower().split('_'))
                if q_words & slug_parts:
                    candidates.append(f'{current_cat}/{slug}.md')
        # De-dup preserving order
        seen = set()
        out = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= 5:
                break
        return out

    def _extract(self, title: str, text: str) -> dict:
        prompt = (
            f"You are ingesting a source into Chloe's wiki. Be COMPREHENSIVE — "
            f"the wiki compounds over time and missing entities can't be backfilled later.\n\n"
            f"Source title: {title}\n"
            f"Source content:\n---\n{text}\n---\n\n"
            f'Return a JSON object with these keys:\n'
            f'  "tldr": 1-2 sentence summary\n'
            f'  "key_points": list of 3-7 key claims\n'
            f'  "entities": list of named things the source ACTUALLY DESCRIBES or EXPLAINS. '
            f'Include a person, project, tool, library, plugin, or service only if the source says '
            f'something substantive about it (what it is, what it does, how it relates) — NOT '
            f'if it is mentioned only by name in passing. Quality over quantity: 3-8 well-supported '
            f'entities is better than padding the list with thin mentions. If you cannot write '
            f'a meaningful page about an entity using only what the source says, do not include it.\n'
            f'  "concepts": list of ideas, methodologies, frameworks, or patterns the source '
            f'actually explains. Same rule as entities: only include concepts the source has real '
            f'content about. 3-7 concepts is plenty.\n'
            f'  "tags": list of 2-5 short topical tags\n\n'
            f"Naming: use the SHORTEST canonical name in snake_case. Each name "
            f"should be 1-3 words MAX. Do NOT include parenthetical expansions, "
            f"acronym expansions, or descriptions in the name itself.\n"
            f"YES                          NO\n"
            f'"rag"                        "rag_retrieval_augmented_generation"\n'
            f'"marp"                       "marp_markdown_based_slide_deck_format"\n'
            f'"dataview"                   "dataview_obsidian_plugin_for_queries"\n'
            f'"claude_code"                "claude_code_anthropic_developer_tool"\n'
            f'"notebooklm"                 "notebooklm_googles_notebook_app"\n'
            f'"andrej_karpathy"            "andrej_karpathy_ai_researcher"\n\n'
            f"If the source mentions an acronym AND its expansion, pick the SHORT form. "
            f"These names become URL-like slugs Edward will type — short slugs are searchable, "
            f"long ones aren\'t.\n\n"
            f"Return ONLY the JSON object."
        )
        result = self._json_call(prompt, 'heavy')
        if not isinstance(result, dict):
            return {'tldr': '', 'key_points': [], 'entities': [], 'concepts': [], 'tags': []}
        return result

    def _validate_and_clean_page(self, response: str):
        """Validate an LLM-generated wiki page response.

        Returns the cleaned page content if valid, or None if the response
        should be skipped (LLM declined to write the page, returned empty,
        returned SKIP_PAGE marker, or response lacks valid YAML frontmatter).

        Strips any preamble before the first --- line so responses like
        "Here is the page:\n---\n..." still work.
        """
        if not response:
            return None
        stripped = response.strip()
        if not stripped:
            return None
        # SKIP_PAGE marker anywhere near the start means LLM declined
        if 'SKIP_PAGE' in stripped[:200]:
            return None
        # Look for YAML frontmatter delimiter. If absent, not a valid page.
        idx = stripped.find('---')
        if idx < 0:
            return None
        # Strip any preamble before the frontmatter
        cleaned = stripped[idx:]
        # Sanity check: must have a title field somewhere in the first 500 chars
        if 'title:' not in cleaned[:500]:
            return None
        return cleaned

    def _update_page(self, page_type: str, name: str, existing,
                     extraction: dict, source_slug: str, source_title: str,
                     source_text: str = '') -> str:
        # Anti-hallucination block shared by both create and update paths.
        # Source text is included so the LLM grounds in what the source actually
        # says about THIS entity, not generic training-data knowledge.
        guardrail = (
            f"\n\nGROUNDING RULES (these are MANDATORY):\n"
            f"1. Use ONLY what the source actually says about '{name}'. Do NOT add "
            f"generic Wikipedia-style background, do NOT invent capabilities or "
            f"features, do NOT pad with plausible-sounding filler.\n"
            f"2. If the source mentions '{name}' only in passing without explaining "
            f"what it is or what it does, return EXACTLY the single line:\n"
            f"   SKIP_PAGE\n"
            f"   on its own with no other text. Do NOT write a page anyway.\n"
            f"3. Quote or paraphrase the source faithfully. If you do not know a fact, "
            f"omit it — do not guess.\n"
            f"4. Related section: only include [[wikilinks]] to things actually "
            f"mentioned in the source. Do NOT invent related entities.\n"
            f"5. Use the current 'updated' date: {self._today()}. Do NOT use any other date.\n"
            f"6. Do NOT include meta-commentary about what you changed or added. The "
            f"output must be the page content only, with no explanatory notes.\n"
        )
        source_excerpt = f"\n\nFull source content (ground your writing in this):\n---\n{source_text[:8000]}\n---\n" if source_text else ''

        if existing:
            instruction = (
                f"Update the existing {page_type} page for '{name}'.\n\n"
                f"Existing page:\n---\n{existing}\n---\n\n"
                f"New source: '{source_title}' (slug: {source_slug})\n"
                f"Source key points: {json.dumps(extraction.get('key_points', []))}"
                f"{source_excerpt}\n\n"
                f"Integrate any NEW information about '{name}' from this source. "
                f"Update the 'updated' date in frontmatter to {self._today()}. Add this "
                f"source to the sources list. Add new content ONLY if the source "
                f"actually adds something specific about '{name}'. Preserve existing "
                f"content unless directly contradicted. Return the COMPLETE updated page."
                f"{guardrail}"
            )
        else:
            instruction = (
                f"Create a new {page_type} page for '{name}'.\n\n"
                f"From source '{source_title}' (slug: {source_slug}):\n"
                f"  Source-wide TLDR: {extraction.get('tldr', '')}\n"
                f"  Source-wide key points: {json.dumps(extraction.get('key_points', []))}\n"
                f"  Tags: {json.dumps(extraction.get('tags', []))}"
                f"{source_excerpt}\n\n"
                f"Write a wiki page with:\n"
                f"  - YAML frontmatter (title, type={page_type}, created={self._today()}, "
                f"updated={self._today()}, tags, sources: [[{source_slug}]])\n"
                f"  - One-paragraph TL;DR specifically about '{name}' (NOT a paraphrase "
                f"of the source-wide TLDR)\n"
                f"  - Body sections covering what the source actually says about '{name}'\n"
                f"  - Related section with [[wikilinks]] only to other things the source mentions\n\n"
                f"Return ONLY the page content."
                f"{guardrail}"
            )
        return self.llm(instruction, 'heavy')

    def _render_source_page(self, slug: str, title: str, extraction: dict) -> str:
        today = self._today()
        tags = extraction.get('tags', [])
        entities = extraction.get('entities', [])
        concepts = extraction.get('concepts', [])
        return (
            f"---\n"
            f"title: {title}\n"
            f"type: source\n"
            f"created: {today}\n"
            f"updated: {today}\n"
            f"tags: {json.dumps(tags)}\n"
            f"---\n\n"
            f"## TL;DR\n\n{extraction.get('tldr', '')}\n\n"
            f"## Key points\n\n"
            + '\n'.join(f"- {p}" for p in extraction.get('key_points', []))
            + ("\n\n## Entities mentioned\n\n"
               + ', '.join(f"[[{self._slug(e)}]]" for e in entities) if entities else '')
            + ("\n\n## Concepts mentioned\n\n"
               + ', '.join(f"[[{self._slug(c)}]]" for c in concepts) if concepts else '')
            + "\n"
        )

    def _update_index(self, slug: str, title: str, entities: list, concepts: list) -> None:
        index_path = self.wiki_dir / 'index.md'
        if not index_path.exists():
            return
        text = index_path.read_text(encoding='utf-8')
        text = self._insert_under_section(text, '## Sources', f"- [[{slug}|{title}]]")
        for entity in entities:
            line = f"- [[{entity}]]"
            if line not in text:
                text = self._insert_under_section(text, '## Entities', line)
        for concept in concepts:
            line = f"- [[{concept}]]"
            if line not in text:
                text = self._insert_under_section(text, '## Concepts', line)
        index_path.write_text(text, encoding='utf-8')

    def _insert_under_section(self, text: str, header: str, line: str) -> str:
        lines = text.splitlines()
        out = []
        i = 0
        inserted = False
        while i < len(lines):
            out.append(lines[i])
            if lines[i].strip() == header and not inserted:
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    out.append(lines[j])
                    j += 1
                if j < len(lines) and lines[j].strip().startswith('(none'):
                    out.append(line)
                    inserted = True
                    i = j + 1
                    continue
                out.append(line)
                inserted = True
                i = j
                continue
            i += 1
        return '\n'.join(out) + ('\n' if not text.endswith('\n') else '')

    def _update_facts_index(self, slug: str, name: str, description: str) -> None:
        index_path = self.facts_dir / 'MEMORY.md'
        line = f"- [{name}]({slug}.md) — {description}"
        if index_path.exists():
            text = index_path.read_text(encoding='utf-8')
            if f'({slug}.md)' not in text:
                index_path.write_text(text.rstrip() + '\n' + line + '\n', encoding='utf-8')
        else:
            index_path.write_text(f"# Facts Index\n\n{line}\n", encoding='utf-8')

    def _update_sessions_index(self, date: str, hook: str) -> None:
        index_path = self.episodic_dir / 'SESSIONS.md'
        line = f"- [{date}]({date}.md) — {hook}"
        if index_path.exists():
            text = index_path.read_text(encoding='utf-8')
            if f'({date}.md)' not in text:
                index_path.write_text(text.rstrip() + '\n' + line + '\n', encoding='utf-8')

    def _log(self, op: str, summary: str) -> None:
        log_path = self.wiki_dir / 'log.md'
        entry = f"\n## [{self._now()}] {op} | {summary}\n"
        if log_path.exists():
            log_path.write_text(log_path.read_text(encoding='utf-8') + entry, encoding='utf-8')
        else:
            log_path.write_text(f"# Wiki Log\n{entry}", encoding='utf-8')

    def _json_call(self, prompt: str, mode: str):
        """Call LLM, parse JSON. Robust to common LLM tics: code fences,
        preamble like 'Here is the JSON:', trailing commentary, and
        Python-repr quirks (None/True/False, single quotes)."""
        raw = self.llm(prompt, mode).strip()
        # Find the JSON region by scanning for the first { or [ and the
        # last matching close. This handles preamble/postamble that ignore
        # the "Return ONLY the JSON" instruction.
        m_start = re.search(r'[\{\[]', raw)
        if m_start is None:
            raise ValueError(f"No JSON in response: {raw[:200]}")
        open_char = raw[m_start.start()]
        close_char = '}' if open_char == '{' else ']'
        m_end = raw.rfind(close_char)
        if m_end == -1 or m_end < m_start.start():
            raise ValueError(f"Malformed JSON in response: {raw[:200]}")
        candidate = raw[m_start.start():m_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Loose: convert Python repr to JSON, single quotes to double
        loose = candidate.replace('None', 'null').replace('True', 'true').replace('False', 'false')
        loose = re.sub(r"(?<![\\\w])'([^']*?)'(?!\w)", r'"\1"', loose)
        try:
            return json.loads(loose)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON: {raw[:200]}") from e

    @staticmethod
    def _slug(text: str) -> str:
        s = re.sub(r'[^\w\s-]', '', text.lower())
        s = re.sub(r'[-\s]+', '_', s).strip('_')
        return s or 'untitled'

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    @staticmethod
    def _today() -> str:
        return datetime.date.today().strftime('%Y-%m-%d')
