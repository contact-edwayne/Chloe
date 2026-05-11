"""queue_processor.py - drain pending tasks from the brain's queue/ folder.

Runs every 2 hours via Windows Task Scheduler. Looks at C:\\Chloe\\brain\\queue\\
for files named:

    RESEARCH-<slug>.md      - topic query against wiki + facts + episodic
    SYNTHESIZE-<slug>.md    - cross-source synthesis from a subset of sources
    DRAFT-<slug>.md         - long-form writing using brain as source material
    ANALYZE-<slug>.md       - patterns / contradictions / gaps in a slice

For each file:
    1. Parse the verb prefix and slug.
    2. Read the file body for instructions / scope / context.
    3. Route to the appropriate handler (one Groq heavy call each).
    4. Write output to:    C:\\Chloe\\brain\\generated\\<date>\\<slug>.md
    5. Move the queue file to:    C:\\Chloe\\brain\\archive\\queue\\<date>-<slug>.md

Idempotent — files already moved out of queue/ won't be reprocessed.

CLI:
    python queue_processor.py             # process every queued file
    python queue_processor.py --dry-run   # parse + route, no LLM, no writes
    python queue_processor.py --once      # process at most 1 file, then exit
"""
import os
import re
import sys
import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def _load_env():
    envf = HERE / ".env"
    if not envf.exists():
        return
    for raw in envf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
_load_env()
sys.path.insert(0, str(HERE))


# ─── Verb → prompt template ─────────────────────────────────────────────────
PROMPT_RESEARCH = """You are Chloe, helping Edward research a topic against his personal brain (a wiki of entities, concepts, and sources he has ingested).

Topic / question:
{user_request}

Below is the user's brain index (top-level page list) and his durable facts. Use them to ground your answer. If the brain has no relevant pages, say so plainly and suggest what to ingest. Cite wiki pages as [[page_name]].

Produce a structured research note with these sections:

# Research: {slug}

## TL;DR
1-2 sentences answering the question if possible.

## What the Brain Knows
What's already in the wiki + facts that's relevant to this topic. Cite pages.

## What's Missing
Gaps in the brain's coverage of this topic. What sources would Edward need to ingest to answer this fully?

## Open Questions
Sub-questions the user might want to dig into.

---

BRAIN INDEX:
{wiki_index}

---

DURABLE FACTS:
{facts}
"""

PROMPT_SYNTHESIZE = """You are Chloe, generating a cross-source synthesis from Edward's brain.

Scope / instructions:
{user_request}

Read the relevant pages below and produce a single well-structured document that synthesizes the material WITHOUT repeating what each source says individually. Pull together claims, surface agreements and disagreements, build a coherent picture.

Output:

# Synthesis: {slug}

## TL;DR
The single insight a reader should walk away with.

## Synthesis
The actual cross-source synthesis, with [[wikilink]] citations to source pages. 3-6 paragraphs.

## Where Sources Disagree
If sources contradict each other, surface the disagreement explicitly.

## What's Underspecified
What the synthesis can't yet say definitively because the brain doesn't have enough material.

---

RELEVANT PAGES FROM BRAIN:
{relevant_pages}
"""

PROMPT_DRAFT = """You are Chloe, drafting long-form content for Edward.

Brief / scope:
{user_request}

Write a full draft using Edward's brain as source material. Cite wiki pages as [[page_name]] inline. Match Edward's voice — direct, terse, opinion-bearing, willing to push back. Don't pad. Don't use marketing language.

Output:

# Draft: {slug}

(then the actual draft, properly structured with headings, no preamble)

---

RELEVANT BRAIN MATERIAL:
{relevant_pages}

---

DURABLE FACTS:
{facts}
"""

PROMPT_ANALYZE = """You are Chloe, running an analysis pass over a slice of Edward's brain.

Analysis scope / question:
{user_request}

Look at the material below and produce a structured analysis. Be specific, cite [[page_name]] for each claim, and don't fall back on generic observations.

Output:

# Analysis: {slug}

## What I Looked At
1-line summary of the slice analyzed.

## Patterns
Notable patterns or themes across the material. Each as a bullet with citations.

## Contradictions
Any places where sources disagree or claims contradict each other.

## Gaps
Things the material implies but doesn't directly cover.

## Recommendations
Concrete next moves Edward could make based on the analysis.

---

BRAIN MATERIAL:
{relevant_pages}
"""


# ─── File parsing ───────────────────────────────────────────────────────────
VERB_PATTERN = re.compile(r"^(RESEARCH|SYNTHESIZE|DRAFT|ANALYZE)-(.+)\.md$",
                          re.IGNORECASE)


def parse_filename(filename: str) -> dict:
    """Return {verb, slug} or None if filename doesn't match the pattern."""
    m = VERB_PATTERN.match(filename)
    if not m:
        return None
    verb = m.group(1).upper()
    slug = m.group(2).strip()
    return {"verb": verb, "slug": slug}


def gather_relevant_pages(brain, user_request: str, max_pages: int = 8) -> str:
    """Use BRAIN's existing keyword-select to pull pages relevant to the
    request. Falls back to LLM page selection if keyword match is empty."""
    try:
        index = brain.read("wiki/index.md")
    except Exception:
        return "(wiki index missing)"

    try:
        pages = brain._keyword_select(user_request, index) or []
    except Exception:
        pages = []
    if not pages:
        try:
            pages = brain._json_call(
                f"You're picking pages from a wiki to inform a task.\n\n"
                f"Task: {user_request[:300]}\n\n"
                f"Wiki index:\n---\n{index}\n---\n\n"
                f'Return a JSON array of up to {max_pages} page paths most likely to be relevant. '
                f'Format: ["entities/foo.md", "concepts/bar.md"]. JSON ONLY.',
                "light",
            ) or []
        except Exception:
            pages = []
        if not isinstance(pages, list):
            pages = []

    parts = []
    for p in pages[:max_pages]:
        try:
            parts.append(f"## {p}\n\n" + brain.read(f"wiki/{p}"))
        except Exception:
            continue
    return "\n\n".join(parts) if parts else "(no relevant wiki pages found for this scope)"


# ─── Routing ────────────────────────────────────────────────────────────────
def build_prompt(verb: str, slug: str, user_request: str, brain) -> str:
    if verb == "RESEARCH":
        try:
            wiki_index = brain.read("wiki/index.md")
        except Exception:
            wiki_index = "(wiki index missing)"
        try:
            facts = brain.facts_only() or "(none)"
        except Exception:
            facts = "(facts read failed)"
        return PROMPT_RESEARCH.format(
            slug=slug, user_request=user_request,
            wiki_index=wiki_index, facts=facts,
        )
    if verb == "SYNTHESIZE":
        return PROMPT_SYNTHESIZE.format(
            slug=slug, user_request=user_request,
            relevant_pages=gather_relevant_pages(brain, user_request),
        )
    if verb == "DRAFT":
        try:
            facts = brain.facts_only() or "(none)"
        except Exception:
            facts = "(facts read failed)"
        return PROMPT_DRAFT.format(
            slug=slug, user_request=user_request,
            relevant_pages=gather_relevant_pages(brain, user_request),
            facts=facts,
        )
    if verb == "ANALYZE":
        return PROMPT_ANALYZE.format(
            slug=slug, user_request=user_request,
            relevant_pages=gather_relevant_pages(brain, user_request),
        )
    raise ValueError(f"unknown verb: {verb}")


# ─── Main loop ──────────────────────────────────────────────────────────────
def process_file(brain, llm_call, queue_file: Path, dry_run: bool = False) -> dict:
    """Process one queue file. Returns result dict."""
    parsed = parse_filename(queue_file.name)
    if not parsed:
        return {"ok": False, "file": queue_file.name,
                "error": f"filename doesn't match VERB-slug.md pattern"}
    verb, slug = parsed["verb"], parsed["slug"]
    try:
        user_request = queue_file.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        return {"ok": False, "file": queue_file.name,
                "error": f"read failed: {type(e).__name__}: {e}"}

    if not user_request:
        # Use the slug itself as the prompt if the file is empty.
        user_request = slug.replace("_", " ").replace("-", " ")

    print(f"[queue] processing {queue_file.name} (verb={verb}, slug={slug})",
          flush=True)

    if dry_run:
        prompt = build_prompt(verb, slug, user_request, brain)
        print(f"  DRY RUN — prompt size {len(prompt)} chars, no LLM call")
        return {"ok": True, "file": queue_file.name, "verb": verb,
                "slug": slug, "dry_run": True}

    try:
        prompt = build_prompt(verb, slug, user_request, brain)
        body = llm_call(prompt, "heavy")
    except Exception as e:
        return {"ok": False, "file": queue_file.name,
                "error": f"LLM call failed: {type(e).__name__}: {e}"}

    if not body or not body.strip():
        return {"ok": False, "file": queue_file.name, "error": "empty LLM response"}

    today = datetime.date.today().isoformat()
    out_dir = brain.root / "generated" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{verb.lower()}-{slug}.md"
    out_path.write_text(body, encoding="utf-8")
    print(f"[queue] wrote {len(body)} bytes -> {out_path}", flush=True)

    # Archive the queue file
    archive_dir = brain.root / "archive" / "queue"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{today}-{queue_file.name}"
    try:
        queue_file.rename(archive_path)
    except Exception as e:
        # If rename fails (cross-volume?), fall back to copy + delete
        try:
            archive_path.write_bytes(queue_file.read_bytes())
            queue_file.unlink()
        except Exception:
            print(f"[queue] failed to archive {queue_file.name}: {e}",
                  flush=True)

    try:
        brain._log("queue_processor", f"{verb} {slug} -> {out_path.name}")
    except Exception:
        pass

    return {"ok": True, "file": queue_file.name, "verb": verb, "slug": slug,
            "out_path": str(out_path)}


def drain(dry_run: bool = False, once: bool = False) -> dict:
    from brain_wiring import BRAIN, chloe_llm_call

    queue_dir = BRAIN.root / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(queue_dir.glob("*.md"))

    if not files:
        print("[queue] no pending tasks", flush=True)
        return {"ok": True, "processed": 0, "files": []}

    results = []
    for qf in files:
        r = process_file(BRAIN, chloe_llm_call, qf, dry_run=dry_run)
        results.append(r)
        if once and r.get("ok"):
            break

    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"[queue] {ok_count}/{len(results)} processed successfully",
          flush=True)
    return {"ok": True, "processed": ok_count, "files": results}


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry-run" in args
    once = "--once" in args
    r = drain(dry_run=dry, once=once)
    if not r.get("ok"):
        sys.exit(1)
