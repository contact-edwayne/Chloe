"""Chloe MCP server — exposes Chloe's memory/wiki/facts to Claude.

v2 (2026-05-17): 10 tools total. Read tools cover full brain scope
(wiki + episodic + generated + queue + raw + overviews + reviews +
facts). Write tools (add_fact, wiki_write) let Claude make persistent
additions through Chloe's own pipelines.

Concurrency: WikiEmbeddingStore + ChloeMemory both use WAL-mode SQLite
with connection-per-op locking, designed to coexist with the running
Chloe instance. Safe to query AND write while jarvis.py is up. Wiki
writes auto-re-embed via wiki_watcher within ~2s.

Setup:
    pip install mcp

Register in Cowork: see the doc at the bottom of this file.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make sure jarvis.py's directory is on sys.path so we can import its modules.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from chloe_memory import ChloeMemory
from wiki_embedding import WikiEmbeddingStore

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp",
          file=sys.stderr)
    sys.exit(1)


# ── Paths (match jarvis.py's setup) ─────────────────────────────────────────
CHLOE_MODE = os.environ.get("CHLOE_MODE", "home")
BRAIN_ROOT = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
MEMORY_DB = _THIS_DIR / "chloe_memory.db"
FACTS_FILE = _THIS_DIR / f"facts_{CHLOE_MODE}.md"
if not FACTS_FILE.exists():
    FACTS_FILE = _THIS_DIR / "facts.md"
ABOUT_FILE = _THIS_DIR / "chloe_about.md"
WIKI_ROOT = BRAIN_ROOT / "wiki"
REVIEWS_ROOT = Path(os.environ.get(
    "CHLOE_REVIEWS_ROOT", r"C:\Chloe\reviews"))


# ── Lazy singletons (only instantiated on first tool call) ─────────────────
_memory: ChloeMemory | None = None
_wiki: WikiEmbeddingStore | None = None


def _get_memory() -> ChloeMemory:
    global _memory
    if _memory is None:
        _memory = ChloeMemory(MEMORY_DB, FACTS_FILE, about_path=ABOUT_FILE)
    return _memory


def _get_wiki() -> WikiEmbeddingStore:
    global _wiki
    if _wiki is None:
        _wiki = WikiEmbeddingStore(wiki_root=WIKI_ROOT, db_path=MEMORY_DB)
    return _wiki


# ── MCP server ──────────────────────────────────────────────────────────────
mcp = FastMCP("chloe")


@mcp.tool()
def recall(query: str, limit: int = 5) -> str:
    """Semantic recall over Chloe's conversation turn log.

    Returns up to `limit` historical chat/voice turns most relevant to
    `query`, ranked by cosine similarity over nomic-embed-text embeddings.
    Falls back to FTS5 keyword search if the embedding model is offline.

    Use when you want to know what Ed and Chloe have actually said about
    a topic — preferred over grepping facts.md when you need the original
    phrasing or context.

    Args:
        query: natural-language search query
        limit: max results (default 5, hard cap 25)

    Returns:
        Formatted markdown list with role/modality/age/snippet per turn,
        or an empty-results message.
    """
    if not query or not query.strip():
        return "Empty query."
    limit = max(1, min(int(limit or 5), 25))
    try:
        hits = _get_memory().search_turns(query, limit=limit,
                                          min_age_hours=0.0)
    except Exception as e:
        return f"recall failed: {type(e).__name__}: {e}"
    if not hits:
        return f"No turns found for: {query}"

    now = datetime.now().timestamp()
    out = [f"**Top recall hits for**: _{query}_  ({len(hits)} of "
           f"up to {limit})\n"]
    for i, h in enumerate(hits, 1):
        age_s = max(0, now - float(h.get("ts", now)))
        if age_s < 60:
            age = f"{int(age_s)}s ago"
        elif age_s < 3600:
            age = f"{int(age_s / 60)}m ago"
        elif age_s < 86400:
            age = f"{int(age_s / 3600)}h ago"
        else:
            age = f"{int(age_s / 86400)}d ago"
        snippet = (h.get("content", "") or "").replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        role = h.get("role", "?")
        modality = h.get("modality", "?")
        out.append(f"{i}. **{role} · {modality} · {age}** — {snippet}")
    return "\n".join(out)


@mcp.tool()
def wiki_search(query: str, limit: int = 5) -> str:
    """Semantic search over Chloe's brain wiki.

    Searches the corpus at `C:\\Chloe\\brain\\wiki\\**\\*.md` — entity
    pages, concept pages, source pages, daily notes. Uses the same
    nomic-embed-text vector store as `recall`. Includes a path-stem
    boost so a query that mentions a page's slug literally tips that
    page above same-topic-but-denser neighbors.

    Use when you want Chloe's curated/synthesized knowledge on a topic
    (her own notes), not transient chat history. Pages include
    auto-persisted Brave search results under `wiki/sources/web_*.md`.

    Args:
        query: natural-language query
        limit: max results (default 5, hard cap 20)

    Returns:
        Formatted markdown list with title/type/score/path/snippet per
        page, or an empty-results message.
    """
    if not query or not query.strip():
        return "Empty query."
    limit = max(1, min(int(limit or 5), 20))
    try:
        store = _get_wiki()
        if store.count_embedded() == 0:
            return ("Wiki store has no embedded pages. Run "
                    "`wiki_watcher.bat --once` to backfill.")
        # apply_staleness_gate=True (explicit -- it's also the default,
        # but Ed asked specifically for wiki_search to exclude stale
        # source pages the way ambient recall already does; making it
        # explicit here means the two callers can never silently drift
        # apart if search()'s default ever changes): excludes superseded
        # pages, ages out old point-in-time quotes/data, and (2026-09-03)
        # collapses same-subject near-duplicate quotes down to the
        # newest -- see WikiEmbeddingStore.search()'s docstring.
        hits = store.search(query, limit=limit, apply_staleness_gate=True)
    except Exception as e:
        return f"wiki_search failed: {type(e).__name__}: {e}"
    if not hits:
        return f"No wiki pages matched: {query}"

    out = [f"**Top wiki hits for**: _{query}_  ({len(hits)} of up to "
           f"{limit}, corpus={store.count_embedded()})\n"]
    for i, h in enumerate(hits, 1):
        title = h.get("title") or h.get("path", "?")
        typ = h.get("type") or "?"
        path = h.get("path", "?")
        score = h.get("score", 0.0)
        snippet = (h.get("snippet") or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        out.append(f"{i}. **{title}** · {typ} · score={score:.2f} · "
                   f"`{path}`\n   {snippet}")
    return "\n".join(out)


@mcp.tool()
def wiki_read(path: str) -> str:
    """Read a full wiki page by relative path.

    Pairs with `wiki_search`: search returns paths (e.g.
    `daily/2026-05-12.md`, `entities/edward_wayne.md`,
    `sources/web_eurovision_2026-05-17.md`), this tool expands the
    full body. Use when a search snippet looks relevant and you need
    the rest of the page.

    Path must be relative to `C:\\Chloe\\brain\\wiki\\` and end in
    `.md`. Path traversal (`..`) is rejected. Body is capped at 50k
    chars (trailing truncation marker if hit).

    Args:
        path: relative wiki path (e.g. "daily/2026-05-12.md")

    Returns:
        Full UTF-8 file contents, or an error message.
    """
    if not path or not path.strip():
        return "Empty path."
    rel = path.strip().lstrip("/\\").replace("\\", "/")
    if not rel.endswith(".md"):
        return f"Refusing non-.md path: {rel}"
    try:
        target = (WIKI_ROOT / rel).resolve()
    except (OSError, ValueError) as e:
        return f"Bad path: {e}"
    try:
        target.relative_to(WIKI_ROOT.resolve())
    except ValueError:
        return f"Path escapes wiki root: {rel}"
    if not target.exists():
        return f"Not found: {rel}"
    if not target.is_file():
        return f"Not a file: {rel}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Read failed: {type(e).__name__}: {e}"
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n_(truncated at 50,000 chars)_"
    return text


@mcp.tool()
def facts() -> str:
    """Return the full body of Chloe's durable facts about Ed.

    Reads `facts.md` (or `facts_<mode>.md` if mode-specific exists).
    Header/instructional preamble is stripped — only the actual facts.

    Use this instead of grepping facts.md from the filesystem when you
    need a definitive list of what Chloe knows about Ed — biographical,
    preferences, opinions, anchors. Updated via the auto-fact extractor
    and the explicit `remember:` command.

    Returns:
        Markdown facts body, or a message if no facts file exists.
    """
    try:
        body = _get_memory().facts_body()
    except Exception as e:
        return f"facts read failed: {type(e).__name__}: {e}"
    if not body.strip():
        return f"No facts recorded in {FACTS_FILE.name}."
    return body


@mcp.tool()
def persona_read() -> str:
    """Return the full body of `chloe_about.md` — Chloe's persona spec.

    `chloe_about.md` lives at `C:\\Users\\eleew\\Documents\\jarvis\\chloe_about.md`
    (outside the brain dir, so `brain_read` cannot reach it). Loaded once
    at Chloe startup and injected into voice + chat system prompts as the
    single source of truth for self-knowledge.

    Sections:
      - Persona (age, style, anti-hedge, no invented bio)
      - Tonal Awareness (mood read, adaptation, no-announce)
      - Voice & speech style (prose, contractions, em-dashes)
      - Seed preferences + Specific favorites
      - Knowledge anchors (character rosters, anti-cross-contamination)
      - Architecture / capabilities / limits

    Use before proposing additions in a persona-evolution pass so you
    don't re-propose what's already there. The meta-header (instructional
    preamble for maintainers) is stripped — only the body Chloe treats
    as self-knowledge is returned.

    Returns:
        Markdown persona body with the meta-header stripped, or a message
        if the file is missing.
    """
    try:
        body = _get_memory().about_body()
    except Exception as e:
        return f"persona_read failed: {type(e).__name__}: {e}"
    if not body.strip():
        return f"No persona body — `{ABOUT_FILE.name}` missing or empty."
    return body


@mcp.tool()
def web_history(window: str = "all") -> str:
    """List Brave search results Chloe has persisted to wiki/sources/.

    Mirrors the `/web_history` slash command in Chloe's chat — reads
    `wiki/sources/web_*.md` files (auto-written by `_persist_brave_to_wiki`
    on every Brave call), parses YAML frontmatter for query+date, returns
    most-recent-first.

    Use to see what Chloe has been researching, or to find a specific
    past lookup. The underlying pages are also searchable via
    `wiki_search` once they've been embedded by the watcher.

    Args:
        window: "today" | "week" | "month" | "all" (default "all").
                Aliases: "1d" / "7d" / "30d".

    Returns:
        Formatted markdown list of {date, query, first citation} entries,
        capped at 25.
    """
    sources_dir = WIKI_ROOT / "sources"
    if not sources_dir.exists():
        return f"No `wiki/sources/` directory at {sources_dir}."
    pages = sorted(sources_dir.glob("web_*.md"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not pages:
        return "No web searches recorded yet."

    win = (window or "all").strip().lower()
    now = datetime.now()
    if win in ("today", "1d"):
        cutoff = now.replace(hour=0, minute=0,
                             second=0, microsecond=0).timestamp()
        pages = [p for p in pages if p.stat().st_mtime >= cutoff]
        label = "today"
    elif win in ("week", "7d"):
        cutoff = (now - timedelta(days=7)).timestamp()
        pages = [p for p in pages if p.stat().st_mtime >= cutoff]
        label = "last 7 days"
    elif win in ("month", "30d"):
        cutoff = (now - timedelta(days=30)).timestamp()
        pages = [p for p in pages if p.stat().st_mtime >= cutoff]
        label = "last 30 days"
    else:
        label = "all-time"
    if not pages:
        return f"No web searches in {label}."

    pages = pages[:25]
    out = [f"**Web search history** ({label}, {len(pages)} of up to 25):\n"]
    for i, p in enumerate(pages, 1):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:1500]
        except Exception:
            continue
        query = ""
        date = ""
        m = re.search(r"^query:\s*['\"]?(.+?)['\"]?\s*$", text, re.M)
        if m:
            query = m.group(1).strip()
        m = re.search(r"^date:\s*(.+)$", text, re.M)
        if m:
            date = m.group(1).strip()
        cite_m = re.search(r"^\d+\.\s+\[(.+?)\]\((.+?)\)", text, re.M)
        cite = ""
        if cite_m:
            title = cite_m.group(1)[:60]
            cite = f" — [{title}]({cite_m.group(2)})"
        if not query:
            stem = p.stem.replace("web_", "")
            query = stem.rsplit("_", 1)[0].replace("_", " ")
        out.append(f"{i}. *{date}* — **{query}**{cite}")
    return "\n".join(out)


# ─── v2 tools ───────────────────────────────────────────────────────────────


@mcp.tool()
def brain_read(path: str) -> str:
    """Read any file under `C:\\Chloe\\brain\\` (full body).

    Wider scope than `wiki_read` — also covers `episodic/` (Daily
    Context Generator output), `generated/` (queue processor output),
    `queue/` (pending verb files), `raw/` (ingest sources), and
    `overviews/` (podcast scripts). Path traversal blocked, body
    capped at 50k chars.

    Args:
        path: relative to brain root (e.g.
              "episodic/CONTEXT-2026-05-17.md",
              "generated/2026-05-10/draft-chloe-pitch.md",
              "queue/some_pending_verb.md")

    Returns:
        Full UTF-8 file contents or an error message.
    """
    if not path or not path.strip():
        return "Empty path."
    rel = path.strip().lstrip("/\\").replace("\\", "/")
    if not rel.endswith(".md") and not rel.endswith(".txt"):
        return f"Refusing non-.md/.txt path: {rel}"
    try:
        target = (BRAIN_ROOT / rel).resolve()
    except (OSError, ValueError) as e:
        return f"Bad path: {e}"
    try:
        target.relative_to(BRAIN_ROOT.resolve())
    except ValueError:
        return f"Path escapes brain root: {rel}"
    if not target.exists():
        return f"Not found: {rel}"
    if not target.is_file():
        return f"Not a file: {rel}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Read failed: {type(e).__name__}: {e}"
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n_(truncated at 50,000 chars)_"
    return text


@mcp.tool()
def brain_list(subdir: str = "", pattern: str = "*.md",
               limit: int = 50) -> str:
    """List files in a brain subdirectory, most-recent-first.

    Use to investigate "what's in episodic/" or "did the queue
    processor write anything recently". Returns name + size + mtime
    per file. Path traversal blocked.

    Args:
        subdir: relative to brain root (e.g. "episodic", "queue",
                "generated/2026-05-10", "overviews"). Empty string
                = brain root itself.
        pattern: glob (default `*.md`). Use `*` for everything.
        limit: max entries (default 50, hard cap 200).

    Returns:
        Formatted markdown list of files or an empty/error message.
    """
    limit = max(1, min(int(limit or 50), 200))
    sd = (subdir or "").strip().lstrip("/\\").replace("\\", "/")
    try:
        target_dir = (BRAIN_ROOT / sd).resolve() if sd else BRAIN_ROOT.resolve()
    except (OSError, ValueError) as e:
        return f"Bad subdir: {e}"
    try:
        target_dir.relative_to(BRAIN_ROOT.resolve())
    except ValueError:
        return f"Subdir escapes brain root: {sd}"
    if not target_dir.exists():
        return f"Not found: {sd or '<brain root>'}"
    if not target_dir.is_dir():
        return f"Not a directory: {sd}"

    try:
        files = [p for p in target_dir.rglob(pattern) if p.is_file()]
    except Exception as e:
        return f"List failed: {type(e).__name__}: {e}"
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return f"No files matching `{pattern}` under `{sd or '<brain root>'}`."

    files = files[:limit]
    out = [f"**Files in `{sd or '<brain root>'}`** (pattern=`{pattern}`, "
           f"{len(files)} of up to {limit}, most-recent-first):\n"]
    for p in files:
        try:
            st = p.stat()
            mt = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes")
            sz = st.st_size
        except OSError:
            mt = "?"
            sz = -1
        rel = p.relative_to(BRAIN_ROOT).as_posix()
        out.append(f"- `{rel}` · {sz} bytes · {mt}")
    return "\n".join(out)


@mcp.tool()
def reviews_read(filename: str = "") -> str:
    """Read Friday meta-review files from `C:\\Chloe\\reviews\\`.

    The Friday 08:00 scheduled task `chloe-friday-meta-review` writes
    `<date>_meta.md` files here — Shipped / Top bugs / 2 proposed
    fixes / Suggested focus. Lives outside the brain dir so the
    other connector tools can't reach it.

    Args:
        filename: exact filename (e.g. "2026-05-15_meta.md"). If
                  empty, returns a listing of all reviews instead.

    Returns:
        Full file body, or a listing if filename is empty, or error.
    """
    if not REVIEWS_ROOT.exists():
        return f"No reviews dir at {REVIEWS_ROOT}."
    if not filename or not filename.strip():
        files = sorted(REVIEWS_ROOT.glob("*.md"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return f"No reviews yet under {REVIEWS_ROOT}."
        out = [f"**Reviews under `{REVIEWS_ROOT}`** "
               f"({len(files)}, most-recent-first):\n"]
        for p in files[:50]:
            mt = datetime.fromtimestamp(
                p.stat().st_mtime).isoformat(timespec="minutes")
            out.append(f"- `{p.name}` · {p.stat().st_size} bytes · {mt}")
        return "\n".join(out)
    rel = filename.strip().lstrip("/\\")
    if "/" in rel or "\\" in rel or ".." in rel:
        return f"Refusing nested path: {rel}"
    if not rel.endswith(".md"):
        return f"Refusing non-.md filename: {rel}"
    target = REVIEWS_ROOT / rel
    if not target.exists():
        return f"Not found: {rel}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Read failed: {type(e).__name__}: {e}"
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n_(truncated at 50,000 chars)_"
    return text


@mcp.tool()
def add_fact(text: str) -> str:
    """Append a durable fact to Chloe's facts.md.

    Wraps `ChloeMemory.add_fact`. Date-stamps automatically. The fact
    becomes part of Chloe's boot context — injected into the system
    prompt on every voice/chat turn after her next restart, so the
    durability is real.

    Use sparingly — facts.md is a high-signal file, not a scratchpad.
    Append biographical / preference / relationship-shaped statements,
    not session details.

    Args:
        text: the fact, no trailing period needed (auto-stripped).

    Returns:
        Confirmation with the appended line, or an error.
    """
    if not text or not text.strip():
        return "Empty fact — nothing to write."
    text = text.strip()
    if len(text) > 500:
        return f"Fact too long ({len(text)} chars; max 500)."
    try:
        ok = _get_memory().add_fact(text)
    except Exception as e:
        return f"add_fact failed: {type(e).__name__}: {e}"
    if not ok:
        return "add_fact returned False — facts file may be unwritable."
    date = datetime.now().strftime("%Y-%m-%d")
    return (f"Fact saved to `{FACTS_FILE.name}`:\n\n"
            f"- {text.rstrip('.')}  *(added {date})*\n\n"
            f"Note: Chloe needs a restart to pick this up in her "
            f"system prompt.")


@mcp.tool()
def wiki_write(topic: str, dry_run: bool = False) -> str:
    """Trigger autonomous wiki research on a topic.

    Calls `brain_wiring.handle_wiki_write` — uses Brave search + Ollama
    synthesis to research the topic, writes the result to
    `wiki/raw/<slug>.md`, then runs entity/concept extraction via
    `BRAIN.ingest`. Heavy: typically 20-60s on the network.

    Use `dry_run=True` to preview the slug + target path without
    burning compute or writing anything.

    Args:
        topic: free-form topic (e.g. "Kelly criterion",
               "Lebron James NBA stats 2026")
        dry_run: if True, preview only — no Groq call, no writes.

    Returns:
        Status string with slug + paths + entity/concept counts, or
        the dry-run preview, or an error.
    """
    if not topic or not topic.strip():
        return "Empty topic — nothing to research."
    if len(topic) > 200:
        return f"Topic too long ({len(topic)} chars; max 200)."
    try:
        from brain_wiring import handle_wiki_write
    except Exception as e:
        return f"wiki_write unavailable: {type(e).__name__}: {e}"
    try:
        return handle_wiki_write(topic.strip(), dry_run=bool(dry_run))
    except Exception as e:
        return f"wiki_write failed: {type(e).__name__}: {e}"


@mcp.tool()
def reviews_write(filename: str, content: str) -> str:
    """Write a file to `C:\\Chloe\\reviews\\`.

    Used by the Friday meta-review scheduled task to save its output.
    Filename must end in `.md`, no path traversal, body capped at
    100k chars. Creates `C:\\Chloe\\reviews\\` if missing. Will NOT
    overwrite an existing file — returns an error if filename
    collides (rename or use a date suffix).

    Args:
        filename: bare filename (e.g. "2026-05-15_meta.md"). No
                  slashes.
        content: markdown body.

    Returns:
        Confirmation with byte count, or an error.
    """
    if not filename or not filename.strip():
        return "Empty filename."
    rel = filename.strip().lstrip("/\\")
    if "/" in rel or "\\" in rel or ".." in rel:
        return f"Refusing nested path: {rel}"
    if not rel.endswith(".md"):
        return f"Refusing non-.md filename: {rel}"
    if not content or not content.strip():
        return "Empty content — nothing to write."
    if len(content) > 100_000:
        return f"Content too long ({len(content)} chars; max 100,000)."
    REVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
    target = REVIEWS_ROOT / rel
    if target.exists():
        return (f"Refusing overwrite: {rel} already exists "
                f"({target.stat().st_size} bytes). Use a date suffix "
                f"or delete the existing file first.")
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Write failed: {type(e).__name__}: {e}"
    return f"Wrote {target} ({len(content)} bytes)."


@mcp.tool()
def queue_add(verb: str, slug: str, body: str) -> str:
    """Drop a verb file into `C:\\Chloe\\brain\\queue\\`.

    The queue processor (every-2h scheduled task) picks up
    `<VERB>-<slug>.md` files, runs the verb against Chloe's brain +
    LLM stack, writes the result to `generated/<date>/<verb>-<slug>.md`,
    and moves the input to `archive/queue/`. Idempotent.

    Use to enqueue autonomous work — research a topic, synthesize
    across pages, draft content, analyze tensions. Lighter than
    wiki_write (deferred to the 2h cadence) and more flexible (any
    verb-shaped instruction).

    Args:
        verb: one of {RESEARCH, SYNTHESIZE, DRAFT, ANALYZE}.
              Case-insensitive; stored uppercase.
        slug: short snake_case identifier (e.g. "mcp_connector_pattern").
              Alnum + underscore + dash, ≤60 chars.
        body: the instruction body the verb operates on. Markdown.

    Returns:
        Confirmation with target path, or an error.
    """
    import re as _re
    verb_u = (verb or "").strip().upper()
    if verb_u not in {"RESEARCH", "SYNTHESIZE", "DRAFT", "ANALYZE"}:
        return (f"Bad verb: {verb!r}. Allowed: RESEARCH, SYNTHESIZE, "
                f"DRAFT, ANALYZE.")
    slug_clean = (slug or "").strip().lower()
    slug_clean = _re.sub(r"[^a-z0-9_-]+", "_", slug_clean)
    slug_clean = _re.sub(r"_+", "_", slug_clean).strip("_")
    if not slug_clean:
        return "Empty slug."
    if len(slug_clean) > 60:
        slug_clean = slug_clean[:60].rstrip("_")
    if not body or not body.strip():
        return "Empty body — nothing to queue."
    if len(body) > 50_000:
        return f"Body too long ({len(body)} chars; max 50,000)."

    queue_dir = BRAIN_ROOT / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{verb_u}-{slug_clean}.md"
    target = queue_dir / filename
    if target.exists():
        return (f"Refusing overwrite: {filename} already in queue. "
                f"Pick a different slug or wait for the processor to "
                f"drain it.")
    try:
        target.write_text(body, encoding="utf-8")
    except OSError as e:
        return f"Write failed: {type(e).__name__}: {e}"
    return (f"Queued `{filename}` for the next processor run.\n"
            f"Path: `{target}`\n"
            f"Expected output: `generated/<date>/{verb_u.lower()}-"
            f"{slug_clean}.md` (within 2h).")


@mcp.tool()
def brain_write(path: str, content: str) -> str:
    """Write a file under `C:\\Chloe\\brain\\` (markdown only).

    Used by scheduled tasks (morning brief, daily finance ingest) to
    drop synthesized content into Chloe's brain without going through
    the heavier research-from-scratch path of `wiki_write`. wiki_watcher
    auto-embeds new files under `wiki/` within ~2s.

    Path-traversal blocked. Refuses non-.md filenames. 100k char cap.
    Will NOT overwrite — returns error on filename collision.

    Args:
        path: relative to brain root (e.g.
              "wiki/sources/finance_news_2026-05-17.md",
              "briefs/morning_brief_2026-05-17.md")
        content: markdown body.

    Returns:
        Confirmation with byte count, or an error.
    """
    if not path or not path.strip():
        return "Empty path."
    rel = path.strip().lstrip("/\\").replace("\\", "/")
    if not rel.endswith(".md"):
        return f"Refusing non-.md path: {rel}"
    if not content or not content.strip():
        return "Empty content."
    if len(content) > 100_000:
        return f"Content too long ({len(content)} chars; max 100,000)."
    try:
        target = (BRAIN_ROOT / rel).resolve()
    except (OSError, ValueError) as e:
        return f"Bad path: {e}"
    try:
        target.relative_to(BRAIN_ROOT.resolve())
    except ValueError:
        return f"Path escapes brain root: {rel}"
    if target.exists():
        return (f"Refusing overwrite: {rel} exists "
                f"({target.stat().st_size} bytes). Pick a different name.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Write failed: {type(e).__name__}: {e}"
    return f"Wrote {target} ({len(content)} bytes)."


@mcp.tool()
def finance_watchlist_read() -> str:
    """Return the contents of finance_watchlist.md.

    The watchlist is Ed's tickers + themes file at
    `C:\\Users\\eleew\\Documents\\jarvis\\finance_watchlist.md`.
    Used by the daily finance ingest scheduled task to know what to
    research. Free-form markdown — Ed edits it directly.

    Returns:
        Full file body, or a default scaffold message if missing.
    """
    watchlist = _THIS_DIR / "finance_watchlist.md"
    if not watchlist.exists():
        return ("Watchlist not configured. Create "
                f"{watchlist} with markdown bullets like:\n\n"
                "## Tickers\n- AAPL\n- TSLA\n\n## Themes\n"
                "- AI infrastructure\n- semiconductor cycle")
    try:
        return watchlist.read_text(encoding="utf-8")
    except OSError as e:
        return f"Read failed: {type(e).__name__}: {e}"


# ─── Lights (v2.1 — wraps lights.py for direct control from Cowork) ────────

def _lazy_lights():
    """Import lights module on first use; cache subsequent imports."""
    import importlib
    return importlib.import_module("lights")


@mcp.tool()
def lights_status() -> str:
    """Return current state of all named bulbs + saved presets.

    Wraps `lights.get_state_snapshot()`. Useful to see which bulbs are
    on, what color/brightness, and what presets are available before
    issuing changes.

    Returns:
        Formatted listing of bulbs (name, mac, ip, on/off, brightness,
        color/ct) plus saved presets, or an error.
    """
    try:
        snap = _lazy_lights().get_state_snapshot()
    except Exception as e:
        return f"lights_status failed: {type(e).__name__}: {e}"
    bulbs = snap.get("bulbs", []) or []
    presets = snap.get("presets", []) or []
    if not bulbs:
        return ("No bulbs configured. Discover with "
                "`python lights.py discover` then name them with "
                "`python lights.py name`.")
    out = [f"**Bulbs** ({len(bulbs)}):"]
    for b in bulbs:
        name = b.get("name") or b.get("mac", "?")
        on = b.get("is_on")
        on_s = "ON" if on else "off" if on is False else "?"
        bri = b.get("brightness")
        bri_s = f"{bri}%" if bri is not None else ""
        color = b.get("color") or ""
        ct = b.get("ct") or ""
        ip = b.get("ip", "?")
        bits = [on_s]
        if bri_s:
            bits.append(bri_s)
        if color:
            bits.append(color)
        if ct:
            bits.append(str(ct))
        bits.append(f"@{ip}")
        out.append(f"- `{name}` — {', '.join(bits)}")
    if presets:
        out.append(f"\n**Presets** ({len(presets)}):")
        for p in presets:
            out.append(f"- `{p.get('name', '?')}`")
    return "\n".join(out)


@mcp.tool()
def lights_set(target: str, on: bool | None = None,
               brightness: int | None = None,
               color: str | None = None,
               ct: str | None = None) -> str:
    """Apply a state change to one or more bulbs.

    Wraps `lights.set_state()`. `target` matches by bulb name (e.g.
    "bedroom", "kitchen") OR special tokens "all" / "everywhere".

    Args:
        target: bulb name, group, or "all".
        on: True to turn on, False to turn off, None to leave.
        brightness: 0-100 percent.
        color: named color from lights.COLORS_RGB (e.g. "warm",
               "blue", "red", "purple").
        ct: color-temperature name or raw kelvin int (e.g.
            "candlelight", "daylight", or 3000).

    Returns:
        Status string with per-bulb result, or an error.
    """
    try:
        r = _lazy_lights().set_state(target, on=on, brightness=brightness,
                                     color=color, ct=ct)
    except Exception as e:
        return f"lights_set failed: {type(e).__name__}: {e}"
    if not r.get("ok"):
        return f"lights_set returned not-ok: {r}"
    results = r.get("results", []) or []
    parts = [f"Set `{target}` (on={on}, bri={brightness}, "
             f"color={color}, ct={ct}):"]
    for res in results:
        name = res.get("bulb_name", "?")
        ok = res.get("ok")
        err = res.get("error", "")
        parts.append(f"- `{name}`: {'OK' if ok else 'FAIL'}"
                     + (f" — {err}" if err else ""))
    return "\n".join(parts)


@mcp.tool()
def lights_command(text: str) -> str:
    """Run a natural-language lights command.

    Wraps `lights.try_handle_lights_command()` — the same parser Chloe
    uses for voice/chat. Handles "turn off the bedroom", "dim the
    kitchen to 30 percent", "set the office to blue", etc.

    Args:
        text: natural-language instruction.

    Returns:
        Reply string from the lights handler, or "not a lights command"
        if the parser didn't recognize the intent.
    """
    if not text or not text.strip():
        return "Empty command."
    try:
        reply = _lazy_lights().try_handle_lights_command(text)
    except Exception as e:
        return f"lights_command failed: {type(e).__name__}: {e}"
    if reply is None:
        return f"Not recognized as a lights command: {text!r}"
    return reply


@mcp.tool()
def lights_preset(name: str) -> str:
    """Apply a saved lights preset.

    Wraps `lights.apply_preset()`. Use `lights_status` to see available
    preset names.

    Args:
        name: preset name (e.g. "movie_night", "morning", "work").

    Returns:
        Status string.
    """
    if not name or not name.strip():
        return "Empty preset name."
    try:
        r = _lazy_lights().apply_preset(name.strip())
    except Exception as e:
        return f"lights_preset failed: {type(e).__name__}: {e}"
    if not r.get("ok"):
        return f"lights_preset returned not-ok: {r}"
    return f"Applied preset `{name}`."


def _lazy_brain_wiring():
    """Import brain_wiring on first use. Returns the module."""
    import importlib
    return importlib.import_module("brain_wiring")


def _lazy_wallet():
    """Import wallet on first use. Returns None if SDK unavailable."""
    try:
        import importlib
        return importlib.import_module("wallet")
    except Exception:
        return None


@mcp.tool()
def wallet_balance() -> str:
    """Return the Lightning wallet's spendable + pending balance.

    Read-only. Calls ``wallet.get_balance()`` via the Breez SDK Liquid.
    Returns a markdown summary with spendable, pending-send, and
    pending-receive amounts in sats.

    Returns:
        Markdown block or an error message.
    """
    w = _lazy_wallet()
    if w is None:
        return ("Wallet unavailable: breez-sdk-liquid not installed or "
                "wallet module missing. See WALLET_SETUP.md.")
    try:
        r = w.get_balance()
    except Exception as e:
        return f"wallet_balance failed: {type(e).__name__}: {e}"
    if not r.get("ok"):
        return f"wallet_balance error: {r.get('error', 'unknown')}"
    lines = [
        "**Wallet balance**",
        f"- Spendable: {r['balance_sat']:,} sats",
        f"- Pending send: {r['pending_send_sat']:,} sats",
        f"- Pending receive: {r['pending_receive_sat']:,} sats",
    ]
    return "\n".join(lines)


@mcp.tool()
def wallet_history(limit: int = 10) -> str:
    """Return recent Lightning wallet payments.

    Read-only. Wraps ``wallet.list_history(limit)``. Returns a markdown
    list of payments (type, status, amount, fees, timestamp, description,
    truncated tx_id) most-recent-first.

    Args:
        limit: max payments to return (1-50, default 10).

    Returns:
        Markdown list or an error message.
    """
    w = _lazy_wallet()
    if w is None:
        return ("Wallet unavailable: breez-sdk-liquid not installed or "
                "wallet module missing.")
    try:
        r = w.list_history(limit=limit)
    except Exception as e:
        return f"wallet_history failed: {type(e).__name__}: {e}"
    if not r.get("ok"):
        return f"wallet_history error: {r.get('error', 'unknown')}"
    payments = r.get("payments", [])
    if not payments:
        return "**Wallet history:** no payments."
    out = [f"**Wallet history** ({len(payments)} of up to {limit})"]
    for p in payments:
        ts = datetime.fromtimestamp(p.get("timestamp", 0)).strftime(
            "%Y-%m-%d %H:%M") if p.get("timestamp") else "?"
        tx = p.get("tx_id", "")
        tx_short = (tx[:8] + "…" + tx[-4:]) if len(tx) > 12 else tx
        desc = p.get("description", "")
        desc_short = (desc[:60] + "…") if len(desc) > 60 else desc
        out.append(
            f"- `{ts}` · {p.get('type', '?')} · {p.get('status', '?')} · "
            f"{p['amount_sat']:,} sat (fees {p['fees_sat']:,}) · "
            f"{desc_short or '(no memo)'} · {tx_short}")
    return "\n".join(out)


@mcp.tool()
def queue_status() -> str:
    """Inspect the brain queue: pending verbs + recent processor output.

    Useful for "did my queued work get picked up yet" checks without
    schema-discovering brain_list + brain_read each time. Returns a
    markdown snapshot of:

    - Pending queue files (``brain/queue/*.md``) with verb + slug + age.
    - Most recent processor outputs (``brain/generated/<latest-date>/``).

    Read-only. Pure listing — no LLM, no triggers.

    Returns:
        Markdown snapshot or an error message.
    """
    import time as _time
    queue_dir = BRAIN_ROOT / "queue"
    generated_dir = BRAIN_ROOT / "generated"
    out: list[str] = []

    # Pending
    if queue_dir.exists():
        pending = sorted(queue_dir.glob("*.md"),
                         key=lambda p: p.stat().st_mtime)
        out.append(f"**Queue pending ({len(pending)}):**")
        if not pending:
            out.append("- _empty_")
        for p in pending[:20]:
            age = _time.time() - p.stat().st_mtime
            age_str = (f"{age/3600:.1f}h" if age >= 3600
                       else f"{age/60:.0f}m")
            out.append(f"- `{p.name}` · {age_str} old")
    else:
        out.append("**Queue pending:** (queue dir missing)")

    # Most recent generated
    out.append("")
    if generated_dir.exists():
        date_dirs = sorted(
            (d for d in generated_dir.iterdir() if d.is_dir()),
            reverse=True)
        if date_dirs:
            latest = date_dirs[0]
            files = sorted(latest.glob("*.md"),
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)
            out.append(f"**Most recent output ({latest.name}, "
                       f"{len(files)} files):**")
            for f in files[:10]:
                size = f.stat().st_size
                out.append(f"- `{f.name}` · {size:,} bytes")
        else:
            out.append("**Generated output:** none yet")
    else:
        out.append("**Generated output:** (dir missing)")

    return "\n".join(out)


@mcp.tool()
def see(question: str = "") -> str:
    """Capture the current desktop and have Chloe describe it.

    Wraps ``brain_wiring.handle_ask(question)`` — the same vision pipeline
    behind the ``/ask`` slash. When ``question`` is empty, returns a
    factual description of what's on screen; otherwise answers the
    specific question grounded in the screenshot.

    Args:
        question: optional question about the visible screen content.
                  Empty string → describe what's on screen.

    Returns:
        Markdown answer or an error message.
    """
    try:
        bw = _lazy_brain_wiring()
        return bw.handle_ask((question or "").strip()
                             or "Describe what's on the screen right now.")
    except Exception as e:
        return f"see failed: {type(e).__name__}: {e}"


@mcp.tool()
def chat(message: str, no_memory: bool = False) -> str:
    """Run a text message through Chloe's chat-handling stack and return the reply.

    Lets Cowork (or any MCP client) drive Chloe's text path end-to-end —
    primarily for closed-loop testing of slash commands, lights handling,
    `remember:` short-circuits, and basic LLM replies — without needing
    Ed to type each test prompt by hand.

    Routing order (mirrors handle_chat in jarvis.py, with the streaming /
    TTS / WebSocket layers stripped):

    1. ``try_handle_brain_command`` — slash commands (``/wiki``,
       ``/recall``, ``/ingest``, ``/wiki_write``, ``/wiki_synth``,
       ``/web_history``, ``/status``, ``/summarize_old``, ``/fact``,
       ``/add``, ``/podcast``, ``/overview``, etc.). Also runs the
       natural-language ``/wiki_*`` aliases shipped 2026-05-18.
    2. ``try_handle_lights_command`` — natural-language lights control.
    3. Fallback: a single non-streaming local Ollama call (Groq retired
       2026-09-01) with a minimal system prompt (chloe_about.md if
       available, plus durable facts). NO Brave hedge-detection fallback,
       NO vision, NO streaming, NO ``remember:`` short-circuit. The
       latter lives in jarvis.py and isn't worth re-importing here — call
       ``add_fact`` directly if you want to persist a fact.

    Both the user message and the assistant reply are pushed into
    ``chloe_memory.db`` (modality ``mcp_chat``) so ``/recall`` and
    ``/summarize_old`` see them. Pass ``no_memory=True`` to skip the
    write (useful for smoke tests you don't want polluting the
    conversation log).

    Args:
        message: the user message to send through the chat stack.
        no_memory: if True, do NOT push the turn into chloe_memory.db.

    Returns:
        The reply text. Slash command output is returned verbatim;
        LLM replies are stripped of leading/trailing whitespace.
    """
    if not message or not message.strip():
        return "Empty message."

    msg = message.strip()
    modality = "mcp_chat"

    def _push(role: str, content: str):
        if no_memory:
            return
        try:
            _get_memory().append_turn(role=role, content=content,
                                      modality=modality)
        except Exception as e:
            print(f"[mcp.chat] memory.append_turn failed: {e}",
                  file=sys.stderr)

    # 1. Brain slash commands (incl. NL /wiki_* aliases)
    try:
        bw = _lazy_brain_wiring()
        brain_reply = bw.try_handle_brain_command(msg)
    except Exception as e:
        brain_reply = None
        print(f"[mcp.chat] brain handler raised: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
    if brain_reply is not None:
        # Slash handlers may return dict ({text, no_tts}) — flatten.
        if isinstance(brain_reply, dict):
            brain_reply = brain_reply.get("text", str(brain_reply))
        _push("user", msg)
        _push("assistant", brain_reply)
        return brain_reply

    # 2. Lights NL
    try:
        lights_reply = _lazy_lights().try_handle_lights_command(msg)
    except Exception as e:
        lights_reply = None
        print(f"[mcp.chat] lights handler raised: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
    if lights_reply is not None:
        _push("user", msg)
        _push("assistant", lights_reply)
        return lights_reply

    # 4. LLM fallback. Groq is fully retired (2026-08-31): llama-3.3-70b-
    #    versatile 404s on this account (moved to an enterprise-only
    #    tier), and the default MCP_CHAT_MODEL was that same model, so
    #    every call here was already falling through to Ollama anyway.
    #    Goes straight to local Ollama now — no API key needed, no wasted
    #    doomed network round-trip first.

    # Assemble a lean system prompt: chloe_about.md (if present) + durable
    # facts. NO recall, NO wiki inject — those need the full chat path.
    sys_parts = []
    try:
        if ABOUT_FILE.exists():
            sys_parts.append(ABOUT_FILE.read_text(encoding="utf-8",
                                                  errors="replace"))
    except Exception:
        pass
    try:
        if FACTS_FILE.exists():
            sys_parts.append("\n=== Durable facts ===\n"
                             + FACTS_FILE.read_text(encoding="utf-8",
                                                    errors="replace"))
    except Exception:
        pass
    system_prompt = "\n".join(sys_parts) if sys_parts else (
        "You are Chloe, Edward's voice + chat assistant.")

    def _ollama_reply() -> str:
        """Local Ollama fallback (no API key needed). '' on failure."""
        try:
            from brain_wiring import _light_call  # type: ignore
            out = _light_call(f"{system_prompt}\n\n[Ed]: {msg}",
                              num_predict=1024)
            return (out or "").strip()
        except Exception as e:
            print(f"[mcp.chat] ollama fallback failed: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return ""

    reply = _ollama_reply()

    if not reply:
        return ("LLM unavailable: local Ollama returned nothing. Check "
                "that Ollama is running (ollama serve) and the model in "
                "OLLAMA_MODEL is pulled.")
    _push("user", msg)
    _push("assistant", reply)
    return reply


@mcp.tool()
def autonomous_status() -> str:
    """Return Stage-4 autonomous self-modification state.

    Reports the enable flag, freeze status, applies-today vs daily cap,
    consecutive_failures vs lockout threshold, and any in-flight watches.
    Read-only.
    """
    import chloe_watchdog
    from chloe_jobs import _read_autonomous_state
    from datetime import datetime
    s = _read_autonomous_state()
    wd = chloe_watchdog.status()
    now = datetime.now().timestamp()
    fz = s.get("frozen_until", 0.0)
    fz_str = (f"frozen for {int((fz - now)/60)} more min"
              if fz > now else "not frozen")
    out = ["**Stage-4 autonomous status:**"]
    out.append(f"- Enabled: **{'ON' if s.get('enabled') else 'OFF'}**")
    out.append(f"- Freeze: {fz_str}")
    out.append(f"- Applied today: {wd['applies_today']} / {wd['max_per_day']}")
    out.append(f"- Consecutive failures: {wd['consecutive_failures']} / "
               f"{wd['max_consecutive_failures']}")
    if wd["consecutive_failures"] >= wd["max_consecutive_failures"]:
        out.append("- **LOCKED.** Reset via `/autonomous reset`.")
    under_watch = wd.get("under_watch", {})
    if under_watch:
        out.append(f"- Under watch: {', '.join(under_watch.keys())}")
    return "\n".join(out)


@mcp.tool()
def autonomous_set_enabled(enabled: bool) -> str:
    """Flip the Stage-4 autonomous enable flag.

    When ENABLED, the `autonomous-fix-recurring-errors` job auto-applies
    proposals with confidence >= 0.85, subject to watchdog rate limits
    (2 applies/day, 30-min interval, 2-strike lockout).

    When DISABLED (default), the job still scans logs + writes proposals
    + a daily digest, but does NOT apply.

    Args:
        enabled: True to enable auto-apply, False to disable.
    """
    from chloe_jobs import _read_autonomous_state, _write_autonomous_state
    s = _read_autonomous_state()
    s["enabled"] = bool(enabled)
    _write_autonomous_state(s)
    if enabled:
        return ("**Stage-4 autonomous ENABLED.** Auto-apply gated by "
                "watchdog rate limits.")
    return ("**Stage-4 autonomous DISABLED.** Proposer writes proposals "
            "but won't apply.")


@mcp.tool()
def autonomous_freeze(minutes: int) -> str:
    """Block autonomous applies for N minutes.

    Useful before a planned restart, during debugging, or any time you
    want to pause Stage-4 without flipping the enable flag.

    Args:
        minutes: 1-1440 (24h max). Use 0 to clear an existing freeze.
    """
    from chloe_jobs import _read_autonomous_state, _write_autonomous_state
    import time as _t
    if minutes < 0 or minutes > 1440:
        return f"minutes must be 0-1440, got {minutes}"
    s = _read_autonomous_state()
    if minutes == 0:
        s["frozen_until"] = 0.0
        _write_autonomous_state(s)
        return "Freeze cleared."
    s["frozen_until"] = _t.time() + (minutes * 60)
    _write_autonomous_state(s)
    return f"Frozen for {minutes} minutes."


@mcp.tool()
def autonomous_reset() -> str:
    """Clear the consecutive_failures counter that locked autonomy.

    After two auto-reverts in a row, the watchdog disables itself. Once
    Ed reviews the failure history (`autonomous_status` + watchdog
    history) and is ready to let autonomy try again, call this.
    """
    import chloe_watchdog
    r = chloe_watchdog.reset_failures()
    return f"Cleared consecutive_failures (was {r['prior_failures']})."


@mcp.tool()
def autonomous_history(limit: int = 20) -> str:
    """Return the last N watchdog events.

    Each entry: action (autonomous_apply / watchdog_watch /
    watchdog_revert / watchdog_cancel / watchdog_reset), outcome,
    slug, reason. Use to audit what the watchdog has been doing.

    Args:
        limit: max events to return. Default 20, hard cap 100.
    """
    import chloe_watchdog
    limit = max(1, min(int(limit), 100))
    rows = chloe_watchdog.history(limit=limit)
    if not rows:
        return "No watchdog history yet."
    out = [f"**Watchdog history ({len(rows)} most recent):**"]
    for h in rows:
        out.append(f"- `{h.get('ts_iso','?')}` · "
                   f"**{h.get('action','?')}** · {h.get('outcome','?')} · "
                   f"`{h.get('slug','?')}` · "
                   f"{(h.get('reason','') or '')[:100]}")
    return "\n".join(out)


@mcp.tool()
def health_full() -> str:
    """Return the same payload as GET /api/health/full but via MCP.

    Bounded health snapshot used by the Stage-4 watchdog. No LLM calls.
    Critical sub-checks: ollama_reachable, memory_db_writable. Returns
    JSON-shaped dict (rendered as text for MCP).
    """
    import json
    try:
        from brain_http import _compute_full_health
        return json.dumps(_compute_full_health(), indent=2, default=str)
    except Exception as e:
        return f"health probe crashed: {type(e).__name__}: {e}"


@mcp.tool()
def propose_and_announce(target: str, kind: str, rationale: str,
                         body: str, test_plan: str,
                         slug: str = "", title: str = "",
                         source: str = "chat",
                         ttl_minutes: int = 5,
                         summary: str = "") -> str:
    """Stage-3 entry point: draft a proposal AND announce it for voice/chat
    confirmation in one shot.

    Combines `chloe_proposals.create_proposal(...)` with
    `chloe_pending_confirms.announce(...)`. The proposal lands on disk
    and a pending-confirm entry is registered. Ed's next yes/no in the
    matching channel resolves it.

    Use case: Cowork-Claude (or a Cowork job) drafts a fix → calls this
    → relays the returned `announce_text` to Ed via the same chat
    surface they're already using. Ed says "yes" / "no" in his next
    reply; the Chloe backend's chat handler picks it up and applies (or
    cancels). No token mint required — Stage 3 IS its own gate.

    Args:
        target: file path (jarvis/ or brain/ relative, or absolute).
        kind: "diff" or "full".
        rationale: Why this change.
        body: unified diff text (kind="diff") or whole-file body
            (kind="full"). Fence wrapping added automatically.
        test_plan: How Ed verifies after applying.
        slug: optional. Auto-derived from title/target if blank.
        title: optional. Markdown H1 of the proposal.
        source: "chat" | "voice" | "any". Channel the confirm resolves on.
        ttl_minutes: minutes Ed has to reply yes/no. Default 5.
        summary: one-line description for the announce text + status.

    Returns:
        Markdown with the announce text + proposal path + slug. Caller
        relays the announce text to Ed.
    """
    import chloe_proposals
    import chloe_pending_confirms

    if source not in ("chat", "voice", "any"):
        return f"Error: source must be chat|voice|any, got {source!r}"
    if ttl_minutes < 1 or ttl_minutes > 60:
        return f"Error: ttl_minutes must be 1-60, got {ttl_minutes}"

    try:
        path = chloe_proposals.create_proposal(
            target=target, kind=kind, rationale=rationale,
            body=body, test_plan=test_plan,
            slug=(slug or None), title=(title or None),
        )
    except ValueError as e:
        return f"create_proposal failed: {e}"
    # Reload the proposal to pick up the auto-derived slug.
    import re as _re_mcp
    m = _re_mcp.match(r"^code_\d{4}-\d{2}-\d{2}_(.+)\.md$", path.name)
    real_slug = m.group(1) if m else slug

    r = chloe_pending_confirms.announce(
        real_slug, source=source, ttl_s=int(ttl_minutes) * 60,
        summary=(summary or f"a change to `{target}`"),
    )
    if not r.get("ok"):
        return f"announce failed (proposal written though): {r.get('error')}"

    return (f"**Drafted + announced** `{real_slug}` → `{target}`.\n\n"
            f"Proposal: `{path.name}`\n"
            f"Channel: `{source}` · TTL: {ttl_minutes} min · "
            f"slot expires {datetime.fromtimestamp(r['expires_at']).isoformat(timespec='seconds')}\n\n"
            f"Speech-shaped announcement (relay this to Ed verbatim, or "
            f"adapt to your tone):\n\n"
            f"> {r['announce_text']}\n\n"
            f"Ed's next non-slash {source}-channel reply with yes / "
            f"yeah / sure / go ahead applies it. Reply with no / nope / "
            f"cancel drops it. Any other reply leaves the pending in "
            f"place until TTL expires.")


@mcp.tool()
def pending_confirms() -> str:
    """List Stage-3 pending-confirm slots.

    Read-only. Use to check whether a proposal you just announced is
    still waiting, OR to see what Ed has queued from other Cowork
    sessions / jobs.

    Returns:
        Markdown table of pending slots (slug, channel, TTL remaining,
        summary).
    """
    import chloe_pending_confirms
    rows = chloe_pending_confirms.pending()
    if not rows:
        return "No pending confirms."
    out = [f"**Pending confirms ({len(rows)}):**"]
    for r in rows:
        ttl_min = int(r["ttl_remaining_s"]) // 60
        ttl_sec = int(r["ttl_remaining_s"]) % 60
        ttl_str = f"{ttl_min}m{ttl_sec}s" if ttl_min else f"{ttl_sec}s"
        out.append(f"- `{r['slug']}` — channel `{r['source']}` · "
                   f"{ttl_str} left · {r['summary'][:80]}")
    return "\n".join(out)


@mcp.tool()
def cancel_pending(slug: str = "") -> str:
    """Cancel a Stage-3 pending confirm without resolving it.

    Args:
        slug: specific pending to drop. Empty string drops all pending.

    Returns:
        Confirmation message.
    """
    import chloe_pending_confirms
    r = chloe_pending_confirms.cancel(slug)
    if r.get("ok"):
        canceled = r.get("canceled", [])
        if not canceled:
            return "Nothing to cancel."
        return f"Canceled {len(canceled)} pending: {', '.join(canceled)}"
    return f"Cancel failed: {r.get('error', 'unknown')}"


@mcp.tool()
def apply_self_patch(slug: str, token: str, dry_run: bool = False) -> str:
    """Tier-2 self-modification: apply a code proposal using a confirm-token.

    Lets Chloe (via this MCP) or any Cowork job apply a previously-drafted
    proposal without Ed retyping `/apply_proposal`. Ed mints the token via
    `/issue_apply_token` (default: 1 apply, 30 min TTL), passes it back to
    the caller, and the caller passes it here.

    All Tier-1 safety rails STILL fire — path whitelist (jarvis/ +
    BRAIN_ROOT), `__pycache__`/`.bak.`/`.git`/`venv*`/`secrets` blocklist,
    ast.parse for `.py`, timestamped backup, max-5-applies-per-session
    counter. The token only relaxes the "human types the slash" gate.

    dry_run=True does NOT consume the token — safe to preview.

    Args:
        slug: the proposal slug. Resolved against newest matching
            `code_*_<slug>.md` under proposals/.
        token: the hex string from `/issue_apply_token`. Constant-time
            compared against active tokens.
        dry_run: if True, preview without writing OR consuming the token.

    Returns:
        Markdown result with backup path + restart hint on success, or
        an error line on rejection.
    """
    import chloe_proposals
    r = chloe_proposals.apply_proposal_with_token(slug, token, dry_run=dry_run)
    if r.get("ok"):
        return r.get("message") or f"Applied `{r.get('slug', slug)}`."
    return f"Apply failed: {r.get('error', 'unknown error')}"


@mcp.tool()
def token_status() -> str:
    """Return active apply-tokens (masked) + per-session apply counter.

    Use to check whether Ed has minted any Tier-2 apply-tokens and how
    many applies are left before the next restart. Read-only.

    Returns:
        Markdown report of `applied_this_session` / `remaining` / per-
        token quota (token IDs masked to first4…last4).
    """
    import chloe_proposals
    st = chloe_proposals.session_state()
    out: list[str] = []
    out.append(f"**Apply session counter:** "
               f"{st['applied_this_session']} used / "
               f"{st['remaining']} remaining "
               f"(cap {st['max_per_session']} per Chloe process lifetime)")
    toks = st.get("tokens", [])
    if not toks:
        out.append("\nNo active apply-tokens.")
    else:
        out.append(f"\n**Active apply-tokens ({len(toks)}):**")
        for t in toks:
            out.append(f"- `{t['token_id']}` — {t['applies_remaining']} "
                       f"applies left, {t['expires_in_minutes']} min until "
                       f"expiry (issued {t['issued_at']})")
    return "\n".join(out)


@mcp.tool()
def capabilities(section: str = "") -> str:
    """Return Chloe's live surface area: slash commands, MCP tools,
    scheduled jobs, env knobs, and modules.

    Ast-derived — accurate by construction, no manual maintenance. Use
    when you need to know "what can Chloe do right now" without reading
    source.

    Args:
        section: Optional. One of "slashes", "tools", "jobs", "env",
            "modules". Empty → full summary.

    Returns:
        Markdown report. JSON shape available via the `/capabilities`
        slash with `--json` if you need structured data.
    """
    import chloe_capabilities
    sect = (section or "").strip().lower()
    if not sect:
        return chloe_capabilities.format_summary_markdown()
    if sect == "slashes":
        items = chloe_capabilities.list_slash_commands()
        out = [f"# Slash commands ({len(items)})"]
        for i in items:
            line = f"- `{i['name']}`"
            if i.get("summary"):
                line += f" — {i['summary']}"
            out.append(line)
        return "\n".join(out)
    if sect == "tools":
        items = chloe_capabilities.list_mcp_tools()
        out = [f"# MCP tools ({len(items)})"]
        for t in items:
            line = f"- `mcp__chloe__{t['name']}` — `{t['signature']}`"
            if t.get("summary"):
                line += f"\n  {t['summary']}"
            out.append(line)
        return "\n".join(out)
    if sect == "jobs":
        items = chloe_capabilities.list_jobs()
        out = [f"# Scheduled jobs ({len(items)})"]
        for j in items:
            if j.get("error"):
                out.append(f"- _error: {j['error']}_")
                continue
            sched = j.get("schedule") or "?"
            health = j.get("health") or "?"
            age = j.get("age_hours")
            age_str = (f"{age:.1f}h ago" if isinstance(age, (int, float))
                       else "never run")
            result = j.get("last_result") or ""
            line = f"- `{j['name']}` — {sched} · **{health}** · {age_str}"
            if result:
                line += f"\n  _{result[:120]}_"
            out.append(line)
        return "\n".join(out)
    if sect == "env":
        items = chloe_capabilities.list_env_knobs()
        out = [f"# Env knobs ({len(items)})"]
        for k in items:
            default = k.get("default", "")
            if len(default) > 80:
                default = default[:77] + "..."
            out.append(f"- `{k['name']}` = {default or '<no default>'}")
        return "\n".join(out)
    if sect == "modules":
        items = chloe_capabilities.list_modules()
        out = [f"# Modules ({len(items)})"]
        for m in items:
            line = (f"- `{m['path']}` — {m['line_count']} lines, "
                    f"{m['function_count']} top-level fn(s)")
            if m.get("summary"):
                line += f" — {m['summary']}"
            out.append(line)
        return "\n".join(out)
    return (f"Unknown section: {sect!r}. "
            f"Try: slashes, tools, jobs, env, modules.")


@mcp.tool()
def explain(module: str) -> str:
    """Ast-introspect a Chloe module by name. Returns module docstring,
    imports, constants, function signatures + first-line docstrings,
    classes + method lists.

    Use to understand a specific module without reading its source —
    e.g. "what does chloe_proposals expose?" / "what's in
    queue_processor?"

    Args:
        module: Module name (with or without `.py` suffix). Must live
            under the jarvis/ directory.

    Returns:
        Markdown report. Raises FileNotFoundError if the module isn't
        under jarvis/.
    """
    import chloe_capabilities
    m = (module or "").strip()
    if not m:
        return ("Usage: explain('<module>') — e.g. explain('brain_wiring'). "
                "List modules via capabilities('modules').")
    try:
        d = chloe_capabilities.describe_module(m)
    except FileNotFoundError as e:
        return f"Unknown module: {m!r}. {e}"
    return chloe_capabilities.format_module_markdown(d)


@mcp.tool()
def about() -> str:
    """Return a categorized menu of all 18 Chloe MCP tools.

    Helps a fresh client (Cowork session, other MCP consumer) bootstrap
    without schema-discovering every tool individually. Use first when
    you don't yet know what's available.

    Returns:
        Markdown menu of tools grouped by category + use case.
    """
    return """**Chloe MCP server — v2.3 (23 tools)**

The server bridges Cowork (Claude) to Chloe's persistent state. Same
SQLite + brain root as the running Chloe instance; WAL-mode concurrent
access is safe.

---

## Read — conversation + memory (4)

- `recall(query, limit=5)` — semantic recall over the turn log
  (nomic-embed-text). Returns role · modality · age · snippet per
  turn. Use when you need original phrasing/context, not summaries.
- `facts()` — full body of facts.md with header stripped. Use for
  the definitive list of what Chloe knows about Ed.
- `persona_read()` — full body of chloe_about.md (persona spec,
  outside the brain dir so brain_read can't reach it). Call before
  proposing persona additions to avoid re-proposing existing
  anchors / favorites / knowledge.
- `web_history(window="all")` — Brave lookups Chloe has persisted.
  Windows: today | week | month | all.

## Read — wiki + brain (5)

- `wiki_search(query, limit=5)` — semantic search over wiki/**/*.md
  with path-stem boost.
- `wiki_read(path)` — full body of a wiki page by relative path
  (e.g. "daily/2026-05-12.md").
- `brain_read(path)` — full body of any .md/.txt under
  C:\\Chloe\\brain\\ (episodic, generated, queue, raw, overviews,
  briefs, facts).
- `brain_list(subdir, pattern, limit)` — list files in a brain
  subdir, most-recent-first. Use for "did this autonomous task run"
  checks.
- `reviews_read(filename)` — Friday meta-review files from
  C:\\Chloe\\reviews\\. Empty filename → listing.

## Read — finance (1)

- `finance_watchlist_read()` — body of finance_watchlist.md.

## Write — durable state (4)

- `add_fact(text)` — append to facts.md (date-stamped, ≤500 chars).
  Chloe restart needed for system-prompt pickup.
- `wiki_write(topic, dry_run=False)` — autonomous research via Brave
  search + Ollama synthesis, writes wiki page, runs entity/concept
  extraction. Heavy (20-60s).
- `reviews_write(filename, content)` — write to C:\\Chloe\\reviews\\.
  No overwrites.
- `brain_write(path, content)` — direct .md write anywhere under
  C:\\Chloe\\brain\\. Wiki paths auto-embed via wiki_watcher within
  ~2s.

## Write — work queue (1)

- `queue_add(verb, slug, body)` — drop a verb file into
  C:\\Chloe\\brain\\queue\\ for the every-2h processor. Verbs:
  RESEARCH, SYNTHESIZE, DRAFT, ANALYZE.

## Lights control (4)

- `lights_status()` — bulbs + presets.
- `lights_set(target, on, brightness, color, ct)` — direct control.
- `lights_command(text)` — natural-language ("dim the kitchen").
- `lights_preset(name)` — apply saved preset.

## Run a chat turn (1, v2.3 — added 2026-05-18)

- `chat(message, no_memory=False)` — drive Chloe's text path end-to-end
  without TTS / streaming / WebSocket scaffolding. Routes through
  brain slash handler → lights NL → local Ollama fallback (Groq
  retired). Use for closed-loop slash testing and basic LLM smoke
  checks. Gaps vs the live chat path: no Brave hedge fallback, no
  vision, no streaming.

## Wallet (read-only, 2 — v2.3)

- `wallet_balance()` — spendable + pending balance in sats. Read-only.
- `wallet_history(limit=10)` — recent payments. Read-only.

## Queue + vision (2 — v2.3)

- `queue_status()` — pending queue files + most-recent processor output.
- `see(question="")` — capture desktop + Chloe describes / answers.

---

## Working rule (set 2026-05-17)

Verify via this connector before pinging Ed. See the
`chloe_collaboration_rules.md` memory entry for the full pattern:
before code changes → check current state; after code changes →
self-verify via connector; escalate to Ed only for module re-import,
voice path testing, HUD/PWA visual checks, or live-chat behavior.

## Deferred to v3

`chat()` (run a chat turn), generic `slash()` (broad surface area),
reverse direction (Chloe → Claude API), persistent event channel,
wallet/vision/queue_status wrappers.
"""


if __name__ == "__main__":
    # Default transport is stdio. Cowork connects over stdio.
    mcp.run()


# ─── Cowork registration ────────────────────────────────────────────────────
# Add this entry to your Cowork MCP config so Claude can call these tools.
# Config file is typically at:
#   %APPDATA%\Claude\claude_desktop_config.json
# or per-Cowork settings. The mcpServers object looks like:
#
# {
#   "mcpServers": {
#     "chloe": {
#       "command": "python",
#       "args": ["C:\\Users\\eleew\\Documents\\jarvis\\chloe_mcp_server.py"],
#       "env": {
#         "CHLOE_MODE": "home",
#         "CHLOE_BRAIN_ROOT": "C:\\Chloe\\brain"
#       }
#     }
#   }
# }
#
# Restart Cowork after editing the config. Then ask Claude:
#   "Call chloe.recall — search for 'wallet'"
# or just
#   "What does Chloe remember about wallet phase 5?"
# and Cowork should auto-route to the recall tool.
