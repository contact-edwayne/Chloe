"""chloe_jobs.py - local equivalents of Cowork-scheduled tasks.

Each Cowork SKILL.md is ported here as a Python function. Invoke via:
    python chloe_jobs.py <job-name>
or via chloe_jobs.bat <job-name> from Windows Task Scheduler.

The point is to NOT burn Anthropic API tokens for routine automation.
Each function uses Chloe's existing local primitives:
  - brain_wiring._heavy_call(prompt)     Groq llama-3.3-70b-versatile
  - brain_wiring._light_call(prompt)     Ollama qwen2.5:32b (local)
  - brain_wiring._search_call(prompt)    Brave search + Ollama synthesis
  - search.web_search(query, count)      Brave API (Ed's free tier)
  - brain_wiring.BRAIN                   Brain singleton (ingest + extract)
  - chloe_memory.ChloeMemory             turns + facts DB

CLI:
    python chloe_jobs.py list
    python chloe_jobs.py <job-name>
    python chloe_jobs.py <job-name> --dry-run    (some jobs honor this)

Logs go to logs/chloe_jobs.log (rotated weekly) AND stdout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import logging.handlers
import os
import shutil
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))


# ─── .env loader ────────────────────────────────────────────────────────────
# jarvis.py loads .env at startup; standalone CLI runs don't go through that
# path. Load it manually so GROQ_API_KEY / BRAVE_API_KEY etc. are present.

def _load_dotenv() -> None:
    env_path = HERE / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        print(f"[chloe_jobs] .env load skipped: {e}", file=sys.stderr)


_load_dotenv()


# ─── Logging ────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("chloe_jobs")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "chloe_jobs.log", when="W0", backupCount=8, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = _setup_logging()


# ─── Lazy primitives ────────────────────────────────────────────────────────

def _brain():
    from brain_wiring import BRAIN  # type: ignore
    return BRAIN


def _heavy(prompt: str) -> str:
    """Synthesis call for background jobs. Defaults to LOCAL Ollama
    (CHLOE_JOBS_LOCAL=1, default on) to keep the shared 100k/day Groq quota
    free for interactive chat/voice — jobs run overnight, so the slower local
    model is an acceptable trade. Set CHLOE_JOBS_LOCAL=0 to force Groq. Falls
    back to Groq if the local call returns empty.

    NOTE: the autonomous code-fix proposer (job_autonomous_fix_recurring_errors)
    calls _heavy_call DIRECTLY, not via this wrapper, so it STAYS on Groq —
    code synthesis needs llama-3.3-70b's structured-output reliability, and
    that job rarely fires anyway."""
    from brain_wiring import _heavy_call, _light_call  # type: ignore
    if os.environ.get("CHLOE_JOBS_LOCAL", "1").strip() != "0":
        out = _light_call(prompt, num_predict=3500)
        if out and out.strip():
            return out
        log.warning("jobs: local synthesis returned empty — falling back to Groq")
    return _heavy_call(prompt)


def _light(prompt: str) -> str:
    from brain_wiring import _light_call  # type: ignore
    return _light_call(prompt)


def _search_llm(prompt: str, *, topic: str = "") -> str:
    """Brave search + Ollama synthesis (Groq compound-mini retired
    2026-08-31). _search_call now returns {"text", "results", "retrieved"}
    so callers that want real citations (see handle_wiki_write in
    brain_wiring.py) can build them from structured data instead of the
    model inventing a sources block -- this wrapper's own callers here
    still just want the body text, so unpack it to keep their existing
    string contract.

    `topic` MUST be passed by every caller -- it's the actual search
    query, distinct from `prompt` (the full write-up instructions).
    2026-09-01 bug, found live: every caller here used to call
    _search_llm(prompt) with no topic, so _search_call's fallback
    silently used the ENTIRE prompt (3000+ chars) as the Brave query,
    which 422'd Brave's length cap on every single call. Four jobs had
    been running fully ungrounded (job_daily_topic_rotation,
    job_daily_finance_ingest, job_daily_morning_brief,
    job_daily_critical_thinking_exercise) with no visible failure signal
    -- "likely a large share of the fabricated citations" found
    2026-08-31. _search_call now also guards query length itself and
    marks the returned text explicitly when no retrieval happened, but
    that's a backstop, not a substitute for passing a real topic here."""
    from brain_wiring import _search_call  # type: ignore
    return _search_call(prompt, topic=topic).get("text", "")


def _brave(query: str, count: int = 8) -> list:
    """Brave Search API. Returns list of {title, url, description, domain}."""
    from search import web_search  # type: ignore
    try:
        return web_search(query, count=count)
    except Exception as e:
        log.warning(f"brave failed for {query!r}: {type(e).__name__}: {e}")
        return []


def _memory():
    """Open a fresh ChloeMemory pointed at the jarvis chloe_memory.db."""
    from chloe_memory import ChloeMemory  # type: ignore
    mode = os.environ.get("CHLOE_MODE", "home")
    facts = HERE / f"facts_{mode}.md"
    if not facts.exists():
        facts = HERE / "facts.md"
    about = HERE / "chloe_about.md"
    return ChloeMemory(HERE / "chloe_memory.db", facts, about_path=about)


# ─── Path helpers ───────────────────────────────────────────────────────────

def _today() -> str:
    return _dt.date.today().isoformat()


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _wiki_dir() -> Path:
    return _brain_root() / "wiki"


def _secrets_dir() -> Path:
    return Path(os.environ.get("CHLOE_SECRETS_DIR", r"C:\Chloe\secrets"))


def _write_brain(rel_path: str, content: str, *, overwrite: bool = False) -> Path:
    """Write under brain root. Path-traversal guarded. Auto-suffix on
    collision unless overwrite=True -- EXCEPT inside wiki/concepts/ and
    wiki/entities/ (2026-08-31, Ed): this _v{n} loop was the actual
    source of the version sprawl he flagged -- every job run that landed
    on an existing slug (or a near-duplicate the caller didn't catch)
    quietly minted another _v2/_v3/_v4/_v5 file instead of merging.
    Concept/entity duplicates are now handled by the caller BEFORE this
    function is reached: wiki_dedup.find_duplicate() decides whether to
    merge (via wiki_dedup.append_dated_revision, then _write_brain(...,
    overwrite=True) on the EXISTING path) or write fresh (a genuinely new
    slug, which shouldn't collide). See _write_concept_with_dedup, used
    by job_daily_topic_rotation and job_daily_critical_thinking_exercise.

    A collision reaching this function for one of those two directories
    with overwrite=False means a caller skipped that check -- raising
    here surfaces that bug immediately instead of silently choosing
    between two wrong behaviors (more sprawl, or a blind overwrite that
    could destroy a genuinely different page that happens to share a
    slug). Every other directory keeps the original auto-suffix
    behavior unchanged."""
    rel = rel_path.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError(f"path traversal: {rel_path!r}")
    target = (_brain_root() / rel).resolve()
    if not str(target).startswith(str(_brain_root().resolve())):
        raise ValueError(f"path escapes brain root: {rel_path!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    top_dir = rel.split("/", 2)[:2]  # e.g. ['wiki', 'concepts']
    is_dedup_scoped = (len(top_dir) == 2 and top_dir[0] == "wiki"
                       and top_dir[1] in ("concepts", "entities"))
    if target.exists() and not overwrite:
        if is_dedup_scoped:
            raise ValueError(
                f"{rel_path!r} already exists and overwrite=False -- "
                f"wiki/concepts/ and wiki/entities/ no longer auto-suffix "
                f"on collision (that's the version-sprawl source). Check "
                f"wiki_dedup.find_duplicate() first and either merge via "
                f"append_dated_revision + overwrite=True, or use a slug "
                f"that's actually new.")
        stem = target.stem
        suffix = target.suffix
        for n in range(2, 10):
            alt = target.with_name(f"{stem}_v{n}{suffix}")
            if not alt.exists():
                target = alt
                break
    target.write_text(content, encoding="utf-8")
    log.info(f"wrote {target.relative_to(_brain_root())} ({len(content)} chars)")
    return target


def _read_brain(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    target = (_brain_root() / rel).resolve()
    if not str(target).startswith(str(_brain_root().resolve())):
        raise ValueError(f"path escapes brain root: {rel_path!r}")
    return target.read_text(encoding="utf-8", errors="replace")


def _write_concept_with_dedup(slug: str, full_content: str, *, wiki_subdir: str,
                              source_label: str, today: str,
                              dedup_query: str = "") -> tuple[Path, bool, str]:
    """Write a concept/entity page, checking wiki_dedup.find_duplicate
    first so a differently-worded slug for the same real-world topic
    merges into the existing page (APPEND a dated revision) instead of
    minting another _v{n} file -- the sprawl Ed flagged: "every daily job
    run mints more _v2/_v3/_v4/_v5 files." Same merge rule Brain.ingest()
    uses (_ingest_typed_page) for the same reason: a concept page is
    durable reference material that legitimately accumulates revisions
    from different runs touching the same topic, so append (not
    supersede, not silent version-fork) is correct here.

    `dedup_query` is the phrase to match against (defaults to `slug`
    converted to space-separated words). Pass it explicitly when `slug`
    carries noise the dedup query shouldn't see -- e.g. a date-stamped
    slug like "thinking_2026-08-31_fed_rate_claim", where the date would
    otherwise dilute the canonical/cosine signal against a different
    day's take on the same underlying claim. Every decision is logged via
    wiki_dedup.log_dedup_decision. Never bulk-merges anything already on
    disk -- only decides where THIS one write lands.

    Returns (written_path, merged: bool, note: str) for the caller's own
    status string.
    """
    from wiki_dedup import (find_duplicate, append_dated_revision,
                            log_dedup_decision, describe_match_score)
    import wiki_embedding

    query = (dedup_query or slug).replace('_', ' ')
    match = None
    try:
        store = wiki_embedding.get_store()
        match = find_duplicate(query, store, scoped_dirs=(wiki_subdir,))
    except Exception as e:
        log.warning(f"[dedup] find_duplicate failed for {slug!r}: "
                   f"{type(e).__name__}: {e}")

    if match:
        dup_rel = f"wiki/{match['path']}"
        try:
            existing_text = _read_brain(dup_rel)
        except Exception as e:
            existing_text = None
            log.warning(f"[dedup] couldn't read matched page {dup_rel}: "
                       f"{type(e).__name__}: {e}")
        if existing_text is not None:
            merged = append_dated_revision(
                existing_text, full_content, date=today, source_label=source_label)
            p = _write_brain(dup_rel, merged, overwrite=True)
            log_dedup_decision('appended', slug, match, target_path=dup_rel,
                               caller=source_label)
            return (p, True,
                    f"merged into {match['path']} "
                    f"({match['match_type']}, {describe_match_score(match)})")

    target_rel = f"wiki/{wiki_subdir}/{slug}.md"
    p = _write_brain(target_rel, full_content)
    log_dedup_decision('new_page', slug, None, target_path=target_rel,
                       caller=source_label)
    return (p, False, "new page")


def _list_brain(rel_subdir: str, pattern: str = "*.md",
                limit: int = 50) -> list[dict]:
    """List files in <brain_root>/<rel_subdir>/, most-recent first."""
    d = (_brain_root() / rel_subdir.replace("\\", "/").lstrip("/")).resolve()
    if not str(d).startswith(str(_brain_root().resolve())):
        return []
    if not d.exists():
        return []
    entries = []
    for p in d.rglob(pattern):
        try:
            st = p.stat()
            entries.append({
                "path": str(p.relative_to(_brain_root())).replace("\\", "/"),
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
        except OSError:
            continue
    entries.sort(key=lambda e: -e["mtime"])
    return entries[:limit]


def _recall(query: str, limit: int = 8, modality: str | None = None) -> list[dict]:
    """Semantic recall via ChloeMemory.search_turns. Optional modality filter."""
    mem = _memory()
    try:
        hits = mem.search_turns(query, limit=limit * 3, min_age_hours=0.0)
    except Exception as e:
        log.warning(f"recall failed for {query!r}: {e}")
        return []
    if modality:
        hits = [h for h in hits if (h.get("modality") or "") == modality]
    return hits[:limit]


def _recent_turns(hours: float = 24.0, modality: str | None = None,
                  limit: int = 200) -> list[dict]:
    """All turns in the last N hours, optionally filtered by modality."""
    mem = _memory()
    cutoff = time.time() - (hours * 3600.0)
    try:
        with mem._connect() as c:  # noqa: SLF001
            if modality:
                rows = c.execute(
                    "SELECT ts, role, content, modality FROM turns "
                    "WHERE ts >= ? AND modality = ? "
                    "ORDER BY ts DESC LIMIT ?",
                    (cutoff, modality, limit)).fetchall()
            else:
                rows = c.execute(
                    "SELECT ts, role, content, modality FROM turns "
                    "WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                    (cutoff, limit)).fetchall()
        return [{"ts": r[0], "role": r[1], "content": r[2], "modality": r[3]}
                for r in rows]
    except sqlite3.Error as e:
        log.warning(f"recent_turns failed: {e}")
        return []


def _dow_local() -> int:
    """0=Mon..6=Sun, in local time. Matches `date +%u - 1` semantics."""
    return _dt.date.today().weekday()


def _facts_body() -> str:
    f = HERE / "facts.md"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def _persona_body() -> str:
    a = HERE / "chloe_about.md"
    return a.read_text(encoding="utf-8") if a.exists() else ""


def _finance_watchlist_body() -> str:
    w = HERE / "finance_watchlist.md"
    return w.read_text(encoding="utf-8") if w.exists() else ""


def _earnings_watchlist_tickers() -> list[str]:
    """Ticker symbols parsed from finance_watchlist.md.

    Symbols are bolded as **TICKER** (1-5 uppercase letters). The bold-span
    regex naturally skips '**136 shares.**' and '**SLV covered calls**'.
    """
    import re
    seen: set = set()
    out: list[str] = []
    for m in re.finditer(r"\*\*([A-Z]{1,5})\*\*", _finance_watchlist_body()):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _earnings_today(tickers: list[str]) -> list[dict] | None:
    """Authoritative 'who reports today' from Finnhub's earnings calendar.

    Returns:
      - None  if FINNHUB_API_KEY is unset, or every fetch failed — caller
        should fall back to LLM web-search rather than assert 'none today'.
      - []    configured + reached, but none of `tickers` report today.
      - [{ticker, hour, eps_estimate, revenue_estimate}, ...] otherwise.
    Best-effort; never raises. `hour` is finnhub's bmo|amc|dmh.
    """
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key or not tickers:
        return None
    import urllib.parse
    import urllib.request
    day = _dt.date.today().isoformat()
    out: list[dict] = []
    reached = False
    for t in tickers:
        q = urllib.parse.urlencode(
            {"from": day, "to": day, "symbol": t, "token": key})
        url = f"https://finnhub.io/api/v1/calendar/earnings?{q}"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "chloe-jobs"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status != 200:
                    continue
                data = json.loads(resp.read().decode("utf-8", "replace"))
            reached = True
        except Exception as e:
            log.warning(f"finnhub earnings fetch failed for {t}: {e}")
            continue
        for row in (data.get("earningsCalendar") or []):
            if (row.get("date") or "") != day:
                continue
            out.append({
                "ticker":           row.get("symbol", t),
                "hour":             (row.get("hour") or "").lower(),
                "eps_estimate":     row.get("epsEstimate"),
                "revenue_estimate": row.get("revenueEstimate"),
            })
    if not reached and not out:
        return None
    return out


# ─── Shared prompt parts ────────────────────────────────────────────────────

_NO_FLUFF = (
    "Hard rules: no fluff openings, no 'in today's complex world', no emoji, "
    "no 'let me know if you'd like more detail'. Concrete examples over "
    "abstract definitions. Real numbers when applicable."
)


# ============================================================================
# JOB: daily-journal-stub
# Drops a third-person daily note if Ed didn't write one.
# Was: chloe-daily-journal-stub (Cowork, 23:01 daily)
# ============================================================================

def job_daily_journal_stub() -> str:
    today = _today()
    target_rel = f"wiki/daily/{today}.md"
    if (_brain_root() / "wiki" / "daily" / f"{today}.md").exists():
        return f"daily note exists for {today}, skipping"

    # Gather observable signal.
    recent_generated = _list_brain("generated", "*.md", limit=5)
    recent_briefs = _list_brain("briefs", "*.md", limit=3)
    web_today = _recall("today", limit=4)
    new_facts = _list_brain("proposals", f"facts_{today}*.md", limit=3)
    finance_news_today = _list_brain("wiki/sources", f"finance_news_{today}*.md",
                                     limit=1)

    parts = []
    for entry in recent_generated:
        parts.append(f"- generated: {entry['path']}")
    for entry in recent_briefs:
        parts.append(f"- brief: {entry['path']}")
    for entry in new_facts:
        parts.append(f"- fact proposals: {entry['path']}")
    for entry in finance_news_today:
        parts.append(f"- finance ingest: {entry['path']}")
    activity = "\n".join(parts) if parts else "(no observable autonomous output today)"

    prompt = (
        "Write a brief third-person daily-note STUB for Ed for the date "
        f"{today}. Ed did not write his own daily note; this is the auto-stub "
        "the Sunday daily_ingest will read.\n\n"
        f"Observable activity today:\n{activity}\n\n"
        f"Recent recall hits:\n" +
        "\n".join(f"- {h.get('role','?')} ({h.get('modality','?')}): "
                  f"{(h.get('content','') or '')[:200]}"
                  for h in web_today) +
        "\n\nWrite 100-200 words. Third person ('Ed worked on...'). "
        "Plain markdown, no bullet points unless naturally listing items. "
        "Single ## Activity heading is enough. " + _NO_FLUFF + "\n\n"
        "Frontmatter to include at top:\n"
        f"---\ntype: daily\ndate: {today}\nauto_stub: chloe_jobs\n---\n"
    )
    body = _heavy(prompt) or ""
    if not body.strip():
        return "LLM returned empty body; not writing stub"
    p = _write_brain(target_rel, body)
    return f"wrote {p.name} ({len(body)} chars)"


# ============================================================================
# JOB: daily-cowork-fact-extract
# Mines today's chat-modality turns (Cowork goes through MCP -> mcp_chat).
# Was: chloe-daily-cowork-fact-extract (Cowork, 22:38 daily)
# ============================================================================

def job_daily_cowork_fact_extract() -> str:
    today = _today()
    turns = _recent_turns(hours=24, modality="mcp_chat", limit=400)
    if not turns:
        # Fall back to "chat" modality if mcp_chat is empty (older versions).
        turns = _recent_turns(hours=24, modality="chat", limit=400)
    if not turns:
        return "no chat-modality turns in last 24h, skipping"

    transcript = "\n".join(
        f"[{_dt.datetime.fromtimestamp(t['ts']).strftime('%H:%M')}] "
        f"{t['role']}: {t['content'][:600]}"
        for t in turns if (t.get("content") or "").strip())

    facts_body = _facts_body()
    today_existing_proposals = []
    for name in (f"proposals/facts_{today}.md",
                 f"proposals/facts_from_cowork_{today}.md",
                 f"proposals/facts_from_voice_{today}.md"):
        try:
            today_existing_proposals.append(_read_brain(name))
        except (FileNotFoundError, ValueError, OSError):
            pass
    existing = "\n\n".join(today_existing_proposals)

    prompt = (
        "Extract NEW fact candidates about Ed from today's Cowork "
        "transcripts. A fact candidate is:\n"
        "- Biographical (relationships, role, history)\n"
        "- A stated preference (likes, dislikes, opinions)\n"
        "- A recurring topic Ed mentioned multiple times\n"
        "- A new project, position, or relationship\n"
        "- Something Ed explicitly asked to remember\n\n"
        "Filter out:\n"
        "- Questions Ed asked (not facts about him)\n"
        "- Generic topical curiosity\n"
        "- Anything already in facts.md (below)\n"
        "- Anything already proposed today (also below)\n"
        "- Trivia and ephemera\n\n"
        f"=== Existing facts.md ===\n{facts_body[:6000]}\n\n"
        f"=== Existing proposals today ===\n{existing[:4000]}\n\n"
        f"=== Today's transcripts ===\n{transcript[:12000]}\n\n"
        "Return up to 5 candidates in this exact markdown format. If zero "
        "candidates clear the bar, return literally:\n"
        "NO_CANDIDATES\n\n"
        "Otherwise:\n"
        "### 1. <one-line statement, no period>\n"
        "**Source:** chat turn ~Nh ago\n"
        "**Evidence:** brief quote (max 150 chars)\n"
        "**Confidence:** high|medium|low\n"
        "**Why it qualifies:** 1-2 sentences\n"
    )
    body = _heavy(prompt) or ""
    if "NO_CANDIDATES" in body or not body.strip():
        return "no candidates passed the bar"

    full = (
        "---\n"
        "type: proposal\n"
        "proposal_type: facts\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.daily-cowork-fact-extract\n"
        "source_channel: cowork\n"
        "status: pending_review\n"
        "---\n\n"
        f"# Fact-candidate proposals (Cowork) — {today}\n\n"
        "Surfaced from Cowork chat in the past 24h. Approve via "
        "`mcp__chloe__add_fact(<text>)` or just edit `facts.md` directly.\n\n"
        "## Candidates\n\n"
        f"{body.strip()}\n"
    )
    p = _write_brain(f"proposals/facts_from_cowork_{today}.md", full)
    return f"wrote {p.name}"


# ============================================================================
# JOB: daily-voice-persona-mining
# Mines voice-modality turns for fact candidates + tonal observations.
# Was: chloe-daily-voice-persona-mining (Cowork, 22:00 daily)
# ============================================================================

def job_daily_voice_persona_mining() -> str:
    today = _today()
    turns = _recent_turns(hours=24, modality="voice", limit=400)
    if not turns:
        return "no voice turns in last 24h, skipping"

    transcript = "\n".join(
        f"[{_dt.datetime.fromtimestamp(t['ts']).strftime('%H:%M')}] "
        f"{t['role']}: {t['content'][:600]}"
        for t in turns if (t.get("content") or "").strip())

    facts_body = _facts_body()
    persona = _persona_body()

    # Part 1: facts
    fact_prompt = (
        "Extract NEW fact candidates about Ed from today's VOICE turns. "
        "Voice carries higher-density signal than typed chat: off-cuff "
        "preferences, biographical details, emotional state. Same bar as "
        "the Cowork fact-extract, but filter out anything already proposed.\n\n"
        f"=== Existing facts.md ===\n{facts_body[:5000]}\n\n"
        f"=== Today's voice transcript ===\n{transcript[:12000]}\n\n"
        "Return up to 5 candidates in this exact format. If zero clear the "
        "bar, return literally:\nNO_CANDIDATES\n\n"
        "### 1. <one-line statement, no period>\n"
        "**Source:** voice turn ~Nh ago\n"
        "**Evidence:** brief quote (max 150 chars)\n"
        "**Confidence:** high|medium|low\n"
        "**Why it qualifies:** 1-2 sentences\n"
    )
    fact_body = _heavy(fact_prompt) or ""
    fact_written = None
    if "NO_CANDIDATES" not in fact_body and fact_body.strip():
        full = (
            "---\ntype: proposal\nproposal_type: facts\n"
            f"date: {today}\n"
            "generated_via: chloe_jobs.daily-voice-persona-mining\n"
            "source_channel: voice\nstatus: pending_review\n---\n\n"
            f"# Voice-channel fact candidates — {today}\n\n"
            f"{fact_body.strip()}\n"
        )
        fact_written = _write_brain(f"proposals/facts_from_voice_{today}.md", full)

    # Part 2: tonal observations
    tonal_prompt = (
        "Extract tonal observations from today's voice turns. Look for:\n"
        "- emotional vocabulary Ed actually uses (specific words)\n"
        "- mood-shift signals (brief/clipped vs verbose/animated)\n"
        "- speech rhythm (short bursts, long pauses)\n"
        "- preference signals about how Chloe should respond\n"
        "- Ed's reactions (positive/negative) to a Chloe phrasing\n\n"
        f"=== Current persona (chloe_about.md) ===\n{persona[:5000]}\n\n"
        f"=== Today's voice transcript ===\n{transcript[:10000]}\n\n"
        "Skip anything ALREADY documented in the persona. Return up to 5 "
        "FALSIFIABLE observations (specific quoted evidence) in this format. "
        "If fewer than 2 solid observations, return literally:\nNO_OBSERVATIONS\n\n"
        "### 1. <pattern in one line>\n"
        "**Evidence:** brief quoted exchange (max 200 chars)\n"
        "**Pattern type:** vocabulary|mood-shift|preference-feedback|"
        "reaction-to-chloe|rhythm\n"
        "**Why it matters:** 1-2 sentences\n"
        "**Already in persona?** yes|no\n"
    )
    tonal_body = _heavy(tonal_prompt) or ""
    tonal_written = None
    if "NO_OBSERVATIONS" not in tonal_body and tonal_body.strip():
        full = (
            "---\ntype: proposal\nproposal_type: tonal_voice\n"
            f"date: {today}\n"
            "generated_via: chloe_jobs.daily-voice-persona-mining\n"
            "source_channel: voice\nstatus: pending_review\n---\n\n"
            f"# Voice tonal observations — {today}\n\n"
            f"{tonal_body.strip()}\n"
        )
        tonal_written = _write_brain(f"proposals/tonal_voice_{today}.md", full)

    bits = [f"voice_turns: {len(turns)}"]
    bits.append(f"facts: {fact_written.name if fact_written else 'none'}")
    bits.append(f"tonal: {tonal_written.name if tonal_written else 'none'}")
    return " · ".join(bits)


# ============================================================================
# JOB: daily-topic-rotation
# One concept page per day, rotating finance/design/macro/etc.
# Was: chloe-daily-topic-rotation (Cowork, 06:00 Mon-Sat)
# ============================================================================

_ROTATION_DOMAINS = {
    0: ("finance", "Finance / options & equity mechanics. "
                   "IV skew, gamma squeezes, wheel-strategy edges, time-decay "
                   "asymmetries, dividend impact on options, assignment risk, "
                   "vol surfaces, pin risk, short-interest dynamics."),
    1: ("design", "Graphic design fundamentals. Typographic hierarchy, color "
                  "theory (split-complementary, triadic, simultaneous contrast), "
                  "grid systems, gestalt principles, contrast types, golden "
                  "ratio in layout, kerning vs tracking, visual rhythm."),
    2: ("macro", "Monetary policy / macro. Yield-curve inversion mechanics, "
                 "M2 vs M3, central-bank balance sheets, repo markets, IRP, "
                 "currency interventions, FOMC dot plot, term premium, "
                 "financial-conditions indices, QT mechanism."),
    3: ("critical_thinking", "Critical thinking / epistemics theory. Bayesian "
                              "updating, steel-manning, type I vs II errors, "
                              "base-rate neglect, sunk-cost vs opportunity-cost "
                              "framing, calibration training, falsifiability."),
    4: ("cross_domain", "Cross-domain integration. Pick a non-obvious connection "
                        "between two of: finance, design, macro, critical-thinking, "
                        "psychology. Bridge the concepts explicitly."),
    5: ("psychology", "Personality / cognitive psychology. Specific cognitive "
                       "biases (one at a time, deep), Big-5 facet detail, "
                       "attachment-style implications, emotional regulation, "
                       "communication styles under pressure."),
}


def job_daily_topic_rotation() -> str:
    dow = _dow_local()
    if dow == 6:
        return "sunday, skipping (Sunday persona-evolution covers persona work)"
    domain, sub_hint = _ROTATION_DOMAINS[dow]
    today = _today()

    # See what concepts already exist so the LLM doesn't repeat.
    existing = _list_brain("wiki/concepts", "*.md", limit=200)
    recent_names = [e["path"].split("/")[-1] for e in existing[:60]]

    prompt = (
        f"Today is {today} ({_dt.date.today().strftime('%A')}). Today's "
        f"domain is **{domain}**. Pick a SPECIFIC intermediate-advanced "
        "subtopic NOT already covered. Existing concept files:\n"
        f"{', '.join(recent_names)}\n\n"
        f"Domain hint: {sub_hint}\n\n"
        "Write a 500-700 word concept page. Use these exact sections (H2):\n"
        "1. Core idea — one tight paragraph\n"
        "2. Why it matters — for Ed when there's a real connection (he runs "
        "covered calls on 136 SLV, holds 212 WU and 300 TE, is building "
        "Chloe). Don't force it.\n"
        "3. The mechanics — concrete details, formulas if applicable, one "
        "worked example with real numbers\n"
        "4. Common misconceptions or traps\n"
        "5. How to apply / detect / use it — actionable\n"
        "6. Related concepts — 3-6 [[wikilinks]] (real or ghost)\n"
        "7. Sources — [1], [2], [3] with URLs\n\n"
        "Include 3-5 real citations from credible primary sources. Use the "
        "web-search ability to find them.\n\n"
        "Output ONLY the slug on line 1, then the markdown body. Slug "
        f"format: `{domain}_<short_subtopic>`. Lowercase, underscores, no "
        "stopwords. Example slug line: `finance_iv_skew_mechanics`\n\n"
        + _NO_FLUFF
    )
    # Real search topic, NOT the prompt (2026-09-01 bug -- see _search_llm's
    # docstring). The specific subtopic isn't known until the LLM responds,
    # so this is the domain hint's own first clause: a short, valid, real
    # description of what's being researched today, not maximally precise
    # but genuinely representative -- the model still picks the specific
    # angle and cites whichever of the retrieved results are relevant.
    #
    # KNOWN LIMITATION (2026-09-01, flagged not fixed): this searches the
    # DOMAIN, not the specific subtopic the model goes on to pick and
    # write about -- e.g. topic="Finance / options & equity mechanics"
    # while the model might write "gamma squeeze mechanics" specifically.
    # Retrieval is real now (the actual bug is fixed: this used to be the
    # whole 3000+ char prompt, 422ing every time), but it may not match
    # the page's specific angle closely. A tighter fix would need a
    # two-phase call (pick the subtopic first, then search for THAT), not
    # done here -- this is a real gap, not a fully solved problem.
    topic = sub_hint.split(".")[0].strip() or domain
    result = _search_llm(prompt, topic=topic) or ""
    if not result.strip():
        return "LLM returned empty; not writing"

    # Parse slug from first line.
    lines = result.strip().split("\n", 1)
    slug = lines[0].strip().lstrip("`").rstrip("`")
    body = lines[1].strip() if len(lines) > 1 else ""
    if not slug or "/" in slug or "\\" in slug:
        slug = f"{domain}_{today.replace('-', '')}"
    if not body:
        return "LLM produced slug but no body; not writing"

    fm = (
        "---\n"
        "type: concept\n"
        f"topic: {domain}\n"
        "depth: intermediate\n"
        f"generated_at: {today}\n"
        "auto_generated: chloe_jobs.daily-topic-rotation\n"
        "---\n\n"
    )
    full = fm + body
    p, merged, note = _write_concept_with_dedup(
        slug, full, wiki_subdir="concepts",
        source_label="chloe_jobs.daily-topic-rotation", today=today)
    return f"topic={domain} slug={slug} → {p.name} ({note})"


# ============================================================================
# JOB: daily-finance-ingest
# Per-ticker + per-theme news → wiki/sources/finance_news_<date>.md.
# Was: chloe-daily-finance-ingest (Cowork, 07:31 weekdays)
# ============================================================================

def job_daily_finance_ingest() -> str:
    today = _today()
    watchlist = _finance_watchlist_body()
    if not watchlist.strip():
        return "finance_watchlist.md is empty/missing; skipping"

    # Pull a quick list of tickers + themes from the watchlist for the prompt.
    # The LLM will route web search internally.
    prompt = (
        f"You are writing today's finance news digest for Ed. Date: {today}.\n\n"
        f"=== Ed's watchlist (jarvis/finance_watchlist.md) ===\n"
        f"{watchlist[:6000]}\n\n"
        "For each TICKER in the watchlist (parse them out): search for "
        "overnight news, price action, analyst moves. For each THEME: search "
        "for relevant macro / sector news. \n\n"
        "Then write a tight markdown digest with these sections:\n"
        "- ## Per-ticker (one subheading per ticker, 2-4 bullets each)\n"
        "  * Include: 1-day price action %, any news, IV / option-flow note "
        "    if relevant.\n"
        "  * Add a 'Strategy implications' paragraph per ticker tied to the "
        "    watchlist's 'Strategies you're actively running' block.\n"
        "  * Hard constraint: do NOT invent strikes or expirations not in "
        "    the watchlist.\n"
        "- ## Themes (macro overlay — DXY, real yields, sector rotation, etc.)\n"
        "- ## What to watch tomorrow\n\n"
        "Cap at 1500 words. Cite sources inline. " + _NO_FLUFF
    )
    # Real search topic (2026-09-01 bug -- see _search_llm's docstring):
    # the watchlist's own ticker symbols, not the 6000-char watchlist
    # dump. A single Brave call can't deeply cover every ticker + theme
    # here (that would need multiple search rounds, out of scope for
    # this fix), but this is a real, valid, representative query instead
    # of a 422 followed by silent ungrounded output.
    tickers = _earnings_watchlist_tickers()
    topic = (" ".join(tickers[:8]) + " stock market news today").strip()
    body = _search_llm(prompt, topic=topic) or ""
    if not body.strip():
        return "LLM returned empty; not writing"

    full = (
        "---\n"
        "type: source\n"
        "source_type: finance_news\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.daily-finance-ingest\n"
        "---\n\n"
        f"# Finance digest — {today}\n\n"
        f"{body.strip()}\n"
    )
    p = _write_brain(f"wiki/sources/finance_news_{today}.md", full)
    return f"wrote {p.name} ({len(body)} chars)"


# ============================================================================
# JOB: daily-morning-brief
# Open loops + overnight changes + focus + sparks + macro + earnings.
# Was: chloe-daily-morning-brief (Cowork, 07:00 daily)
# ============================================================================

def job_daily_morning_brief() -> str:
    today = _today()

    # Gather context.
    contexts = _list_brain("episodic", "CONTEXT-*.md", limit=3)
    latest_ctx_body = ""
    if contexts:
        try:
            latest_ctx_body = _read_brain(contexts[0]["path"])
        except Exception:
            pass
    open_loop_hits = _recall("open loop OR todo OR pending OR stuck", limit=8)
    fact_hits = _recall("Ed preferences biographical holdings", limit=5)
    facts = _facts_body()
    watchlist = _finance_watchlist_body()
    recent_generated = _list_brain("generated", "*.md", limit=5)

    recent_meta = ""
    reviews_dir = Path(r"C:\Chloe\reviews")
    if reviews_dir.exists():
        latest = sorted(reviews_dir.glob("*_meta.md"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if latest and time.time() - latest[0].stat().st_mtime < 86400:
            try:
                recent_meta = latest[0].read_text(encoding="utf-8")[:4000]
            except Exception:
                pass

    open_loop_text = "\n".join(f"- ({h.get('modality','?')}): "
                               f"{(h.get('content','') or '')[:200]}"
                               for h in open_loop_hits)
    fact_hit_text = "\n".join(f"- {(h.get('content','') or '')[:200]}"
                              for h in fact_hits)
    gen_text = "\n".join(f"- {e['path']}" for e in recent_generated)

    # Authoritative earnings-day surfacing (Finnhub). When FINNHUB_API_KEY is
    # configured we inject the real "who reports today" list so the LLM stops
    # guessing dates; otherwise we fall back to a hardened web-search prompt.
    _tickers = _earnings_watchlist_tickers()
    _earnings = _earnings_today(_tickers)
    if _earnings is None:
        earnings_block = (
            "## Earnings today\n"
            f"Web-search whether any of these watchlist tickers report today: "
            f"{', '.join(_tickers) or '(none parsed)'}. Include a ticker ONLY "
            "with a credible source confirming today's date; cite it. Format: "
            "**TICKER** · BMO|AMC · consensus EPS X · rev Y, plus position "
            "context tied to active strategies. Omit the section if none "
            "confirmed.\n\n"
        )
    elif not _earnings:
        earnings_block = (
            f"(Skip the 'Earnings today' section — Finnhub confirms none of "
            f"[{', '.join(_tickers)}] report today.)\n\n"
        )
    else:
        _hourmap = {"bmo": "BMO", "amc": "AMC", "dmh": "during market hours"}
        _lines = []
        for e in _earnings:
            hour = _hourmap.get(e["hour"], e["hour"] or "time TBD")
            eps = e.get("eps_estimate")
            rev = e.get("revenue_estimate")
            _lines.append(
                f"- **{e['ticker']}** · {hour}"
                + (f" · consensus EPS {eps}" if eps is not None else "")
                + (f" · rev est {rev}" if rev is not None else ""))
        earnings_block = (
            "## Earnings today\n"
            "AUTHORITATIVE (Finnhub) — these watchlist tickers report today. "
            "Do NOT web-search the schedule; just add position context tied "
            "to Ed's active strategies:\n"
            + "\n".join(_lines) + "\n\n"
        )

    prompt = (
        f"Produce Ed's morning brief for {today}. Output is a markdown "
        "document (no preamble, no chat-fillers). Target 500 words max.\n\n"
        f"=== Latest CONTEXT (Daily Context Generator) ===\n"
        f"{latest_ctx_body[:6000]}\n\n"
        f"=== Open-loop recall hits ===\n{open_loop_text}\n\n"
        f"=== Fact-shape recall hits ===\n{fact_hit_text}\n\n"
        f"=== Recent autonomous output ===\n{gen_text}\n\n"
        f"=== Current facts.md (dedupe against this) ===\n{facts[:4000]}\n\n"
        f"=== Watchlist (for macro/earnings tie-ins) ===\n{watchlist[:3000]}\n\n"
        f"=== Recent meta-review (last 24h, if any) ===\n{recent_meta[:3000]}\n\n"
        "Structure:\n"
        "# Morning Brief — {date}\n\n"
        "## Open loops\n"
        "3-5 specific items. If nothing's open: 'clear deck.'\n\n"
        "## What changed overnight\n"
        "- new generated/ files (if any)\n"
        "- new finance_news (mention if landed)\n"
        "- meta-review summary (if recent)\n\n"
        "## Today's focus suggestion\n"
        "ONE sentence tied to highest-priority loop.\n\n"
        "## Macro calendar today\n"
        "Use web search to find any high-impact US release scheduled today "
        "(CPI, jobs, FOMC, Fed speeches, PCE, retail sales, GDP, ISM). "
        "Format: **HH:MM ET — Release** · prior X · consensus Y. One-line "
        "implication for SLV (silver, sensitive to DXY/real yields), WU "
        "(remittance, FX/employment), TE (solar, rates/tariffs). If nothing "
        "notable: 'no high-impact US releases scheduled.'\n\n"
        + earnings_block +
        "## Sparks (optional)\n"
        "2-3 short items.\n\n"
        "## Fact candidates queued\n"
        "Skip this — fact extraction runs as a separate job.\n\n"
        + _NO_FLUFF
    )
    # Real search topic (2026-09-01 bug -- see _search_llm's docstring):
    # matches what the "Macro calendar today" section actually needs.
    topic = f"US economic calendar {today} CPI jobs FOMC Fed Reserve"
    body = _search_llm(prompt, topic=topic) or ""
    if not body.strip():
        return "LLM returned empty; not writing"
    p = _write_brain(f"briefs/morning_brief_{today}.md", body)
    return f"wrote {p.name} ({len(body)} chars)"


# ============================================================================
# JOB: daily-critical-thinking-exercise
# Picks one news claim, deconstructs it with a fixed framework.
# Was: chloe-daily-critical-thinking-exercise (Cowork, 13:00 weekdays)
# ============================================================================

def job_daily_critical_thinking_exercise() -> str:
    today = _today()
    prompt = (
        f"Today is {today}. Pick ONE specific factual or causal claim from "
        "today's news. Avoid pure opinion, sports, and irreducibly partisan "
        "framings. Prefer claims touching finance, monetary policy, tech, "
        "or energy/solar (Ed runs 136 SLV covered calls, holds 212 WU and "
        "300 TE). Use web search to find both the original source AND a "
        "credible critical response.\n\n"
        "Output line 1: the slug `thinking_YYYY-MM-DD_<five-word-claim-slug>`. "
        "Then a blank line, then the markdown body with these exact sections "
        "(H2):\n"
        "1. <the claim in one sentence as H1>\n"
        "   Then: 'Reported in <publication> on <date>, attributed to "
        "<speaker>. Link: <URL>'\n"
        "## 1. What's actually being claimed?\n"
        "## 2. Premises (P1, P2, P3 each load-bearing)\n"
        "## 3. Evidence cited (and quality: Strong / Mixed / Weak per "
        "premise)\n"
        "## 4. What's missing from the evidence\n"
        "## 5. Steelman of the counterargument\n"
        "## 6. What would change my mind? (a specific falsifiable "
        "observation, and the time horizon on which it would resolve — "
        "when would you expect to know if this observation has or hasn't "
        "happened)\n"
        "## 7. Implications for Ed (SLV / WU / TE / Chloe / none)\n"
        "## Related concepts (3-5 [[wikilinks]])\n"
        "## Sources ([1], [2], ... with URLs)\n\n"
        "600-900 words. Steel-man must be present even when you find the "
        "claim convincing. " + _NO_FLUFF
    )
    # Real search topic (2026-09-01 bug -- see _search_llm's docstring).
    # "Pick ONE claim from today's news" has no single topic known in
    # advance -- unlike job_daily_topic_rotation's fixed per-day domain,
    # this job's whole premise is picking from CURRENT news, which a
    # non-search call has no way to do. This is the same fixed interest-
    # area phrase already stated in the prompt above (finance, monetary
    # policy, tech, energy/solar), used as a real, valid, on-topic query
    # so the model has actual current headlines to pick a claim from --
    # narrower per-claim searching would need multiple search rounds,
    # out of scope for this fix.
    topic = f"finance monetary policy tech energy solar news {today}"
    result = _search_llm(prompt, topic=topic) or ""
    if not result.strip():
        return "LLM returned empty; not writing"
    lines = result.strip().split("\n", 1)
    slug = lines[0].strip().lstrip("`").rstrip("`")
    body = lines[1].strip() if len(lines) > 1 else ""
    if not slug.startswith("thinking_"):
        slug = f"thinking_{today}_unspecified"
    if not body:
        return "LLM produced slug but no body"
    fm = (
        "---\n"
        "type: concept\n"
        "topic: critical_thinking\n"
        "subtopic: applied_exercise\n"
        "depth: practice\n"
        f"generated_at: {today}\n"
        "auto_generated: chloe_jobs.daily-critical-thinking-exercise\n"
        "---\n\n"
    )
    # Dedup query: strip the "thinking_YYYY-MM-DD_" prefix before matching
    # so the date doesn't dilute the canonical/cosine signal -- two
    # different days' takes on the same underlying claim should still
    # merge; the date is provenance, not part of the topic.
    import re as _re_local
    claim_part = _re_local.sub(r"^thinking_\d{4}-\d{2}-\d{2}_", "", slug)
    p, merged, note = _write_concept_with_dedup(
        slug, fm + body, wiki_subdir="concepts",
        source_label="chloe_jobs.daily-critical-thinking-exercise",
        today=today, dedup_query=claim_part)
    return f"slug={slug} → {p.name} ({note})"


# ============================================================================
# JOB: weekly-backup
# Wraps the existing backup_chloe.py script.
# Was: chloe-weekly-backup (Cowork, Sun 03:07)
# ============================================================================

def job_weekly_backup() -> str:
    backup_script = HERE / "backup_chloe.py"
    if not backup_script.exists():
        return f"backup_chloe.py not found at {backup_script}"
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(backup_script)],
            cwd=str(HERE), capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.error(f"backup_chloe.py failed: {result.stderr[:500]}")
            return f"FAILED (exit {result.returncode}): {result.stderr[:300]}"
        return f"OK\n{result.stdout[-1500:]}"
    except Exception as e:
        return f"FAILED: {type(e).__name__}: {e}"


# ============================================================================
# JOB: weekly-autonomous-audit
# Inspects pipeline state, classifies HEALTHY/WARN/DARK per task.
# Was: chloe-weekly-autonomous-audit (Cowork, Sun 04:01)
# ============================================================================

def job_weekly_autonomous_audit() -> str:
    today = _today()
    now = time.time()

    def _age_days(p: Path) -> float:
        try:
            return (now - p.stat().st_mtime) / 86400.0
        except OSError:
            return float("inf")

    def _classify(latest_age_days: float, expected_max_days: float) -> str:
        if latest_age_days <= expected_max_days:
            return "HEALTHY"
        if latest_age_days <= expected_max_days * 3:
            return "WARN"
        return "DARK"

    checks = []

    # 1) Daily morning brief — expect daily.
    briefs = _list_brain("briefs", "morning_brief_*.md", limit=1)
    age = _age_days(_brain_root() / briefs[0]["path"]) if briefs else float("inf")
    checks.append(("morning_brief", _classify(age, 1.5), age, briefs[0]["path"] if briefs else "(none)"))

    # 2) Daily finance ingest — expect weekdays.
    fin = _list_brain("wiki/sources", "finance_news_*.md", limit=1)
    age = _age_days(_brain_root() / fin[0]["path"]) if fin else float("inf")
    checks.append(("finance_ingest", _classify(age, 3.0), age, fin[0]["path"] if fin else "(none)"))

    # 3) Daily topic rotation — expect Mon-Sat.
    concepts_recent = _list_brain("wiki/concepts", "*.md", limit=1)
    age = _age_days(_brain_root() / concepts_recent[0]["path"]) if concepts_recent else float("inf")
    checks.append(("topic_rotation_or_other_concept", _classify(age, 2.0), age,
                   concepts_recent[0]["path"] if concepts_recent else "(none)"))

    # 4) Daily journal — expect daily.
    dailies = _list_brain("wiki/daily", "*.md", limit=1)
    age = _age_days(_brain_root() / dailies[0]["path"]) if dailies else float("inf")
    checks.append(("daily_journal", _classify(age, 1.5), age, dailies[0]["path"] if dailies else "(none)"))

    # 5) Weekly meta-review — expect weekly.
    reviews_dir = Path(r"C:\Chloe\reviews")
    review_age = float("inf")
    if reviews_dir.exists():
        rev = sorted(reviews_dir.glob("*_meta.md"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if rev:
            review_age = (now - rev[0].stat().st_mtime) / 86400.0
    checks.append(("meta_review", _classify(review_age, 8.0), review_age, "C:/Chloe/reviews/"))

    # 6) Queue processor — expect every 2h.
    generated = _list_brain("generated", "*.md", limit=1)
    age = _age_days(_brain_root() / generated[0]["path"]) if generated else float("inf")
    checks.append(("queue_processor", _classify(age, 0.5), age, generated[0]["path"] if generated else "(none)"))

    # 7) Persona proposals (weekly).
    persona_recent = _list_brain("proposals", "persona_*.md", limit=1)
    age = _age_days(_brain_root() / persona_recent[0]["path"]) if persona_recent else float("inf")
    checks.append(("persona_proposals", _classify(age, 9.0), age,
                   persona_recent[0]["path"] if persona_recent else "(none)"))

    # Build report.
    body_lines = [
        "---", "type: report", "report_type: autonomous_audit",
        f"date: {today}",
        "generated_via: chloe_jobs.weekly-autonomous-audit",
        "---", "",
        f"# Autonomous-pipeline audit — {today}", "",
        f"Generated at {_now_iso()} local.", "",
        "## Pipeline status", "",
        "| Pipeline | Status | Last seen (days) | Latest output |",
        "|---|---|---|---|",
    ]
    for name, status, age_d, path in checks:
        age_str = f"{age_d:.1f}" if age_d != float("inf") else "—"
        body_lines.append(f"| {name} | **{status}** | {age_str} | `{path}` |")
    body_lines.append("")

    dark = [c for c in checks if c[1] == "DARK"]
    warn = [c for c in checks if c[1] == "WARN"]
    if dark:
        body_lines.append("## DARK pipelines (immediate attention)")
        for name, _, age_d, path in dark:
            body_lines.append(f"- **{name}** — last output {age_d:.1f} days ago. "
                              f"Check Windows Task Scheduler entry for `chloe-{name}`.")
        body_lines.append("")
    if warn:
        body_lines.append("## WARN pipelines (degraded)")
        for name, _, age_d, path in warn:
            body_lines.append(f"- {name} — last output {age_d:.1f} days ago.")
        body_lines.append("")
    if not dark and not warn:
        body_lines.append("## All pipelines HEALTHY")
        body_lines.append("")
        body_lines.append("No DARK or WARN states. Skip the rest of the report.")

    p = _write_brain(f"autonomous_status_{today}.md", "\n".join(body_lines))
    return f"wrote {p.name} | dark={len(dark)} warn={len(warn)}"


# ============================================================================
# JOB: weekly-persona-drift
# Compares recent assistant turns vs persona rules, flags violations.
# Was: chloe-weekly-persona-drift (Cowork, Sun 05:02)
# ============================================================================

def job_weekly_persona_drift() -> str:
    today = _today()
    persona = _persona_body()
    if not persona.strip():
        return "chloe_about.md empty; skipping"

    turns = _recent_turns(hours=168, modality=None, limit=400)  # 7 days
    assistant_turns = [t for t in turns if t.get("role") == "assistant"]
    if not assistant_turns:
        return "no assistant turns in last 7 days; skipping"

    sample = "\n---\n".join(
        f"[{_dt.datetime.fromtimestamp(t['ts']).strftime('%m-%d %H:%M')}] "
        f"({t.get('modality','?')}): {(t.get('content','') or '')[:600]}"
        for t in assistant_turns[:60])

    prompt = (
        "Audit Chloe's recent assistant turns for drift from her persona. "
        "Surface ONLY actual violations — banned phrasings used, clinical-list "
        "output where prose was specified, tonal-read leakage ('you sound "
        "off'), tail-pads ('let me know if you want more details'), pet names, "
        "hedging on knowable facts. Skip turns that follow the persona — "
        "the point is to surface drift, not praise compliance.\n\n"
        f"=== chloe_about.md (the source of truth) ===\n{persona[:8000]}\n\n"
        f"=== Recent assistant turns (last 7 days, sampled) ===\n"
        f"{sample[:14000]}\n\n"
        "Output a drift report in this exact markdown format. If zero "
        "violations found, return literally:\nNO_DRIFT\n\n"
        "## Drift verdict\n"
        "One paragraph: how bad is it this week?\n\n"
        "## Specific violations\n"
        "### 1. <one-line summary>\n"
        "**Rule violated:** <quote from chloe_about.md>\n"
        "**Evidence:** <quote from a turn, max 200 chars>\n"
        "**Severity:** high|medium|low\n"
        "**Suggested fix:** what specifically should Ed tighten in the persona "
        "doc, or what mid-flight correction would have helped?\n\n"
        "Cap at 5 violations.\n"
    )
    body = _heavy(prompt) or ""
    if "NO_DRIFT" in body or not body.strip():
        return "no drift detected"

    full = (
        "---\ntype: proposal\nproposal_type: persona_drift\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.weekly-persona-drift\nstatus: pending_review\n"
        "---\n\n"
        f"# Persona drift audit — {today}\n\n"
        f"{body.strip()}\n"
    )
    p = _write_brain(f"proposals/persona_drift_{today}.md", full)
    return f"wrote {p.name}"


# ============================================================================
# JOB: weekly-persona-evolution
# Proposes ADDITIONS to chloe_about.md from a week of patterns.
# Was: chloe-weekly-persona-evolution (Cowork, Sun 06:03)
# ============================================================================

def job_weekly_persona_evolution() -> str:
    today = _today()
    persona = _persona_body()
    turns = _recent_turns(hours=168, modality=None, limit=600)
    ctx_files = _list_brain("episodic", "CONTEXT-*.md", limit=7)
    ctx_text = ""
    for entry in ctx_files:
        try:
            ctx_text += "\n\n=== " + entry["path"] + " ===\n" + _read_brain(entry["path"])[:3000]
        except Exception:
            continue

    user_turns = [t for t in turns if t.get("role") == "user"]
    sample = "\n".join(
        f"[{_dt.datetime.fromtimestamp(t['ts']).strftime('%m-%d')} "
        f"{t.get('modality','?')}]: {(t.get('content','') or '')[:400]}"
        for t in user_turns[:80])

    prompt = (
        "Propose drop-in ADDITIONS to chloe_about.md based on the past 7 "
        "days of Ed's behavior + CONTEXT files. The Sunday-drift task "
        "catches violations; this task catches NEW patterns worth encoding.\n\n"
        f"=== Current chloe_about.md ===\n{persona[:8000]}\n\n"
        f"=== Episodic CONTEXT files ===\n{ctx_text[:8000]}\n\n"
        f"=== Sample of Ed's user-side turns ===\n{sample[:8000]}\n\n"
        "Return up to 4 proposals in this exact format. Each must include "
        "an `**Already in persona?**` line — duplicates are dropped. If "
        "zero new patterns meet the bar, return literally:\nNO_PROPOSALS\n\n"
        "### 1. <one-line proposed addition>\n"
        "**Section to add to:** Seed preferences | Tonal Awareness | "
        "Specific favorites | Knowledge anchors | etc.\n"
        "**Evidence:** quote 2 specific turns or context lines\n"
        "**Already in persona?** yes|no (you MUST check)\n"
        "**Proposed text (drop-in ready):**\n"
        "```markdown\n"
        "<exact addition text to splice into chloe_about.md>\n"
        "```\n"
    )
    body = _heavy(prompt) or ""
    if "NO_PROPOSALS" in body or not body.strip():
        return "no new patterns met the bar"
    full = (
        "---\ntype: proposal\nproposal_type: persona_evolution\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.weekly-persona-evolution\n"
        "status: pending_review\n---\n\n"
        f"# Persona evolution proposals — {today}\n\n"
        "Drop-in ADDITIONS to chloe_about.md. Splice the proposed text "
        "verbatim under the named section to accept.\n\n"
        f"{body.strip()}\n"
    )
    p = _write_brain(f"proposals/persona_{today}.md", full)
    return f"wrote {p.name}"


# ============================================================================
# JOB: weekly-cross-domain-synthesis
# Proposes verified-target wikilinks to connect recent / orphan / cross-domain.
# Was: chloe-weekly-cross-domain-synthesis (Cowork, Sun 09:00)
# ============================================================================

def job_weekly_cross_domain_synthesis() -> str:
    today = _today()

    # Inventory recent + all existing pages.
    all_pages: dict[str, dict] = {}
    for sub in ("entities", "concepts", "sources", "comparisons", "explorations"):
        for entry in _list_brain(f"wiki/{sub}", "*.md", limit=400):
            all_pages[entry["path"]] = entry
    recent_cutoff = time.time() - 7 * 86400
    recent = [e for e in all_pages.values() if e["mtime"] >= recent_cutoff]
    recent.sort(key=lambda e: -e["mtime"])
    # Trimmed 2026-05-20: was 30 pages x 600 chars + 400 paths (~30KB prompt),
    # which overflowed Ollama's 8192 ctx on Groq-quota-out days and produced
    # empty output. Top-10 x 300 chars + 150 paths keeps the prompt ~12KB so
    # the Ollama fallback still returns a real report.
    recent_top = recent[:10]

    # Build a digest of recent pages — title + first 300 chars.
    digest_parts = []
    for e in recent_top:
        try:
            content = _read_brain(e["path"])[:300]
        except Exception:
            continue
        digest_parts.append(f"=== {e['path']} ===\n{content}")
    digest = "\n\n".join(digest_parts)

    all_paths = sorted(all_pages.keys())
    paths_text = "\n".join(all_paths[:150])

    prompt = (
        "Propose specific wikilinks that would connect recent/orphan pages "
        "to existing pages. Hard rule: every link target MUST exist in the "
        "provided list (otherwise it becomes a ghost — we just shipped UI "
        "to clean those up). \n\n"
        "Three buckets:\n"
        "- A: Recently-added pages (last 7d) — what existing pages should "
        "link IN to them, and what existing pages should they link OUT to?\n"
        "- B: Orphans (pages mentioned in the digest with very few links) — "
        "missing in-links.\n"
        "- C: Cross-domain bridges — pairs spanning different top-level "
        "dirs (finance ↔ design, etc.) where the bridge is non-obvious.\n\n"
        f"=== All existing pages (link targets must come from here) ===\n"
        f"{paths_text}\n\n"
        f"=== Recent pages digest ===\n{digest[:4000]}\n\n"
        "Output a markdown report with the three buckets. Cap 20 proposals "
        "total. For each:\n"
        "- Specificity (1-5)\n"
        "- Likelihood Ed accepts (1-5)\n"
        "- Cross-domain bonus (+2 if so)\n"
        "Drop any proposal scoring under 7.\n\n"
        "Format each proposal as:\n"
        "- From `<source path>` → to `[[<target path>]]` — "
        "insert near phrase: '<exact phrase from source>' — "
        "*because: <one-line reason>* — score: N/15\n\n"
        + _NO_FLUFF
    )
    body = _heavy(prompt) or ""
    if not body.strip():
        return "LLM returned empty; not writing"
    full = (
        "---\ntype: proposal\nproposal_type: cross_domain_links\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.weekly-cross-domain-synthesis\n"
        "status: pending_review\n---\n\n"
        f"# Cross-domain link proposals — {today}\n\n"
        f"{body.strip()}\n"
    )
    p = _write_brain(f"proposals/cross_domain_{today}.md", full)
    return f"wrote {p.name}"


# ============================================================================
# JOB: friday-meta-review
# Weekly meta-review at Fridays 08:00 → C:\Chloe\reviews\<date>_meta.md.
# Was: chloe-friday-meta-review (Cowork, Fri 08:06)
# ============================================================================

def job_friday_meta_review() -> str:
    today = _today()
    contexts = _list_brain("episodic", "CONTEXT-*.md", limit=7)
    briefs = _list_brain("briefs", "morning_brief_*.md", limit=7)

    ctx_text = ""
    for entry in contexts:
        try:
            ctx_text += f"\n\n=== {entry['path']} ===\n" + _read_brain(entry["path"])[:2500]
        except Exception:
            continue
    brief_text = ""
    for entry in briefs:
        try:
            brief_text += f"\n\n=== {entry['path']} ===\n" + _read_brain(entry["path"])[:2000]
        except Exception:
            continue

    open_loops = _recall("open loop OR stuck OR pending OR todo", limit=10)
    open_loops_text = "\n".join(
        f"- ({h.get('modality','?')}): {(h.get('content','') or '')[:200]}"
        for h in open_loops)

    # Current facts (2026-09-03, Ed): the job never included this before --
    # confirmed live, "Ed's father's name addition request unresolved" was
    # listed as an open bug in a review generated MONTHS after facts.md
    # already had "my dad's name is Earle Wayne (added 2026-05-17)". The
    # semantic open-loop recall above can surface an old turn from BEFORE
    # something was resolved with nothing to tell the model it's since been
    # closed -- same fabrication-class bug as the wiki-write grounding fix
    # (09-01), just in this generation path instead. facts.md is the
    # cheapest, most authoritative "is this actually still open" check
    # available, so it's now fed in explicitly with an instruction to use it.
    facts_text = _facts_body()

    # Past meta-reviews.
    reviews_dir = Path(r"C:\Chloe\reviews")
    prior_reviews = ""
    if reviews_dir.exists():
        latest = sorted(reviews_dir.glob("*_meta.md"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if latest:
            try:
                prior_reviews = latest[0].read_text(encoding="utf-8")[:5000]
            except Exception:
                pass

    # Append-only behavior-change log if present.
    changelog = ""
    cl_path = HERE / "CHLOE_CHANGELOG.md"
    if cl_path.exists():
        try:
            changelog = cl_path.read_text(encoding="utf-8")[-6000:]
        except Exception:
            pass

    prompt = (
        f"Produce Friday's meta-review for the week ending {today}. Today's "
        f"date is exactly {today} -- use ONLY this date for the review's "
        f"own heading and 'this week'/'currently' framing. Do NOT reuse a "
        f"date that appears inside the 'Last meta-review' section below; "
        f"that section is prior context, not today.\n\n"
        f"Audience: Ed. Tone: terse, evidence-based, actionable.\n\n"
        f"GROUNDING RULE (mandatory): before listing anything under "
        f"'Top bugs / open loops' as still open, check it against the "
        f"CURRENT FACTS section below and the dates on the Open-loop recall "
        f"hits. If a fact file already documents that something is "
        f"resolved, or a hit is old and nothing since contradicts it being "
        f"resolved, do NOT list it as open -- either omit it or, if it's "
        f"otherwise worth a callout, note explicitly that it was resolved. "
        f"An item carried forward from the 'Last meta-review' section must "
        f"be re-verified against CURRENT FACTS the same way, not copied "
        f"forward on the assumption it's still true.\n\n"
        f"Do NOT write your own title/heading line -- start directly with "
        f"section 1, no leading '# ...' line.\n\n"
        "Sections:\n"
        "1. **Shipped this week** — concrete deliverables, with file paths.\n"
        "2. **Top bugs / open loops** — 3-5 items with severity.\n"
        "3. **Two proposed fixes** — for the top two bugs, specific changes.\n"
        "4. **Suggested focus next week** — one paragraph.\n"
        "5. **Persona / behavior notes** — patterns Chloe drifted in or "
        "improved on this week.\n\n"
        f"=== CURRENT FACTS (authoritative, check before calling anything "
        f"'still open') ===\n{facts_text[:4000]}\n\n"
        f"=== CONTEXT files (past 7d) ===\n{ctx_text[:9000]}\n\n"
        f"=== Morning briefs (past 7d) ===\n{brief_text[:8000]}\n\n"
        f"=== Open-loop recall hits (may include OLD, since-resolved items "
        f"-- verify against CURRENT FACTS above) ===\n{open_loops_text[:3000]}\n\n"
        f"=== CHLOE_CHANGELOG.md tail ===\n{changelog[:5000]}\n\n"
        f"=== Last meta-review, for continuity ONLY -- its date is NOT "
        f"today, and its open items are NOT guaranteed still open "
        f"===\n{prior_reviews[:4000]}\n\n"
        "Cap at 1200 words. Markdown only. " + _NO_FLUFF
    )
    body = _heavy(prompt) or ""
    if not body.strip():
        return "LLM returned empty; not writing"
    # Backstop (2026-09-03): the model sometimes writes its own leading
    # "# ..." heading despite the instruction above -- confirmed live, this
    # is exactly how a stale self-generated date ("Weekly Meta-Review —
    # 2026-08-26") ended up displayed as if it were the review's real date,
    # while the correctly-dated wrapper heading below it went unnoticed.
    # Strip any such leading heading line rather than trust the instruction
    # alone -- same "don't just ask nicely, verify" lesson as the wiki-write
    # citation fix.
    import re as _re_heading
    body = _re_heading.sub(r"^\s*#[^\n]*\n+", "", body.strip(), count=1).strip()
    if not body:
        return "LLM returned empty; not writing"

    reviews_dir.mkdir(parents=True, exist_ok=True)
    out_path = reviews_dir / f"{today}_meta.md"
    full = (
        "---\n"
        "type: review\n"
        "review_type: weekly_meta\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.friday-meta-review\n"
        "---\n\n"
        f"# Weekly meta-review — {today}\n\n"
        f"{body.strip()}\n"
    )
    out_path.write_text(full, encoding="utf-8")
    log.info(f"wrote {out_path}")

    # Code-fix hook (2026-05-20): if the review surfaced a concrete single-file
    # fix, draft a *pending* code_<date>_<slug>.md proposal so Ed has the patch
    # ready to /apply_proposal --dry-run, not just prose. Best-effort — a
    # failure here must never break the meta-review write above.
    proposal_note = ""
    try:
        proposal_note = _draft_code_proposal_from_review(body)
    except Exception as e:
        log.warning(f"meta-review code-fix hook failed: {type(e).__name__}: {e}")

    return f"wrote {out_path.name} ({len(full)} chars){proposal_note}"


def _draft_code_proposal_from_review(review_body: str) -> str:
    """Extract at most one concrete, small, single-file code fix from the
    meta-review prose and write it as a PENDING proposal via
    chloe_proposals.create_proposal (no apply). Returns a short log suffix
    or "" if nothing was drafted.

    The proposal is a draft for Ed to verify with `--dry-run` and apply by
    hand — the autonomous refused-targets list deliberately does NOT apply
    here, since nothing is auto-applied.
    """
    extract_prompt = (
        "From the weekly meta-review below, extract AT MOST ONE concrete, "
        "single-file code fix that is small (unified diff under ~40 lines) "
        "and names a clear target file in Chloe's source. If no such fix "
        "exists, return has_code_fix=false.\n\n"
        "Return STRICT JSON ONLY — no prose, no markdown fence:\n"
        '{"has_code_fix": <bool>, "target": "<jarvis-relative .py path, e.g. '
        'brain_wiring.py>", "slug": "<short_snake_case>", "title": "<short '
        'title>", "rationale": "<why, grounded in the review>", "diff": '
        '"<real unified diff: ---/+++/@@ hunks with 2-3 context lines>", '
        '"test_plan": "<how Ed verifies>"}\n\n'
        "Rules: exactly one .py target; the diff MUST be a real unified diff; "
        "if you are unsure the diff is correct, set has_code_fix=false rather "
        "than guess.\n\n"
        "=== META-REVIEW ===\n" + (review_body or "")[:8000]
    )
    try:
        raw = (_heavy(extract_prompt) or "").strip()
    except Exception as e:
        log.warning(f"meta-review code-fix extract failed: {e}")
        return ""
    if not raw:
        return ""
    # Strip a ```json ... ``` fence if the model added one anyway.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1] if "\n" in raw else ""
        if "```" in raw:
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if a == -1 or b <= a:
            return ""
        try:
            data = json.loads(raw[a:b + 1])
        except Exception:
            log.info("meta-review code-fix: unparseable JSON; skipping")
            return ""
    if not isinstance(data, dict) or not data.get("has_code_fix"):
        return ""

    target = str(data.get("target", "")).replace("\\", "/").strip().lstrip("/")
    if target.startswith("jarvis/"):
        target = target[len("jarvis/"):]
    diff = str(data.get("diff", "")).strip()
    if not target or not diff:
        return ""
    # No traversal, must be an existing .py under jarvis/ (no hallucinated paths).
    if not target.endswith(".py") or ".." in target.split("/"):
        return ""
    tgt = (HERE / target).resolve()
    if not str(tgt).startswith(str(HERE)) or not tgt.exists():
        log.info(f"meta-review code-fix: target {target!r} missing; skipping")
        return ""
    if diff.count("\n") > 60:
        log.info("meta-review code-fix: diff too large; skipping")
        return ""

    try:
        from chloe_proposals import create_proposal  # type: ignore
    except Exception as e:
        log.warning(f"chloe_proposals import failed: {e}")
        return ""
    slug = (str(data.get("slug") or "").strip() or None)
    title = (str(data.get("title") or "").strip() or None)
    rationale = (str(data.get("rationale") or "").strip() +
                 "\n\n_(LLM-drafted from the weekly meta-review. Verify with "
                 "`/apply_proposal <slug> --dry-run` before applying — the diff "
                 "may need hand-tuning to match current source context.)_")
    test_plan = (str(data.get("test_plan") or "").strip() or
                 "Run `/apply_proposal <slug> --dry-run`; confirm the hunk "
                 "anchors match current source. If clean, apply, then run "
                 "verify_proposals.bat.")
    try:
        p = create_proposal(target=target, kind="diff", rationale=rationale,
                             body=diff, test_plan=test_plan, slug=slug,
                             title=title)
    except Exception as e:
        log.warning(f"create_proposal failed: {type(e).__name__}: {e}")
        return ""
    log.info(f"drafted code proposal {p.name}")
    return f"; drafted code proposal {p.name}"


# ============================================================================
# Stage 4 — autonomous fix-recurring-errors job
# ============================================================================
# Scans the last 24h of logs for recurring traceback patterns. For each
# pattern >=3 occurrences, ast-introspects the implicated module, calls
# the heavy LLM for a fix proposal, requires self-rated confidence
# >=0.85 AND diff <50 lines, then if `autonomous_state.enabled` is True
# AND the watchdog rate limits pass: mints a token, applies, supervises.
#
# Default: DISABLED. Ed must flip the enable flag via `/autonomous on`
# or by editing C:\Chloe\brain\autonomous_state.json.
#
# REFUSED PATHS (the proposer can't touch its own brakes):
#   - chloe_proposals.py
#   - chloe_watchdog.py
#   - chloe_pending_confirms.py
#   - chloe_jobs.py
#   - chloe_capabilities.py
#   - chloe_mcp_server.py
#   - brain_wiring.py     (slash dispatch + ack-gate live here; risk is too high)
#   - jarvis.py           (voice + chat handlers; bricking-prone)


_AUTONOMOUS_REFUSED_TARGETS = frozenset({
    "chloe_proposals.py",
    "chloe_watchdog.py",
    "chloe_pending_confirms.py",
    "chloe_jobs.py",
    "chloe_capabilities.py",
    "chloe_mcp_server.py",
    "brain_wiring.py",
    "jarvis.py",
})


def _autonomous_state_path() -> Path:
    """Live next to the watchdog state under brain root."""
    p = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
    p.mkdir(parents=True, exist_ok=True)
    return p / "autonomous_state.json"


def _read_autonomous_state() -> dict:
    """Returns {enabled: bool, frozen_until: float, last_proposed_slug: str}."""
    p = _autonomous_state_path()
    default = {"enabled": False, "frozen_until": 0.0, "last_proposed_slug": ""}
    if not p.exists():
        return default
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        for k, v in default.items():
            s.setdefault(k, v)
        return s
    except (OSError, json.JSONDecodeError):
        return default


def _write_autonomous_state(state: dict) -> None:
    p = _autonomous_state_path()
    tmp = p.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _autonomous_gate() -> tuple[bool, str]:
    """Combined gate: state.enabled + freeze + watchdog rate limits."""
    state = _read_autonomous_state()
    if not state.get("enabled"):
        return False, "autonomous is DISABLED (see /autonomous on)"
    fz = state.get("frozen_until", 0.0)
    now = time.time()
    if fz > now:
        return False, (f"frozen for {int((fz - now) / 60)} more minutes "
                       f"(/autonomous freeze)")
    try:
        import chloe_watchdog
        ok, reason = chloe_watchdog.autonomous_can_apply_now()
        if not ok:
            return False, f"watchdog gate: {reason}"
    except Exception as e:
        return False, f"watchdog import failed: {type(e).__name__}: {e}"
    return True, ""


# Traceback signature normalization — strips line numbers, memory addresses,
# bare numbers, AND quoted-string error values so "KeyError: 'x'" and
# "KeyError: 'y'" hash identically. Same Python file + same line position +
# same exception class = same signature for grouping purposes.
import re as _re_auto
_TB_NORMALIZE = [
    (_re_auto.compile(r'File "[^"]+/([^/\\"]+)", line \d+'), r'File ".../\1", line N'),
    (_re_auto.compile(r'File "[^"]+\\([^/\\"]+)", line \d+'), r'File ".../\1", line N'),
    (_re_auto.compile(r'0x[0-9a-fA-F]+'),                    'ADDR'),
    (_re_auto.compile(r'\b\d{2,}\b'),                        'NUM'),
    # Quoted string values inside error messages — collapse contents so
    # "KeyError: 'foo'" and "KeyError: 'bar'" group to one signature.
    (_re_auto.compile(r"'[^']*'"),                           "'STR'"),
    (_re_auto.compile(r'"[^"]*"'),                           '"STR"'),
]


def _normalize_traceback(text: str) -> str:
    s = text
    for rx, repl in _TB_NORMALIZE:
        s = rx.sub(repl, s)
    return s.strip()


def _extract_tracebacks(log_text: str, hours: int = 24) -> list[dict]:
    """Walk a log file's lines and extract Python tracebacks from the last
    `hours`. Returns list of {ts, body, normalized}."""
    now = time.time()
    cutoff = now - (hours * 3600)
    out: list[dict] = []
    lines = log_text.splitlines()
    i = 0
    ts_re = _re_auto.compile(
        r'^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})')
    while i < len(lines):
        line = lines[i]
        # Heuristic traceback start
        if "Traceback (most recent call last):" in line:
            # Walk forward until non-indented non-Traceback line.
            j = i + 1
            body = [line]
            while j < len(lines):
                nxt = lines[j]
                if (nxt.startswith(("  ", "\t"))
                        or nxt.startswith("File ")
                        or nxt.startswith(("Exception", "Error", "TypeError",
                                           "ValueError", "RuntimeError",
                                           "KeyError", "AttributeError",
                                           "OSError", "IOError", "ImportError",
                                           "ModuleNotFoundError",
                                           "ConnectionError", "TimeoutError"))
                        or " Error:" in nxt or " Exception:" in nxt):
                    body.append(nxt)
                    j += 1
                    continue
                break
            block = "\n".join(body)
            # Try to find a timestamp in the next-prev line
            ts_line = lines[max(0, i - 1)]
            m = ts_re.match(ts_line)
            ts = 0.0
            if m:
                try:
                    ts = _dt.datetime.fromisoformat(
                        m.group(1).replace("T", " ")).timestamp()
                except ValueError:
                    pass
            if ts >= cutoff or ts == 0.0:
                out.append({
                    "ts": ts,
                    "body": block[:4000],
                    "normalized": _normalize_traceback(block)[:1500],
                })
            i = j
            continue
        i += 1
    return out


def _group_by_signature(tracebacks: list[dict]) -> list[dict]:
    """Bucket by normalized text. Returns list of
    {signature, count, sample_body, implicated_files}."""
    buckets: dict[str, dict] = {}
    # Extract implicated files from the RAW body, NOT the normalized signature.
    # Normalization rewrites `File "C:\\...\\x.py"` to `File ".../x.py"` (rules
    # 1-2) but then the generic quoted-string rule collapses it to `File "STR"`,
    # so the old `File "\.\.\./..."` match against the signature ALWAYS failed
    # → implicated_files was always empty → Stage 4 could never pick a target.
    # Capture the basename straight off the real path in the raw traceback.
    # (bug fixed 2026-05-20)
    file_re = _re_auto.compile(r'File "[^"]*[\\/]([^"\\/]+\.py)"')
    for tb in tracebacks:
        sig = tb["normalized"]
        b = buckets.setdefault(sig, {
            "signature": sig,
            "count": 0,
            "sample_body": tb["body"],
            "implicated_files": [],
        })
        b["count"] += 1
        for m in file_re.finditer(tb.get("body", "")):
            fn = m.group(1)
            if fn not in b["implicated_files"]:
                b["implicated_files"].append(fn)
    rows = list(buckets.values())
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


_AUTONOMOUS_FIX_PROMPT = (
    "You are reviewing a Python traceback that has fired {count} times in "
    "the last 24 hours of Chloe's logs. Chloe is a voice/chat assistant "
    "on Windows; she has the ability to apply small code patches to "
    "herself via a reviewed-proposal pipeline.\n\n"
    "TRACEBACK (recurring {count}x):\n"
    "---\n{traceback}\n---\n\n"
    "IMPLICATED MODULE SOURCE (`{module_name}`):\n"
    "---\n{module_source}\n---\n\n"
    "Propose a minimal fix. Constraints:\n"
    "- The fix MUST be a unified-diff hunk against `{module_name}`.\n"
    "- Diff <= 30 lines total (additions + removals).\n"
    "- No new external dependencies.\n"
    "- The fix must address the ROOT CAUSE shown in the traceback, not "
    "  hide the symptom.\n"
    "- If you're not at least 85% confident this fix is correct AND safe, "
    "  set confidence below 0.85 and the apply won't happen.\n\n"
    "Output ONLY a JSON object with this exact shape, no preamble:\n"
    "{{\n"
    '  "confidence": 0.0,        // float 0.0-1.0; self-rated\n'
    '  "rationale": "...",       // 2-4 sentences, root-cause focused\n'
    '  "diff": "@@ -... @@\\n..." // unified diff hunk against the module\n'
    "}}"
)


def job_autonomous_fix_recurring_errors() -> str:
    """Stage 4 — scan logs for repeating tracebacks, draft a fix per
    pattern, optionally apply + supervise via the watchdog.

    Default behavior (state.enabled=False): writes proposals to
    `proposals/code_<date>_autonomous_<slug>.md` for Ed's manual review,
    but does NOT apply. Flip `/autonomous on` to enable auto-apply.
    """
    log = logging.getLogger("chloe_jobs")
    log.info("=== Stage-4 autonomous fix-recurring-errors START ===")

    # Read both Chloe log files
    log_dir = HERE / "logs"
    sources = []
    for name in ("backend.log", "chloe_jobs.log"):
        p = log_dir / name
        if p.exists():
            try:
                sources.append((name, p.read_text(encoding="utf-8",
                                                  errors="replace")))
            except OSError as e:
                log.warning(f"couldn't read {name}: {e}")

    if not sources:
        return "no log files found under logs/"

    all_tbs: list[dict] = []
    for _, text in sources:
        all_tbs.extend(_extract_tracebacks(text, hours=24))

    if not all_tbs:
        return "no tracebacks in last 24h — nothing to fix"

    groups = _group_by_signature(all_tbs)
    # Filter to patterns >=3 occurrences
    candidates = [g for g in groups if g["count"] >= 3]
    if not candidates:
        return (f"{len(groups)} unique tracebacks but none >=3 occurrences "
                f"(top: {groups[0]['count']}x). Nothing autonomous-worthy.")

    # Always-on portion — even when disabled, write a digest report.
    digest = _write_autonomous_digest(groups, candidates)
    log.info(f"wrote digest: {digest}")

    # Gate the apply path. If disabled/frozen/rate-limited, return after
    # writing proposals as drafts.
    gate_ok, gate_reason = _autonomous_gate()

    proposals_written = 0
    proposals_applied = 0

    for cand in candidates[:3]:  # at most 3 candidates per run
        implicated = cand.get("implicated_files", [])
        # Filter to .py files under jarvis/ that aren't in the refused list.
        target_name = None
        for fn in implicated:
            base = os.path.basename(fn)
            if not base.endswith(".py"):
                continue
            if base in _AUTONOMOUS_REFUSED_TARGETS:
                continue
            module_path = HERE / base
            if module_path.exists():
                target_name = base
                break
        if not target_name:
            log.info(f"skipping pattern {cand['count']}x — no eligible "
                     f"target in {implicated}")
            continue

        # Synthesize the fix
        try:
            module_src = (HERE / target_name).read_text(encoding="utf-8")
        except OSError:
            continue
        # Cap module source for prompt (heavy LLM ctx + token cost)
        capped_src = module_src[:8000]

        try:
            from brain_wiring import _heavy_call
        except Exception as e:
            log.error(f"can't import _heavy_call: {e}")
            break

        prompt = _AUTONOMOUS_FIX_PROMPT.format(
            count=cand["count"],
            traceback=cand["sample_body"][:1500],
            module_name=target_name,
            module_source=capped_src,
        )
        try:
            raw = _heavy_call(prompt)
        except Exception as e:
            log.error(f"_heavy_call failed: {e}")
            continue

        # Parse JSON
        try:
            # LLMs often wrap in code fences — strip them
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = _re_auto.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = _re_auto.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned)
            confidence = float(parsed.get("confidence", 0))
            rationale = str(parsed.get("rationale", ""))
            diff = str(parsed.get("diff", ""))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            log.warning(f"LLM output didn't parse: {e}; raw[:200]={raw[:200]}")
            continue

        if not diff or "@@" not in diff:
            log.info(f"no valid diff in proposal — skipping")
            continue

        # Diff size cap
        diff_lines = diff.count("\n") + 1
        if diff_lines > 50:
            log.info(f"diff too large ({diff_lines} lines) — skipping")
            continue

        # Write the proposal
        slug = f"autonomous_{target_name.replace('.py', '')}_{int(time.time())}"
        try:
            import chloe_proposals
            path = chloe_proposals.create_proposal(
                target=f"jarvis/{target_name}",
                kind="diff",
                rationale=(f"AUTONOMOUS PROPOSAL — recurring traceback "
                           f"({cand['count']}x in last 24h).\n\n{rationale}\n\n"
                           f"Confidence: {confidence:.2f}"),
                body=diff,
                test_plan=("Watch the next backend.log for the same "
                           "traceback signature; if it stops firing in "
                           "the next 24h, the fix took."),
                slug=slug,
                title=f"Autonomous fix: {target_name} ({cand['count']}x error)",
            )
            proposals_written += 1
            log.info(f"wrote proposal {path.name} (confidence={confidence:.2f})")
        except Exception as e:
            log.error(f"create_proposal failed: {e}")
            continue

        # Apply gate
        if not gate_ok:
            log.info(f"NOT applying (gate: {gate_reason}); proposal "
                     f"left in proposals/ for manual review")
            continue
        if confidence < 0.85:
            log.info(f"confidence {confidence:.2f} < 0.85 — not applying")
            continue

        # Apply via the watchdog-supervised path. Use Tier-1 directly
        # since we're invoking from inside Chloe's process; no token
        # round-trip needed.
        try:
            import chloe_watchdog
            result = chloe_proposals.apply_proposal(slug, dry_run=False)
            if not result.get("ok"):
                chloe_watchdog.record_autonomous_failure(
                    slug, f"apply failed: {result.get('error')}")
                log.error(f"apply failed: {result.get('error')}")
                continue
            chloe_watchdog.record_autonomous_apply(slug)
            log.info(f"APPLIED {slug} — entering watchdog supervision")
            # Supervise (blocking, ~5 min)
            sup = chloe_watchdog.supervise_apply(
                slug, watch_minutes=5, expected_to_restart=False)
            log.info(f"watch outcome: {sup}")
            if sup.get("outcome") == "reverted":
                log.warning(f"WATCHDOG REVERTED {slug}: {sup.get('reason')}")
                break  # stop — consecutive_failures bumped, gate will lock
            proposals_applied += 1
        except Exception as e:
            log.error(f"apply pipeline crashed: {e}")
            break

    summary = (
        f"scanned {len(all_tbs)} tracebacks → {len(groups)} unique → "
        f"{len(candidates)} candidates >=3x · "
        f"wrote {proposals_written} proposal(s) · "
        f"applied {proposals_applied} · "
        f"gate: {'OPEN' if gate_ok else 'CLOSED ('+gate_reason+')'}"
    )
    log.info(summary)
    return summary


def _write_autonomous_digest(all_groups: list[dict],
                              candidates: list[dict]) -> str:
    """Write a daily digest of error patterns under brain/proposals/
    even when no autonomous applies happen."""
    today = _dt.date.today().isoformat()
    brain_root = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
    out_dir = brain_root / "proposals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"autonomous_digest_{today}.md"

    lines = [
        f"# Autonomous error digest — {today}",
        f"",
        f"Total unique traceback signatures (24h): {len(all_groups)}",
        f"Candidates for autonomous fix (count >= 3): {len(candidates)}",
        f"",
        f"## Top patterns",
        f"",
    ]
    for g in all_groups[:10]:
        files = g.get("implicated_files") or []
        first_file = files[0] if files else "(no file path matched)"
        lines.append(f"### {g['count']}x — {first_file}")
        lines.append("```")
        lines.append(g["sample_body"][:600])
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


# ============================================================================
# Registry + CLI
# ============================================================================

# ============================================================================
# JOB: daily-arcade-distill
# Mines arcade_watch + nearby chat turns from the last 24h, distils durable
# observations into the per-game brain page (brain/games/<slug>.md), and
# surfaces high-confidence personal-preference candidates as facts proposals.
# ============================================================================

def job_daily_arcade_distill() -> str:
    today = _today()
    arcade = _recent_turns(hours=24, modality="arcade_watch", limit=400)
    chat = _recent_turns(hours=24, modality="chat", limit=400)
    if not arcade and not chat:
        return "no arcade_watch or chat turns in last 24h, skipping"

    # Group by which game was active around each turn. Heuristic: the per-game
    # brain page already aggregates per session, so for this distill pass we
    # focus on (a) facts-worthy preferences/jokes and (b) any per-game
    # observations that survived a session block but never made it into a
    # page (e.g. user typed about a game outside watch mode).
    transcript = "\n".join(
        f"[{_dt.datetime.fromtimestamp(t['ts']).strftime('%H:%M')}] "
        f"{t.get('modality') or '?'}/{t['role']}: {t['content'][:400]}"
        for t in (arcade + chat)
        if (t.get("content") or "").strip()
    )

    facts_body = _facts_body()

    prompt = (
        "Below are the last 24h of Chloe's arcade-watch commentary and chat "
        "with Ed. Extract up to 4 DURABLE candidates for long-term memory:\n"
        "- gaming preferences (favourite franchises, starters, playstyle)\n"
        "- recurring in-jokes between Ed and Chloe about the games\n"
        "- specific games Ed is playing and what stage / progress he's at\n"
        "- emotional reactions worth remembering (frustration, nostalgia)\n\n"
        "Skip:\n- generic reactions, one-off jokes, screen descriptions\n"
        "- anything already in facts.md (below)\n"
        "- ephemera tied to a single moment\n\n"
        f"=== Existing facts.md ===\n{facts_body[:6000]}\n\n"
        f"=== Last 24h transcripts (arcade_watch + chat) ===\n"
        f"{transcript[:12000]}\n\n"
        "If zero candidates clear the bar, return literally:\nNO_CANDIDATES\n\n"
        "Otherwise return up to 4 candidates in this exact format:\n\n"
        "### 1. <one-line fact about Ed, no period>\n"
        "**Source:** arcade_watch | chat\n"
        "**Evidence:** brief quote (max 150 chars)\n"
        "**Confidence:** high|medium|low\n"
        "**Why it qualifies:** 1-2 sentences\n"
    )
    body = _heavy(prompt) or ""
    if "NO_CANDIDATES" in body or not body.strip():
        return "no arcade candidates passed the bar"

    full = (
        "---\n"
        "type: proposal\n"
        "proposal_type: facts\n"
        f"date: {today}\n"
        "generated_via: chloe_jobs.daily-arcade-distill\n"
        "source_channel: arcade_watch+chat\n"
        "status: pending_review\n"
        "---\n\n"
        f"# Fact-candidate proposals (arcade) — {today}\n\n"
        "Surfaced from last 24h of arcade_watch + chat modalities. "
        "Approve via `mcp__chloe__add_fact(<text>)` or just edit "
        "`facts.md` directly.\n\n"
        "## Candidates\n\n"
        f"{body.strip()}\n"
    )
    p = _write_brain(f"proposals/facts_from_arcade_{today}.md", full)
    return f"wrote {p.name}"


def job_daily_reflect() -> str:
    """Phase 4E reflection: read recent turns + facts and merge higher-order
    observations about Ed into the typed user-model (ed_model.json) via
    chloe_ed_profile.build(). Turns the conversation log into understanding."""
    try:
        import chloe_reflect
        out = chloe_reflect.reflect(n_turns=40)
        return ("reflect: merged insights into ed_model"
                if out else "reflect: no update produced")
    except Exception as e:
        return f"reflect: failed — {e}"


JOBS = {
    "daily-reflect":                   job_daily_reflect,
    "daily-journal-stub":              job_daily_journal_stub,
    "daily-cowork-fact-extract":       job_daily_cowork_fact_extract,
    "daily-voice-persona-mining":      job_daily_voice_persona_mining,
    "daily-arcade-distill":            job_daily_arcade_distill,
    "daily-topic-rotation":            job_daily_topic_rotation,
    "daily-finance-ingest":            job_daily_finance_ingest,
    "daily-morning-brief":             job_daily_morning_brief,
    "daily-critical-thinking-exercise": job_daily_critical_thinking_exercise,
    "weekly-backup":                   job_weekly_backup,
    "weekly-autonomous-audit":         job_weekly_autonomous_audit,
    "weekly-persona-drift":            job_weekly_persona_drift,
    "weekly-persona-evolution":        job_weekly_persona_evolution,
    "weekly-cross-domain-synthesis":   job_weekly_cross_domain_synthesis,
    "friday-meta-review":              job_friday_meta_review,
    "autonomous-fix-recurring-errors": job_autonomous_fix_recurring_errors,
}

# Human-readable schedule per job — mirrors register_chloe_jobs.ps1 exactly.
# Used by the HUD CH03 panel to show "when does this run next" hints.
SCHEDULES = {
    "daily-reflect":                    "21:00 daily",
    "daily-journal-stub":               "23:00 daily",
    "daily-cowork-fact-extract":        "22:30 daily",
    "daily-voice-persona-mining":       "22:00 daily",
    "daily-arcade-distill":             "22:15 daily",
    "daily-topic-rotation":             "06:00 Mon-Sat",
    "daily-finance-ingest":             "07:30 weekdays",
    "daily-morning-brief":              "07:00 daily",
    "daily-critical-thinking-exercise": "13:00 weekdays",
    "weekly-backup":                    "Sun 03:00",
    "weekly-autonomous-audit":          "Sun 04:00",
    "weekly-persona-drift":             "Sun 05:00",
    "weekly-persona-evolution":         "Sun 06:00",
    "weekly-cross-domain-synthesis":    "Sun 09:00",
    "friday-meta-review":               "Fri 08:00",
    "autonomous-fix-recurring-errors":  "04:00 daily (NOT auto-scheduled — manual until trusted)",
}


# ─── Run-state introspection (consumed by HUD CH03 channel) ─────────────────

import re as _re

# Track jobs currently running via WS-triggered "run now". Thread-safe.
import threading as _threading
_RUNNING_LOCK = _threading.Lock()
_RUNNING: set[str] = set()


def _parse_log_tail(max_lines: int = 4000) -> dict:
    """Parse logs/chloe_jobs.log tail, return {job_name: {last_started_ts,
    last_completed_ts, last_status, last_result, last_duration_s}}."""
    log_path = HERE / "logs" / "chloe_jobs.log"
    if not log_path.exists():
        return {}
    try:
        # Read all, take last `max_lines` to keep this cheap.
        lines = log_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines()[-max_lines:]
    except Exception:
        return {}

    start_re = _re.compile(
        r"^(\S+ \S+) \[\w+\] chloe_jobs - === START (\S+) ===")
    ok_re = _re.compile(
        r"^(\S+ \S+) \[\w+\] chloe_jobs - === OK\s+(\S+) \(([\d.]+)s\): (.*)$")
    fail_re = _re.compile(
        r"^(\S+ \S+) \[\w+\] chloe_jobs - === FAIL\s+(\S+) \(([\d.]+)s\): (.*)$")

    def _parse_ts(s: str) -> float:
        try:
            return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0.0

    state: dict = {}
    for line in lines:
        m = start_re.match(line)
        if m:
            ts, name = m.group(1), m.group(2)
            d = state.setdefault(name, {})
            d["last_started_ts"] = _parse_ts(ts)
            continue
        m = ok_re.match(line) or fail_re.match(line)
        if m:
            ts, name, dur, result = m.group(1), m.group(2), m.group(3), m.group(4)
            d = state.setdefault(name, {})
            d["last_completed_ts"] = _parse_ts(ts)
            d["last_duration_s"] = float(dur)
            d["last_result"] = result[:300]
            d["last_status"] = "OK" if "=== OK " in line else "FAIL"
    return state


def state() -> dict:
    """Public: full HUD-facing snapshot of every job's runtime status.

    Returns:
      {
        "jobs": [
          { "name", "schedule", "last_started_ts", "last_completed_ts",
            "last_status", "last_result", "last_duration_s",
            "running", "health", "age_hours" },
          ...
        ],
        "summary": {
          "total": 13, "ran_today": N, "ok": N, "fail": N, "running": N
        },
        "computed_at": <epoch>
      }
    Health classification:
      - 'running' if currently in-flight
      - 'never_run' if no log entry
      - 'fail' if last_status == FAIL
      - 'healthy' if last completion within expected window for the cron
      - 'stale' otherwise
    """
    log_state = _parse_log_tail()
    with _RUNNING_LOCK:
        running_now = set(_RUNNING)
    now = time.time()
    today_start = _dt.datetime.combine(
        _dt.date.today(), _dt.time.min).timestamp()

    # Per-job expected max age (hours) for "healthy" classification.
    expected_max = {
        "daily-journal-stub":               36,   # daily
        "daily-cowork-fact-extract":        36,
        "daily-voice-persona-mining":       36,
        "daily-topic-rotation":             48,   # Mon-Sat
        "daily-finance-ingest":             96,   # weekdays only
        "daily-morning-brief":              36,
        "daily-critical-thinking-exercise": 96,   # weekdays
        "weekly-backup":                    192,  # 8 days
        "weekly-autonomous-audit":          192,
        "weekly-persona-drift":             192,
        "weekly-persona-evolution":         192,
        "weekly-cross-domain-synthesis":    192,
        "friday-meta-review":               192,
    }

    jobs_out = []
    ran_today = 0
    ok_count = 0
    fail_count = 0
    for name in JOBS.keys():
        ls = log_state.get(name, {})
        last_done = ls.get("last_completed_ts", 0.0)
        last_status = ls.get("last_status", "")
        running = name in running_now
        if running:
            health = "running"
        elif not last_done:
            health = "never_run"
        elif last_status == "FAIL":
            health = "fail"
        else:
            age_h = (now - last_done) / 3600.0
            health = "healthy" if age_h <= expected_max.get(name, 36) else "stale"
        age_hours = ((now - last_done) / 3600.0) if last_done else None
        if last_done >= today_start:
            ran_today += 1
        if last_status == "OK":
            ok_count += 1
        elif last_status == "FAIL":
            fail_count += 1
        jobs_out.append({
            "name":              name,
            "schedule":          SCHEDULES.get(name, ""),
            "last_started_ts":   ls.get("last_started_ts", 0.0),
            "last_completed_ts": last_done,
            "last_status":       last_status,
            "last_result":       ls.get("last_result", ""),
            "last_duration_s":   ls.get("last_duration_s", 0.0),
            "running":           running,
            "health":            health,
            "age_hours":         age_hours,
        })
    return {
        "jobs": jobs_out,
        "summary": {
            "total":    len(JOBS),
            "ran_today": ran_today,
            "ok":        ok_count,
            "fail":      fail_count,
            "running":   len(running_now),
        },
        "computed_at": now,
    }


def run_async(name: str, on_complete=None) -> bool:
    """Kick off a job in a background thread. Non-blocking.

    Returns True if started, False if unknown job or already running.
    on_complete (optional callable) is invoked with (name, result_str, ok: bool)
    after the job finishes — used by the WS layer to push a fresh state_update.
    """
    if name not in JOBS:
        return False
    with _RUNNING_LOCK:
        if name in _RUNNING:
            return False
        _RUNNING.add(name)

    def _worker():
        t0 = time.time()
        log.info(f"=== START {name} === (WS-triggered)")
        ok = False
        result = ""
        try:
            result = JOBS[name]()
            ok = True
            log.info(f"=== OK    {name} ({time.time()-t0:.1f}s): {result}")
        except Exception as e:
            result = f"{type(e).__name__}: {e}"
            log.error(f"=== FAIL  {name} ({time.time()-t0:.1f}s): {result}\n"
                      f"{traceback.format_exc()}")
        finally:
            with _RUNNING_LOCK:
                _RUNNING.discard(name)
            if on_complete:
                try:
                    on_complete(name, result, ok)
                except Exception as e:
                    log.error(f"on_complete callback failed: {e}")

    _threading.Thread(target=_worker, name=f"chloe-job-{name}",
                      daemon=True).start()
    return True


# ─── Catch-up: run jobs missed while the PC was off ─────────────────────────
# Jobs fire via external Windows Task Scheduler, which silently no-ops when the
# machine is asleep/off — that's why the off-hours jobs (night / Sunday /
# Friday) go dark. This sweep runs at Chloe boot (i.e. when the PC is back on):
# for each scheduled job it computes the most recent time the job SHOULD have
# fired and, if the job's last successful completion predates that, runs it now
# (staggered). Complements Task Scheduler rather than replacing it.

_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3,
        "fri": 4, "sat": 5, "sun": 6}


def _schedule_spec(sched: str):
    """Parse a SCHEDULES string -> (hour, minute, weekdays:set[int]) or None.

    Recognizes an 'HH:MM' time plus a day constraint: 'daily', 'weekdays'
    (Mon-Fri), 'Mon-Sat', or explicit day tokens ('Sun', 'Fri', ...). Returns
    None for unparseable schedules and for the Stage-4 'NOT auto-scheduled'
    job, which must never be caught up automatically."""
    if not sched:
        return None
    s = sched.lower()
    if "not auto-scheduled" in s:
        return None
    m = _re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    if "weekday" in s:
        days = {0, 1, 2, 3, 4}
    elif "mon-sat" in s:
        days = {0, 1, 2, 3, 4, 5}
    elif "daily" in s:
        days = set(range(7))
    else:
        toks = set(_re.findall(r"\b(mon|tue|wed|thu|fri|sat|sun)\b", s))
        days = {_DOW[t] for t in toks} if toks else set(range(7))
    return hh, mm, days


def _most_recent_fire(spec, now):
    """Latest datetime <= now matching the spec's time on an allowed weekday,
    looking back up to 8 days. None if no match (shouldn't happen for a valid
    spec)."""
    hh, mm, days = spec
    for back in range(0, 8):
        cand_date = (now - _dt.timedelta(days=back)).date()
        if cand_date.weekday() in days:
            cand = _dt.datetime.combine(cand_date, _dt.time(hh, mm))
            if cand <= now:
                return cand
    return None


def due_jobs(now=None, grace_min: int = 20) -> list[str]:
    """Names of scheduled jobs that missed their most-recent window — the job's
    last successful completion is older than the most recent time it should
    have fired. Skips fires within `grace_min` minutes (Task Scheduler is
    likely handling those right now) and the manual Stage-4 job."""
    now = now or _dt.datetime.now()
    log_state = _parse_log_tail()
    due = []
    for name in JOBS:
        spec = _schedule_spec(SCHEDULES.get(name, ""))
        if not spec:
            continue
        fire = _most_recent_fire(spec, now)
        if not fire:
            continue
        if (now - fire).total_seconds() < grace_min * 60:
            continue
        last_done = log_state.get(name, {}).get("last_completed_ts", 0.0)
        if last_done < fire.timestamp():
            due.append(name)
    return due


def run_catchup(stagger_s: int = 45, max_jobs: int = 8) -> list[str]:
    """Fire missed jobs (from due_jobs), staggered to avoid a thundering herd.
    Non-blocking — each job is launched via run_async on a daemon stagger
    timer. Returns the names scheduled to run."""
    try:
        due = due_jobs()
    except Exception as e:
        log.error(f"=== CATCHUP === due_jobs failed: {e}")
        return []
    fired = due[:max_jobs]
    for i, name in enumerate(fired):
        t = _threading.Timer(i * stagger_s, run_async, args=(name,))
        t.daemon = True
        t.start()
    log.info(f"=== CATCHUP === now={_now_iso()} due={due} firing={fired}"
             + (f" (capped at {max_jobs})" if len(due) > max_jobs else ""))
    return fired


def _sweep_decision(gap_s, since_sweep_s, tick_s=60, jump_factor=4,
                    interval_s=1800):
    """(should_sweep, woke). `woke` means the gap between ticks was far larger
    than tick_s — implying the thread was frozen because the machine slept and
    has now resumed, so we sweep immediately rather than waiting out the
    interval. Pure; unit-tested."""
    woke = gap_s > tick_s * jump_factor
    return (woke or since_sweep_s >= interval_s), woke


def run_periodic_catchup(interval_s: int = 1800, tick_s: int = 60,
                         jump_factor: int = 4) -> None:
    """Blocking loop (run in a daemon thread): re-run missed jobs every
    `interval_s`, AND immediately after the machine resumes from sleep.

    This is the path that matters when the PC sleeps with Chloe still running —
    no fresh boot sweep ever happens, but this loop's timer thread is frozen
    during sleep and resumes on wake, so the next tick sees a large wall-clock
    gap and sweeps right away. Polls every `tick_s`; a gap > tick_s*jump_factor
    means the thread was frozen (machine asleep). Never returns."""
    last_tick = time.time()
    last_sweep = last_tick  # boot sweep already covered "now"; wait a full interval
    log.info(f"=== CATCHUP === periodic loop started "
             f"(interval={interval_s}s tick={tick_s}s)")
    while True:
        time.sleep(tick_s)
        now = time.time()
        should, woke = _sweep_decision(now - last_tick, now - last_sweep,
                                       tick_s, jump_factor, interval_s)
        if should:
            try:
                fired = run_catchup()
                log.info(f"=== CATCHUP === periodic "
                         f"({'post-wake' if woke else 'interval'}) fired={fired}")
                last_sweep = now
            except Exception as e:
                log.error(f"periodic catchup sweep failed: {e}")
        last_tick = now


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", help="job name, or 'list' to enumerate")
    p.add_argument("--dry-run", action="store_true",
                   help="for jobs that support it (unused in current set)")
    args = p.parse_args(argv)

    if args.job == "list":
        print("\n".join(JOBS.keys()))
        return 0

    if args.job == "catchup":
        # Synchronous, loggable sweep — runs each missed job inline so this is
        # safe to wire to a single Windows "at logon / on wake" task as an
        # alternative to the in-process boot sweep.
        due = due_jobs()
        print("due: " + (", ".join(due) if due else "(none)"))
        for name in due:
            t0 = time.time()
            log.info(f"=== START {name} === (catchup)")
            try:
                result = JOBS[name]()
                log.info(f"=== OK    {name} ({time.time()-t0:.1f}s): {result}")
                print(f"  {name}: OK")
            except Exception as e:
                log.error(f"=== FAIL  {name} ({time.time()-t0:.1f}s): "
                          f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                print(f"  {name}: FAIL {type(e).__name__}: {e}")
        return 0

    if args.job not in JOBS:
        log.error(f"unknown job: {args.job}")
        log.info("available: " + ", ".join(JOBS.keys()))
        return 2

    t0 = time.time()
    log.info(f"=== START {args.job} ===")
    try:
        result = JOBS[args.job]()
        log.info(f"=== OK    {args.job} ({time.time()-t0:.1f}s): {result}")
        print(result)
        return 0
    except Exception as e:
        log.error(f"=== FAIL  {args.job} ({time.time()-t0:.1f}s): "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
