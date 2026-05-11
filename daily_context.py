"""daily_context.py - autonomous morning synthesis.

Runs at 6am via Windows Task Scheduler. Reads recent activity across the
brain layers, calls Groq heavy with a structured synthesis prompt, and
writes the result to episodic/CONTEXT-<today>.md.

Sections produced:
    PROJECT STATUS    - where each active project stands
    OPEN LOOPS        - unresolved items from the past 7 days
    EMERGING PATTERNS - themes appearing across recent material
    SUGGESTED FOCUS   - one specific thing to work on today

CLI:
    python daily_context.py             # generate today's context
    python daily_context.py --dry-run   # show prompt + would-be path, no LLM call
    python daily_context.py --date 2026-05-09   # backfill a specific date

The .env is loaded explicitly because this script is invoked by Task
Scheduler, not by jarvis.py — so we don't get the environment for free.
"""
import os
import sys
import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def _load_env():
    """Load .env at script start so Task Scheduler context has GROQ_API_KEY."""
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

# Make sibling imports work
sys.path.insert(0, str(HERE))


# ─── Source gathering ───────────────────────────────────────────────────────
def _find_recent_files(folder: Path, hours: int) -> list:
    """List markdown files modified within the last N hours, newest first."""
    if not folder.exists():
        return []
    cutoff = datetime.datetime.now().timestamp() - hours * 3600
    out = []
    for p in folder.rglob("*.md"):
        try:
            if p.stat().st_mtime >= cutoff:
                out.append(p)
        except OSError:
            continue
    return sorted(out, key=lambda p: -p.stat().st_mtime)


def _read_truncated(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception as e:
        return f"<read failed: {type(e).__name__}: {e}>"


def gather_context(brain, target_date: datetime.date) -> dict:
    """Pull the source material for today's context.

    Returns a dict of named sections (yesterday_episodic, recent_episodic,
    facts, recent_raw, recent_wiki) — each value is a string blob.
    """
    yesterday = target_date - datetime.timedelta(days=1)
    week_ago  = target_date - datetime.timedelta(days=7)

    # Yesterday's episodic file
    yest_file = brain.episodic_dir / f"{yesterday.isoformat()}.md"
    yesterday_episodic = (yest_file.read_text(encoding="utf-8", errors="replace")
                          if yest_file.exists() else "(no episodic file for yesterday)")

    # Past 7 days of episodic for "open loops" context
    recent_episodic = []
    cur = week_ago
    while cur < target_date:
        ef = brain.episodic_dir / f"{cur.isoformat()}.md"
        if ef.exists():
            recent_episodic.append(f"### {cur.isoformat()}\n\n" +
                                   ef.read_text(encoding="utf-8", errors="replace")[:2000])
        cur += datetime.timedelta(days=1)
    recent_episodic_blob = "\n\n---\n\n".join(recent_episodic) if recent_episodic else "(no episodic activity in the last 7 days)"

    # Facts (from BRAIN.facts_only)
    try:
        facts_blob = brain.facts_only() or "(no durable facts)"
    except Exception as e:
        facts_blob = f"(facts read failed: {e})"

    # Last 48h of raw/ — sources newly added or refreshed
    recent_raw_files = _find_recent_files(brain.raw_dir, 48)
    if recent_raw_files:
        recent_raw_blob = "\n\n".join(
            f"### {p.name}\n\n{_read_truncated(p, 2000)}"
            for p in recent_raw_files[:5]
        )
    else:
        recent_raw_blob = "(no new sources in the last 48 hours)"

    # Last 48h of modified wiki pages — what the brain has actively touched
    recent_wiki_files = _find_recent_files(brain.wiki_dir, 48)
    if recent_wiki_files:
        recent_wiki_blob = "\n\n".join(
            f"### {p.relative_to(brain.wiki_dir).with_suffix('').as_posix()}\n\n"
            f"{_read_truncated(p, 1200)}"
            for p in recent_wiki_files[:8]
        )
    else:
        recent_wiki_blob = "(no wiki pages modified in the last 48 hours)"

    return {
        "target_date":         target_date.isoformat(),
        "yesterday_episodic":  yesterday_episodic,
        "recent_episodic":     recent_episodic_blob,
        "facts":               facts_blob,
        "recent_raw":          recent_raw_blob,
        "recent_wiki":         recent_wiki_blob,
    }


# ─── Prompt + LLM call ──────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are Chloe, generating Edward's daily context note for {date}.

Use ONLY the source material below — do not invent or assume anything not present in it. If a section has no signal, say so plainly.

Output a markdown document with these EXACT sections, in this order:

# DAILY CONTEXT - {date}

## Project Status
For each active project or topic that's been touched in the last 48 hours, give a 2-3 sentence status. If nothing has moved, say "no active project activity."

## Open Loops
Unresolved questions, half-finished tasks, or pending decisions surfaced in the past 7 days of activity. Each as a bullet. If none, say so.

## Emerging Patterns
Themes that appear across multiple sources, episodic entries, or facts. Connections that aren't already wikilinked. Each as a bullet with one-line justification. If nothing notable, say so.

## Suggested Focus
ONE specific thing Edward should work on today, with one sentence of reasoning grounded in the source material above. Pick based on momentum, blocked items, or strategic value.

Keep the whole document under 600 words. Use [[wikilink]] format when referencing pages from the wiki.

---

YESTERDAY'S EPISODIC ({yesterday}):
{yesterday_episodic}

---

PAST 7 DAYS OF EPISODIC:
{recent_episodic}

---

DURABLE FACTS:
{facts}

---

NEW SOURCES (last 48h):
{recent_raw}

---

RECENTLY UPDATED WIKI PAGES (last 48h):
{recent_wiki}
"""


def build_prompt(ctx: dict) -> str:
    target = datetime.date.fromisoformat(ctx["target_date"])
    return PROMPT_TEMPLATE.format(
        date=ctx["target_date"],
        yesterday=(target - datetime.timedelta(days=1)).isoformat(),
        yesterday_episodic=ctx["yesterday_episodic"],
        recent_episodic=ctx["recent_episodic"],
        facts=ctx["facts"],
        recent_raw=ctx["recent_raw"],
        recent_wiki=ctx["recent_wiki"],
    )


def generate(target_date: datetime.date = None, dry_run: bool = False) -> dict:
    """Generate today's daily context. Returns dict {ok, path, bytes, error?}."""
    target_date = target_date or datetime.date.today()
    from brain_wiring import BRAIN, chloe_llm_call

    ctx = gather_context(BRAIN, target_date)
    prompt = build_prompt(ctx)

    out_path = BRAIN.episodic_dir / f"CONTEXT-{target_date.isoformat()}.md"

    if dry_run:
        print(f"[daily-context] DRY RUN — would write to {out_path}")
        print(f"[daily-context] prompt size: {len(prompt)} chars")
        print("---PROMPT PREVIEW---")
        print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
        return {"ok": True, "path": str(out_path), "bytes": 0, "dry_run": True}

    print(f"[daily-context] generating context for {target_date}", flush=True)
    try:
        body = chloe_llm_call(prompt, "heavy")
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if not body or not body.strip():
        return {"ok": False, "error": "empty response from LLM"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"[daily-context] wrote {len(body)} bytes to {out_path}", flush=True)

    # Log into wiki/log.md so the lint pipeline can see it ran.
    try:
        BRAIN._log("daily_context", f"{target_date.isoformat()} ({len(body)} bytes)")
    except Exception:
        pass

    return {"ok": True, "path": str(out_path), "bytes": len(body)}


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    target = datetime.date.today()
    dry = False
    if "--dry-run" in args:
        dry = True
    if "--date" in args:
        i = args.index("--date")
        target = datetime.date.fromisoformat(args[i + 1])

    r = generate(target_date=target, dry_run=dry)
    if r.get("ok"):
        if dry:
            print("[daily-context] dry run complete")
        else:
            print(f"[daily-context] OK ({r['bytes']} bytes -> {r['path']})")
    else:
        print(f"[daily-context] FAILED: {r.get('error')}")
        sys.exit(1)
