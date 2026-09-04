"""daily_ingest.py - autonomous Obsidian-daily-note ingestion.

Runs at 8am via Windows Task Scheduler. Reads yesterday's Obsidian daily
note from `wiki/daily/<yesterday>.md`, copies it into `BRAIN.raw_dir` with
a provenance header, and runs `BRAIN.ingest()` so the content surfaces in
the entity/concept extraction pipeline.

This complements wiki_watcher.bat, which already keeps daily notes embedded
for `/wiki` semantic search the moment they're saved. Ingest is the heavier
pass that materialises named entities and concepts into wiki/entities/
and wiki/concepts/, with full TLDR + cross-link generation.

Scheduling rationale: 8am runs *after* daily_context.bat (6am) by design.
The 6am job synthesises from chat history; the 8am job folds in whatever
Edward actually typed into Obsidian the previous day. Running them in
the other order would deny the context job that evening's hand-written
material.

CLI:
    python daily_ingest.py                  # ingest yesterday's note
    python daily_ingest.py --dry-run        # show plan, no writes
    python daily_ingest.py --date 2026-05-09  # backfill a specific date
                                              # (the date OF the note, not
                                              #  the date you're running on)

Exit codes:
    0 - success OR clean no-op (missing/empty daily note)
    1 - failure (ingest error, IO error, etc.)

The .env is loaded explicitly because Task Scheduler doesn't inherit the
parent environment, so we don't get GROQ_API_KEY for free. Same pattern
as daily_context.py.
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

# Make sibling imports work even when invoked by Task Scheduler with an odd cwd.
sys.path.insert(0, str(HERE))


# Below this many non-whitespace chars we treat the note as effectively empty
# and skip rather than ingest garbage. A header alone is ~20 chars; this gives
# a small buffer above that.
MIN_BODY_CHARS = 30


# --- Helpers ---------------------------------------------------------------

def _strip_frontmatter(body: str) -> str:
    """Strip a leading YAML frontmatter block if present.

    Obsidian daily notes often start with `---\\nproperty: value\\n---`.
    That metadata isn't useful for the ingest extractor and can mislead it
    into treating property values (tags, aliases, cssclass...) as durable
    concepts or entities. Cleaner to drop it before handing to BRAIN.ingest.
    """
    if not body.startswith("---"):
        return body
    lines = body.split("\n")
    if not lines or lines[0].strip() != "---":
        return body
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]).lstrip("\n")
    return body  # unterminated frontmatter - leave alone rather than mangle


def _wrap_for_ingest(date_iso: str, source_path: Path, body: str) -> str:
    """Prepend a small provenance header so the ingest pipeline has a clear
    title and so a human reviewing wiki/sources/ can see where this came
    from. The original frontmatter (if any) is stripped to keep property
    values out of the extractor."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    clean_body = _strip_frontmatter(body).strip()
    return (
        f"# Daily Note - {date_iso}\n\n"
        f"_Ingested from `{source_path}` at {ts}._\n\n"
        f"---\n\n"
        f"{clean_body}\n"
    )


def _slug_for(date_iso: str) -> str:
    """Slug-validate-safe filename for the raw/ copy.

    ISO dates are [0-9-]; the `daily_` prefix makes the row type obvious
    when browsing raw/. The full slug `daily_2026-05-12` satisfies the
    BRAIN validator (alnum + underscore + dash, <= 80 chars).
    """
    return f"daily_{date_iso}"


# --- Core ------------------------------------------------------------------

def ingest(target_date=None, dry_run=False):
    """Ingest the daily note for `target_date` (defaults to yesterday).

    Returns one of:
        {"ok": True, "skipped": "<reason>"}                      - no work
        {"ok": True, "dry_run": True, "raw_path": "..."}         - dry run
        {"ok": True, "slug": ..., "tldr": ..., "raw_path": ...,
                     "entities_touched": int, "concepts_touched": int}
        {"ok": False, "error": "..."}
    """
    target = target_date or (datetime.date.today() - datetime.timedelta(days=1))
    date_iso = target.isoformat()

    # Late import - matches daily_context.py. brain_wiring's module-load reads
    # CHLOE_BRAIN_ROOT from env, so _load_env() must have run first.
    from brain_wiring import BRAIN

    src_path = BRAIN.wiki_dir / "daily" / f"{date_iso}.md"
    slug = _slug_for(date_iso)

    if not src_path.exists():
        print(f"[daily-ingest] no daily note for {date_iso} at {src_path} "
              f"- skipping", flush=True)
        return {"ok": True, "skipped": f"no note for {date_iso}"}

    try:
        body = src_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": f"read failed: {type(e).__name__}: {e}"}

    if len(body.strip()) < MIN_BODY_CHARS:
        print(f"[daily-ingest] daily note for {date_iso} is effectively empty "
              f"({len(body.strip())} chars) - skipping", flush=True)
        return {"ok": True, "skipped": f"note for {date_iso} too short "
                                       f"({len(body.strip())} chars)"}

    wrapped = _wrap_for_ingest(date_iso, src_path, body)
    raw_path = BRAIN.raw_dir / f"{slug}.md"

    if dry_run:
        print(f"[daily-ingest] DRY RUN")
        print(f"  source:        {src_path}")
        print(f"  would write:   {raw_path}")
        print(f"  wrapped size:  {len(wrapped)} chars")
        print(f"  would call:    BRAIN.ingest('{slug}.md')")
        print("--- WRAPPED PREVIEW ---")
        preview = wrapped[:1500]
        print(preview + ("..." if len(wrapped) > 1500 else ""))
        return {"ok": True, "dry_run": True, "raw_path": str(raw_path)}

    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(wrapped, encoding="utf-8")
    except Exception as e:
        return {"ok": False, "error": f"raw write failed: {type(e).__name__}: {e}"}

    print(f"[daily-ingest] wrote {len(wrapped)} bytes to {raw_path}", flush=True)
    print(f"[daily-ingest] running BRAIN.ingest('{slug}.md')", flush=True)

    try:
        r = BRAIN.ingest(f"{slug}.md")
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"ok": False, "error": f"ingest failed: {type(e).__name__}: {e}",
                "raw_path": str(raw_path)}

    ents = len(r.get("entities_touched", []) or [])
    cons = len(r.get("concepts_touched", []) or [])
    print(f"[daily-ingest] OK - touched {ents} entities, {cons} concepts",
          flush=True)

    # Log into wiki/log.md so /lint and the wiki dashboard can see it ran.
    # Best-effort; a logging hiccup must not mask a successful ingest.
    try:
        BRAIN._log("daily_ingest",
                   f"{date_iso} ({ents} entities, {cons} concepts)")
    except Exception:
        pass

    return {
        "ok":               True,
        "slug":             r.get("slug", slug),
        "tldr":             r.get("tldr", ""),
        "raw_path":         str(raw_path),
        "entities_touched": ents,
        "concepts_touched": cons,
    }


# --- CLI -------------------------------------------------------------------

def _parse_args(argv):
    """Tiny hand-rolled parser. argparse would work but mirrors daily_context.py."""
    target = None
    dry = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--dry-run", "--dryrun", "-n"):
            dry = True
            i += 1
        elif a == "--date" and i + 1 < len(argv):
            try:
                target = datetime.date.fromisoformat(argv[i + 1])
            except ValueError:
                print(f"bad --date value: {argv[i+1]} "
                      f"(want YYYY-MM-DD)", file=sys.stderr)
                sys.exit(2)
            i += 2
        elif a in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            print(f"unknown arg: {a}\nrun with --help for usage", file=sys.stderr)
            sys.exit(2)
    return target, dry


if __name__ == "__main__":
    target, dry = _parse_args(sys.argv[1:])
    r = ingest(target_date=target, dry_run=dry)
    if r.get("ok"):
        if r.get("skipped"):
            print(f"[daily-ingest] SKIPPED: {r['skipped']}")
        elif r.get("dry_run"):
            print(f"[daily-ingest] dry run complete")
        else:
            print(f"[daily-ingest] OK -> {r['slug']} "
                  f"({r['entities_touched']} entities, "
                  f"{r['concepts_touched']} concepts) "
                  f"raw={r['raw_path']}")
        sys.exit(0)
    else:
        print(f"[daily-ingest] FAILED: {r.get('error')}", file=sys.stderr)
        sys.exit(1)
