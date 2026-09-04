"""
tools/ungrounded_pages_report.py — READ-ONLY report of pages written by the
four chloe_jobs.py daily jobs affected by the 2026-09-01 search-topic bug
(each called _search_llm(prompt) with no `topic` kwarg, so the Brave query
became the full multi-thousand-char prompt, which 422'd Brave's length cap
on every call, silently falling through to _heavy_call with zero web
retrieval -- see brain_wiring._search_call's docstring for the full
history). Writes nothing except its own dated report file. Does not touch,
mark, merge, or delete anything under C:\\Chloe\\brain\\.

IMPORTANT — NOT EVERY PAGE THESE JOBS EVER WROTE IS AFFECTED. These four
jobs existed before the Groq -> Brave/_search_call migration (this
session's Groq-removal work, ~2026-08-31). Before that date they called
Groq's compound-mini, which has built-in agentic web search with no
separate query-length constraint -- pages from that era are NOT subject to
this specific bug (they may have other, already-addressed issues from the
earlier fabrication work this session, but that's a different problem).
Only pages written ON OR AFTER the migration went through the broken
_search_call path.

This script cannot detect "written via Groq vs written via broken Brave"
from a page's content or frontmatter -- the job-identifying tag
(auto_generated / generated_via) is the same either way, since it names
the JOB, not the search backend it happened to use that day. What it CAN
do reliably is date each page and split the report at a migration-date
estimate, so the split is visible and adjustable rather than silently
wrong. --migration-date defaults to 2026-08-31 (this session's best
knowledge of when chloe_jobs.py started calling brain_wiring._search_call
for these four jobs) -- override it if you know a more precise date.

IDENTIFICATION IS ALSO NOT UNIFORM ACROSS THE FOUR JOBS -- this matters for
reading the report:

  job_daily_topic_rotation            -> wiki/concepts/*.md
                                          frontmatter: auto_generated: chloe_jobs.daily-topic-rotation
  job_daily_critical_thinking_exercise -> wiki/concepts/thinking_*.md
                                          frontmatter: auto_generated: chloe_jobs.daily-critical-thinking-exercise
  job_daily_finance_ingest            -> wiki/sources/finance_news_<date>.md
                                          frontmatter: generated_via: chloe_jobs.daily-finance-ingest
                                          (DIFFERENT field name than the two above -- 'generated_via', not
                                          'auto_generated'. Ed's original ask named "auto_generated frontmatter
                                          tag" as the identifier; that field alone would silently miss this job.)
  job_daily_morning_brief             -> briefs/morning_brief_<date>.md
                                          NO frontmatter tag at all -- the job writes the LLM's raw body with
                                          no wrapper. Identified by FILENAME PATTERN only. If Ed ever renamed or
                                          moved a brief, or another process writes to briefs/morning_brief_*.md,
                                          this would over-match; spot-check before acting on this job's rows.

DATE per page, in priority order (method recorded per row so it's never
ambiguous which one fired):
  1. `generated_at:` frontmatter field (topic_rotation, critical_thinking)
  2. `date:` frontmatter field (finance_ingest)
  3. YYYY-MM-DD embedded in the filename itself (morning_brief, and a
     fallback for the other three if their own frontmatter field is
     missing/malformed)
  4. File mtime (last resort, least trustworthy -- a later edit or a
     filesystem copy changes this without changing when the page was
     originally generated)

Usage:
    python tools/ungrounded_pages_report.py
    python tools/ungrounded_pages_report.py --migration-date 2026-08-28
    python tools/ungrounded_pages_report.py --brain-root "C:\\Chloe\\brain"
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

_DEFAULT_BRAIN_ROOT = Path(r"C:\Chloe\brain")
_DEFAULT_MIGRATION_DATE = "2026-08-31"

_DATE_IN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
_GENERATED_AT_RE = re.compile(r"^generated_at:\s*(\S+)", re.MULTILINE)
_DATE_FIELD_RE = re.compile(r"^date:\s*(\S+)", re.MULTILINE)

# tag_re confirms a file belongs to this job; tag_field names which
# frontmatter field carries it (None for morning_brief, which has none).
_JOB_SIGNATURES = [
    {
        "job": "job_daily_topic_rotation",
        "subdir": "wiki/concepts",
        "glob": "*.md",
        "tag_re": re.compile(
            r"^auto_generated:\s*chloe_jobs\.daily-topic-rotation\s*$",
            re.MULTILINE),
        "tag_field": "auto_generated",
    },
    {
        "job": "job_daily_critical_thinking_exercise",
        "subdir": "wiki/concepts",
        "glob": "thinking_*.md",
        "tag_re": re.compile(
            r"^auto_generated:\s*chloe_jobs\.daily-critical-thinking-exercise\s*$",
            re.MULTILINE),
        "tag_field": "auto_generated",
    },
    {
        "job": "job_daily_finance_ingest",
        "subdir": "wiki/sources",
        "glob": "finance_news_*.md",
        "tag_re": re.compile(
            r"^generated_via:\s*chloe_jobs\.daily-finance-ingest\s*$",
            re.MULTILINE),
        "tag_field": "generated_via",
    },
    {
        "job": "job_daily_morning_brief",
        "subdir": "briefs",
        "glob": "morning_brief_*.md",
        "tag_re": None,  # no frontmatter at all -- path pattern is the only signal
        "tag_field": None,
    },
]


def _extract_date(text: str, path: Path) -> tuple[str, str]:
    """Returns (date_string, method) using the priority order documented
    in the module docstring. date_string is 'unknown' only if every
    method fails (shouldn't happen -- mtime is always available)."""
    fm_m = _FRONTMATTER_RE.match(text)
    fm = fm_m.group(1) if fm_m else ""

    m = _GENERATED_AT_RE.search(fm)
    if m:
        return m.group(1)[:10], "generated_at frontmatter"

    m = _DATE_FIELD_RE.search(fm)
    if m:
        return m.group(1)[:10], "date frontmatter"

    m = _DATE_IN_NAME_RE.search(path.name)
    if m:
        return m.group(1), "filename"

    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).date().isoformat(), "file mtime (least trustworthy)"
    except OSError:
        return "unknown", "unavailable"


def scan(brain_root: Path) -> dict:
    """Read-only scan. Returns {'jobs': {job_name: [{'path', 'date',
    'date_method', 'size'}, ...]}, 'errors': [...]}. Every page these four
    jobs ever wrote, undated by migration status -- splitting that is the
    caller's job (see main()), since this function has no opinion on when
    the migration happened."""
    results = defaultdict(list)
    errors = []

    for sig in _JOB_SIGNATURES:
        target_dir = brain_root / sig["subdir"]
        if not target_dir.exists():
            continue
        for p in sorted(target_dir.glob(sig["glob"])):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                errors.append(f"{p}: {type(e).__name__}: {e}")
                continue

            if sig["tag_re"] is not None:
                if not sig["tag_re"].search(text):
                    continue  # filename pattern matched but tag didn't -- not this job

            date_str, date_method = _extract_date(text, p)
            results[sig["job"]].append({
                "path": str(p.relative_to(brain_root)).replace("\\", "/"),
                "date": date_str,
                "date_method": date_method,
                "size": len(text),
            })

    return {"jobs": dict(results), "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain-root", type=Path, default=_DEFAULT_BRAIN_ROOT)
    ap.add_argument("--migration-date", type=str, default=_DEFAULT_MIGRATION_DATE,
                    help=f"Best-knowledge date the Groq->Brave/_search_call "
                         f"migration landed for these jobs (default "
                         f"{_DEFAULT_MIGRATION_DATE}). Pages dated on or "
                         f"after this are the actually-affected subset; "
                         f"pages before it used Groq's agentic search and "
                         f"are unaffected by this bug.")
    args = ap.parse_args()

    if not args.brain_root.exists():
        print(f"brain root not found: {args.brain_root}", file=sys.stderr)
        return 1

    scanned = scan(args.brain_root)
    jobs = scanned["jobs"]
    errors = scanned["errors"]
    migration_date = args.migration_date

    today = date.today().isoformat()
    reports_dir = args.brain_root / "dedup_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"ungrounded_pages_report_{today}.md"

    lines = []
    lines.append(f"# chloe_jobs Search-Job Pages Report — {today}")
    lines.append("")
    lines.append("Read-only report. Nothing was marked, merged, edited, or deleted.")
    lines.append("")
    lines.append(
        "Lists every page written by the four chloe_jobs.py jobs that "
        "were silently calling brain_wiring._search_call with no `topic`, "
        "the 2026-09-01 bug (see brain_wiring._search_call's docstring). "
        "**Not every row here is actually ungrounded** -- only pages dated "
        f"on or after **{migration_date}** (this session's best-knowledge "
        "estimate of when these jobs started using Brave/_search_call "
        "instead of Groq compound-mini's built-in agentic search; pass "
        "--migration-date to correct it) went through the broken path. "
        "Earlier pages used Groq and are unaffected by THIS bug. Each "
        "job's table below is split into 'likely affected' and "
        "'pre-migration' accordingly.")
    lines.append("")
    lines.append(
        "**Identification is not uniform across the four jobs** -- see "
        "this script's module docstring for the full breakdown. In "
        "short: two jobs tag with `auto_generated:`, one tags with "
        "`generated_via:` (a different field), and one "
        "(job_daily_morning_brief) writes no frontmatter tag at all and "
        "is identified by filename pattern only.")
    lines.append("")

    total_affected = 0
    total_pre = 0
    for sig in _JOB_SIGNATURES:
        job = sig["job"]
        rows = jobs.get(job, [])
        affected = sorted([r for r in rows if r["date"] >= migration_date],
                          key=lambda r: r["date"])
        pre = sorted([r for r in rows if r["date"] < migration_date],
                     key=lambda r: r["date"])
        total_affected += len(affected)
        total_pre += len(pre)
        tag_desc = (f"`{sig['tag_field']}: ...` frontmatter field"
                   if sig["tag_field"] else "filename pattern only, NO frontmatter tag")

        lines.append(f"## {job}")
        lines.append("")
        lines.append(f"- Location: `{sig['subdir']}/{sig['glob']}`")
        lines.append(f"- Identified by: {tag_desc}")
        lines.append(f"- Total pages found: {len(rows)}")
        lines.append(f"- **Likely affected** (>= {migration_date}): **{len(affected)}**")
        lines.append(f"- Pre-migration, likely Groq-grounded (< {migration_date}): {len(pre)}")
        lines.append("")
        if affected:
            lines.append(f"### Likely affected ({len(affected)})")
            lines.append("")
            lines.append("| Date | Date source | Path | Size |")
            lines.append("|---|---|---|---|")
            for r in affected:
                lines.append(f"| {r['date']} | {r['date_method']} | "
                            f"`{r['path']}` | {r['size']:,} chars |")
            lines.append("")
        if pre:
            lines.append(f"<details><summary>Pre-migration, not part of this "
                         f"bug ({len(pre)}) — click to expand</summary>\n")
            lines.append("| Date | Date source | Path | Size |")
            lines.append("|---|---|---|---|")
            for r in pre:
                lines.append(f"| {r['date']} | {r['date_method']} | "
                            f"`{r['path']}` | {r['size']:,} chars |")
            lines.append("\n</details>")
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Likely-affected pages (>= {migration_date}) across all four jobs: **{total_affected}**")
    lines.append(f"- Pre-migration pages (< {migration_date}, not part of this bug): {total_pre}")
    lines.append(f"- Total pages from these four jobs, either era: {total_affected + total_pre}")
    if errors:
        lines.append(f"- Files that couldn't be read ({len(errors)}):")
        for e in errors:
            lines.append(f"  - {e}")
    lines.append("")
    lines.append(
        "No action taken. Decide whether to retroactively mark the "
        "likely-affected pages with the no-retrieval notice, or delete "
        "and regenerate them now that _search_call actually passes a "
        "real topic.")

    report_text = "\n".join(lines) + "\n"
    out_path.write_text(report_text, encoding="utf-8")

    print(f"Scanned {args.brain_root}")
    print(f"Migration-date split: {migration_date} (pass --migration-date to adjust)")
    print(f"Likely-affected total: {total_affected}   Pre-migration total: {total_pre}")
    for sig in _JOB_SIGNATURES:
        rows = jobs.get(sig["job"], [])
        affected = [r for r in rows if r["date"] >= migration_date]
        pre = [r for r in rows if r["date"] < migration_date]
        print(f"  {sig['job']}: {len(affected)} affected, {len(pre)} pre-migration")
    if errors:
        print(f"  {len(errors)} file(s) could not be read -- see report")
    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
