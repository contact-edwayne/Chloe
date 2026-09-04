"""
tools/apply_fabrication_fixes.py — Applies Ed's two approved fixes from
the 2026-09-03 fabrication_check.py findings:

  1. Strip confirmed-dead (404) citation URLs from the LIVE section of a
     page -- replaces just the URL text with an honest
     "[dead link removed]" marker, in place, so citation numbering and
     any inline [N] prose references stay intact. Never touches archived
     "## Superseded duplicate: ..." blocks (13 of the 98 affected pages
     have these, from the 2026-09-03 concept-version dedup merge) --
     those are explicitly historical/frozen content, not live-recall
     content, so rewriting citations inside them serves no purpose and
     would blur an intentionally preserved record.

  2. Add ONE honest caveat line under a page's References/Sources
     heading when its numbered citations are bare source/service names
     with no URL (e.g. "[1] Bloomberg", "[2] OptionFlow") -- Ed decided
     these are worth fixing retroactively (2026-09-03), unlike a genuine
     book citation (author/title/publisher/year), which already IS a
     complete, correct citation format and is left untouched. Heuristic
     for "bare name" vs "real bibliographic citation": a real citation
     has a comma-separated author-like prefix or a parenthesized year;
     a bare name is short and has neither.

SAFETY: backs up every page's ORIGINAL full text (before either fix) to
C:\\Chloe\\brain\\dedup_reports\\fabrication_fix_backup_<date>\\<same
relative path>, one-time per page, before any write -- mirrors the
existing entitys_merge_backup_2026-09-03 convention. Writes a report of
exactly what changed. --dry-run previews every change with no writes at
all (default off; you must pass --apply to actually write).

Usage:
    python tools/apply_fabrication_fixes.py --dry-run
    python tools/apply_fabrication_fixes.py --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import date
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from url_verify import verify_url  # noqa: E402
from tools.ungrounded_pages_report import scan, _DEFAULT_MIGRATION_DATE  # noqa: E402

_DEFAULT_BRAIN_ROOT = Path(r"C:\Chloe\brain")

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_SUPERSEDED_MARKER_RE = re.compile(r"\n## Superseded duplicate:")
_CITATION_HEADING_RE = re.compile(
    r"^(#+\s*(?:sources|citations)\s*|References:)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBERED_CITE_LINE_RE = re.compile(r"^(\s*\[\d+\]\s*)(.+)$", re.MULTILINE)

_DEAD_MARKER = "[dead link removed 2026-09-03]"
_NAME_ONLY_CAVEAT = (
    "_(Named sources below have no direct link and are not independently "
    "verifiable from this page alone.)_"
)

# Real bibliographic citation heuristic: "Author, A. (YEAR)." or a
# parenthesized year anywhere -- either is a strong signal this is a
# proper citation, not a bare source name.
_LOOKS_BIBLIOGRAPHIC_RE = re.compile(r"\(\d{4}\)|,\s*[A-Z]\.\s")


def _clean_url(u: str) -> str:
    while u and u[-1] in ".,;:!?":
        u = u[:-1]
    if u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    return u


def _split_live_and_archive(text: str) -> tuple[str, str]:
    """Return (live, archive) -- archive is everything from the first
    '## Superseded duplicate:' marker onward, live is everything before
    it. archive is '' if the page has no archived blocks."""
    m = _SUPERSEDED_MARKER_RE.search(text)
    if not m:
        return text, ""
    return text[:m.start()], text[m.start():]


def _strip_dead_urls_in_live_section(live: str) -> tuple[str, list[str]]:
    """Replace every confirmed-dead URL found in `live` with
    _DEAD_MARKER. Returns (new_live, [urls actually stripped])."""
    raw_matches = sorted(set(_URL_RE.findall(live)))
    stripped = []
    for raw in raw_matches:
        cleaned = _clean_url(raw)
        if not verify_url(cleaned):
            if raw in live:
                live = live.replace(raw, _DEAD_MARKER)
                stripped.append(cleaned)
    return live, stripped


def _should_add_name_only_caveat(original_live: str) -> bool:
    """Decide against the ORIGINAL (pre-dead-url-strip) live text --
    NOT the post-strip text. Bug caught in dry-run review, 2026-09-03:
    checking post-strip text means a citation that HAD a real (now-
    stripped-as-dead) URL looks identical to one that never had a URL
    at all, so a page mixing a couple of dead weblinks with one genuine
    book citation got mis-flagged as "all bare" once the dead links were
    replaced by the marker text. Whether a citation counts as
    "name-only" must be judged on what the page ORIGINALLY said."""
    hm = _CITATION_HEADING_RE.search(original_live)
    if not hm:
        return False
    after_heading = original_live[hm.end():]
    lines = after_heading.splitlines()
    entries = []
    for ln in lines:
        cm = _NUMBERED_CITE_LINE_RE.match(ln)
        if cm:
            entries.append(cm.group(2).strip())
        elif ln.strip() == "":
            continue
        else:
            break
    if not entries:
        return False
    return all(
        not _URL_RE.search(e) and not _LOOKS_BIBLIOGRAPHIC_RE.search(e)
        for e in entries
    )


def _add_name_only_caveat(live: str) -> tuple[str, bool]:
    """Insert _NAME_ONLY_CAVEAT right after `live`'s References/Sources/
    Citations heading. Caller must have already decided this is wanted
    via _should_add_name_only_caveat(ORIGINAL live text) -- see that
    function's docstring for why the decision can't be made here, after
    dead-URL stripping has already run. Idempotent -- does nothing if
    the caveat is already present. Returns (new_live, added: bool)."""
    if _NAME_ONLY_CAVEAT in live:
        return live, False
    hm = _CITATION_HEADING_RE.search(live)
    if not hm:
        return live, False
    insert_at = hm.end()
    new_live = live[:insert_at] + "\n" + _NAME_ONLY_CAVEAT + live[insert_at:]
    return new_live, True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain-root", type=Path, default=_DEFAULT_BRAIN_ROOT)
    ap.add_argument("--migration-date", type=str, default=_DEFAULT_MIGRATION_DATE)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write changes. Without this, runs as a "
                         "dry run (default) -- previews every change, "
                         "writes nothing.")
    args = ap.parse_args()

    scanned = scan(args.brain_root)
    pre_migration_paths = [
        row["path"] for rows in scanned["jobs"].values() for row in rows
        if row["date"] < args.migration_date
    ]

    today = date.today().isoformat()
    backup_root = args.brain_root / "dedup_reports" / f"fabrication_fix_backup_{today}"
    report_lines = [f"# Fabrication fixes applied — {today}", ""]
    if not args.apply:
        report_lines.append("**DRY RUN — nothing was written.**")
    report_lines.append("")

    n_dead_fixed = 0
    n_caveat_added = 0
    n_pages_touched = 0

    for rel_path in pre_migration_paths:
        full = args.brain_root / rel_path
        try:
            original = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        live, archive = _split_live_and_archive(original)
        wants_caveat = _should_add_name_only_caveat(live)
        new_live, dead_stripped = _strip_dead_urls_in_live_section(live)
        caveat_added = False
        if wants_caveat:
            new_live, caveat_added = _add_name_only_caveat(new_live)

        if not dead_stripped and not caveat_added:
            continue

        n_pages_touched += 1
        n_dead_fixed += len(dead_stripped)
        n_caveat_added += 1 if caveat_added else 0

        report_lines.append(f"## `{rel_path}`")
        if dead_stripped:
            report_lines.append(f"- Stripped {len(dead_stripped)} dead URL(s):")
            for u in dead_stripped:
                report_lines.append(f"    - {u}")
        if caveat_added:
            report_lines.append("- Added name-only-citation caveat")
        if archive:
            report_lines.append("- (archived Superseded-duplicate section(s) "
                                "left untouched)")
        report_lines.append("")

        if args.apply:
            backup_path = backup_root / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                backup_path.write_text(original, encoding="utf-8")
            full.write_text(new_live + archive, encoding="utf-8")

    report_lines.insert(2 if not args.apply else 1,
                        f"Pages touched: **{n_pages_touched}**  |  "
                        f"Dead URLs stripped: **{n_dead_fixed}**  |  "
                        f"Name-only-citation caveats added: **{n_caveat_added}**")
    report_lines.insert(3 if not args.apply else 2, "")

    report_path = (args.brain_root / "dedup_reports" /
                  f"fabrication_fixes_{'applied' if args.apply else 'dryrun'}_{today}.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"{'APPLIED' if args.apply else 'DRY RUN'}: "
          f"{n_pages_touched} pages, {n_dead_fixed} dead URLs stripped, "
          f"{n_caveat_added} caveats added", flush=True)
    print(f"Report: {report_path}", flush=True)
    if args.apply:
        print(f"Backups: {backup_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
