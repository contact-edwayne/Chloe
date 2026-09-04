"""
tools/backfill_point_in_time.py — retroactively mark existing
wiki/sources/web_*.md pages that contain price/rate/quantity claims with
point_in_time frontmatter + a prominent as-of body marker.

Regex-only (no LLM pass), reusing wiki_dedup.classify_point_in_time — the
exact same classifier _persist_brave_to_wiki now runs on every new page,
so old and new pages get identical treatment.

Classifies the SYNTHESIZED-ANSWER PORTION ONLY (the text between the
"_Synthesized from Brave search on ..._" line and the "## Citations"
heading) -- not the whole file. Classifying the whole file would pick up
keyword false-positives from citation link titles/domains (e.g. a
"Stock Price" article title, or "stocktwits.com" containing "stock" as a
substring) that have nothing to do with whether the synthesized ANSWER
itself made a quantity claim.

Does NOT run supersede/duplicate detection retroactively -- that wasn't
asked for the backfill specifically (only for the go-forward persist
path). Several of these pages likely still coexist with a same-subject
page post-backfill even after staleness gating; flagged in the report.

Idempotent: skips any file that already has point_in_time frontmatter.

Usage:
    python tools/backfill_point_in_time.py            # write changes
    python tools/backfill_point_in_time.py --dry-run   # report only
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from wiki_dedup import classify_point_in_time, POINT_IN_TIME_LABELS  # noqa: E402
import wiki_dedup as _wiki_dedup  # noqa: E402

SOURCES_DIR = Path(r"C:\Chloe\brain\wiki\sources")

_ANSWER_SPAN_RE = re.compile(
    r"_Synthesized from Brave search on [^_]*?_\n\n(.*?)\n\n## Citations",
    re.DOTALL,
)
_FRONTMATTER_RE = re.compile(r"^(---\n)(.*?)(\n---\n?)(.*)$", re.DOTALL)
_DATE_RE = re.compile(r"^date:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
_GENERATED_AT_RE = re.compile(r"^generated_at:\s*(\S+)", re.MULTILINE)
_ALREADY_MARKED_RE = re.compile(r"^point_in_time:\s*true", re.MULTILINE | re.IGNORECASE)
_H1_RE = re.compile(r"(^#\s+.*$)", re.MULTILINE)


def process_one(path: Path, dry_run: bool) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {"file": path.name, "kind": None, "action": "skip", "as_of": None}

    if _ALREADY_MARKED_RE.search(text):
        result["action"] = "already-marked"
        return result

    m = _ANSWER_SPAN_RE.search(text)
    answer_text = m.group(1) if m else text  # fall back to whole body if the
                                              # expected structure isn't found
    kind = classify_point_in_time(answer_text)
    result["kind"] = kind
    if not kind:
        result["action"] = "not-point-in-time"
        return result

    ga_m = _GENERATED_AT_RE.search(text)
    d_m = _DATE_RE.search(text)
    as_of = ga_m.group(1) if ga_m else (d_m.group(1) if d_m else "unknown date")
    result["as_of"] = as_of

    fm_m = _FRONTMATTER_RE.match(text)
    if not fm_m:
        result["action"] = "no-frontmatter-skip"
        return result

    new_frontmatter = (
        fm_m.group(1) + fm_m.group(2)
        + f"\npoint_in_time: true\npoint_in_time_kind: {kind}"
        + fm_m.group(3)
    )
    body_rest = fm_m.group(4)

    kind_label = POINT_IN_TIME_LABELS.get(kind, kind)
    ceiling = (_wiki_dedup.QUOTE_STALENESS_DAYS if kind == "quote"
              else _wiki_dedup.DATA_STALENESS_DAYS)
    marker = (
        f"\n> ⚠ **Point-in-time data ({kind_label}) — valid as of "
        f"{as_of} (backfilled retroactively).** Do not treat as current "
        f"without re-checking. (Ambient recall drops this page after "
        f"{ceiling:.0f} day(s).)\n"
    )
    h1_m = _H1_RE.search(body_rest)
    if h1_m:
        insert_at = h1_m.end()
        new_body_rest = body_rest[:insert_at] + "\n" + marker + body_rest[insert_at:]
    else:
        new_body_rest = marker + body_rest

    new_text = new_frontmatter + new_body_rest
    result["action"] = "would-mark" if dry_run else "marked"
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="Report what would change without writing.")
    args = ap.parse_args()

    files = sorted(SOURCES_DIR.glob("web_*.md"))
    results = [process_one(f, args.dry_run) for f in files]

    marked = [r for r in results if r["action"] in ("marked", "would-mark")]
    already = [r for r in results if r["action"] == "already-marked"]
    not_pit = [r for r in results if r["action"] == "not-point-in-time"]
    skipped = [r for r in results if r["action"] == "no-frontmatter-skip"]

    print(f"Total web_*.md pages scanned: {len(files)}")
    print(f"{'Would mark' if args.dry_run else 'Marked'}: {len(marked)}")
    print(f"Already marked (idempotent skip): {len(already)}")
    print(f"Not point-in-time: {len(not_pit)}")
    print(f"Skipped (no frontmatter found): {len(skipped)}")
    print()
    for r in marked:
        print(f"  [{r['kind']:5s}] as_of={r['as_of']:<25s} {r['file']}")
    if skipped:
        print()
        print("Skipped (no frontmatter block matched — inspect manually):")
        for r in skipped:
            print(f"  {r['file']}")
    print()
    print("NOTE: supersede/duplicate detection was NOT run retroactively — "
          "only the point_in_time marking above. Some of these pages may "
          "still coexist with a same-subject page even after staleness "
          "gating (e.g. two pages a few days apart, both inside their "
          "ceiling). Ask if you want a retroactive supersede pass too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
