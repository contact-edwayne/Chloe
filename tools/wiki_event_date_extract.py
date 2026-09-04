"""
tools/wiki_event_date_extract.py — READ-ONLY event_date extraction pass
over concepts/thinking_*.md.

Per Ed's instruction: try filename + body-text regex extraction FIRST and
report coverage before considering an LLM-assisted pass on the remainder.
Writes nothing to the wiki -- only a review file under
C:\\Chloe\\brain\\dedup_reports\\ listing what was extracted (or not) per
file, so Ed can eyeball correctness before anything gets backfilled into
frontmatter.

Two extraction layers, in priority order:
  1. FILENAME slug (e.g. "fed-cut-rates-july-2026" -> 2026-07-01). Cheap,
     but many filenames only carry a bare month ("fed-raises-rates-june")
     with no year, or no date-bearing token at all ("fed-raises-rate-50bps").
  2. BODY TEXT — regex for explicit "<Month> <Year>" / "at its <Month>
     <Year> meeting" phrases in the first ~2000 chars (the claim + premises
     sections, where the dated claim is stated). More reliable than the
     filename slug when it's present, since the model's own prose is
     usually fully-specified even when its filename slug abbreviated the
     date away.

A bare month with no year (filename or body) is resolved against the
page's own `generated_at` frontmatter date: if the named month is on or
after generated_at's month in the same year, use that year; otherwise
assume the next calendar year (these are forward-looking predictions,
so a claim generated in November about "March" almost always means next
March, not eight months ago).
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

WIKI_CONCEPTS = Path(r"C:\Chloe\brain\wiki\concepts")
OUT_DIR = Path(r"C:\Chloe\brain\dedup_reports")

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

# Filename: month optionally followed by a 4-digit year, as a separate
# hyphen-joined token (matches "...-july-2026", "...-june", "...-sept-2026").
FILENAME_DATE_RE = re.compile(
    rf"\b({_MONTH_ALT})(?:-(\d{{4}}))?\b", re.IGNORECASE)

# Body text: "<Month> <Year>" possibly preceded by "at its"/"in"/"by" and
# followed by "meeting". Captures the most common phrasing observed in the
# actual pages ("at its June 2026 meeting", "in July 2026").
BODY_DATE_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.IGNORECASE)

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
GENERATED_AT_RE = re.compile(r"^generated_at:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
EXISTING_EVENT_DATE_RE = re.compile(r"^event_date:\s*(\S+)", re.MULTILINE)


def _resolve_year(month: int, generated_at: date) -> int:
    """Bare month, no year given -- pick the year such that the resolved
    date is on/after generated_at (predictions point forward), same
    calendar year if the month hasn't passed yet this year, else next."""
    if month >= generated_at.month:
        return generated_at.year
    return generated_at.year + 1


def extract_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    fm_m = FRONTMATTER_RE.match(text)
    fm = fm_m.group(1) if fm_m else ""
    body = text[fm_m.end():] if fm_m else text

    ga_m = GENERATED_AT_RE.search(fm)
    generated_at = None
    if ga_m:
        try:
            generated_at = date.fromisoformat(ga_m.group(1))
        except ValueError:
            pass

    existing = EXISTING_EVENT_DATE_RE.search(fm)
    if existing:
        return {"file": path.name, "source": "frontmatter (already present)",
                 "event_date": existing.group(1), "confidence": "existing"}

    # Layer 2 first (body text) -- more often fully-specified than the slug.
    body_head = body[:2500]
    bm = BODY_DATE_RE.search(body_head)
    if bm:
        mon = MONTHS[bm.group(1).lower()]
        yr = int(bm.group(2))
        return {"file": path.name, "source": "body-text",
                 "event_date": f"{yr:04d}-{mon:02d}-01", "confidence": "month+year"}

    # Layer 1: filename slug.
    fm_match = FILENAME_DATE_RE.search(path.stem)
    if fm_match:
        mon = MONTHS[fm_match.group(1).lower()]
        yr_str = fm_match.group(2)
        if yr_str:
            return {"file": path.name, "source": "filename",
                     "event_date": f"{int(yr_str):04d}-{mon:02d}-01",
                     "confidence": "month+year"}
        if generated_at:
            yr = _resolve_year(mon, generated_at)
            return {"file": path.name, "source": "filename+generated_at inference",
                     "event_date": f"{yr:04d}-{mon:02d}-01",
                     "confidence": "month-only, year inferred"}
        return {"file": path.name, "source": "filename (month only, no generated_at to infer year)",
                 "event_date": None, "confidence": "insufficient"}

    return {"file": path.name, "source": "none found",
             "event_date": None, "confidence": "insufficient"}


def main() -> int:
    files = sorted(WIKI_CONCEPTS.glob("thinking_*.md"))
    results = [extract_one(f) for f in files]

    found = [r for r in results if r["event_date"]]
    missing = [r for r in results if not r["event_date"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"event_date_extraction_{date.today().isoformat()}.md"
    lines = []
    lines.append(f"# event_date extraction pass — {date.today().isoformat()}")
    lines.append("")
    lines.append("READ-ONLY. Nothing written to wiki/. Review before any frontmatter backfill.")
    lines.append("")
    lines.append(f"- Total thinking_* files: {len(files)}")
    lines.append(f"- Extracted via filename/body regex: {len(found)} "
                 f"({len(found) / len(files) * 100:.1f}%)")
    lines.append(f"- Unresolved (would need LLM pass): {len(missing)} "
                 f"({len(missing) / len(files) * 100:.1f}%)")
    lines.append("")
    lines.append("## Extracted (review these — automated month/year parsing, not verified)")
    lines.append("")
    lines.append("| file | event_date | source | confidence |")
    lines.append("|---|---|---|---|")
    for r in found:
        lines.append(f"| {r['file']} | {r['event_date']} | {r['source']} | {r['confidence']} |")
    lines.append("")
    lines.append("## Unresolved — candidates for an LLM-assisted pass")
    lines.append("")
    for r in missing:
        lines.append(f"- {r['file']} — {r['source']}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"Total: {len(files)}  Extracted: {len(found)} ({len(found)/len(files)*100:.1f}%)  "
          f"Unresolved: {len(missing)} ({len(missing)/len(files)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
