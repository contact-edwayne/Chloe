"""
tools/fabrication_check.py — READ-ONLY content-level fabrication check
for the ~329 pre-2026-08-31-migration pages written by chloe_jobs.py's
four daily jobs (job_daily_topic_rotation, job_daily_critical_thinking_exercise,
job_daily_finance_ingest, job_daily_morning_brief).

Why this is a SEPARATE script from tools/ungrounded_pages_report.py: that
script explicitly documents it "cannot detect [fabrication] from a page's
content" -- it only classifies pages by DATE against the Groq->Brave
migration. This script is the actual content check that report called
for but never ran: it reuses that script's scan()/migration-date split
to get the same pre-migration page set (no need to re-derive it), then
checks each page's content for the fabricated-citation pattern found in
8 post-migration pages on 2026-09-01 (see brain_wiring.py's
_search_call/handle_wiki_write docstrings for that history):
  1. Dead/unreachable citation URLs (url_verify.py's HEAD-check, the
     same live verifier _search_call already uses going forward).
  2. Self-referential wikilinks -- a page citing [[its own slug]] as if
     it were an external source.
  3. Name-only "citations" with no URL at all ([1] Bloomberg, [2]
     OptionFlow, ...) -- not proof of fabrication by itself (Groq
     compound-mini doesn't always return a URL), but unverifiable by
     construction, so surfaced as its own category rather than silently
     skipped.

Deliberately does NOT flag "Confidence the claim is true: NN%" as
fabrication: that's a REQUIRED section of the critical-thinking-exercise
page template (job_daily_critical_thinking_exercise's own prompt asks
for it), confirmed by inspecting real thinking_*.md pages -- a stated
confidence number there is by design, not an invented statistic. Ed's
earlier "remove confidence-percentage prompt section" fix (this
session's memory: obs 961) was a different page type. This script does
NOT attempt to distinguish a legitimate calibration number from a
fabricated one semantically -- that needs a human or LLM read, not a
regex -- so it reports raw counts of the pattern per page TYPE (to show
where it's template-expected vs unusual) and leaves the judgment call to
Ed rather than guessing.

Writes nothing except its own dated report under
C:\\Chloe\\brain\\dedup_reports\\. Does not mark, merge, edit, or delete
any wiki page -- confirmed fabrication (dead URL, self-referential link)
is reported for Ed's decision, never auto-fixed here.

Usage:
    python tools/fabrication_check.py
    python tools/fabrication_check.py --migration-date 2026-08-31
    python tools/fabrication_check.py --brain-root "C:\\Chloe\\brain"
    python tools/fabrication_check.py --workers 16   # concurrent URL checks
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from url_verify import verify_url  # noqa: E402
from tools.ungrounded_pages_report import scan, _DEFAULT_MIGRATION_DATE  # noqa: E402

_DEFAULT_BRAIN_ROOT = Path(r"C:\Chloe\brain")

# Deliberately allows '(' and ')' inside the match -- academic DOIs
# routinely embed a literal parenthesized year, e.g.
# "https://doi.org/10.1016/0010-0285(92)90002-J". An earlier version of
# this regex excluded ')' outright to avoid capturing a trailing paren
# that's part of surrounding prose ("(see https://x.com)"), which
# instead truncated every such DOI mid-URL and made it look 404 --
# confirmed live, 2026-09-03, before this was caught: ~98 pages were
# about to be reported as having dead citation URLs, many of which were
# this exact truncation artifact, not a real dead link. _clean_url below
# handles the trailing-paren-from-prose case properly instead.
_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def _clean_url(u: str) -> str:
    """Strip trailing punctuation that's almost certainly prose, not
    part of the URL -- but only strip a trailing ')' if it's UNBALANCED
    (more ')' than '(' in the match), so a legitimately paren-bearing
    URL like a DOI is left intact."""
    while u and u[-1] in ".,;:!?":
        u = u[:-1]
    if u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    return u
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")
_CITATION_HEADING_RE = re.compile(
    r"^#+\s*(?:sources|citations|references)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBERED_CITE_LINE_RE = re.compile(r"^\s*\[\d+\]\s*(.+)$", re.MULTILINE)
_CONFIDENCE_PCT_RE = re.compile(
    r"confidence[^.\n]{0,40}?(\d{1,3})\s*%", re.IGNORECASE)


def _page_slug(path: str) -> str:
    """Best-effort slug for self-reference checking: the filename stem,
    and the same stem with a trailing _v2/_v3/... version suffix
    stripped (a v2 page legitimately citing its own v1 slug, or vice
    versa, is still self-reference in spirit -- these are the same
    underlying topic across chloe_jobs.py's known _v{n} sprawl)."""
    stem = Path(path).stem
    base = re.sub(r"_v\d+$", "", stem)
    return stem, base


def _analyze_page(brain_root: Path, rel_path: str) -> dict:
    p = brain_root / rel_path
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"path": rel_path, "error": f"{type(e).__name__}: {e}"}

    stem, base = _page_slug(rel_path)
    urls = sorted({_clean_url(u) for u in _URL_RE.findall(text)})

    wikilinks = [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]
    self_refs = [w for w in wikilinks
                if Path(w).stem in (stem, base) or w in (stem, base)]

    has_citation_section = bool(_CITATION_HEADING_RE.search(text)) or \
        bool(re.search(r"^References:\s*$", text, re.MULTILINE))
    numbered_lines = _NUMBERED_CITE_LINE_RE.findall(text)
    name_only_cites = [ln.strip() for ln in numbered_lines
                       if not _URL_RE.search(ln)]

    confidence_hits = _CONFIDENCE_PCT_RE.findall(text)

    return {
        "path": rel_path,
        "urls": urls,
        "self_referential_wikilinks": self_refs,
        "has_citation_section": has_citation_section,
        "numbered_citation_count": len(numbered_lines),
        "name_only_citation_count": len(name_only_cites),
        "name_only_examples": name_only_cites[:3],
        "confidence_pct_hits": confidence_hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain-root", type=Path, default=_DEFAULT_BRAIN_ROOT)
    ap.add_argument("--migration-date", type=str, default=_DEFAULT_MIGRATION_DATE,
                    help="Same meaning as ungrounded_pages_report.py: pages "
                         "dated on/after this used the (separately, already "
                         "audited) Brave path; this script only scans pages "
                         "BEFORE it.")
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent URL HEAD-checks (url_verify.py's cache "
                         "is thread-safe).")
    args = ap.parse_args()

    if not args.brain_root.exists():
        print(f"brain root not found: {args.brain_root}", file=sys.stderr)
        return 1

    scanned = scan(args.brain_root)
    jobs = scanned["jobs"]

    pre_migration_paths = []
    for job_name, rows in jobs.items():
        for row in rows:
            if row["date"] < args.migration_date:
                pre_migration_paths.append((job_name, row["path"]))

    print(f"Scanning {len(pre_migration_paths)} pre-migration pages "
          f"(dated before {args.migration_date})...", flush=True)

    analyses = []
    for job_name, rel_path in pre_migration_paths:
        a = _analyze_page(args.brain_root, rel_path)
        a["job"] = job_name
        analyses.append(a)

    all_urls = sorted({u for a in analyses for u in a.get("urls", [])})
    print(f"Found {len(all_urls)} unique URLs across all pages; "
          f"checking liveness with {args.workers} workers "
          f"(cached results from a prior run are instant)...", flush=True)

    url_citable: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for url, citable in zip(all_urls, ex.map(verify_url, all_urls)):
            url_citable[url] = citable

    dead_url_pages = []
    self_ref_pages = []
    name_only_pages = []
    no_citation_at_all_pages = []
    confidence_by_job: dict[str, int] = {}

    for a in analyses:
        if a.get("error"):
            continue
        dead = [u for u in a["urls"] if not url_citable.get(u, True)]
        if dead:
            dead_url_pages.append((a["path"], dead))
        if a["self_referential_wikilinks"]:
            self_ref_pages.append((a["path"], a["self_referential_wikilinks"]))
        if a["name_only_citation_count"] > 0:
            name_only_pages.append(
                (a["path"], a["name_only_citation_count"], a["name_only_examples"]))
        if a["has_citation_section"] and not a["urls"] and a["numbered_citation_count"] == 0:
            no_citation_at_all_pages.append(a["path"])
        if a["confidence_pct_hits"]:
            confidence_by_job[a["job"]] = confidence_by_job.get(a["job"], 0) + 1

    today = date.today().isoformat()
    reports_dir = args.brain_root / "dedup_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"fabrication_check_pre_migration_{today}.md"

    lines = []
    lines.append(f"# Pre-migration fabrication check — {today}")
    lines.append("")
    lines.append("Read-only. Nothing was marked, merged, edited, or deleted. "
                 "Scanned every page dated before "
                 f"**{args.migration_date}** from the four chloe_jobs.py "
                 "daily jobs (see ungrounded_pages_report_2026-09-01.md for "
                 "the page inventory this reuses).")
    lines.append("")
    lines.append(f"- Pages scanned: **{len(analyses)}**")
    lines.append(f"- Unique URLs found: **{len(all_urls)}**")
    lines.append(f"- Pages with a confirmed-dead (404) citation URL: "
                 f"**{len(dead_url_pages)}**")
    lines.append(f"- Pages with a self-referential wikilink: "
                 f"**{len(self_ref_pages)}**")
    lines.append(f"- Pages with at least one name-only (no URL) numbered "
                 f"citation: **{len(name_only_pages)}**")
    lines.append(f"- Pages with a 'Sources/Citations/References' heading "
                 f"but literally zero citations under it: "
                 f"**{len(no_citation_at_all_pages)}**")
    lines.append("")
    lines.append(
        "**Not checked / not auto-flagged**: stated confidence percentages "
        "(e.g. 'Confidence the claim is true: 70%'). That's a REQUIRED "
        "template field for job_daily_critical_thinking_exercise pages "
        "specifically (confirmed by inspecting real pages) -- a number "
        "there is by design, not evidence of fabrication. Raw counts by "
        "job, for reference only:")
    for job, n in sorted(confidence_by_job.items()):
        lines.append(f"  - {job}: {n} page(s) contain a confidence percentage")
    lines.append("")

    lines.append("## Confirmed dead citation URLs (HTTP 404)")
    lines.append("")
    if dead_url_pages:
        for path, dead in dead_url_pages:
            lines.append(f"- `{path}`")
            for u in dead:
                lines.append(f"    - {u}")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Self-referential wikilinks (page cites itself)")
    lines.append("")
    if self_ref_pages:
        for path, refs in self_ref_pages:
            lines.append(f"- `{path}` -> {', '.join(f'[[{r}]]' for r in refs)}")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Name-only citations (numbered, no URL -- unverifiable by construction)")
    lines.append("")
    if name_only_pages:
        for path, n, examples in name_only_pages:
            ex = "; ".join(examples)
            lines.append(f"- `{path}` -- {n} name-only citation(s), e.g. {ex}")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Citation heading present, zero citations under it")
    lines.append("")
    if no_citation_at_all_pages:
        for path in no_citation_at_all_pages:
            lines.append(f"- `{path}`")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Decision needed")
    lines.append("")
    lines.append(
        "No page was edited or marked. For any page above with a confirmed "
        "dead URL or a self-referential wikilink, decide whether to add a "
        "no-retrieval-style notice, strip just that citation, or regenerate "
        "the page. Name-only citations and empty citation sections are a "
        "different, lower-severity gap (unverifiable, not necessarily "
        "false) -- decide whether that's worth fixing retroactively or just "
        "worth knowing about going forward.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}", flush=True)
    print(f"dead-URL pages={len(dead_url_pages)}  "
          f"self-ref pages={len(self_ref_pages)}  "
          f"name-only-citation pages={len(name_only_pages)}  "
          f"empty-citation-section pages={len(no_citation_at_all_pages)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
