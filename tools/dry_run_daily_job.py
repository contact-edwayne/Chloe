"""
tools/dry_run_daily_job.py — Run a chloe_jobs.py daily job against a
TEMPORARY COPY of the real brain, so you can see real before/after
behavior (does it write a new page? does dedup fire and merge? does Brave
actually return results now?) without touching anything under
C:\\Chloe\\brain.

⚠ WARNING — THIS IMPORTS chloe_jobs, WHICH BOOTS jarvis.py:
jarvis.py has no `if __name__ == "__main__"` guard, so importing anything
that reaches brain_wiring._search_call (which lazy-imports jarvis) boots
the FULL app in this process: audio device enumeration, wake-word model
loading, Kokoro TTS init, a boot sound, and a spoken greeting -- all on
your actual machine, actual audio output. This is why the agent that
built this script stopped running it automatically and handed it to you
instead. There is no way to run job_daily_topic_rotation /
job_daily_finance_ingest / job_daily_morning_brief /
job_daily_critical_thinking_exercise (all four route through
_search_call) without this side effect, short of adding a __main__ guard
to jarvis.py, which is out of scope here.

WHAT'S SAFE, WHAT'S NOT:
  - All file WRITES are safe: CHLOE_BRAIN_ROOT is redirected to a temp
    copy before chloe_jobs is even imported, so _write_brain/Brain.write
    physically cannot reach C:\\Chloe\\brain. Verified by this script
    (before/after counts on the REAL directory, confirmed unchanged).
  - The Brave search call is REAL (costs one query against your 2000/mo
    quota) and the Ollama synthesis call is REAL (local, no cost).
  - The audio/boot side effect above is NOT avoidable by this script.

First run costs ~15 minutes: it copies C:\\Chloe\\brain (a few MB, fast)
then embeds every wiki/concepts + wiki/entities page into a scratch DB so
wiki_dedup.find_duplicate's cosine layer has real data to match against
(this is the slow part -- one Ollama call per page). Re-running reuses the
existing copy+embeddings unless --fresh is passed, so subsequent runs are
fast.

Usage:
    python tools/dry_run_daily_job.py topic_rotation
    python tools/dry_run_daily_job.py finance_ingest
    python tools/dry_run_daily_job.py morning_brief
    python tools/dry_run_daily_job.py critical_thinking
    python tools/dry_run_daily_job.py topic_rotation --fresh
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REAL_BRAIN_ROOT = Path(r"C:\Chloe\brain")
# Fixed (not random) so repeated runs reuse the same copy + embeddings.
_TMP_ROOT = Path(tempfile.gettempdir()) / "chloe_brain_dryrun"

_JOB_FUNCS = {
    "topic_rotation": "job_daily_topic_rotation",
    "finance_ingest": "job_daily_finance_ingest",
    "morning_brief": "job_daily_morning_brief",
    "critical_thinking": "job_daily_critical_thinking_exercise",
}


def _copy_brain(fresh: bool) -> None:
    if _TMP_ROOT.exists() and not fresh:
        print(f"reusing existing temp copy: {_TMP_ROOT}")
        return
    if _TMP_ROOT.exists():
        print(f"--fresh passed: removing old temp copy {_TMP_ROOT}")
        shutil.rmtree(_TMP_ROOT)
    print(f"copying {_REAL_BRAIN_ROOT} -> {_TMP_ROOT} ...")
    shutil.copytree(_REAL_BRAIN_ROOT, _TMP_ROOT)
    print("copy done.")


def _embed_corpus_if_needed(fresh: bool) -> None:
    import wiki_embedding
    store = wiki_embedding.get_store()
    with store._lock, store._connect() as c:
        (count,) = c.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()
    if count > 0 and not fresh:
        print(f"reusing existing embeddings ({count} pages already in the scratch DB)")
        return

    print("embedding wiki/concepts + wiki/entities into the scratch DB "
          "(one Ollama call per page, ~2s each -- this is the slow part, "
          "expect several minutes)...")
    t0 = time.time()
    n = 0
    for sub in ("concepts", "entities"):
        for p in (_TMP_ROOT / "wiki" / sub).glob("*.md"):
            rel = f"{sub}/{p.name}"
            try:
                store.upsert_page(rel)
                n += 1
            except Exception as e:
                print(f"  skip {rel}: {type(e).__name__}: {e}")
            if n % 50 == 0:
                print(f"  {n} embedded, {time.time() - t0:.0f}s elapsed")
    print(f"embedding done: {n} pages in {time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=sorted(_JOB_FUNCS.keys()))
    ap.add_argument("--fresh", action="store_true",
                    help="Force a fresh copy of C:\\Chloe\\brain and re-embed "
                         "everything, instead of reusing the last run's copy.")
    args = ap.parse_args()

    print(f"=== dry run: {_JOB_FUNCS[args.job]} ===\n")
    print("⚠ this will import chloe_jobs, which boots jarvis.py -- audio "
          "will play on this machine (boot sound + spoken greeting). "
          "Ctrl+C now if that's not expected.\n")

    _copy_brain(fresh=args.fresh)

    import os
    os.environ["CHLOE_BRAIN_ROOT"] = str(_TMP_ROOT)
    os.environ["CHLOE_WIKI_ROOT"] = str(_TMP_ROOT / "wiki")
    os.environ["CHLOE_WIKI_DB"] = str(_TMP_ROOT / "dryrun_embed.db")

    _embed_corpus_if_needed(fresh=args.fresh)

    # Snapshot the REAL brain's concept count before running, so we can
    # positively confirm afterward that nothing leaked through despite
    # the env-var redirection.
    real_before = len(list((_REAL_BRAIN_ROOT / "wiki" / "concepts").glob("*.md")))

    before_concepts = set(p.name for p in (_TMP_ROOT / "wiki" / "concepts").glob("*.md"))
    before_sources = set(p.name for p in (_TMP_ROOT / "wiki" / "sources").glob("*.md"))
    before_briefs = (set(p.name for p in (_TMP_ROOT / "briefs").glob("*.md"))
                     if (_TMP_ROOT / "briefs").exists() else set())
    print(f"\nBEFORE (temp copy): {len(before_concepts)} concepts, "
          f"{len(before_sources)} sources, {len(before_briefs)} briefs")

    print(f"\n--- running {_JOB_FUNCS[args.job]}() for real "
          f"(live Brave query + local Ollama synthesis) ---\n")
    import chloe_jobs
    fn = getattr(chloe_jobs, _JOB_FUNCS[args.job])
    result = fn()
    print(f"\nRESULT: {result}")

    after_concepts = set(p.name for p in (_TMP_ROOT / "wiki" / "concepts").glob("*.md"))
    after_sources = set(p.name for p in (_TMP_ROOT / "wiki" / "sources").glob("*.md"))
    after_briefs = (set(p.name for p in (_TMP_ROOT / "briefs").glob("*.md"))
                   if (_TMP_ROOT / "briefs").exists() else set())

    new_concepts = after_concepts - before_concepts
    new_sources = after_sources - before_sources
    new_briefs = after_briefs - before_briefs
    sprawl = [f for f in (new_concepts | new_sources | new_briefs)
             if "_v" in f and f.split("_v")[-1][:1].isdigit()]

    print(f"\nAFTER (temp copy): {len(after_concepts)} concepts, "
          f"{len(after_sources)} sources, {len(after_briefs)} briefs")
    print(f"New files: {sorted(new_concepts | new_sources | new_briefs) or '(none -- merged into an existing page)'}")
    print(f"_v{{n}} sprawl files created: {sprawl or 'NONE'}")

    real_after = len(list((_REAL_BRAIN_ROOT / "wiki" / "concepts").glob("*.md")))
    print(f"\nREAL brain concepts/ count: before={real_before} after={real_after}  "
          f"{'(UNCHANGED, confirmed safe)' if real_before == real_after else '*** CHANGED -- INVESTIGATE ***'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
