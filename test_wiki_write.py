"""test_wiki_write.py - Sanity tests for /wiki_write.

Tests handle_wiki_write and _slugify_topic without hitting Groq or
running the real ingest pipeline. Monkey-patches brain_wiring._search_call
and brain_wiring.BRAIN.ingest to controllable fakes.

Run from the jarvis dir:
    python test_wiki_write.py
Exit code 0 on success, non-zero on any failure.
"""

import os
import sys
import tempfile
from pathlib import Path

# Brain.__init__ requires the brain root to exist. Set up a temp one BEFORE
# importing brain_wiring so the module-load BRAIN instantiation succeeds.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="wiki_write_test_"))
for sub in ("raw", "wiki/entities", "wiki/concepts", "wiki/sources",
            "episodic", "facts"):
    (_TEST_ROOT / sub).mkdir(parents=True, exist_ok=True)
os.environ["CHLOE_BRAIN_ROOT"] = str(_TEST_ROOT)

import brain_wiring as bw


# --- Test harness ---------------------------------------------------------

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  ({detail})")


# Replace _search_call with a controllable fake. Lists are used as one-
# element queues so tests can mutate the return value without rebinding.
# _search_call returns {"text", "results"} (2026-08-31, was a bare string --
# handle_wiki_write needs the structured `results` to build real citations
# instead of trusting the model's own fabricated sources block).
_search_return = ["DEFAULT FAKE RESEARCH BODY"]
_search_results_return = [[]]  # list of {title, url, domain} dicts, per call


def _fake_search_call(prompt, *, topic=""):
    return {
        "text": _search_return[0] if _search_return else "",
        "results": _search_results_return[0] if _search_results_return else [],
    }


bw._search_call = _fake_search_call


# Replace BRAIN.ingest with a fake that records calls and returns a stub.
# Mirrors the real BRAIN.ingest return shape.
_ingest_calls = []
_ingest_raise_holder = [None]  # use list so nested funcs can mutate


def _fake_ingest(filename, dry_run=False):
    _ingest_calls.append({"filename": filename, "dry_run": dry_run})
    if _ingest_raise_holder[0]:
        raise _ingest_raise_holder[0]
    return {
        "slug": filename[:-3] if filename.endswith(".md") else filename,
        "tldr": "Fake TLDR from test harness.",
        "entities_touched": ["thing_a", "thing_b"],
        "concepts_touched": ["concept_x"],
    }


bw.BRAIN.ingest = _fake_ingest


def _reset_state():
    """Clear fake state between tests."""
    _search_return.clear()
    _search_return.append("DEFAULT FAKE RESEARCH BODY")
    _search_results_return.clear()
    _search_results_return.append([])
    _ingest_calls.clear()
    _ingest_raise_holder[0] = None
    for p in (_TEST_ROOT / "raw").glob("*.md"):
        p.unlink()


# --- Tests ----------------------------------------------------------------

def test_slugify_topic():
    print("\n_slugify_topic:")
    cases = [
        ("Kelly criterion", "kelly_criterion"),
        ("Lebron James (NBA)", "lebron_james_nba"),
        ("  Polyphasic Sleep!! ", "polyphasic_sleep"),
        ("x86-64 assembly", "x86-64_assembly"),
        ('"Kelly criterion"', "kelly_criterion"),
        ("!!!!!", ""),
        ("A" * 100, "a" * 80),
    ]
    for inp, expected in cases:
        got = bw._slugify_topic(inp)
        check(f"slugify {inp!r} -> {expected!r}", got == expected,
              f"got {got!r}")


def test_empty_topic_returns_usage():
    _reset_state()
    print("\nempty topic -> usage message:")
    r = bw.handle_wiki_write("")
    check("usage hint present", "Usage:" in r and "/wiki_write" in r, r[:200])
    check("no ingest called", _ingest_calls == [])
    check("no raw file written", not any((_TEST_ROOT / "raw").iterdir()))


def test_too_long_topic_returns_error():
    _reset_state()
    print("\ntoo-long topic (>200 chars) -> error:")
    r = bw.handle_wiki_write("x" * 201)
    check("max-length warning", "too long" in r.lower() or "200" in r, r[:200])
    check("no ingest called", _ingest_calls == [])


def test_garbage_topic_returns_error():
    _reset_state()
    print("\ntopic that produces empty slug -> error:")
    r = bw.handle_wiki_write("!!!!!")
    check("slug error reported",
          "slug" in r.lower() or "valid" in r.lower(), r[:200])
    check("no ingest called", _ingest_calls == [])


def test_normal_flow():
    _reset_state()
    _search_return[0] = ("This is fake research content about the Kelly "
                         "criterion. It's a betting strategy used in finance.")
    print("\nnormal flow (topic -> research -> file -> ingest):")
    r = bw.handle_wiki_write("Kelly criterion")
    check("returns success header", "Wrote + ingested" in r, r[:200])
    check("contains TLDR field", "TLDR" in r)
    check("raw file exists",
          (_TEST_ROOT / "raw" / "kelly_criterion.md").exists(),
          str(list((_TEST_ROOT / "raw").iterdir())))
    check("BRAIN.ingest called exactly once", len(_ingest_calls) == 1,
          str(_ingest_calls))
    check("BRAIN.ingest got correct filename",
          _ingest_calls and _ingest_calls[0]["filename"] == "kelly_criterion.md",
          str(_ingest_calls))

    body = (_TEST_ROOT / "raw" / "kelly_criterion.md").read_text(encoding="utf-8")
    check("provenance frontmatter present",
          "generated_via: /wiki_write" in body)
    check("requested_topic in frontmatter",
          "requested_topic: Kelly criterion" in body)
    check("heading derived from slug",
          "# Kelly Criterion" in body)
    check("fake research body included",
          "betting strategy" in body)


def test_dry_run_no_writes():
    _reset_state()
    print("\n--dry-run skips writes and ingest:")
    r = bw.handle_wiki_write("Kelly criterion", dry_run=True)
    check("DRY RUN header in result", "DRY RUN" in r, r[:200])
    check("no raw file written",
          not (_TEST_ROOT / "raw" / "kelly_criterion.md").exists())
    check("BRAIN.ingest not called", _ingest_calls == [])


def test_search_empty_returns_error_no_writes():
    _reset_state()
    _search_return[0] = ""
    print("\nempty search result -> error, no writes, no ingest:")
    r = bw.handle_wiki_write("Some obscure topic")
    check("error message returned",
          "empty" in r.lower() or "nothing" in r.lower(), r[:200])
    check("no raw file written", not any((_TEST_ROOT / "raw").iterdir()))
    check("BRAIN.ingest not called", _ingest_calls == [])


def test_ingest_error_surfaced():
    _reset_state()
    _search_return[0] = "fake body content long enough to ingest"
    _ingest_raise_holder[0] = RuntimeError("simulated ingest crash")
    print("\nBRAIN.ingest raising -> error in return, raw file persisted:")
    r = bw.handle_wiki_write("Test topic")
    check("error reported", "ingest failed" in r.lower(), r[:200])
    check("exception type in error", "RuntimeError" in r)
    # The raw file should still exist so the user can retry /ingest manually
    # without burning another Groq research call.
    check("raw file persisted for manual retry",
          (_TEST_ROOT / "raw" / "test_topic.md").exists(),
          str(list((_TEST_ROOT / "raw").iterdir())))


def test_quoted_topics_get_cleaned():
    _reset_state()
    _search_return[0] = "fake body"
    print("\nquoted topics get stripped before slugify:")
    r = bw.handle_wiki_write('"Kelly criterion"')
    check("raw file at clean slug (no leading underscore)",
          (_TEST_ROOT / "raw" / "kelly_criterion.md").exists(),
          str(list((_TEST_ROOT / "raw").iterdir())))


def test_real_citations_from_search_results():
    # 2026-08-31 regression test: handle_wiki_write must build its
    # Citations section from _search_call's structured `results`, never
    # from the model's own output. Before the fix, _search_call returned
    # only text, so the model free-generated its own sources block --
    # confirmed live on topic "silver" to fabricate a self-referential
    # [[wikilink]] and an invented /wiki_write/silver URL. Verify real
    # URLs land in the page and no fabricated pattern does.
    _reset_state()
    _search_return[0] = "Silver is a precious metal with many industrial uses."
    _search_results_return[0] = [
        {"title": "Silver Price Today", "url": "https://kitco.com/silver",
         "domain": "kitco.com"},
        {"title": "Silver - Wikipedia", "url": "https://en.wikipedia.org/wiki/Silver",
         "domain": "en.wikipedia.org"},
    ]
    print("\nreal citations built from structured search results:")
    r = bw.handle_wiki_write("silver test topic")
    body = (_TEST_ROOT / "raw" / "silver_test_topic.md").read_text(encoding="utf-8")
    check("Citations section present", "## Citations" in body, body[:400])
    check("real URL #1 present", "https://kitco.com/silver" in body)
    check("real URL #2 present", "https://en.wikipedia.org/wiki/Silver" in body)
    check("source_urls frontmatter present", "source_urls:" in body)
    check("frontmatter has real URL",
          "  - https://kitco.com/silver" in body)
    check("no fabricated self-referential wikilink pattern",
          "[[silver_test_topic]]" not in body and "[[silver test topic]]" not in body,
          body[:400])
    check("no fabricated /wiki_write/ path pattern",
          "/wiki_write/" not in body.replace("`/wiki_write ", ""), body[:400])


def test_citations_section_present_even_with_no_results():
    # No Brave results (e.g. heavy-fallback path with no web search) should
    # still produce a well-formed Citations section, just an honest empty
    # one -- never silently omitted, never backfilled by the model.
    _reset_state()
    _search_return[0] = "Fake body with no web results behind it."
    _search_results_return[0] = []
    print("\nempty results -> honest empty citations, not omitted:")
    r = bw.handle_wiki_write("obscure test topic")
    body = (_TEST_ROOT / "raw" / "obscure_test_topic.md").read_text(encoding="utf-8")
    check("Citations section present", "## Citations" in body)
    check("honest 'no citations' placeholder",
          "(no citations returned)" in body, body[:400])


# --- Run ------------------------------------------------------------------

if __name__ == "__main__":
    test_slugify_topic()
    test_empty_topic_returns_usage()
    test_too_long_topic_returns_error()
    test_garbage_topic_returns_error()
    test_normal_flow()
    test_dry_run_no_writes()
    test_search_empty_returns_error_no_writes()
    test_ingest_error_surfaced()
    test_quoted_topics_get_cleaned()
    test_real_citations_from_search_results()
    test_citations_section_present_even_with_no_results()

    print(f"\n{'=' * 50}")
    print(f"PASSED: {PASSED}")
    print(f"FAILED: {FAILED}")
    if FAILED:
        sys.exit(1)
