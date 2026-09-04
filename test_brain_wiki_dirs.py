"""test_brain_wiki_dirs.py - Regression test for the entitys/ misspelled-
directory bug (2026-09-01/09-02): Brain._ingest_typed_page used to derive
its wiki/ subdirectory via naive pluralization (f'{page_type}s'), which
is correct for 'concept' -> 'concepts' but wrong for the irregular plural
'entity' -> 'entitys' (should be 'entities'). Every new/updated entity
page silently landed in the wrong directory for two days before an audit
caught it (brain/log.md 2026-09-03 16:30).

Fixed by centralizing the page_type -> directory mapping into a single
Brain._wiki_dir_for_page_type() staticmethod (see brain.py), used by
every write path in the class instead of each site re-deriving it. This
test guards that mapping directly -- if anyone reintroduces a second,
independently-pluralizing copy of this logic anywhere in brain.py, this
test does NOT catch that (it only tests the one canonical function), but
it does catch the mapping itself silently changing or a new page_type
being added without an explicit, correct entry.

Run from the jarvis dir:
    python test_brain_wiki_dirs.py
Exit code 0 on success, non-zero on any failure.
"""

from pathlib import Path

from brain import Brain

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def test_entity_maps_to_entities():
    check("page_type 'entity' -> 'entities' (irregular plural, NOT 'entitys')",
          Brain._wiki_dir_for_page_type("entity") == "entities",
          Brain._wiki_dir_for_page_type("entity"))


def test_concept_maps_to_concepts():
    check("page_type 'concept' -> 'concepts' (regular plural)",
          Brain._wiki_dir_for_page_type("concept") == "concepts",
          Brain._wiki_dir_for_page_type("concept"))


def test_unknown_page_type_fails_honestly():
    try:
        Brain._wiki_dir_for_page_type("bogus")
        check("unknown page_type raises ValueError instead of guessing", False)
    except ValueError as e:
        check("unknown page_type raises ValueError instead of guessing",
              "bogus" in str(e))
    except Exception as e:
        check("unknown page_type raises ValueError instead of guessing", False,
              f"raised {type(e).__name__} instead of ValueError: {e}")


def test_no_naive_pluralization_pattern_in_source():
    """Belt-and-suspenders static check: the exact buggy shape
    (f'{page_type}s' or f"{page_type}s") must not appear as CODE anywhere
    in brain.py (comment lines mentioning it in prose, like this test's
    own docstring references, are excluded). Catches a regression even
    if someone adds a THIRD call site that bypasses
    _wiki_dir_for_page_type entirely, which the mapping-level tests above
    can't see."""
    src = (Path(__file__).parent / "brain.py").read_text(encoding="utf-8")
    code_hits = []
    for lineno, line in enumerate(src.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        if "{page_type}s" in line and ("f'" in line or 'f"' in line):
            code_hits.append((lineno, line.strip()))
    check("no naive f'{page_type}s' pluralization anywhere in brain.py",
          len(code_hits) == 0, code_hits)


if __name__ == "__main__":
    test_entity_maps_to_entities()
    test_concept_maps_to_concepts()
    test_unknown_page_type_fails_honestly()
    test_no_naive_pluralization_pattern_in_source()

    print(f"\n{'=' * 50}")
    print(f"PASSED: {PASSED}")
    print(f"FAILED: {FAILED}")
    if FAILED:
        import sys
        sys.exit(1)
