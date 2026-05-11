"""Smoke test for Brain._validate_and_clean_page / _normalize_frontmatter.

Covers the two corruption patterns seen in the wild on 2026-05-11:
  - Double leading '---' (hybrid_llm_router.md)
  - Missing closing '---' (pyqt6.md)

Plus happy paths and a few rejection cases. Run from the live folder:
    python test_validate_page.py
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


# Brain.__init__ wants a real directory. Give it a temp one and a dummy llm.
def _make_brain():
    from brain import Brain
    tmp = tempfile.mkdtemp(prefix="brain_test_")
    return Brain(root=tmp, llm_call=lambda prompt, mode: "")


def _assert(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return False
    print(f"  ok:   {msg}")
    return True


def _normalize(brain, s):
    return brain._validate_and_clean_page(s)


def main():
    brain = _make_brain()
    ok = True

    # 1. Happy path: well-formed page passes through unchanged.
    good = (
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "tags: [a, b]\n"
        "---\n"
        "# Body\n"
        "\n"
        "Some content.\n"
    )
    out = _normalize(brain, good)
    ok &= _assert(out == good, "well-formed page unchanged")

    # 2. Double leading '---' (hybrid_llm_router shape).
    double_open = (
        "---\n"
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "---\n"
        "Body here.\n"
    )
    expected = (
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "---\n"
        "Body here.\n"
    )
    out = _normalize(brain, double_open)
    ok &= _assert(out == expected, "double '---' opener collapsed")

    # 3. Double leading '---' with blank line in between.
    double_open_blank = (
        "---\n"
        "\n"
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "---\n"
        "Body.\n"
    )
    expected_db = (
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "---\n"
        "Body.\n"
    )
    out = _normalize(brain, double_open_blank)
    ok &= _assert(out == expected_db, "double opener with blank between collapsed")

    # 4. Missing closing '---' (pyqt6 shape).
    no_close = (
        "---\n"
        "title: PyQt6\n"
        "type: entity\n"
        "created: 2026-05-11\n"
        "updated: 2026-05-11\n"
        "tags: [gui]\n"
        "# What the Source Says\n"
        "\n"
        "PyQt6 is a Python binding for Qt6.\n"
    )
    expected_close = (
        "---\n"
        "title: PyQt6\n"
        "type: entity\n"
        "created: 2026-05-11\n"
        "updated: 2026-05-11\n"
        "tags: [gui]\n"
        "---\n"
        "# What the Source Says\n"
        "\n"
        "PyQt6 is a Python binding for Qt6.\n"
    )
    out = _normalize(brain, no_close)
    ok &= _assert(out == expected_close, "missing closing '---' inserted before heading")

    # 5. Missing closing '---' with blank between YAML and body.
    no_close_blank = (
        "---\n"
        "title: Foo\n"
        "type: entity\n"
        "\n"
        "# Heading\n"
        "\n"
        "Body.\n"
    )
    expected_ncb = (
        "---\n"
        "title: Foo\n"
        "type: entity\n"
        "---\n"
        "# Heading\n"
        "\n"
        "Body.\n"
    )
    out = _normalize(brain, no_close_blank)
    ok &= _assert(out == expected_ncb, "missing close with blank-then-heading inserted correctly")

    # 6. Multi-line YAML with indented list (sources: - [[bar]]).
    with_list = (
        "---\n"
        "title: Foo\n"
        "sources:\n"
        "  - [[bar]]\n"
        "  - [[baz]]\n"
        "tags: [a]\n"
        "---\n"
        "Body.\n"
    )
    out = _normalize(brain, with_list)
    ok &= _assert(out == with_list, "YAML with indented list passes through unchanged")

    # 7. Preamble before frontmatter is stripped (existing behavior preserved).
    preamble = (
        "Here is the page:\n"
        "---\n"
        "title: Foo\n"
        "---\n"
        "Body.\n"
    )
    expected_pre = (
        "---\n"
        "title: Foo\n"
        "---\n"
        "Body.\n"
    )
    out = _normalize(brain, preamble)
    ok &= _assert(out == expected_pre, "preamble before '---' stripped")

    # 8. Idempotent: normalizing a clean page again gives the same result.
    out1 = _normalize(brain, good)
    out2 = _normalize(brain, out1)
    ok &= _assert(out1 == out2, "normalize is idempotent on well-formed input")

    # 9. Rejection: SKIP_PAGE marker returns None.
    skip = "SKIP_PAGE\n"
    ok &= _assert(_normalize(brain, skip) is None, "SKIP_PAGE marker returns None")

    # 10. Rejection: empty string returns None.
    ok &= _assert(_normalize(brain, "") is None, "empty string returns None")
    ok &= _assert(_normalize(brain, "   \n  \n") is None, "whitespace-only returns None")

    # 11. Rejection: no '---' at all returns None.
    no_fm = "title: Foo\nBody.\n"
    ok &= _assert(_normalize(brain, no_fm) is None, "no frontmatter returns None")

    # 12. Rejection: frontmatter without 'title:' field returns None.
    no_title = "---\ntype: concept\n---\nBody.\n"
    ok &= _assert(_normalize(brain, no_title) is None, "frontmatter without title returns None")

    # 13. Combined corruption: double opener AND missing close.
    combined = (
        "---\n"
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "# Heading\n"
        "Body.\n"
    )
    expected_combined = (
        "---\n"
        "title: Foo\n"
        "type: concept\n"
        "---\n"
        "# Heading\n"
        "Body.\n"
    )
    out = _normalize(brain, combined)
    ok &= _assert(out == expected_combined, "double opener + missing close fixed together")

    print()
    if ok:
        print("ALL PASS")
        return 0
    print("SOME FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
