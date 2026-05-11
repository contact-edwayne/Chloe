"""One-shot cleanup: re-run _normalize_frontmatter over every wiki .md file.

Fixes the two corruption patterns observed 2026-05-11:
  - Double leading '---' (hybrid_llm_router.md)
  - Missing closing '---' (pyqt6.md)

Defaults to dry-run; pass --apply to actually write.

    python fix_wiki_frontmatter.py            # dry-run, show what would change
    python fix_wiki_frontmatter.py --apply    # actually write
"""

import sys
import tempfile
from pathlib import Path

BRAIN_ROOT = Path(r"C:\Chloe\brain")
WIKI = BRAIN_ROOT / "wiki"


def main():
    apply = "--apply" in sys.argv
    if not WIKI.exists():
        print(f"Wiki not found at {WIKI}")
        return 1

    # Brain.__init__ wants a real root + an llm_call. Both are unused by
    # the normalize helper — give it dummies.
    from brain import Brain
    brain = Brain(root=str(BRAIN_ROOT), llm_call=lambda p, m: "")

    pages = sorted(WIKI.rglob("*.md"))
    print(f"scanning {len(pages)} pages under {WIKI}")
    print(f"mode: {'APPLY (will rewrite files)' if apply else 'DRY-RUN (no writes)'}")
    print()

    touched = 0
    skipped = 0
    for p in pages:
        try:
            original = p.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  skip {p.relative_to(BRAIN_ROOT)}: read error {e}")
            skipped += 1
            continue

        # Run the same logic the ingest pipeline runs on fresh LLM output.
        # Note: _validate_and_clean_page returns None if SKIP_PAGE-marked,
        # empty, or missing 'title:'. None of those should match a real
        # on-disk wiki page; if any do, leave them alone (skipped).
        cleaned = brain._validate_and_clean_page(original)
        if cleaned is None:
            print(f"  skip {p.relative_to(BRAIN_ROOT)}: validator returned None")
            skipped += 1
            continue

        if cleaned == original:
            continue  # already clean, no diff

        touched += 1
        rel = p.relative_to(BRAIN_ROOT)
        # Brief diff summary — first few lines of each side.
        orig_head = original.splitlines()[:6]
        new_head = cleaned.splitlines()[:6]
        print(f"  CHANGED {rel}")
        print(f"    before head: {orig_head}")
        print(f"    after  head: {new_head}")
        size_delta = len(cleaned) - len(original)
        print(f"    size delta: {size_delta:+d} bytes")

        if apply:
            p.write_text(cleaned, encoding="utf-8")

    print()
    print(f"summary: {touched} would change, {skipped} skipped, "
          f"{len(pages) - touched - skipped} already clean")
    if not apply and touched > 0:
        print("re-run with --apply to write the fixes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
