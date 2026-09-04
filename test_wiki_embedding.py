"""
test_wiki_embedding.py — Sanity tests for WikiEmbeddingStore.

Uses a deterministic toy embedding (monkey-patched onto the class) so the
test runs without Ollama. Verifies:
  1. backfill_all inserts every .md file
  2. upsert is idempotent — same content = 'unchanged' on re-run
  3. content change => 'updated' + new embedding
  4. delete_page actually removes the row
  5. purge_missing GCs orphan rows
  6. search returns rows by cosine, honors the threshold, caps at limit
  7. frontmatter parsing extracts title/type when present
  8. path-stem boost tips bullseye-shorter cases the right way
  9. apply_staleness_gate collapses near-duplicate point-in-time hits to
     the single best match, without merging different subjects, non-
     point-in-time pages, or unparseable-age pages into the collapse

Run from the jarvis dir:
    python test_wiki_embedding.py
Exit code 0 on success, non-zero on any failure.
"""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import wiki_embedding as we


# Toy embedding: deterministic, content-derived, L2-normalized.
def _toy_embed(self, text: str) -> bytes | None:
    if not text or not text.strip():
        return None
    h = hashlib.sha256(text.strip().encode('utf-8')).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
    topic = (text.strip().lower().split() + ['', '', '', ''])[:4]
    for i, word in enumerate(topic):
        arr[i] += sum(ord(c) for c in word) / 1000.0
    n = float(np.linalg.norm(arr))
    if not n:
        return None
    return (arr / n).tobytes()


PASSED = 0
FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  ({detail})")


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding='utf-8')


def test_frontmatter_parsing():
    print("\n[frontmatter parsing]")
    title, typ, body = we._parse_frontmatter(
        "---\ntitle: Edward Wayne\ntype: entity\ntags: [a]\n---\nHe is the user.\n"
    )
    check("title extracted", title == "Edward Wayne", f"got {title!r}")
    check("type extracted", typ == "entity", f"got {typ!r}")
    check("body extracted", body == "He is the user.", f"got {body!r}")
    title, typ, body = we._parse_frontmatter("no frontmatter here\nbody text")
    check("no-frontmatter title empty", title == "")
    check("no-frontmatter body retains full text", "body text" in body)


def test_store_lifecycle(tmp: Path):
    print("\n[store lifecycle]")
    wiki = tmp / "wiki"
    wiki.mkdir()
    _write(wiki, "entities/alpha.md",
           "---\ntitle: Alpha\ntype: entity\n---\nAlpha is the first letter.\n")
    _write(wiki, "concepts/beta.md",
           "---\ntitle: Beta\ntype: concept\n---\nBeta is the second letter.\n")
    _write(wiki, "index.md", "# Wiki Index\n\n- [[alpha]]\n- [[beta]]\n")

    db = tmp / "test.db"
    we.WikiEmbeddingStore._embed = _toy_embed
    store = we.WikiEmbeddingStore(wiki_root=wiki, db_path=db)

    counts = store.backfill_all()
    check("backfill inserts 3 pages", counts['inserted'] == 3, f"counts={counts}")
    check("count_pages = 3", store.count_pages() == 3, f"got {store.count_pages()}")
    check("count_embedded = 3", store.count_embedded() == 3, f"got {store.count_embedded()}")

    again = store.upsert_page("entities/alpha.md")
    check("re-upsert unchanged file => 'unchanged'", again == 'unchanged', f"got {again!r}")

    _write(wiki, "entities/alpha.md",
           "---\ntitle: Alpha\ntype: entity\n---\n"
           "Alpha is the first Greek letter, now updated.\n")
    upd = store.upsert_page("entities/alpha.md")
    check("modified file => 'updated'", upd == 'updated', f"got {upd!r}")

    (wiki / "concepts" / "beta.md").unlink()
    purged = store.purge_missing()
    check("purge_missing returns 1", purged == 1, f"got {purged}")
    check("count_pages now 2 after purge", store.count_pages() == 2, f"got {store.count_pages()}")

    ok = store.delete_page("index.md")
    check("delete_page returns True for present row", ok is True)
    check("delete_page of missing row returns False",
          store.delete_page("nonexistent.md") is False)

    outside = tmp / "outside.md"
    outside.write_text("hi")
    res = store.upsert_page("../outside.md")
    check("path-traversal refused", res == 'missing', f"got {res!r}")


def test_search(tmp: Path):
    print("\n[search]")
    wiki = tmp / "wiki2"
    wiki.mkdir()
    _write(wiki, "entities/wallet.md",
           "---\ntitle: Wallet\ntype: entity\n---\n"
           "Chloe wallet bitcoin lightning breez sdk liquid keys secrets.\n")
    _write(wiki, "entities/hud.md",
           "---\ntitle: HUD\ntype: entity\n---\n"
           "The HUD is the desktop overlay window with the orb and waveform.\n")
    _write(wiki, "concepts/cooking.md",
           "---\ntitle: Cooking\ntype: concept\n---\n"
           "Recipes pasta tomatoes garlic; nothing about Chloe.\n")
    db = tmp / "search.db"
    we.WikiEmbeddingStore._embed = _toy_embed
    store = we.WikiEmbeddingStore(wiki_root=wiki, db_path=db)
    store.backfill_all()

    hits = store.search("wallet bitcoin lightning", limit=5, threshold=0.0)
    check("search returns results", len(hits) >= 1)
    check("top hit is wallet page",
          hits[0]['path'] == 'entities/wallet.md',
          f"top={hits[0]['path']!r}, all={[h['path'] for h in hits]}")
    check("top hit has title", hits[0]['title'] == 'Wallet')
    check("top hit has snippet", 'wallet' in hits[0]['snippet'].lower(),
          f"snippet={hits[0]['snippet']!r}")
    check("scores monotone non-increasing",
          all(hits[i]['score'] >= hits[i+1]['score']
              for i in range(len(hits) - 1)))

    check("empty query returns []", store.search("") == [])

    hits2 = store.search("wallet hud cooking", limit=2, threshold=0.0)
    check("limit honored", len(hits2) <= 2, f"got {len(hits2)}")


def test_path_boost(tmp: Path):
    """The path-stem boost should tip a bullseye-but-shorter page above
    a denser sibling that shares the same topic."""
    print("\n[path-stem boost]")
    wiki = tmp / "wiki_boost"
    wiki.mkdir()
    _write(wiki, "entities/ollama.md",
           "---\ntitle: Ollama\ntype: entity\n---\n"
           "Ollama runs local models.\n")
    repeated = " ".join(["Ollama"] * 18)
    _write(wiki, "concepts/router.md",
           "---\ntitle: Router\ntype: concept\n---\n"
           f"{repeated} - a much denser page about ollama.\n")
    db = tmp / "boost.db"
    we.WikiEmbeddingStore._embed = _toy_embed
    store = we.WikiEmbeddingStore(wiki_root=wiki, db_path=db)
    store.backfill_all()

    hits = store.search("ollama", limit=5, threshold=0.0)
    check("ollama query returns hits", len(hits) >= 2)
    check("path-stem-matching page is first",
          hits[0]['path'] == 'entities/ollama.md',
          f"ranking: {[(h['path'], round(h['score'], 3)) for h in hits]}")
    check("top hit has positive path_boost",
          hits[0].get('path_boost', 0) > 0,
          f"path_boost={hits[0].get('path_boost')}")
    check("non-matching page has zero path_boost",
          hits[1].get('path_boost', 0) == 0,
          f"path_boost={hits[1].get('path_boost')}")

    tokens = we.WikiEmbeddingStore._query_tokens("an ai is on the page")
    check("short tokens filtered (>=3 chars)",
          all(len(t) >= 3 for t in tokens), f"tokens={tokens}")
    check("alnum-only tokens", all(t.isalnum() for t in tokens), f"tokens={tokens}")


def test_collapse_near_duplicate_point_in_time(tmp: Path):
    """Regression test for the 2026-09-03 "pile of near-duplicate SLV
    answers" fix: WikiEmbeddingStore.search()'s apply_staleness_gate=True
    path should collapse same-subject point-in-time hits down to the
    single best match, while leaving a different subject (gold), a
    non-point-in-time near-duplicate, and an unparseable-age point-in-
    time page all untouched."""
    print("\n[collapse near-duplicate point-in-time]")
    wiki = tmp / "wiki_collapse"
    wiki.mkdir()
    slv_body = "SLV silver price current quote today market figure.\n"
    _write(wiki, "sources/web_slv_a.md",
           "---\ntitle: Web What Is The Current Price Of Slv A\ntype: source\n"
           "point_in_time_kind: quote\ngenerated_at: 2026-09-01T09:00:00\n"
           f"---\n{slv_body}")
    _write(wiki, "sources/web_slv_b.md",
           "---\ntitle: Web What Is The Current Price Of Slv B\ntype: source\n"
           "point_in_time_kind: quote\ngenerated_at: 2026-09-01T15:00:00\n"
           f"---\n{slv_body}")
    # Same pit_kind, DIFFERENT subject -- must never collapse with the SLV
    # pair just because both are 'quote' kind (the exact gold-vs-SLV
    # failure mode this whole mechanism exists to avoid).
    _write(wiki, "sources/web_gold_a.md",
           "---\ntitle: Web What Is The Current Price Of Gold\ntype: source\n"
           "point_in_time_kind: quote\ngenerated_at: 2026-09-01T15:00:00\n---\n"
           "Gold price current quote today market figure.\n")
    # Near-duplicate SLV content but NO point_in_time_kind -- collapse only
    # ever applies to point-in-time pages, this must survive untouched.
    _write(wiki, "sources/web_slv_c_no_pit.md",
           f"---\ntitle: Web What Is The Current Price Of Slv C\ntype: source\n---\n{slv_body}")
    # point_in_time_kind set but no generated_at/date -> asof_epoch=0.0 ->
    # age_days=None -- must never be collapsed (can't judge freshness).
    _write(wiki, "sources/web_slv_d_no_date.md",
           "---\ntitle: Web What Is The Current Price Of Slv D\ntype: source\n"
           f"point_in_time_kind: quote\n---\n{slv_body}")

    db = tmp / "collapse.db"
    we.WikiEmbeddingStore._embed = _toy_embed
    store = we.WikiEmbeddingStore(wiki_root=wiki, db_path=db)
    store.backfill_all()

    hits = store.search("what's SLV's price", limit=10, threshold=0.0)
    paths = [h["path"] for h in hits]

    slv_ab_survivors = [p for p in paths
                        if p in ("sources/web_slv_a.md", "sources/web_slv_b.md")]
    check("only ONE of the two same-subject/same-age-known SLV quote "
         "pages survives the collapse",
          len(slv_ab_survivors) == 1, f"paths={paths}")

    check("gold quote page NOT collapsed away by the SLV cluster "
         "(different subject, same pit_kind)",
          "sources/web_gold_a.md" in paths, f"paths={paths}")

    check("non-point-in-time near-duplicate untouched by collapse",
          "sources/web_slv_c_no_pit.md" in paths, f"paths={paths}")

    check("point-in-time page with no parseable date (age_days=None) "
         "untouched by collapse",
          "sources/web_slv_d_no_date.md" in paths, f"paths={paths}")


def test_idempotent_init(tmp: Path):
    print("\n[idempotent init]")
    wiki = tmp / "wiki3"
    wiki.mkdir()
    db = tmp / "idem.db"
    we.WikiEmbeddingStore._embed = _toy_embed
    we.WikiEmbeddingStore(wiki_root=wiki, db_path=db)
    we.WikiEmbeddingStore(wiki_root=wiki, db_path=db)
    check("two stores share schema without conflict", True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="wiki_emb_test_"))
    try:
        test_frontmatter_parsing()
        test_store_lifecycle(tmp)
        test_search(tmp)
        test_path_boost(tmp)
        test_collapse_near_duplicate_point_in_time(tmp)
        test_idempotent_init(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    print(f"summary: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
