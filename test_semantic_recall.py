"""Smoke test for ChloeMemory's vector-recall path.

Mocks ChloeMemory._embed with a deterministic 3-dim toy embedder so we
don't need a live Ollama for the test.

Run from the live folder:
    python test_semantic_recall.py
"""

import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


_TOY_EMBED_MAP = {
    "I have a golden retriever named Bella": np.array([1.0, 0.0, 0.0]),
    "Bella is my favorite dog":               np.array([0.95, 0.05, 0.0]),
    "My puppy loves to play in the yard":     np.array([0.9, 0.1, 0.0]),
    "I drive a Honda Civic to work":          np.array([0.0, 1.0, 0.0]),
    "Traffic was terrible on the freeway":    np.array([0.0, 0.9, 0.1]),
    "My favorite breakfast is pancakes":      np.array([0.0, 0.0, 1.0]),
    "tell me about dogs":                     np.array([0.97, 0.0, 0.0]),
    "what do I drive":                        np.array([0.0, 0.95, 0.05]),
    "food I like":                            np.array([0.0, 0.0, 0.97]),
}


def _toy_embed(self, text):
    if not text or not text.strip():
        return None
    vec = _TOY_EMBED_MAP.get(text.strip())
    if vec is None:
        # Unknown text → negative-octant direction. After L2-normalize this
        # gives ~-0.577 cosine with each positive-axis cluster vector, well
        # below the 0.35 threshold. Models the real-world case where an
        # unrelated query embeds to a direction far from anything stored.
        h = abs(hash(text)) % 1000
        vec = np.array([-1.0, -1.0, -1.0]) + np.array([0.01 * (h % 3), 0, 0])
    vec = vec.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return None
    return (vec / norm).tobytes()


def _embed_returns_none(self, text):
    return None


def _assert(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return False
    print(f"  ok:   {msg}")
    return True


def _seed(db_path, role, content, vec_text, ts_offset=-7200, modality="chat"):
    from chloe_memory import ChloeMemory
    mem = ChloeMemory(db_path, db_path.parent / "facts.md")
    ChloeMemory._embed = _toy_embed
    blob = mem._embed(vec_text)
    with sqlite3.connect(str(db_path)) as c:
        c.execute(
            "INSERT INTO turns(ts, role, content, modality, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time() + ts_offset, role, content, modality, blob),
        )


def main():
    from chloe_memory import ChloeMemory

    tmp = Path(tempfile.mkdtemp(prefix="memtest_"))
    db_path = tmp / "memory.db"
    facts = tmp / "facts.md"

    # Migration on fresh DB
    mem = ChloeMemory(db_path, facts)
    with sqlite3.connect(str(db_path)) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(turns)").fetchall()}
    ok = True
    ok &= _assert("embedding" in cols, "migration adds embedding column on fresh DB")

    # Migration is idempotent on a legacy DB
    legacy_db = tmp / "legacy.db"
    with sqlite3.connect(str(legacy_db)) as c:
        c.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, modality TEXT DEFAULT 'voice')")
        c.execute("INSERT INTO turns(ts, role, content) VALUES (?, ?, ?)", (time.time(), "user", "legacy seed"))
    ChloeMemory(legacy_db, tmp / "legacy_facts.md")
    with sqlite3.connect(str(legacy_db)) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(turns)").fetchall()}
        seed = c.execute("SELECT content FROM turns WHERE content='legacy seed'").fetchone()
    ok &= _assert("embedding" in cols, "migration adds embedding column to legacy DB")
    ok &= _assert(seed and seed[0] == "legacy seed", "legacy data survives the ALTER")

    # Seed turns
    ChloeMemory._embed = _toy_embed
    for line, role in [
        ("I have a golden retriever named Bella", "user"),
        ("Bella is my favorite dog",              "user"),
        ("My puppy loves to play in the yard",    "user"),
        ("I drive a Honda Civic to work",         "user"),
        ("Traffic was terrible on the freeway",   "user"),
        ("My favorite breakfast is pancakes",     "user"),
    ]:
        _seed(db_path, role, line, line)

    # Top-k by semantic direction
    hits = mem.search_turns("tell me about dogs", limit=3, min_age_hours=0.5)
    ok &= _assert(len(hits) == 3, "dogs query returns 3 hits")
    if hits:
        ok &= _assert("Bella" in hits[0]["content"] or "puppy" in hits[0]["content"],
                      f"dogs query top hit is dog-related (got: {hits[0]['content']!r})")
    if len(hits) >= 3:
        contents = [h["content"] for h in hits]
        ok &= _assert(not any("Civic" in c for c in contents),
                      "Civic is not in top-3 for dogs query")

    hits = mem.search_turns("what do I drive", limit=2, min_age_hours=0.5)
    ok &= _assert(len(hits) == 2, "drive query returns 2 hits")
    if hits:
        ok &= _assert("Civic" in hits[0]["content"] or "Traffic" in hits[0]["content"],
                      f"drive query top hit is vehicle-related (got: {hits[0]['content']!r})")

    hits = mem.search_turns("food I like", limit=1, min_age_hours=0.5)
    ok &= _assert(len(hits) == 1, "food query returns 1 hit")
    if hits:
        ok &= _assert("pancakes" in hits[0]["content"],
                      f"food query top hit mentions pancakes (got: {hits[0]['content']!r})")

    # Return shape
    hits = mem.search_turns("tell me about dogs", limit=1, min_age_hours=0.5)
    if hits:
        ok &= _assert(set(hits[0].keys()) == {"ts", "role", "content", "modality"},
                      f"hit shape is {{ts,role,content,modality}} (got: {set(hits[0].keys())})")

    # min_age_hours filter
    _seed(db_path, "user", "I have a golden retriever named Bella",
          "I have a golden retriever named Bella", ts_offset=0)
    hits = mem.search_turns("tell me about dogs", limit=5, min_age_hours=1.0)
    age_violations = [h for h in hits if (time.time() - float(h["ts"])) < 3600]
    ok &= _assert(len(age_violations) == 0,
                  f"min_age_hours=1.0 excludes turns younger than 1h ({len(age_violations)} violations)")

    # Noise filter: slash commands
    _seed(db_path, "user", "/recall tell me about dogs",
          "I have a golden retriever named Bella")
    hits = mem.search_turns("tell me about dogs", limit=5, min_age_hours=0.5)
    slash_in_hits = [h for h in hits if h["content"].startswith("/")]
    ok &= _assert(len(slash_in_hits) == 0,
                  f"user slash-command turns are filtered from recall (found: {[h['content'] for h in slash_in_hits]})")

    # Noise filter: recall-output assistant turns
    _seed(db_path, "assistant",
          "**Top recall hits for**: _dogs_ (3 of up to 10)\n1. ...",
          "I have a golden retriever named Bella")
    hits = mem.search_turns("tell me about dogs", limit=5, min_age_hours=0.5)
    recall_out = [h for h in hits if h["content"].startswith("**Top recall hits for**")]
    ok &= _assert(len(recall_out) == 0,
                  f"assistant recall-output turns are filtered (found {len(recall_out)})")

    # Similarity threshold
    hits = mem.search_turns("garbage query xyz", limit=5, min_age_hours=0.5)
    ok &= _assert(len(hits) == 0,
                  f"low-similarity queries return 0 hits via threshold (got {len(hits)})")

    # FTS5 fallback
    ChloeMemory._embed = _embed_returns_none
    hits = mem.search_turns("Bella", limit=3, min_age_hours=0.5)
    ok &= _assert(len(hits) >= 1, "FTS5 fallback returns hits when embed fails")
    if hits:
        bella_hits = [h for h in hits if "Bella" in h["content"]]
        ok &= _assert(len(bella_hits) >= 1, "FTS5 fallback finds 'Bella' by keyword match")

    # FTS5 fallback also applies the noise filter
    slash_in_fts = [h for h in hits if h["content"].startswith("/")]
    ok &= _assert(len(slash_in_fts) == 0, "FTS5 fallback also filters slash-command turns")

    print()
    if ok:
        print("ALL PASS")
        return 0
    print("SOME FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
