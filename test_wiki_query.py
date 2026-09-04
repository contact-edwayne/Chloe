"""test_wiki_query.py - Sanity tests for the wiki auto-inject path.

Covers looks_like_wiki_query, format_wiki_block, and wiki_context_for_query
(all in wiki_embedding.py) without hitting Ollama or the real wiki DB.
The wiki_context_for_query tests monkey-patch wiki_embedding.get_store with
controllable fakes - same pattern as test_wiki_write.py.

Run from the jarvis dir:
    python test_wiki_query.py
Exit code 0 on success, non-zero on any failure.
"""

import sys

import wiki_embedding as we
from wiki_embedding import (
    looks_like_wiki_query,
    format_wiki_block,
    wiki_context_for_query,
)

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


# --- looks_like_wiki_query: positives -------------------------------------
print("looks_like_wiki_query - positives")
for q in [
    "what is the kelly criterion",
    "What's a flow field?",
    "tell me about obsidian",
    "tell me more about the brain pipeline",
    "explain how nomic embeddings work",
    "can you explain cosine similarity",
    "who is edward wayne",
    "define idempotence",
    "describe the watcher",
    "how does the wiki watcher stay in sync",
    "how do i tune the threshold",
    "how to back up jarvis",
    "why is ollama needed",
    "do you know about tailscale",
    "catch me up on the social media plan",
    "remind me what the path-stem boost does",
    "what's the deal with the boot signal race",
]:
    check(f"positive: {q!r}", looks_like_wiki_query(q) is True, "expected True")

# --- looks_like_wiki_query: negatives -------------------------------------
print("looks_like_wiki_query - negatives")
for q in [
    "",
    "   ",
    "hey chloe",
    "good morning",
    "thanks!",
    "lol that was great",
    "yeah exactly",
    "turn the lights on",
    "play some music",
    "remember: i take my coffee black",
    "/wiki kelly criterion",
    "/recall the brain project",
    "/wiki_write kelly criterion",
    "/help",
]:
    check(f"negative: {q!r}", looks_like_wiki_query(q) is False, "expected False")
check("negative: None", looks_like_wiki_query(None) is False, "expected False")

# --- looks_like_wiki_query: case & whitespace -----------------------------
print("looks_like_wiki_query - case & whitespace")
check("uppercase fires", looks_like_wiki_query("WHAT IS X") is True)
check("padded fires", looks_like_wiki_query("   tell me about Y   ") is True)
check("padded slash command still skipped",
      looks_like_wiki_query("   /wiki foo") is False)
# Documented known looseness: substring match means a phrase that merely
# CONTAINS a keyword fires ("somewhat isolated" contains "what is"). That's
# acceptable - it costs one embed call the threshold then drops. Pinned
# here so any future tightening is a conscious choice, not an accident.
check("known-loose: 'somewhat isolated' contains 'what is' -> True",
      looks_like_wiki_query("that was somewhat isolated") is True)

# --- format_wiki_block ----------------------------------------------------
print("format_wiki_block")
check("empty list -> ''", format_wiki_block([]) == "")
check("None -> ''", format_wiki_block(None) == "")

one = [{"path": "concepts/kelly_criterion.md", "title": "Kelly Criterion",
        "type": "concept", "snippet": "Optimal bet sizing formula.",
        "score": 0.91}]
b = format_wiki_block(one)
check("one hit: has header", "## Possibly relevant wiki pages:" in b)
check("one hit: has title", "Kelly Criterion" in b)
check("one hit: has type tag", "[concept]" in b)
check("one hit: has snippet", "Optimal bet sizing formula." in b)
check("one hit: has guidance line", "from your own wiki" in b)

multi = [
    {"path": "a.md", "title": "Alpha", "type": "concept",
     "snippet": "first body", "score": 0.9},
    {"path": "b.md", "title": "Beta", "type": "entity",
     "snippet": "second body", "score": 0.8},
]
b2 = format_wiki_block(multi)
check("multi: both titles present", "Alpha" in b2 and "Beta" in b2)
check("multi: both snippets present", "first body" in b2 and "second body" in b2)

b3 = format_wiki_block([{"path": "concepts/foo.md", "snippet": "x",
                         "type": "concept", "score": 0.7}])
check("missing title -> path fallback", "concepts/foo.md" in b3)

b4 = format_wiki_block([{"path": "p.md", "title": "P", "snippet": "x",
                         "score": 0.7}])
check("missing type -> 'page'", "[page]" in b4)

long_snip = "z" * 600
b5 = format_wiki_block([{"path": "p.md", "title": "P", "type": "concept",
                         "snippet": long_snip, "score": 0.7}])
check("long snippet truncated with marker", "[…]" in b5 and len(b5) < 800)

b6 = format_wiki_block([{"path": "p.md", "title": "P", "type": "concept",
                         "snippet": "", "score": 0.7}])
check("empty snippet doesn't crash", "P" in b6)

# --- wiki_context_for_query -----------------------------------------------
print("wiki_context_for_query")

_orig_get_store = we.get_store


class _FakeStore:
    def __init__(self, embedded=5, hits=None, search_raises=False):
        self._embedded = embedded
        self._hits = hits or []
        self._search_raises = search_raises
        self.search_calls = []

    def count_embedded(self):
        return self._embedded

    def search(self, text, limit=5, threshold=None):
        self.search_calls.append((text, limit, threshold))
        if self._search_raises:
            raise RuntimeError("boom")
        return self._hits


def _set_store(store_or_exc):
    if isinstance(store_or_exc, Exception):
        def _raiser(*a, **k):
            raise store_or_exc
        we.get_store = _raiser
    else:
        we.get_store = lambda *a, **k: store_or_exc


try:
    # not a wiki query -> '' and the store is never touched
    sentinel = _FakeStore(embedded=99, hits=[
        {"path": "x.md", "title": "X", "type": "concept",
         "snippet": "s", "score": 1.0}])
    _set_store(sentinel)
    r = wiki_context_for_query("good morning chloe")
    check("non-wiki query -> ''", r == "")
    check("non-wiki query -> store.search not called",
          sentinel.search_calls == [])

    # wiki query but empty corpus -> ''
    empty_store = _FakeStore(embedded=0)
    _set_store(empty_store)
    r = wiki_context_for_query("what is the kelly criterion")
    check("empty corpus -> ''", r == "")

    # wiki query with hits -> formatted block; search gets limit + threshold
    hit_store = _FakeStore(embedded=10, hits=[
        {"path": "concepts/kelly_criterion.md", "title": "Kelly Criterion",
         "type": "concept", "snippet": "Optimal bet sizing.", "score": 0.88}])
    _set_store(hit_store)
    r = wiki_context_for_query("what is the kelly criterion", limit=2)
    check("hits -> formatted block",
          "Kelly Criterion" in r and "## Possibly relevant wiki pages:" in r)
    check("search called with limit=2",
          bool(hit_store.search_calls) and hit_store.search_calls[0][1] == 2)
    check("search called with the inject threshold",
          bool(hit_store.search_calls)
          and hit_store.search_calls[0][2] == we._WIKI_INJECT_THRESHOLD)

    # wiki query, nothing clears threshold -> ''
    nohit_store = _FakeStore(embedded=10, hits=[])
    _set_store(nohit_store)
    r = wiki_context_for_query("what is some obscure thing")
    check("query but no hits -> ''", r == "")

    # store.search raises -> '' (must never propagate)
    boom_store = _FakeStore(embedded=10, search_raises=True)
    _set_store(boom_store)
    try:
        r = wiki_context_for_query("what is the kelly criterion")
        check("search raises -> '' (swallowed)", r == "")
    except Exception as e:
        check("search raises -> '' (swallowed)", False, f"raised: {e}")

    # get_store itself raises -> '' (must never propagate)
    _set_store(RuntimeError("store init failed"))
    try:
        r = wiki_context_for_query("what is the kelly criterion")
        check("get_store raises -> '' (swallowed)", r == "")
    except Exception as e:
        check("get_store raises -> '' (swallowed)", False, f"raised: {e}")
finally:
    we.get_store = _orig_get_store

# --- summary --------------------------------------------------------------
print()
print(f"{PASSED} passed, {FAILED} failed")
sys.exit(0 if FAILED == 0 else 1)
