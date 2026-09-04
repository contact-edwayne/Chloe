"""Shared, cached Ollama query-embedding for Chloe.

Every substantive turn embeds the user's query *twice* — once for conversation
recall (`chloe_memory.search_turns`) and once for wiki auto-inject
(`wiki_embedding.wiki_context_for_query`) — hitting Ollama's `/api/embeddings`
with the identical string back-to-back. Both stores use the same model
(`nomic-embed-text`) and the same L2-normalized float32 layout, so the second
call is pure waste.

This module centralizes the embed call behind a tiny success-only cache keyed on
`(model, url, text)`. nomic embeddings are deterministic for a fixed model, so a
cached vector is always valid — there's no staleness to worry about. The two
call sites delegate here, so the recall+wiki pair collapses to one round-trip per
turn (and repeated queries within a session are free).

Design contract: best-effort, never raises. Returns the L2-normalized float32
`bytes` (BLOB-ready) or `None` on any failure — callers already treat `None` as
"no embedding, fall back to FTS5". Failures are NOT cached, so a transient Ollama
hiccup doesn't poison later turns.
"""

from __future__ import annotations

import threading

import numpy as np
import requests

# Small LRU-ish cache. Embedding a few KB vector is cheap to hold; 64 distinct
# queries is plenty for a single user's session and bounds memory.
_MAX_ENTRIES = 64
_CACHE: "dict[tuple[str, str, str], bytes]" = {}
_CACHE_LOCK = threading.Lock()

# Cheap hit/miss counters for the optional [embed] cache log / debugging.
_STATS = {"hits": 0, "misses": 0}


def _cache_get(key) -> bytes | None:
    with _CACHE_LOCK:
        v = _CACHE.get(key)
        if v is not None:
            _STATS["hits"] += 1
            # Move to most-recently-used end.
            _CACHE.pop(key, None)
            _CACHE[key] = v
        return v


def _cache_put(key, val: bytes) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = val
        while len(_CACHE) > _MAX_ENTRIES:
            # Drop the oldest insertion (FIFO/LRU approximation).
            _CACHE.pop(next(iter(_CACHE)))


def embed(text: str, *, model: str, url: str, timeout: float,
          keep_alive=None, tag: str = "embed") -> bytes | None:
    """Embed `text` via Ollama and return an L2-normalized float32 byte string,
    or None on failure. Identical (model, url, text) requests are served from a
    process-wide cache so recall + wiki only pay the round-trip once per turn."""
    if not text or not text.strip():
        return None
    t = text.strip()
    key = (model, url, t)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        _STATS["misses"] += 1
    try:
        payload = {"model": model, "prompt": t}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        r = requests.post(f"{url}/api/embeddings", json=payload, timeout=timeout)
        r.raise_for_status()
        emb = r.json().get("embedding") or []
        if not emb:
            return None
        arr = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if not norm or not np.isfinite(norm):
            return None
        blob = (arr / norm).tobytes()
        _cache_put(key, blob)
        return blob
    except requests.exceptions.HTTPError as e:
        # requests.HTTPError.__str__ ("400 Client Error: Bad Request for
        # url: ...") drops the response BODY -- which is where Ollama's
        # actual error message lives (e.g. "time: missing unit in
        # duration \"-1\""). Read it explicitly. This was the actual root
        # cause of the 2026-08-31 outage being invisible here: this
        # exact call site 400'd on every turn, but silently, because this
        # handler swallowed the body -- the real message was only ever
        # seen because a different call site's error handler happened to
        # print r.text.
        body = ""
        try:
            if e.response is not None:
                body = e.response.text[:300]
        except Exception:
            pass
        print(f"[{tag}] embed failed ({model} @ {url}): {e} — body: {body}",
              flush=True)
        return None
    except Exception as e:
        print(f"[{tag}] embed failed ({model} @ {url}): {e}", flush=True)
        return None


def stats() -> dict:
    """Return a copy of {hits, misses} for diagnostics."""
    with _CACHE_LOCK:
        return dict(_STATS)
