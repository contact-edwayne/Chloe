"""
url_verify.py — Live HEAD-check for citation URLs, with a persistent cache.

Used by brain_wiring._search_call to drop dead/fabricated-looking citation
URLs before they land in a wiki page. A confirmed 404 means "don't cite
this." Anything else (200/30x, or inconclusive — 401/403/429/timeout,
which paywalled news sites return to every request regardless of
validity) is treated as citable; we'd rather occasionally keep a URL we
couldn't fully verify than strip real citations from sites that just
bot-block HEAD requests.

Cache is a flat JSON file so repeated job runs (and repeated pages citing
the same source, e.g. bls.gov/cpi/) don't re-check the same URL. Entries
expire after CHLOE_URL_VERIFY_CACHE_DAYS (default 7) so a since-fixed or
since-broken link gets re-evaluated eventually.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_CACHE_PATH = Path(os.environ.get(
    "CHLOE_URL_VERIFY_CACHE",
    str(Path(__file__).resolve().parent / "url_verify_cache.json")))
_CACHE_TTL_S = float(os.environ.get("CHLOE_URL_VERIFY_CACHE_DAYS", "7")) * 86400
_TIMEOUT_S = float(os.environ.get("CHLOE_URL_VERIFY_TIMEOUT", "6"))

_lock = threading.Lock()
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(_cache, indent=0), encoding="utf-8")
    except Exception:
        pass  # cache is a pure optimization; a write failure is never fatal


def verify_url(url: str) -> bool:
    """Return False only when `url` is CONFIRMED dead (a clean HTTP 404).
    Everything else — resolves fine, blocked (401/403/429), timeout,
    network error, or malformed URL we can't even attempt — returns True
    ("citable"), since a bot-block on a paywalled site is not evidence of
    fabrication and we'd rather under-strip than remove real citations.
    """
    if not url or not url.startswith(("http://", "https://")):
        return True
    with _lock:
        cache = _load_cache()
        entry = cache.get(url)
        now = time.time()
        if entry and (now - entry.get("checked_at", 0)) < _CACHE_TTL_S:
            return entry["citable"]

    citable = True
    try:
        req = urllib.request.Request(url, method="HEAD", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        })
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
            citable = r.status != 404
    except urllib.error.HTTPError as e:
        citable = e.code != 404
    except Exception:
        citable = True  # network error / timeout — inconclusive, don't penalize

    with _lock:
        cache = _load_cache()
        cache[url] = {"citable": citable, "checked_at": time.time()}
        _save_cache()
    return citable


def verify_urls(urls: list[str]) -> dict[str, bool]:
    """Batch convenience wrapper. Returns {url: citable}."""
    return {u: verify_url(u) for u in urls}
