"""
Brave Search Web API client for Chloe.

Independent web search backend. Use cases:

  1. Explicit /search slash command — user types `/search hans zimmer career`,
     Chloe synthesizes an answer with cited sources.
  2. Auto-fallback when local qwen hedges and Groq compound-mini is also
     unavailable (rate-limited or down). jarvis.py's hedge-retry chain calls
     web_search() as the final web-data step before giving up.

Brave free tier: 2000 queries/month, 1 query/second. API key from
https://api-dashboard.search.brave.com/app/keys → set BRAVE_API_KEY in
C:\\Users\\eleew\\Documents\\jarvis\\.env.

Surfaces:
    web_search(query, count=5)         -> list of {title, url, description, domain}
    format_for_context(results, query) -> str block to inject into LLM prompt
    try_handle_search_command(text)    -> {query, results, error?} or None

CLI:
    python search.py "your query here"
    python search.py --test
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_COUNT = 5
DEFAULT_TIMEOUT_S = 8.0
USER_AGENT = "Chloe/0.1 (+https://github.com/contact-edwayne/Chloe)"

# Free tier is 1 query/second. Track last call so back-to-back queries
# (rare in chat, common during scripted tests) don't 429.
_last_call_ts: float = 0.0
_MIN_INTERVAL_S = 1.05

# Tiny in-memory cache so repeat queries during demos don't burn quota.
# {(query_lower, count): (ts, results)}
_cache: dict = {}
_CACHE_TTL_S = 600  # 10 minutes


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _get_api_key() -> Optional[str]:
    """Look up the Brave subscription token. Accepts BRAVE_API_KEY or
    BRAVE_SEARCH_API_KEY for flexibility with different .env conventions."""
    key = os.environ.get("BRAVE_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY")
    return key.strip() if key else None


def _throttle() -> None:
    """Sleep just enough to stay under Brave's 1 QPS free-tier cap."""
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _last_call_ts = time.time()


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _dedup_by_domain(results: list, max_per_domain: int = 2) -> list:
    """Drop results past the Nth from the same domain, preserving order.
    Ed, 2026-09-01 (info-quality pass, found via live testing): Brave
    sometimes returns several near-duplicate hits from the same
    aggregator domain, which burns result slots without adding any real
    cross-checking value for the synthesis step -- worse, it can make a
    single source's figure look independently "confirmed" by 2-3 results
    that are really all the same page/data. Capped at 2/domain rather
    than 1 so a domain that genuinely has two distinct relevant pages
    (e.g. a live quote page AND a historical-data page) doesn't lose one
    for no reason."""
    counts: dict = {}
    out = []
    for r in results:
        d = r.get("domain", "")
        counts[d] = counts.get(d, 0) + 1
        if counts[d] <= max_per_domain:
            out.append(r)
    return out


_HTML_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(text: str) -> str:
    """Brave wraps matched terms in <strong> in description/title fields."""
    if not text:
        return ""
    return _HTML_TAG_RE.sub("", text)


def _load_dotenv_if_present() -> None:
    """For standalone CLI use only. jarvis.py already loads .env at startup;
    when search.py is imported from jarvis, this is a no-op."""
    if _get_api_key():
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        print(f"[search] .env load skipped: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Core API call                                                               #
# --------------------------------------------------------------------------- #

def web_search(query: str, count: int = DEFAULT_COUNT, *, fresh: bool = False) -> list:
    """Query Brave Search.

    Returns list of {title, url, description, domain}, length <= count.

    Soft failures (HTTP error, network timeout, malformed response) print to
    stderr and return []. Caller decides whether to retry or surface the
    error to the user. Hard failure (missing API key) raises RuntimeError.

    fresh=True bypasses the in-memory cache.
    """
    if not query or not query.strip():
        return []

    q = query.strip()
    cache_key = (q.lower(), count)

    if not fresh:
        cached = _cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL_S:
            return cached[1]

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY not set. Add it to "
            r"C:\Users\eleew\Documents\jarvis\.env  "
            "(get one at https://api-dashboard.search.brave.com/app/keys)"
        )

    _throttle()

    params = urlencode({
        "q": q,
        "count": max(1, min(int(count), 20)),
        "safesearch": "moderate",
        "result_filter": "web",
        "text_decorations": "false",
    })
    url = f"{BRAVE_API_ENDPOINT}?{params}"
    req = Request(url, headers={
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
        "User-Agent": USER_AGENT,
    })

    try:
        with urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            payload = json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            raw_body = e.read()
            # The request sent Accept-Encoding: gzip, and Brave honors it on
            # error responses too -- the success path already decompresses
            # (see above); this branch didn't, so error bodies were being
            # decoded as UTF-8 straight off compressed bytes, producing
            # unreadable garbage instead of the actual error detail.
            if e.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw_body = gzip.decompress(raw_body)
            body = raw_body.decode("utf-8", errors="replace")[:500]
        except Exception as decode_err:
            body = f"(could not decode error body: {decode_err})"
        print(f"[search] Brave API HTTP {e.code}: {body}", file=sys.stderr)
        return []
    except URLError as e:
        print(f"[search] network error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[search] unexpected error: {e}", file=sys.stderr)
        return []

    web = (payload.get("web") or {}).get("results") or []
    results = []
    for r in web[:count]:
        results.append({
            "title":       _strip_html(r.get("title", "") or ""),
            "url":         r.get("url", "") or "",
            "description": _strip_html(r.get("description", "") or ""),
            "domain":      _domain_of(r.get("url", "") or ""),
            # Brave's own freshness signal for this result, when it has
            # one -- "age" is the human-readable form ("3 days ago"),
            # page_age an ISO timestamp; either is a lot better than the
            # LLM guessing which of several same-looking results is
            # newest. Both keys are absent on plenty of results (Brave
            # doesn't always have a crawl/publish date for a page), so
            # this is best-effort -- format_for_context only shows it
            # when non-empty.
            "age":         (r.get("age") or r.get("page_age") or "").strip(),
        })
    results = _dedup_by_domain(results)

    _cache[cache_key] = (time.time(), results)
    return results


# --------------------------------------------------------------------------- #
# LLM context formatter                                                       #
# --------------------------------------------------------------------------- #

def format_for_context(results: list, query: str = "") -> str:
    """Format search results as a compact context block for an LLM prompt.
    Numbered citations [1], [2], ... let the model cite specific sources.
    Capped snippet length keeps the block under ~1.5K tokens for 5 results."""
    if not results:
        return f"(no web search results for: {query})" if query else "(no results)"

    lines = []
    if query:
        lines.append(f"Web search results for: {query}")
        lines.append("")
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "(untitled)").strip()
        domain = (r.get("domain") or "").strip()
        age = (r.get("age") or "").strip()
        snippet = (r.get("description") or "").strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        header = f"[{i}] {title} ({domain}" + (f", {age}" if age else "") + ")"
        lines.append(header)
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Slash-command handler                                                       #
# --------------------------------------------------------------------------- #

def try_handle_search_command(text: str, *, fetch: bool = True) -> Optional[dict]:
    """If `text` is an explicit /search-family slash command, return a dict
    {query, results, error?}. Otherwise return None so later handlers (or
    the LLM router) can take the turn.

    Accepted forms:
        /search <query>
        /lookup <query>     (alias)
        /web <query>        (alias)

    Natural-language triggers ("look up X", "what's the latest on Y") are
    intentionally NOT handled here. Those route through the existing
    realtime-keyword path in jarvis.py, which can fall back to compound-mini
    or local qwen as appropriate.

    Note: this handler does NOT synthesize a reply — it just fetches the
    results. jarvis.py owns the LLM call so it can stream a Chloe-voiced
    answer with citations. Returning structured data here keeps the LLM
    code path in one place.
    """
    if not text:
        return None
    cleaned = text.strip()
    lower = cleaned.lower()

    query = None
    if lower.startswith("/search "):
        query = cleaned[8:].strip()
    elif lower.startswith("/lookup "):
        query = cleaned[8:].strip()
    elif lower.startswith("/web "):
        query = cleaned[5:].strip()
    elif lower in ("/search", "/lookup", "/web"):
        return {"query": "", "results": [], "error": "usage: /search <your query>"}

    if query is None:
        return None  # not a search command at all
    if not query:
        return {"query": "", "results": [], "error": "usage: /search <your query>"}

    if not fetch:
        # Parse-only: caller (jarvis.py's /search handling) is about to
        # fetch+synthesize itself via _brave_search_core -- a second
        # web_search() call here would just be a wasted, redundant Brave
        # API hit against the same query.
        return {"query": query, "results": []}

    try:
        results = web_search(query, count=DEFAULT_COUNT)
    except RuntimeError as e:
        return {"query": query, "results": [], "error": str(e)}

    return {"query": query, "results": results}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _cli() -> int:
    _load_dotenv_if_present()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--test":
        query = "Hans Zimmer recent score"
    else:
        query = " ".join(args)

    print(f"searching: {query}")
    try:
        results = web_search(query, count=5)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not results:
        print("(no results — see [search] errors above for cause)")
        return 1

    print(format_for_context(results, query))
    print()
    print(f"({len(results)} result(s))")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
