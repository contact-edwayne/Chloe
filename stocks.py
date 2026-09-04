"""
Real-time stock/ETF quotes for Chloe — Yahoo Finance's chart endpoint, no
API key.

Public API
----------
quote_reply(symbol_or_name: str) -> str
    A spoken-friendly last-price line with the actual quote date/time from
    Yahoo Finance, for a ticker symbol or a name resolvable via the alias
    config.

maybe_stock_reply(user_text: str) -> str | None
    Intent-detect a bare price question ("price of X", "what's X trading
    at", "X stock"), resolve X to a ticker, and return quote_reply(...).
    Returns None when the text isn't a resolvable price ask, so the
    caller falls through to normal routing.

Why this exists (Ed, 2026-09-01, found via live testing): Brave web
search returns crawled snippets, not live data -- for a fast-moving
number like a stock price, the "most recent result" turned out to be a
week or more stale, and different snippets disagreed because they were
never simultaneous quotes to begin with (confirmed live: SLV synthesis
citing an August 24 result for a September 1 question, three different
prices for Duolingo with no dates on any of them). weather.py already
solved this exact class of problem for weather -- a real API instead of
search-and-guess -- checked before the Brave/real-time route ever
fires. This is the same fix, same wiring pattern, for prices.

Provider (revised 2026-09-01, same day): first built against Stooq's
free CSV quote endpoint (stooq.com/q/l/), which turned out to be dead --
404s live, and search turned up reports of Stooq gating its CSV/bulk
downloads behind a CAPTCHA since. Verified live in Ed's own browser
before switching: Yahoo Finance's unofficial chart JSON endpoint
(query2.finance.yahoo.com/v8/finance/chart/<TICKER>) returns a clean,
keyless response with an actual regularMarketPrice + regularMarketTime
(unix timestamp, exact to the second, with the exchange's own timezone
info alongside it) -- the same genuinely-dated-not-ambiguously-"recent"
property Stooq was meant to provide, from a still-live source. No
cookie/crumb auth needed for this specific endpoint (that dance is only
required for some of Yahoo's other endpoints); a normal browser-like
User-Agent is enough. Like any unofficial endpoint it could break or
start rate-limiting without notice -- if quote_reply starts failing
across the board, that's the first thing to check.

Ticker resolution: `symbol_or_name` is used AS-IS if it already looks
like a ticker (1-5 letters, optionally a class suffix like BRK.B).
Otherwise it's looked up in the alias config at
C:\\Chloe\\secrets\\stock_aliases.json -- same pattern as lights.json /
youtube_playlists.json: {"aliases": {"dick's sporting goods": "DKS"}}
mapping a lowercased company name/nickname to its ticker, merged over a
small seed list of names that came up in live testing. An unresolved
name returns None from maybe_stock_reply -- NEVER a guessed ticker --
which falls through to the normal chat/search route, same "honest
miss, don't claim it" gating lights.py/youtube_playlists.py use for
target resolution. Add more names without touching code:
    python stocks.py --alias "Some Company" TICK

Deliberately narrow scope: this only replaces the ONE thing Brave
search does badly for finance questions -- today's actual last price.
Earnings, news, "why did it move", "should I buy" all still go through
the normal Brave/chat route, which is the right tool for those (and now
has its own freshness/conflict-disclosure fix, see search.py).

Phrasing coverage is best-effort, not a full parser -- it catches the
common shapes ("what's the price of X", "X stock", "how much is X
trading at") and safely falls through to Brave on anything messier,
same trade-off already accepted elsewhere in this codebase (e.g.
youtube_playlists.py's skip/stop phrasing gates).

CLI:
    python stocks.py SLV
    python stocks.py "Dick's Sporting Goods"
    python stocks.py --alias "Duolingo" DUOL
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone as _dt_timezone
from pathlib import Path
from typing import Optional

_TIMEOUT = 6.0
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
# Yahoo's chart endpoint is unofficial/keyless but does appear to gate
# on User-Agent -- Chloe's own descriptive UA got nothing back in
# testing against Stooq (which didn't seem to care either way), so
# this one deliberately looks like a real browser rather than
# identifying itself, matching what actually worked live.

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CONFIG_PATH = SECRETS_DIR / "stock_aliases.json"

# Seed aliases for names that came up in live testing (2026-09-01) --
# the config file (built via --alias) is merged OVER these, so a config
# entry always wins if Ed ever wants to override one.
_SEED_ALIASES = {
    "silver": "SLV",                        # iShares Silver Trust
    "dick's sporting goods": "DKS",
    "dicks sporting goods": "DKS",
    "duolingo": "DUOL",
    # A handful of obvious household names, added defensively (2026-09-01)
    # so the common case works out of the box -- everything else grows
    # via --alias as Ed actually asks about it, same as the rest of this
    # seed list.
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "nvidia": "NVDA",
}

# Quote cache: short TTL -- unlike weather's slow-moving numbers, the
# whole point here is freshness, so this only exists to stop a rapid
# back-to-back re-ask (a voice follow-up window retry, e.g.) from
# hitting Yahoo twice for the same symbol within the same few seconds.
_quote_cache: dict = {}
_QUOTE_TTL_S = 30.0

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

_PRICE_TRIGGERS = (
    "price of", "price for", "stock price", "share price",
    "trading at", "trades at", "worth right now", "what's it worth",
    "how much is", "quote for", "quote on",
)
_STOCK_WORD_RE = re.compile(r"\b(?:stock|shares?|ticker|etf|fund)\b", re.I)


def _load_aliases() -> dict:
    aliases = dict(_SEED_ALIASES)
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in (data.get("aliases") or {}).items():
                if k and v:
                    aliases[k.strip().lower()] = v.strip().upper()
    except Exception as e:
        print(f"[stocks] alias config read failed: {e}", file=sys.stderr)
    return aliases


def _save_alias(name: str, ticker: str) -> None:
    data = {"aliases": {}}
    try:
        if CONFIG_PATH.exists():
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except Exception:
        pass
    data.setdefault("aliases", {})
    data["aliases"][name.strip().lower()] = ticker.strip().upper()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Short ALL-CAPS words that are common in ordinary speech and would
# otherwise false-positive as a ticker in the standalone-caps-token scan
# below (resolve_ticker's last resort). Not exhaustive -- just the ones
# likely to actually come up.
_CAPS_FALSE_POSITIVES = {
    "OK", "US", "USA", "CEO", "CFO", "ETF", "IRS", "ASAP", "FYI",
    "ID", "IT", "AI", "TV", "PC", "DIY", "FAQ", "PM", "AM",
}


def resolve_ticker(text: str) -> Optional[str]:
    """Try, in order: (1) the whole cleaned text as a bare ticker --
    covers direct use (CLI, quote_reply called with just a symbol); (2)
    an alias-key substring match anywhere in the text, longest/most-
    specific key wins on overlap -- same reverse-containment resolution
    style as lights.py's _resolve_targets / youtube_playlists.py's
    _resolve_playlist, so "thank you, what's the price of Dick's
    Sporting Goods" resolves without needing the sentence scaffolding
    stripped first; (3) a standalone ALL-CAPS 1-5 letter word in the
    ORIGINAL (not lowercased) text -- Whisper does preserve casing for
    tickers/acronyms it recognizes (confirmed live: "SLV" transcribed
    in caps), so a bare ticker said aloud amid a full sentence is often
    directly recoverable this way even with no alias entry. None if
    nothing resolves -- callers must never guess."""
    raw = (text or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if _TICKER_RE.match(upper):
        return upper

    low = raw.lower()
    aliases = _load_aliases()
    hits = [(key, tick) for key, tick in aliases.items() if key in low]
    if hits:
        hits.sort(key=lambda kv: len(kv[0]), reverse=True)
        return hits[0][1]

    for word in re.findall(r"[A-Za-z]{1,5}", raw):
        if (word.isupper() and len(word) >= 2
                and word not in _CAPS_FALSE_POSITIVES
                and _TICKER_RE.match(word)):
            return word
    return None


def _fmt_when(meta: dict) -> str:
    """Human-readable quote timestamp from Yahoo's meta block, in the
    exchange's own local time (gmtoffset is seconds-from-UTC, timezone
    is its abbreviation, e.g. "EDT") -- not Ed's local time and not
    UTC, so it always matches what a human would read off the exchange
    itself. Built manually rather than with %-d/%-I strftime flags --
    those are POSIX-only and this runs on Windows, where the equivalent
    is %#d/%#I; doing it by hand avoids the platform split entirely."""
    ts = meta.get("regularMarketTime")
    if not ts:
        return "an unknown time"
    try:
        gmtoffset = int(meta.get("gmtoffset") or 0)
        dt = datetime.fromtimestamp(
            int(ts), tz=_dt_timezone(timedelta(seconds=gmtoffset)))
        hour12 = dt.strftime("%I").lstrip("0") or "12"
        stamp = f"{dt.strftime('%B')} {dt.day}, {dt.year} {hour12}:{dt.strftime('%M %p')}"
    except Exception:
        return "an unknown time"
    tz_abbr = (meta.get("timezone") or "").strip()
    return f"{stamp} {tz_abbr}".strip()


def _fetch_quote(ticker: str) -> Optional[dict]:
    """Raw Yahoo Finance chart-endpoint quote for `ticker`. None on any
    failure -- caller reports honestly rather than raising into a
    voice/chat turn."""
    key = ticker.upper()
    now = time.time()
    hit = _quote_cache.get(key)
    if hit and now - hit[0] < _QUOTE_TTL_S:
        return hit[1]
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(key)}?" +
           urllib.parse.urlencode({"range": "1d", "interval": "1d"}))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[stocks] Yahoo fetch failed for {key}: {e}", file=sys.stderr)
        return None
    chart = payload.get("chart") or {}
    if chart.get("error"):
        print(f"[stocks] Yahoo reported an error for {key}: {chart['error']}",
              file=sys.stderr)
        return None
    results = chart.get("result") or []
    if not results:
        return None
    meta = results[0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        return None
    result = {
        "symbol": meta.get("symbol", key),
        "name": meta.get("longName") or meta.get("shortName") or "",
        "price": price,
        "when": _fmt_when(meta),
    }
    _quote_cache[key] = (now, result)
    return result


def quote_reply(symbol_or_name: str) -> str:
    ticker = resolve_ticker(symbol_or_name)
    if not ticker:
        return (f"I don't have a ticker for \"{symbol_or_name}\" — tell me "
                f"the symbol, or I can add it if you give me one.")
    q = _fetch_quote(ticker)
    if not q:
        return (f"I couldn't get a quote for {ticker} just now — "
                f"the symbol might be wrong, or the quote service is down.")
    try:
        price = float(q["price"])
    except (TypeError, ValueError):
        return f"I got a response for {ticker} but couldn't read a price out of it."
    label = f"{q['name']} ({ticker})" if q.get("name") else ticker
    return f"{label} last traded at ${price:,.2f}, as of {q['when']}."


def maybe_stock_reply(user_text: str):
    """Return a stock-quote reply if `user_text` is a bare price question
    AND resolves to a known ticker, else None. Deliberately narrow: only
    the "what's the price of X" shape. News, earnings, "why did it
    move", and "should I buy" all fall through to the normal Brave/chat
    route, which is right for those -- this module only replaces the one
    thing Brave search does badly (today's actual last price). Resolves
    against the FULL original text (not a stripped fragment) -- see
    resolve_ticker's docstring for why that's the more robust order."""
    if not user_text:
        return None
    low = user_text.lower()
    has_trigger = (any(t in low for t in _PRICE_TRIGGERS)
                   or bool(_STOCK_WORD_RE.search(low)))
    if not has_trigger:
        return None
    ticker = resolve_ticker(user_text)
    if not ticker:
        return None  # unresolved name -- fall through, never guess
    return quote_reply(ticker)


def _cli() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--alias":
        if len(args) < 3:
            print('usage: python stocks.py --alias "<name>" <TICKER>')
            return 1
        _save_alias(args[1], args[2])
        print(f"saved alias {args[1].strip().lower()!r} -> {args[2].strip().upper()}")
        return 0
    print(quote_reply(" ".join(args)))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
