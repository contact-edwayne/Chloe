"""Callback-novelty TTL for Chloe — Phase 5.

The persona's proactive-callback habit ("oh, like that mic thing last week!")
is charming the first time and grating the third. This module remembers which
recalled memories Chloe actually SURFACED in a reply and stops re-injecting
them for a while, so a callback lands once and then rests.

Flow per turn (wired in jarvis):
  1. recall hits selected → `filter_hits(hits, probe=…)` drops any hit whose
     content she already surfaced within the TTL. Explicit memory probes
     ("what did I say about …") BYPASS the filter — if Ed asks, he gets it.
  2. the surviving hits are registered via `note_injected([contents])`.
  3. when the reply is emitted — `_push_history("assistant", …)`, the single
     choke point every reply flows through — `note_reply(text)` checks which
     injected items the model actually used (content-word overlap, so
     paraphrase counts) and stamps those into a TTL ledger persisted at
     `<brain>/raw/callback_ttl.json`.

Suppression is by USE, not by injection: a memory that was injected but never
voiced stays available — only ones Ed actually heard go to rest.

Best-effort + never raises. `CHLOE_CALLBACK_TTL_MIN=0` disables entirely.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

try:
    from chloe_lock import locked
except Exception:  # pragma: no cover - lock is best-effort
    import contextlib

    @contextlib.contextmanager
    def locked(*_a, **_k):
        yield

# Minutes a surfaced callback rests before it may be injected again. The 6h
# dialogue-state session gap makes ~2h ≈ "not again this session".
TTL_MIN = float(os.environ.get("CHLOE_CALLBACK_TTL_MIN", "120"))
ENABLED = TTL_MIN > 0
_MAX_LEDGER = 64
_MAX_WORDS = 12          # distinctive words sampled per item
_MIN_SHARED = 3          # floor for the overlap test

_STOP = set("the a an and or but to of in on for is it i you he she we they that "
            "this with as at be do my your me him her them so if not no yes was "
            "are were have has had about what when where which there here just "
            "like really very been being its his hers".split())

_LOCK = threading.Lock()
_PENDING: list = []          # [(key, [words])] — items injected this turn
_LEDGER: dict | None = None  # {key: ts_surfaced}; lazy-loaded


def _path() -> str:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    return os.path.join(root, "raw", "callback_ttl.json")


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _key(content: str) -> str:
    return _norm(content)[:60]


def _words(content: str) -> list:
    out = []
    for w in re.findall(r"[a-z']+", (content or "").lower()):
        if len(w) > 3 and w not in _STOP and w not in out:
            out.append(w)
        if len(out) >= _MAX_WORDS:
            break
    return out


def _load() -> dict:
    global _LEDGER
    if _LEDGER is not None:
        return _LEDGER
    try:
        with open(_path(), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        _LEDGER = {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except Exception:
        _LEDGER = {}
    return _LEDGER


def _prune(led: dict) -> dict:
    cutoff = time.time() - TTL_MIN * 60.0
    fresh = {k: v for k, v in led.items() if v >= cutoff}
    if len(fresh) > _MAX_LEDGER:
        fresh = dict(sorted(fresh.items(), key=lambda kv: kv[1])[-_MAX_LEDGER:])
    return fresh


def _save(led: dict) -> None:
    try:
        with locked("callback_ttl"):
            p = _path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(led, fh)
            os.replace(p + ".tmp", p)
    except Exception:
        pass


def filter_hits(hits, probe: bool = False):
    """Drop recall hits Chloe already surfaced within the TTL. Returns
    (kept_hits, suppressed_count). `probe=True` (explicit memory question)
    bypasses the filter entirely. Never raises."""
    if not ENABLED or probe or not hits:
        return list(hits or []), 0
    try:
        with _LOCK:
            led = _prune(_load())
            _LEDGER.clear()
            _LEDGER.update(led)
        kept, suppressed = [], 0
        for h in hits:
            if _key(str(h.get("content") or "")) in led:
                suppressed += 1
                continue
            kept.append(h)
        return kept, suppressed
    except Exception:
        return list(hits or []), 0


def note_injected(contents) -> None:
    """Register the recall contents injected into THIS turn's prompt (replaces
    the previous turn's pending set). Never raises."""
    if not ENABLED:
        return
    try:
        pend = []
        for c in (contents or []):
            c = str(c or "")
            if c.strip():
                pend.append((_key(c), _words(c)))
        with _LOCK:
            _PENDING.clear()
            _PENDING.extend(pend)
    except Exception:
        pass


def note_reply(reply_text: str) -> int:
    """Called on every assistant reply: any pending injected memory whose
    distinctive words substantially appear in the reply was SURFACED — stamp it
    into the TTL ledger so it rests. Clears the pending set. Returns the number
    rested. Never raises."""
    if not ENABLED:
        return 0
    try:
        with _LOCK:
            pend = list(_PENDING)
            _PENDING.clear()
        if not pend or not (reply_text or "").strip():
            return 0
        reply_words = set(re.findall(r"[a-z']+", reply_text.lower()))
        now = time.time()
        rested = 0
        with _LOCK:
            led = _prune(_load())
            for key, words in pend:
                if not words:
                    continue
                need = max(_MIN_SHARED, (len(words) + 1) // 2)
                if len([w for w in words if w in reply_words]) >= need:
                    led[key] = now
                    rested += 1
            if rested:
                _LEDGER.clear()
                _LEDGER.update(led)
        if rested:
            _save(led)
            print(f"[callbacks] {rested} memory callback(s) resting for "
                  f"{TTL_MIN:.0f}min", flush=True)
        return rested
    except Exception:
        return 0
