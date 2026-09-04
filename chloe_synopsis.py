"""Rolling conversation synopsis for Chloe — Phase 1 of the 2026-06-01 plan.

Long sessions fall off a cliff: the model only sees the last ~30 turns
(`_trim_messages_for_model`), and anything older survives only if it happens to
be semantically recalled. This module keeps a running summary of the turns that
scroll out of that window and injects it as a compact `## Conversation so far`
block, so the arc of a marathon session isn't lost.

Cost discipline (this is the one Phase-1 piece that needs an LLM call):
- Fires ONLY on genuinely long sessions (> SUMM_KEEP + MIN_OLDER turns). Short
  sessions return '' immediately — zero cost on the common path.
- Caches the summary and only re-summarizes once every REFRESH_STEP newly
  evicted turns, so the call is infrequent and amortized.
- Summarizes with a small fast local model via Ollama; runs behind
  `asyncio.to_thread` at the call site so it never blocks the event loop.
- Pure-ish + never raises: any failure (Ollama down, timeout, bad cache) yields
  '' so a turn can never stall or break here.

Public API:
    synopsis_block(messages) -> str   # '' unless the session is long enough
"""

from __future__ import annotations

import json
import os
import time

import requests

# 2026-08-27: lowered from 30/6/10. The trigger (SUMM_KEEP + MIN_OLDER = 36)
# was never reachable in practice -- _voice_history/_HISTORY_MAX and the HUD
# clients' own history caps kept the live turn count well under 36, so this
# safety net could never fire and long sessions just silently lost anything
# older than the raw window. 14/8/6 (threshold 22) is reachable within the
# current _HISTORY_MAX=30 window with real headroom to spare.
SUMM_KEEP = int(os.environ.get("CHLOE_SYNOPSIS_KEEP", "14"))      # turns the model still sees
MIN_OLDER = int(os.environ.get("CHLOE_SYNOPSIS_MIN_OLDER", "8"))  # min evicted turns worth summarizing
REFRESH_STEP = int(os.environ.get("CHLOE_SYNOPSIS_REFRESH", "6"))  # re-summarize every N new evictions
MODEL = os.environ.get("CHLOE_SYNOPSIS_MODEL", "llama3.2:3b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
TIMEOUT = int(os.environ.get("CHLOE_SYNOPSIS_TIMEOUT", "20"))
ENABLED = os.environ.get("CHLOE_SYNOPSIS", "1") != "0"


def _cache_path() -> str:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    return os.path.join(root, "raw", "synopsis_cache.json")


def _load_cache() -> dict:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_cache(d: dict) -> None:
    try:
        p = _cache_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False)
        os.replace(p + ".tmp", p)
    except Exception:
        pass


def _turns(messages) -> list:
    return [
        m for m in (messages or [])
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]


def _summarize(older: list) -> str:
    convo = []
    for m in older:
        who = "Ed" if m.get("role") == "user" else "Chloe"
        c = " ".join(str(m.get("content") or "").split())
        convo.append(f"{who}: {c[:500]}")
    text = "\n".join(convo)[:6000]
    prompt = (
        "Summarize the earlier part of this conversation between Ed and his "
        "assistant Chloe in 3-4 sentences. Capture the concrete topics, any "
        "decisions reached, and any still-open threads. Be specific and "
        "factual; no preamble, no bullet points.\n\n"
        + text + "\n\nSummary:"
    )
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 180, "temperature": 0.2},
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return " ".join((r.json().get("response") or "").split()).strip()


def synopsis_block(messages, allow_build: bool = True) -> str:
    """Return a `## Conversation so far` block for long sessions, else ''.

    Safe to call behind asyncio.to_thread on every turn — it short-circuits to
    '' for normal-length sessions and only hits the model when enough turns have
    scrolled out of the model's window. Never raises.

    `allow_build=False` is cache-only: it returns whatever summary already exists
    but never triggers the (blocking) LLM summarize — used on the sync voice path
    so a spoken turn can show the synopsis without stalling on a model call.
    """
    if not ENABLED:
        return ""
    try:
        turns = _turns(messages)
        if len(turns) <= SUMM_KEEP + MIN_OLDER:
            return ""
        older = turns[:-SUMM_KEEP]
        if len(older) < MIN_OLDER:
            return ""

        cache = _load_cache()
        covered = int(cache.get("covered", 0))
        text = cache.get("text", "") or ""

        stale = (
            not text
            or covered > len(older)                 # new/shorter session → rebuild
            or (len(older) - covered) >= REFRESH_STEP
        )
        if stale and allow_build:
            new = _summarize(older)
            if new:
                text = new
                _save_cache({"covered": len(older), "text": text, "ts": time.time()})

        if not text:
            return ""
        return "\n\n## Conversation so far (summary):\n" + text + "\n"
    except Exception:
        return ""
