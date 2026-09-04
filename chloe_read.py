"""LLM read-pass for Chloe — Phase 4C (deferred half).

A small, fast structured read of Ed's latest message: mood / intent / subtext /
engagement / certainty as JSON. This is the part heuristics can't do — sarcasm,
an "I'm fine" that isn't, a flat reply to something he usually engages with.

Cost discipline:
- Gated to non-trivial turns (>= CHLOE_READ_MIN_CHARS).
- Tight timeout; small fast model (llama3.2:3b); num_predict 80.
- Designed to run in `asyncio.to_thread` IN PARALLEL with the recall/wiki
  lookups that already happen, so it adds ~no wall-clock latency.
- Returns {} on anything — disabled, short turn, timeout, bad JSON — so callers
  fall straight back to the heuristic read in chloe_dialogue_state.
- Default ON; `CHLOE_LLM_READ=0` turns it off entirely.

Public API:
    llm_read(messages) -> dict   # {} or {mood,intent,subtext,engagement,certainty}
"""

from __future__ import annotations

import json
import os

import requests

try:
    from chloe_lock import locked
except Exception:  # pragma: no cover - lock is best-effort
    import contextlib

    @contextlib.contextmanager
    def locked(*_a, **_k):
        yield

MODEL = os.environ.get("CHLOE_READ_MODEL", "llama3.2:3b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
TIMEOUT = float(os.environ.get("CHLOE_READ_TIMEOUT", "4"))
ENABLED = os.environ.get("CHLOE_LLM_READ", "1") != "0"
MIN_CHARS = int(os.environ.get("CHLOE_READ_MIN_CHARS", "45"))
# Cadence governor: a 3B doing structured ToM JSON on EVERY turn is mostly
# theater and adds round-trip + parse latency you feel in voice. Run it on every
# Nth substantial turn instead; heuristics in chloe_dialogue_state carry the
# turns in between. CHLOE_READ_EVERY=1 restores the per-turn behavior.
READ_EVERY = max(1, int(os.environ.get("CHLOE_READ_EVERY", "3")))


def _gate_path() -> str:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    return os.path.join(root, "raw", "read_gate.json")


def _gate_should_run() -> bool:
    """Advance the substantial-turn counter and return True on the 1st turn and
    every READ_EVERY-th turn after. Persisted + locked so chat and voice share
    one cadence. Degrades to True (run) on any failure."""
    if READ_EVERY <= 1:
        return True
    try:
        with locked("read_gate"):
            p = _gate_path()
            n = 0
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    n = int(json.load(fh).get("count", 0))
            except Exception:
                n = 0
            n += 1
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                tmp = p + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"count": n}, fh)
                os.replace(tmp, p)
            except Exception:
                pass
            return (n - 1) % READ_EVERY == 0
    except Exception:
        return True

_MOODS = {"frustrated", "sad", "tired", "excited", "happy", "tense", "neutral"}
_INTENTS = {"vent", "decide", "task", "chat", "test", "smalltalk"}


def _last_user(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


def llm_read(messages) -> dict:
    """Structured read of the latest user turn. {} unless enabled, the turn is
    substantial, and the model returns clean JSON. Never raises."""
    if not ENABLED:
        return {}
    try:
        ut = _last_user(messages)
        if len(ut.strip()) < MIN_CHARS:
            return {}
        # Cadence gate: only spend the 3B call on every Nth substantial turn.
        if not _gate_should_run():
            return {}
        prompt = (
            "Read this message from Ed to his AI companion Chloe. Output ONLY a "
            'JSON object: {"mood":"frustrated|sad|tired|excited|happy|tense|'
            'neutral","intent":"vent|decide|task|chat|test|smalltalk","subtext":'
            '"<short note or empty>","engagement":"high|med|low","certainty":'
            "0.0-1.0}. Judge real tone and subtext — sarcasm, an \"I'm fine\" that "
            "isn't, enthusiasm vs flatness. No prose.\n\nMessage: "
            + ut[:600] + "\n\nJSON:"
        )
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"num_predict": 80, "temperature": 0.1}},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        d = json.loads((r.json().get("response") or "").strip())
        if not isinstance(d, dict):
            return {}
        out = {}
        if d.get("mood") in _MOODS:
            out["mood"] = d["mood"]
        if d.get("intent") in _INTENTS:
            out["intent"] = d["intent"]
        if isinstance(d.get("subtext"), str) and d["subtext"].strip():
            out["subtext"] = d["subtext"].strip()[:120]
        if d.get("engagement") in ("high", "med", "low"):
            out["engagement"] = d["engagement"]
        try:
            out["certainty"] = max(0.0, min(1.0, float(d.get("certainty", 0))))
        except Exception:
            out["certainty"] = 0.0
        return out
    except Exception:
        return {}
