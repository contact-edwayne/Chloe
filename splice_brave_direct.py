"""Splice script — bug #4 fix: force Brave on temporal+result queries.

Closes the hedge-detection blind spot where confident confabulations slip
through. The Eurovision smoke test exposed this: the model answered
"Norway/Aurora" (wrong) instead of hedging, so _looks_like_hedge never
fired and Brave fallback never escalated.

Adds:
  1. _RESULT_SEEKING_WORDS tuple + _needs_brave_direct() classifier near
     _needs_realtime (jarvis.py line ~935).
  2. Forced-Brave check in handle_chat after the /search block, before
     URL fetch (line ~1470).
  3. Forced-Brave check in _ask_groq (voice handler) after _push_history
     (line ~4770).

When _needs_brave_direct fires, bypass the LLM entirely and route through
Brave + Groq synthesis. On empty Brave return, fall through to the normal
LLM path so the user still gets some answer.
"""

from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "jarvis.py"
STAMP = datetime.now().strftime("%Y-%m-%d-bravedirect")
BACKUP = ROOT / f"jarvis.py.bak.{STAMP}"


# ── Edit 1: classifier + word list inserted before def _needs_realtime ──────
CLASSIFIER_BLOCK = '''# Words/nouns that ask for a specific recent factual outcome. Combined
# with a temporal marker (current year, "this year", "recent", etc.) they
# signal "this question needs ground-truth web data — the model will
# happily confabulate if asked from training alone." Bug #4 fix
# (2026-05-17): the Eurovision smoke test showed confident hallucination
# slipping past _looks_like_hedge because the model didn't hedge — it
# just lied with confidence. _needs_brave_direct catches this class
# pre-emptively and forces a Brave round-trip.
_RESULT_SEEKING_WORDS = (
    "winner", "won", "winning", "wins", "champion", "champions",
    "champ", "title", "trophy", "crown", "victor",
    "result", "results", "outcome", "final score",
    "finalist", "finalists", "runner-up", "second place", "third place",
    "elected", "appointed", "nominated", "named", "announced",
    "released", "launched", "shipped", "debuted", "unveiled", "premiered",
    "awarded", "earned", "took home", "took the",
)


def _needs_brave_direct(text: str) -> bool:
    """Return True if the query asks for a specific recent factual outcome
    that the model would confidently confabulate from training data.

    Triggers when BOTH:
      1. Query references the current/next/last year (literal 4-digit) or
         contains a recency marker (`this year`, `current`, `recent`,
         `latest`, `last year`, `this season`).
      2. Query contains a result-seeking word (`winner`, `won`, `champion`,
         `finalist`, `elected`, `released`, `awarded`, etc.).

    When True, the chat and voice handlers bypass the LLM and route
    directly to _brave_fallback_search / _brave_voice_synth. Closes the
    hedge-detection blind spot for confident confabulations (bug #4).
    """
    if not text:
        return False
    import datetime as _d
    t = text.lower()
    _yr = _d.datetime.now().year
    year_literals = (str(_yr - 1), str(_yr), str(_yr + 1))
    has_temporal = (
        any(y in t for y in year_literals)
        or "this year" in t
        or "last year" in t
        or "current" in t
        or "recent" in t
        or "latest" in t
        or "this season" in t
        or "this month" in t
    )
    has_result = any(w in t for w in _RESULT_SEEKING_WORDS)
    return has_temporal and has_result


'''

CLASSIFIER_ANCHOR = "def _needs_realtime(text: str) -> bool:"


# ── Edit 2: forced-Brave check in handle_chat ──────────────────────────────
CHAT_OLD = """    # If the user message contains URLs, fetch them server-side and inject the
    # readable text into the message before sending to Groq. Browser CORS makes
    # client-side fetch unreliable, so it has to live here."""

CHAT_NEW = """    # Force-route to Brave when the query asks for a recent factual outcome
    # (winner/result/election/etc.) tied to a current/recent year. The model
    # will confidently confabulate these from training (bug #4: the
    # Eurovision smoke test caught this) and _looks_like_hedge never fires
    # because the reply doesn't hedge — it just lies with confidence.
    # Pre-empt by routing directly to Brave. Reuses _brave_fallback_search
    # which already streams start/delta/sources/done and auto-persists via
    # _persist_brave_to_wiki.
    if messages:
        _bd_user_q = _last_user_text(messages)
        if _bd_user_q and _needs_brave_direct(_bd_user_q):
            print(f"[chloe] forced Brave route - temporal+result query: "
                  f"{_bd_user_q!r}", flush=True)
            _push_history("user", _bd_user_q, modality="chat")
            _bd_brave_reply = await _brave_fallback_search(
                websocket, _bd_user_q, data
            )
            if _bd_brave_reply.strip():
                _push_history("assistant", _bd_brave_reply, modality="chat")
                if not data.get("no_tts"):
                    _bd_tts = _re.sub(r"\\[\\d+\\]", "",
                                      _bd_brave_reply).strip()
                    try:
                        await _reply_audio_or_speak(
                            _bd_tts, data, label="chat-brave-direct"
                        )
                    except Exception as e:
                        print(f"[chloe] chat TTS error on Brave-direct: {e}",
                              flush=True)
                        hud_server.broadcast_sync("idle")
                return
            print("[chloe] Brave-direct returned empty - falling through "
                  "to LLM", flush=True)

    # If the user message contains URLs, fetch them server-side and inject the
    # readable text into the message before sending to Groq. Browser CORS makes
    # client-side fetch unreliable, so it has to live here."""


# ── Edit 3: forced-Brave check in _ask_groq (voice) ────────────────────────
VOICE_OLD = """    if not _sync_groq and not _ollama_available():
        return ""
    _push_history("user", user_text)

    route = _pick_route(user_text)"""

VOICE_NEW = """    if not _sync_groq and not _ollama_available():
        return ""
    _push_history("user", user_text)

    # Forced-Brave route for temporal+result queries (bug #4 fix). Same
    # logic as the chat path: bypass the LLM whose confident confabulation
    # would slip past _looks_like_hedge.
    if _needs_brave_direct(user_text):
        print(f"[voice] forced Brave route - temporal+result query: "
              f"{user_text!r}", flush=True)
        _bd_brave_reply = _brave_voice_synth(user_text)
        if _bd_brave_reply:
            print(f"[voice] Brave-direct succeeded "
                  f"({len(_bd_brave_reply)} chars)", flush=True)
            _push_history("assistant", _bd_brave_reply)
            return _bd_brave_reply
        print("[voice] Brave-direct returned empty - falling through "
              "to normal route", flush=True)

    route = _pick_route(user_text)"""


def fail(msg: str, restore: bool = True) -> None:
    print(f"[splice] FAIL: {msg}", file=sys.stderr)
    if restore and BACKUP.exists():
        shutil.copy2(BACKUP, TARGET)
        print(f"[splice] restored {TARGET} from {BACKUP.name}",
              file=sys.stderr)
    sys.exit(1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        fail(f"{label}: anchor not found")
    if count > 1:
        fail(f"{label}: anchor matched {count} times (expected 1)")
    return text.replace(old, new, 1)


def insert_before(text: str, anchor: str, payload: str, label: str) -> str:
    count = text.count(anchor)
    if count == 0:
        fail(f"{label}: anchor not found")
    if count > 1:
        fail(f"{label}: anchor matched {count} times (expected 1)")
    idx = text.find(anchor)
    return text[:idx] + payload + text[idx:]


def main() -> None:
    if not TARGET.exists():
        fail(f"target missing: {TARGET}", restore=False)
    shutil.copy2(TARGET, BACKUP)
    print(f"[splice] backup -> {BACKUP.name}")

    src = TARGET.read_text(encoding="utf-8")
    orig_lines = src.count("\n")

    new = src
    new = insert_before(new, CLASSIFIER_ANCHOR, CLASSIFIER_BLOCK,
                        "classifier block")
    new = replace_once(new, CHAT_OLD, CHAT_NEW, "chat forced-Brave check")
    new = replace_once(new, VOICE_OLD, VOICE_NEW, "voice forced-Brave check")

    new_lines = new.count("\n")
    delta = new_lines - orig_lines
    if not (80 <= delta <= 160):
        fail(f"line-delta {delta} outside [80, 160]")

    try:
        ast.parse(new)
    except SyntaxError as e:
        fail(f"ast.parse failed: {e}")

    # Tail-diff: edits are far from EOF (line ~4770 of ~5000+), tail
    # should be untouched.
    bak_tail = src.splitlines()[-50:]
    new_tail = new.splitlines()[-50:]
    if bak_tail != new_tail:
        fail("tail-diff: last 50 lines diverged (edits drifted to EOF)")

    TARGET.write_text(new, encoding="utf-8")
    print(f"[splice] OK - {delta} lines added")
    print(f"[splice] verify: restart Chloe and ask "
          f"'who won the 2026 Eurovision' — should see "
          f"'[chloe] forced Brave route' in logs and a "
          f"citation-shaped answer.")


if __name__ == "__main__":
    main()
