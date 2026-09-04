"""Splice script — bug fix from 2026-05-12 meta-review: ack-gate.

Closes the "Thank you" → grep_source / "So," → _speak function false-fire
class of bug. When the user emits a trivial ≤3-token acknowledgement,
short-circuit the LLM entirely and emit a persona-shaped reply.

Three edits to jarvis.py:
  1. Add `_THANKS_TOKENS`, `_SHORT_ACK_TOKENS`, and
     `_maybe_pick_ack_reply()` after `_INTROSPECTION_KEYWORDS`.
  2. Insert ack-gate in `handle_chat` before the lights handler.
  3. Insert ack-gate in `_ask_groq` (voice) after _push_history, before
     the forced-Brave check.

Per chloe_editing_jarvis_py.md: backup, ast.parse, tail-diff,
line-delta bound. Auto-restore on fail.
"""

from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "jarvis.py"
STAMP = datetime.now().strftime("%Y-%m-%d-ackgate")
BACKUP = ROOT / f"jarvis.py.bak.{STAMP}"


# ── Edit 1: ack-set + helper inserted after _INTROSPECTION_KEYWORDS ─────────
HELPER_BLOCK = '''
# ─── Ack-gate (2026-05-12 meta-review fix) ─────────────────────────────────
# Bare "thanks" / "thank you" / "so," / "ok" etc. were false-firing the
# introspection/grep_source routing path (3 transcripts in the 5/12 weekly
# review). Short-circuit the LLM on ≤3-token utterances that match one of
# the ack sets — emit a persona-shaped reply directly, never reach Ollama.
_THANKS_TOKENS = frozenset({
    "thanks", "thank you", "thank ya", "thanks ed", "thanks chloe",
    "appreciate it", "appreciated", "ty", "tysm", "thx",
})
_SHORT_ACK_TOKENS = frozenset({
    "ok", "okay", "k", "kk", "cool", "nice", "alright", "right",
    "sure", "fine", "yeah", "yep", "nope", "yes", "no", "got it",
    "so", "hmm", "uh", "um", "well", "mm", "mhm",
})


def _maybe_pick_ack_reply(text: str) -> str | None:
    """Return a persona-shaped ack reply if `text` is a trivial
    acknowledgement, else None.

    Triggers on ≤3-token utterances matching one of the ack sets. For
    thanks-shaped acks, picks from the chloe_about.md "When thanked"
    response pool. For short acks (ok/so/hmm/etc), picks a low-content
    acknowledgement to keep the conversational rhythm.

    Returns None for anything else — caller falls through to normal LLM
    routing.
    """
    if not text:
        return None
    t = text.lower().strip().rstrip("?!.,;:")
    if len(t.split()) > 3:
        return None
    if t in _THANKS_TOKENS:
        return random.choice((
            "anytime.", "of course.", "happy to.",
            "you got it.", "always.",
        ))
    if t in _SHORT_ACK_TOKENS:
        return random.choice(("mhm.", "yeah.", "got it.", "right."))
    return None

'''

HELPER_ANCHOR_OLD = '''    "in jarvis.py", "in hud_server", "in chloe_memory",
    "in chloe-mobile", "in start_jarvis",
)


# Phrases the fast model emits when it\'s stalling on a question that really'''

HELPER_ANCHOR_NEW = '''    "in jarvis.py", "in hud_server", "in chloe_memory",
    "in chloe-mobile", "in start_jarvis",
)

''' + HELPER_BLOCK + '''
# Phrases the fast model emits when it\'s stalling on a question that really'''


# ── Edit 2: chat handler ack-gate before lights ─────────────────────────────
CHAT_OLD = '''    # Lights: /lights status and natural-language ("turn off the bedroom")
    if messages:
        _last_user_l = _user_text_from_message(messages[-1]) or ""
        lights_reply = await asyncio.to_thread(try_handle_lights_command, _last_user_l)'''

CHAT_NEW = '''    # Ack-gate: short-circuit ≤3-token thanks/acknowledgements with a
    # persona-shaped reply. Prevents "Thank you" → grep_source false-fires
    # (2026-05-12 weekly review). No LLM call, no tool routing.
    if messages:
        _last_user_a = _user_text_from_message(messages[-1]) or ""
        _ack_reply = _maybe_pick_ack_reply(_last_user_a)
        if _ack_reply is not None:
            print(f"[chloe] ack-gate fired: {_last_user_a!r} -> "
                  f"{_ack_reply!r}", flush=True)
            _push_history("user", _last_user_a, modality="chat")
            _push_history("assistant", _ack_reply, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": _ack_reply})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts"):
                try:
                    await _reply_audio_or_speak(
                        _ack_reply, data, label="chat-ack")
                except Exception as e:
                    print(f"[chloe] chat TTS error on ack-reply: {e}",
                          flush=True)
                finally:
                    hud_server.broadcast_sync("idle")
            return

    # Lights: /lights status and natural-language ("turn off the bedroom")
    if messages:
        _last_user_l = _user_text_from_message(messages[-1]) or ""
        lights_reply = await asyncio.to_thread(try_handle_lights_command, _last_user_l)'''


# ── Edit 3: voice handler ack-gate before forced-Brave ──────────────────────
VOICE_OLD = '''    _push_history("user", user_text)

    # Forced-Brave route for temporal+result queries (bug #4 fix). Same'''

VOICE_NEW = '''    _push_history("user", user_text)

    # Ack-gate: short-circuit ≤3-token thanks/acknowledgements. Prevents
    # voice false-fires like "Thank you" → grep_source and "So," →
    # _speak function dump (2026-05-12 weekly review).
    _ack_voice_reply = _maybe_pick_ack_reply(user_text)
    if _ack_voice_reply is not None:
        print(f"[voice] ack-gate fired: {user_text!r} -> "
              f"{_ack_voice_reply!r}", flush=True)
        _push_history("assistant", _ack_voice_reply)
        return _ack_voice_reply

    # Forced-Brave route for temporal+result queries (bug #4 fix). Same'''


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


def main() -> None:
    if not TARGET.exists():
        fail(f"target missing: {TARGET}", restore=False)
    shutil.copy2(TARGET, BACKUP)
    print(f"[splice] backup -> {BACKUP.name}")

    src = TARGET.read_text(encoding="utf-8")
    orig_lines = src.count("\n")

    new = src
    new = replace_once(new, HELPER_ANCHOR_OLD, HELPER_ANCHOR_NEW,
                       "helper block")
    new = replace_once(new, CHAT_OLD, CHAT_NEW, "chat ack-gate")
    new = replace_once(new, VOICE_OLD, VOICE_NEW, "voice ack-gate")

    new_lines = new.count("\n")
    delta = new_lines - orig_lines
    if not (70 <= delta <= 140):
        fail(f"line-delta {delta} outside [70, 140]")

    try:
        ast.parse(new)
    except SyntaxError as e:
        fail(f"ast.parse failed: {e}")

    bak_tail = src.splitlines()[-50:]
    new_tail = new.splitlines()[-50:]
    if bak_tail != new_tail:
        fail("tail-diff: last 50 lines diverged")

    TARGET.write_text(new, encoding="utf-8")
    print(f"[splice] OK - {delta} lines added")
    print(f"[splice] verify: after restart, try chat 'thanks' or voice "
          f"'thank you' — should see '[chloe] ack-gate fired' log line "
          f"and get a short persona reply, no grep_source call.")


if __name__ == "__main__":
    main()
