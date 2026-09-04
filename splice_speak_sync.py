"""
Lift the greeting sync fix to all voice replies.

Two changes:

  1. _speak_kokoro: broadcast "speaking" at first audio onset (after the
     first sentence is pulled from the queue, right before
     _play_audio_with_barge_in) and "idle" in the finally block.

  2. Remove the 22 early `hud_server.broadcast_sync("speaking")` calls
     that fire BEFORE _speak() / _reply_audio_or_speak() — they cause
     the HUD orb to pulse during the Kokoro synth gap. The chat path
     (reply_audio=True) is already protected by the HUD's
     `expectingAudio` flag; the voice path is not. With this fix the
     state stays "thinking" through synth and transitions to "speaking"
     at audio onset.

  3. Matching `hud_server.broadcast_sync("idle")` lines at the call
     sites are kept — they're idempotent with the engine-level idle
     and provide a safety net if _speak fails entirely.

Edge-tts and ElevenLabs engines still have the original behavior;
since Kokoro is the current default (USE_KOKORO=1, USE_ELEVENLABS=0)
this covers daily-driver use. Backlog item to lift the same pattern
to the other engines.

Run from C:\\Users\\eleew\\Documents\\jarvis\\:
    python splice_speak_sync.py
"""
from __future__ import annotations
import ast
import re
import shutil
from datetime import date
from pathlib import Path


JARVIS = Path(__file__).parent / "jarvis.py"
BACKUP = JARVIS.with_name(f"jarvis.py.bak.{date.today().isoformat()}-speaksync")


# 1) Patch _speak_kokoro's consumer loop to broadcast on first onset + idle in finally.

KOKORO_OLD = '''    _speaking.set()
    _barge_in_request.clear()
    _barge_in_via_wake.clear()
    threading.Thread(target=_barge_in_monitor, daemon=True,
                     name="chloe-barge-in").start()
    try:
        while True:
            item = audio_queue.get()
            if item is SENTINEL:
                break
            samples, sr = item
            try:
                if not _barge_in_request.is_set():
                    completed = _play_audio_with_barge_in(samples, sr)'''


KOKORO_NEW = '''    _speaking.set()
    _barge_in_request.clear()
    _barge_in_via_wake.clear()
    threading.Thread(target=_barge_in_monitor, daemon=True,
                     name="chloe-barge-in").start()
    # Broadcast "speaking" on first audio onset (after first sentence is
    # synthesized + about to play) so the HUD pulse animation lines up
    # with actual sound, not synth-gap silence. Broadcast "idle" in the
    # finally so call sites can drop their pre-_speak() "speaking" call
    # without leaving the HUD stuck.
    _spoke_at_least_once = False
    try:
        while True:
            item = audio_queue.get()
            if item is SENTINEL:
                break
            samples, sr = item
            try:
                if not _barge_in_request.is_set():
                    if not _spoke_at_least_once:
                        try:
                            hud_server.broadcast_sync("speaking")
                        except Exception:
                            pass
                        _spoke_at_least_once = True
                    completed = _play_audio_with_barge_in(samples, sr)'''


# The existing _speak_kokoro `finally:` clause clears _speaking.clear().
# We extend it to broadcast idle on the way out.

KOKORO_FINALLY_OLD = '''    finally:
        _speaking.clear()


# ─── REGISTER + START VOICE LOOP ON IMPORT ───────────────────────────────────'''


KOKORO_FINALLY_NEW = '''    finally:
        _speaking.clear()
        if _spoke_at_least_once:
            try:
                hud_server.broadcast_sync("idle")
            except Exception:
                pass


# ─── REGISTER + START VOICE LOOP ON IMPORT ───────────────────────────────────'''


# 2) Remove the early call-site "speaking" broadcasts. We DO NOT touch the
# matching "idle" calls — they remain as idempotent safety nets.
#
# Two patterns to handle:
#
#   Pattern B (chat path, gated):
#       if not _hud_via_audio:
#           hud_server.broadcast_sync("speaking")
#
#   Pattern A (voice path, bare):
#       hud_server.broadcast_sync("speaking")
#
# B is matched FIRST so its wrapper line is dropped along with the body.
# Otherwise Pattern A would strip the body and leave an empty `if` block.

GATED_SPEAKING_RE = re.compile(
    r'^[ \t]*if not _hud_via_audio:[ \t]*\r?\n'
    r'[ \t]+hud_server\.broadcast_sync\("speaking"\)[ \t]*\r?\n',
    flags=re.MULTILINE,
)

BARE_SPEAKING_RE = re.compile(
    r'^[ \t]*hud_server\.broadcast_sync\("speaking"\)[ \t]*\r?\n',
    flags=re.MULTILINE,
)


def main() -> None:
    src = JARVIS.read_text(encoding="utf-8")

    # Idempotence guard
    if KOKORO_NEW.split('\n', 1)[1] in src and 'hud_server.broadcast_sync("speaking")' not in src:
        print("[splice] already patched — no-op.")
        return

    shutil.copy2(JARVIS, BACKUP)
    print(f"[splice] backup: {BACKUP.name}")

    # Order matters: do removals BEFORE we insert the new broadcast inside
    # _speak_kokoro, otherwise the bare-broadcast regex eats our own insert.

    # Step 1a: drop gated chat-path "speaking" broadcasts (if-wrapper + body)
    gated_matches = GATED_SPEAKING_RE.findall(src)
    gated_removed = len(gated_matches)
    new_src = GATED_SPEAKING_RE.sub('', src)

    # Step 1b: drop bare voice-path "speaking" broadcasts
    bare_matches = BARE_SPEAKING_RE.findall(new_src)
    bare_removed = len(bare_matches)
    new_src = BARE_SPEAKING_RE.sub('', new_src)

    removed = gated_removed + bare_removed

    # Step 2: _speak_kokoro consumer loop
    if KOKORO_OLD not in new_src:
        raise SystemExit("[splice] FAIL — _speak_kokoro consumer block not found.")
    new_src = new_src.replace(KOKORO_OLD, KOKORO_NEW, 1)

    # Step 3: _speak_kokoro finally block
    if KOKORO_FINALLY_OLD not in new_src:
        raise SystemExit("[splice] FAIL — _speak_kokoro finally block not found.")
    new_src = new_src.replace(KOKORO_FINALLY_OLD, KOKORO_FINALLY_NEW, 1)

    # Syntax check
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"[splice] FAIL — ast.parse: {e}")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    # Tail integrity
    old_tail = src.splitlines()[-20:]
    new_tail = new_src.splitlines()[-20:]
    if old_tail != new_tail:
        print("[splice] FAIL — tail diverged; restoring backup")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    # Sanity on what got removed
    expected_remove_min = 18
    expected_remove_max = 28
    if not (expected_remove_min <= removed <= expected_remove_max):
        print(f"[splice] FAIL — removed {removed} early broadcasts, expected "
              f"{expected_remove_min}-{expected_remove_max}; restoring backup")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    delta = len(new_src.splitlines()) - len(src.splitlines())
    JARVIS.write_text(new_src, encoding="utf-8")
    print(f"[splice] OK — removed {gated_removed} gated + {bare_removed} bare = "
          f"{removed} early speaking broadcasts")
    print(f"[splice] OK — patched _speak_kokoro consumer + finally")
    print(f"[splice] OK — wrote {JARVIS.name} ({delta:+d} lines)")
    print("[splice] restart Chloe to pick up the change.")


if __name__ == "__main__":
    main()
