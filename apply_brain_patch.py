"""apply_brain_patch.py — one-shot patcher for jarvis.py.

Adds the two integration points for the brain module:
  1. Import of BRAIN + try_handle_brain_command
  2. Command intercept block inside handle_chat()

Idempotent — safe to run more than once. Makes a backup at jarvis.py.bak
before modifying (only on the first run; subsequent runs preserve the
original pristine backup).

Usage from cmd:
    python apply_brain_patch.py
    python apply_brain_patch.py "C:\\Chloe\\jarvis.py"
"""

import sys
import shutil
from pathlib import Path


DEFAULT_PATH = r"C:\Chloe\jarvis.py"

IMPORT_ANCHOR = "from groq import AsyncGroq, Groq"
IMPORT_LINE = "from brain_wiring import BRAIN, try_handle_brain_command"

# End of the _try_handle_remember block. Brain command intercept slots in
# right after this return, before the URL-fetch comment.
INTERCEPT_ANCHOR = '''                    hud_server.broadcast_sync("idle")
            return

    # If the user message contains URLs, fetch them server-side'''

INTERCEPT_REPLACEMENT = '''                    hud_server.broadcast_sync("idle")
            return

    # Brain commands: /ingest, /query, /lint, /fact, /brain
    if messages:
        _last_user = _user_text_from_message(messages[-1]) or ""
        brain_reply = await asyncio.to_thread(try_handle_brain_command, _last_user)
        if brain_reply is not None:
            _push_history("user", _last_user, modality="chat")
            _push_history("assistant", brain_reply, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": brain_reply})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts"):
                hud_server.broadcast_sync("speaking")
                try:
                    await _reply_audio_or_speak(brain_reply, data, label="chat-brain")
                except Exception as e:
                    print(f"[chloe] chat TTS error on brain reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

    # If the user message contains URLs, fetch them server-side'''

INTERCEPT_MARKER = "Brain commands: /ingest, /query, /lint"


def patch(path: Path) -> int:
    if not path.exists():
        print(f"[ERROR] file not found: {path}")
        return 1

    original_text = path.read_text(encoding='utf-8')
    text = original_text

    # ── Backup (only on first run; never overwrite the pristine copy) ──
    backup = path.with_suffix(path.suffix + '.bak')
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[ok]   backup created at {backup.name}")
    else:
        print(f"[skip] backup already exists at {backup.name}")

    # ── Edit 1: import line ──
    if IMPORT_LINE in text:
        print(f"[skip] import line already present")
    else:
        if IMPORT_ANCHOR not in text:
            print(f"[ERROR] could not find groq import anchor:")
            print(f"        '{IMPORT_ANCHOR}'")
            print(f"        Add the import manually near the top:")
            print(f"        {IMPORT_LINE}")
            return 1
        text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + IMPORT_LINE, 1)
        print(f"[ok]   added import line")

    # ── Edit 2: command intercept ──
    if INTERCEPT_MARKER in text:
        print(f"[skip] command intercept block already present")
    else:
        count = text.count(INTERCEPT_ANCHOR)
        if count == 0:
            print(f"[ERROR] could not find intercept anchor (end of _try_handle_remember block)")
            print(f"        Add the brain command block manually — see BRAIN_WIRING.md")
            return 1
        if count > 1:
            print(f"[ERROR] intercept anchor matches {count} times — too ambiguous to patch safely")
            return 1
        text = text.replace(INTERCEPT_ANCHOR, INTERCEPT_REPLACEMENT, 1)
        print(f"[ok]   added brain command intercept block in handle_chat()")

    if text == original_text:
        print(f"\n[done] no changes needed — jarvis.py is already patched.")
        return 0

    path.write_text(text, encoding='utf-8')
    print(f"\n[done] {path.name} patched.")
    print(f"       Restart Chloe and type /brain in chat to verify.")
    return 0


if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
    sys.exit(patch(target))
