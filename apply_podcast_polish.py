"""apply_podcast_polish.py — one-shot patcher to update the brain command
intercept in jarvis.py so it supports dict-shape returns from try_handle_brain_command.

This lets /podcast return {text, no_tts: True} so the chat reply text is
shown in chat but NOT also read aloud — preventing the status message
from playing over the actual podcast audio.

Idempotent. Safe to run more than once.

Usage:
    python apply_podcast_polish.py
    python apply_podcast_polish.py "C:\\Users\\eleew\\Documents\\jarvis\\jarvis.py"
"""

import sys
from pathlib import Path

DEFAULT_PATH = r"C:\Users\eleew\Documents\jarvis\jarvis.py"

OLD = '''    # Brain commands: /ingest, /query, /lint, /fact, /brain
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
            return'''

NEW = '''    # Brain commands: /ingest, /query, /lint, /fact, /brain, /podcast, /add
    if messages:
        _last_user = _user_text_from_message(messages[-1]) or ""
        brain_reply = await asyncio.to_thread(try_handle_brain_command, _last_user)
        if brain_reply is not None:
            # brain_reply may be a string (normal) or a dict {text, no_tts}
            # for commands that produce their own audio (e.g. /podcast plays
            # a WAV via os.startfile and shouldn\\'t also TTS the status text).
            if isinstance(brain_reply, dict):
                _brain_text = brain_reply.get("text", "")
                _brain_silent = bool(brain_reply.get("no_tts"))
            else:
                _brain_text = brain_reply
                _brain_silent = False
            _push_history("user", _last_user, modality="chat")
            _push_history("assistant", _brain_text, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": _brain_text})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts") and not _brain_silent:
                hud_server.broadcast_sync("speaking")
                try:
                    await _reply_audio_or_speak(_brain_text, data, label="chat-brain")
                except Exception as e:
                    print(f"[chloe] chat TTS error on brain reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return'''

ALREADY_MARKER = "_brain_silent = bool(brain_reply.get"


def patch(path: Path) -> int:
    if not path.exists():
        print(f"[ERROR] file not found: {path}")
        return 1
    original = path.read_text(encoding='utf-8')
    if ALREADY_MARKER in original:
        print(f"[skip] intercept already supports dict-shape returns")
        return 0
    if OLD not in original:
        print("[ERROR] couldn't find original brain intercept block to update.")
        print("        File may have been edited; patch manually instead.")
        return 1
    backup = path.with_suffix(path.suffix + '.bak2')
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
        print(f"[ok]   secondary backup created at {backup.name}")
    new_text = original.replace(OLD, NEW, 1)
    path.write_text(new_text, encoding='utf-8')
    print(f"[ok]   intercept updated for dict-shape support")
    print(f"[done] {path.name} patched. Restart Chloe.")
    return 0


if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
    sys.exit(patch(target))
