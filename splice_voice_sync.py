"""splice_voice_sync.py - wire inline streaming TTS into the use_ollama
branch of handle_chat so spoken audio tracks the typed text instead of
waiting for the whole reply.

Run (Chloe stopped):
    stop_chloe.bat
    venv\\Scripts\\activate
    python splice_voice_sync.py
    start_chloe.vbs

Changes:
  - jarvis.py: replace the use_ollama branch with the inline-TTS version
    in lat_useollama_v2.txt.
  - jarvis.py: in the post-stream TTS call, skip _reply_audio_or_speak
    when inline TTS already fired (so we don't double-speak).

Same anchored-edit + ast.parse + tail-diff verification as splice_latency.
Aborts (no write) on any mismatch. Backup at jarvis.py.bak.2026-05-14d.
"""
import ast
import io
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JP = HERE / "jarvis.py"
BAK_SUFFIX = ".bak.2026-05-14d"


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write_fsync(p, text):
    with io.open(p, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def fail(m):
    print("ABORT:", m)
    sys.exit(1)


def find_one(lines, pred, label):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    if len(hits) != 1:
        fail(f"{label}: matched {len(hits)} lines (need exactly 1)")
    return hits[0]


def backup(path):
    bak = path.parent / (path.name + BAK_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup: {bak.name}  ({bak.stat().st_size} bytes)")
    else:
        print(f"  backup exists, keeping: {bak.name}")
    return bak


if not JP.exists():
    fail(f"missing: {JP}")
if not (HERE / "lat_useollama_v2.txt").exists():
    fail("missing staged file: lat_useollama_v2.txt")

print("Making backup...")
JBAK = backup(JP)
print()

NEW_USEOLLAMA = read(HERE / "lat_useollama_v2.txt").rstrip("\n").split("\n")

jsrc = read(JP)
jbak = read(JBAK)
if "\r\n" in jsrc:
    fail("jarvis.py has CRLF (splice assumes LF)")
if "_emit_tts_chunk" in jsrc or "inline_tts_fired" in jsrc:
    fail("jarvis.py already patched for voice sync - refusing to double-apply")
if "_ollama_chat_stream" not in jsrc:
    fail("jarvis.py: expected latency patch (_ollama_chat_stream) not found")

lines = jsrc.split("\n")

# ---- edit 1: replace the use_ollama branch with the inline-TTS version ----
i_start = find_one(lines, lambda l: l.strip() == "if use_ollama:", "edit1-start")
# find the end: the `await _ws_send(websocket, {"type": "done"})` inside this branch
# (the FIRST one after `if use_ollama:`, scanning forward)
i_end = None
for j in range(i_start, min(i_start + 60, len(lines))):
    if lines[j].strip() == 'await _ws_send(websocket, {"type": "done"})':
        i_end = j
        break
if i_end is None:
    fail("edit1: end anchor (done) not found within 60 lines of 'if use_ollama:'")
oldblock = "\n".join(lines[i_start:i_end + 1])
for marker in ("async for _delta in _ollama_chat_stream",
               "_OllamaToolCallNeeded",
               "use_ollama = False",
               "Ollama empty — falling back"):
    if marker not in oldblock:
        fail(f"edit1: expected marker not in old block: {marker!r}")
lines[i_start:i_end + 1] = NEW_USEOLLAMA

# ---- edit 2: gate the post-stream TTS call on inline_tts_fired ----
# anchor: the line that calls _reply_audio_or_speak with label="chat-ollama"
i_t = find_one(
    lines,
    lambda l: l.strip() == 'await _reply_audio_or_speak(full_reply, data, label="chat-ollama")',
    "edit2-anchor")
ind = lines[i_t][:len(lines[i_t]) - len(lines[i_t].lstrip())]
lines[i_t:i_t + 1] = [
    ind + "if not inline_tts_fired:",
    ind + "    await _reply_audio_or_speak(full_reply, data, label=\"chat-ollama\")",
]

# ---- validate + write ----
new_j = "\n".join(lines)
try:
    ast.parse(new_j)
except SyntaxError as e:
    fail(f"jarvis.py would not parse: {e}")
# tail unchanged (we don't touch the file end)
if new_j.splitlines()[-6:] != jbak.splitlines()[-6:]:
    fail("jarvis.py tail differs from backup pre-write")

write_fsync(JP, new_j)
jchk = read(JP)
ast.parse(jchk)
if jchk.splitlines()[-6:] != jbak.splitlines()[-6:]:
    fail("jarvis.py TAIL MISMATCH after write")
for needle in ("_emit_tts_chunk", "inline_tts_fired", "tts_audio_chunk",
               "if not inline_tts_fired:",
               "tts_tasks.append(asyncio.create_task("):
    if needle not in jchk:
        fail(f"jarvis.py post-write missing {needle!r}")
print(f"OK  jarvis.py  {len(jbak)} -> {len(jchk)} chars "
      f"({len(jbak.splitlines())} -> {len(jchk.splitlines())} lines)")
print()
print("VOICE SYNC SPLICE OK")
print()
print("Restart Chloe (start_chloe.vbs), ask a question, voice should now")
print("follow the typed text within ~1-2s instead of waiting for the whole")
print("reply. Log lines to watch in logs\\backend.log:")
print("    [chloe] chat-ollama inline: chunk N (M bytes, wav)")
print("    [chloe] chat-ollama inline: chunk N (M bytes, wav) [final]")
