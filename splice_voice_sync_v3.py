"""splice_voice_sync_v3.py - serialize the inline TTS synth so concurrent
Kokoro calls can't corrupt later chunks. Surgical edit of three spots in
the inline-TTS block from v2.

Run (Chloe stopped):
    stop_chloe.bat
    python splice_voice_sync_v3.py
    start_chloe.vbs

Backup: jarvis.py.bak.2026-05-14e
"""
import ast
import io
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JP = HERE / "jarvis.py"
BAK = JP.parent / (JP.name + ".bak.2026-05-14e")


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write_fsync(p, text):
    with io.open(p, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def fail(m):
    print("ABORT:", m); sys.exit(1)


print("Making backup...")
if not BAK.exists():
    shutil.copy2(JP, BAK)
    print(f"  backup: {BAK.name}  ({BAK.stat().st_size} bytes)")
else:
    print(f"  backup exists, keeping: {BAK.name}")

src = read(JP)
bsrc = read(BAK)
if "_emit_tts_chunk" not in src:
    fail("jarvis.py: inline-TTS block (v2) not present; nothing to upgrade")
if "tts_tasks" not in src:
    fail("jarvis.py: v2 markers missing (tts_tasks)")
if "await asyncio.gather(*tts_tasks" not in src:
    fail("jarvis.py: v2 final-gather block already removed?")

# --- replacement 1: drop the tts_tasks list (serial synth doesn't need it) ---
OLD_1 = "        tts_tasks: list = []\n        inline_tts_fired = False"
NEW_1 = "        inline_tts_fired = False"
if src.count(OLD_1) != 1:
    fail(f"replacement 1 anchor count = {src.count(OLD_1)} (need 1)")
src = src.replace(OLD_1, NEW_1, 1)

# --- replacement 2: inline create_task -> await (serialize mid-stream synth) ---
OLD_2 = """                        tts_tasks.append(asyncio.create_task(
                            _emit_tts_chunk(sent_text, is_final=False)))
                        inline_tts_fired = True"""
NEW_2 = """                        await _emit_tts_chunk(sent_text, is_final=False)
                        inline_tts_fired = True"""
if src.count(OLD_2) != 1:
    fail(f"replacement 2 anchor count = {src.count(OLD_2)} (need 1)")
src = src.replace(OLD_2, NEW_2, 1)

# --- replacement 3: final-flush block -> single await ---
OLD_3 = """            if wants_audio and inline_tts_fired:
                tts_tasks.append(asyncio.create_task(
                    _emit_tts_chunk(tts_buf, is_final=True)))
                if tts_tasks:
                    await asyncio.gather(*tts_tasks, return_exceptions=True)"""
NEW_3 = """            if wants_audio and inline_tts_fired:
                await _emit_tts_chunk(tts_buf, is_final=True)"""
if src.count(OLD_3) != 1:
    fail(f"replacement 3 anchor count = {src.count(OLD_3)} (need 1)")
src = src.replace(OLD_3, NEW_3, 1)

# Sanity: tts_tasks should no longer appear at all
if "tts_tasks" in src:
    fail(f"tts_tasks still present after replacements ({src.count('tts_tasks')} occurrences)")

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"jarvis.py would not parse: {e}")
if src.splitlines()[-6:] != bsrc.splitlines()[-6:]:
    fail("jarvis.py tail differs from backup pre-write")

write_fsync(JP, src)
chk = read(JP)
ast.parse(chk)
if chk.splitlines()[-6:] != bsrc.splitlines()[-6:]:
    fail("jarvis.py TAIL MISMATCH after write")
if "tts_tasks" in chk:
    fail("tts_tasks still present after write")
for needle in ("await _emit_tts_chunk(sent_text, is_final=False)",
               "await _emit_tts_chunk(tts_buf, is_final=True)"):
    if needle not in chk:
        fail(f"post-write missing: {needle!r}")
print(f"OK  jarvis.py  {len(bsrc)} -> {len(chk)} chars")
print()
print("VOICE SYNC v3 (serial synth) SPLICE OK")
