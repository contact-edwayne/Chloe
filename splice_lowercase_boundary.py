"""splice_lowercase_boundary.py - widen _SENT_BOUNDARY_RE so it matches
lowercase sentence starts (Chloe's persona is lowercase, so the old regex
that required [A-Z0-9] after the period never fired and inline TTS
streaming silently fell through to the post-stream chunked path)."""
import ast, io, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JP = HERE / "jarvis.py"
BAK = JP.parent / (JP.name + ".bak.2026-05-14f")

if not BAK.exists():
    shutil.copy2(JP, BAK)
    print(f"backup: {BAK.name}")
else:
    print(f"backup exists, keeping: {BAK.name}")

src = io.open(JP, "r", encoding="utf-8").read()
OLD = "_SENT_BOUNDARY_RE = _re.compile(r'(?<=[.!?])\\s+(?=[A-Z0-9])')"
NEW = "_SENT_BOUNDARY_RE = _re.compile(r'(?<=[.!?])\\s+(?=[A-Za-z0-9])')"
if src.count(OLD) != 1:
    print(f"ABORT: anchor count = {src.count(OLD)} (need 1)")
    sys.exit(1)
new_src = src.replace(OLD, NEW, 1)
ast.parse(new_src)
bsrc = io.open(BAK, "r", encoding="utf-8").read()
if new_src.splitlines()[-6:] != bsrc.splitlines()[-6:]:
    print("ABORT: tail differs"); sys.exit(1)
with io.open(JP, "wb") as f:
    f.write(new_src.encode("utf-8")); f.flush(); os.fsync(f.fileno())
chk = io.open(JP, "r", encoding="utf-8").read()
ast.parse(chk)
if NEW not in chk or OLD in chk:
    print("ABORT: post-write content mismatch"); sys.exit(1)
print("OK")
