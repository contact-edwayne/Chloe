"""splice_latency.py - apply the latency patch across jarvis.py,
chloe_memory.py, and wiki_embedding.py.

How to run (Chloe stopped):
    stop_chloe.bat
    venv\\Scripts\\activate
    python splice_latency.py
    start_chloe.vbs

What it does:
  - Makes .bak.2026-05-14c backups for all 3 files
  - Splices in (anchored, with assertions):
      jarvis.py:
        1. OLLAMA_KEEP_ALIVE constant
        2. keep_alive in the _ollama_chat payload
        3. _OllamaToolCallNeeded + _ollama_chat_stream + _warm_ollama_models
        4. real streaming in the handle_chat use_ollama branch
        5. concurrent (non-blocking) recall + wiki lookups
        6. _warm_ollama_models kicked off at boot
      chloe_memory.py / wiki_embedding.py:
        _EMBED_KEEP_ALIVE constant + keep_alive in the _embed payload
  - ast.parse the new source before writing
  - fsync the write
  - re-read, ast.parse again, tail-diff vs backup
  - Aborts (no write to that file) on any mismatch

This script lives next to jarvis.py. After a successful run you can
delete it and the three lat_jarvis_*.txt files - they're not used at
runtime by Chloe.
"""
import ast
import io
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JP = HERE / "jarvis.py"
CMP = HERE / "chloe_memory.py"
WEP = HERE / "wiki_embedding.py"
BAK_SUFFIX = ".bak.2026-05-14c"


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


def indent_of(line):
    return line[:len(line) - len(line.lstrip())]


def backup(path):
    bak = path.parent / (path.name + BAK_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup: {bak.name}  ({bak.stat().st_size} bytes)")
    else:
        print(f"  backup exists, keeping: {bak.name}")
    return bak


# ----- preflight: are all 3 source files there? -----
for p in (JP, CMP, WEP):
    if not p.exists():
        fail(f"missing: {p}")
for staged in ("lat_jarvis_funcs.txt", "lat_jarvis_useollama.txt",
               "lat_jarvis_recallwiki.txt"):
    if not (HERE / staged).exists():
        fail(f"missing staged file: {HERE / staged}")

print("Making backups...")
JBAK = backup(JP)
CMP_BAK = backup(CMP)
WEP_BAK = backup(WEP)
print()

FUNCS = read(HERE / "lat_jarvis_funcs.txt").rstrip("\n").split("\n")
USEOLLAMA = read(HERE / "lat_jarvis_useollama.txt").rstrip("\n").split("\n")
RECALLWIKI = read(HERE / "lat_jarvis_recallwiki.txt").rstrip("\n").split("\n")


# ============================ jarvis.py =============================
jsrc = read(JP)
jbak = read(JBAK)
if "\r\n" in jsrc:
    fail("jarvis.py has CRLF (splice assumes LF)")
if "OLLAMA_KEEP_ALIVE" in jsrc or "_ollama_chat_stream" in jsrc:
    fail("jarvis.py already patched - refusing to double-apply")
lines = jsrc.split("\n")

# edit 1: OLLAMA_KEEP_ALIVE constant
i = find_one(lines, lambda l: l.startswith("OLLAMA_FALLBACK_ENABLED"), "edit1")
lines[i + 1:i + 1] = [
    "# How long Ollama keeps a model resident after a request. Ollama's default",
    "# (5m) lets the 19GB qwen2.5:32b get evicted between turns, so the next",
    '# question pays a full cold reload. "30m" covers normal use; set',
    "# OLLAMA_KEEP_ALIVE=-1 to pin the model in VRAM permanently.",
    'OLLAMA_KEEP_ALIVE       = os.environ.get("OLLAMA_KEEP_ALIVE", "30m").strip()',
]

# edit 2: keep_alive in _ollama_chat payload
i = find_one(lines, lambda l: l.strip() == '"stream":   False,', "edit2")
ind = indent_of(lines[i])
lines[i + 1:i + 1] = [ind + '"keep_alive": OLLAMA_KEEP_ALIVE,']

# edit 3: insert streaming + warm-up funcs before def _ollama_chat
i = find_one(lines, lambda l: l.startswith("def _ollama_chat("), "edit3")
lines[i:i] = FUNCS + ["", ""]

# edit 4: rewrite use_ollama branch with real streaming
i_start = find_one(lines, lambda l: l.strip() == "if use_ollama:", "edit4-start")
i_end = None
for j in range(i_start, min(i_start + 45, len(lines))):
    if lines[j].strip() == 'await _ws_send(websocket, {"type": "done"})':
        i_end = j
        break
if i_end is None:
    fail("edit4: end anchor not found within 45 lines of 'if use_ollama:'")
block = "\n".join(lines[i_start:i_end + 1])
for marker in ("for word in ollama_reply.split():",
               "_ollama_chat, ollama_msgs, max_tok",
               "traceback.print_exc()"):
    if marker not in block:
        fail(f"edit4: expected marker not in old block: {marker!r}")
lines[i_start:i_end + 1] = USEOLLAMA

# edit 5: concurrent recall + wiki lookups
i_u = find_one(
    lines,
    lambda l: l.strip() == "user_text_for_recall = _last_user_text(messages)",
    "edit5-anchor")
if lines[i_u - 1].strip() != 'recall_block = ""':
    fail(f"edit5: line above anchor is {lines[i_u - 1]!r}")
i_w = None
for j in range(i_u, min(i_u + 30, len(lines))):
    if lines[j].strip() == 'print(f"[wiki] chat inject failed: {e}", flush=True)':
        i_w = j
        break
if i_w is None:
    fail("edit5: wiki-fail end anchor not found within 30 lines")
oldblock = "\n".join(lines[i_u - 1:i_w + 1])
for marker in ("looks_like_recall_query", "wiki_context_for_query",
               "_memory.search_turns"):
    if marker not in oldblock:
        fail(f"edit5: expected marker not in old block: {marker!r}")
lines[i_u - 1:i_w + 1] = RECALLWIKI

# edit 6: warm-up kickoff at end of file
i = find_one(
    lines,
    lambda l: l.strip() == 'threading.Thread(target=_voice_thread_entry, daemon=True, name="chloe-voice").start()',
    "edit6")
lines[i + 1:i + 1] = [
    "",
    "# Warm the Ollama chat + embedding models in the background so the first",
    "# question after boot doesn't pay a cold load. Daemon thread - any failure",
    "# is logged and ignored, and it never blocks startup.",
    "threading.Thread(target=_warm_ollama_models, daemon=True,",
    '                 name="ollama-warm").start()',
]

new_j = "\n".join(lines)
try:
    ast.parse(new_j)
except SyntaxError as e:
    fail(f"jarvis.py would not parse: {e}")
# Edit 6 deliberately appends lines after the previous last line, so a tail-
# equality check is wrong here. Instead verify the original last line is
# still present (catches truncation) and the new warm-up thread is the new
# final functional line (catches a botched insert).
VOICE_LINE = 'threading.Thread(target=_voice_thread_entry, daemon=True, name="chloe-voice").start()'
WARM_LINE = '                 name="ollama-warm").start()'
if VOICE_LINE not in new_j:
    fail("jarvis.py: original voice-thread line missing after edits")
if not new_j.rstrip().endswith(WARM_LINE):
    fail("jarvis.py: warm-up thread kickoff not at end after edits")
write_fsync(JP, new_j)
jchk = read(JP)
ast.parse(jchk)
if VOICE_LINE not in jchk:
    fail("jarvis.py: voice-thread line missing on disk after write")
if not jchk.rstrip().endswith(WARM_LINE):
    fail("jarvis.py: warm-up kickoff not at end on disk after write")
for needle in ("OLLAMA_KEEP_ALIVE", "_ollama_chat_stream", "_OllamaToolCallNeeded",
               "_warm_ollama_models", "_ollama_chat_stream(ollama_msgs",
               "asyncio.gather("):
    if needle not in jchk:
        fail(f"jarvis.py post-write missing {needle!r}")
print(f"OK  jarvis.py  {len(jbak)} -> {len(jchk)} chars "
      f"({len(jbak.splitlines())} -> {len(jchk.splitlines())} lines)")


def patch_embed_file(path, bak):
    src = read(path)
    bsrc = read(bak)
    if "\r\n" in src:
        fail(f"{path.name} has CRLF")
    if "_EMBED_KEEP_ALIVE" in src:
        fail(f"{path.name} already patched")
    ls = src.split("\n")
    i = find_one(ls, lambda l: l.startswith("_EMBED_TIMEOUT"), f"{path.name}-const")
    ls[i + 1:i + 1] = [
        "# Keep the embedding model resident between calls (same knob as the",
        "# chat model) so recall + wiki lookups don't pay a cold reload.",
        '_EMBED_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m").strip()',
    ]
    j = find_one(
        ls,
        lambda l: l.strip() == 'json={"model": _EMBED_MODEL, "prompt": text.strip()},',
        f"{path.name}-payload")
    ind = indent_of(ls[j])
    ls[j:j + 1] = [
        ind + 'json={"model": _EMBED_MODEL, "prompt": text.strip(),',
        ind + '      "keep_alive": _EMBED_KEEP_ALIVE},',
    ]
    new = "\n".join(ls)
    try:
        ast.parse(new)
    except SyntaxError as e:
        fail(f"{path.name} would not parse: {e}")
    if new.splitlines()[-4:] != bsrc.splitlines()[-4:]:
        fail(f"{path.name} tail differs from backup pre-write")
    write_fsync(path, new)
    chk = read(path)
    ast.parse(chk)
    if chk.splitlines()[-4:] != bsrc.splitlines()[-4:]:
        fail(f"{path.name} TAIL MISMATCH after write")
    if "_EMBED_KEEP_ALIVE" not in chk or '"keep_alive": _EMBED_KEEP_ALIVE' not in chk:
        fail(f"{path.name} post-write missing keep_alive wiring")
    print(f"OK  {path.name}  {len(bsrc)} -> {len(chk)} chars")


patch_embed_file(CMP, CMP_BAK)
patch_embed_file(WEP, WEP_BAK)

print()
print("=" * 56)
print("  ALL LATENCY SPLICES OK")
print("=" * 56)
print()
print("Next steps:")
print("  1. start_chloe.vbs              (loads the patched code)")
print("  2. python bench_ollama.py       (confirm streaming first-token ~2s)")
print("  3. Ask Chloe a 'what is X' question; first word should appear fast.")
print()
print("Watch logs\\backend.log for these lines:")
print("    [chloe] Ollama warm-up: qwen2.5:32b resident (N.Ns)")
print("    [chloe] Ollama (qwen2.5:32b) stream: first token N.Ns, total N.Ns")
print()
print("If anything goes sideways, the backups are next to each file:")
print("    jarvis.py.bak.2026-05-14c")
print("    chloe_memory.py.bak.2026-05-14c")
print("    wiki_embedding.py.bak.2026-05-14c")
