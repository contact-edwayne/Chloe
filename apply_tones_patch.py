"""Splice script for tts_tones (Kokoro tonal shifts).

**Requires apply_nsfw_patch.py to have been run first** — anchors against
the `import nsfw_mode` line that NSFW splice inserts.

Applies four patches to jarvis.py:
  1. import tts_tones
  2. /tone chat slash handler (in handle_chat, before /search block)
  3. _kokoro_to_wav_bytes — parse tag + use per-tone voice/speed
  4. _speak_kokoro — parse tag once + producer uses per-tone voice/speed

Safety: idempotent, backed up, ast.parse-validated, tail-diff output.
Run from C:\\Users\\eleew\\Documents\\jarvis\\:

    python apply_tones_patch.py
"""

from __future__ import annotations

import ast
import datetime
import shutil
import sys
from pathlib import Path

JARVIS_DIR = Path(__file__).parent.resolve()
JARVIS_PY = JARVIS_DIR / "jarvis.py"
TONES_MOD = JARVIS_DIR / "tts_tones.py"

for required in (JARVIS_PY, TONES_MOD):
    if not required.exists():
        sys.exit(f"FAIL: missing {required.name} in {JARVIS_DIR}")


# ─── P1: import tts_tones ──────────────────────────────────────────────────
# Anchors on the nsfw_mode import line — fails if apply_nsfw_patch.py
# hasn't been run yet.
P1_ANCHOR = "import nsfw_mode  # permissive-mode toggle + persona block\n"
P1_REPLACE = (
    "import nsfw_mode  # permissive-mode toggle + persona block\n"
    "import tts_tones  # tonal style tags + Kokoro voice/speed mapping\n"
)


# ─── P2: /tone chat slash handler ──────────────────────────────────────────
# Insert before the /search block. After nsfw is applied, the order
# in handle_chat becomes: /nsfw → /tone → /search.
P2_ANCHOR = "    # Explicit /search /lookup /web slash commands — Brave Search backend.\n"
P2_REPLACE = (
    "    # /tone <name> | /tone reset | /tone status — TTS tonal style.\n"
    "    # Sticky across turns; auto-resets to neutral when /nsfw flips off.\n"
    "    if messages:\n"
    "        _last_user_t = _user_text_from_message(messages[-1]) or \"\"\n"
    "        tone_reply = tts_tones.try_handle_command(_last_user_t)\n"
    "        if tone_reply is not None:\n"
    "            _push_history(\"user\", _last_user_t, modality=\"chat\")\n"
    "            _push_history(\"assistant\", tone_reply, modality=\"chat\")\n"
    "            await _ws_send(websocket, {\"type\": \"start\"})\n"
    "            await _ws_send(websocket, {\"type\": \"delta\", \"text\": tone_reply})\n"
    "            await _ws_send(websocket, {\"type\": \"done\"})\n"
    "            if not data.get(\"no_tts\"):\n"
    "                try:\n"
    "                    await _reply_audio_or_speak(tone_reply, data, label=\"chat-tone\")\n"
    "                except Exception as e:\n"
    "                    print(f\"[chloe] chat TTS error on tone reply: {e}\")\n"
    "            return\n"
    "\n"
    "    # Explicit /search /lookup /web slash commands — Brave Search backend.\n"
)


# ─── P3: _kokoro_to_wav_bytes — parse tag + per-tone voice/speed ───────────
P3_ANCHOR = (
    "def _kokoro_to_wav_bytes(text: str):\n"
    "    \"\"\"Kokoro synthesis → WAV bytes (no local playback). Mobile path.\"\"\"\n"
    "    engine = _get_kokoro()\n"
    "    if engine is None:\n"
    "        return None\n"
    "    try:\n"
    "        t0 = time.time()\n"
    "        samples, sr = engine.create(\n"
    "            text,\n"
    "            voice=KOKORO_VOICE,\n"
    "            speed=KOKORO_SPEED,\n"
    "            lang=\"en-us\",\n"
    "        )\n"
)
P3_REPLACE = (
    "def _kokoro_to_wav_bytes(text: str):\n"
    "    \"\"\"Kokoro synthesis → WAV bytes (no local playback). Mobile path.\n"
    "\n"
    "    Parses a leading tone tag (e.g. [intimate]) and uses the matching\n"
    "    Kokoro voice + speed for synthesis. See tts_tones.PALETTE.\"\"\"\n"
    "    engine = _get_kokoro()\n"
    "    if engine is None:\n"
    "        return None\n"
    "    text, _kvoice, _kspeed = tts_tones.parse_and_get(\n"
    "        text, default_voice=KOKORO_VOICE, default_speed=KOKORO_SPEED)\n"
    "    if not text.strip():\n"
    "        return None\n"
    "    try:\n"
    "        t0 = time.time()\n"
    "        samples, sr = engine.create(\n"
    "            text,\n"
    "            voice=_kvoice,\n"
    "            speed=_kspeed,\n"
    "            lang=\"en-us\",\n"
    "        )\n"
)


# ─── P4: _speak_kokoro — parse once before split, producer uses _kvoice/_kspeed ──
# Multi-line anchor including the unique producer setup so this only matches
# inside _speak_kokoro (not _kokoro_to_wav_bytes).
P4_ANCHOR = (
    "    sentences = _split_sentences_for_tts(text)\n"
    "    if not sentences:\n"
    "        return\n"
    "\n"
    "    audio_queue: queue.Queue = queue.Queue(maxsize=3)\n"
    "    SENTINEL = object()\n"
    "\n"
    "    def _producer():\n"
    "        try:\n"
    "            for idx, sent in enumerate(sentences):\n"
    "                if _barge_in_request.is_set():\n"
    "                    break\n"
    "                try:\n"
    "                    samples, sample_rate = kokoro.create(\n"
    "                        text=sent,\n"
    "                        voice=KOKORO_VOICE,\n"
    "                        speed=KOKORO_SPEED,\n"
    "                        lang=\"en-us\",\n"
    "                    )\n"
)
P4_REPLACE = (
    "    # Parse leading tone tag once (sticky across all sentences in this reply).\n"
    "    text, _kvoice, _kspeed = tts_tones.parse_and_get(\n"
    "        text, default_voice=KOKORO_VOICE, default_speed=KOKORO_SPEED)\n"
    "    sentences = _split_sentences_for_tts(text)\n"
    "    if not sentences:\n"
    "        return\n"
    "\n"
    "    audio_queue: queue.Queue = queue.Queue(maxsize=3)\n"
    "    SENTINEL = object()\n"
    "\n"
    "    def _producer():\n"
    "        try:\n"
    "            for idx, sent in enumerate(sentences):\n"
    "                if _barge_in_request.is_set():\n"
    "                    break\n"
    "                try:\n"
    "                    samples, sample_rate = kokoro.create(\n"
    "                        text=sent,\n"
    "                        voice=_kvoice,\n"
    "                        speed=_kspeed,\n"
    "                        lang=\"en-us\",\n"
    "                    )\n"
)


JARVIS_PATCHES = [
    ("P1 import tts_tones", P1_ANCHOR, P1_REPLACE),
    ("P2 /tone chat slash handler", P2_ANCHOR, P2_REPLACE),
    ("P3 _kokoro_to_wav_bytes per-tone synthesis", P3_ANCHOR, P3_REPLACE),
    ("P4 _speak_kokoro per-tone synthesis", P4_ANCHOR, P4_REPLACE),
]


def apply_patches(src: str, patches: list[tuple[str, str, str]]) -> str:
    current = src
    for name, anchor, replace in patches:
        count = current.count(anchor)
        if count == 0:
            raise RuntimeError(f"{name}: anchor not found")
        if count > 1:
            raise RuntimeError(f"{name}: anchor found {count} times — not unique")
        new = current.replace(anchor, replace, 1)
        if new == current:
            raise RuntimeError(f"{name}: replace produced no change")
        current = new
        print(f"  ✓ applied: {name}")
    return current


def tail_diff(label: str, src: str, anchor_first_line: str) -> None:
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if anchor_first_line.rstrip("\n") in line:
            lo = max(0, i - 1)
            hi = min(len(lines), i + 8)
            print(f"\n--- {label} (line ~{i+1}) ---")
            for j in range(lo, hi):
                print(f"  {j+1:5d} | {lines[j]}")
            break


def main() -> int:
    print(f"[apply_tones] working dir: {JARVIS_DIR}")
    src = JARVIS_PY.read_text(encoding="utf-8")

    # Dependency check: nsfw patch must be applied first (its `import
    # nsfw_mode` line is our P1 anchor).
    if "import nsfw_mode" not in src:
        print("ABORT: jarvis.py is missing 'import nsfw_mode'.")
        print("Run apply_nsfw_patch.py first — tones depends on the nsfw_mode import.")
        return 1

    # Idempotency check
    if "import tts_tones" in src:
        print("ABORT: jarvis.py already contains 'import tts_tones'.")
        print("Looks already-patched. Inspect manually before re-running.")
        return 1

    # Backup
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    bak = JARVIS_PY.with_suffix(JARVIS_PY.suffix + f".bak.{ts}")
    shutil.copy2(JARVIS_PY, bak)
    print(f"backup: {bak.name}")

    print("\napplying patches:")
    try:
        new_src = apply_patches(src, JARVIS_PATCHES)
    except RuntimeError as e:
        print(f"\nFAIL: {e}")
        print("No files modified. Backup can be deleted.")
        return 1

    print("\nvalidating syntax with ast.parse...")
    try:
        ast.parse(new_src, filename=str(JARVIS_PY))
        print("  ✓ jarvis.py parses cleanly")
    except SyntaxError as e:
        print(f"FAIL: syntax error after patch: {e}")
        print("No files modified.")
        return 1

    JARVIS_PY.write_text(new_src, encoding="utf-8")
    print("\nfile written.")

    print("\n========== tail-diff ==========")
    tail_diff("P1 import tts_tones", new_src, "import tts_tones")
    tail_diff("P2 /tone slash handler", new_src, "# /tone <name> | /tone reset")
    tail_diff("P3 _kokoro_to_wav_bytes", new_src, "Parses a leading tone tag")
    tail_diff("P4 _speak_kokoro", new_src, "Parse leading tone tag once (sticky")

    print("\n========== done ==========")
    print("Next:")
    print("  1. stop_chloe.bat")
    print("  2. start_chloe.vbs")
    print("  3. /tone status  → 'tone: neutral'")
    print("  4. /tone playful → 'tone set to playful.' Next reply should sound quicker.")
    print("  5. /tone reset → back to neutral.")
    print("  6. With /nsfw on, ask an intimate-coded question. Chloe should emit")
    print("     a tag like [intimate] at the start, which gets stripped before TTS")
    print("     but shifts the voice to af_sky at 0.88x speed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
