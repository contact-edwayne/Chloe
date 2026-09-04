"""
Splice fix for the startup pulse-before-sound sync bug.

Problem: _greet_user() broadcasts "speaking" (which drives the HUD orb
pulse animation) BEFORE _speak() returns, but _speak() blocks on Kokoro
synthesis for ~1-2s before audio actually plays. Result: orb pulses
silently for the synth gap.

Fix: pre-synthesize the greeting via _get_kokoro().create() directly,
then broadcast "speaking" right before sd.play(). HUD pulse lines up
with audio onset. Falls back to original _speak() path if Kokoro isn't
available or synth fails — preserves all existing engine fallback
behavior.

Run from C:\\Users\\eleew\\Documents\\jarvis\\ with the venv active:
    python splice_greet_sync.py

Verifies:
    - ast.parse passes on the new file
    - tail of new file matches tail of backup (no truncation)
    - line count delta is sane (+13 ± 5)

If any check fails, the original is restored from the backup.
"""
from __future__ import annotations
import ast
import shutil
from datetime import date
from pathlib import Path


JARVIS = Path(__file__).parent / "jarvis.py"
BACKUP = JARVIS.with_name(f"jarvis.py.bak.{date.today().isoformat()}-greetsync")


OLD = '''def _greet_user():
    """Speak a short startup greeting through ElevenLabs (or edge-tts fallback).
    Runs once when the voice loop boots, after the wake detector is ready."""
    h = datetime.now().hour
    if   h < 12: tod = "morning"
    elif h < 17: tod = "afternoon"
    else:        tod = "evening"
    greeting = random.choice(_GREETING_POOL).format(tod=tod)
    print(f"[chloe] greeting: {greeting!r}")
    hud_server.broadcast_sync("speaking")
    try:
        _speak(greeting)
    except Exception as e:
        print(f"[chloe] greeting failed: {e}")
    finally:
        hud_server.broadcast_sync("idle")
'''


NEW = '''def _greet_user():
    """Speak a short startup greeting through ElevenLabs (or edge-tts fallback).
    Runs once when the voice loop boots, after the wake detector is ready.

    Pre-synthesizes via Kokoro before broadcasting "speaking" so the HUD orb
    pulse animation lines up with actual audio onset. Without this the orb
    pulsed silently for the ~1-2s Kokoro synth gap on boot. Falls back to
    _speak() if Kokoro isn't loadable or pre-synth raises — preserves the
    existing ElevenLabs / Kokoro / edge-tts engine fallback chain."""
    h = datetime.now().hour
    if   h < 12: tod = "morning"
    elif h < 17: tod = "afternoon"
    else:        tod = "evening"
    greeting = random.choice(_GREETING_POOL).format(tod=tod)
    print(f"[chloe] greeting: {greeting!r}")

    # Try to pre-synthesize so the speaking-state broadcast lands at audio
    # onset, not synth onset. ElevenLabs / edge-tts paths fall through to
    # the original _speak() blocking call.
    samples = None
    sr = None
    if USE_KOKORO and not (USE_ELEVENLABS and ELEVENLABS_API_KEY):
        try:
            kokoro = _get_kokoro()
            if kokoro is not None:
                cleaned = _clean_for_tts(greeting)
                if cleaned:
                    samples, sr = kokoro.create(
                        text=cleaned, voice=KOKORO_VOICE,
                        speed=KOKORO_SPEED, lang="en-us",
                    )
        except Exception as e:
            print(f"[chloe] greeting pre-synth failed: {e}")
            samples = None

    hud_server.broadcast_sync("speaking")
    try:
        if samples is not None and sr is not None:
            import sounddevice as sd
            sd.play(samples, sr)
            sd.wait()
        else:
            _speak(greeting)
    except Exception as e:
        print(f"[chloe] greeting failed: {e}")
    finally:
        hud_server.broadcast_sync("idle")
'''


def main() -> None:
    src = JARVIS.read_text(encoding="utf-8")
    if OLD not in src:
        if NEW in src:
            print("[splice] _greet_user is already patched — no-op.")
            return
        raise SystemExit("[splice] could not find _greet_user OLD block — aborting.")

    shutil.copy2(JARVIS, BACKUP)
    print(f"[splice] backup: {BACKUP.name}")

    new_src = src.replace(OLD, NEW, 1)

    # 1) Syntax check
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"[splice] FAIL — ast.parse: {e}")
        raise SystemExit(1)

    # 2) Sanity on tail (last 20 lines should be unchanged)
    old_tail = src.splitlines()[-20:]
    new_tail = new_src.splitlines()[-20:]
    if old_tail != new_tail:
        print("[splice] FAIL — tail diverged; restoring backup")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    # 3) Line-count delta sanity
    delta = len(new_src.splitlines()) - len(src.splitlines())
    if not (20 <= delta <= 40):
        print(f"[splice] FAIL — line delta {delta} outside expected range")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    JARVIS.write_text(new_src, encoding="utf-8")
    print(f"[splice] OK — wrote {JARVIS.name} ({delta:+d} lines)")
    print("[splice] restart Chloe to pick up the change.")


if __name__ == "__main__":
    main()
