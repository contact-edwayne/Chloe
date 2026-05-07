"""
mic_test.py — Standalone mic diagnostic. Run this from your jarvis venv:

    cd Documents\\jarvis
    venv\\Scripts\\activate
    python mic_test.py

What it does:
  1. Lists every audio input device.
  2. Records 4 seconds from each input device that looks like a microphone.
  3. Prints the peak / mean RMS for each device's recording.
  4. Tells you which device(s) actually picked up audio.

While it's recording from a device, the script will print "*** RECORDING — TALK NOW ***".
Speak loudly when you see that. Each device gets 4 seconds.

This bypasses Jarvis, openwakeword, and the WebSocket entirely. If NO device shows
voice here, the issue is your mic hardware or Windows audio settings, not Jarvis.
"""

import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
DURATION_S  = 4.0
SILENCE_RMS = 0.008  # same threshold jarvis.py uses

def list_input_devices():
    print("\n=== ALL INPUT DEVICES ===")
    devs = sd.query_devices()
    inputs = []
    default_in = sd.default.device[0] if sd.default.device else None
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            marker = "  ← DEFAULT" if i == default_in else ""
            print(f"  [{i:>2}] {d['name']!r:50s}  ch={d['max_input_channels']}, sr={int(d.get('default_samplerate', 0))}{marker}")
            inputs.append((i, d))
    return inputs

def test_device(dev_index, dev_info):
    name = dev_info["name"]
    print(f"\n--- Testing device [{dev_index}] {name!r} ---")
    print(f"    *** RECORDING {DURATION_S}s — TALK LOUDLY NOW ***")
    try:
        rec = sd.rec(
            int(DURATION_S * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=dev_index,
        )
        sd.wait()
    except Exception as e:
        print(f"    FAILED: {e}")
        return None

    samples = rec.flatten().astype(np.float32) / 32768.0
    rms_overall = float(np.sqrt(np.mean(samples ** 2)))
    rms_peak = float(np.max(np.abs(samples)))

    # Per-100ms RMS so we can see if voice landed at any moment
    chunk_n = SAMPLE_RATE // 10
    chunks = samples.reshape(-1, chunk_n)[:40]  # max 40 chunks of 100ms
    chunk_rms = np.sqrt(np.mean(chunks ** 2, axis=1))
    chunk_peak = float(chunk_rms.max())

    saw_voice = chunk_peak > SILENCE_RMS

    verdict = "✓ AUDIO DETECTED" if saw_voice else "✗ silent (no voice picked up)"
    print(f"    overall_rms={rms_overall:.4f}  peak_sample={rms_peak:.4f}  best_100ms_rms={chunk_peak:.4f}  → {verdict}")

    return {
        "index": dev_index,
        "name": name,
        "overall_rms": rms_overall,
        "peak_chunk": chunk_peak,
        "voice": saw_voice,
    }


def main():
    inputs = list_input_devices()

    # Filter to only Samson + Realtek + Microsoft Sound Mapper (skip Line/Analog connectors)
    candidates = []
    for idx, d in inputs:
        n = d["name"].lower()
        if any(k in n for k in ("samson", "realtek", "sound mapper")):
            # Skip Line/Analog — those aren't a mic
            if "line" in n or "analog connector" in n:
                continue
            candidates.append((idx, d))

    if not candidates:
        candidates = inputs  # fall back to testing all

    print(f"\n=== TESTING {len(candidates)} DEVICES — TALK LOUDLY DURING EACH ===")
    print("(There's a 1-second pause between devices. You'll see 'TALK NOW' before each.)")
    time.sleep(2)

    results = []
    for idx, d in candidates:
        r = test_device(idx, d)
        if r is not None:
            results.append(r)
        time.sleep(1)

    # Summary
    print("\n\n=== SUMMARY ===")
    voiced = [r for r in results if r["voice"]]
    silent = [r for r in results if not r["voice"]]

    if voiced:
        print(f"\n✓ {len(voiced)} device(s) DETECTED YOUR VOICE:")
        for r in sorted(voiced, key=lambda x: -x["peak_chunk"]):
            print(f"    [{r['index']:>2}] {r['name']!r}  best_100ms_rms={r['peak_chunk']:.4f}")
        best = max(voiced, key=lambda x: x["peak_chunk"])
        print(f"\nBest device: [{best['index']}] {best['name']}")
        print(f"\n→ To force jarvis to use this one, set this env var BEFORE running jarvis:")
        print(f"     set JARVIS_MIC={best['index']}")
        print(f"     python start_jarvis.py")
    else:
        print("\n✗ NO device picked up your voice. Possible causes:")
        print("    - Samson C01U gain knob turned all the way down (it has a physical knob on the front)")
        print("    - Windows microphone privacy: Settings → Privacy → Microphone → Allow apps")
        print("    - Samson is muted in Windows Sound settings → Recording → Properties")
        print("    - The mic is being held exclusively by another app (Discord, OBS, Teams, etc.)")

    if silent:
        print(f"\n  silent devices ({len(silent)}):")
        for r in silent:
            print(f"    [{r['index']:>2}] {r['name']!r}  best_100ms_rms={r['peak_chunk']:.4f}")


if __name__ == "__main__":
    main()
