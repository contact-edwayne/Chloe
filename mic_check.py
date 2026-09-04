"""
mic_check.py — One-shot diagnostic for the mic-pick + InputStream-open path.

Runs the same logic Chloe uses to pick a mic, then tries to open an InputStream
for half a second to confirm the device is actually usable. Prints a short
verdict you can paste back to Claude.

Usage (from Documents\\jarvis with the venv active):
    python mic_check.py
"""

import os
import sys
import time
import traceback

# Load .env so JARVIS_MIC / CHLOE_MIC reflect what Chloe would actually see.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # if python-dotenv isn't installed, env vars from the shell still apply

import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # matches jarvis.py's openwakeword frame length

MIC_DEVICE_OVERRIDE = (
    os.environ.get("CHLOE_MIC") or os.environ.get("JARVIS_MIC", "")
).strip() or None


def _resolve_mic_device():
    """Mirror of jarvis.py:_resolve_mic_device with the post-fix three-pass
    substring match."""
    if MIC_DEVICE_OVERRIDE is not None:
        if MIC_DEVICE_OVERRIDE.isdigit():
            return int(MIC_DEVICE_OVERRIDE), "numeric override"
        needle = MIC_DEVICE_OVERRIDE.lower()
        host_apis = sd.query_hostapis()

        def _hostname(d):
            h = d.get("hostapi")
            if h is None or not (0 <= h < len(host_apis)):
                return ""
            return host_apis[h]["name"].upper()

        devs = list(enumerate(sd.query_devices()))
        for label, predicate in (
            ("WASAPI", lambda h: "WASAPI" in h),
            ("non-WDM-KS", lambda h: "WDM-KS" not in h and "KERNEL STREAMING" not in h),
            ("any-host", lambda h: True),
        ):
            for i, d in devs:
                if (d.get("max_input_channels", 0) > 0
                    and needle in d["name"].lower()
                    and predicate(_hostname(d))):
                    return i, f"substring '{MIC_DEVICE_OVERRIDE}' / {label} pass / {_hostname(d) or '?'}"
        return None, f"substring '{MIC_DEVICE_OVERRIDE}' matched no device"

    # No override → prefer a WASAPI entry of the system default mic
    try:
        default_idx = sd.default.device[0] if sd.default.device else None
        if default_idx is None:
            return None, "no system default mic"
        default_name = sd.query_devices(default_idx).get("name", "").lower().strip()
        host_apis = sd.query_hostapis()
        wasapi_idx = None
        for i, h in enumerate(host_apis):
            if "WASAPI" in h["name"].upper():
                wasapi_idx = i
                break
        if wasapi_idx is None:
            return None, "no WASAPI host API present"

        tokens = [t.strip("()") for t in default_name.split()]
        skip = {"microphone", "input", "audio", "mic", "(", ")"}
        needles = [t for t in tokens if t and t not in skip]
        if not needles:
            needles = [tokens[0]] if tokens else []

        for i, d in enumerate(sd.query_devices()):
            if (d.get("max_input_channels", 0) > 0
                and d.get("hostapi") == wasapi_idx):
                dn = d["name"].lower()
                if any(n in dn for n in needles):
                    return i, f"auto-WASAPI / matched on {needles[0]!r}"
        return None, f"auto-WASAPI: no WASAPI device matched needles {needles!r}"
    except Exception as e:
        return None, f"auto-WASAPI failed: {type(e).__name__}: {e}"


def _list_inputs():
    host_apis = sd.query_hostapis()
    default_in = sd.default.device[0] if sd.default.device else None
    print("=== INPUT DEVICES ===")
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            api = host_apis[d["hostapi"]]["name"] if d.get("hostapi") is not None else "?"
            mark = "  <-- DEFAULT" if i == default_in else ""
            print(f"  [{i:>2}] {d['name']!r:48s}  ({api}){mark}")


def _try_open(device):
    """Attempt to open + read one chunk. Mirrors what the wake loop does."""
    print(f"\n=== TRYING TO OPEN device={device} ===")
    try:
        # Same negotiation as jarvis._pick_device_samplerate
        try:
            sd.check_input_settings(device=device, samplerate=SAMPLE_RATE,
                                    channels=1, dtype="int16")
            rate = SAMPLE_RATE
        except Exception:
            info = sd.query_devices(device) if device is not None else {}
            rate = int(info.get("default_samplerate") or 48000)
        block = CHUNK_SAMPLES if rate == SAMPLE_RATE else int(round(CHUNK_SAMPLES * rate / SAMPLE_RATE))

        stream = sd.InputStream(
            samplerate=rate, channels=1, dtype="int16",
            blocksize=block, device=device,
        )
        stream.start()
        time.sleep(0.5)
        chunk, _ = stream.read(block)
        stream.stop()
        stream.close()
        print(f"  PASS  rate={rate} block={block} read_ok shape={getattr(chunk, 'shape', '?')}")
        return True
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}")
        return False


def main():
    print(f"sounddevice {sd.__version__}")
    print(f"MIC_DEVICE_OVERRIDE = {MIC_DEVICE_OVERRIDE!r}  (from CHLOE_MIC / JARVIS_MIC)")
    print()
    _list_inputs()
    print()
    picked, why = _resolve_mic_device()
    print(f"=== PICKER RESULT ===")
    print(f"  device = {picked}")
    print(f"  reason = {why}")

    # Try the picked device first; if None (OS default), try None.
    ok = _try_open(picked)
    if not ok and picked is not None:
        print("\n  -> retrying with device=None (OS default)")
        _try_open(None)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
