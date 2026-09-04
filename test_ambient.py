"""Smoke tests for ambient_vision.py.

Run on Windows from the jarvis venv:
    python test_ambient.py

Tests monkeypatch screen_vision + BRAIN so they don't need a display, Groq
key, or real brain folder.
"""
import os
import sys
import time
import threading
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _load_env_file():
    envf = Path(__file__).parent / ".env"
    if not envf.exists():
        return
    for raw in envf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
_load_env_file()

# Override brain root for non-Windows test runners
if not os.environ.get("CHLOE_BRAIN_ROOT") or not Path(os.environ["CHLOE_BRAIN_ROOT"]).exists():
    _TMP = tempfile.mkdtemp(prefix="chloe_test_brain_")
    os.environ["CHLOE_BRAIN_ROOT"] = _TMP


def t_status_when_off():
    import ambient_vision as av
    s = av.status()
    assert s["running"] is False
    assert s["ticks_total"] == 0
    print("PASS: status when off")


def _patch_for_test(blocked=False, idx_init=False):
    """Install fake screen_vision + brain. Returns (restore_fn, captured_list)."""
    import brain_wiring
    import screen_vision

    captured = []

    class FakeBrain:
        def episodic_append(self, line):
            captured.append(line)

    sv_originals = {
        'get_frontmost_app': screen_vision.get_frontmost_app,
        'is_blocked':        screen_vision.is_blocked,
        'capture_screen':    screen_vision.capture_screen,
        'describe_screen':   screen_vision.describe_screen,
    }
    original_brain = brain_wiring.BRAIN
    brain_wiring.BRAIN = FakeBrain()

    fake_app = {"ok": True, "title": "Test Window", "exe": "test.exe",
                "hwnd": 1, "rect": (0, 0, 100, 100)}
    screen_vision.get_frontmost_app = lambda: fake_app
    screen_vision.is_blocked = (lambda app: "1password") if blocked else (lambda app: "")
    screen_vision.capture_screen = lambda monitor_index=None: {
        "ok": True, "png": b"\x89PNG\r\n\x1a\nFAKE",
        "monitor": {"left": 0, "top": 0, "width": 100, "height": 100},
        "app": fake_app,
    }
    screen_vision.describe_screen = lambda image_bytes, prompt="": {
        "ok": True, "text": "fake summary of test screen", "model": "fake",
    }

    def restore():
        brain_wiring.BRAIN = original_brain
        for k, v in sv_originals.items():
            setattr(screen_vision, k, v)

    return restore, captured


def t_start_stop_lifecycle():
    import ambient_vision as av
    restore, captured = _patch_for_test(blocked=False)
    try:
        s = av.start(minutes=0.05)
        assert s["running"] is True
        assert abs(s["minutes"] - 0.05) < 1e-6
        time.sleep(7.5)
        s = av.status()
        assert s["ticks_total"] >= 2, f"expected >=2 ticks, got {s['ticks_total']}"
        assert len(captured) >= 2
        assert captured[0].startswith("[test]"), f"bad format: {captured[0]!r}"
        assert "fake summary" in captured[0]
        print(f"PASS: lifecycle ({s['ticks_total']} ticks, {len(captured)} entries)")
        r = av.stop()
        assert r["running"] is False
        time.sleep(0.2)
        s = av.status()
        assert s["running"] is False
        print("PASS: stop")
    finally:
        try:
            av.stop()
        except Exception:
            pass
        restore()


def t_blocklist_silent_skip():
    import ambient_vision as av
    restore, captured = _patch_for_test(blocked=True)
    try:
        av.start(minutes=0.05)
        time.sleep(4.0)
        s = av.status()
        av.stop()
        assert len(captured) == 0, f"blocklist leak: {captured}"
        assert s["ticks_skipped_blocked"] >= 1, f"expected blocked skips, got {s}"
        print(f"PASS: blocklist silent skip ({s['ticks_skipped_blocked']} skipped, 0 logged)")
    finally:
        try:
            av.stop()
        except Exception:
            pass
        restore()


def t_disabled_kill_switch():
    import ambient_vision as av
    restore, captured = _patch_for_test(blocked=False)
    os.environ["CHLOE_VISION_DISABLED"] = "1"
    try:
        av.start(minutes=0.05)
        time.sleep(4.0)
        s = av.status()
        av.stop()
        assert len(captured) == 0
        assert s["ticks_skipped_disabled"] >= 1
        print(f"PASS: disabled kill switch ({s['ticks_skipped_disabled']} skipped, 0 logged)")
    finally:
        os.environ.pop("CHLOE_VISION_DISABLED", None)
        try:
            av.stop()
        except Exception:
            pass
        restore()


def main():
    t_status_when_off()
    t_start_stop_lifecycle()
    t_blocklist_silent_skip()
    t_disabled_kill_switch()


if __name__ == "__main__":
    main()
