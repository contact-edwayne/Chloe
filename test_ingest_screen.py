"""Smoke tests for /ingest_screen handler.

Run on Windows from the jarvis venv:
    python test_ingest_screen.py
"""
import os
import sys
import tempfile
import shutil
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

# Fresh tmp brain for each run so ingest tests don't accumulate
if not os.environ.get("CHLOE_BRAIN_ROOT") or not Path(os.environ["CHLOE_BRAIN_ROOT"]).exists():
    _TMP = tempfile.mkdtemp(prefix="chloe_test_brain_")
    os.environ["CHLOE_BRAIN_ROOT"] = _TMP


def _patch(blocked=False, capture_ok=True, vision_ok=True, vision_text=None,
           ingest_ok=True):
    """Install fakes for screen_vision + BRAIN.ingest."""
    import brain_wiring
    import screen_vision

    sv_originals = {
        'get_frontmost_app': screen_vision.get_frontmost_app,
        'is_blocked':        screen_vision.is_blocked,
        'capture_screen':    screen_vision.capture_screen,
        'describe_screen':   screen_vision.describe_screen,
    }
    original_ingest = brain_wiring.BRAIN.ingest

    fake_app = {"ok": True, "title": "Groq Console", "exe": "chrome.exe",
                "hwnd": 1, "rect": (0, 0, 100, 100)}
    screen_vision.get_frontmost_app = lambda: fake_app
    screen_vision.is_blocked = (lambda app: "1password") if blocked else (lambda app: "")
    if capture_ok:
        screen_vision.capture_screen = lambda monitor_index=None: {
            "ok": True, "png": b"\x89PNG\r\n\x1a\nFAKE",
            "monitor": {"left": 0, "top": 0, "width": 2560, "height": 1080,
                        "name": "LG ULTRAWIDE"},
            "app": fake_app,
        }
    else:
        screen_vision.capture_screen = lambda monitor_index=None: {
            "ok": False, "error": "fake capture failure",
        }
    if vision_ok:
        text = vision_text if vision_text is not None else (
            "# Groq Console\n\n"
            "URL: https://console.groq.com/dashboard\n\n"
            "## Visible Text\n\n"
            "Models: llama-3.3-70b-versatile, llama-4-scout, compound-mini.\n"
            "Daily quota: 14,400 requests / day.\n"
            "Reset: 00:00 UTC.\n"
        )
        screen_vision.describe_screen = lambda image_bytes, prompt="": {
            "ok": True, "text": text, "model": "fake",
        }
    else:
        screen_vision.describe_screen = lambda image_bytes, prompt="": {
            "ok": False, "error": "fake vision failure",
        }

    if ingest_ok:
        def fake_ingest(filename):
            return {
                "slug": Path(filename).stem,
                "tldr": "Groq Console dashboard showing models and quotas.",
                "entities_touched": ["groq", "llama_3_3_70b"],
                "concepts_touched": ["daily_quota"],
            }
    else:
        def fake_ingest(filename):
            raise RuntimeError("fake ingest crash")
    brain_wiring.BRAIN.ingest = fake_ingest

    def restore():
        for k, v in sv_originals.items():
            setattr(screen_vision, k, v)
        brain_wiring.BRAIN.ingest = original_ingest

    return restore


def t_slug_validation():
    from brain_wiring import handle_ingest_screen, _validate_slug
    assert _validate_slug("") != ""
    assert _validate_slug("foo bar") != ""
    assert _validate_slug("foo/bar") != ""
    assert _validate_slug("../etc") != ""
    assert _validate_slug("foo!") != ""
    assert _validate_slug("a" * 100) != ""
    assert _validate_slug("good_slug") == ""
    assert _validate_slug("good-slug-2") == ""
    # Handler should reject without capturing
    r = handle_ingest_screen("")
    assert "Usage" in r and "failed" in r
    r = handle_ingest_screen("bad slug")
    assert "must not contain spaces" in r or "snake_case" in r
    print("PASS: slug validation")


def t_happy_path_writes_file_and_ingests():
    from brain_wiring import handle_ingest_screen, BRAIN
    restore = _patch()
    slug = "groq_console_quotas_test"
    # Pre-clean
    raw_path = BRAIN.raw_dir / f"{slug}.md"
    if raw_path.exists():
        raw_path.unlink()
    try:
        r = handle_ingest_screen(slug)
        assert "Captured + ingested" in r, f"unexpected reply: {r[:300]!r}"
        assert raw_path.exists(), f"source file not written at {raw_path}"
        body = raw_path.read_text(encoding="utf-8")
        # Frontmatter sanity
        assert body.startswith("---\n"), "missing frontmatter"
        assert "captured_at:" in body
        assert "capture_app_exe: chrome.exe" in body
        assert "capture_monitor: LG ULTRAWIDE" in body
        # Vision body landed in
        assert "Groq Console" in body and "Daily quota" in body
        print("PASS: happy path writes source + runs ingest")
    finally:
        if raw_path.exists():
            raw_path.unlink()
        restore()


def t_blocklist_refuses_no_write():
    from brain_wiring import handle_ingest_screen, BRAIN
    restore = _patch(blocked=True)
    slug = "should_not_appear"
    raw_path = BRAIN.raw_dir / f"{slug}.md"
    if raw_path.exists():
        raw_path.unlink()
    try:
        r = handle_ingest_screen(slug)
        assert "Skipped" in r and "blocklist" in r.lower(), f"got: {r!r}"
        assert not raw_path.exists(), "source file written despite blocklist"
        print("PASS: blocklist refuses, no file written")
    finally:
        restore()


def t_kill_switch_refuses():
    from brain_wiring import handle_ingest_screen, BRAIN
    restore = _patch()
    slug = "kill_switch_test"
    raw_path = BRAIN.raw_dir / f"{slug}.md"
    if raw_path.exists():
        raw_path.unlink()
    os.environ["CHLOE_VISION_DISABLED"] = "1"
    try:
        r = handle_ingest_screen(slug)
        assert "disabled" in r.lower(), f"got: {r!r}"
        assert not raw_path.exists()
        print("PASS: kill switch refuses, no file written")
    finally:
        os.environ.pop("CHLOE_VISION_DISABLED", None)
        restore()


def t_vision_failure_no_write():
    from brain_wiring import handle_ingest_screen, BRAIN
    restore = _patch(vision_ok=False)
    slug = "vision_fail_test"
    raw_path = BRAIN.raw_dir / f"{slug}.md"
    if raw_path.exists():
        raw_path.unlink()
    try:
        r = handle_ingest_screen(slug)
        assert "Vision call failed" in r, f"got: {r!r}"
        assert not raw_path.exists()
        print("PASS: vision failure, no file written")
    finally:
        restore()


def t_ingest_failure_keeps_source_file():
    """If BRAIN.ingest crashes, source file should still exist with a hint."""
    from brain_wiring import handle_ingest_screen, BRAIN
    restore = _patch(ingest_ok=False)
    slug = "ingest_fail_test"
    raw_path = BRAIN.raw_dir / f"{slug}.md"
    if raw_path.exists():
        raw_path.unlink()
    try:
        r = handle_ingest_screen(slug)
        assert "ingest failed" in r, f"got: {r!r}"
        assert raw_path.exists(), "source file should still be written"
        assert f"/ingest {slug}.md" in r, "should hint at retry command"
        print("PASS: ingest failure keeps source file with retry hint")
    finally:
        if raw_path.exists():
            raw_path.unlink()
        restore()


def main():
    t_slug_validation()
    t_happy_path_writes_file_and_ingests()
    t_blocklist_refuses_no_write()
    t_kill_switch_refuses()
    t_vision_failure_no_write()
    t_ingest_failure_keeps_source_file()


if __name__ == "__main__":
    main()
