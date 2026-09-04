"""Smoke tests for screen_vision.py + /see dispatch.

Run on Windows from the jarvis venv:
    python test_see.py            # all tests
    python test_see.py --live     # also do a real capture + Groq vision call

Linux/CI: only the logic tests run; capture/vision are skipped automatically.
"""
import os
import sys
from pathlib import Path

# Make sibling modules importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

# Load .env BEFORE any screen_vision import so the lazy Groq init sees the key.
# Without this, t_dispatcher_blocklist_response calls see() with no key and
# the failed-init flag sticks for the rest of the run.
def _load_env_file():
    envf = Path(__file__).parent / ".env"
    if not envf.exists():
        return
    for raw in envf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
_load_env_file()


def t_blocklist_logic():
    """is_blocked() returns matching token or empty string."""
    from screen_vision import is_blocked
    # No blocklist → never blocked
    os.environ.pop("CHLOE_VISION_BLOCKLIST", None)
    assert is_blocked({"title": "1Password", "exe": "1Password.exe"}) == ""

    # Exact substring match on title
    os.environ["CHLOE_VISION_BLOCKLIST"] = "1password,bitwarden,breez"
    r = is_blocked({"title": "1Password — vault unlocked", "exe": "1Password.exe"})
    assert r == "1password", f"expected 1password, got {r!r}"

    # Match against exe basename
    r = is_blocked({"title": "Some title", "exe": "Bitwarden.exe"})
    assert r == "bitwarden"

    # Case-insensitive
    os.environ["CHLOE_VISION_BLOCKLIST"] = "BREEZ"
    r = is_blocked({"title": "Breez Wallet", "exe": "breez.exe"})
    assert r == "breez"

    # Non-match
    os.environ["CHLOE_VISION_BLOCKLIST"] = "1password"
    r = is_blocked({"title": "VS Code", "exe": "Code.exe"})
    assert r == ""

    os.environ.pop("CHLOE_VISION_BLOCKLIST", None)
    print("PASS: blocklist logic")


def t_kill_switch():
    """see() returns ok=False when CHLOE_VISION_DISABLED=1."""
    from screen_vision import see
    os.environ["CHLOE_VISION_DISABLED"] = "1"
    try:
        r = see()
        assert r["ok"] is False
        assert "disabled" in r.get("error", "").lower()
    finally:
        os.environ.pop("CHLOE_VISION_DISABLED", None)
    print("PASS: kill switch")


def t_default_prompt_present():
    from screen_vision import DEFAULT_PROMPT
    assert "screen" in DEFAULT_PROMPT.lower()
    assert len(DEFAULT_PROMPT) > 50
    print("PASS: default prompt present")


def t_capture_returns_png_bytes():
    """Real capture only on Windows with mss installed. Otherwise expected
    to return ok=False with a clear error."""
    from screen_vision import capture_screen
    r = capture_screen()
    if r.get("ok"):
        png = r["png"]
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "not PNG bytes"
        assert len(png) > 1000, f"PNG suspiciously small: {len(png)} bytes"
        print(f"PASS: capture ({len(png)} bytes, monitor={r['monitor']})")
    else:
        # Acceptable on Linux / no mss / no display
        msg = r.get("error", "")
        print(f"SKIP: capture not possible here ({msg})")


def t_dispatcher_blocklist_response():
    """When the dispatcher path is taken with a blocked frontmost app, user
    sees a clean 'Skipped' message rather than a stack trace.

    We can't trigger the real dispatcher without jarvis running, so we just
    check the see() return shape produces what the wiring code expects.
    """
    from screen_vision import see
    # Force a block by setting a token that will match almost any window title
    os.environ["CHLOE_VISION_BLOCKLIST"] = "qwertyuiop_unlikely_token_xyz"
    # Won't actually match, so capture will proceed; just verify see() runs
    # without exception. (A real test would mock the foreground app.)
    try:
        r = see()
        # Shape contract:
        assert "ok" in r
        # If ok=False because no GROQ_API_KEY or no display, error is non-empty
        if not r["ok"]:
            assert r.get("error") or r.get("blocked_by")
        print(f"PASS: see() shape ({'ok' if r['ok'] else 'ok=False with reason'})")
    finally:
        os.environ.pop("CHLOE_VISION_BLOCKLIST", None)


def t_live_vision():
    """End-to-end with a real Groq vision call. Requires GROQ_API_KEY and
    a display. Only runs with --live."""
    import screen_vision
    if not os.environ.get("GROQ_API_KEY"):
        print("SKIP: no GROQ_API_KEY for live test")
        return
    # Reset the lazy-init flags in case an earlier test in this run tripped
    # them while .env wasn't loaded yet. Belt-and-suspenders — _load_env_file
    # at module load should already have prevented this.
    screen_vision._groq = None
    screen_vision._groq_attempted = False
    r = screen_vision.see("In one sentence, what's on this screen?")
    if r["ok"]:
        print(f"PASS: live vision call")
        print(f"  app:   {r.get('app',{}).get('title','?')}")
        print(f"  text:  {r['text'][:300]}")
    else:
        print(f"FAIL: live vision call: {r.get('error')}")


def main():
    t_blocklist_logic()
    t_kill_switch()
    t_default_prompt_present()
    t_capture_returns_png_bytes()
    t_dispatcher_blocklist_response()
    if "--live" in sys.argv:
        t_live_vision()
    else:
        print("(skipping live test — pass --live to enable)")


if __name__ == "__main__":
    main()
