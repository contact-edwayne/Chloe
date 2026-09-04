"""Smoke tests for the /ask vision-augmented query handler.

Run on Windows from the jarvis venv:
    python test_ask.py

Tests monkeypatch screen_vision and BRAIN/chloe_llm_call so they don't
need a display, Groq key, or real brain folder.
"""
import os
import sys
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

# Set tmp brain root for non-Windows runners
if not os.environ.get("CHLOE_BRAIN_ROOT") or not Path(os.environ["CHLOE_BRAIN_ROOT"]).exists():
    _TMP = tempfile.mkdtemp(prefix="chloe_test_brain_")
    os.environ["CHLOE_BRAIN_ROOT"] = _TMP
    # Seed minimal wiki so BRAIN.read('wiki/index.md') works
    wiki = Path(_TMP) / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Concepts\n- [[qmd]] — local search engine\n",
        encoding="utf-8",
    )
    (wiki / "concepts").mkdir(exist_ok=True)
    (wiki / "concepts" / "qmd.md").write_text(
        "# qmd\n\nLocal search engine for markdown files using BM25 + vector search.\n",
        encoding="utf-8",
    )


def _patch(blocked=False, capture_ok=True, vision_ok=True):
    """Install fakes for screen_vision + chloe_llm_call. Returns (restore, captured_prompt_list)."""
    import brain_wiring
    import screen_vision

    captured_prompts = []

    sv_originals = {
        'get_frontmost_app': screen_vision.get_frontmost_app,
        'is_blocked':        screen_vision.is_blocked,
        'capture_screen':    screen_vision.capture_screen,
        'describe_screen':   screen_vision.describe_screen,
    }
    original_llm = brain_wiring.chloe_llm_call

    fake_app = {"ok": True, "title": "VS Code", "exe": "Code.exe",
                "hwnd": 1, "rect": (0, 0, 100, 100)}
    screen_vision.get_frontmost_app = lambda: fake_app
    screen_vision.is_blocked = (lambda app: "1password") if blocked else (lambda app: "")
    if capture_ok:
        screen_vision.capture_screen = lambda monitor_index=None: {
            "ok": True, "png": b"\x89PNG\r\n\x1a\nFAKE",
            "monitor": {"left": 0, "top": 0, "width": 100, "height": 100},
            "app": fake_app,
        }
    else:
        screen_vision.capture_screen = lambda monitor_index=None: {
            "ok": False, "error": "fake capture failure",
        }
    if vision_ok:
        screen_vision.describe_screen = lambda image_bytes, prompt="": {
            "ok": True, "text": "Editing screen_vision.py in VS Code with a TypeError on line 42",
            "model": "fake",
        }
    else:
        screen_vision.describe_screen = lambda image_bytes, prompt="": {
            "ok": False, "error": "fake vision failure",
        }

    def fake_llm(prompt, mode):
        captured_prompts.append((mode, prompt))
        return "FAKE_ANSWER: combined screen + wiki answer"
    brain_wiring.chloe_llm_call = fake_llm

    def restore():
        for k, v in sv_originals.items():
            setattr(screen_vision, k, v)
        brain_wiring.chloe_llm_call = original_llm

    return restore, captured_prompts


def t_usage_when_empty():
    from brain_wiring import handle_ask
    r = handle_ask("")
    assert "Usage" in r, f"expected usage hint, got: {r!r}"
    print("PASS: empty question -> usage hint")


def t_full_path_with_screen_and_wiki():
    """Happy path: capture works, vision describes, wiki has matches, heavy
    answer prompt is composed with both."""
    from brain_wiring import handle_ask
    restore, prompts = _patch(blocked=False, capture_ok=True, vision_ok=True)
    try:
        r = handle_ask("what is qmd and why am I getting a TypeError")
        assert r.startswith("FAKE_ANSWER"), f"unexpected reply: {r[:200]!r}"
        assert "(vision skipped" not in r
        assert len(prompts) == 1
        mode, prompt = prompts[0]
        assert mode == "heavy", f"expected heavy mode, got {mode}"
        assert "Screen (Code)" in prompt, f"app label missing in: {prompt[:300]}"
        assert "TypeError on line 42" in prompt, "screen description not in prompt"
        assert "qmd" in prompt.lower(), "wiki content not in prompt"
        print("PASS: full pipeline composes screen + wiki")
    finally:
        restore()


def t_blocklist_falls_back_to_brain_only():
    """When blocklist matches, vision is skipped but brain still answers."""
    from brain_wiring import handle_ask
    restore, prompts = _patch(blocked=True, capture_ok=True, vision_ok=True)
    try:
        r = handle_ask("what is qmd")
        assert "(vision skipped:" in r, f"missing skip note: {r!r}"
        assert "blocklist" in r, f"skip note should mention blocklist: {r!r}"
        # The heavy LLM was still called with brain-only prompt
        assert len(prompts) == 1
        mode, prompt = prompts[0]
        assert mode == "heavy"
        assert "Screen" not in prompt, "screen section should not be in fallback prompt"
        assert "qmd" in prompt.lower(), "wiki content missing in fallback prompt"
        print("PASS: blocklist falls back to brain-only with note")
    finally:
        restore()


def t_kill_switch_falls_back():
    from brain_wiring import handle_ask
    os.environ["CHLOE_VISION_DISABLED"] = "1"
    restore, prompts = _patch(blocked=False)
    try:
        r = handle_ask("what is qmd")
        assert "(vision skipped:" in r and "kill switch" in r, f"got: {r!r}"
        assert len(prompts) == 1
        mode, prompt = prompts[0]
        assert "Screen" not in prompt
        print("PASS: kill switch falls back to brain-only with note")
    finally:
        os.environ.pop("CHLOE_VISION_DISABLED", None)
        restore()


def t_capture_failure_falls_back():
    from brain_wiring import handle_ask
    restore, prompts = _patch(blocked=False, capture_ok=False)
    try:
        r = handle_ask("what is qmd")
        assert "(vision skipped:" in r and "capture failed" in r, f"got: {r!r}"
        assert len(prompts) == 1
        print("PASS: capture failure falls back to brain-only with note")
    finally:
        restore()


def t_no_wiki_no_screen_returns_clean_message():
    from brain_wiring import handle_ask
    restore, prompts = _patch(blocked=True)  # vision will be skipped
    try:
        # Question that won't match the seeded qmd page
        r = handle_ask("what is the airspeed velocity of an unladen swallow")
        # Vision skipped + no wiki match = "Nothing to answer from"
        assert "Nothing to answer from" in r, f"got: {r!r}"
        # No LLM call should have happened
        assert len(prompts) == 0, f"unexpected LLM calls: {prompts}"
        print("PASS: no vision + no wiki = clean refusal")
    finally:
        restore()


def main():
    t_usage_when_empty()
    t_full_path_with_screen_and_wiki()
    t_blocklist_falls_back_to_brain_only()
    t_kill_switch_falls_back()
    t_capture_failure_falls_back()
    t_no_wiki_no_screen_returns_clean_message()


if __name__ == "__main__":
    main()
