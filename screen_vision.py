"""screen_vision.py — on-demand screen capture + vision description.

Wired into jarvis.py via the /see command in brain_wiring.py.

Capture path:
    1. Kill switch CHLOE_VISION_DISABLED=1 short-circuits everything.
    2. Detect foreground window (Win32 GetForegroundWindow + GetWindowText).
    3. Frontmost-app blocklist check against CHLOE_VISION_BLOCKLIST
       (comma-separated, case-insensitive substring match on window title or
       executable basename). Refuses cleanly if matched.
    4. Find the monitor containing that window (MonitorFromWindow), capture
       it via mss, encode to PNG bytes.
    5. Send to Groq llama-4-scout with the user's prompt (or a default).

Returns a dict {ok, text, app, error, blocked_by, model} so the caller can
shape the chat reply. Default blocklist is intentionally empty — Edward sets
the env var himself.

Hard deps:
    pip install mss pywin32

mss is cross-platform; pywin32 is Windows-only. The win32 imports are guarded
so this module loads on non-Windows for testing, but get_frontmost_app()
returns a "not supported" stub off Windows.
"""

import base64
import io
import os
import sys
from pathlib import Path

# ─── Optional Windows imports (guarded) ──────────────────────────────────────
_WIN32_AVAILABLE = False
try:
    if sys.platform.startswith("win"):
        import win32gui
        import win32process
        import win32api
        import win32con  # noqa: F401  (loaded for completeness)
        try:
            import psutil
        except ImportError:
            psutil = None  # exe-name fallback off, still works via window title
        _WIN32_AVAILABLE = True
except ImportError as _e:
    print(f"[vision] pywin32 not importable: {_e}", flush=True)


# ─── Groq client (lazy, mirrors brain_wiring.py pattern) ─────────────────────
try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment]

MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"

_groq = None
_groq_attempted = False


def _get_groq():
    """Lazy Groq init. Reads GROQ_API_KEY at first call so .env has loaded."""
    global _groq, _groq_attempted
    if _groq is not None:
        return _groq
    if _groq_attempted:
        return None
    _groq_attempted = True
    if Groq is None:
        print("[vision] groq package not installed", flush=True)
        return None
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("[vision] no GROQ_API_KEY in env", flush=True)
        return None
    _groq = Groq(api_key=key)
    print("[vision] Groq client initialized lazily", flush=True)
    return _groq


# ─── Foreground window detection ─────────────────────────────────────────────
def get_frontmost_app() -> dict:
    """Return info about the foreground window.

    {
      'ok': bool,
      'title': str,        # window title text
      'exe': str,          # process executable basename (e.g. 'chrome.exe')
      'hwnd': int,         # window handle
      'rect': (x, y, w, h) # window bounds in virtual-screen coords
    }

    On non-Windows or if win32 calls fail, returns {'ok': False, 'error': ...}.
    """
    if not _WIN32_AVAILABLE:
        return {"ok": False, "error": "pywin32 unavailable (non-Windows or not installed)"}
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return {"ok": False, "error": "no foreground window"}
        title = win32gui.GetWindowText(hwnd) or ""
        exe = ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if "psutil" in globals() and psutil is not None and pid:
                exe = Path(psutil.Process(pid).exe()).name
        except Exception:
            pass
        # Window rect (left, top, right, bottom) in screen coords
        try:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            rect = (l, t, r - l, b - t)
        except Exception:
            rect = (0, 0, 0, 0)
        return {"ok": True, "title": title, "exe": exe, "hwnd": int(hwnd), "rect": rect}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _monitor_for_window(sct, hwnd: int) -> dict:
    """Pick the mss monitor dict whose rectangle contains the window's center.

    sct is an open mss.mss() instance; hwnd is the window handle. Returns the
    monitor dict (with left/top/width/height keys). Falls back to monitor[1]
    (primary) if the window center can't be located.
    """
    monitors = sct.monitors  # [0]=virtual screen union, [1..]=individual
    try:
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        cx, cy = (l + r) // 2, (t + b) // 2
        for m in monitors[1:]:
            mx, my, mw, mh = m["left"], m["top"], m["width"], m["height"]
            if mx <= cx < mx + mw and my <= cy < my + mh:
                return m
    except Exception:
        pass
    # Fallback: primary monitor
    return monitors[1] if len(monitors) > 1 else monitors[0]


# ─── Privacy gate ────────────────────────────────────────────────────────────
def _blocklist_tokens() -> list:
    raw = os.environ.get("CHLOE_VISION_BLOCKLIST", "").strip()
    if not raw:
        return []
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def is_blocked(app_info: dict) -> str:
    """Return the matching blocklist token if frontmost app is blocked,
    else empty string. Substring-matches against window title and exe name.
    """
    tokens = _blocklist_tokens()
    if not tokens:
        return ""
    title = (app_info.get("title") or "").lower()
    exe = (app_info.get("exe") or "").lower()
    haystack = f"{title} {exe}"
    for t in tokens:
        if t in haystack:
            return t
    return ""


# ─── Capture ─────────────────────────────────────────────────────────────────
def capture_screen(monitor_index: int = None) -> dict:
    """Capture screen and return PNG bytes.

    monitor_index:
        None → auto: monitor containing the foreground window
        0    → all monitors stitched (mss virtual-screen union)
        1+   → specific monitor by mss index

    Returns {'ok': bool, 'png': bytes, 'monitor': dict, 'app': dict, 'error': str?}.
    """
    try:
        import mss  # type: ignore
    except ImportError:
        return {"ok": False, "error": "mss not installed — pip install mss"}

    app = get_frontmost_app() if monitor_index is None else {"ok": False}

    try:
        with mss.mss() as sct:
            if monitor_index is None and app.get("ok"):
                mon = _monitor_for_window(sct, app["hwnd"])
            elif monitor_index is None:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            else:
                if monitor_index >= len(sct.monitors):
                    return {"ok": False, "error": f"monitor {monitor_index} not found "
                                                   f"({len(sct.monitors) - 1} attached)"}
                mon = sct.monitors[monitor_index]
            shot = sct.grab(mon)
            # Encode to PNG via Pillow if available, else mss's tools
            try:
                from PIL import Image
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()
            except ImportError:
                # mss has its own PNG encoder
                import mss.tools as mt
                png_bytes = mt.to_png(shot.rgb, shot.size)
            return {"ok": True, "png": png_bytes,
                    "monitor": dict(mon), "app": app}
    except Exception as e:
        return {"ok": False, "error": f"capture failed: {type(e).__name__}: {e}"}


# ─── Vision call ─────────────────────────────────────────────────────────────
DEFAULT_PROMPT = (
    "Describe what's on this screen concisely. Identify the app or website, "
    "the visible content, and any obvious context about what the user is doing. "
    "Be specific — name UI elements and visible text rather than vague summary. "
    "If the screen contains code, summarize what it does. Skip preamble."
)


ANTI_HALLUCINATION_SUFFIX = (
    "\n\nIMPORTANT: Answer based ONLY on what is visible in the image. If the "
    "image does not contain the information needed to answer, say so plainly "
    "(e.g. \"no code editor is visible\", \"I can't see the file name\"). "
    "Do not guess filenames, contents, or context that aren't on screen."
)


def describe_screen(image_bytes: bytes, prompt: str = "") -> dict:
    """Send PNG bytes to Groq Llama 4 Scout and return its description.

    Returns {'ok': bool, 'text': str, 'model': str, 'error': str?}.
    """
    client = _get_groq()
    if client is None:
        return {"ok": False, "error": "Groq client unavailable (no GROQ_API_KEY or groq pkg missing)"}
    user_prompt_raw = (prompt or "").strip()
    if user_prompt_raw:
        # User asked a specific question — append the anti-hallucination clause
        # so the model says "not visible" instead of inventing detail.
        user_prompt = user_prompt_raw + ANTI_HALLUCINATION_SUFFIX
    else:
        user_prompt = DEFAULT_PROMPT
    b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        resp = client.with_options(timeout=60.0).chat.completions.create(
            model=MODEL_VISION,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=900,
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "text": text, "model": MODEL_VISION}
    except Exception as e:
        return {"ok": False, "error": f"vision call failed: {type(e).__name__}: {e}",
                "model": MODEL_VISION}


# ─── Orchestrator (the thing /see calls) ─────────────────────────────────────
def see(prompt: str = "") -> dict:
    """End-to-end: kill switch → frontmost → blocklist → capture → describe.

    Returns:
        {'ok': bool, 'text': str, 'app': dict, 'monitor': dict,
         'blocked_by': str, 'error': str?}
    """
    if os.environ.get("CHLOE_VISION_DISABLED", "").strip() == "1":
        return {"ok": False, "error": "screen vision is disabled "
                                       "(CHLOE_VISION_DISABLED=1)"}

    app = get_frontmost_app()
    if app.get("ok"):
        match = is_blocked(app)
        if match:
            return {"ok": False, "blocked_by": match, "app": app,
                    "error": f"refusing to capture — frontmost app matches "
                             f"blocklist token '{match}' "
                             f"(title={app.get('title','?')[:60]}, "
                             f"exe={app.get('exe','?')})"}
    cap = capture_screen(monitor_index=None)
    if not cap.get("ok"):
        return {"ok": False, "error": cap.get("error", "capture failed"), "app": app}

    desc = describe_screen(cap["png"], prompt=prompt)
    if not desc.get("ok"):
        return {"ok": False, "error": desc.get("error"), "app": app,
                "monitor": cap.get("monitor")}

    return {
        "ok": True,
        "text": desc["text"],
        "app": app,
        "monitor": cap.get("monitor"),
        "model": desc.get("model"),
    }


# ─── CLI smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    # Quick smoke test: capture only (no Groq call) unless --vision is passed.
    if "--app" in sys.argv:
        print(json.dumps(get_frontmost_app(), indent=2, default=str))
        sys.exit(0)
    if "--capture" in sys.argv:
        r = capture_screen()
        if r.get("ok"):
            out = Path("vision_test.png")
            out.write_bytes(r["png"])
            print(f"captured {len(r['png'])} bytes to {out.resolve()}")
            print(f"monitor: {r['monitor']}")
            print(f"app: {r['app'].get('title','?')} ({r['app'].get('exe','?')})")
        else:
            print(f"FAIL: {r.get('error')}")
        sys.exit(0)
    if "--vision" in sys.argv:
        prompt = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
        r = see(prompt)
        print(json.dumps({k: v for k, v in r.items() if k != "monitor"},
                         indent=2, default=str))
        sys.exit(0)
    print("usage: python screen_vision.py [--app|--capture|--vision] [prompt...]")
