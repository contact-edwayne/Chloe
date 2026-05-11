"""ambient_vision.py — periodic screen capture → 1-line summary → brain.episodic_append.

Started by `/see ambient on [N]` from brain_wiring.py. Runs on a daemon
thread so it dies cleanly when jarvis.py exits.

Each tick:
    1. Honor CHLOE_VISION_DISABLED kill switch (skip the tick).
    2. Detect foreground window via screen_vision.get_frontmost_app.
    3. If frontmost matches CHLOE_VISION_BLOCKLIST → skip silently. No
       episodic entry, no leak of the blocked app name.
    4. Capture current monitor and call vision model with a tight 1-line
       prompt (TIGHT_PROMPT below — caps the output length).
    5. Append a single line to today's episodic file via BRAIN.episodic_append:
           [HH:MM] (added by Brain) + "[<app>] <description>"

Loop control is via threading.Event so stop() returns immediately even mid-wait.

Public API:
    start(minutes=None) -> dict   # idempotent — second call updates cadence
    stop() -> dict                # safe to call when not running
    status() -> dict              # {running, minutes, last_tick, last_text,
                                  #  ticks_total, ticks_skipped_blocked,
                                  #  ticks_failed}
"""

import os
import threading
import time
import datetime
from typing import Optional


# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MINUTES = 10  # overridden by CHLOE_VISION_AMBIENT_MINUTES env var

# Tight prompt for the 1-line summary. Chosen to:
#   - cap length (one short sentence, ~20 words)
#   - name the app + task specifically
#   - bail clean when the screen is idle/locked
#   - skip the model's usual preamble
TIGHT_PROMPT = (
    "In ONE short sentence (max 20 words), describe what the user is doing. "
    "Name the specific app or website and the specific task or content. "
    "If the screen is locked, idle, or shows only a desktop, just say "
    "'idle desktop' or 'screen locked'. "
    "No preamble, no markdown, no quotes, no bullet points — return only "
    "the sentence."
)


# ─── State (module-level singleton, thread-safe) ─────────────────────────────
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_minutes: float = DEFAULT_MINUTES
_started_at: Optional[datetime.datetime] = None
_last_tick: Optional[datetime.datetime] = None
_last_text: str = ""
_ticks_total: int = 0
_ticks_skipped_blocked: int = 0
_ticks_skipped_disabled: int = 0
_ticks_failed: int = 0
_last_error: str = ""


def _read_env_minutes() -> float:
    raw = os.environ.get("CHLOE_VISION_AMBIENT_MINUTES", "").strip()
    if not raw:
        return DEFAULT_MINUTES
    try:
        v = float(raw)
        return max(0.05, v)  # 3-second floor, prevents accidentally tight loops
    except ValueError:
        return DEFAULT_MINUTES


# ─── Tick logic (the per-iteration work) ─────────────────────────────────────
def _format_app_label(app: dict) -> str:
    """Compact app identifier for the episodic line. Prefers exe basename
    (without .exe) over the full window title, which is often noisy.
    """
    exe = (app.get("exe") or "").strip()
    if exe.lower().endswith(".exe"):
        exe = exe[:-4]
    if exe:
        return exe
    title = (app.get("title") or "").strip()
    return title[:40] if title else "unknown"


def _do_tick(brain) -> dict:
    """One ambient capture-describe-log cycle. Returns a result dict for
    state updates. Doesn't raise — all errors are caught and reported.
    """
    global _ticks_total, _ticks_skipped_blocked, _ticks_skipped_disabled
    global _ticks_failed, _last_text, _last_error, _last_tick

    _last_tick = datetime.datetime.now()
    _ticks_total += 1

    # Kill switch
    if os.environ.get("CHLOE_VISION_DISABLED", "").strip() == "1":
        _ticks_skipped_disabled += 1
        return {"action": "skip_disabled"}

    try:
        # Late import — keeps module light at jarvis startup time
        from screen_vision import (
            get_frontmost_app, is_blocked, capture_screen, describe_screen,
        )
    except Exception as e:
        _ticks_failed += 1
        _last_error = f"import: {type(e).__name__}: {e}"
        return {"action": "fail", "error": _last_error}

    app = get_frontmost_app()
    if app.get("ok") and is_blocked(app):
        # Silent skip — do NOT write the blocked app name to episodic memory.
        _ticks_skipped_blocked += 1
        return {"action": "skip_blocked"}

    cap = capture_screen()
    if not cap.get("ok"):
        _ticks_failed += 1
        _last_error = f"capture: {cap.get('error')}"
        return {"action": "fail", "error": _last_error}

    desc = describe_screen(cap["png"], prompt=TIGHT_PROMPT)
    if not desc.get("ok"):
        _ticks_failed += 1
        _last_error = f"vision: {desc.get('error')}"
        return {"action": "fail", "error": _last_error}

    text = (desc.get("text") or "").strip()
    # Strip the model's occasional leading/trailing quotes
    text = text.strip('"').strip("'").strip()
    if not text:
        _ticks_failed += 1
        _last_error = "vision: empty text"
        return {"action": "fail", "error": _last_error}

    label = _format_app_label(app)
    line = f"[{label}] {text}"
    _last_text = line

    try:
        brain.episodic_append(line)
    except Exception as e:
        _ticks_failed += 1
        _last_error = f"episodic_append: {type(e).__name__}: {e}"
        return {"action": "fail", "error": _last_error}

    return {"action": "logged", "text": line}


# ─── Loop runner ─────────────────────────────────────────────────────────────
def _runner(brain, stop_event: threading.Event):
    """Worker function for the daemon thread. Sleeps on the event so stop()
    interrupts the wait immediately. First tick happens IMMEDIATELY (not
    after a wait) so users see something in the log without delaying.
    """
    print(f"[ambient] loop started, cadence={_minutes:.2f} min", flush=True)
    while not stop_event.is_set():
        try:
            r = _do_tick(brain)
            print(f"[ambient] tick: {r.get('action')}"
                  + (f" — {r.get('text','')[:80]}" if r.get('action') == 'logged' else "")
                  + (f" — {r.get('error','')}" if r.get('action') == 'fail' else ""),
                  flush=True)
        except Exception as e:
            global _ticks_failed, _last_error
            _ticks_failed += 1
            _last_error = f"runner: {type(e).__name__}: {e}"
            print(f"[ambient] runner error: {_last_error}", flush=True)
        # Wait until next tick or until stop. .wait returns True if event set.
        if stop_event.wait(timeout=_minutes * 60):
            break
    print("[ambient] loop stopped", flush=True)


# ─── Public API ──────────────────────────────────────────────────────────────
def start(minutes: float = None) -> dict:
    """Start (or restart with new cadence) the ambient loop.

    If the loop is already running and a new cadence is supplied, the loop is
    stopped and restarted with the new value. Returns status().
    """
    global _thread, _stop_event, _minutes, _started_at
    global _ticks_total, _ticks_skipped_blocked, _ticks_skipped_disabled, _ticks_failed
    with _lock:
        new_minutes = float(minutes) if minutes is not None else _read_env_minutes()
        new_minutes = max(0.05, new_minutes)
        # If running, stop first so the new cadence applies
        if _thread is not None and _thread.is_alive():
            if _stop_event is not None:
                _stop_event.set()
            _thread.join(timeout=2.0)
        # Reset counters on a fresh start
        _ticks_total = 0
        _ticks_skipped_blocked = 0
        _ticks_skipped_disabled = 0
        _ticks_failed = 0
        _minutes = new_minutes
        _started_at = datetime.datetime.now()
        _stop_event = threading.Event()
        # Late import of BRAIN to avoid circular import at module-load time
        from brain_wiring import BRAIN
        _thread = threading.Thread(target=_runner, args=(BRAIN, _stop_event),
                                   name="chloe-ambient-vision", daemon=True)
        _thread.start()
    return status()


def stop() -> dict:
    """Stop the loop. Safe to call when not running."""
    global _thread, _stop_event
    with _lock:
        if _thread is None or not _thread.is_alive():
            return {"running": False, "note": "not running"}
        if _stop_event is not None:
            _stop_event.set()
        _thread.join(timeout=2.0)
        was_alive = _thread.is_alive()
        _thread = None
        _stop_event = None
    return {"running": False, "stopped_cleanly": not was_alive}


def status() -> dict:
    with _lock:
        running = _thread is not None and _thread.is_alive()
        return {
            "running": running,
            "minutes": _minutes,
            "started_at": _started_at.isoformat() if _started_at else None,
            "last_tick": _last_tick.isoformat() if _last_tick else None,
            "last_text": _last_text,
            "ticks_total": _ticks_total,
            "ticks_skipped_blocked": _ticks_skipped_blocked,
            "ticks_skipped_disabled": _ticks_skipped_disabled,
            "ticks_failed": _ticks_failed,
            "last_error": _last_error,
        }
