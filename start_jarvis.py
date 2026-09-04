"""
start_jarvis.py — Launches the Chloe HUD window + WebSocket backend.

Run modes:
  - Dev:    python start_jarvis.py
  - Frozen: double-click Chloe.exe (built from Jarvis.spec)

When frozen, sys.executable's directory is treated as the "app directory"
where user data lives (_env, facts.md, chloe_memory.db, models/, sounds/,
kokoro_models/, etc). Bundled read-only resources (Python source, hud.html,
openwakeword model files) are extracted to sys._MEIPASS by PyInstaller.
"""

import threading
import time
import sys
import os
import asyncio
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage, QWebEngineProfile

# ─── PATH RESOLUTION ─────────────────────────────────────────────────────────
# Two distinct directories matter when running as a frozen exe:
#   bundled_dir  — where PyInstaller extracted code + bundled resources
#   app_dir      — where the exe actually lives, where user files belong
# In dev mode they're the same (the project folder).
if getattr(sys, "frozen", False):
    bundled_dir = Path(sys._MEIPASS)
    app_dir     = Path(sys.executable).resolve().parent
else:
    bundled_dir = Path(__file__).resolve().parent
    app_dir     = bundled_dir

# All user-data paths (env files, facts, memory db, models, sounds) resolve
# against the cwd in jarvis.py, so we set cwd = app_dir.
os.chdir(app_dir)

# ─── LOG REDIRECT (frozen / no-console mode) ─────────────────────────────────
# With console=False in the .spec, prints have nowhere to go and a
# crash-on-import would leave the user staring at a closed window.
# Redirect stdout/stderr to chloe.log next to the exe so problems are
# diagnosable.
if getattr(sys, "frozen", False):
    try:
        log_path = app_dir / "chloe.log"
        # line-buffered so log lines flush as they happen
        _log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _log_fp
        sys.stderr = _log_fp
        print(f"\n=== Chloe started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"app_dir={app_dir}  bundled_dir={bundled_dir}")
    except Exception:
        # If redirect fails, just continue without one — better than crashing.
        pass

# ─── BACKEND BOOTSTRAP ───────────────────────────────────────────────────────
# WebSocket server starts in a daemon thread; jarvis (voice loop) starts
# in another after a brief delay to let the WS server bind its port.
from hud_server import start_server


def run_hud_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_server())


threading.Thread(target=run_hud_server, daemon=True).start()
time.sleep(1)


def run_jarvis():
    time.sleep(3)
    import jarvis  # noqa: F401  side-effect import: starts voice thread


threading.Thread(target=run_jarvis, daemon=True).start()


def run_catchup_sweep():
    # Run scheduled jobs that missed their window while the PC was off/asleep
    # (Windows Task Scheduler silently no-ops those). Delay first so Ollama, the
    # WS server, and the memory DB are warm. After the initial boot sweep, enter
    # the periodic loop — which re-sweeps on a timer AND right after the machine
    # resumes from sleep (the usual case here: the PC sleeps with Chloe running).
    time.sleep(90)
    try:
        import chloe_jobs
        fired = chloe_jobs.run_catchup()
        print(f"[catchup] boot sweep fired: {fired}")
        chloe_jobs.run_periodic_catchup()  # blocks this daemon thread forever
    except Exception as e:
        print(f"[catchup] sweep failed: {e}")


threading.Thread(target=run_catchup_sweep, daemon=True).start()


# ─── HUD WINDOW ──────────────────────────────────────────────────────────────
class Page(QWebEnginePage):
    """Forwards JS console messages to our redirected stdout so they show
    up in chloe.log when running as exe."""
    def javaScriptConsoleMessage(self, level, message, line, source):
        # "ResizeObserver loop completed with undelivered notifications" is a
        # benign Chromium warning — it fires when a resize callback can't deliver
        # every notification within one frame. EmulatorJS + responsive layouts
        # emit it in a flood; drop it so it doesn't bury chloe.log.
        if message and "ResizeObserver loop" in message:
            return
        print(f"JS: {message}")


app = QApplication(sys.argv)
app.setApplicationName("CHLOE")

window = QMainWindow()
window.setWindowTitle("CHLOE")
# Default size matches the HUD design canvas (1100x760) plus a small
# allowance for window chrome. Below this size the .jr frame goes fluid
# and the chat panel auto-fits, so a smaller window is still usable.
window.resize(1140, 800)
window.setMinimumSize(880, 600)

view = QWebEngineView()
view.setPage(Page(view))

profile = QWebEngineProfile.defaultProfile()
profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
# Give the profile a persistent storage path so EmulatorJS save states / SRAM
# (IndexedDB + localStorage) actually survive — without one, browser storage
# can be ephemeral and save state silently no-ops.
try:
    profile.setPersistentStoragePath(str(app_dir / "webdata"))
except Exception:
    pass

settings = view.settings()
settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

# Fullscreen + storage + downloads for embedded panels (EmulatorJS, etc.).
# QWebEngine disables fullscreen by default and ignores download requests unless
# we opt in — without these the emulator's fullscreen button does nothing and
# "save state to file" / screenshot exports silently fail.
settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)


def _on_fullscreen_requested(req):
    req.accept()
    window.showFullScreen() if req.toggleOn() else window.showNormal()


view.page().fullScreenRequested.connect(_on_fullscreen_requested)


def _on_download_requested(item):
    # Accept downloads (EmulatorJS save-state exports, screenshots) so they
    # actually write instead of being silently dropped.
    try:
        item.accept()
    except Exception:
        pass


try:
    profile.downloadRequested.connect(_on_download_requested)
except Exception:
    pass

# hud.html is a bundled resource (shipped inside the exe), so it lives in
# bundled_dir even when frozen. Loading it via setHtml(content, base_url)
# rather than setUrl(...) avoids file-system permission quirks inside
# QWebEngine's sandboxed renderer.
html_path = bundled_dir / "hud.html"
print(f"Loading HUD from: {html_path}")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()
print(f"HUD file size: {len(html_content)} chars")
view.setHtml(html_content, QUrl("http://localhost/"))

window.setCentralWidget(view)
window.show()

sys.exit(app.exec())
