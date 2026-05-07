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


# ─── HUD WINDOW ──────────────────────────────────────────────────────────────
class Page(QWebEnginePage):
    """Forwards JS console messages to our redirected stdout so they show
    up in chloe.log when running as exe."""
    def javaScriptConsoleMessage(self, level, message, line, source):
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

settings = view.settings()
settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

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
