@echo off
REM ════════════════════════════════════════════════════════════════════════════
REM start_chloe_full.bat — one-click launcher for Chloe (desktop HUD + mobile PWA)
REM ════════════════════════════════════════════════════════════════════════════
REM What this does:
REM   1. Opens "Chloe Backend"  — runs `python start_jarvis.py` (or Chloe.exe
REM                                if you flip the switch below). This is the
REM                                voice loop + WebSocket server on port 6789.
REM   2. Opens "Chloe Static"   — runs `python -m http.server 8080`. Static
REM                                file server so the iPhone can load
REM                                chloe-mobile.html via Tailscale Serve.
REM   3. Optionally opens the desktop test URL in your default browser.
REM
REM What this assumes is already running (set-and-forget services from the
REM mobile setup):
REM   - Tailscale itself (Windows service, auto-starts on boot)
REM   - tailscale serve proxies for / and /chloe-ws (configured once, persists)
REM   - Ollama (Windows service, auto-starts)
REM
REM To stop Chloe: close the "Chloe Backend" and "Chloe Static" windows.

cd /d "%~dp0"

REM ─────────────────────────────────────────────────────────────────────────
REM Quick sanity check — Tailscale Serve config
REM ─────────────────────────────────────────────────────────────────────────
echo Checking Tailscale Serve config...
tailscale serve status >nul 2>&1
if errorlevel 1 (
    echo   WARNING: tailscale serve isn't responding. Mobile won't reach the PC
    echo   until you re-run:
    echo     tailscale serve --bg http://localhost:8080
    echo     tailscale serve --bg --set-path=/chloe-ws http://localhost:6789
    echo.
)

REM ─────────────────────────────────────────────────────────────────────────
REM 1. BACKEND
REM ─────────────────────────────────────────────────────────────────────────
REM Default: run from source via the venv. This always picks up your latest
REM jarvis.py edits without needing to rebuild.
REM
REM Once you've rebuilt with build.bat and want to launch the bundled exe
REM instead (no terminal window, faster startup), comment out the python
REM line below and uncomment the .exe line.

start "Chloe Backend" cmd /k "venv\Scripts\activate.bat && python start_jarvis.py"
REM start "Chloe Backend" "dist\Chloe\Chloe.exe"

REM ─────────────────────────────────────────────────────────────────────────
REM 2. STATIC FILE SERVER for the mobile PWA
REM ─────────────────────────────────────────────────────────────────────────
start "Chloe Static" cmd /k "python -m http.server 8080"

REM ─────────────────────────────────────────────────────────────────────────
REM (Optional desktop browser test — commented out by default. Uncomment
REM if you ever want to test the mobile UI on desktop without opening
REM Chrome manually. The iPhone doesn't need this — it loads the page
REM straight from Tailscale Serve. start_jarvis.py already opens the
REM real desktop HUD via PyQt6.)
REM
REM timeout /t 2 /nobreak >nul
REM start "" "http://localhost:8080/chloe-mobile.html?ws=ws://localhost:6789"

echo.
echo ════════════════════════════════════════════════════════════════════
echo  Chloe is starting up.
echo.
echo  Two new windows are opening:
echo    - "Chloe Backend"  ^(WebSocket + voice loop, watch this for logs^)
echo    - "Chloe Static"   ^(static file server for the mobile PWA^)
echo.
echo  Desktop URL: http://localhost:8080/chloe-mobile.html?ws=ws://localhost:6789
echo  iPhone URL:  https://desktop-lgv51k8.tail4c6ace.ts.net/chloe-mobile.html
echo.
echo  To stop: close those two windows.
echo  This launcher window can be closed any time.
echo ════════════════════════════════════════════════════════════════════
echo.
pause
