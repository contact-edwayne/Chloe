@echo off
REM ════════════════════════════════════════════════════════════════════════════
REM start_chloe_full.bat — launcher for Chloe (desktop HUD + mobile PWA)
REM ════════════════════════════════════════════════════════════════════════════
REM HOW TO RUN: normally you don't run this directly — double-click
REM start_chloe.vbs, which runs this script hidden. If you DO run it directly it
REM still works; it just shows its own console briefly. Either way the three
REM services below are launched as HIDDEN background processes (no windows).
REM
REM What this does:
REM   1. backend — `python start_jarvis.py`. Voice loop + WebSocket server on
REM                port 6789.                          Logs -> logs\backend.log
REM   2. static  — `python -m http.server 8080`. Static file server so the
REM                iPhone can load chloe-mobile.html.   Logs -> logs\static.log
REM   3. watcher — `python wiki_watcher.py`. Keeps wiki_pages embeddings in
REM                sync with disk.                      Logs -> logs\watcher.log
REM
REM   Each service runs with NO console window (run_hidden.vbs + svc.bat).
REM     Watch logs live:  show_chloe_logs.bat
REM     Stop Chloe:       stop_chloe.bat
REM
REM What this assumes is already running (set-and-forget services from the
REM mobile setup):
REM   - Tailscale itself (Windows service, auto-starts on boot)
REM   - tailscale serve proxies for / and /chloe-ws (configured once, persists)
REM   - Ollama (started hidden at login by the chloe_ollama_serve Startup
REM            shortcut; this script also launches it hidden on demand if it
REM            isn't responding — see the health check below)

cd /d "%~dp0"
if not exist logs mkdir logs

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
REM Quick sanity check — Ollama (needed by wiki watcher + brain light path)
REM ─────────────────────────────────────────────────────────────────────────
REM curl is built into Windows 10+. We accept a 2-second timeout so the
REM check doesn't stall the launcher when Ollama is genuinely missing.
echo Checking Ollama...
curl -s -o nul --max-time 2 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo   Ollama not responding on localhost:11434 - launching it hidden
    wscript "%~dp0run_hidden.vbs" "cmd /c %~dp0start_ollama.bat"
    REM Give Ollama ~3s to bind the port before the watcher starts hammering it
    timeout /t 3 /nobreak >nul
) else (
    echo   Ollama is up.
)

REM ─────────────────────────────────────────────────────────────────────────
REM 1. BACKEND
REM ─────────────────────────────────────────────────────────────────────────
REM Default: run from source via the venv. This always picks up your latest
REM jarvis.py edits without needing to rebuild. To launch the bundled exe
REM instead, edit the "backend" branch of svc.bat.
REM
REM Launched hidden via run_hidden.vbs -> svc.bat backend. Output goes to
REM logs\backend.log. The process is fully detached, so it survives this
REM launcher exiting.
wscript "%~dp0run_hidden.vbs" "cmd /c %~dp0svc.bat backend"

REM ─────────────────────────────────────────────────────────────────────────
REM 2. STATIC FILE SERVER for the mobile PWA
REM ─────────────────────────────────────────────────────────────────────────
wscript "%~dp0run_hidden.vbs" "cmd /c %~dp0svc.bat static"

REM ─────────────────────────────────────────────────────────────────────────
REM 3. WIKI WATCHER — keeps wiki_pages embeddings in sync with disk
REM ─────────────────────────────────────────────────────────────────────────
REM Polls C:\Chloe\brain\wiki every 2 seconds. Embeds Obsidian edits AND new
REM pages created by daily_ingest (8am). Idempotent on unchanged files via
REM hash compare, so it costs ~nothing when nothing has changed.
wscript "%~dp0run_hidden.vbs" "cmd /c %~dp0svc.bat watcher"

REM ─────────────────────────────────────────────────────────────────────────
REM (Optional desktop browser test — commented out by default. Uncomment if
REM you ever want to test the mobile UI on desktop without opening Chrome
REM manually. The iPhone doesn't need this — it loads the page straight from
REM Tailscale Serve. start_jarvis.py already opens the real desktop HUD via
REM PyQt6.)
REM
REM timeout /t 2 /nobreak >nul
REM start "" "http://localhost:8080/chloe-mobile.html?ws=ws://localhost:6789"

echo.
echo ════════════════════════════════════════════════════════════════════
echo  Chloe is starting up.
echo.
echo  Three background services were launched ^(no windows^):
echo    - backend  ^(WebSocket + voice loop^)    logs\backend.log
echo    - static   ^(static file server^)        logs\static.log
echo    - watcher  ^(wiki_pages embedding sync^)  logs\watcher.log
echo.
echo  The Chloe desktop HUD will appear in a few seconds.
echo.
echo  Desktop URL: http://localhost:8080/chloe-mobile.html?ws=ws://localhost:6789
echo  iPhone URL:  https://desktop-lgv51k8.tail4c6ace.ts.net/chloe-mobile.html
echo.
echo  Watch logs live: show_chloe_logs.bat
echo  Stop Chloe:      stop_chloe.bat
echo ════════════════════════════════════════════════════════════════════

REM No `pause` here — this script is normally run hidden (via start_chloe.vbs)
REM and a hidden `pause` would hang forever waiting for a keypress nobody can
REM give. Exit cleanly; the three services are already detached and running.
exit /b 0
