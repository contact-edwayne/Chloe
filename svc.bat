@echo off
REM ============================================================================
REM svc.bat - internal: run ONE Chloe service in this (hidden) console.
REM ============================================================================
REM Not meant to be double-clicked. start_chloe_full.bat invokes this through
REM run_hidden.vbs so each service runs with no window. All output is
REM redirected to logs\<name>.log (use show_chloe_logs.bat to watch them live).
REM
REM   svc.bat backend   -> venv + python start_jarvis.py    -> logs\backend.log
REM   svc.bat static    -> python -m http.server 8080       -> logs\static.log
REM   svc.bat watcher   -> venv + python wiki_watcher.py    -> logs\watcher.log
REM ============================================================================

cd /d "%~dp0"
if not exist logs mkdir logs

REM Force Python to UTF-8 for stdout/stderr and locale-defaulted I/O.
REM Without this, Python's stdout (redirected to logs\*.log in this setup)
REM falls back to the system's locale-preferred encoding, which on a non-
REM UTF-8 locale (e.g. CP1251) can't represent the Unicode chars used in
REM Chloe's diagnostic prints (->, ..., -). This bites only in the windowless
REM setup; under the old cmd /k path the console handled it. PYTHONUTF8=1 is
REM the broad PEP 540 fix; PYTHONIOENCODING is the targeted I/O knob.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if /i "%~1"=="backend" (
    call venv\Scripts\activate.bat
    echo ==================================================================>  logs\backend.log
    echo  Chloe Backend started %date% %time%                              >> logs\backend.log
    echo ==================================================================>> logs\backend.log
    python start_jarvis.py >> logs\backend.log 2>&1
    goto :eof
)

if /i "%~1"=="static" (
    echo ==================================================================>  logs\static.log
    echo  Chloe Static server started %date% %time%                        >> logs\static.log
    echo ==================================================================>> logs\static.log
    python -m http.server 8080 >> logs\static.log 2>&1
    goto :eof
)

if /i "%~1"=="watcher" (
    call venv\Scripts\activate.bat
    echo ==================================================================>  logs\watcher.log
    echo  Chloe Watcher started %date% %time%                              >> logs\watcher.log
    echo ==================================================================>> logs\watcher.log
    python wiki_watcher.py >> logs\watcher.log 2>&1
    goto :eof
)

echo svc.bat: unknown service "%~1"  (expected: backend ^| static ^| watcher)
exit /b 1
