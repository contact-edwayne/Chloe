@echo off
REM daily_context.bat - run by Windows Task Scheduler at 6:00 AM daily.
REM Generates Edward's daily synthesis: episodic/CONTEXT-<today>.md
REM Stdout/stderr captured into logs/daily_context.log for debugging.

set JARVIS=%~dp0
cd /d "%JARVIS%"

if not exist "%JARVIS%logs" mkdir "%JARVIS%logs"

REM Use venv if present
if exist "%JARVIS%.venv\Scripts\python.exe" (
    "%JARVIS%.venv\Scripts\python.exe" daily_context.py >> "%JARVIS%logs\daily_context.log" 2>&1
) else if exist "%JARVIS%venv\Scripts\python.exe" (
    "%JARVIS%venv\Scripts\python.exe" daily_context.py >> "%JARVIS%logs\daily_context.log" 2>&1
) else (
    python daily_context.py >> "%JARVIS%logs\daily_context.log" 2>&1
)

exit /b %ERRORLEVEL%
