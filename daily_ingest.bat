@echo off
REM daily_ingest.bat - run by Windows Task Scheduler at 8:00 AM daily.
REM Ingests yesterday's Obsidian daily note (C:\Chloe\brain\wiki\daily\<date>.md)
REM into Chloe's brain via BRAIN.ingest().
REM
REM Scheduled to run after daily_context.bat (6am) so the 6am context
REM synthesis already wrote, and the 8am job folds yesterday's hand-typed
REM material into the entity/concept extraction pipeline.
REM
REM Stdout/stderr captured into logs/daily_ingest.log for debugging.
REM
REM Pass-through args:
REM     daily_ingest.bat --dry-run            (show plan, no writes)
REM     daily_ingest.bat --date 2026-05-09    (backfill specific date)

set JARVIS=%~dp0
cd /d "%JARVIS%"

if not exist "%JARVIS%logs" mkdir "%JARVIS%logs"

REM Use venv if present
if exist "%JARVIS%.venv\Scripts\python.exe" (
    "%JARVIS%.venv\Scripts\python.exe" daily_ingest.py %* >> "%JARVIS%logs\daily_ingest.log" 2>&1
) else if exist "%JARVIS%venv\Scripts\python.exe" (
    "%JARVIS%venv\Scripts\python.exe" daily_ingest.py %* >> "%JARVIS%logs\daily_ingest.log" 2>&1
) else (
    python daily_ingest.py %* >> "%JARVIS%logs\daily_ingest.log" 2>&1
)

exit /b %ERRORLEVEL%
