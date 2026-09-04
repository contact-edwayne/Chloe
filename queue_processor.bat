@echo off
REM queue_processor.bat - run by Windows Task Scheduler every 2 hours.
REM Drains C:\Chloe\brain\queue\*.md, writes outputs to brain\generated\<date>\,
REM moves processed files to brain\archive\queue\.
REM Stdout/stderr captured into logs/queue_processor.log.

set JARVIS=%~dp0
cd /d "%JARVIS%"

if not exist "%JARVIS%logs" mkdir "%JARVIS%logs"

if exist "%JARVIS%.venv\Scripts\python.exe" (
    "%JARVIS%.venv\Scripts\python.exe" queue_processor.py >> "%JARVIS%logs\queue_processor.log" 2>&1
) else if exist "%JARVIS%venv\Scripts\python.exe" (
    "%JARVIS%venv\Scripts\python.exe" queue_processor.py >> "%JARVIS%logs\queue_processor.log" 2>&1
) else (
    python queue_processor.py >> "%JARVIS%logs\queue_processor.log" 2>&1
)

exit /b %ERRORLEVEL%
