@echo off
REM wiki_watcher.bat - keep Chloe's wiki embeddings fresh.
REM
REM Watches C:\Chloe\brain\wiki\**\*.md. When you edit a page in Obsidian
REM (or any editor) and save, the corresponding embedding gets refreshed
REM within ~2 seconds so /wiki <query> reflects the change.
REM
REM Run from cmd:
REM     wiki_watcher.bat
REM
REM Pass-through args:
REM     wiki_watcher.bat --once          (backfill only, then exit)
REM     wiki_watcher.bat --rebuild       (force re-embed everything)
REM     wiki_watcher.bat --interval 5    (slower polling)
REM
REM Logs stream to the console. Ctrl-C exits cleanly.

set JARVIS=%~dp0
cd /d "%JARVIS%"

if not exist "%JARVIS%logs" mkdir "%JARVIS%logs"

REM Use venv if present
if exist "%JARVIS%.venv\Scripts\python.exe" (
    "%JARVIS%.venv\Scripts\python.exe" wiki_watcher.py %*
) else if exist "%JARVIS%venv\Scripts\python.exe" (
    "%JARVIS%venv\Scripts\python.exe" wiki_watcher.py %*
) else (
    python wiki_watcher.py %*
)
