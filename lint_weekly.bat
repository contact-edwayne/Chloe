@echo off
REM ─────────────────────────────────────────────────────────────────────
REM lint_weekly.bat — invoked by Task Scheduler weekly.
REM Activates the venv, runs lint_weekly.py, appends output to
REM lint_weekly.log so you can review on Mondays.
REM ─────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
call venv\Scripts\activate.bat
python lint_weekly.py >> lint_weekly.log 2>&1
