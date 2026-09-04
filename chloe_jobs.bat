@echo off
REM Windowless wrapper for chloe_jobs.py. Invoked by Windows Task Scheduler.
REM Usage: chloe_jobs.bat <job-name>

setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

REM Prefer venv_py314 (the active Chloe venv) so dependency resolution
REM matches the live process. Fall back to system python if the venv is
REM missing.
set PY=
if exist "%~dp0venv_py314\Scripts\python.exe" set PY=%~dp0venv_py314\Scripts\python.exe
if "%PY%"=="" set PY=python

"%PY%" -X utf8 "%~dp0chloe_jobs.py" %*
exit /b %ERRORLEVEL%
