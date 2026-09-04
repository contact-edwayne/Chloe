@echo off
REM Stage-4 canary harness launcher. Runs the formal apply + 5-min healthy
REM watch in a standalone process (no MCP truncation). Safe + self-cleaning.
REM Preconditions: full Chloe up (:6790 serving) + daily apply cap not hit
REM (resets at local midnight).
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
venv_py314\Scripts\python.exe canary_apply_test.py
echo.
echo [run_canary_test] exit code: %ERRORLEVEL%  (0 = applied, 1 = no apply)
pause
