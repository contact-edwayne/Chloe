@echo off
REM ============================================================================
REM stop_chloe.bat - stop Chloe's background services.
REM ============================================================================
REM Since the services run with no windows now, this is how you stop them.
REM (Closing the Chloe HUD stops the backend, but the static server and the
REM wiki watcher keep running until stopped here.)
REM
REM This kills every python.exe owned by your account - the same "nuclear
REM option" from the session handoffs. That's fine on this box because Chloe
REM is the only thing using python.exe. If that ever changes, stop the
REM services from Task Manager instead.
REM ============================================================================

echo This will stop Chloe by killing all python.exe processes.
choice /c YN /m "Proceed"
if errorlevel 2 (
    echo Cancelled - nothing was stopped.
    pause
    exit /b 0
)

echo.
echo Stopping Chloe...
taskkill /im python.exe /f >nul 2>&1
if errorlevel 1 (
    echo   No python.exe processes were running - Chloe was already stopped.
) else (
    echo   Chloe services stopped.
)

echo.
echo Ollama was left running (it's shared and cheap to keep up).
echo To stop it too:  taskkill /im ollama.exe /f
echo.
pause
