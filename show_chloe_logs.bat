@echo off
REM ============================================================================
REM show_chloe_logs.bat - open live-tailing windows for Chloe's service logs.
REM ============================================================================
REM Chloe's services run with no console windows now, so this is how you watch
REM their output. Opens a PowerShell window per log, each live-tailing (the
REM equivalent of the old "Chloe Backend / Static / Watcher" windows).
REM Close the windows when done - it does not affect the running services.
REM
REM   show_chloe_logs.bat            -> all three logs
REM   show_chloe_logs.bat backend   -> just backend  (also: static, watcher)
REM ============================================================================

cd /d "%~dp0"
if not exist logs mkdir logs

REM Make sure each log exists so Get-Content -Wait doesn't error on a fresh box.
if not exist "logs\backend.log" type nul > "logs\backend.log"
if not exist "logs\static.log"  type nul > "logs\static.log"
if not exist "logs\watcher.log" type nul > "logs\watcher.log"

if /i "%~1"=="backend" goto :one
if /i "%~1"=="static"  goto :one
if /i "%~1"=="watcher" goto :one

REM No arg (or unrecognized): open all three.
start "Chloe backend log" powershell -NoExit -NoProfile -Command "Get-Content -LiteralPath '%~dp0logs\backend.log' -Wait -Tail 40"
start "Chloe static log"  powershell -NoExit -NoProfile -Command "Get-Content -LiteralPath '%~dp0logs\static.log' -Wait -Tail 40"
start "Chloe watcher log" powershell -NoExit -NoProfile -Command "Get-Content -LiteralPath '%~dp0logs\watcher.log' -Wait -Tail 40"
goto :eof

:one
start "Chloe %~1 log" powershell -NoExit -NoProfile -Command "Get-Content -LiteralPath '%~dp0logs\%~1.log' -Wait -Tail 40"
goto :eof
