@echo off
REM ============================================================================
REM create_desktop_shortcut.bat - ONE-TIME: put a "Chloe" icon on your Desktop.
REM ============================================================================
REM Creates (or updates) a Desktop shortcut named "Chloe" that points at
REM start_chloe.vbs -- the existing one-click, no-console launcher. Double-
REM clicking that icon:
REM   - checks Tailscale Serve and Ollama, launching Ollama hidden if it's down
REM   - launches the backend, static file server, and wiki watcher hidden
REM     (no terminal windows appear at any point)
REM   - the Chloe HUD window then opens on its own a few seconds later
REM
REM (start_jarvis.py also now checks/starts Ollama itself on top of this, so
REM Ollama comes up automatically even if Chloe is ever launched another way.)
REM
REM Uses chloe_icon.ico (already in this folder) as the shortcut's icon.
REM Safe to run more than once -- it just overwrites the same shortcut.
REM ============================================================================

echo Creating "Chloe" shortcut on your Desktop...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Chloe.lnk'; $sc = $ws.CreateShortcut($lnk); $sc.TargetPath = Join-Path '%~dp0' 'start_chloe.vbs'; $sc.WorkingDirectory = '%~dp0'; $sc.IconLocation = (Join-Path '%~dp0' 'chloe_icon.ico'); $sc.Description = 'Start Chloe -- one click, no terminal windows'; $sc.Save(); Write-Host ('  Created: ' + $lnk); Write-Host ('  Target:  ' + $sc.TargetPath); Write-Host ('  Icon:    ' + $sc.IconLocation)"

echo.
if errorlevel 1 (
    echo There was a problem creating the shortcut - see the message above.
) else (
    echo Done. Double-click "Chloe" on your Desktop any time to start her.
    echo   - No terminal windows will appear.
    echo   - Ollama is checked/started automatically if it isn't already running.
    echo   - The Chloe HUD will open on its own after a few seconds.
    echo.
    echo To stop her later:  stop_chloe.bat  ^(also in this folder^)
)
echo.
pause
