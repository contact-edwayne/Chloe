@echo off
REM ============================================================================
REM setup_ollama_startup.bat - ONE-TIME: make Ollama start hidden at login.
REM ============================================================================
REM Repoints the existing "chloe_ollama_serve" shortcut in your Windows Startup
REM folder so it launches start_ollama_hidden.vbs (no window) instead of
REM start_ollama.bat directly (which flashed a console at every login).
REM
REM Safe to run more than once. Run it once and you're done - the change takes
REM effect at your next login (your current Ollama process is untouched).
REM ============================================================================

echo Repointing the chloe_ollama_serve Startup shortcut...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $lnk = Join-Path $ws.SpecialFolders('Startup') 'chloe_ollama_serve.lnk'; $sc = $ws.CreateShortcut($lnk); $sc.TargetPath = Join-Path '%~dp0' 'start_ollama_hidden.vbs'; $sc.WorkingDirectory = '%~dp0'; $sc.Arguments = ''; $sc.WindowStyle = 7; $sc.Description = 'Launch Ollama hidden at login (for Chloe)'; $sc.Save(); Write-Host ('  Updated: ' + $lnk); Write-Host ('  Target:  ' + $sc.TargetPath)"

echo.
if errorlevel 1 (
    echo There was a problem updating the shortcut - see the message above.
) else (
    echo Done. Ollama will start with no window at your next login.
)
echo.
pause
