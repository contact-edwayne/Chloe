@echo off
REM ─────────────────────────────────────────────────────────────────────
REM  build.bat — One-shot rebuild for Chloe.
REM
REM  1. Kills any running Chloe.exe so PyInstaller can clean dist\Chloe\
REM     without hitting "Access is denied" on locked .pyd files.
REM  2. Runs pyinstaller with --clean (wipes cache + dist\Chloe\).
REM  3. Re-copies user-data files (.env, models, sounds, kokoro_models, etc.)
REM     into dist\Chloe\ so the exe finds them at runtime.
REM
REM  Run from the project root with your venv activated:
REM      (venv) C:\Users\eleew\Documents\jarvis> build.bat
REM ─────────────────────────────────────────────────────────────────────

echo.
echo ====================================
echo   Killing any running Chloe.exe
echo ====================================
echo.

REM 2^>nul suppresses the "process not found" message when nothing is running.
REM /F = force, /IM = match by image name. Errors are non-fatal — we always
REM continue to the build step.
taskkill /F /IM Chloe.exe 2>nul
if errorlevel 0 (
    REM Brief pause so Windows fully releases the file handles before
    REM PyInstaller tries to clean dist\Chloe\.
    timeout /t 1 /nobreak >nul
)

echo.
echo ====================================
echo   Building Chloe.exe
echo ====================================
echo.

pyinstaller Jarvis.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo BUILD FAILED — see errors above.
    exit /b 1
)

echo.
echo ====================================
echo   Copying user data files
echo ====================================
echo.

if exist .env             xcopy .env             dist\Chloe\               /Y
if exist _env             xcopy _env             dist\Chloe\               /Y
if exist env              xcopy env              dist\Chloe\               /Y
if exist chloe_about.md   xcopy chloe_about.md   dist\Chloe\               /Y
if exist facts.md         xcopy facts.md         dist\Chloe\               /Y
if exist models           xcopy models           dist\Chloe\models\        /E /I /Y
if exist sounds           xcopy sounds           dist\Chloe\sounds\        /E /I /Y
if exist kokoro_models    xcopy kokoro_models    dist\Chloe\kokoro_models\ /E /I /Y

echo.
echo ====================================
echo   Done. Run dist\Chloe\Chloe.exe
echo ====================================
echo.
