@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM start_demo.bat — Launch Chloe in DEMO mode without touching .env.
REM
REM Sets the three demo-only env vars at process level. Python-dotenv won't
REM override env vars that are already set, so the values here win over .env.
REM Your daily .env stays untouched — close this window and run jarvis the
REM normal way to go back to local-first chat + Kokoro.
REM ─────────────────────────────────────────────────────────────────────────

cd /d "%~dp0"

set CHLOE_OLLAMA_PRIMARY=0
set USE_ELEVENLABS=1
set USE_KOKORO=0

echo.
echo ============================================================
echo   Chloe — DEMO RECORDING MODE
echo ============================================================
echo   CHLOE_OLLAMA_PRIMARY=0   ^(cloud-first chat, ~1s response^)
echo   USE_ELEVENLABS=1         ^(premium voice — burns credits^)
echo   USE_KOKORO=0             ^(local TTS off for this run^)
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo WARNING: venv not found at venv\Scripts\activate.bat
    echo Continuing with system Python...
)

python start_jarvis.py
