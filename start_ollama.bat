@echo off
REM start_ollama.bat - launch ollama serve in the background, then exit.
REM Registered with Task Scheduler via /sc ONLOGON so Ollama is always up
REM before Chloe (and the wiki watcher) try to reach localhost:11434.
REM
REM Idempotent: if Ollama is already listening on 11434, the second
REM `ollama serve` instance exits immediately ("address already in use"),
REM so it's safe to also call this from start_chloe_full.bat as a belt-
REM and-suspenders check.

set OLLAMA_EXE=C:\Users\eleew\AppData\Local\Programs\Ollama\ollama.exe

if not exist "%OLLAMA_EXE%" (
    echo [start_ollama] FATAL: ollama.exe not found at %OLLAMA_EXE%
    exit /b 1
)

REM Idempotence check. If something already answers on :11434, assume it's
REM Ollama (the tray app, a previous invocation, etc.) and exit silently.
REM Without this, every re-invocation prints a noisy
REM     Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each ...
REM to stderr because the second `ollama serve` tries to bind the same port.
curl -s -o nul --max-time 2 http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 exit /b 0

REM `start "" /b` detaches Ollama from this cmd. The bat then exits and
REM Ollama keeps running as a background process tied to the user session.
start "" /b "%OLLAMA_EXE%" serve

exit /b 0
