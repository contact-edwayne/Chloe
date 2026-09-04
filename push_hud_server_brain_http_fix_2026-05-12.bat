@echo off
REM Commit + push the hud_server.py brain_http regression fix from
REM 2026-05-12 morning. A prior file rewrite (when social-media WS handlers
REM were added) had dropped the brain_http startup block from
REM hud_server.start_server(); this restores it.
REM
REM Run from cmd:   push_hud_server_brain_http_fix_2026-05-12.bat

cd /d "C:\Users\eleew\Documents\jarvis"
if errorlevel 1 (
    echo Failed to cd to jarvis folder. Aborting.
    exit /b 1
)

echo.
echo === Status before ===
git status --short
echo.

echo === Commit: restore brain_http.start() in hud_server ===
git add hud_server.py
if errorlevel 1 (
    echo git add hud_server.py failed. Aborting.
    exit /b 1
)
git commit -m "fix(hud-server): restore brain_http.start() in start_server()" -m "A prior file rewrite (when social-media WS handlers like social_drafts_list, social_draft_now, etc. were added to handler()) dropped the brain_http startup block from hud_server.start_server(). Symptom: http://localhost:6790/brain-graph.html returned 'localhost refused to connect' because port 6790 was never bound and no error printed either (because the code path was gone, not failing). Restored the try/except import-and-start block so brain_http boots alongside the WS server. Also restored the explanatory comment about non-fatal failure."
if errorlevel 1 (
    echo Commit failed. Aborting.
    exit /b 1
)

echo.
echo === Pushing to origin/main ===
git push origin main
if errorlevel 1 (
    echo Push failed. Commit is still local; resolve and retry `git push origin main`.
    exit /b 1
)

echo.
echo === Done ===
git log --oneline -3
echo.
pause
