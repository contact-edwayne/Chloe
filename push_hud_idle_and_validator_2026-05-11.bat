@echo off
REM Commit + push two independent fixes from the 2026-05-11 late-evening session:
REM   1) HUD-idle race in the brain-reply path (jarvis.py)
REM   2) Frontmatter validator hardening + test + on-disk cleanup tool (brain.py + helpers)
REM
REM Run from cmd:   push_hud_idle_and_validator_2026-05-11.bat
REM
REM Repo already has http.version=HTTP/1.1 configured from earlier today, so
REM no extra git config needed.

cd /d "C:\Users\eleew\Documents\jarvis"
if errorlevel 1 (
    echo Failed to cd to jarvis folder. Aborting.
    exit /b 1
)

echo.
echo === Status before ===
git status --short
echo.

echo === Commit 1: HUD-idle race in chat-brain path ===
git add jarvis.py
if errorlevel 1 (
    echo git add jarvis.py failed. Aborting.
    exit /b 1
)
git commit -m "fix(hud): stop chat-brain path from broadcasting idle mid-speech" -m "In reply_audio mode the HUD plays TTS in-browser; its TtsAudio.play onStart/onEnd callbacks already drive the speaking->idle transition in lock-step with actual playback. The backend's broadcast_sync('idle') in the finally clause raced ahead of the audio (most visible on long /query replies, where the backend 'idle' arrives while decodeAudioData is still resolving) and the orb flipped back to idle mid-speech. Skip the manual broadcasts in that mode; the audio callbacks handle state. Local _speak path still needs them. Keep an idle backstop in the exception branch so unexpected errors don't leave the HUD stuck."
if errorlevel 1 (
    echo Commit 1 failed. Aborting.
    exit /b 1
)

echo.
echo === Commit 2: frontmatter validator hardening ===
git add brain.py test_validate_page.py fix_wiki_frontmatter.py
if errorlevel 1 (
    echo git add for commit 2 failed. Aborting.
    exit /b 1
)
git commit -m "harden(brain): normalize wiki frontmatter to exactly one block" -m "_validate_and_clean_page now calls a new _normalize_frontmatter helper that fixes two corruption patterns seen in real LLM output 2026-05-11: (1) double leading '---' (LLM emitted an empty frontmatter block before the real one, e.g. hybrid_llm_router.md, chloe.md, pyqt6.md and 6 others), (2) missing closing '---' (YAML bleeds straight into the body). Idempotent on well-formed input. Adds test_validate_page.py (14 cases) and fix_wiki_frontmatter.py for one-shot cleanup of already-broken pages on disk."
if errorlevel 1 (
    echo Commit 2 failed. Aborting.
    exit /b 1
)

echo.
echo === Pushing both commits to origin/main ===
git push origin main
if errorlevel 1 (
    echo Push failed. Commits are still local; resolve and retry `git push origin main`.
    exit /b 1
)

echo.
echo === Done ===
git log --oneline -3
echo.
pause
