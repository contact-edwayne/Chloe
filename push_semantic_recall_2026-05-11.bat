@echo off
REM Commit + push the semantic-recall groundwork from the 2026-05-11
REM late-evening session (option E from the handoff). Two commits:
REM   1) chloe_memory.py + backfill + test = the infra
REM   2) brain_wiring.py = /recall slash command for visibility
REM
REM Run from cmd:   push_semantic_recall_2026-05-11.bat
REM
REM Repo already has http.version=HTTP/1.1 configured from earlier today.

cd /d "C:\Users\eleew\Documents\jarvis"
if errorlevel 1 (
    echo Failed to cd to jarvis folder. Aborting.
    exit /b 1
)

echo.
echo === Status before ===
git status --short
echo.

echo === Commit 1: episodic semantic recall infra ===
git add chloe_memory.py backfill_embeddings.py test_semantic_recall.py
if errorlevel 1 (
    echo git add for commit 1 failed. Aborting.
    exit /b 1
)
git commit -m "feat(memory): episodic recall via Ollama embeddings (nomic-embed-text)" -m "Replaces ChloeMemory.search_turns' FTS5 keyword search with vector embeddings. Public API unchanged so all jarvis.py callers keep working. FTS5 stays as fallback when embed fails (Ollama unreachable, model not pulled). Embeddings stored as L2-normalized float32 BLOBs in a new embedding column; brute-force cosine via numpy matmul at query time (sub-ms on hundreds-of-thousands of vectors). Idempotent ALTER TABLE migration runs on init. Cosine threshold (CHLOE_RECALL_THRESHOLD, default 0.35) drops 'least-bad junk' for small corpora. Noise filter drops voice false-positives (single-char content), user slash commands, assistant recall-output turns, and brain-graph [CONTEXT -] injections. backfill_embeddings.py walks NULL-embedding rows for retroactive coverage. test_semantic_recall.py exercises the full path with a mocked toy embedder."
if errorlevel 1 (
    echo Commit 1 failed. Aborting.
    exit /b 1
)

echo.
echo === Commit 2: /recall slash command for visibility ===
git add brain_wiring.py
if errorlevel 1 (
    echo git add brain_wiring.py failed. Aborting.
    exit /b 1
)
git commit -m "feat(brain): /recall slash command for semantic recall visibility" -m "Demo + debug hook over ChloeMemory.search_turns. Formats top-10 hits as a markdown list with role/modality/relative-time. min_age_hours=0.25 (15 min) skips trivially-recent self-reference; the threshold and noise filter (in chloe_memory) handle the rest. Late 'from jarvis import _memory' avoids a circular import."
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
git log --oneline -4
echo.
pause
