@echo off
REM Run from C:\Users\eleew\Documents\jarvis to commit the 2026-05-12 evening
REM punch-list fixes (ack short-circuit + recall FTS5 fallback).

cd /d C:\Users\eleew\Documents\jarvis

echo === Backing up jarvis.py first ===
copy /Y jarvis.py jarvis.py.bak.2026-05-12-evening

echo.
echo === git diff (review before commit) ===
git diff --stat jarvis.py chloe_memory.py
echo.

echo === Commit ===
git add jarvis.py chloe_memory.py
git commit -m "Fix punch-list bugs: ack short-circuit + recall FTS5 fallback" ^
           -m "" ^
           -m "Bug A (jarvis.py): bare 'Thank you' / 'thanks chloe' / 'goodnight' etc." ^
           -m "no longer reach the LLM, where qwen2.5:32b would sometimes hallucinate" ^
           -m "a grep_source tool call. New _try_handle_acknowledgement() short-circuits" ^
           -m "in handle_chat, PTT path, and _process_voice_turn." ^
           -m "" ^
           -m "Bug B (chloe_memory.py): search_turns now falls back to FTS5 when" ^
           -m "the embedded-rows window is empty (line 369) AND when the cosine loop" ^
           -m "produces zero above-threshold hits (line 431). Previously both cases" ^
           -m "returned [] silently, so older discussions with NULL embeddings or" ^
           -m "below-threshold paraphrase mismatch never surfaced." ^
           -m "" ^
           -m "Both bugs surfaced by 2026-05-12 weekly_review.py first run."

echo.
echo === git log (verify commit) ===
git log --oneline -1

echo.
echo === Push ===
echo If the diff looks right above, run:  git push
