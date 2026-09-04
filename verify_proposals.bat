@echo off
REM Tier 1 self-modification verifier.
REM
REM Runs:
REM   1) ast.parse on brain_wiring.py + chloe_proposals.py
REM   2) chloe_proposals module-level smoke tests (no live Chloe needed)
REM   3) End-to-end /apply_proposal + /revert_proposal through brain_wiring's
REM      try_handle_brain_command using a sandboxed proposal that points at
REM      a temp file (not real source).
REM
REM Run from the jarvis/ directory. No restart required to verify.

setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if exist "venv_py314\Scripts\python.exe" (
    set PY=venv_py314\Scripts\python.exe
) else (
    set PY=python
)

%PY% verify_proposals.py
endlocal
