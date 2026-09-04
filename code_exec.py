"""
code_exec.py — Sandboxed one-shot Python execution for Chloe.

Backs the `run_python` LLM tool (jarvis.py) so Chloe can actually
compute an answer instead of guessing at arithmetic or letting the LLM
hallucinate a result. Ed, 2026-09-02: math/computation was flagged as a
real gap after the wallet/stocks/weather work proved out the "give the
model a real tool instead of trusting its output" pattern for anything
checkable -- this is the same fix applied to computation in general
(arithmetic, unit conversions, quick data-munging) rather than just
prices/weather.

Threat model: Ed is the only person who talks to Chloe, on his own
machine, but the CODE BEING RUN is written by an LLM from a voice/chat
prompt, not by Ed directly -- a hallucinated or misdirected snippet
should not be able to delete files, hang the process, exhaust memory,
or reach the network. This is NOT a security boundary against a hostile
user (that would need a real container); it's a blast-radius limiter
against an unsupervised LLM's own code. Mitigations:
  - Runs as a genuinely separate OS process (`python -I -`, isolated
    mode: ignores PYTHONPATH/site customization), not an in-process
    exec() -- a crash or hang there can't take Chloe down with it.
  - Hard wall-clock timeout (default 6s, capped at 20s); subprocess.run
    kills the child on TimeoutExpired.
  - stdout/stderr captured and truncated so a runaway print loop can't
    balloon the LLM context or a TTS turn.
  - cwd is a fresh temp directory per call, never the jarvis source
    tree or C:\\Chloe -- an accidental relative-path write can't touch
    real config/secrets.
  - Source text is checked against a small denylist (see
    _BLOCKED_PATTERNS) before it ever runs: file deletion, subprocess/
    os.system, network access, ctypes/winreg, eval/exec/__import__.
    Deliberately a denylist, not an allowlist -- an allowlist narrow
    enough to be safe would also be too narrow for genuine "write a
    quick script" requests (loops, json, re, datetime, etc. all need to
    keep working). A caught pattern returns an honest refusal instead
    of running, same "honest miss" contract as the rest of this
    codebase (lights/local_media/stocks all refuse rather than guess).
  - File *reading* is not blocked (needed for basic usefulness, and
    Chloe already exposes brain_read/grep_source with their own scope
                                       independently of this tool).

Public API
----------
run_python(code: str, timeout: float = 6.0) -> dict
    {"ok": bool, "stdout": str, "stderr": str, "returncode": int|None,
     "blocked": str|None, "timed_out": bool}
    ok is True only for returncode == 0 and not blocked/timed_out.

run_python_tool(code: str, timeout: float = 6.0) -> str
    Same execution, formatted as a single string for the LLM tool-call
    result (jarvis.py's RUN_PYTHON_SCHEMA dispatch target).

CLI:
    python code_exec.py "print(2**10)"
    echo "print(1+1)" | python code_exec.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile

_MAX_OUTPUT_CHARS = 3000
_DEFAULT_TIMEOUT_S = 6.0
_MAX_TIMEOUT_S = 20.0

_BLOCKED_PATTERNS = [
    (re.compile(r"\bimport\s+subprocess\b"), "subprocess"),
    (re.compile(r"\bimport\s+socket\b"), "socket (network)"),
    (re.compile(r"\bimport\s+ctypes\b"), "ctypes"),
    (re.compile(r"\bimport\s+winreg\b"), "winreg"),
    (re.compile(r"\bimport\s+shutil\b"), "shutil"),
    (re.compile(r"\bos\.(system|popen|remove|unlink|rmdir)\s*\("), "a destructive os.* call"),
    (re.compile(r"\bshutil\.(rmtree|move)\s*\("), "shutil.rmtree/move"),
    (re.compile(r"\b(eval|exec)\s*\("), "eval/exec"),
    (re.compile(r"__import__\s*\("), "__import__"),
    (re.compile(r"\brequests\.|urllib\.request\.|http\.client\."), "network access"),
]


def _blocked_reason(code: str):
    for pattern, label in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return label
    return None


def run_python(code: str, timeout: float = _DEFAULT_TIMEOUT_S) -> dict:
    code = code or ""
    try:
        timeout = float(timeout or _DEFAULT_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_S
    timeout = min(max(timeout, 1.0), _MAX_TIMEOUT_S)

    reason = _blocked_reason(code)
    if reason:
        return {"ok": False, "stdout": "", "stderr": "",
                "returncode": None, "blocked": reason, "timed_out": False}

    with tempfile.TemporaryDirectory(prefix="chloe_exec_") as scratch:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "",
                    "returncode": None, "blocked": None, "timed_out": True}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}",
                    "returncode": None, "blocked": None, "timed_out": False}

    stdout = (proc.stdout or "")[:_MAX_OUTPUT_CHARS]
    stderr = (proc.stderr or "")[:_MAX_OUTPUT_CHARS]
    return {"ok": proc.returncode == 0, "stdout": stdout, "stderr": stderr,
            "returncode": proc.returncode, "blocked": None, "timed_out": False}


def run_python_tool(code: str, timeout: float = _DEFAULT_TIMEOUT_S) -> str:
    r = run_python(code, timeout=timeout)
    if r["blocked"]:
        return (f"Refused to run this: it contains {r['blocked']}, which is "
                f"blocked for safety. Rewrite without it.")
    if r["timed_out"]:
        return f"Timed out after {timeout:.0f}s (likely an infinite loop) -- killed."
    parts = []
    if r["stdout"]:
        parts.append(f"stdout:\n{r['stdout']}")
    if r["stderr"]:
        parts.append(f"stderr:\n{r['stderr']}")
    if not parts:
        parts.append("(no output)")
    status = "ok" if r["ok"] else f"exited with code {r['returncode']}"
    return f"[{status}]\n" + "\n".join(parts)


def _cli() -> int:
    code = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    print(run_python_tool(code))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
