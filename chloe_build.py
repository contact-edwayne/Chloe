"""chloe_build.py -- run Claude Code on the Chloe repo, billed to API credits.

The API key is injected into the child process only, so your other Claude Code
sessions stay on the subscription. Key comes from CHLOE_BUILD_API_KEY (.env or
shell), falling back to ANTHROPIC_API_KEY.

    python chloe_build.py "add X to jarvis.py"      # headless (-p), budget-capped
    python chloe_build.py                           # interactive session
    python chloe_build.py --opus "redo the proposal flow"
    python chloe_build.py --budget 15 --turns 80 "big refactor"
    python chloe_build.py --status                  # which auth is active

Headless runs stop at --budget USD (client-side estimate, can overshoot a bit)
or --turns. Interactive mode has no budget flag: set a monthly spend limit on
the key's workspace in the Console. First interactive run asks once to approve
the key; answer yes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass


def build_cmd(a: argparse.Namespace, claude: str) -> list[str]:
    cmd = [claude]
    if a.status:
        return [claude, "auth", "status"]
    model = "opus" if a.opus else a.model
    cmd += ["--model", model, "--permission-mode", a.perm]
    if a.task:
        cmd += ["-p", a.task, "--max-turns", str(a.turns),
                "--max-budget-usd", str(a.budget), "--output-format", a.fmt]
    return cmd


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", nargs="?", help="prompt; omit for interactive")
    p.add_argument("--model", default=os.environ.get("CHLOE_BUILD_MODEL", "sonnet"))
    p.add_argument("--opus", action="store_true")
    p.add_argument("--budget", type=float, default=float(os.environ.get("CHLOE_BUILD_BUDGET", "5")))
    p.add_argument("--turns", type=int, default=40)
    p.add_argument("--perm", default="acceptEdits",
                   choices=["default", "acceptEdits", "plan", "auto", "dontAsk", "bypassPermissions"])
    p.add_argument("--fmt", default="text", choices=["text", "json", "stream-json"])
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the command, don't run")
    a = p.parse_args(argv)

    key = (os.environ.get("CHLOE_BUILD_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        print("no CHLOE_BUILD_API_KEY / ANTHROPIC_API_KEY set (.env or shell)", file=sys.stderr)
        return 2
    claude = shutil.which("claude")
    if not claude and not a.dry_run:
        print("claude CLI not found on PATH (npm i -g @anthropic-ai/claude-code)", file=sys.stderr)
        return 2

    cmd = build_cmd(a, claude or "claude")
    if a.dry_run:
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        print(f"(key ...{key[-4:]} injected into child env only)")
        return 0
    env = dict(os.environ, ANTHROPIC_API_KEY=key)
    for k in ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):  # outrank/confuse billing
        env.pop(k, None)
    return subprocess.call(cmd, cwd=HERE, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
