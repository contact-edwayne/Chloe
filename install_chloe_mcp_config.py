"""One-shot installer: register chloe MCP server in Cowork's config.

Reads `%APPDATA%\\Claude\\claude_desktop_config.json` (creates if missing),
adds the `chloe` entry to `mcpServers` (merges with whatever's already
there), backs up the original, writes the result.

Idempotent: re-running just overwrites the `chloe` entry with the latest
config. Other MCP servers in the file are preserved.

After running, fully quit and relaunch the Claude desktop app for
Cowork to pick up the new server.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


CONFIG = Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
SERVER = Path(__file__).resolve().parent / "chloe_mcp_server.py"


def detect_python() -> str:
    """Pick the python interpreter Cowork should use. Prefer the venv that
    has `mcp` installed; fall back to the one running this script."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "venv_py314" / "Scripts" / "python.exe",
        here / "venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            # Check if mcp is importable from this interpreter — but skip
            # the subprocess probe to keep this script side-effect free.
            # Ed will see a Cowork-side error if the wrong one is picked,
            # and can re-edit the config manually.
            return str(c)
    # Last resort: the current interpreter (whatever ran this script).
    return sys.executable


def main() -> None:
    if not SERVER.exists():
        print(f"ERROR: server script missing at {SERVER}", file=sys.stderr)
        sys.exit(1)

    CONFIG.parent.mkdir(parents=True, exist_ok=True)

    if CONFIG.exists():
        try:
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"ERROR: existing config is invalid JSON: {e}",
                  file=sys.stderr)
            print(f"Fix or delete {CONFIG} and re-run.", file=sys.stderr)
            sys.exit(2)
        backup = CONFIG.with_suffix(
            f".json.bak.{datetime.now():%Y-%m-%d-%H%M%S}")
        shutil.copy2(CONFIG, backup)
        print(f"[install] backup -> {backup.name}")
    else:
        data = {}
        print(f"[install] creating new config at {CONFIG}")

    if not isinstance(data, dict):
        print(f"ERROR: config root is not an object (got {type(data).__name__})",
              file=sys.stderr)
        sys.exit(3)

    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        print("ERROR: mcpServers is not an object — refusing to overwrite",
              file=sys.stderr)
        sys.exit(4)

    py = detect_python()
    entry = {
        "command": py,
        "args": [str(SERVER)],
        "env": {
            "CHLOE_MODE": os.environ.get("CHLOE_MODE", "home"),
            "CHLOE_BRAIN_ROOT": os.environ.get(
                "CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"),
        },
    }

    existed = "chloe" in servers
    servers["chloe"] = entry

    CONFIG.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    verb = "updated" if existed else "added"
    print(f"[install] {verb} chloe entry in {CONFIG}")
    print(f"[install]   command: {py}")
    print(f"[install]   args:    [{SERVER}]")
    print(f"[install] now fully quit + relaunch the Claude desktop app.")
    print(f"[install] in a fresh Cowork session, ask: "
          f"'What does Chloe remember about <topic>?'")


if __name__ == "__main__":
    main()
