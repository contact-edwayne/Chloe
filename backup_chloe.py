"""Weekly backup of Chloe's durable state.

Copies:
  - C:\\Chloe\\brain\\         (wiki, episodic, generated, queue, raw,
                                overviews, briefs, facts.md, gaps.md,
                                log.md, index.md, etc.)
  - C:\\Chloe\\secrets\\        (wallet.seed, PIN hash, API key snapshots)
  - jarvis/chloe_memory.db    (+ -wal + -shm if present)
  - jarvis/facts*.md
  - jarvis/chloe_about.md
  - jarvis/chloe_self.md
  - jarvis/finance_watchlist.md

To: C:\\Users\\eleew\\OneDrive\\ChloeBackups\\<YYYY-MM-DD>\\

Rotation: keeps the last 4 weekly snapshots, deletes older.

Idempotent: re-running on the same day overwrites the day's snapshot.

Run manually:
    python C:\\Users\\eleew\\Documents\\jarvis\\backup_chloe.py

Run from a scheduled task (Cowork or Windows): same command.
"""

from __future__ import annotations

import shutil
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# ── Sources ─────────────────────────────────────────────────────────────────
JARVIS_DIR = Path(__file__).resolve().parent
BRAIN_DIR = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
SECRETS_DIR = Path(r"C:\Chloe\secrets")

# Jarvis-side individual files to back up. Globs are also supported.
JARVIS_FILES = [
    "chloe_memory.db",
    "chloe_memory.db-wal",
    "chloe_memory.db-shm",
    "facts.md",
    "facts_home.md",
    "facts_office.md",
    "chloe_about.md",
    "chloe_self.md",
    "finance_watchlist.md",
    "chloe_handoff.md",
    "CHLOE_CHANGELOG.md",
]

# ── Destination ─────────────────────────────────────────────────────────────
BACKUP_ROOT = Path(os.environ.get(
    "CHLOE_BACKUP_ROOT",
    r"C:\Users\eleew\OneDrive\ChloeBackups"))

KEEP_WEEKS = int(os.environ.get("CHLOE_BACKUP_KEEP_WEEKS", "4"))


def log(msg: str) -> None:
    print(f"[backup_chloe] {msg}", flush=True)


def copy_dir(src: Path, dst: Path) -> dict:
    """Recursively copy `src` to `dst`. Skip __pycache__ and .git.
    Returns {files_copied, bytes_copied, errors}."""
    files_copied = 0
    bytes_copied = 0
    errors = 0
    if not src.exists():
        log(f"skip (missing): {src}")
        return {"files_copied": 0, "bytes_copied": 0, "errors": 0}
    for root, dirs, files in os.walk(src):
        # Skip noise dirs in-place so os.walk doesn't recurse into them.
        dirs[:] = [d for d in dirs if d not in (
            "__pycache__", ".git", "node_modules", "venv", "venv_py314")]
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_f = Path(root) / f
            dst_f = out_dir / f
            try:
                shutil.copy2(src_f, dst_f)
                files_copied += 1
                bytes_copied += src_f.stat().st_size
            except (OSError, shutil.Error) as e:
                log(f"  ERR copying {src_f}: {e}")
                errors += 1
    return {"files_copied": files_copied, "bytes_copied": bytes_copied,
            "errors": errors}


def copy_files(src_dir: Path, dst_dir: Path, names: list[str]) -> dict:
    """Copy individual named files from src_dir → dst_dir (if they exist).
    Glob characters are supported."""
    files_copied = 0
    bytes_copied = 0
    errors = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        # Glob support
        for p in src_dir.glob(name):
            if not p.is_file():
                continue
            try:
                shutil.copy2(p, dst_dir / p.name)
                files_copied += 1
                bytes_copied += p.stat().st_size
            except (OSError, shutil.Error) as e:
                log(f"  ERR copying {p}: {e}")
                errors += 1
    return {"files_copied": files_copied, "bytes_copied": bytes_copied,
            "errors": errors}


def rotate(root: Path, keep: int) -> int:
    """Delete snapshot dirs older than `keep` count. Returns dirs deleted."""
    if not root.exists():
        return 0
    snapshots = sorted(
        (p for p in root.iterdir()
         if p.is_dir() and len(p.name) == 10 and p.name[4] == p.name[7] == "-"),
        key=lambda p: p.name, reverse=True)
    deleted = 0
    for old in snapshots[keep:]:
        try:
            shutil.rmtree(old)
            log(f"rotated out: {old.name}")
            deleted += 1
        except OSError as e:
            log(f"  ERR rotating {old}: {e}")
    return deleted


def main() -> int:
    date = datetime.now().strftime("%Y-%m-%d")
    snapshot = BACKUP_ROOT / date
    log(f"snapshot target: {snapshot}")
    snapshot.mkdir(parents=True, exist_ok=True)

    totals = {"files_copied": 0, "bytes_copied": 0, "errors": 0}

    log(f"copying brain: {BRAIN_DIR}")
    r = copy_dir(BRAIN_DIR, snapshot / "brain")
    for k in totals:
        totals[k] += r[k]
    log(f"  brain: {r['files_copied']} files, "
        f"{r['bytes_copied'] / 1e6:.1f} MB, {r['errors']} errors")

    log(f"copying secrets: {SECRETS_DIR}")
    r = copy_dir(SECRETS_DIR, snapshot / "secrets")
    for k in totals:
        totals[k] += r[k]
    log(f"  secrets: {r['files_copied']} files, "
        f"{r['bytes_copied'] / 1e6:.1f} MB, {r['errors']} errors")

    log(f"copying jarvis files from: {JARVIS_DIR}")
    r = copy_files(JARVIS_DIR, snapshot / "jarvis", JARVIS_FILES)
    for k in totals:
        totals[k] += r[k]
    log(f"  jarvis: {r['files_copied']} files, "
        f"{r['bytes_copied'] / 1e6:.1f} MB, {r['errors']} errors")

    log(f"rotating snapshots (keep last {KEEP_WEEKS})...")
    deleted = rotate(BACKUP_ROOT, KEEP_WEEKS)
    log(f"  rotated out: {deleted} old snapshot(s)")

    log(f"---")
    log(f"DONE: {totals['files_copied']} files copied, "
        f"{totals['bytes_copied'] / 1e6:.1f} MB total, "
        f"{totals['errors']} errors")
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
