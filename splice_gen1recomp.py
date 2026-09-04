"""
Splice: gen1recomp (native Pokemon Gen-1 LOVE2D recompilation) process
management, wired into the existing arcade-watch/commentary pipeline.

Context: Ed is replacing the browser-embedded ROM emulator (EmulatorJS/
WasmBoy, driven from emulator_lite.html) with gen1recomp.exe, a standalone
native Windows app (fused LOVE2D build) that reimplements Pokemon Red/Blue/
Yellow. It is not a ROM and cannot be embedded in the arcade iframe, so
instead of loading it into a canvas, Chloe launches it as its own OS window
and falls back to whole-screen capture for watch/commentary (the fallback
path already exists in _arcade_comment_once for exactly this case: no fresh
in-iframe canvas frame was ever posted, so it always uses
screen_vision.capture_screen(), which auto-targets the monitor holding
gen1recomp's own foreground window).

This adds a small process-management block near the existing _arcade_watch
state (subprocess launch / status / stop, gated by CHLOE_GEN1RECOMP_PATH).
It does NOT touch _arcade_comment_once, the watch loop, or KB/facts code —
those are already engine-agnostic (game is just a string key) and work
unmodified once gen1recomp_panel.html sends game_watch_start/stop with a
"pokemon_<version>" game name over the same WS protocol emulator_lite.html
already uses.

Run from C:\\Users\\eleew\\Documents\\jarvis\\ with the venv active:
    python splice_gen1recomp.py

Verifies:
    - ast.parse passes on the new file
    - tail of new file matches tail of backup (no truncation)
    - line count delta is sane (+75 to +100)

If any check fails, the original is restored from the backup.
"""
from __future__ import annotations
import ast
import shutil
from datetime import date
from pathlib import Path


JARVIS = Path(__file__).parent / "jarvis.py"
BACKUP = JARVIS.with_name(f"jarvis.py.bak.{date.today().isoformat()}-gen1recomp")


ANCHOR = '''_arcade_frame_lock = threading.Lock()
_arcade_frame = {"png": b"", "ts": 0.0}
_ARCADE_FRAME_MAX_AGE_S = 30.0
'''


NEW_BLOCK = '''

# ─── gen1recomp: native Pokemon Gen-1 recompilation (replaces the browser
# ROM emulator on desktop). It runs as its own OS window, not an iframe, so
# there is no canvas to POST frames from — _arcade_comment_once's existing
# mss whole-screen fallback (screen_vision.capture_screen, which auto-picks
# the monitor holding the foreground window) covers watch/commentary with no
# further changes needed there. This block only owns launching/tracking the
# process. Exe path defaults to a "gen1recomp" folder next to jarvis.py so no
# extra folder permissions are needed; override with CHLOE_GEN1RECOMP_PATH if
# it lives elsewhere. Ed provides his own legally-owned ROM the first time he
# plays a given version (red/blue/yellow) via gen1recomp's own import screen
# — that one-time step can't be automated from here.
import subprocess as _sp_gen1recomp

_gen1recomp_proc = None  # subprocess.Popen | None — set by _gen1recomp_launch
_gen1recomp_game = ""    # last-launched version string ("red"/"blue"/"yellow")


def _gen1recomp_exe_path() -> Path:
    default = Path(__file__).resolve().parent / "gen1recomp" / "gen1recomp.exe"
    return Path(os.environ.get("CHLOE_GEN1RECOMP_PATH", str(default)))


def _gen1recomp_is_running() -> bool:
    """True iff the process we launched is still alive. Clears the handle
    (so a later launch isn't blocked) once it has exited."""
    global _gen1recomp_proc
    if _gen1recomp_proc is None:
        return False
    if _gen1recomp_proc.poll() is not None:
        _gen1recomp_proc = None
        return False
    return True


def _gen1recomp_launch(game: str = "") -> dict:
    """Launch gen1recomp.exe with --game <version>. Refuses if we already
    have one tracked and running (closing the window quits back to its own
    launcher rather than the process, so a second launch would just be a
    second window). Returns {ok, pid, game} or {ok: False, error}."""
    global _gen1recomp_proc, _gen1recomp_game
    if _gen1recomp_is_running():
        return {"ok": False, "error": "already running",
                "pid": _gen1recomp_proc.pid}
    exe = _gen1recomp_exe_path()
    if not exe.exists():
        return {"ok": False,
                 "error": f"gen1recomp.exe not found at {exe} "
                          f"(set CHLOE_GEN1RECOMP_PATH or drop it in "
                          f"{exe.parent})"}
    version = (game or os.environ.get("CHLOE_GEN1RECOMP_GAME") or "red").strip().lower()
    if version not in ("red", "blue", "yellow"):
        version = "red"
    try:
        _gen1recomp_proc = _sp_gen1recomp.Popen(
            [str(exe), "--game", version], cwd=str(exe.parent))
        _gen1recomp_game = version
        print(f"[gen1recomp] launched pid={_gen1recomp_proc.pid} game={version}",
              flush=True)
        return {"ok": True, "pid": _gen1recomp_proc.pid, "game": version}
    except Exception as e:
        _gen1recomp_proc = None
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _gen1recomp_status() -> dict:
    running = _gen1recomp_is_running()
    return {"ok": True, "running": running,
            "pid": (_gen1recomp_proc.pid if running else None),
            "game": _gen1recomp_game if running else ""}


def _gen1recomp_stop() -> dict:
    """Best-effort terminate. The game autosaves on its own; this is a
    convenience kill switch, not a clean-shutdown request."""
    global _gen1recomp_proc
    if not _gen1recomp_is_running():
        return {"ok": True, "running": False}
    try:
        _gen1recomp_proc.terminate()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _gen1recomp_proc = None
    return {"ok": True, "running": False}
'''


def main() -> None:
    src = JARVIS.read_text(encoding="utf-8")
    if "_gen1recomp_launch" in src:
        print("[splice] gen1recomp block already present — no-op.")
        return
    if ANCHOR not in src:
        raise SystemExit("[splice] could not find the _arcade_frame anchor — aborting.")

    shutil.copy2(JARVIS, BACKUP)
    print(f"[splice] backup: {BACKUP.name}")

    new_src = src.replace(ANCHOR, ANCHOR + NEW_BLOCK, 1)

    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"[splice] FAIL — ast.parse: {e}")
        raise SystemExit(1)

    old_tail = src.splitlines()[-20:]
    new_tail = new_src.splitlines()[-20:]
    if old_tail != new_tail:
        print("[splice] FAIL — tail diverged; restoring backup")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    delta = len(new_src.splitlines()) - len(src.splitlines())
    if not (75 <= delta <= 100):
        print(f"[splice] FAIL — line delta {delta} outside expected range")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    JARVIS.write_text(new_src, encoding="utf-8")
    print(f"[splice] OK — wrote {JARVIS.name} ({delta:+d} lines)")
    print("[splice] next: drop gen1recomp.exe (delivered separately) into "
          "jarvis\\gen1recomp\\, apply the brain_http.py routes, then "
          "restart Chloe.")


if __name__ == "__main__":
    main()
