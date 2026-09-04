"""
Follow-up splice: fixes a launch race in _gen1recomp_launch.

Symptom Ed hit: clicking Launch produced TWO gen1recomp.exe processes
("launched pid=25536 game=red" immediately followed by "launched pid=19612
game=red"). brain_http.py runs on ThreadingHTTPServer, so two concurrent
POST /api/gen1recomp/launch requests can both evaluate
_gen1recomp_is_running() as False before either one sets the global
_gen1recomp_proc handle -- classic check-then-act race between threads.

Fix: a threading.Lock held for the whole check+spawn+record sequence, so
only one thread can be inside _gen1recomp_launch at a time.

NOTE: this does NOT kill the orphaned first process from the race that
already happened (pid 25536 in Ed's log) -- that one is untracked now
(_gen1recomp_proc only ever holds the LAST launch). Close it by hand once
via Task Manager if it's still running.

Run from C:\\Users\\eleew\\Documents\\jarvis\\ with the venv active:
    venv_py314\\Scripts\\python.exe splice_gen1recomp_lock.py

Requires splice_gen1recomp.py to have already been applied (it has, per
Ed's confirmation). Verifies ast.parse, tail match, and a sane line delta;
restores the backup on any failure.
"""
from __future__ import annotations
import ast
import shutil
from datetime import date
from pathlib import Path


JARVIS = Path(__file__).parent / "jarvis.py"
BACKUP = JARVIS.with_name(f"jarvis.py.bak.{date.today().isoformat()}-gen1recomp-lock")


OLD = '''_gen1recomp_proc = None  # subprocess.Popen | None — set by _gen1recomp_launch
_gen1recomp_game = ""    # last-launched version string ("red"/"blue"/"yellow")'''

NEW = '''_gen1recomp_proc = None  # subprocess.Popen | None — set by _gen1recomp_launch
_gen1recomp_game = ""    # last-launched version string ("red"/"blue"/"yellow")
# Guards the whole check-then-spawn-then-record sequence in _gen1recomp_launch.
# brain_http.py's ThreadingHTTPServer can run two POST /api/gen1recomp/launch
# handlers concurrently; without this lock both can see _gen1recomp_is_running()
# as False before either sets _gen1recomp_proc, spawning two exes.
_gen1recomp_lock = threading.Lock()'''


OLD_BODY = '''    global _gen1recomp_proc, _gen1recomp_game
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
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}'''

NEW_BODY = '''    global _gen1recomp_proc, _gen1recomp_game
    with _gen1recomp_lock:
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
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}'''


def main() -> None:
    src = JARVIS.read_text(encoding="utf-8")
    if "_gen1recomp_lock" in src:
        print("[splice] lock fix already present — no-op.")
        return
    if OLD not in src:
        raise SystemExit("[splice] could not find the _gen1recomp_proc anchor — "
                          "has splice_gen1recomp.py been applied yet?")
    if OLD_BODY not in src:
        raise SystemExit("[splice] could not find _gen1recomp_launch's body — aborting.")

    shutil.copy2(JARVIS, BACKUP)
    print(f"[splice] backup: {BACKUP.name}")

    new_src = src.replace(OLD, NEW, 1)
    new_src = new_src.replace(OLD_BODY, NEW_BODY, 1)

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
    if not (3 <= delta <= 12):
        print(f"[splice] FAIL — line delta {delta} outside expected range")
        shutil.copy2(BACKUP, JARVIS)
        raise SystemExit(1)

    JARVIS.write_text(new_src, encoding="utf-8")
    print(f"[splice] OK — wrote {JARVIS.name} ({delta:+d} lines)")
    print("[splice] restart Chloe. If pid 25536 from the earlier double-launch "
          "is still running, close it by hand in Task Manager.")


if __name__ == "__main__":
    main()
