"""Cross-thread + cross-process advisory lock for Chloe's shared JSON state.

`dialogue_state.json` and `ed_model.json` are read-modify-written from both the
async chat handler (jarvis) and the synchronous voice path, and a *second*
process (`brain_http`) also touches the model. `os.replace` already makes every
individual write atomic (no torn / half-written JSON) — but it does NOT prevent a
**lost update** when two read-modify-write cycles interleave:

    chat:  load -> mutate(A) ----------------> save(A)
    voice:        load -> mutate(B) -> save(B)              # A is now lost

This module closes that window. Wrap the whole load->mutate->save in a short
advisory lock named after the file:

    from chloe_lock import locked
    with locked("ed_model"):
        m = _load_model(); ...; _save_model(m)

Contract — **best-effort, never raises, never deadlocks a turn**:
- An in-process `threading.Lock` serializes threads in the same interpreter
  (chat coroutine vs. the voice worker thread).
- An `msvcrt` byte-range lock on a sidecar `.lock` file serializes across
  processes (jarvis vs. brain_http) on Windows.
- If the lock can't be acquired within `timeout`, or the platform lacks
  `msvcrt`, the block runs anyway. Correctness degrades to the *pre-lock*
  behavior (a possible lost update) rather than blocking or crashing — exactly
  what the old code did, so this can only ever help.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

try:
    import msvcrt  # Windows only
except Exception:  # pragma: no cover - non-Windows dev boxes
    msvcrt = None

# One re-entrant-free Lock per logical name, created lazily. Guarded by a
# registry lock so two threads racing to first-use the same name agree on one
# Lock object.
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _thread_lock(name: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        lk = _THREAD_LOCKS.get(name)
        if lk is None:
            lk = threading.Lock()
            _THREAD_LOCKS[name] = lk
        return lk


def _lock_path(name: str) -> str:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    d = os.path.join(root, "raw", "locks")
    os.makedirs(d, exist_ok=True)
    # keep the filename filesystem-safe
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return os.path.join(d, safe + ".lock")


@contextlib.contextmanager
def locked(name: str, timeout: float = 3.0):
    """Hold an advisory lock named `name` for the duration of the block.

    Never raises; yields even if the lock can't be taken within `timeout`
    (degrades to no-lock). Reentrancy is NOT supported — don't nest the same
    name on one thread."""
    tlk = _thread_lock(name)
    got_thread = tlk.acquire(timeout=max(0.0, timeout))
    fh = None
    if msvcrt is not None:
        try:
            fh = open(_lock_path(name), "a+b")
            deadline = time.time() + timeout
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break  # got the cross-process lock
                except OSError:
                    if time.time() >= deadline:
                        break  # give up; proceed unlocked
                    time.sleep(0.05)
        except Exception:
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass
            fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass
        if got_thread:
            try:
                tlk.release()
            except Exception:
                pass
