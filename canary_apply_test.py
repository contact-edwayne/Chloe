"""Stage-4 canary harness — formally exercises autonomous apply + the full
5-minute healthy watchdog watch in a STANDALONE process.

Why this exists: driving `/autonomous run-now` through the Cowork MCP `chat()`
tool can't sustain the 5-min blocking `supervise_apply` — the MCP request times
out and is cancelled mid-watch, so `watchdog_watch=healthy` never logs and the
watch is orphaned. Run THIS from the Windows venv instead; nothing cancels the
block, so the watch completes and logs a real outcome.

    venv_py314\\Scripts\\python.exe canary_apply_test.py

Safe by construction: the target is an unimported throwaway module
(`chloe_canary.py`), and the script cleans up after itself (deletes the canary
+ any .bak, removes its own seeded log lines, sets autonomous back OFF).

Preconditions for a HEALTHY result:
  - brain_http must be serving on :6790 (full Chloe up). Probe printed below.
  - autonomous daily apply cap not already hit (resets at LOCAL midnight).
"""

import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The job summary prints contain non-ASCII (→ · em-dash); the Windows console is
# CP1251/cp1252, which would crash these prints with UnicodeEncodeError
# (lesson #3). Force UTF-8 stdout so the harness can't die on its own output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CANARY = HERE / "chloe_canary.py"
LOG = HERE / "logs" / "chloe_jobs.log"
MAX_ATTEMPTS = 8  # LLM diff-match rate ~1/3; failed drafts are free (no apply)

# def on LINE 1 + SINGLE quotes — both required so the LLM's @@ line number and
# context/removal lines match strict-patch (_apply_diff). See handoff lessons.
CANARY_SRC = (
    "def compute_total(data):\n"
    "    return data['count'] * data['price']\n"
)


def _seed_text() -> str:
    """5x identical recent KeyError tracebacks pointing at the canary.
    Timestamps are 'now minus a few minutes' so they're inside the job's
    24h window. KeyError is in the extractor's captured-exception whitelist."""
    blocks = []
    for i in range(5):
        ts = time.strftime("%Y-%m-%d %H:%M:%S",
                           time.localtime(time.time() - 300 + i))
        blocks.append(
            f"{ts} [ERROR] chloe_jobs - canary compute_total failed\n"
            "Traceback (most recent call last):\n"
            f'  File "{CANARY}", line 2, in compute_total\n'
            "    return data['count'] * data['price']\n"
            "KeyError: 'count'\n"
        )
    return "".join(blocks)


def _probe_6790() -> bool:
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:6790/api/health/full", timeout=5) as r:
            ok = r.status == 200
            print(f"[canary] :6790 probe -> HTTP {r.status}")
            return ok
    except Exception as e:
        print(f"[canary] :6790 probe FAILED ({e.__class__.__name__}: {e}). "
              "With the endpoint-grace fix the watch will return INCONCLUSIVE "
              "(patch kept, not reverted) rather than HEALTHY. Start full Chloe "
              "for a HEALTHY result.")
        return False


def main() -> int:
    import chloe_jobs
    import chloe_watchdog

    # 1) Enable autonomous (read-modify-write preserves other keys).
    st = chloe_jobs._read_autonomous_state()
    st["enabled"] = True
    chloe_jobs._write_autonomous_state(st)

    # 2) Stage fixtures.
    CANARY.write_text(CANARY_SRC, encoding="utf-8")
    seed = _seed_text()
    with LOG.open("a", encoding="utf-8") as f:
        f.write(seed)
    print(f"[canary] staged {CANARY.name} + seeded 5x traceback")
    _probe_6790()

    outcome = "no apply"
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            chloe_watchdog.reset_failures()  # clear cf so a prior mismatch
            #                                  doesn't lock us out mid-retry
            print(f"\n[canary] attempt {attempt}/{MAX_ATTEMPTS} — running job "
                  "(BLOCKS ~5 min once a diff matches and applies)...")
            summary = chloe_jobs.job_autonomous_fix_recurring_errors()
            print(f"[canary] {summary}")
            if "applied 1" in summary:
                outcome = "applied"
                break
            low = summary.lower()
            if "daily cap" in low or "min interval" in low or "locked" in low:
                print("[canary] gate is CLOSED — stopping. "
                      "(cap resets at local midnight; interval is 30 min.)")
                break
            print("[canary] no apply (diff mismatch or no eligible target) — "
                  "resetting and retrying.")
    finally:
        # 3) Restore safe state + clean fixtures regardless of outcome.
        st = chloe_jobs._read_autonomous_state()
        st["enabled"] = False
        chloe_jobs._write_autonomous_state(st)
        try:
            CANARY.unlink()
        except FileNotFoundError:
            pass
        for bak in HERE.glob("chloe_canary.py.bak.*"):
            try:
                bak.unlink()
            except OSError:
                pass
        txt = LOG.read_text(encoding="utf-8", errors="replace")
        if seed in txt:
            LOG.write_text(txt.replace(seed, ""), encoding="utf-8")
        print("\n[canary] cleanup done: canary + .bak removed, log seed "
              "removed, autonomous set OFF.")

    print("\n[canary] recent watchdog history:")
    for h in chloe_watchdog.history(6):
        print(f"   {h.get('ts_iso','?')}  {h.get('action',''):16s}  "
              f"{h.get('outcome',''):13s}  {h.get('slug','')[:34]}  "
              f"{h.get('reason','')[:60]}")

    print(f"\n[canary] RESULT: {outcome}. Look for an `autonomous_apply ok` "
          "followed by `watchdog_watch healthy` above — that's the formal "
          "clean-watch completion. (`inconclusive` = :6790 was down; patch "
          "kept, not a failure.)")
    return 0 if outcome == "applied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
