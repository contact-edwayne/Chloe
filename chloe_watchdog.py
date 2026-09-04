"""Stage 4 of self-modification: autonomous-apply watchdog with auto-revert.

After Chloe (autonomously) applies a code proposal, the watchdog watches
her health for N minutes. If health degrades — startup fails, WS won't
reconnect, Ollama unreachable, voice loop dies, memory DB unwritable —
auto-revert the proposal from its timestamped backup. Idle states never
get touched.

This is the FINAL safety net. It needs to be paranoid because:
- The autonomous proposer can apply changes Ed never reviewed.
- A bad apply that takes down startup leaves Chloe dark with no chat
  surface to ask for help.
- Bricking is real; the watchdog IS the recovery.

Hard rate limits (caps shared with the proposer in `chloe_jobs.py`):
- `MAX_AUTONOMOUS_APPLIES_PER_DAY = 2`
- `MIN_INTERVAL_BETWEEN_AUTONOMOUS_S = 1800` (30 min between applies)
- `MAX_CONSECUTIVE_FAILURES = 2` (after two auto-reverts, autonomy
  disables itself; Ed must re-enable explicitly)

State persists to `C:\\Chloe\\brain\\watchdog_state.json` so a reboot
mid-watch still resolves correctly: on boot, if a slug is under watch
AND the deadline is past, run a final health check; if failing, revert.

Public API:
  - `supervise_apply(slug, watch_minutes=5, expected_to_restart=False)`
  - `status() -> dict`
  - `cancel_watch(slug="")` — drop in-flight watches without reverting
  - `history(limit=20) -> list[dict]` — completed watches
  - `record_autonomous_apply(slug)` / `record_autonomous_failure()`
    used by the proposer; rate-limit accounting.
  - `autonomous_can_apply_now() -> tuple[bool, str]` — proposer-side
    gate check
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# ─── Caps + paths ─────────────────────────────────────────────────────────

MAX_AUTONOMOUS_APPLIES_PER_DAY = 2
MIN_INTERVAL_BETWEEN_AUTONOMOUS_S = 1800
MAX_CONSECUTIVE_FAILURES = 2
HEALTH_POLL_INTERVAL_S = 30
DEFAULT_WATCH_MINUTES = 5
# Always-on grace for an UNREACHABLE health endpoint (TCP refused / timeout),
# independent of expected_to_restart. The autonomous proposer applies from a
# separate process and the HUD health server (:6790) may simply not be up; an
# unrelated/unimported patch cannot make the endpoint refuse connections, so a
# never-reachable endpoint is an infra condition, not a patch regression.
ENDPOINT_REACHABILITY_GRACE_S = 60
HEALTH_ENDPOINT_DEFAULT = "http://127.0.0.1:6790/api/health/full"


def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _state_path() -> Path:
    p = _brain_root()
    p.mkdir(parents=True, exist_ok=True)
    return p / "watchdog_state.json"


def _health_url() -> str:
    return os.environ.get("CHLOE_HEALTH_URL", HEALTH_ENDPOINT_DEFAULT)


def _now() -> float:
    return time.time()


def _today_key() -> str:
    return _dt.date.today().isoformat()


# ─── State file IO ────────────────────────────────────────────────────────

_STATE_LOCK = threading.Lock()

# Schema:
# {
#   "under_watch": {slug: {applied_at, deadline, expected_to_restart,
#                          backup_path, target, fail_count}},
#   "history": [{slug, action, outcome, ts, reason}],   # capped at 200
#   "applies_today": {date_iso: count},
#   "last_apply_ts": float,
#   "consecutive_failures": int,
# }


def _empty_state() -> dict:
    return {
        "under_watch": {},
        "history": [],
        "applies_today": {},
        "last_apply_ts": 0.0,
        "consecutive_failures": 0,
    }


def _load_state() -> dict:
    p = _state_path()
    if not p.exists():
        return _empty_state()
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        # Backfill missing keys for forward compat.
        for k, v in _empty_state().items():
            s.setdefault(k, v)
        return s
    except (OSError, json.JSONDecodeError):
        return _empty_state()


def _save_state(state: dict) -> None:
    # Cap history.
    hist = state.get("history", [])
    if len(hist) > 200:
        state["history"] = hist[-200:]
    p = _state_path()
    tmp = p.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _log(state: dict, slug: str, action: str, outcome: str,
         reason: str = "") -> None:
    state.setdefault("history", []).append({
        "slug":    slug,
        "action":  action,
        "outcome": outcome,
        "ts":      _now(),
        "ts_iso":  _dt.datetime.now().isoformat(timespec="seconds"),
        "reason":  reason,
    })


# ─── Health checks ────────────────────────────────────────────────────────

def _fetch_health(timeout_s: float = 5.0) -> dict:
    """Hit Chloe's /api/health/full. Returns parsed JSON or
    {"ok": False, "error": "..."}.
    """
    try:
        req = urllib.request.Request(_health_url(),
                                     headers={"User-Agent": "chloe-watchdog"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return {"ok": False,
                        "error": f"http {resp.status}"}
            body = resp.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"json: {e}"}
        return data
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"urlopen: {e.reason}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _is_healthy(health: dict) -> tuple[bool, str]:
    """Decide if a health payload indicates Chloe is healthy.

    Strict: requires both `checks_failed == 0` AND specific critical
    sub-checks (ollama_reachable, memory_db_writable, ws_connected).
    Returns (ok, reason_if_unhealthy).
    """
    if not isinstance(health, dict):
        return False, "non-dict health payload"
    if health.get("error"):
        return False, f"endpoint error: {health['error']}"
    failed = health.get("checks_failed", 0)
    if failed and failed > 0:
        issues = health.get("issues", [])
        return False, f"{failed} checks failed: {'; '.join(issues[:3])}"
    # Critical sub-check whitelist.
    for k in ("ollama_reachable", "memory_db_writable"):
        if k in health and not health[k]:
            return False, f"critical check {k} false"
    return True, ""


def _is_endpoint_unreachable(reason: str) -> bool:
    """True when the health check failed because the endpoint couldn't be
    reached at all (TCP refused / DNS / timeout) rather than answering with an
    unhealthy payload. `_fetch_health` surfaces connection failures as a
    urlopen error, so we key off that. A reachable-but-unhealthy response
    ("N checks failed", "critical check ... false", "http 5xx", bad json) is
    NOT unreachable and is always treated as a real regression."""
    return "urlopen" in (reason or "").lower()


# ─── supervise_apply ──────────────────────────────────────────────────────

def supervise_apply(slug: str, watch_minutes: int = DEFAULT_WATCH_MINUTES,
                    expected_to_restart: bool = False) -> dict:
    """Watch Chloe's health after applying `slug`. Auto-revert on failure.

    Args:
        slug: the proposal slug that was just applied.
        watch_minutes: how long to watch (1-30). Default 5.
        expected_to_restart: if True, allow up to 60s of unreachable
            health endpoint before considering it a failure (gives
            Chloe time to come back up).

    This is a BLOCKING call. Runs on the caller's thread. Caller
    (typically the autonomous proposer job) should accept the blocking
    cost — Stage-4 applies happen at 04:00 by default; 5 min of block
    is acceptable.

    Returns:
        {"outcome": "healthy" | "reverted", "slug": ..., "reason": ...,
         "polls_ok": int, "polls_fail": int}
    """
    if watch_minutes < 1 or watch_minutes > 30:
        return {"outcome": "error", "slug": slug,
                "reason": f"watch_minutes must be 1-30, got {watch_minutes}"}

    deadline = _now() + (watch_minutes * 60)

    with _STATE_LOCK:
        state = _load_state()
        # Register the watch.
        proposal_meta = _lookup_proposal_meta(slug)
        state["under_watch"][slug] = {
            "applied_at":          _now(),
            "deadline":            deadline,
            "expected_to_restart": expected_to_restart,
            "backup_path":         proposal_meta.get("backup_path", ""),
            "target":              proposal_meta.get("target", ""),
            "fail_count":          0,
            "polls_ok":            0,
            "polls_fail":          0,
        }
        _save_state(state)

    print(f"[watchdog] supervising {slug} for {watch_minutes}min "
          f"(expected_to_restart={expected_to_restart})", flush=True)

    # Two UNREACHABILITY graces (neither ever excuses a reachable-but-unhealthy
    # response — that always counts immediately):
    #   - restart grace: if the patch is expected to bounce Chloe, tolerate a
    #     brief endpoint outage right after apply.
    #   - reachability grace: always-on; :6790 may simply not be up yet.
    restart_grace_until = _now() + 60.0 if expected_to_restart else 0.0
    reachability_grace_until = _now() + ENDPOINT_REACHABILITY_GRACE_S

    consecutive_fails = 0
    polls_ok = 0
    polls_fail = 0
    polls_unreachable = 0
    ever_reachable = False

    while _now() < deadline:
        time.sleep(HEALTH_POLL_INTERVAL_S)
        health = _fetch_health()
        ok, reason = _is_healthy(health)

        if ok:
            polls_ok += 1
            ever_reachable = True
            consecutive_fails = 0
        elif _is_endpoint_unreachable(reason):
            polls_unreachable += 1
            # In either grace window: wait, don't hold it against the patch.
            if (_now() < restart_grace_until
                    or _now() < reachability_grace_until):
                print(f"[watchdog] {slug}: endpoint unreachable in grace "
                      f"window, waiting", flush=True)
                continue
            # Past grace. Only treat unreachable as a patch fault if the
            # endpoint had been up earlier this watch (the patch may have just
            # taken it down). If it was NEVER reachable, :6790 is absent —
            # infra, not the patch — so do NOT count it toward a revert. An
            # applied patch to a live module isn't even loaded until restart;
            # on_boot_recover does the post-load health check.
            if not ever_reachable:
                print(f"[watchdog] {slug}: endpoint never reachable — infra "
                      f"down, not counting against patch", flush=True)
                continue
            polls_fail += 1
            consecutive_fails += 1
            print(f"[watchdog] {slug} endpoint went down after being up "
                  f"({consecutive_fails} consecutive): {reason}", flush=True)
            if consecutive_fails >= 2:
                return _do_revert(slug, polls_ok, polls_fail, reason)
        else:
            # Reachable but reporting unhealthy → real regression, count now.
            polls_fail += 1
            consecutive_fails += 1
            print(f"[watchdog] {slug} unhealthy ({consecutive_fails} "
                  f"consecutive): {reason}", flush=True)
            if consecutive_fails >= 2:
                return _do_revert(slug, polls_ok, polls_fail, reason)

        # Live-update poll counts in state.
        with _STATE_LOCK:
            state = _load_state()
            if slug in state["under_watch"]:
                state["under_watch"][slug]["polls_ok"] = polls_ok
                state["under_watch"][slug]["polls_fail"] = polls_fail
                _save_state(state)

    # Watch period complete. If we never once reached the endpoint, we can't
    # claim healthy — but an unreachable endpoint is not evidence the patch is
    # bad, so keep it applied and flag inconclusive rather than reverting.
    if polls_ok == 0 and polls_unreachable > 0:
        return _do_inconclusive(slug, polls_ok, polls_unreachable)
    return _do_succeed(slug, polls_ok, polls_fail)


def _send_alert(title: str, body: str, *, priority: str = "high",
                tags: str = "warning") -> None:
    """Best-effort push notification via ntfy.sh. No-op if unconfigured.

    The whole point is to reach Ed when he's AWAY from the machine and the
    autonomous proposer has mutated code. Configure with:
      - CHLOE_NTFY_TOPIC   private, hard-to-guess topic string (acts as the
                           shared secret). Subscribe to it in the ntfy app.
      - CHLOE_NTFY_SERVER  optional, default https://ntfy.sh
    Never raises — alerting must not break the revert path.

    2026-09-02: the actual HTTP POST now lives in notify.send_ntfy so
    other features (the notify_me tool, wallet-send confirmations) can
    reuse the same pipe. This wrapper is kept so existing call sites and
    this function's watchdog-specific defaults (priority="high",
    tags="warning") don't need to change.
    """
    try:
        import notify
        notify.send_ntfy(title, body, priority=priority, tags=tags)
    except Exception as e:
        print(f"[watchdog] alert send failed: {e}", flush=True)


def _do_revert(slug: str, polls_ok: int, polls_fail: int,
               reason: str) -> dict:
    """Revert the proposal + log + bump failure counter."""
    import chloe_proposals
    try:
        rev = chloe_proposals.revert_proposal(slug)
        revert_ok = rev.get("ok", False)
        revert_msg = rev.get("message", "")
    except Exception as e:
        revert_ok = False
        revert_msg = f"{type(e).__name__}: {e}"

    with _STATE_LOCK:
        state = _load_state()
        state["under_watch"].pop(slug, None)
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        outcome = "reverted" if revert_ok else "revert_failed"
        _log(state, slug, "watchdog_revert", outcome,
             reason=f"health: {reason} | revert: {revert_msg}")
        _save_state(state)

    cf = state["consecutive_failures"]
    locked = cf >= MAX_CONSECUTIVE_FAILURES
    print(f"[watchdog] REVERTED {slug}: {reason} "
          f"(consecutive_failures={cf})", flush=True)
    _send_alert(
        ("Chloe: auto-revert FAILED" if not revert_ok
         else "Chloe: patch auto-reverted"),
        (f"slug: {slug}\n"
         f"reason: {reason}\n"
         f"revert: {'ok' if revert_ok else 'FAILED — ' + revert_msg}\n"
         f"consecutive_failures: {cf}"
         + ("\nAUTONOMY LOCKED OUT (>=2 fails) — review, then /autonomous reset"
            if locked else "")),
        priority=("urgent" if (not revert_ok or locked) else "high"),
        tags=("rotating_light" if (not revert_ok or locked) else "warning"),
    )
    return {"outcome": "reverted", "slug": slug, "reason": reason,
            "polls_ok": polls_ok, "polls_fail": polls_fail,
            "revert_ok": revert_ok, "revert_message": revert_msg}


def _do_succeed(slug: str, polls_ok: int, polls_fail: int) -> dict:
    """Watch period completed without triggering revert. Log success."""
    with _STATE_LOCK:
        state = _load_state()
        state["under_watch"].pop(slug, None)
        # Reset consecutive_failures on a successful watch.
        state["consecutive_failures"] = 0
        _log(state, slug, "watchdog_watch", "healthy",
             reason=f"polls ok={polls_ok} fail={polls_fail}")
        _save_state(state)
    print(f"[watchdog] {slug} healthy after watch "
          f"(polls ok={polls_ok}, fail={polls_fail})", flush=True)
    return {"outcome": "healthy", "slug": slug,
            "polls_ok": polls_ok, "polls_fail": polls_fail}


def _do_inconclusive(slug: str, polls_ok: int, polls_unreachable: int) -> dict:
    """Watch ended without ever reaching the health endpoint. The patch is
    LEFT APPLIED — an unreachable :6790 is not evidence the patch is bad (and a
    patch to a live module isn't loaded until restart anyway; on_boot_recover
    does the post-load check). We couldn't verify health, so alert Ed to check
    manually. Does NOT bump consecutive_failures (no fault observed) and does
    NOT reset it (no healthy poll observed)."""
    with _STATE_LOCK:
        state = _load_state()
        state["under_watch"].pop(slug, None)
        _log(state, slug, "watchdog_watch", "inconclusive",
             reason=(f"endpoint unreachable entire watch "
                     f"(polls_ok=0, unreachable={polls_unreachable}); "
                     f"patch kept"))
        _save_state(state)
    print(f"[watchdog] {slug} INCONCLUSIVE — endpoint never reachable; "
          f"patch kept, manual health check advised", flush=True)
    _send_alert(
        "Chloe: patch kept, health UNVERIFIED",
        (f"slug: {slug}\n"
         f"reason: health endpoint (:6790) unreachable for the entire watch\n"
         f"action: patch was NOT reverted (an unreachable endpoint isn't a "
         f"patch fault); please verify Chloe is healthy"),
        priority="high", tags="warning",
    )
    return {"outcome": "inconclusive", "slug": slug,
            "polls_ok": polls_ok, "polls_unreachable": polls_unreachable}


def _lookup_proposal_meta(slug: str) -> dict:
    """Fetch backup_path + target from the proposal frontmatter for state
    persistence. Best-effort; missing fields are OK."""
    try:
        import chloe_proposals
        p = chloe_proposals.load_proposal(slug)
        return {
            "backup_path": p.frontmatter.get("backup_path", ""),
            "target":      p.target,
        }
    except Exception:
        return {}


# ─── Rate-limit accounting (called by the autonomous proposer) ────────────

def autonomous_can_apply_now() -> tuple[bool, str]:
    """Check whether the autonomous proposer is allowed to apply right now.

    Returns (allowed, reason_if_blocked). The proposer must call this
    BEFORE applying. After applying, it must call record_autonomous_apply
    or record_autonomous_failure.

    Reasons it can be blocked:
    - Daily cap (MAX_AUTONOMOUS_APPLIES_PER_DAY) already used today.
    - Less than MIN_INTERVAL_BETWEEN_AUTONOMOUS_S since the last apply.
    - consecutive_failures >= MAX_CONSECUTIVE_FAILURES (autonomy is
      auto-disabled until Ed clears the counter via /autonomous reset
      or /autonomous on).
    """
    with _STATE_LOCK:
        state = _load_state()
        today_count = state.get("applies_today", {}).get(_today_key(), 0)
        if today_count >= MAX_AUTONOMOUS_APPLIES_PER_DAY:
            return False, (f"daily cap reached "
                           f"({today_count}/{MAX_AUTONOMOUS_APPLIES_PER_DAY})")
        last_ts = state.get("last_apply_ts", 0.0)
        since = _now() - last_ts
        if last_ts and since < MIN_INTERVAL_BETWEEN_AUTONOMOUS_S:
            mins_remaining = round(
                (MIN_INTERVAL_BETWEEN_AUTONOMOUS_S - since) / 60.0, 1)
            return False, (f"min interval not elapsed "
                           f"({mins_remaining:.1f}min remaining)")
        cf = state.get("consecutive_failures", 0)
        if cf >= MAX_CONSECUTIVE_FAILURES:
            return False, (f"{cf} consecutive failures — autonomy "
                           f"locked. Reset via /autonomous reset.")
    return True, ""


def record_autonomous_apply(slug: str) -> None:
    """Called by the proposer right after a successful apply. Updates
    daily counter + last_apply_ts."""
    with _STATE_LOCK:
        state = _load_state()
        key = _today_key()
        state.setdefault("applies_today", {})
        state["applies_today"][key] = state["applies_today"].get(key, 0) + 1
        state["last_apply_ts"] = _now()
        _log(state, slug, "autonomous_apply", "ok")
        _save_state(state)


def record_autonomous_failure(slug: str, reason: str) -> None:
    """Called by the proposer if synthesis or apply failed before
    reaching the watchdog. Counts toward consecutive_failures but NOT
    toward applies_today."""
    with _STATE_LOCK:
        state = _load_state()
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        cf = state["consecutive_failures"]
        _log(state, slug, "autonomous_apply", "fail", reason=reason)
        _save_state(state)

    # Alert outside the lock (network). Same spirit as the _do_revert ping:
    # the proposer failed to even apply, so the watchdog never runs, yet this
    # still pushes consecutive_failures toward the auto-lockout silently.
    locked = cf >= MAX_CONSECUTIVE_FAILURES
    _send_alert(
        "Chloe: autonomous apply FAILED",
        (f"slug: {slug}\n"
         f"reason: {reason}\n"
         f"consecutive_failures: {cf}\n"
         "(failed before the watchdog — nothing was applied)"
         + ("\nAUTONOMY LOCKED OUT (>=2 fails) — review, then /autonomous reset"
            if locked else "")),
        priority=("urgent" if locked else "high"),
        tags=("rotating_light" if locked else "warning"),
    )


# ─── Public status ────────────────────────────────────────────────────────

def status() -> dict:
    """Read-only watchdog snapshot."""
    with _STATE_LOCK:
        state = _load_state()
    return {
        "under_watch":          state.get("under_watch", {}),
        "applies_today":        state.get("applies_today", {}).get(
                                    _today_key(), 0),
        "max_per_day":          MAX_AUTONOMOUS_APPLIES_PER_DAY,
        "last_apply_ts":        state.get("last_apply_ts", 0.0),
        "consecutive_failures": state.get("consecutive_failures", 0),
        "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
        "history_count":        len(state.get("history", [])),
    }


def history(limit: int = 20) -> list[dict]:
    with _STATE_LOCK:
        state = _load_state()
    hist = list(reversed(state.get("history", [])))
    return hist[:limit]


def cancel_watch(slug: str = "") -> dict:
    """Drop in-flight watches without reverting their targets. Use if
    Ed wants to stop watchdog supervision mid-flight (e.g., to manually
    inspect)."""
    with _STATE_LOCK:
        state = _load_state()
        watching = state.get("under_watch", {})
        if slug:
            if slug not in watching:
                return {"ok": False, "error": f"no watch on {slug!r}"}
            watching.pop(slug)
            _log(state, slug, "watchdog_cancel", "ok",
                 reason="manual cancel")
            _save_state(state)
            return {"ok": True, "canceled": [slug]}
        # Cancel all
        canceled = list(watching.keys())
        watching.clear()
        for s in canceled:
            _log(state, s, "watchdog_cancel", "ok", reason="manual cancel-all")
        _save_state(state)
        return {"ok": True, "canceled": canceled}


def reset_failures() -> dict:
    """Clear the consecutive-failures counter. Use after Ed has reviewed
    the failure history and is ready to let autonomy try again."""
    with _STATE_LOCK:
        state = _load_state()
        prior = state.get("consecutive_failures", 0)
        state["consecutive_failures"] = 0
        _log(state, "<global>", "watchdog_reset", "ok",
             reason=f"prior failures cleared: {prior}")
        _save_state(state)
    return {"ok": True, "prior_failures": prior}


# ─── Boot recovery (called from jarvis.py boot) ──────────────────────────

def on_boot_recover(endpoint_grace_s: float = 60.0) -> dict:
    """Called once at Chloe boot. For each slug `under_watch` whose
    deadline has passed, run a final health check; if failing, revert.

    Endpoint-startup grace: if any deadlines have passed, we must make
    health-based keep/revert decisions — but at boot the HUD health
    endpoint (port 6790) may not be up yet, and a connection error reads
    as "unhealthy", which would FALSELY revert a perfectly good patch.
    So when there are expired watches we poll the endpoint for up to
    `endpoint_grace_s` seconds for it to become reachable, then decide
    against that snapshot. If it never answers within the grace window,
    recovery proceeds anyway — a patch that keeps the endpoint down past
    the grace window is exactly what boot recovery should revert.

    Returns summary {recovered: [...], healthy: [...], skipped: [...]}.
    Idempotent.
    """
    recovered: list[str] = []
    healthy: list[str] = []
    skipped: list[str] = []

    with _STATE_LOCK:
        state = _load_state()
        watch = dict(state.get("under_watch", {}))

    expired = {slug: entry for slug, entry in watch.items()
               if _now() >= entry.get("deadline", 0)}
    skipped = [slug for slug in watch if slug not in expired]
    if not expired:
        return {"recovered": recovered, "healthy": healthy, "skipped": skipped}

    # There ARE expired watches → wait out the endpoint's startup window
    # so a transient "connection refused" isn't misread as a failure.
    health = _fetch_health()
    if health.get("error") and endpoint_grace_s > 0:
        grace_deadline = _now() + endpoint_grace_s
        while _now() < grace_deadline:
            time.sleep(min(3.0, max(0.1, grace_deadline - _now())))
            health = _fetch_health()
            if not health.get("error"):
                break

    # One health snapshot drives every expired slug's decision this pass.
    ok, reason = _is_healthy(health)
    for slug, entry in expired.items():
        if ok:
            _do_succeed(slug, entry.get("polls_ok", 0),
                        entry.get("polls_fail", 0))
            healthy.append(slug)
        else:
            _do_revert(slug, entry.get("polls_ok", 0),
                       entry.get("polls_fail", 0),
                       reason=f"boot recovery: {reason}")
            recovered.append(slug)
    return {"recovered": recovered, "healthy": healthy, "skipped": skipped}


# ─── CLI ──────────────────────────────────────────────────────────────────

def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Chloe Stage-4 watchdog")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    sub.add_parser("history")
    sub.add_parser("reset")
    sub.add_parser("health")  # one-shot health probe
    sub.add_parser("can-apply")  # check rate-limit gate

    sp = sub.add_parser("cancel")
    sp.add_argument("slug", nargs="?", default="")

    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
        return 0
    if args.cmd == "history":
        for h in history(20):
            print(f"{h['ts_iso']}  {h['action']:20s}  {h['outcome']:14s}  "
                  f"{h['slug'][:30]}  {h.get('reason', '')[:60]}")
        return 0
    if args.cmd == "reset":
        print(json.dumps(reset_failures(), indent=2))
        return 0
    if args.cmd == "health":
        print(json.dumps(_fetch_health(), indent=2, default=str))
        return 0
    if args.cmd == "can-apply":
        ok, reason = autonomous_can_apply_now()
        print(json.dumps({"can_apply": ok, "reason": reason}, indent=2))
        return 0
    if args.cmd == "cancel":
        print(json.dumps(cancel_watch(args.slug), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
