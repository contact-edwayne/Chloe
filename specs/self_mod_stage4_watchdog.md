# Stage 4: autonomous self-mod with watchdog rollback

**Status:** designed, NOT built. **Unlocks ONLY after Stage 3 logs ≥10
successful voice-confirm cycles AND Stage 2/3 combined run 2+ months
incident-free.** This is the highest-risk tier — real bricking
possible. Do not skip the gate.

**Goal:** Chloe scans her own logs for recurring errors, drafts a
fix, applies it, restarts the affected service, watches startup and
health checks for N minutes, and auto-reverts on any failure. No
human in the loop at apply-time.

## Why this gate is so high

- A bad apply that breaks startup leaves Chloe dark until manually
  recovered. No voice path to ask for help.
- A bad apply that *silently* degrades a path (e.g. fact extraction
  silently producing garbage) can compound over hours of normal
  operation.
- The watchdog itself is now part of the trusted base. If its
  health check has false-positives (auto-reverts a good patch), Chloe
  loses fixes. If false-negatives (passes a bad patch), Chloe corrupts
  state.

## Surface area to build

### 1. Health-check endpoint in `brain_http.py`

New `GET /api/health/full` returning:

```json
{
  "ws_connected": true,
  "ollama_reachable": true,
  "groq_key_present": true,
  "wiki_embedded_count": 415,
  "auto_fact_last_run_ts": 1747700000,
  "voice_loop_alive": true,
  "memory_db_writable": true,
  "checks_ok": 7,
  "checks_failed": 0,
  "issues": []
}
```

Each check has a clear pass/fail with a one-line `issues` entry on
fail. The watchdog asks for this every 30 seconds.

### 2. New module `chloe_watchdog.py` (~200 lines)

State machine:

```
IDLE → APPLIED → WATCHING → (OK | ROLLBACK) → IDLE
```

Public API:

```python
def supervise_apply(slug: str, watch_minutes: int = 5,
                    expected_to_restart: bool = True) -> dict:
    """Called by the autonomous proposer AFTER apply_proposal succeeds.

    1. Mark this slug as 'under watch' with a deadline = now + watch_minutes.
    2. If expected_to_restart: trigger a hot-restart via WTS (`stop_chloe.bat`
       then `start_chloe.vbs`). Watch for the WS to come back within 60s.
    3. Poll /api/health/full every 30s for `watch_minutes` minutes.
    4. On any consecutive 2 failures OR WS-fail-to-reconnect: auto-revert.
    5. On a clean watch period: log success, free the slot, return ok.
    """

def status() -> dict:
    """{under_watch: [slugs], history: [{slug, outcome, ts}], counters}"""
```

### 3. Autonomous proposer job

New `chloe_jobs.py` job `autonomous-fix-recurring-errors`:

```
cron: 0 4 * * *  (4am daily, low-traffic window)
prompt:
  1. Read last 24h of logs/backend.log + logs/jobs.log.
  2. Identify error patterns that fired ≥3 times.
  3. For each: ast-introspect the implicated file via chloe_capabilities,
     synthesize a proposed fix via Groq heavy.
  4. Write proposals/code_<date>_autonomous_<slug>.md with rationale
     citing log lines.
  5. If proposer's confidence >= 0.85 (LLM-estimated) AND the diff
     touches <50 lines AND target is .py under jarvis/:
     - mint an apply-token via issue_token(applies=1, minutes=10)
     - apply_proposal_with_token(slug, token)
     - chloe_watchdog.supervise_apply(slug, watch_minutes=5,
                                     expected_to_restart=True)
  6. If watch passes: log to CHLOE_CHANGELOG.md as autonomous.
  7. If watch fails: log to wiki/episodic/autonomous_revert_<date>.md
     with full trace for Ed's morning review.
```

### 4. Hard rate limits

- `MAX_AUTONOMOUS_APPLIES_PER_DAY = 2` — even if 10 errors qualify, only
  the top 2 get applied. Rest go in `proposals/` for manual review.
- `MIN_INTERVAL_BETWEEN_AUTONOMOUS_S = 1800` (30 min). Prevents back-to-
  back applies cascading failures.
- `MAX_CONSECUTIVE_FAILURES = 2` — after two auto-reverts, the
  autonomous job is **disabled** until Ed manually re-enables.

### 5. Manual override slashes

- `/autonomous on|off` — toggle the daily job
- `/autonomous status` — last N apply attempts, current daily counter
- `/autonomous freeze <minutes>` — disable for N minutes (default 60)

## Safety design

- **No restart attempts during watch window.** Stage 4 is for in-place
  applies that are believed restart-safe. Restart-required changes
  (slash dispatch, daemon thread changes) stay manual.
- **Watchdog state survives Chloe restart.** Persisted to
  `C:\Chloe\watchdog_state.json`. On boot, if a slug is under watch
  AND the deadline is past, run the post-mortem path immediately:
  health check fail → auto-revert.
- **Failed startup detector.** `stop_chloe.bat` writes a marker file
  before exit. `start_chloe.vbs` clears it on successful boot. If the
  watchdog sees the marker file >2 minutes old, startup failed → revert
  + alert.
- **Alert path on auto-revert.** Email Ed via SMTP (configure in `.env`)
  and write a high-priority entry to `wiki/episodic/CONTEXT-<today>.md`.
- **Sandbox-test layer (optional, more work).** Before applying live,
  spin up a second Chloe instance with the proposed change, run a
  regression-test suite against it (subset of `verify_proposals.bat`
  + new behavioral tests), and only apply live if the sandbox passes.
  Estimated +10h for the sandbox. Recommend deferring to Stage 4.5.

## Build estimate

~15 hours for the core (health endpoint, watchdog state machine,
autonomous proposer job, hard limits, override slashes) WITHOUT the
sandbox layer. Add 10h for the sandbox.

## Trust gate for retiring Stage 4

This stage doesn't graduate — it just stays gated by its own rate
limits forever. There's no "Stage 5: unlimited autonomy." Two
self-mods per 24h with watchdog is the ceiling.

## Open questions for the build

1. **What counts as "Ed's morning review"?** Recommendation: the
   morning-brief job (`chloe-daily-morning-brief`) gains a section
   "Yesterday's autonomous activity" that shows applied / reverted /
   queued-for-review proposals.

2. **Should the autonomous proposer use the heavy or search model?**
   Recommendation: heavy. Web search isn't needed for fixing local
   code issues from logs, and search adds latency + a Groq quota
   risk.

3. **Should Stage 4 apply proposals that ed-authored Stage 1
   proposals haven't been touched in N days?** I.e., should Stage 4
   also process the `proposals/` backlog? Recommendation: NO. Stage
   4 only acts on errors IT detected. Ed-authored proposals stay
   manual-apply only. This keeps the two paths cleanly separated.
