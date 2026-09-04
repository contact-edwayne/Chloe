# chloe_jobs — local scheduled tasks (Cowork-free)

Ported 2026-05-19 (afternoon → evening). Replaces 13 Cowork-scheduled
tasks with local Python equivalents so they don't burn Anthropic API
tokens on every run.

## What this is

A single Python module — `chloe_jobs.py` — that exposes one function
per Cowork scheduled task. Each function uses Chloe's existing local
primitives (Groq llama-3.3-70b via `_heavy_call`, Ollama qwen2.5:32b
via `_light_call`, Groq compound-mini via `_search_call`, Brave via
`search.web_search`, `BRAIN.ingest`, `ChloeMemory.search_turns`). No
MCP, no Claude API tokens.

Plus:
- `chloe_jobs.bat` — Windows wrapper invoked by Task Scheduler.
- `register_chloe_jobs.ps1` — one-time setup script that creates all
  13 Windows Task Scheduler entries under `\Chloe\<job-name>` with
  the same cron times as the Cowork originals.

## What's ported

| Job | Cron | Old Cowork task |
|---|---|---|
| `daily-journal-stub` | 23:00 daily | `chloe-daily-journal-stub` |
| `daily-cowork-fact-extract` | 22:30 daily | `chloe-daily-cowork-fact-extract` |
| `daily-voice-persona-mining` | 22:00 daily | `chloe-daily-voice-persona-mining` |
| `daily-topic-rotation` | 06:00 Mon-Sat | `chloe-daily-topic-rotation` |
| `daily-finance-ingest` | 07:30 weekdays | `chloe-daily-finance-ingest` |
| `daily-morning-brief` | 07:00 daily | `chloe-daily-morning-brief` |
| `daily-critical-thinking-exercise` | 13:00 weekdays | `chloe-daily-critical-thinking-exercise` |
| `weekly-backup` | Sun 03:00 | `chloe-weekly-backup` |
| `weekly-autonomous-audit` | Sun 04:00 | `chloe-weekly-autonomous-audit` |
| `weekly-persona-drift` | Sun 05:00 | `chloe-weekly-persona-drift` |
| `weekly-persona-evolution` | Sun 06:00 | `chloe-weekly-persona-evolution` |
| `weekly-cross-domain-synthesis` | Sun 09:00 | `chloe-weekly-cross-domain-synthesis` |
| `friday-meta-review` | Fri 08:00 | `chloe-friday-meta-review` |

## Setup

1. **One-time:** register the Windows Task Scheduler entries.

   ```
   powershell -ExecutionPolicy Bypass -File register_chloe_jobs.ps1
   ```

   This creates `\Chloe\daily-journal-stub`, `\Chloe\daily-finance-ingest`,
   etc. Tasks run as the current user, only while logged on (matches
   Cowork's "runs while app is open" semantics — set `/ru SYSTEM` in the
   script if you want background runs, but the jobs need user-scope
   env vars from `.env`).

2. **Verify one job manually:**

   ```
   chloe_jobs.bat daily-journal-stub
   ```

   Tail `logs\chloe_jobs.log` for output. Should see `=== START daily-journal-stub ===` and `=== OK    daily-journal-stub (...): <result>`.

3. **Run each job once before its first scheduled fire** to confirm
   the LLM / Brave / disk pipeline works end-to-end. Some jobs require
   the day-of-week to match (e.g. `daily-topic-rotation` exits early
   on Sunday) — manually set a different DOW or just verify on the
   right day.

4. **After a job's local version is verified, disable the matching
   Cowork task** to avoid duplicate output:

   ```python
   # In Cowork:
   mcp__scheduled-tasks__update_scheduled_task(
     taskId="chloe-daily-journal-stub",
     enabled=False)
   ```

   Or just toggle them off in the Cowork "Scheduled" sidebar.

## Token / cost picture (back-of-envelope)

Per Cowork run estimated 2-10k Claude API tokens depending on task
complexity. 13 jobs × ~5k avg × 7 runs/week ≈ 450k tokens/week saved.

After migration:
- Groq llama-3.3-70b (free tier): ~100k tokens/day = 700k/week ceiling.
  Plenty of headroom.
- Groq compound-mini (free tier, separate quota): used by jobs that
  need web search; quota-shared with `_heavy_call` so `_search_call`
  falls through to Ollama on 429 (already wired in `brain_wiring._search_call`).
- Ollama qwen2.5:32b: free + unlimited, ~85s cold load / ~10 tok/s
  steady. Used as the bottom of the fallback chain.
- Brave Search API: $5/mo free credit, ~1000 queries/mo. The daily-
  finance-ingest is the heaviest user — one query per watchlist
  ticker per weekday. Stay under by capping ticker count.

## Troubleshooting

**Job logs only show `=== START ===` and no completion?**
Check `logs/chloe_jobs.log`. The job probably crashed inside an LLM
call — look for the `traceback`. Run manually with `chloe_jobs.bat
<job>` for live output.

**LLM returns empty / `(no specific discussion)`?**
Probably hit a Groq quota AND Ollama wasn't running. Check `ollama
serve` is up via `curl http://localhost:11434/api/tags`. The fallback
chain is `_heavy_call` → `_light_call` (Ollama) → empty string.

**Job runs but writes nothing?**
The job's input source may be empty (e.g. `daily-voice-persona-mining`
returns "no voice turns in last 24h, skipping" when you didn't talk
to Chloe yesterday). That's normal.

**Path errors writing to brain?**
Confirm `CHLOE_BRAIN_ROOT` env var (default `C:\Chloe\brain`) is
correct + writable from the task's run context.

**Task Scheduler says "Running" forever?**
The python.exe may be waiting on a cold Ollama load (~85s) or a slow
Brave response. 5-minute timeout in `backup_chloe.py` subprocess call.
Other jobs don't have explicit timeouts; if a job hangs > 5 min, end
the task manually and check logs.

## File map

- `chloe_jobs.py` — the module + CLI entry point
- `chloe_jobs.bat` — Windows wrapper
- `register_chloe_jobs.ps1` — Task Scheduler setup
- `logs/chloe_jobs.log` — rotating weekly log
- `backup_chloe.py` — invoked by `weekly-backup` (durable, already existed)

## Reverting

If a local job is producing worse output than its Cowork twin:

1. Re-enable the Cowork task: `mcp__scheduled-tasks__update_scheduled_task(taskId="chloe-...", enabled=True)`.
2. Delete the local Task Scheduler entry: `schtasks /delete /tn "Chloe\<name>" /f`.

No data loss — both write to the same brain folder; output formats
are aligned so downstream consumers don't care which produced a given
file.
