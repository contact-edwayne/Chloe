# Autonomous Brain Workflows

Two background workflows make Chloe's brain write back to itself while you're not actively using it.

## 1. Daily Context Generator (6:00 AM daily)

Reads:
- yesterday's episodic file
- the past 7 days of episodic
- all durable facts
- last 48 hours of new sources in `raw/`
- last 48 hours of modified wiki pages

Calls Groq heavy with a structured synthesis prompt and writes the result to `episodic/CONTEXT-<today>.md` with these sections:

- Project Status
- Open Loops
- Emerging Patterns
- Suggested Focus

## 2. Queue Processor (every 2 hours)

Watches `C:\Chloe\brain\queue\` for files named:

```
RESEARCH-<slug>.md
SYNTHESIZE-<slug>.md
DRAFT-<slug>.md
ANALYZE-<slug>.md
```

For each file: reads the body for the user's instructions, routes to the right handler, calls Groq heavy, writes output to `brain/generated/<date>/<verb>-<slug>.md`, and moves the queue file to `brain/archive/queue/<date>-<filename>`.

Drop a file in the queue at midnight, output is in `generated/` by 2 AM.

### Verbs

- **RESEARCH** — Topic query against the wiki + facts. Output sections: TL;DR, What the Brain Knows, What's Missing, Open Questions.
- **SYNTHESIZE** — Cross-source synthesis using relevant wiki pages. Surfaces agreement, disagreement, gaps.
- **DRAFT** — Long-form writing using the brain as source material. Outputs an article-shaped document.
- **ANALYZE** — Pattern / contradiction / gap analysis on a slice of the brain.

### Example queue file

`C:\Chloe\brain\queue\RESEARCH-rag_vs_llm_wiki.md`

```
What does my brain know about the difference between RAG and the
Karpathy LLM Wiki pattern? What's still missing if I want to write
a comparison article?
```

After the next 2-hour drain, expect:

```
C:\Chloe\brain\generated\2026-05-10\research-rag_vs_llm_wiki.md
C:\Chloe\brain\archive\queue\2026-05-10-RESEARCH-rag_vs_llm_wiki.md
```

## Setup (Windows Task Scheduler)

Open Task Scheduler. Create two new tasks:

### Chloe Daily Context

- Trigger: Daily at 6:00 AM
- Action: Start a program
- Program: `C:\Users\eleew\Documents\jarvis\daily_context.bat`
- Start in: `C:\Users\eleew\Documents\jarvis`

### Chloe Queue Processor

- Trigger: Daily at 12:00 AM, then "Repeat task every 2 hours" for 1 day
- Action: Start a program
- Program: `C:\Users\eleew\Documents\jarvis\queue_processor.bat`
- Start in: `C:\Users\eleew\Documents\jarvis`

Logs land in `C:\Users\eleew\Documents\jarvis\logs\daily_context.log` and `queue_processor.log`.

## Manual / on-demand use

Both scripts accept CLI flags. Useful for testing.

```cmd
cd C:\Users\eleew\Documents\jarvis
.venv\Scripts\activate

REM Generate today's context interactively
python daily_context.py

REM Show what would be sent without calling the LLM
python daily_context.py --dry-run

REM Backfill a missed day
python daily_context.py --date 2026-05-09

REM Drain the queue right now
python queue_processor.py

REM Drain just the first task and exit
python queue_processor.py --once

REM Show what would be processed without calling the LLM
python queue_processor.py --dry-run
```

## Cost estimate

- Daily Context: 1 heavy Groq call per day (~3000 tokens out, ~$0.05)
- Queue Processor: 1 heavy Groq call per queued task. If you queue 5 tasks per week that's ~$0.25/week.

Both well within Groq's free daily quota for normal use.
