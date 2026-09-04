"""weekly_review.py — Chloe's self-fitness review.

Reads the last 7 days of Chloe's activity across her data layers and
produces a markdown brief at C:\\Chloe\\reviews\\<date>_weekly.md.

The "At a glance" metrics are deterministic (counts, lists, samples
straight from the data). The qualitative sections (Quality observations,
Coverage gaps, Prompt improvements) are LLM-synthesized via
chloe_llm_call — Groq heavy with Ollama qwen2.5:32b fallback.

This is the meta-workflow the Day-7 step of the AI-employee playbook
prescribes: a weekly pass that reviews the system's own output and
proposes improvements. Output lands in a review folder (not the repo)
so it's an operational artifact, not a portfolio one.

CLI:
    python weekly_review.py                # last 7 days, write to file
    python weekly_review.py --dry-run      # show prompt + paths, no LLM
    python weekly_review.py --days 14      # custom window
    python weekly_review.py --print-only   # write to stdout, no file
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent.resolve()


# --------------------------------------------------------------------------- #
# Standalone env loader (Task Scheduler doesn't inherit jarvis's env)         #
# --------------------------------------------------------------------------- #

def _load_env():
    envf = HERE / ".env"
    if not envf.exists():
        return
    for raw in envf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
_load_env()

sys.path.insert(0, str(HERE))


# --------------------------------------------------------------------------- #
# Data sources                                                                #
# --------------------------------------------------------------------------- #

MEMORY_DB    = HERE / "chloe_memory.db"
SOCIAL_DB    = HERE / "chloe_social.db"
FACTS_PATH   = HERE / "facts.md"
BRAIN_ROOT   = Path(r"C:\Chloe\brain")
WIKI_ENT_DIR = BRAIN_ROOT / "wiki" / "entities"
WIKI_CON_DIR = BRAIN_ROOT / "wiki" / "concepts"
REVIEWS_DIR  = Path(r"C:\Chloe\reviews")


# --------------------------------------------------------------------------- #
# Conversation activity                                                       #
# --------------------------------------------------------------------------- #

def collect_turns(days: int) -> dict:
    """Read chloe_memory.db turns from the last `days` days.

    Returns: {
        counts: {voice_user, voice_asst, chat_user, chat_asst, total},
        longest_assistant: [list of dicts ts/role/content/modality],
        random_pairs: [list of (user_turn, assistant_turn) tuples],
    }
    """
    if not MEMORY_DB.exists():
        return {"counts": {}, "longest_assistant": [], "random_pairs": [],
                "error": f"DB missing: {MEMORY_DB}"}

    cutoff = time.time() - days * 86400
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ts, role, content, modality FROM turns "
            "WHERE ts >= ? ORDER BY ts ASC",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    turns = [dict(r) for r in rows]

    counts = {
        "voice_user": sum(1 for t in turns
                          if t["modality"] == "voice" and t["role"] == "user"),
        "voice_asst": sum(1 for t in turns
                          if t["modality"] == "voice" and t["role"] == "assistant"),
        "chat_user":  sum(1 for t in turns
                          if t["modality"] == "chat"  and t["role"] == "user"),
        "chat_asst":  sum(1 for t in turns
                          if t["modality"] == "chat"  and t["role"] == "assistant"),
        "total":      len(turns),
    }

    # Top 5 longest assistant replies — proxy for highest-effort moments.
    longest = sorted(
        (t for t in turns if t["role"] == "assistant"),
        key=lambda t: -len(t["content"] or "")
    )[:5]

    # Random sample of 5 user→assistant pairs for context.
    pairs = []
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
    random.seed(int(cutoff))   # deterministic for a given window
    sample = random.sample(pairs, min(5, len(pairs))) if pairs else []

    return {
        "counts": counts,
        "longest_assistant": longest,
        "random_pairs": sample,
    }


# --------------------------------------------------------------------------- #
# Brain wiki growth                                                           #
# --------------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DATE_RE = re.compile(r"^(created|updated):\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)


def _page_dates(path: Path) -> dict:
    """Extract created/updated from frontmatter, falling back to file mtime."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:1024]
    except Exception:
        return {}
    m = _FRONTMATTER_RE.match(head)
    if not m:
        return {}
    dates = {}
    for kind, val in _DATE_RE.findall(m.group(1)):
        try:
            dates[kind] = datetime.date.fromisoformat(val)
        except ValueError:
            continue
    return dates


def collect_brain_growth(days: int) -> dict:
    """List wiki pages created or updated in the last `days` days."""
    cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
    out = {"entities_created": [], "entities_updated": [],
           "concepts_created": [], "concepts_updated": []}

    for label, folder in (("entities", WIKI_ENT_DIR), ("concepts", WIKI_CON_DIR)):
        if not folder.exists():
            continue
        for p in folder.glob("*.md"):
            dates = _page_dates(p)
            created = dates.get("created")
            updated = dates.get("updated")
            slug = p.stem
            if created and created >= cutoff_date:
                out[f"{label}_created"].append(slug)
            elif updated and updated >= cutoff_date:
                out[f"{label}_updated"].append(slug)
    for k in out:
        out[k].sort()
    return out


# --------------------------------------------------------------------------- #
# Social activity                                                             #
# --------------------------------------------------------------------------- #

def collect_social(days: int) -> dict:
    if not SOCIAL_DB.exists():
        return {"counts": {}, "samples": [], "error": f"DB missing: {SOCIAL_DB}"}

    cutoff = time.time() - days * 86400
    conn = sqlite3.connect(SOCIAL_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, platform, status, body, edited_body, rationale, "
            "source_trigger, created_at "
            "FROM drafts WHERE created_at >= ? ORDER BY created_at ASC",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    drafts = [dict(r) for r in rows]
    counts = {}
    for d in drafts:
        key = f"{d['platform']}/{d['status']}"
        counts[key] = counts.get(key, 0) + 1

    # Pick up to 3 examples spanning the status range
    samples = []
    seen_status = set()
    for d in drafts:
        if d["status"] in seen_status:
            continue
        samples.append(d)
        seen_status.add(d["status"])
        if len(samples) >= 3:
            break

    return {"counts": counts, "samples": samples, "total": len(drafts)}


# --------------------------------------------------------------------------- #
# Facts added                                                                 #
# --------------------------------------------------------------------------- #

_FACT_LINE_RE = re.compile(
    r"^-\s+(.*?)\s+\*\(added (\d{4}-\d{2}-\d{2})\)\*\s*$",
    re.MULTILINE,
)


def collect_facts(days: int) -> dict:
    if not FACTS_PATH.exists():
        return {"new_facts": [], "total_known": 0}

    text = FACTS_PATH.read_text(encoding="utf-8", errors="replace")
    cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
    new_facts = []
    total = 0
    for body, datestr in _FACT_LINE_RE.findall(text):
        total += 1
        try:
            added = datetime.date.fromisoformat(datestr)
        except ValueError:
            continue
        if added >= cutoff_date:
            new_facts.append({"fact": body.strip(), "added": datestr})
    return {"new_facts": new_facts, "total_known": total}


# --------------------------------------------------------------------------- #
# Deterministic summary builder                                               #
# --------------------------------------------------------------------------- #

def _truncate(text: str, n: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[:n - 1].rstrip() + "…"


def render_at_a_glance(turns, brain, social, facts, window_days: int) -> str:
    c = turns.get("counts", {})
    lines = [
        f"- **Conversation:** {c.get('total', 0)} turns "
        f"(voice {c.get('voice_user', 0)}↔{c.get('voice_asst', 0)}, "
        f"chat {c.get('chat_user', 0)}↔{c.get('chat_asst', 0)})",
    ]

    ents_new = brain["entities_created"]
    ents_upd = brain["entities_updated"]
    cons_new = brain["concepts_created"]
    cons_upd = brain["concepts_updated"]
    if any((ents_new, ents_upd, cons_new, cons_upd)):
        parts = []
        if ents_new: parts.append(f"{len(ents_new)} new entities")
        if ents_upd: parts.append(f"{len(ents_upd)} updated entities")
        if cons_new: parts.append(f"{len(cons_new)} new concepts")
        if cons_upd: parts.append(f"{len(cons_upd)} updated concepts")
        lines.append(f"- **Brain growth:** {', '.join(parts)}")
    else:
        lines.append("- **Brain growth:** none")

    sc = social.get("counts", {})
    if sc:
        breakdown = ", ".join(f"{k} ×{v}" for k, v in sorted(sc.items()))
        lines.append(f"- **Social:** {social.get('total', 0)} drafts ({breakdown})")
    else:
        lines.append("- **Social:** none")

    nf = facts.get("new_facts", [])
    if nf:
        lines.append(f"- **Facts added:** {len(nf)} "
                     f"(running total: {facts.get('total_known', 0)})")
    else:
        lines.append(f"- **Facts added:** 0 "
                     f"(running total: {facts.get('total_known', 0)})")

    return "\n".join(lines)


def render_notable_exchanges(turns) -> str:
    longest = turns.get("longest_assistant", [])
    pairs = turns.get("random_pairs", [])
    blocks = []

    if longest:
        blocks.append("**Highest-effort assistant replies (by length):**")
        for t in longest:
            ts = datetime.datetime.fromtimestamp(t["ts"]).strftime("%a %H:%M")
            blocks.append(f"- *{ts} · {t['modality']}* — {_truncate(t['content'], 200)}")
        blocks.append("")

    if pairs:
        blocks.append("**Random sample exchanges:**")
        for u, a in pairs:
            ts = datetime.datetime.fromtimestamp(u["ts"]).strftime("%a %H:%M")
            blocks.append(f"\n- *{ts} · {u['modality']}*")
            blocks.append(f"  - **Ed:** {_truncate(u['content'], 180)}")
            blocks.append(f"  - **Chloe:** {_truncate(a['content'], 220)}")

    return "\n".join(blocks) if blocks else "_(no conversation data in window)_"


def render_brain_detail(brain) -> str:
    sections = []
    if brain["entities_created"]:
        sections.append("- new entities: "
                        + ", ".join(f"`{s}`" for s in brain["entities_created"]))
    if brain["entities_updated"]:
        sections.append("- updated entities: "
                        + ", ".join(f"`{s}`" for s in brain["entities_updated"]))
    if brain["concepts_created"]:
        sections.append("- new concepts: "
                        + ", ".join(f"`{s}`" for s in brain["concepts_created"]))
    if brain["concepts_updated"]:
        sections.append("- updated concepts: "
                        + ", ".join(f"`{s}`" for s in brain["concepts_updated"]))
    return "\n".join(sections) if sections else "_(no wiki changes)_"


def render_social_detail(social) -> str:
    samples = social.get("samples", [])
    if not samples:
        return "_(no drafts in window)_"
    lines = []
    for d in samples:
        body = _truncate(d.get("edited_body") or d.get("body", ""), 180)
        rat = _truncate(d.get("rationale", ""), 120)
        lines.append(f"- **{d['platform']}/{d['status']}** "
                     f"(trigger: {d.get('source_trigger', '?')}) — {body}")
        if rat:
            lines.append(f"  - *angle:* {rat}")
    return "\n".join(lines)


def render_facts_detail(facts) -> str:
    nf = facts.get("new_facts", [])
    if not nf:
        return "_(no new facts)_"
    return "\n".join(f"- `{f['added']}` — {f['fact']}" for f in nf)


# --------------------------------------------------------------------------- #
# LLM synthesis                                                               #
# --------------------------------------------------------------------------- #

# Behavior-shaping files Edward actually edits. The LLM must reference one of
# these symbols (or admit when the fix isn't covered) — never invent paths.
# Edit this constant when new behavior-shaping files are added to the project.
BEHAVIOR_FILE_ROSTER = """\
PERSONA & VOICE:
  - chloe_about.md  — main persona file: voice, capabilities, social rules,
                      knowledge anchors, seed preferences. Edit here for any
                      tone / voice / character / formatting fix.
  - facts.md        — Ed's long-term facts about himself, append-only. Edit
                      here for stale or missing personal context.

ROUTING & SYSTEM PROMPTS (inside jarvis.py):
  - jarvis.py:_REALTIME_KEYWORDS      — substring keywords that route to
                                         Groq compound-mini for web search.
  - jarvis.py:_INTROSPECTION_KEYWORDS — keywords that force Groq fast Llama
                                         for source-code questions. Bare
                                         words like "thanks" false-positive
                                         here; tighten if seen in samples.
  - jarvis.py:_HEDGING_PATTERNS       — fingerprints that trigger the
                                         hedge-fallback retry chain (Groq
                                         compound -> Brave search).
  - jarvis.py:_pick_route             — voice/chat routing function.
  - jarvis.py voice/chat system preamble blocks — the per-turn system
                                         prompts injected before model
                                         calls. Search for "preamble".

KNOWLEDGE LAYER PROMPTS:
  - brain.py                 — wiki ingestion: entity & concept extraction.
  - brain_wiring.py          — /ingest /query /add /lint command routing.
  - daily_context.py         — 6am daily context-synthesis prompt.
  - weekly_review.py         — THIS script's qualitative-section prompt.

TOOL PROMPTS:
  - social_composer.py       — Bluesky / LinkedIn post drafting.
  - lights.py:parse_intent   — natural-language light command parsing.
  - search.py + jarvis.py /search handler — Brave web-search synthesis.

MEMORY:
  - chloe_memory.py          — semantic recall threshold, FTS5 fallback,
                                turn log schema. Edit for recall precision.

If a needed fix is genuinely not addressable by any file above, say so
explicitly in the proposal — do not invent a file name."""


QUALITATIVE_PROMPT = """You are reviewing Chloe's past {days}-day activity.
Chloe is a personal AI assistant — voice + chat with one user (Edward), backed
by a brain wiki, semantic memory, and tool integrations (web search, lights,
social drafts, Lightning wallet).

Below is a structured summary of last week's signal. Use ONLY this data.
Don't invent metrics or claims not present below.

Produce a markdown document with EXACTLY these three sections, in order:

## Quality observations
2–5 bullets. What patterns do you see in the conversation samples? Where does
Chloe seem to be doing good work? Where does she sound like a corporate AI,
or hedge, or confabulate? Cite specific exchanges by quoting a few words.

## Coverage gaps
2–5 bullets. What kinds of questions or tasks does the data suggest she's
missing — or handling poorly? What capability would have helped on the
weakest exchanges? Be concrete, not abstract.

## Prompt improvement proposals
EXACTLY TWO numbered proposals. Each one MUST:
  (a) Name a specific file (and symbol/section if relevant) from the
      EDITABLE BEHAVIOR FILES list below. Do NOT invent file names — if
      the right fix isn't covered by the list, say so explicitly.
  (b) Describe the exact change: what to add, remove, or modify.
  (c) Explain why — what behaviour the change fixes, citing the
      conversation evidence (quote a few words from a specific exchange).

Keep the whole document under 600 words. No preamble before "## Quality
observations". No closing summary.

---

EDITABLE BEHAVIOR FILES (use these in proposals, never invent paths):

{file_roster}

---

WINDOW: last {days} days, ending {end_date}

AT A GLANCE:
{at_a_glance}

NOTABLE EXCHANGES:
{notable}

BRAIN ACTIVITY:
{brain_detail}

SOCIAL ACTIVITY:
{social_detail}

NEW FACTS ABOUT EDWARD:
{facts_detail}
"""


def build_prompt(window_days, end_date,
                 turns, brain, social, facts) -> str:
    return QUALITATIVE_PROMPT.format(
        days=window_days,
        end_date=end_date,
        file_roster=BEHAVIOR_FILE_ROSTER,
        at_a_glance=render_at_a_glance(turns, brain, social, facts, window_days),
        notable=render_notable_exchanges(turns),
        brain_detail=render_brain_detail(brain),
        social_detail=render_social_detail(social),
        facts_detail=render_facts_detail(facts),
    )


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #

def render_full_document(window_days, end_date,
                         turns, brain, social, facts,
                         qualitative: str) -> str:
    return f"""# Weekly Review — {end_date} (last {window_days} days)

*Generated by `weekly_review.py` on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}*

## At a glance

{render_at_a_glance(turns, brain, social, facts, window_days)}

## Notable exchanges

{render_notable_exchanges(turns)}

## Brain activity

{render_brain_detail(brain)}

## Social activity

{render_social_detail(social)}

## New facts about Edward

{render_facts_detail(facts)}

---

{qualitative.strip()}
"""


def _output_path(end_date: str) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    base = REVIEWS_DIR / f"{end_date}_weekly.md"
    if not base.exists():
        return base
    # Collision protection — never overwrite
    for n in range(2, 100):
        candidate = REVIEWS_DIR / f"{end_date}_weekly-{n}.md"
        if not candidate.exists():
            return candidate
    return base  # give up; will overwrite


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def generate(window_days: int = 7,
             dry_run: bool = False,
             print_only: bool = False) -> dict:
    end_date = datetime.date.today().isoformat()

    print(f"[weekly-review] gathering data ({window_days} days)…", flush=True)
    turns  = collect_turns(window_days)
    brain  = collect_brain_growth(window_days)
    social = collect_social(window_days)
    facts  = collect_facts(window_days)

    prompt = build_prompt(window_days, end_date, turns, brain, social, facts)

    if dry_run:
        out = _output_path(end_date)
        print(f"[weekly-review] DRY RUN — would write to {out}")
        print(f"[weekly-review] prompt size: {len(prompt)} chars")
        print("--- PROMPT PREVIEW ---")
        print(prompt[:3000] + ("…" if len(prompt) > 3000 else ""))
        return {"ok": True, "dry_run": True, "path": str(out)}

    # Call the LLM. chloe_llm_call is in brain_wiring; "heavy" tries Groq
    # first and falls back to local Ollama qwen2.5:32b on failure.
    print("[weekly-review] calling LLM for qualitative synthesis…", flush=True)
    try:
        from brain_wiring import chloe_llm_call
        qualitative = chloe_llm_call(prompt, "heavy")
    except Exception as e:
        print(f"[weekly-review] LLM call failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        qualitative = ("## Quality observations\n\n_(LLM synthesis failed — "
                       f"deterministic sections above are still accurate. "
                       f"Error: {type(e).__name__})_\n")

    full = render_full_document(window_days, end_date,
                                turns, brain, social, facts, qualitative)

    if print_only:
        print(full)
        return {"ok": True, "bytes": len(full), "print_only": True}

    out_path = _output_path(end_date)
    out_path.write_text(full, encoding="utf-8")
    print(f"[weekly-review] wrote {len(full):,} bytes → {out_path}")
    return {"ok": True, "bytes": len(full), "path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=7,
                    help="window size in days (default 7)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show prompt + paths without calling LLM")
    ap.add_argument("--print-only", action="store_true",
                    help="print the full doc to stdout instead of writing")
    args = ap.parse_args()

    try:
        result = generate(window_days=args.days,
                          dry_run=args.dry_run,
                          print_only=args.print_only)
    except KeyboardInterrupt:
        print("\n[weekly-review] interrupted", file=sys.stderr)
        return 130

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
