"""Reflection job for Chloe — Phase 4E of the 2026-06-01 plan.

The generative-agents move: periodically read recent conversation, distill
higher-order observations about Ed, and MERGE them into the typed user-model
(`ed_model.json`) with evidence — so memory becomes *understanding*, not just a
log. Self-contained: reads the turn log (SQLite) and facts.md directly and hands
them to `chloe_ed_profile.build()`, which asks a local model for slot updates and
merges them incrementally (never wholesale).

Cold path only — meant to run from a scheduled job (off-hours) or on demand.
Never on a live turn. Pure-ish + never raises.

Run manually:  python chloe_reflect.py
From a job:    chloe_reflect.reflect()
"""

from __future__ import annotations

import os
import sqlite3


def _here(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _recent_turns_text(db_path: str, n: int) -> str:
    """Pull the last `n` turns straight from the log (read-only). Skips slash
    commands and recall echoes so reflection sees real conversation."""
    try:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT role, content FROM turns ORDER BY ts DESC LIMIT ?",
                (n,),
            ).fetchall()
        finally:
            con.close()
        rows = list(reversed(rows))
        out = []
        for role, content in rows:
            c = " ".join(str(content or "").split())
            if not c or c.startswith("/"):
                continue
            out.append(f"{role}: {c[:300]}")
        return "\n".join(out)
    except Exception:
        return ""


def reflect(n_turns: int = 40, db_path: str = None, facts_path: str = None) -> str:
    """Read recent turns + facts, synthesize higher-order observations, and write
    them as a PENDING proposal (Phase-5 demote — no longer mutates ed_model.json
    unattended; Ed applies it with /accept_reflection). Returns a short summary."""
    try:
        import chloe_ed_profile
    except Exception:
        return ""
    db_path = db_path or _here("chloe_memory.db")
    facts_path = facts_path or _here("facts.md")

    recent = _recent_turns_text(db_path, n_turns)
    facts = ""
    try:
        with open(facts_path, "r", encoding="utf-8") as fh:
            facts = fh.read()
    except Exception:
        facts = ""

    if not (recent.strip() or facts.strip()):
        return ""
    # propose() synthesizes slot updates and stages them as a pending proposal;
    # it does NOT merge. Ed reviews via /reflection and applies via
    # /accept_reflection. This removes the autonomous daily model mutation.
    payload = chloe_ed_profile.propose(facts_body=facts, recent=recent)
    try:
        chloe_ed_profile.tick_relationship()
    except Exception:
        pass
    if not payload:
        return "[reflect] nothing worth proposing"
    n = sum(len(v) for v in (payload.get("updates") or {}).values())
    return f"[reflect] staged {n} observation(s) — review with /reflection, apply with /accept_reflection"


if __name__ == "__main__":
    out = reflect()
    print(out or "[reflect] no update produced")
