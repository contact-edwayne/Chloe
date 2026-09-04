#!/usr/bin/env python
"""Smoke test for the 2026-06-02 stability + observability pass.

Run from the jarvis folder:
    venv_py314\\Scripts\\python.exe test_chloe_changes.py

Validates that everything touched this session loads cleanly in YOUR venv, runs
the Grok-hardening unit checks, and exercises the per-turn trace end-to-end
(record -> SQLite/WAL -> /whathappened + /debug turn). It points CHLOE_BRAIN_ROOT
at a throwaway temp dir, so it does NOT touch your real brain or traces.db.

NOTE: this tests the modules in isolation. The trace only fires on REAL turns via
handle_chat / the voice path — so after this passes, chat a couple of turns in the
HUD/PWA, then type `/whathappened` and `/debug turn <id>` to see live traces. (The
chloe MCP `chat` tool bypasses handle_chat and will NOT generate traces.)
"""

import os
import sys
import ast
import json
import tempfile
import traceback

JARVIS = os.path.dirname(os.path.abspath(__file__))
os.chdir(JARVIS)
sys.path.insert(0, JARVIS)

# Isolate ALL file writes to a temp brain so the real brain is never touched.
_TMP = tempfile.mkdtemp(prefix="chloe_smoketest_")
os.environ["CHLOE_BRAIN_ROOT"] = _TMP
os.makedirs(os.path.join(_TMP, "raw"), exist_ok=True)

_ok = True


def check(name, fn):
    global _ok
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        _ok = False
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


print("\n[1] Syntax / import of touched modules")
# Big, side-effecty files: parse only (don't execute their module bodies).
for _m in ("jarvis.py", "brain_wiring.py"):
    check(f"ast.parse {_m}",
          lambda _m=_m: ast.parse(open(_m, encoding="utf-8").read()))
# Standalone modules: real import (this is the check the handoff asks for).
for _m in ("chloe_lock", "chloe_embed", "chloe_read", "chloe_ed_profile",
           "chloe_dialogue_state", "chloe_memory", "wiki_embedding",
           "chloe_context", "chloe_trace"):
    check(f"import {_m}", lambda _m=_m: __import__(_m))


print("\n[2] Grok-hardening unit checks")


def _merge_checks():
    import chloe_ed_profile as P
    # conflict: opposite-polarity observation wins, flagged, lower conf
    base = [{"text": "prefers terse replies", "conf": 0.9,
             "last_seen": "2026-05-01", "evidence": 5}]
    r = P._merge_slot(base, ["no longer prefers terse replies"], source="observed")
    c = [x for x in r if "terse" in x["text"]][0]
    assert c["text"] == "no longer prefers terse replies", c
    assert c.get("conflict") and c["conf"] < 0.9, c
    # reflection quarantine: can't overwrite a confident observed belief
    base2 = [{"text": "values momentum over caution", "conf": 0.95,
              "last_seen": "2026-05-01", "evidence": 8}]
    r2 = P._merge_slot(base2, ["does not value momentum over caution"],
                       source="reflection")
    assert r2[0]["text"] == "values momentum over caution", r2


check("merge conflict + reflection quarantine", _merge_checks)
check("read-pass READ_EVERY present",
      lambda: __import__("chloe_read").READ_EVERY >= 1)
check("dialogue_state.current() exists",
      lambda: callable(__import__("chloe_dialogue_state").current))
check("embed shared-cache stats() exists",
      lambda: isinstance(__import__("chloe_embed").stats(), dict))


print("\n[3] Trace end-to-end (temp brain, WAL, full-prompt, /debug)")


def _trace_e2e():
    import chloe_trace
    # seed a dialogue-state read so the trace has mood/mode/flow to report
    open(os.path.join(_TMP, "raw", "dialogue_state.json"), "w").write(json.dumps(
        {"mood": "tired", "mode": "processing", "turn": 42, "length": "short",
         "flow": {"engagement": "low", "subtext": "says it's fine"}}))
    blocks = [
        {"name": "identity", "text": "P" * 400, "priority": 0},
        {"name": "facts",    "text": "F" * 200, "priority": 1},
        {"name": "recall",   "text": "R" * 120, "priority": 2},
        {"name": "wiki",     "text": "",        "priority": 2},
        {"name": "nsfw",     "text": "x",        "priority": 3},
    ]
    tok = chloe_trace.begin("chat")
    chloe_trace.record(
        tok, modality="chat", model="ollama:qwen2.5:32b",
        route_reason="ollama-primary", blocks=blocks,
        dropped=[("wiki", "budget"), ("nsfw", "dup")], ctx_used=5800,
        recall_block="recalled text", wiki_block="",
        read={"mood": "tired", "intent": "vent", "certainty": 0.7,
              "subtext": "says fine, isn't"},
        system="ASSEMBLED SYSTEM PROMPT (full text would be here)\n...persona...",
        retrieval_worthwhile=True)
    db = os.path.join(_TMP, "raw", "traces.db")
    assert os.path.exists(db), "traces.db not written"
    # confirm WAL took
    import sqlite3
    con = sqlite3.connect(db)
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    assert mode.lower() == "wal", f"journal_mode is {mode}, expected wal"
    print("\n--- /whathappened ---")
    print(chloe_trace.format_last(1))
    print("\n--- /debug turn 42 ---")
    print(chloe_trace.format_turn(42))


check("record + WAL + format_last + format_turn", _trace_e2e)

print("\n" + ("=" * 48))
print("ALL PASS" if _ok else "SOME CHECKS FAILED — see above")
print("=" * 48)
sys.exit(0 if _ok else 1)
