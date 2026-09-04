"""Per-turn observability trace for Chloe.

When a reply feels off, you currently can't see *why*: which mood she read, what
recall surfaced, which context blocks the composer dropped for budget, whether
the read-pass fired. This records one compact trace per turn — the decisions that
shaped the reply — to a small SQLite table plus an in-memory ring buffer, so
"why did that reply feel off?" becomes a lookup (`/whathappened`) instead of a
guess.

It captures, not recomputes: callers hand it what they already have at the
compose point (model, route, recall/wiki blocks, the block list + dropped list,
tokens used, the read-pass result); the trace enriches that with the persisted
dialogue-state read and the embed-cache delta. Best-effort + never raises —
tracing must never break a turn, so every failure degrades to a smaller trace.

Public API:
    begin(modality) -> dict | None          # mark turn start (captures t0)
    record(token=None, **fields) -> dict     # enrich + persist one turn
    format_last(n=1) -> str                  # human gloss (/whathappened)
    recent(n=20) -> list[dict]
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

RING_MAX = int(os.environ.get("CHLOE_TRACE_RING", "50"))
ENABLED = os.environ.get("CHLOE_TRACE", "1") != "0"
# Store the full assembled system prompt per turn so /debug turn N can replay
# exactly what the model saw. Adds ~prompt-size bytes/turn to traces.db; set 0 to
# keep only the length.
STORE_PROMPT = os.environ.get("CHLOE_TRACE_PROMPT", "1") != "0"

_RING: "deque[dict]" = deque(maxlen=RING_MAX)
_LAST_EMBED = {"hits": 0, "misses": 0}
_DB_READY = False


def _db_path() -> str:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    return os.path.join(root, "raw", "traces.db")


def _connect():
    import sqlite3
    global _DB_READY
    p = _db_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p, timeout=2.0)
    if not _DB_READY:
        # WAL from day one: concurrent reader (/whathappened) + writer don't block,
        # and it's the on-ramp for later folding dialogue_state into this DB and
        # retiring the JSON files + advisory lock.
        try:
            con.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        con.execute(
            "CREATE TABLE IF NOT EXISTS turn_traces ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, turn_id INTEGER, "
            "modality TEXT, data TEXT)")
        con.commit()
        _DB_READY = True
    return con


def begin(modality: str = "chat") -> dict | None:
    """Mark the start of a turn so record() can compute pre-LLM latency.
    Returns an opaque token (or None if tracing is off)."""
    if not ENABLED:
        return None
    return {"t0": time.perf_counter(), "modality": modality}


def _embed_delta() -> dict:
    """Per-turn embed-cache hit/miss delta from chloe_embed's cumulative stats."""
    try:
        import chloe_embed
        cur = chloe_embed.stats()
        d = {"hits": cur.get("hits", 0) - _LAST_EMBED["hits"],
             "misses": cur.get("misses", 0) - _LAST_EMBED["misses"]}
        _LAST_EMBED["hits"] = cur.get("hits", 0)
        _LAST_EMBED["misses"] = cur.get("misses", 0)
        return {"hits": max(0, d["hits"]), "misses": max(0, d["misses"])}
    except Exception:
        return {"hits": 0, "misses": 0}


def _state_read() -> dict:
    """Pull this turn's persisted dialogue-state read (best-effort)."""
    try:
        import chloe_dialogue_state
        st = chloe_dialogue_state.current()
        return {
            "turn": st.get("turn"),
            "mood": st.get("mood"),
            "mode": st.get("mode"),
            "length": st.get("length"),
            "flow": st.get("flow", {}) or {},
        }
    except Exception:
        return {}


def record(token: dict = None, *, modality: str = None, model: str = None,
           route_reason: str = None, blocks=None, dropped=None, ctx_used=None,
           recall_block: str = None, wiki_block: str = None, read: dict = None,
           system: str = None, retrieval_worthwhile=None, **extra) -> dict:
    """Assemble + persist one turn's trace. Never raises; returns the trace dict
    (possibly partial). `blocks` is the candidate list handed to the composer;
    `dropped` is the composer's list of (name, reason)."""
    if not ENABLED:
        return {}
    try:
        st = _state_read()
        dropped = list(dropped or [])
        dropped_names = {d[0] for d in dropped if isinstance(d, (list, tuple)) and d}
        included = [b.get("name") for b in (blocks or [])
                    if isinstance(b, dict) and b.get("text")
                    and b.get("name") not in dropped_names]
        # Per-block token estimate (the composer's own len/4) so the trace shows
        # the priority+token math behind every keep/drop.
        try:
            import chloe_context
            block_tokens = {b.get("name"): chloe_context.est_tokens(b.get("text", ""))
                            for b in (blocks or [])
                            if isinstance(b, dict) and b.get("text")}
        except Exception:
            block_tokens = {}
        # Mood ownership (Phase 5): the heuristic owns mood; the read-pass only
        # advises. Record the owned mood's source ("heuristic") plus what the
        # read-pass thought, and flag drift — so you can SEE when the heuristic
        # and the 3B disagree without the read-pass silently overriding.
        _r = read or {}
        owned_mood = st.get("mood")
        read_mood = _r.get("mood")
        mood_source = "heuristic"
        try:
            mood_drift = bool(read_mood and read_mood != owned_mood
                              and float(_r.get("certainty", 0) or 0) >= 0.5)
        except Exception:
            mood_drift = False
        tr = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "turn_id": st.get("turn"),
            "modality": modality or (token or {}).get("modality") or "chat",
            "model": model,
            "route_reason": route_reason,
            "mood": st.get("mood"),
            "mode": st.get("mode"),
            "length": st.get("length"),
            "flow": st.get("flow", {}),
            "read_fired": bool(read),
            "read": read or {},
            "mood_source": mood_source,
            "mood_read": read_mood,
            "mood_drift": mood_drift,
            "recall_chars": len(recall_block or ""),
            "wiki_chars": len(wiki_block or ""),
            "embed": _embed_delta(),
            "ctx_used": ctx_used,
            "blocks_included": included,
            "blocks_dropped": [f"{d[0]}({d[1]})" for d in dropped
                               if isinstance(d, (list, tuple)) and len(d) >= 2],
            "block_tokens": block_tokens,
            "system_len": len(system or "") if system is not None else None,
            "system_full": (system if (STORE_PROMPT and system is not None) else None),
            "retrieval_worthwhile": retrieval_worthwhile,
        }
        if token and token.get("t0") is not None:
            tr["pre_llm_ms"] = round((time.perf_counter() - token["t0"]) * 1000)
        if extra:
            tr.update(extra)

        _RING.append(tr)
        try:
            con = _connect()
            try:
                con.execute(
                    "INSERT INTO turn_traces (ts, turn_id, modality, data) "
                    "VALUES (?,?,?,?)",
                    (tr["ts"], tr.get("turn_id"), tr["modality"], json.dumps(tr)))
                con.commit()
            finally:
                con.close()
        except Exception:
            pass  # ring buffer still has it
        return tr
    except Exception:
        return {}


def recent(n: int = 20) -> list:
    """Last n traces, newest last. From the ring; falls back to the DB."""
    if _RING:
        return list(_RING)[-n:]
    try:
        con = _connect()
        try:
            rows = con.execute(
                "SELECT data FROM turn_traces ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        finally:
            con.close()
        out = []
        for (data,) in reversed(rows):
            try:
                out.append(json.loads(data))
            except Exception:
                pass
        return out
    except Exception:
        return []


def _gloss(tr: dict) -> str:
    """One-line summary: the fast 'why did this reply feel off' read."""
    bits = []
    head = f"turn {tr.get('turn_id', '?')} {tr.get('modality', '?')}"
    mood = tr.get("mood", "?")
    mode = tr.get("mode", "?")
    if tr.get("read_fired") and tr.get("mood_read"):
        drift = "⚠drift" if tr.get("mood_drift") else ""
        head += (f": {mood}/{mode} (llm-read {tr.get('mood_read')} "
                 f"c{tr.get('read', {}).get('certainty', '?')}{drift})")
    else:
        head += f": {mood}/{mode}"
    bits.append(head)

    flow = tr.get("flow", {}) or {}
    flags = [k for k in ("circling", "repeated_correction") if flow.get(k)]
    if flow.get("engagement") and flow["engagement"] != "med":
        flags.append(f"eng={flow['engagement']}")
    if flow.get("subtext"):
        flags.append("subtext")
    if flags:
        bits.append(",".join(flags))

    rc = "recall✓" if tr.get("recall_chars") else "recall✗"
    if tr.get("recall_suppressed"):
        rc += f"(✂{tr['recall_suppressed']})"   # dups suppressed vs live window
    if tr.get("callback_suppressed"):
        rc += f"(ttl{tr['callback_suppressed']})"  # callbacks resting
    wk = "wiki✓" if tr.get("wiki_chars") else "wiki✗"
    emb = tr.get("embed", {}) or {}
    bits.append(f"{rc} {wk} embed h{emb.get('hits', 0)}/m{emb.get('misses', 0)}")

    ctx = f"ctx {tr.get('ctx_used', '?')}tok"
    if tr.get("blocks_dropped"):
        ctx += " drop:" + ",".join(tr["blocks_dropped"])
    bits.append(ctx)

    if tr.get("pre_llm_ms") is not None:
        bits.append(f"{tr['pre_llm_ms']}ms")
    return " | ".join(bits)


def format_last(n: int = 1) -> str:
    """Human-readable last-n traces for the /whathappened slash + MCP."""
    trs = recent(n)
    if not trs:
        return "no turns traced yet."
    out = []
    for tr in trs[-n:]:
        out.append(_gloss(tr))
        if n == 1:
            # Expanded detail for a single-turn look.
            out.append(f"  model: {tr.get('model')}  route: {tr.get('route_reason')}"
                       f"  mood_src: {tr.get('mood_source')}")
            if tr.get("read_fired"):
                r = tr.get("read", {})
                sub = (r.get("subtext") or "").strip()
                out.append(f"  read-pass: mood={r.get('mood')} intent={r.get('intent')}"
                           f" cert={r.get('certainty')}" + (f" subtext=“{sub}”" if sub else ""))
            out.append(f"  kept: {', '.join(tr.get('blocks_included') or []) or '—'}")
            out.append(f"  worthwhile={tr.get('retrieval_worthwhile')}"
                       f"  sys_len={tr.get('system_len')}")
    return "\n".join(out)


def get_turn(turn_id) -> dict | None:
    """Fetch the trace for a specific turn_id (ring first, then DB). None if
    absent. Returns the most recent record if a turn_id recurs."""
    try:
        tid = int(turn_id)
    except Exception:
        return None
    for tr in reversed(_RING):
        if tr.get("turn_id") == tid:
            return tr
    try:
        con = _connect()
        try:
            row = con.execute(
                "SELECT data FROM turn_traces WHERE turn_id=? ORDER BY id DESC LIMIT 1",
                (tid,)).fetchone()
        finally:
            con.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def format_turn(turn_id, full: bool = True) -> str:
    """Full dump for one turn — gloss + per-block token math + the assembled
    system prompt (if stored). Backs `/debug turn N`."""
    tr = get_turn(turn_id)
    if not tr:
        return f"no trace for turn {turn_id} (ring holds the last {RING_MAX})."
    lines = [_gloss(tr)]
    lines.append(f"  model: {tr.get('model')}  route: {tr.get('route_reason')}"
                 f"  mood_src: {tr.get('mood_source')}")
    bt = tr.get("block_tokens") or {}
    if bt:
        lines.append("  blocks(tok): " + ", ".join(f"{k}={v}" for k, v in bt.items()))
    if tr.get("blocks_dropped"):
        lines.append("  dropped: " + ", ".join(tr["blocks_dropped"]))
    if tr.get("read_fired"):
        r = tr.get("read", {})
        sub = (r.get("subtext") or "").strip()
        lines.append(f"  read-pass: mood={r.get('mood')} cert={r.get('certainty')}"
                     + (f" subtext=“{sub}”" if sub else ""))
    if full and tr.get("system_full"):
        lines.append("  ── assembled system prompt ──")
        lines.append(tr["system_full"])
    elif full:
        lines.append("  (system prompt not stored — set CHLOE_TRACE_PROMPT=1)")
    return "\n".join(lines)
