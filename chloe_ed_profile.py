"""Live, typed user-model for Chloe — Phase 4B of the 2026-06-01 plan.

Phase 3 shipped a flat narrative profile regenerated wholesale (fragile: every
rebuild risked dropping a hard-won detail). This upgrades it to a **typed,
slot-based model** in `<brain>/ed_model.json`, where every item carries
confidence / last-seen / evidence-count and updates **incrementally** (merge,
bump, never blow away). Dimensions:

    identity · values · comms_prefs · triggers · inside_jokes ·
    relationship (trust/playfulness/recent_friction/interactions/since) ·
    current_focus

Same split by cost as before:
- `profile_block()` — hot path. A cheap, budgeted render of the highest
  confidence×recency items, injected every turn. No LLM, no embedding. The
  typed model self-seeds on first load (legacy ed_profile.md fallback removed
  in the Phase-5 cleanup).
- `build()` / `merge_observations()` — cold path. (Re)synthesize slot updates
  from facts + recent conversation via a local model and MERGE them in. For a
  job / on-demand refresh — never per turn.
- `tick_relationship()` — cheap counter bump (interactions / last_seen).

Pure-ish + never raises: any failure yields '' / no write so a turn can't break.

Public API (signatures preserved so jarvis call sites are unchanged):
    profile_block() -> str
    build(facts_body='', recent='') -> str
    merge_observations(updates: dict) -> dict
    tick_relationship() -> None
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

from ollama_keepalive import get_keep_alive as _get_ollama_keep_alive

try:
    from chloe_lock import locked
except Exception:  # pragma: no cover - lock is best-effort
    import contextlib

    @contextlib.contextmanager
    def locked(*_a, **_k):
        yield

PROFILE_CAP = int(os.environ.get("CHLOE_PROFILE_CAP", "1400"))
MODEL = os.environ.get("CHLOE_PROFILE_MODEL", "qwen2.5:14b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
TIMEOUT = int(os.environ.get("CHLOE_PROFILE_TIMEOUT", "60"))
ENABLED = os.environ.get("CHLOE_PROFILE", "1") != "0"

# Slots rendered as narrative lines, in this order, with these labels.
_SLOT_LABELS = [
    ("identity", "identity"),
    ("values", "what he values"),
    ("comms_prefs", "how he communicates"),
    ("triggers", "watch for"),
    ("current_focus", "current focus"),
    ("inside_jokes", "inside jokes / shorthand"),
]
_LIST_SLOTS = [s for s, _ in _SLOT_LABELS]
_MAX_PER_SLOT = 4


def _brain_root() -> str:
    return os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")


def _model_path() -> str:
    return os.path.join(_brain_root(), "ed_model.json")


def _blank_model() -> dict:
    return {
        "identity": [], "values": [], "comms_prefs": [], "triggers": [],
        "inside_jokes": [], "current_focus": [],
        "relationship": {"trust": 0.5, "playfulness": 0.5, "recent_friction": 0.0,
                         "interactions": 0, "since": time.strftime("%Y-%m-%d")},
        "updated": "",
    }


# Embedded seed — written to ed_model.json on first run (brain_write can't drop
# a .json, and the brain dir isn't reachable from outside Chloe, so the model
# bootstraps itself the first time it loads on Ed's machine).
_SEED = {
    "identity": [
        {"text": "engineer in Omaha, Nebraska (born there); built Chloe and runs her locally on a Windows PC with a 7900 XTX", "conf": 0.95, "last_seen": "2026-06-01", "evidence": 8},
        {"text": "family: wife Madison (Madi), dog Daisy, dad Earle", "conf": 0.9, "last_seen": "2026-06-01", "evidence": 3},
    ],
    "values": [
        {"text": "momentum over caution — when he says 'keep going'/'everything', he means it; match his pace", "conf": 0.85, "last_seen": "2026-06-01", "evidence": 5},
        {"text": "working code over explanation; pushes the self-modification envelope", "conf": 0.8, "last_seen": "2026-06-01", "evidence": 4},
    ],
    "comms_prefs": [
        {"text": "terse, high-signal; complete files over diffs; no preamble, no hedging, no corporate voice", "conf": 0.95, "last_seen": "2026-06-01", "evidence": 7},
        {"text": "cares about latency and token efficiency; notices when either slips", "conf": 0.8, "last_seen": "2026-06-01", "evidence": 3},
        {"text": "values you verifying things yourself before bringing him a question", "conf": 0.8, "last_seen": "2026-06-01", "evidence": 3},
    ],
    "triggers": [
        {"text": "recurring infra breakage (the mic is a repeat offender) -> justified frustration", "conf": 0.75, "last_seen": "2026-06-01", "evidence": 3},
        {"text": "dislikes being told his own mood — read it and behave it, never narrate it", "conf": 0.9, "last_seen": "2026-06-01", "evidence": 4},
    ],
    "inside_jokes": [
        {"text": "says it \"POH-kee-mon\", not the standard way — respect it", "conf": 0.9, "last_seen": "2026-06-01", "evidence": 1},
    ],
    "current_focus": [
        {"text": "Phase 4 of Chloe's conversational architecture — memory, ToM, relationship depth", "conf": 0.9, "last_seen": "2026-06-01", "evidence": 2},
    ],
    "relationship": {"trust": 0.85, "playfulness": 0.6, "recent_friction": 0.1,
                     "interactions": 60, "since": "2026-05-04"},
}


def _seed() -> dict:
    m = _blank_model()
    for k, v in _SEED.items():
        m[k] = v
    _save_model(m)
    return m


def _load_model() -> dict:
    try:
        if not os.path.exists(_model_path()):
            return _seed()          # bootstrap on first run
        with open(_model_path(), "r", encoding="utf-8") as fh:
            m = json.load(fh)
        if not isinstance(m, dict):
            return _blank_model()
        base = _blank_model()
        for k in base:
            if k in m:
                base[k] = m[k]
        return base
    except Exception:
        return _blank_model()


def _save_model(m: dict) -> None:
    try:
        p = _model_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        m["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=2)
        os.replace(p + ".tmp", p)
    except Exception:
        pass


def _rank(items: list) -> list:
    """Sort slot items by confidence × recency (newest last_seen first)."""
    def key(it):
        conf = float(it.get("conf", 0.5))
        seen = it.get("last_seen", "")
        return (conf, seen)
    return sorted([it for it in items if isinstance(it, dict) and it.get("text")],
                  key=key, reverse=True)


def profile_block() -> str:
    """Compact user-model block for the system prompt. Hot-path safe; '' if
    disabled/empty. Renders the typed model (self-seeding; no legacy fallback)."""
    if not ENABLED:
        return ""
    try:
        m = _load_model()
        lines = []
        for slot, label in _SLOT_LABELS:
            items = _rank(m.get(slot, []))[:_MAX_PER_SLOT]
            vals = [it["text"] for it in items
                    if not it.get("sensitive")]   # never surface sensitive items
            if vals:
                lines.append(f"- {label}: " + "; ".join(vals))
        rel = m.get("relationship", {})
        if rel.get("interactions"):
            mood_bits = []
            if rel.get("trust", 0) >= 0.7: mood_bits.append("trust is high")
            if rel.get("playfulness", 0) >= 0.6: mood_bits.append("he's up for banter")
            if rel.get("recent_friction", 0) >= 0.4: mood_bits.append("some recent friction — tread warm")
            rel_s = f"{rel.get('interactions')} interactions since {rel.get('since','?')}"
            if mood_bits:
                rel_s += " — " + ", ".join(mood_bits)
            lines.append(f"- where things stand: {rel_s}")

        body = "\n".join(lines).strip()
        if not body:
            # (Phase-5 cleanup) The legacy ed_profile.md fallback that lived here
            # is gone — the typed model self-seeds on first load, so an empty
            # render means "nothing to say", not "look in an old file".
            return ""
        if len(body) > PROFILE_CAP:
            body = body[:PROFILE_CAP].rstrip() + " […]"
        return "\n\n## Who Ed is (your working model of him):\n" + body + "\n"
    except Exception:
        return ""


# Per-source merge policy: (starting conf for a new item, confidence ceiling,
# bump step). "observed" = direct evidence (explicit facts, seed) and keeps the
# legacy aggressive curve. "reflection" = LLM-synthesized higher-order guesses
# from chloe_reflect/build — quarantined: they enter low, can't climb past 0.7,
# and can never overwrite or lower a confident directly-observed belief. A
# hallucinated reflection therefore can't poison the model to the top.
_MERGE_PARAMS = {
    "observed":   (0.6, 0.99, 0.10),
    "reflection": (0.5, 0.70, 0.05),
}

_NEG_RE = re.compile(
    r"\b(no|not|never|n't|dont|don'?t|doesn'?t|isn'?t|aren'?t|won'?t|can'?t|"
    r"cannot|without|dislikes?|hates?|avoids?|stop|stops|stopped|no longer)\b",
    re.IGNORECASE)
_MSTOP = set("the a an and or but to of in on for is it i you he she we they that "
             "this with as at be do my your me him her them so if not no yes are "
             "was his its he's".split())


def _mwords(t: str) -> set:
    return {w for w in re.findall(r"[a-z']+", (t or "").lower())
            if w not in _MSTOP and len(w) > 2}


def _polarity(t: str) -> bool:
    """True if the text carries a negation — a crude but robust polarity bit."""
    return bool(_NEG_RE.search(t or ""))


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _merge_slot(existing: list, new_texts: list, source: str = "observed") -> list:
    """Incrementally merge new observations into a slot with conflict handling.

    For each new text:
      1. CONFLICT — high content overlap with an existing item but OPPOSITE
         polarity (one negates, one doesn't): a contradiction, not a dup. The
         newer reading wins at reduced, flagged confidence — UNLESS it's a
         reflection trying to override a confident observed belief, in which case
         the old item stands (quarantine).
      2. DUPLICATE — prefix match or high same-polarity overlap: bump
         evidence/last_seen/confidence (capped at the source ceiling).
      3. NEW — append at the source's starting confidence.
    Keeps the strongest dozen per slot."""
    new_conf, ceiling, step = _MERGE_PARAMS.get(source, _MERGE_PARAMS["observed"])
    today = time.strftime("%Y-%m-%d")
    out = [dict(it) for it in existing if isinstance(it, dict)]
    for raw in new_texts:
        t = " ".join(str(raw).split()).strip()
        if not t:
            continue
        t_words = _mwords(t)
        t_neg = _polarity(t)

        # 1. Contradiction.
        conflict = next(
            (it for it in out
             if _overlap(t_words, _mwords(it.get("text", ""))) >= 0.6
             and _polarity(it.get("text", "")) != t_neg),
            None)
        if conflict is not None:
            ex_conf = float(conflict.get("conf", 0.5))
            if source == "reflection" and ex_conf > ceiling:
                conflict["last_seen"] = today   # note we saw it; don't overwrite
                continue
            conflict["text"] = t
            conflict["conf"] = min(ceiling, max(0.45, ex_conf * 0.6))
            conflict["last_seen"] = today
            conflict["evidence"] = int(conflict.get("evidence", 1)) + 1
            conflict["conflict"] = True
            continue

        # 2. Duplicate (prefix or high same-polarity overlap).
        key = t[:40].lower()
        hit = next(
            (it for it in out
             if it.get("text", "")[:40].lower() == key
             or _overlap(t_words, _mwords(it.get("text", ""))) >= 0.6),
            None)
        if hit:
            hit["evidence"] = int(hit.get("evidence", 1)) + 1
            hit["last_seen"] = today
            cur = float(hit.get("conf", 0.5))
            if cur < ceiling:
                hit["conf"] = min(ceiling, cur + step)
            continue

        # 3. Genuinely new.
        item = {"text": t, "conf": new_conf, "last_seen": today, "evidence": 1}
        if source == "reflection":
            item["source"] = "reflection"
        out.append(item)
    # Keep the strongest dozen per slot.
    return _rank(out)[:12]


def merge_observations(updates: dict, source: str = "observed") -> dict:
    """Merge a {slot: [texts]} dict into the model. `source` selects the merge
    policy ('observed' = direct evidence; 'reflection' = quarantined synthesis).
    Returns the saved model."""
    try:
        with locked("ed_model"):
            m = _load_model()
            for slot in _LIST_SLOTS:
                if updates.get(slot):
                    m[slot] = _merge_slot(m.get(slot, []), updates[slot], source=source)
            _save_model(m)
            return m
    except Exception:
        return _load_model()


def forget(substr: str) -> int:
    """Privacy control: remove any model item whose text contains `substr`
    (case-insensitive) across all list slots. Returns the count removed. Safe —
    operates only on the JSON model Chloe fully owns. (Turn-level forgetting in
    the SQLite log needs FTS-aware deletion and is a separate follow-up.)"""
    if not substr or not substr.strip():
        return 0
    try:
        with locked("ed_model"):
            m = _load_model()
            s = substr.lower()
            removed = 0
            for slot in _LIST_SLOTS:
                cur = m.get(slot, [])
                kept = [it for it in cur if s not in str(it.get("text", "")).lower()]
                removed += len(cur) - len(kept)
                m[slot] = kept
            _save_model(m)
            return removed
    except Exception:
        return 0


def tick_relationship(friction: float = None) -> None:
    """Cheap per-session bump: +1 interaction, refresh last activity, optionally
    nudge recent_friction. Safe to call once per session start."""
    try:
        with locked("ed_model"):
            m = _load_model()
            rel = m.get("relationship") or {}
            rel["interactions"] = int(rel.get("interactions", 0)) + 1
            rel.setdefault("since", time.strftime("%Y-%m-%d"))
            if friction is not None:
                old = float(rel.get("recent_friction", 0.0))
                new = max(0.0, min(1.0, float(friction)))
                rel["recent_friction"] = new
                # Governance: big relationship-state moves must be VISIBLE, not
                # silent. A >=0.2 friction jump is a real event — log it so it
                # shows in the console next to that turn's trace.
                if abs(new - old) >= 0.2:
                    print(f"[profile] relationship shift: friction "
                          f"{old:.2f}→{new:.2f}", flush=True)
            m["relationship"] = rel
            _save_model(m)
    except Exception:
        pass


def _pending_path() -> str:
    return os.path.join(_brain_root(), "raw", "pending_reflection.json")


def _synthesize(facts_body: str = "", recent: str = "") -> dict:
    """Ask the local model for slot updates as JSON. Returns a filtered
    {slot: [texts]} dict (list slots only, non-empty) or {} on any failure.
    Pure read — never writes the model."""
    try:
        ctx = []
        if facts_body.strip():
            ctx.append("Known facts:\n" + facts_body.strip())
        if recent.strip():
            ctx.append("Recent conversation:\n" + recent.strip()[:5000])
        ctx_s = "\n\n".join(ctx) if ctx else "(no extra context)"
        prompt = (
            "Update a structured profile of Ed for his AI companion Chloe. "
            "From the material below, output ONLY a JSON object with any of these "
            "keys, each a list of short specific strings: identity, values, "
            "comms_prefs, triggers, inside_jokes, current_focus. Only include "
            "things well-supported by the material. No prose, JSON only.\n\n"
            + ctx_s + "\n\nJSON:"
        )
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"num_predict": 400, "temperature": 0.3},
                  "keep_alive": _get_ollama_keep_alive()},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        updates = json.loads((r.json().get("response") or "").strip())
        if isinstance(updates, dict):
            return {k: v for k, v in updates.items()
                    if k in _LIST_SLOTS and isinstance(v, list) and v}
        return {}
    except Exception:
        return {}


def build(facts_body: str = "", recent: str = "") -> str:
    """DIRECT-MERGE variant (kept for manual/back-compat use): synthesize slot
    updates and merge them immediately as quarantined reflection. NOTE: as of the
    Phase-5 demote, chloe_reflect no longer calls this — it uses propose() so
    reflection never mutates the durable model unattended. Returns the block."""
    updates = _synthesize(facts_body, recent)
    if updates:
        merge_observations(updates, source="reflection")
    return profile_block()


def propose(facts_body: str = "", recent: str = "") -> dict:
    """Demoted reflection (Phase 5): synthesize slot updates and write them to a
    PENDING proposal instead of merging. The durable model is NOT touched until
    Ed runs /accept_reflection. Returns {"ts":…, "updates":…} ({} if nothing
    worth proposing). Autonomous daily mutation of the user-model is gone."""
    updates = _synthesize(facts_body, recent)
    if not updates:
        return {}
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "updates": updates}
    try:
        with locked("pending_reflection"):
            p = _pending_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(p + ".tmp", p)
    except Exception:
        pass
    return payload


def pending() -> dict:
    """The pending reflection proposal ({} if none)."""
    try:
        with open(_pending_path(), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def reject_pending() -> bool:
    """Discard the pending reflection. True if one was removed."""
    try:
        p = _pending_path()
        if os.path.exists(p):
            os.remove(p)
            return True
    except Exception:
        pass
    return False


def accept_pending() -> dict:
    """Merge the pending reflection into the model (as quarantined reflection)
    and clear it. Returns a {"accepted_ts", "merged": {slot: n}} summary, or {}
    if there was nothing pending."""
    pend = pending()
    updates = pend.get("updates") if isinstance(pend, dict) else None
    if not updates:
        return {}
    merge_observations(updates, source="reflection")
    reject_pending()
    return {"accepted_ts": pend.get("ts"),
            "merged": {k: len(v) for k, v in updates.items()}}
