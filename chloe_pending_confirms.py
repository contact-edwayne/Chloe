"""Stage 3 of self-modification: voice/chat-confirm per apply.

Chloe (or any Cowork-driven proposer) drafts a proposal, then *announces*
it conversationally: "I drafted a fix for X — want me to apply?" Ed
replies with "yes" / "yeah" / "go ahead" / "no" / "cancel" in the same
channel, and that response decides whether the proposal applies.

This module is the state machine. Both the Chloe backend (which catches
Ed's voice/chat replies in `jarvis.py` + `brain_wiring.py`) and the MCP
server (which writes pending confirms when a Cowork job calls
`propose_and_announce`) need to see the same state. Since those are
separate processes, the state lives in a JSON file under
`C:\\Chloe\\brain\\pending_confirms.json` — atomic write via tmp+rename
to avoid torn reads.

Public API:
  - `announce(slug, source, ttl_s=120, summary="")` -> dict
  - `resolve(user_text, source) -> Optional[dict]`
  - `pending(source=None) -> list[dict]`
  - `cancel(slug="") -> dict`

Source separation: voice-announced confirms can only be resolved by
voice replies. Chat-announced by chat replies. Source `"any"` matches
both (use sparingly — increases ambiguity if multiple channels are
active).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Optional


# ─── State file location ──────────────────────────────────────────────────

def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _state_path() -> Path:
    p = _brain_root()
    p.mkdir(parents=True, exist_ok=True)
    return p / "pending_confirms.json"


def _now() -> float:
    return time.time()


# ─── Phrase matching ──────────────────────────────────────────────────────

# Affirmative replies that resolve a pending confirm to APPLY.
# Kept deliberately small; expand based on real usage / log misses.
_AFFIRMATIVE = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "go", "apply", "approve",
    "do it", "go ahead", "send it", "ship it", "confirm", "confirmed",
    "ok", "okay", "alright",
})

# Negative replies that cancel a pending confirm.
_NEGATIVE = frozenset({
    "no", "nope", "nevermind", "never mind", "cancel", "hold off",
    "not yet", "wait", "skip", "abort", "negative",
})

# Strip leading/trailing punctuation + lowercase for matching.
_PUNCT_RE = re.compile(r"^[\s.,!?\"';:()\[\]]+|[\s.,!?\"';:()\[\]]+$")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "").strip().lower())


def classify_reply(text: str) -> str:
    """Return "yes" / "no" / "" for the user's reply."""
    norm = _normalize(text)
    if not norm:
        return ""
    # Only consider short replies (<= 5 tokens) to avoid false positives
    # from incidental "yes" inside a longer sentence ("I have yes-people").
    tokens = norm.split()
    if len(tokens) > 5:
        return ""
    if norm in _AFFIRMATIVE:
        return "yes"
    if norm in _NEGATIVE:
        return "no"
    # Multi-word affirmative/negative (e.g. "go ahead", "hold off"):
    if norm in _AFFIRMATIVE:
        return "yes"
    if norm in _NEGATIVE:
        return "no"
    # First-token check for things like "yes please" / "no thanks".
    first = tokens[0]
    if first in _AFFIRMATIVE and len(tokens) <= 3:
        return "yes"
    if first in _NEGATIVE and len(tokens) <= 3:
        return "no"
    return ""


# ─── State file IO (atomic) ───────────────────────────────────────────────

def _load_state() -> dict:
    """Read the state file. Returns {slug: entry} or {} on missing/corrupt."""
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    """Atomic write: tmp file + rename. Survives partial-write crashes."""
    p = _state_path()
    tmp = p.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # On Windows, os.replace handles the atomic move even if target exists.
    os.replace(tmp, p)


def _prune_expired(state: dict) -> dict:
    """Drop entries past their deadline. Returns the cleaned state."""
    now = _now()
    return {
        slug: entry for slug, entry in state.items()
        if entry.get("expires_at", 0) > now
    }


# ─── Public API ───────────────────────────────────────────────────────────

def announce(slug: str, source: str = "chat", ttl_s: int = 120,
             summary: str = "") -> dict:
    """Register a proposal as pending Ed's confirmation.

    Args:
        slug: the proposal slug (matches `proposals/code_*_<slug>.md`).
        source: "voice" | "chat" | "any". Only matching-source replies
            can resolve. "any" matches both.
        ttl_s: seconds until the pending entry auto-expires. Default 120.
        summary: optional one-line description for status output.

    Returns:
        Dict shape: {ok, slug, source, expires_at, announce_text}.
        The `announce_text` is the speech-shaped string the caller
        should relay to Ed.
    """
    slug = (slug or "").strip()
    if not slug:
        return {"ok": False, "error": "missing slug"}
    source = (source or "chat").strip().lower()
    if source not in ("voice", "chat", "any"):
        return {"ok": False,
                "error": f"source must be voice|chat|any, got {source!r}"}
    if ttl_s < 5 or ttl_s > 3600:
        return {"ok": False, "error": "ttl_s must be 5-3600"}

    now = _now()
    entry = {
        "slug": slug,
        "source": source,
        "ttl_s": int(ttl_s),
        "asked_at": now,
        "expires_at": now + int(ttl_s),
        "summary": summary[:200],
    }

    state = _prune_expired(_load_state())
    # Re-announcing an existing pending = refresh TTL, keep source.
    state[slug] = entry
    _save_state(state)

    nice_summary = summary or f"the change in `{slug}`"
    announce_text = (
        f"i drafted {nice_summary}. want me to apply it? "
        f"say yes / no within {int(ttl_s)//60} minutes."
        if int(ttl_s) >= 60 else
        f"i drafted {nice_summary}. apply? yes or no."
    )
    return {"ok": True, "slug": slug, "source": source,
            "expires_at": entry["expires_at"],
            "announce_text": announce_text}


def resolve(user_text: str, source: str = "chat") -> Optional[dict]:
    """Check whether `user_text` resolves a pending confirm. Apply or
    cancel if so.

    Called by the Chloe backend on every non-slash user turn.

    Args:
        user_text: the user's message.
        source: "voice" | "chat". Channel of the incoming reply.

    Returns:
        None if no pending confirm matched (caller continues normally).
        A dict with shape:
          {action: "applied"|"canceled", slug, result, reply_text}
        if a confirm WAS resolved. `result` is the apply_proposal return
        dict (for "applied") or `{"ok": True}` (for "canceled").
        `reply_text` is what Chloe should say back.
    """
    decision = classify_reply(user_text)
    if not decision:
        return None
    source = (source or "chat").strip().lower()

    state = _prune_expired(_load_state())
    # Save back the pruned state so expired entries don't accumulate.
    _save_state(state)

    # Find the most-recent pending matching this source.
    candidates = []
    for slug, entry in state.items():
        if entry.get("source") not in (source, "any"):
            continue
        candidates.append(entry)
    if not candidates:
        return None
    # Newest first by asked_at — Ed's "yes" applies to the most recent
    # announcement.
    candidates.sort(key=lambda e: e["asked_at"], reverse=True)
    target = candidates[0]
    slug = target["slug"]

    # Remove the entry — single-resolution.
    state.pop(slug, None)
    _save_state(state)

    if decision == "no":
        return {
            "action": "canceled",
            "slug": slug,
            "result": {"ok": True, "slug": slug},
            "reply_text": f"got it. canceled `{slug}` — nothing applied.",
        }

    # decision == "yes" — apply via the Tier-1 pipeline. All safety rails
    # fire. No token needed; Stage 3 is its own gate (Ed said yes).
    try:
        import chloe_proposals
        result = chloe_proposals.apply_proposal(slug, dry_run=False)
    except Exception as e:
        return {
            "action": "applied",
            "slug": slug,
            "result": {"ok": False, "error": f"{type(e).__name__}: {e}"},
            "reply_text": (f"hmm, tried to apply `{slug}` but hit "
                           f"{type(e).__name__}: {e}"),
        }

    if result.get("ok"):
        msg = result.get("message", "")
        # Drop the verbose "Backup at... Rollback..." tail; voice-friendly.
        first_sentence = msg.split(". ")[0] if msg else f"applied `{slug}`"
        reply = f"done. {first_sentence}."
    else:
        reply = (f"that didn't take — {result.get('error', 'unknown error')}")

    return {"action": "applied", "slug": slug, "result": result,
            "reply_text": reply}


def pending(source: Optional[str] = None) -> list[dict]:
    """List active pending-confirms. Optionally filter by source."""
    state = _prune_expired(_load_state())
    _save_state(state)
    now = _now()
    rows: list[dict] = []
    for entry in state.values():
        if source and entry.get("source") not in (source, "any"):
            continue
        ttl_remaining = max(0, entry.get("expires_at", 0) - now)
        rows.append({
            "slug":              entry["slug"],
            "source":            entry["source"],
            "ttl_remaining_s":   round(ttl_remaining, 0),
            "asked_at_iso":      _dt.datetime.fromtimestamp(
                                     entry["asked_at"]).isoformat(
                                     timespec="seconds"),
            "summary":           entry.get("summary", ""),
        })
    rows.sort(key=lambda r: r["ttl_remaining_s"], reverse=True)
    return rows


def cancel(slug: str = "") -> dict:
    """Cancel a specific pending or all pending if slug=''."""
    state = _prune_expired(_load_state())
    if slug:
        if slug in state:
            del state[slug]
            _save_state(state)
            return {"ok": True, "canceled": [slug]}
        _save_state(state)
        return {"ok": False, "error": f"no pending confirm for {slug!r}"}
    n = list(state.keys())
    state.clear()
    _save_state(state)
    return {"ok": True, "canceled": n}


# ─── CLI ──────────────────────────────────────────────────────────────────

def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Chloe pending-confirm state")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("announce")
    sp.add_argument("slug")
    sp.add_argument("--source", default="chat",
                    choices=["voice", "chat", "any"])
    sp.add_argument("--ttl", type=int, default=120)
    sp.add_argument("--summary", default="")

    sp = sub.add_parser("resolve")
    sp.add_argument("text")
    sp.add_argument("--source", default="chat", choices=["voice", "chat"])

    sub.add_parser("list")

    sp = sub.add_parser("cancel")
    sp.add_argument("slug", nargs="?", default="")

    args = ap.parse_args(argv)
    if args.cmd == "announce":
        r = announce(args.slug, source=args.source,
                     ttl_s=args.ttl, summary=args.summary)
        print(json.dumps(r, indent=2))
        return 0 if r.get("ok") else 1
    if args.cmd == "resolve":
        r = resolve(args.text, source=args.source)
        if r is None:
            print("(no pending match)")
            return 0
        print(json.dumps(r, indent=2, default=str))
        return 0
    if args.cmd == "list":
        for p in pending():
            print(f"{p['slug']:30s}  {p['source']:6s}  "
                  f"{int(p['ttl_remaining_s'])}s left  "
                  f"{p['summary'][:60]}")
        return 0
    if args.cmd == "cancel":
        r = cancel(args.slug)
        print(json.dumps(r, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
