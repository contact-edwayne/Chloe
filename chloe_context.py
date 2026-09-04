"""Context Composer for Chloe — Phase 4A keystone of the 2026-06-01 plan.

Through Phases 0-3, `handle_chat` and `_augmented_voice_system` grew into an
unbudgeted concatenation of ~13 independent blocks (preamble, now, about, facts,
recent-context, ed_profile, synopsis, dialogue-state, recall, wiki, nsfw, …).
Each block decided on its own whether to inject, with no shared token budget, no
deduplication, and no priority. At 16k context on a local model that silently
overflows and truncates — usually the persona, the thing it was built to protect.

This module is the single front door for system-prompt assembly. Callers hand it
a list of candidate blocks; it:

  1. **Prioritizes** — priority 0 ("identity": the persona core, the date) is
     NEVER dropped. Higher numbers are dropped first when over budget.
  2. **Budgets** — estimates tokens and drops lowest-priority blocks until the
     prompt fits `ctx_tokens - reserve` (reserve leaves room for the reply).
  3. **De-duplicates** — blocks sharing a `key` are included once (so the same
     thread doesn't arrive from recall AND synopsis AND profile).
  4. **Emits in stable reading order** — inclusion is decided by priority, but
     output is ordered by `order` so dropping a low block never scrambles the
     rest.
  5. **Logs** — prints tokens used / blocks kept / blocks dropped every turn so
     context pressure is visible instead of discovered later as "forgetting".

Pure + never raises: on any failure it falls back to a plain concatenation of
every block's text, so a turn can never break here. Token counting (Phase 5):
REAL tokenizer via Ollama /api/tokenize, hash-cached so stable blocks
(persona/facts/profile) cost one call ever and only the volatile blocks
(recall/dstate/recent) re-tokenize; len/4 stays as the never-fail fallback when
the endpoint is disabled, missing, or down (10-min cooldown, no per-block
re-probing).

Public API:
    block(name, text, priority, *, key=None, order=None) -> dict
    compose(blocks, ctx_tokens=..., reserve=..., log=True) -> (text, used, dropped)
    est_tokens(text) -> int        # real-tokenizer-backed, cached, never raises
    tok_mode() -> 'real' | 'est'
"""

from __future__ import annotations

import hashlib
import os
import threading
import time

DEFAULT_CTX = int(os.environ.get("CHLOE_OLLAMA_CTX", "16384"))
DEFAULT_RESERVE = int(os.environ.get("CHLOE_CTX_RESERVE", "1200"))
LOG = os.environ.get("CHLOE_CTX_LOG", "1") != "0"

# ── Real tokenizer (hash-cached, graceful-degrade) ──────────────────────────
TOK_REAL = os.environ.get("CHLOE_REAL_TOKENS", "1") != "0"
TOK_MODEL = (os.environ.get("CHLOE_TOK_MODEL")
             or os.environ.get("OLLAMA_MODEL", "llama3.2:3b"))
TOK_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
TOK_TIMEOUT = float(os.environ.get("CHLOE_TOK_TIMEOUT", "1.5"))
TOK_DEAD_COOLDOWN_S = 600.0   # after a failure, don't re-probe for 10 min
_TOK_CACHE: dict = {}
_TOK_LOCK = threading.Lock()
_TOK_MAX = 256
_TOK_DEAD_UNTIL = 0.0


def _fallback_tokens(text: str) -> int:
    """Legacy ~4 chars/token heuristic — the never-fail floor."""
    return (len(text) + 3) // 4 if text else 0


def _real_token_count(text: str):
    """Ask Ollama to tokenize `text`. Int count or None on any failure."""
    import requests  # lazy — module stays importable without it
    r = requests.post(
        f"{TOK_URL}/api/tokenize",
        # some builds take "text", some "prompt" — send both, extras ignored
        json={"model": TOK_MODEL, "text": text, "prompt": text},
        timeout=TOK_TIMEOUT,
    )
    r.raise_for_status()
    toks = r.json().get("tokens")
    return len(toks) if isinstance(toks, list) else None


def est_tokens(text: str) -> int:
    """Token count for budgeting. Tries the REAL tokenizer, hash-cached per
    block text; falls back to len/4 when disabled or the endpoint is
    missing/down (cooldown prevents per-block re-probing). Never raises."""
    global _TOK_DEAD_UNTIL
    if not text:
        return 0
    if not TOK_REAL or time.time() < _TOK_DEAD_UNTIL:
        return _fallback_tokens(text)
    try:
        key = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]
        with _TOK_LOCK:
            n = _TOK_CACHE.get(key)
        if n is not None:
            return n
        n = _real_token_count(text)
        if n is None:
            _TOK_DEAD_UNTIL = time.time() + TOK_DEAD_COOLDOWN_S
            return _fallback_tokens(text)
        with _TOK_LOCK:
            _TOK_CACHE[key] = n
            while len(_TOK_CACHE) > _TOK_MAX:
                _TOK_CACHE.pop(next(iter(_TOK_CACHE)))
        return n
    except Exception:
        _TOK_DEAD_UNTIL = time.time() + TOK_DEAD_COOLDOWN_S
        return _fallback_tokens(text)


def tok_mode() -> str:
    """'real' when the Ollama tokenizer is active, 'est' when on the len/4
    fallback — shown in the [context] log so traces record which produced
    the numbers."""
    return "real" if (TOK_REAL and time.time() >= _TOK_DEAD_UNTIL) else "est"


def block(name: str, text: str, priority: int, *, key=None, order=None) -> dict:
    """Build a candidate block. `order` defaults to the priority so the natural
    reading order matches importance unless a caller overrides it."""
    return {
        "name": name,
        "text": text or "",
        "priority": int(priority),
        "key": key,
        "order": priority if order is None else order,
    }


def compose(blocks, ctx_tokens: int = None, reserve: int = None, log: bool = None):
    """Assemble a system prompt from candidate blocks under a token budget.

    Returns (assembled_text, tokens_used, dropped) where `dropped` is a list of
    (name, reason) for visibility. Never raises — on any error returns a plain
    concatenation of all block texts.
    """
    ctx_tokens = DEFAULT_CTX if ctx_tokens is None else ctx_tokens
    reserve = DEFAULT_RESERVE if reserve is None else reserve
    log = LOG if log is None else log
    try:
        budget = max(ctx_tokens - reserve, 1000)
        cands = [b for b in (blocks or []) if isinstance(b, dict) and b.get("text")]
        # Decide inclusion by priority (0 first); tie-break on reading order.
        ranked = sorted(cands, key=lambda b: (b.get("priority", 9), b.get("order", 0)))
        kept, used, seen, dropped = [], 0, set(), []
        for b in ranked:
            key = b.get("key")
            if key and key in seen:
                dropped.append((b.get("name", "?"), "dup"))
                continue
            cost = est_tokens(b["text"])
            if int(b.get("priority", 9)) > 0 and used + cost > budget:
                dropped.append((b.get("name", "?"), "budget"))
                continue
            kept.append(b)
            used += cost
            if key:
                seen.add(key)
        # Emit in stable reading order.
        kept.sort(key=lambda b: b.get("order", 0))
        out = "".join(b["text"] for b in kept)
        if log:
            drop_s = ", ".join(f"{n}({r})" for n, r in dropped) or "none"
            print(f"[context] {used}/{budget} tok ({tok_mode()}), "
                  f"{len(kept)} blocks kept, dropped: {drop_s}", flush=True)
        return out, used, dropped
    except Exception as e:
        try:
            print(f"[context] compose failed ({e}); falling back to concat", flush=True)
            return "".join((b.get("text", "") for b in (blocks or [])
                            if isinstance(b, dict))), 0, []
        except Exception:
            return "", 0, []
