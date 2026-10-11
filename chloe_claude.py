"""
chloe_claude.py -- optional Claude escalation for hard turns.

Local qwen stays the default for every turn (zero added latency). Only turns
that match the heuristics below go to Claude, streamed token-by-token so the
existing sentence-level TTS starts on the first sentence.

Enabled iff ANTHROPIC_API_KEY is set and CHLOE_CLAUDE_ESCALATE != 0.
Any failure before the first token yields nothing, so the caller falls back to
local Ollama as if Claude was never tried.

Env:
  ANTHROPIC_API_KEY        required
  CHLOE_CLAUDE_ESCALATE    0 to disable (default on when key present)
  CHLOE_CLAUDE_SONNET      model id, default claude-sonnet-5-5
  CHLOE_CLAUDE_OPUS        model id, default claude-opus-5-5
  CHLOE_CLAUDE_FIRST_TOKEN_TIMEOUT  seconds, default 6
"""

import asyncio
import os
import re
from typing import AsyncIterator, Optional

_env_loaded = False


def _load_env_once():
    """Import order must not matter: if jarvis.py imports this before it loads
    .env, read the key from .env next to this file ourselves."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")  # never overrides existing env
    except Exception:
        pass


def api_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k:
        _load_env_once()
        k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return k


def enabled() -> bool:
    return bool(api_key()) and os.environ.get("CHLOE_CLAUDE_ESCALATE", "1").strip().lower() not in ("0", "false", "no", "off")


def __getattr__(name):  # PEP 562: _cc.ENABLED stays valid but is evaluated live
    if name == "ENABLED":
        _e = enabled()
        _dbg(f"ENABLED checked -> {_e}")
        return _e
    if name == "API_KEY":
        return api_key()
    raise AttributeError(name)


HISTORY_MSGS = int(os.environ.get("CHLOE_CLAUDE_HISTORY", "6"))
_FOCUS = ("\n\nAnswer the user's LATEST message directly, on its own topic. Earlier turns are background "
          "only: do not resume, redo, or finish an earlier topic unless the latest message asks you to.")
FIRST_TOKEN_TIMEOUT = float(os.environ.get("CHLOE_CLAUDE_FIRST_TOKEN_TIMEOUT", "6"))

MODELS = {
    "sonnet": os.environ.get("CHLOE_CLAUDE_SONNET", "claude-sonnet-5-5"),
    "opus": os.environ.get("CHLOE_CLAUDE_OPUS", "claude-opus-5-5"),
}
MAX_TOKENS = {"sonnet": 700, "opus": 1200}

_OPUS_RE = re.compile(
    r"\b(think (really )?hard|deep dive|ultrathink|use opus|figure this out carefully)\b", re.I)
_SONNET_RE = re.compile(
    r"\b(use claude|use sonnet|plan|architect(ure)?|design|refactor|debug|"
    r"write (a |the )?(script|code|function|class)|analy[sz]e|compare|"
    r"step[- ]by[- ]step|explain (in detail|how .* works)|"
    r"why (is|does|did|isn't|doesn't)|trade-?offs?|strategy)\b", re.I)
# Never leave the machine: money, credentials, destructive/personal actions.
_NEVER_RE = re.compile(
    r"\b(wallet|bitcoin|btc|sats?|lightning|invoice|pin|password|passcode|seed phrase|"
    r"private key|send (money|payment)|delete|trash)\b", re.I)

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.AsyncAnthropic(api_key=api_key(), timeout=30.0, max_retries=0)
    return _client


def _dbg(msg: str) -> None:
    if os.environ.get("CHLOE_CLAUDE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        print(f"[claude:debug] {msg}", flush=True)
        try:
            import time
            from pathlib import Path
            with open(Path(__file__).parent / "claude_escalation.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except Exception:
            pass


def pick_tier(user_text: str) -> Optional[str]:
    """'sonnet' | 'opus' | None (stay local). Pure regex, microseconds."""
    if not enabled():
        _dbg(f"disabled (key set={bool(api_key())}, ESCALATE={os.environ.get('CHLOE_CLAUDE_ESCALATE')!r})")
        return None
    if not user_text:
        _dbg("empty user text")
        return None
    m = _NEVER_RE.search(user_text)
    if m:
        _dbg(f"blocked by never-list word {m.group(0)!r} in {user_text[:80]!r}")
        return None
    if _OPUS_RE.search(user_text):
        _dbg("-> opus")
        return "opus"
    if _SONNET_RE.search(user_text) or len(user_text) > 600:
        _dbg("-> sonnet")
        return "sonnet"
    _dbg(f"no trigger in {user_text[:80]!r} (len={len(user_text)})")
    return None


def _to_anthropic(messages: list):
    """OpenAI-style [system, user, assistant, ...] -> (system_str, anthropic msgs).
    Merges consecutive same-role turns and guarantees the list starts with user."""
    system_parts, out = [], []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            continue  # skip image/list content; escalation is text-only
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            if out and out[-1]["role"] == role:
                out[-1]["content"] += "\n" + content
            else:
                out.append({"role": role, "content": content})
    keep = HISTORY_MSGS
    if keep > 0 and len(out) > keep:
        out = out[-keep:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    system = "\n\n".join(system_parts)
    if os.environ.get("CHLOE_CLAUDE_FOCUS", "1").strip().lower() not in ("0", "false", "no", "off"):
        system += _FOCUS
    return system, out


# ---- prompt caching -------------------------------------------------------
# jarvis builds one big system string per turn: a stable base plus per-turn
# blocks (time, recall...). Cache only the stable prefix: the longest common
# prefix with the previous turn's system, rounded down and kept sticky so the
# cache breakpoint lands on the same boundary every turn.
_prev_system = ""
_stable_len = 0
_CACHE_ROUND = 1000
_CACHE_MIN_CHARS = 4500   # ~1024 tokens, Sonnet's minimum cacheable prefix


def _system_blocks(system: str):
    global _prev_system, _stable_len
    if os.environ.get("CHLOE_CLAUDE_CACHE", "1").strip().lower() in ("0", "false", "no", "off"):
        return system
    lcp = 0
    n = min(len(system), len(_prev_system))
    while lcp < n and system[lcp] == _prev_system[lcp]:
        lcp += 1
    _prev_system = system
    cand = (lcp // _CACHE_ROUND) * _CACHE_ROUND
    if _stable_len == 0:
        _stable_len = cand
    elif cand < _stable_len:
        _stable_len = cand          # only shrink, so the boundary stays put
    if _stable_len < _CACHE_MIN_CHARS or _stable_len >= len(system):
        return system
    head, tail = system[:_stable_len], system[_stable_len:]
    _dbg(f"cache breakpoint at {_stable_len} of {len(system)} chars")
    return [
        {"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": tail},
    ]


async def stream(messages: list, tier: str, max_tokens: Optional[int] = None) -> AsyncIterator[str]:
    """Yield text deltas from Claude. Yields nothing (and never raises) if Claude
    is unreachable or slow to first token, so the caller can fall back to local."""
    import time
    try:
        system, msgs = _to_anthropic(messages)
        if not msgs:
            _dbg("stream: no user messages after conversion -> yielding nothing")
            return
        print(f"[claude] escalating -> {tier} ({MODELS[tier]})", flush=True)
        _dbg(f"stream start tier={tier} model={MODELS[tier]} msgs={len(msgs)} system_chars={len(system)}")
        t0 = time.time()
        cm = _get_client().messages.stream(
            model=MODELS[tier],
            max_tokens=min(max_tokens or MAX_TOKENS[tier], MAX_TOKENS[tier]),
            system=_system_blocks(system),
            messages=msgs,
        )
        async with cm as s:
            it = s.text_stream.__aiter__()
            try:
                first = await asyncio.wait_for(it.__anext__(), timeout=FIRST_TOKEN_TIMEOUT)
            except StopAsyncIteration:
                _dbg("stream ended with no tokens")
                return
            except asyncio.TimeoutError:
                _dbg(f"no first token within {FIRST_TOKEN_TIMEOUT}s -> falling back to local")
                return
            _dbg(f"first token after {time.time() - t0:.2f}s")
            yield first
            async for t in it:
                yield t
            try:
                u = (await s.get_final_message()).usage
                _dbg(f"usage in={u.input_tokens} cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
                     f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} out={u.output_tokens}")
            except Exception:
                pass
        _dbg(f"stream complete in {time.time() - t0:.2f}s")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # network, auth, rate limit, etc.
        print(f"[claude] escalation failed: {type(e).__name__}: {e}", flush=True)
        _dbg(f"FAILED {type(e).__name__}: {e}")
        return
