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
        return enabled()
    if name == "API_KEY":
        return api_key()
    raise AttributeError(name)


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


def pick_tier(user_text: str) -> Optional[str]:
    """'sonnet' | 'opus' | None (stay local). Pure regex, microseconds."""
    if not enabled() or not user_text:
        return None
    if _NEVER_RE.search(user_text):
        return None
    if _OPUS_RE.search(user_text):
        return "opus"
    if _SONNET_RE.search(user_text) or len(user_text) > 600:
        return "sonnet"
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
    while out and out[0]["role"] != "user":
        out.pop(0)
    return "\n\n".join(system_parts), out


async def stream(messages: list, tier: str, max_tokens: Optional[int] = None) -> AsyncIterator[str]:
    """Yield text deltas from Claude. Yields nothing (and never raises) if Claude
    is unreachable or slow to first token, so the caller can fall back to local."""
    try:
        system, msgs = _to_anthropic(messages)
        if not msgs:
            return
        print(f"[claude] escalating -> {tier} ({MODELS[tier]})", flush=True)
        cm = _get_client().messages.stream(
            model=MODELS[tier],
            max_tokens=min(max_tokens or MAX_TOKENS[tier], MAX_TOKENS[tier]),
            system=system,
            messages=msgs,
        )
        async with cm as s:
            it = s.text_stream.__aiter__()
            try:
                first = await asyncio.wait_for(it.__anext__(), timeout=FIRST_TOKEN_TIMEOUT)
            except (StopAsyncIteration, asyncio.TimeoutError):
                return
            yield first
            async for t in it:
                yield t
    except asyncio.CancelledError:
        raise
    except Exception as e:  # network, auth, rate limit, etc.
        print(f"[claude] escalation failed: {type(e).__name__}: {e}", flush=True)
        return
