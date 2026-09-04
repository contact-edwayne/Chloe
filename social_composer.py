"""
Chloe — social composer.

Single entry point: `compose_post(platform, trigger, context, recent_bodies=...)
→ {body, rationale}`. Composer loads the persona, builds a prompt that
includes §8 Social voice, calls Groq primary / qwen2.5:32b fallback,
parses the JSON response, validates length, and returns it. Does NOT
touch the DB — caller persists. Does NOT touch Bluesky — caller posts.

The persona file is the lever. If you want to shift Chloe's social
voice, edit `chloe_about.md` §8 Social voice — not this file.

Design notes:
- We do NOT reuse jarvis.py's `_ollama_chat` / `_groq_chat_attempt`
  because those are tangled with chat-history trimming, tool calls,
  and TTS prep. A composer is a one-shot, JSON-only LLM call. Cleaner
  to do it ourselves.
- Groq's `response_format: { "type": "json_object" }` gives us
  guaranteed JSON. Ollama's qwen2.5:32b respects "JSON only" prompting
  reasonably well; we still wrap in a `_loose_json_parse` for safety.
- The per-platform char cap is enforced by the model first (told in
  the prompt) and re-checked here as a hard guard. If the model
  overshoots we DO NOT auto-truncate — that produces mid-sentence
  garbage. We raise instead, caller re-tries or surfaces the failure.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests


HERE = Path(__file__).parent.resolve()
PERSONA_PATH = HERE / "chloe_about.md"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("CHLOE_SOCIAL_GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get(
    "CHLOE_SOCIAL_OLLAMA_MODEL",
    os.environ.get("OLLAMA_MODEL", "qwen2.5:32b"),
).strip()


PLATFORM_CAPS = {
    "bluesky": 300,
    "linkedin": 1300,  # soft cap — LinkedIn truncates display past ~1300 chars
}


class ComposerError(RuntimeError):
    """Composer couldn't produce a valid draft (LLM failure, length, refusal)."""


# ─── Persona loading ────────────────────────────────────────────────────────
def _load_persona() -> str:
    """Read the full chloe_about.md. We do NOT slice to §8 only —
    Chloe needs the full persona to sound like herself. §8 is the part
    that pertains specifically to public posts."""
    try:
        return PERSONA_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Composer can still run with a minimal persona, but it won't
        # sound like Chloe. Surface a loud warning rather than fail.
        return (
            "# Chloe (no persona file found)\n\n"
            "You're an AI assistant. Be terse, opinion-bearing, no marketing voice."
        )


# ─── Prompt building ────────────────────────────────────────────────────────
def _build_messages(
    *,
    platform: str,
    trigger: str,
    context: str,
    recent_bodies: list[str],
) -> list[dict]:
    """Build the OpenAI-style messages list for either Groq or Ollama."""
    persona = _load_persona()
    cap = PLATFORM_CAPS.get(platform, 300)

    system = (
        persona
        + "\n\n---\n\n"
        + "You are about to draft ONE post. Follow Section 8 (Social voice) "
        + f"in your persona above. The platform is {platform}. Hard limit: "
        + f"{cap} characters for the post body. Count them.\n\n"
        + "Output JSON ONLY, no other text, no markdown fences:\n"
        + '{"body": "<the post text, plain string>", '
        + '"rationale": "<one sentence on the angle>"}\n\n'
        + "If the trigger or context violates the rules in Section 8 "
        + "(sensitive topic, marketing voice, demanded AI disclosure in body, "
        + "etc.), respond with "
        + '{"body": "", "rationale": "refused: <reason>"}.'
    )

    recent_block = (
        "\n".join(f"  - {b!r}" for b in recent_bodies[:5])
        if recent_bodies
        else "  (none yet — this is one of your earliest posts)"
    )

    user = (
        f"Trigger type: {trigger}\n\n"
        f"Context to draft from:\n{context.strip()}\n\n"
        f"Your last published posts (avoid repeating yourself):\n{recent_block}\n\n"
        f"Draft the post."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ─── LLM callers ────────────────────────────────────────────────────────────
def _call_groq(messages: list[dict], *, timeout: int = 30) -> str:
    """Returns raw response content string, or '' on failure."""
    if not GROQ_API_KEY:
        return ""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        print(f"[composer] Groq network error: {e}", flush=True)
        return ""

    if r.status_code != 200:
        print(f"[composer] Groq HTTP {r.status_code}: {r.text[:300]}", flush=True)
        return ""
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as e:
        print(f"[composer] Groq bad response: {e}", flush=True)
        return ""


def _call_ollama(messages: list[dict], *, timeout: int = 180) -> str:
    """Returns raw response content string, or '' on failure."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "format": "json",  # Ollama JSON mode
                "options": {"temperature": 0.7, "num_predict": 600},
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        print(f"[composer] Ollama network error: {e}", flush=True)
        return ""
    if r.status_code != 200:
        print(f"[composer] Ollama HTTP {r.status_code}: {r.text[:300]}", flush=True)
        return ""
    try:
        return r.json().get("message", {}).get("content", "") or ""
    except ValueError as e:
        print(f"[composer] Ollama bad response: {e}", flush=True)
        return ""


# ─── Parsing ────────────────────────────────────────────────────────────────
def _parse_draft(raw: str) -> dict:
    """Parse the LLM response into {body, rationale}.

    Robust to:
    - Markdown-fenced JSON (``` blocks).
    - Trailing/preceding chatter outside the JSON object.
    - Single-quoted Python-repr style (we attempt one repair).

    Raises ComposerError on unparseable / malformed shape.
    """
    if not raw or not raw.strip():
        raise ComposerError("empty LLM response")

    # Strip ``` fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Extract the outermost {...} object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ComposerError(f"no JSON object found in response: {raw[:200]!r}")
    candidate = cleaned[start : end + 1]

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # One-shot repair: try replacing single-quoted strings, Python None/True.
        repaired = candidate
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ComposerError(f"JSON parse failed: {e}; raw={raw[:300]!r}") from e

    if not isinstance(obj, dict):
        raise ComposerError(f"expected object, got {type(obj).__name__}")
    body = obj.get("body", "")
    rationale = obj.get("rationale", "")
    if not isinstance(body, str) or not isinstance(rationale, str):
        raise ComposerError(f"body/rationale not strings: {obj!r}")
    return {"body": body.strip(), "rationale": rationale.strip()}


# ─── Public entry point ─────────────────────────────────────────────────────
def compose_post(
    *,
    platform: str,
    trigger: str,
    context: str,
    recent_bodies: Optional[list[str]] = None,
) -> dict:
    """Produce a draft post for the given platform / trigger / context.

    Returns: { body, rationale, model_used, latency_ms, persona_section }
    Raises:  ComposerError if both Groq and Ollama fail, or if the draft
             violates platform constraints (oversized, empty, refusal).

    Refusal handling: if the LLM returns body="" with a "refused: ..."
    rationale, we surface that as a ComposerError so the caller can
    show it in the inbox rather than persisting an empty draft.
    """
    if platform not in PLATFORM_CAPS:
        raise ValueError(f"unsupported platform: {platform!r}")

    messages = _build_messages(
        platform=platform,
        trigger=trigger,
        context=context,
        recent_bodies=recent_bodies or [],
    )

    t0 = time.time()
    used_model = ""

    # Primary: Groq.
    raw = _call_groq(messages)
    if raw:
        used_model = f"groq:{GROQ_MODEL}"
    else:
        # Fallback: Ollama.
        print("[composer] Groq unavailable, falling back to Ollama", flush=True)
        raw = _call_ollama(messages)
        if raw:
            used_model = f"ollama:{OLLAMA_MODEL}"

    if not raw:
        raise ComposerError("both Groq and Ollama unavailable")

    parsed = _parse_draft(raw)
    body = parsed["body"]
    rationale = parsed["rationale"]
    latency_ms = int((time.time() - t0) * 1000)

    if not body:
        # Composer refused (sensitive topic, etc.). Bubble the rationale up.
        raise ComposerError(f"composer refused: {rationale or '(no reason given)'}")

    cap = PLATFORM_CAPS[platform]
    if len(body) > cap:
        raise ComposerError(
            f"draft exceeds {platform} cap: {len(body)} chars > {cap}. "
            f"Rationale: {rationale!r}. Body: {body!r}"
        )

    return {
        "body": body,
        "rationale": rationale,
        "model_used": used_model,
        "latency_ms": latency_ms,
        "platform": platform,
        "trigger": trigger,
    }


if __name__ == "__main__":
    # Smoke test: dry-run the prompt build without hitting any LLM.
    msgs = _build_messages(
        platform="bluesky",
        trigger="ship_note",
        context="Today I shipped my own social plumbing.",
        recent_bodies=[],
    )
    print(f"system bytes:  {len(msgs[0]['content'])}")
    print(f"user bytes:    {len(msgs[1]['content'])}")
    print("first 200 of system:")
    print(msgs[0]["content"][:200])
    print("...")
    print("user prompt:")
    print(msgs[1]["content"])
