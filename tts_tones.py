"""Tonal style mapping for Kokoro TTS output.

Chloe prefixes a reply with a tag like `[intimate]` / `[whispering]` /
`[submissive]`. This module parses the leading tag, updates sticky state
(so subsequent turns inherit until a new tag is emitted), strips the
tag from the text that gets spoken, and returns the (voice, speed) pair
the Kokoro engine should use.

Palette is editable — the (voice_id, speed) values are tuned conservatively
to stay within Kokoro's safe range (0.80–1.20 speed). Ed can iterate by
editing PALETTE directly; no jarvis.py changes needed.

Tone tags are sticky across turns: once Chloe enters [intimate], she stays
there until she emits a different tag. Drops back to neutral when:
  - she emits [neutral] (or any tag mapping to neutral)
  - nsfw_mode flips off (via the reset hook in nsfw_mode.set_enabled)
  - reset_tone() is called explicitly (e.g. from /tone reset slash)
"""

from __future__ import annotations

import re
import threading

# ─── Palette ───────────────────────────────────────────────────────────────
# (blend_with, mix, speed):
#   blend_with — name of an af_* Kokoro voice to mix toward, or None for the
#                pure baseline voice (KOKORO_VOICE, normally af_heart).
#   mix        — blend weight in [0, 1]: the spoken style embedding is
#                (1-mix)*style(baseline) + mix*style(blend_with). Keep mix
#                below ~0.5 so the baseline (Chloe's identity) stays dominant.
#   speed      — playback speed, or None to inherit KOKORO_SPEED. Clamp to
#                Kokoro's safe 0.5–2.0 range; we stay 0.80–1.12.
#
# DESIGN NOTE (2026-05-24): the 2026-05-15 note kept voice=None (speed-only)
# because a full voice SWAP (af_sky vs af_jessica) reads as a different woman,
# not a softer mood — so every tone blurred into neutral. The fix is
# voice-embedding BLENDING rather than swapping: kokoro_onnx.create() accepts
# voice as either a name string OR a float32 style ndarray, and
# get_voice_style(name) returns that array. A weighted sum of two style banks
# (baseline dominant) shifts timbre toward a mood while keeping Chloe's
# identity. parse_and_get returns the (blend_with, mix, speed) spec; jarvis's
# _kokoro_voice_arg(engine, blend_with, mix, base) resolves it against the
# loaded engine (returns the baseline NAME when there's nothing to blend, an
# ndarray otherwise, and the baseline NAME on any error). Ed tunes the blend
# targets + mixes below by ear; an unknown/failed blend safely falls back to
# the baseline voice.
PALETTE: dict[str, tuple[str | None, float, float | None]] = {
    # Always-available
    "neutral":    (None,        0.00, None),   # pure KOKORO_VOICE/SPEED
    "playful":    ("af_nicole", 0.30, 1.12),   # brighter, quicker, lighter
    "gentle":     ("af_sky",    0.20, 0.92),   # softer/slower — tired/flat/down reads
    "warm":       ("af_river",  0.20, 0.96),   # warmer timbre — affection/reassurance
    "bright":     ("af_nicole", 0.22, 1.08),   # genuine delight (vs playful's teasing)
    # Permissive
    "intimate":   ("af_sky",    0.35, 0.88),   # calmer timbre, lingering
    "whispering": ("af_sky",    0.45, 0.80),   # softest register, near-paused
    "submissive": ("af_sky",    0.25, 0.94),   # subtle soften, gentle pacing
    "breathy":    ("af_nova",   0.40, 0.90),   # airy timbre, drawn-out
    "sultry":     ("af_river",  0.35, 0.85),   # warmer/darker, deliberate
}

_DEFAULT_TONE = "neutral"
_LOCK = threading.Lock()
_CURRENT: dict = {"tone": _DEFAULT_TONE}

# Leading-tag regex. Tolerant of:
#  - Optional bold/italic wrappers (`**[intimate]**`, `*[intimate]*`)
#  - Optional leading whitespace
#  - Letters/digits/underscores inside the bracket (so `[x_rated]` would parse)
_TAG_RE = re.compile(
    r"""^\s*           # optional leading whitespace
        \**            # optional markdown bold prefix
        \*?            # optional markdown italic prefix
        \[\s*          # opening bracket
        ([a-zA-Z_][a-zA-Z0-9_-]*)  # tag name (captured)
        \s*\]          # closing bracket
        \*?\**         # optional closing markdown
        [\s:.,-]*      # optional trailing punctuation/whitespace
    """,
    re.VERBOSE,
)


def parse_and_get(
    text: str,
    default_speed: float | None = None,
) -> tuple[str, str | None, float, float | None]:
    """Parse a leading tone tag (if present), update sticky state, return
    (stripped_text, blend_with, mix, speed) for the current sticky tone.

    blend_with/mix describe how to mix the baseline voice toward a mood
    (see PALETTE); the caller (jarvis._kokoro_voice_arg) resolves them
    against the loaded Kokoro engine. speed falls back to default_speed
    for 'neutral' or when no tag has been seen.

    Always safe to call. Unknown tags are stripped but the sticky tone
    is left unchanged.
    """
    if not text:
        return text, None, 0.0, default_speed

    m = _TAG_RE.match(text)
    if m:
        tag = m.group(1).lower()
        if tag in PALETTE:
            with _LOCK:
                _CURRENT["tone"] = tag
        # Strip the tag span whether or not it was a known tag — we don't
        # want Chloe pronouncing 'open bracket made up word close bracket'.
        text = text[m.end():]

    with _LOCK:
        tone = _CURRENT["tone"]
    blend_with, mix, speed = PALETTE.get(tone, (None, 0.0, None))
    return (
        text,
        blend_with,
        mix,
        speed if speed is not None else default_speed,
    )


def current_tone() -> str:
    with _LOCK:
        return _CURRENT["tone"]


def reset_tone() -> None:
    """Force the sticky tone back to neutral. Called by nsfw_mode when the
    permissive flag flips off, and by the /tone reset slash if added."""
    with _LOCK:
        _CURRENT["tone"] = _DEFAULT_TONE


def set_tone(tone: str) -> bool:
    """Explicit override (e.g. from a slash command). Returns True on success,
    False if the tone isn't in the palette."""
    if tone not in PALETTE:
        return False
    with _LOCK:
        _CURRENT["tone"] = tone
    return True


# ─── Slash command handler ─────────────────────────────────────────────────
def try_handle_command(text: str) -> str | None:
    """Match `/tone`, `/tone status`, `/tone reset`, `/tone <name>`.
    Returns the spoken reply, or None to fall through."""
    if not text:
        return None
    t = text.strip().lower()
    if t in ("/tone", "/tone status"):
        return f"tone: {current_tone()}"
    if t == "/tone reset":
        reset_tone()
        return "tone reset to neutral."
    if t.startswith("/tone "):
        name = t.split(None, 1)[1].strip()
        if set_tone(name):
            return f"tone set to {name}."
        return f"unknown tone '{name}'. available: {', '.join(sorted(PALETTE))}."
    return None
