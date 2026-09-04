"""Per-turn persona composition for system-prompt injection.

`chloe_about.md` is the single source of truth for Chloe's self-knowledge, but
it has grown to ~5k tokens and jarvis.py injects it verbatim into EVERY chat and
voice system prompt. Two costs:

  1. It burns the Groq daily token budget on every turn.
  2. The longer it gets, the more the *behavioral* rules (the anti-corporate
     bans, tonal awareness, conversational texture) get diluted and ignored by
     the model — which is exactly the drift we see in the turn log.

This module trims the always-on footprint WITHOUT editing `chloe_about.md`, so
`persona_read`, the weekly backups, and the persona drift/evolution jobs all
keep seeing the whole file. The split happens only at injection time.

Sections are classified into four buckets:

  - **core**        — always injected. Identity, tonal awareness, the
                      thinking/processing modes, the anti-corporate voice bans,
                      conversational texture, warmth, architecture, Notes from
                      Ed. The stuff that must bind on every single turn.
  - **preference**  — seed preferences, specific favorites, "commit to ONE
                      favorite", knowledge anchors. Only needed when the turn is
                      actually about taste/opinions/fandom.
  - **voice**       — the TTS tone-tag catalog. Useless in a text reply; only
                      injected on the voice path.
  - **social**      — the public-posting voice. The social composer has its own
                      prompt path, so chat/voice never need this unless the turn
                      is explicitly about posting.

Safety contract: `compose()` only ever REMOVES gated sections. If parsing finds
nothing recognizable (or anything goes wrong), it returns the full body
unchanged. The worst case is "no trimming" — today's behavior — never a
stripped-down persona.
"""

import re

# A markdown header at depth 2 or 3 starts a new section.
_HEADER_RE = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")

# Header-text substrings (lowercased) → gated category. Anything that matches
# none of these is treated as CORE. Order matters: social/voice checked before
# the generic "voice"/"preference" words so e.g. "Voice & speech style" (a CORE
# bans section) is NOT mistaken for the tone-tag catalog.
_SOCIAL_KEYS = ("social voice",)
_VOICE_KEYS = ("voice tone tags",)
_PREFERENCE_KEYS = (
    "seed preferences",
    "specific favorites",
    "commit to one favorite",
    "knowledge anchors",
)


def _classify(header: str) -> str:
    h = header.lower()
    if any(k in h for k in _SOCIAL_KEYS):
        return "social"
    if any(k in h for k in _VOICE_KEYS):
        return "voice"
    if any(k in h for k in _PREFERENCE_KEYS):
        return "preference"
    return "core"


def _split_sections(body: str):
    """Split a persona body into a flat list of (category, text) blocks, one
    per `##`/`###` header. Any leading text before the first header is its own
    CORE block."""
    blocks = []
    preamble = []
    cur_header = None
    cur = []
    seen = False
    for ln in body.split("\n"):
        m = _HEADER_RE.match(ln)
        if m:
            if not seen:
                if "\n".join(preamble).strip():
                    blocks.append(("core", "\n".join(preamble).rstrip()))
                seen = True
            else:
                blocks.append((_classify(cur_header), "\n".join(cur).rstrip()))
            cur_header = m.group(2)
            cur = [ln]
        elif seen:
            cur.append(ln)
        else:
            preamble.append(ln)
    if seen:
        blocks.append((_classify(cur_header), "\n".join(cur).rstrip()))
    elif "\n".join(preamble).strip():
        blocks.append(("core", "\n".join(preamble).rstrip()))
    return blocks


# Turn-intent triggers for the gated buckets. Deliberately GENEROUS for
# preferences — losing her favorites exactly when Ed asks about them is the
# expensive failure, so we err toward including them.
_PREF_INTENT = re.compile(
    r"\b(favou?rite|prefer(?:ence)?|your\s+taste|your\s+opinion|"
    r"do\s+you\s+(?:like|enjoy|love|hate|prefer)|"
    r"what\s+do\s+you\s+think\s+(?:of|about)|"
    r"what\s+(?:are|do)\s+you\s+(?:into|enjoy)|"
    r"what(?:'?s| is)\s+your)\b",
    re.IGNORECASE,
)
_PREF_TOPIC = re.compile(
    r"\b(music|songs?|artists?|bands?|movies?|films?|shows?|anime|"
    r"books?|reading|colou?rs?|food|drinks?|tea|coffee|aesthetic|"
    r"characters?|madoka|death\s*note|blade\s*runner|interstellar|"
    r"zimmer|kajiura|bonobo|tycho|reznor)\b",
    re.IGNORECASE,
)
_SOCIAL_INTENT = re.compile(
    r"\b(bluesky|linkedin|tweet|posts?|posting|draft\s+a\s+post|"
    r"social\s+media|caption)\b",
    re.IGNORECASE,
)


def compose(about_body: str, user_text: str = "", voice: bool = False) -> str:
    """Return a trimmed persona body for system-prompt injection.

    Keeps every CORE section. Includes PREFERENCE sections when the turn looks
    like it's about taste/opinions (or when `user_text` is empty and we can't
    tell — err toward keeping). Includes the VOICE tone-tag catalog only on the
    voice path. Includes the SOCIAL voice only when the turn is about posting.

    Never raises; on any failure returns `about_body` unchanged.
    """
    try:
        body = (about_body or "").strip()
        if not body:
            return about_body
        blocks = _split_sections(body)
        # If parsing yielded nothing gated, there's nothing to trim — return
        # the body as-is rather than risk reflowing it.
        if not blocks or all(cat == "core" for cat, _ in blocks):
            return body

        text = user_text or ""
        want_pref = (not text.strip()) or bool(
            _PREF_INTENT.search(text) or _PREF_TOPIC.search(text)
        )
        want_social = bool(text.strip()) and bool(_SOCIAL_INTENT.search(text))

        kept = []
        for cat, chunk in blocks:
            if cat == "core":
                kept.append(chunk)
            elif cat == "preference" and want_pref:
                kept.append(chunk)
            elif cat == "voice" and voice:
                kept.append(chunk)
            elif cat == "social" and want_social:
                kept.append(chunk)

        out = "\n\n".join(c for c in kept if c.strip())
        return out if out.strip() else body
    except Exception:
        return about_body
