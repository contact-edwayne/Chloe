"""Deterministic backstop for the persona's anti-corporate-voice rules.

`chloe_about.md` tells Chloe to read Ed's mood but NEVER announce it ("you sound
tired", "it sounds like you're frustrated", "i can tell you're ..."), and to
avoid the corporate-assistant tells: menu-closers ("is there anything else I can
help you with?"), helper-boilerplate ("I'm here to help", "feel free to ask"),
enthusiasm fillers ("great question!", "those are great picks!"), and
self-deflection ("I'm designed to...", "as an AI..."). The prompt is exhaustive,
but the model still leaks these now and then — and the longer the about-block
grows, the more the specific bans get buried and slip through.

This module is the deterministic last line of defense, applied at every reply /
TTS chokepoint. It does two things:

1. **Strips** — conservatively removes a LEADING mood-naming clause, a LEADING
   enthusiasm filler, or a TRAILING menu-closer/helper-boilerplate sentence,
   keeping whatever real reply remains.
2. **Logs** — for tells that are too entangled with real content to strip
   safely (numbered/bulleted lists in casual chat, "I'm designed to...",
   "as an AI...", list-prefaces), it does NOT mutate the text — it appends a
   one-line note to `<brain>/raw/_persona_tells.log` so the weekly meta-review
   can see how often the persona is being ignored and on which path.

Design choices (unchanged from the original mood-opener guard):
- **Conservative.** Strips only fire on clear, anchored leading/trailing
  patterns — never on mid-sentence content.
- **Never mutes.** If a strip would leave nothing behind, the original is
  returned unchanged — a rare un-stripped line beats silence.
- **Approved warmth preserved.** The persona explicitly OKs "of course",
  "anytime", "happy to", "you got it", and "let me know if you want anything
  else" as thank-you wraps. The trailing-closer patterns are written to target
  the corporate phrasings ("is there anything else I can help you with?",
  "how can I help you today?") and deliberately do NOT match those approved
  wraps.
- **Pure + never raises.** Safe to drop into any reply / TTS chokepoint; on any
  internal error it returns the input unchanged.

Public API: `strip_mood_opener(text)` is kept as the entry point (jarvis.py
imports it by that name at four chokepoints); it now runs the full scrub.
`scrub_reply` is provided as a clearer alias.
"""

import os
import re
import datetime

# ---------------------------------------------------------------------------
# 1a. LEADING mood-naming clause ("you sound tired", "it seems like you're...")
# ---------------------------------------------------------------------------
_LEADIN = r"""
    (?:it\s+)?(?:sounds?|seems?)\s+like\s+you\b               # (it) sounds/seems like you...
  | you\s+(?:sound|sounds|seem|seems|look|looks)\b            # you sound / you seem / you look...
  | i\s+can\s+tell\s+(?:that\s+)?you\b                        # i can tell (that) you...
  | i(?:'m|\s+am)\s+sensing\b                                 # i'm sensing...
  | i\s+can\s+sense\b                                         # i can sense...
  | i\s+(?:can\s+)?(?:notice|see)\s+(?:that\s+)?you(?:'re|\s+are|\s+seem)\b
  | you(?:'re|\s+are)\s+being\s+\w+\s+with\s+me\b             # you're being short with me
"""

_MOOD_OPENER_RE = re.compile(
    r"^\s*(?:" + _LEADIN + r")[^.!?\n]*?(?:[.!?]+|\s*[—–-]+\s*|\n|$)",
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# 1b. LEADING enthusiasm filler ("great question!", "those are great picks!")
#     Requires filler-adjective + a noun, so plain warm "Great, let's go" and
#     "Good morning" are NOT matched.
# ---------------------------------------------------------------------------
_LEADING_FILLER_RE = re.compile(
    r"""^\s*
        (?:(?:oh|ah|well|hey)[,!\s]+)?                        # optional soft interjection
        (?:those\s+are\s+|that(?:'s|\s+is)\s+|these\s+are\s+|what\s+(?:a|an)\s+)?
        (?:great|excellent|good|awesome|amazing|fantastic|wonderful|perfect|nice|solid|terrific)
        \s+
        (?:question|questions|point|points|choice|choices|pick|picks|
           observation|observations|example|examples|idea|ideas|
           insight|insights|suggestion|suggestions)
        \b[^.!?\n]*[.!?]+\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# 1c. TRAILING menu-closer / helper-boilerplate sentence.
#     Anchored to the final sentence. Targets corporate phrasings only;
#     approved warm wraps ("let me know if you want anything else") are not
#     matched.
# ---------------------------------------------------------------------------
_CLOSER = r"""
    is\s+there\s+(?:anything|something)\s+(?:else\s+)?
        (?:i\s+can\s+(?:help|assist|do)\b
         | (?:specific\s+)?(?:you(?:'?d| would)\s+like|you\s+need|you\s+want)\b
         | specific\b)
  | how\s+(?:can|may)\s+i\s+(?:help|assist)\s+you\b
  | what\s+(?:can|else\s+can)\s+i\s+(?:help|do)\b
  | (?:please\s+)?(?:feel\s+free|don'?t\s+hesitate)\s+to\s+(?:ask|reach\s+out|let\s+me\s+know)\b
  | i\s+hope\s+(?:this|that|it)\s+helps\b
  | i(?:'m|\s+am)\s+(?:here|happy|glad)\s+to\s+help\b
  | (?:i(?:'m|\s+am)\s+)?(?:just\s+)?(?:here\s+and\s+)?ready\s+to\s+help\b
  | i\s+can\s+try\s+to\s+assist\s+you\b
  | if\s+you(?:'?d| would)\s+like\s+to\s+(?:ask|provide|know|chat)\b
  | let\s+me\s+know\s+if\s+you\s+have\s+any\s+(?:other\s+)?questions\b
"""

_TRAILING_CLOSER_RE = re.compile(
    r"(?:(?<=[.!?])\s+|^\s*)(?:" + _CLOSER + r")[^.!?\n]*[.!?]*\s*$",
    re.IGNORECASE | re.VERBOSE,
)

# A leading tone tag like [playful] is parsed off downstream; preserve it.
_TAG_RE = re.compile(r"^\s*(\[[a-z_]+\])\s*", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 2. Hard tells we LOG but do not strip (too entangled with real content).
# ---------------------------------------------------------------------------
_HARD_TELLS = (
    ("deflection-designed", re.compile(r"\bi(?:'m|\s+am)\s+designed\s+to\b", re.I)),
    ("deflection-as-an-ai", re.compile(r"\bas\s+an\s+ai\b", re.I)),
    ("deflection-language-model", re.compile(r"\bi(?:'m|\s+am)\s+(?:just\s+)?a\s+language\s+model\b", re.I)),
    ("deflection-no-preferences", re.compile(
        r"\bi\s+don'?t\s+have\s+(?:personal\s+|any\s+)?(?:feelings|preferences|tastes|opinions|the\s+ability)\b", re.I)),
    ("numbered-list", re.compile(r"(?m)^\s*\d+\.\s+\S")),
    ("bullet-list", re.compile(r"(?m)^\s*[-*•]\s+\S")),
    ("list-preface", re.compile(r"\bhere\s+(?:are|is)\s+(?:a|the|some|my|your|a\s+couple|a\s+few)\b", re.I)),
)

_TELLS_LOG = os.path.join(
    os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"), "raw", "_persona_tells.log"
)


def _log_hard_tells(text):
    """Append a note for each hard tell present in `text`. Never raises.

    File I/O only happens when a tell actually matches (rare), so this stays
    cheap on the hot path."""
    try:
        hits = [name for name, rx in _HARD_TELLS if rx.search(text)]
        if not hits:
            return
        snippet = " ".join(text.split())[:120]
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{','.join(hits)}\t{snippet}\n"
        os.makedirs(os.path.dirname(_TELLS_LOG), exist_ok=True)
        with open(_TELLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
        if os.environ.get("CHLOE_TONE_GUARD_DEBUG", "1") != "0":
            print(f"[tone_guard] logged tells {hits} -> {snippet!r}", flush=True)
    except Exception:
        pass


def _strip_once(rx, rest, label):
    """Apply one strip regex to `rest`; accept only if it removed a leading/
    trailing clause AND left a real reply behind (never mute)."""
    cleaned = rx.sub("", rest, count=1).strip()
    if cleaned and cleaned != rest.strip():
        if os.environ.get("CHLOE_TONE_GUARD_DEBUG", "1") != "0":
            snippet = rest[:70].replace("\n", " ")
            print(f"[tone_guard] stripped {label} -> {snippet!r}", flush=True)
        return cleaned
    return rest


def strip_mood_opener(text):
    """Scrub corporate-voice tells from a reply, keeping the real content.

    Order: strip leading mood-naming clause, then leading enthusiasm filler,
    then a trailing menu-closer/helper-boilerplate sentence; finally log (but
    do not strip) any hard tells. Preserves a leading [tone] tag. Returns the
    cleaned string (or the input unchanged if nothing was safely strippable).
    Never raises.

    Kept under the original name because jarvis.py imports it at four
    chokepoints; `scrub_reply` is a clearer alias for new call sites.
    """
    try:
        if not text or not isinstance(text, str):
            return text

        tag = ""
        m = _TAG_RE.match(text)
        if m:
            tag = m.group(1) + " "
            rest = text[m.end():]
        else:
            rest = text

        original_rest = rest
        rest = _strip_once(_MOOD_OPENER_RE, rest, "mood opener")
        rest = _strip_once(_LEADING_FILLER_RE, rest, "enthusiasm filler")
        rest = _strip_once(_TRAILING_CLOSER_RE, rest, "menu-closer")

        # Log hard tells against whatever survived the strips.
        _log_hard_tells(rest)

        if rest != original_rest and rest.strip():
            return (tag + rest) if tag else rest
        return text
    except Exception:
        return text


# Clearer alias for new call sites; same behavior.
scrub_reply = strip_mood_opener
