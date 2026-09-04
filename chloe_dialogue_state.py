"""Per-session dialogue-state scratchpad for Chloe.

Phase 1 of the 2026-06-01 conversational-optimization plan. The persona
(`chloe_about.md`) leans hard on reading Ed's mood and tracking whether he's
*processing* (thinking out loud) or *distilling* (wants a crisp answer) — but
nothing persisted that read across turns, so every turn re-derived it from
scratch and the mode/mood never stuck. The "stay in processing mode, don't jump
to advice" rule had no memory of having entered processing mode last turn.

This module maintains a tiny, deterministic working-memory record for the active
session and renders it as a compact system-prompt block, e.g.:

    ## Session state (your running read — behave it, never announce it)
    mode: processing (since 3 turns) — listen first, don't race to advice
    read: flat/tired — quieter, fewer words, no advice unless asked
    length: short — match his brevity
    open loops: whether to enable Stage-4; the deploy approach
    on the table: deploy script, qwen2.5:32b, Friday meta-review

Design contract (mirrors `_recent_context_block` / the tone guard):
- **Heuristic only.** No LLM/embedding call → no added latency.
- **Pure + never raises.** Any failure yields an empty block and a best-effort
  state; a turn can never break here.
- **Single-user, single active session.** State lives in one JSON file under
  `<brain>/raw/`. A gap > SESSION_GAP_H hours since the last turn starts a fresh
  session (mode/loops/entities reset).
- **Guidance, not output.** The block is a private read injected into the system
  prompt. The header reasserts the persona's show-don't-tell rule so the model
  behaves the read instead of narrating it.

Public API:
    block_for(messages) -> str   # update state from the convo, return the block
    update(messages)    -> dict  # update + persist, return the state (for tests)
    format_block(state) -> str   # render a state dict (pure)
"""

from __future__ import annotations

import json
import os
import re
import time

try:
    from chloe_lock import locked
except Exception:  # pragma: no cover - lock is best-effort
    import contextlib

    @contextlib.contextmanager
    def locked(*_a, **_k):
        yield

# ─── Config ────────────────────────────────────────────────────────────────

SESSION_GAP_H = 6.0          # hours of silence that starts a fresh session
LOOP_TTL_TURNS = 8           # drop an open loop after this many turns unseen
MAX_LOOPS = 3
MAX_ENTITIES = 6


def _state_path() -> str:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    return os.path.join(root, "raw", "dialogue_state.json")


# ─── Persistence (best-effort) ──────────────────────────────────────────────

def _blank_state() -> dict:
    return {
        "mode": "neutral",
        "mode_since_turn": 0,
        "mood": "neutral",
        "length": "medium",
        "loops": [],         # list of {"text": str, "turn": int}
        "entities": [],      # recency-ordered list[str], most-recent first
        "flow": {},          # last flow signals (engagement/circling/subtext/…)
        "turn": 0,
        "updated_ts": 0.0,
    }


def _load() -> dict:
    try:
        with open(_state_path(), "r", encoding="utf-8") as fh:
            st = json.load(fh)
        if not isinstance(st, dict):
            return _blank_state()
        base = _blank_state()
        base.update({k: st[k] for k in base if k in st})
        return base
    except Exception:
        return _blank_state()


def _save(state: dict) -> None:
    try:
        p = _state_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        pass


# ─── Heuristic reads ────────────────────────────────────────────────────────

_DISTILL_RE = re.compile(
    r"\b(what should i|give me (?:a|the) plan|make me a plan|"
    r"summari[sz]e|tl;?dr|bottom line|just tell me|my options|"
    r"what(?:'s| is) the (?:plan|move|call|verdict)|"
    r"steps?\b|decide|pick one|which (?:one|should))\b",
    re.IGNORECASE,
)
_PROCESS_RE = re.compile(
    r"\b(what do you think|help me (?:figure|think)|i don'?t know where|"
    r"i'?m not sure|not sure (?:if|what|how|whether)|"
    r"go(?:ing)? back and forth|on the fence|thinking (?:about|through)|"
    r"trying to (?:figure|work) (?:out|through)|talk(?:ing)? (?:it|this) (?:out|through)|"
    r"i keep (?:thinking|wondering))\b",
    re.IGNORECASE,
)

# Mood cue tables. First match (in this order) wins; lean to the gentler read
# when uncertain (persona rule).
_MOOD_CUES = (
    ("frustrated", re.compile(
        r"\b(frustrat\w+|annoy\w+|pissed|fed up|this isn'?t working|"
        r"still (?:broken|not working|failing)|wtf|ffs|so done|"
        r"why (?:won'?t|isn'?t|does)\w*\b.*\?)\b", re.IGNORECASE)),
    ("sad", re.compile(
        r"\b(i'?m (?:sad|down|low|depressed)|rough (?:day|week)|"
        r"feel(?:ing)? (?:awful|terrible|like shit|hopeless)|"
        r"hard (?:day|week|time)|burn(?:t|ed) out)\b", re.IGNORECASE)),
    ("tired", re.compile(
        r"\b(tired|exhausted|drained|wiped|knackered|so sleepy|"
        r"no energy|can'?t focus|fried|meh|whatever|i guess|fine\.)\b",
        re.IGNORECASE)),
    ("excited", re.compile(
        r"(!!|\b(let'?s go|stoked|pumped|so excited|this is huge|"
        r"can'?t wait|amazing|insane|let'?s gooo+|hyped)\b)",
        re.IGNORECASE)),
    ("happy", re.compile(
        r"\b(haha+|lol|lmao|nice|love (?:it|this|that)|"
        r"that'?s great|sweet|awesome)\b", re.IGNORECASE)),
)

# Behavioral guidance appended to each read so the model knows what to DO.
_MOOD_GUIDE = {
    "frustrated": "skip preamble, go straight to substance, one concrete step at a time",
    "sad": "presence over advice — reflect before suggesting anything",
    "tired": "brief and gentle, no advice unless asked",
    "excited": "match the energy, riff with him",
    "happy": "warm, tease back, lean into the moment",
    "tense": "quieter, fewer words, one grounded follow-up",
    "neutral": "default mode — be yourself",
}
_MODE_GUIDE = {
    "processing": "listen first, reflect, ONE good question — don't race to advice",
    "distilling": "find the through-line, name the crux, give only what moves the needle",
    "neutral": "",
}

# Domain lexicon for entity extraction — Chloe's own world. Case-insensitive;
# the canonical (display) form is the value.
_LEXICON = {
    "stage 4": "Stage-4", "stage-4": "Stage-4", "stage4": "Stage-4",
    "stage 3": "Stage-3", "stage-3": "Stage-3",
    "deploy": "deploy", "rollback": "rollback",
    "watchdog": "watchdog", "proposal": "proposal", "autonomous": "autonomous",
    "recall": "recall", "wiki": "wiki", "persona": "persona",
    "mic": "mic", "samson": "Samson", "voice": "voice", "wake word": "wake-word",
    "ollama": "Ollama", "groq": "Groq", "qwen": "qwen", "kokoro": "Kokoro",
    "elevenlabs": "ElevenLabs", "whisper": "whisper", "embed": "embeddings",
    "wallet": "wallet", "lightning": "Lightning", "lights": "lights",
    "chess": "chess", "arcade": "arcade", "emulator": "emulator",
    "pokemon": "Pokémon", "pokémon": "Pokémon",
    "vision": "vision", "hud": "HUD", "tts": "TTS", "stt": "STT",
    "meta-review": "meta-review", "meta review": "meta-review",
    "backup": "backup", "schedule": "scheduling", "job": "jobs",
    "context window": "context-window", "num_ctx": "num_ctx",
}

# Patterns for non-lexicon entities: model versions (qwen2.5:32b, llama-3.3),
# ALLCAPS acronyms (3-6), and CamelCase / dotted identifiers.
_MODELVER_RE = re.compile(r"\b[a-z][a-z0-9.]*(?::|-)\d[\w.:-]*\b", re.IGNORECASE)
_ACRONYM_RE = re.compile(r"\b[A-Z]{3,6}\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b")
_DOTPY_RE = re.compile(r"\b\w+\.py\b")

_STOP_ACR = {"AND", "THE", "FOR", "YOU", "ARE", "NOT", "BUT", "WAS",
             "ALL", "ANY", "CAN", "GET", "GOT", "HOW", "WHY", "WHO"}


def _read_length(text: str) -> str:
    n = len(text.strip())
    if n < 25:
        return "short"
    if n < 200:
        return "medium"
    return "long"


def _read_mood(text: str) -> str:
    t = text or ""
    for name, rx in _MOOD_CUES:
        if rx.search(t):
            return name
    # Punctuation/structure tells for "tense": very short + sharp, or curt
    # one-word with a period.
    stripped = t.strip()
    if stripped and len(stripped) < 18 and stripped.endswith(".") and " " not in stripped:
        return "tense"
    return "neutral"


def _read_mode(text: str, prev_mode: str) -> str:
    t = text or ""
    distill = bool(_DISTILL_RE.search(t))
    process = bool(_PROCESS_RE.search(t))
    long_dump = len(t) > 400 and "?" not in t[-60:]
    if distill and not process:
        return "distilling"
    if process or long_dump:
        return "processing"
    # Ambiguous turn: stay sticky toward processing (persona: don't jump out
    # of it too fast), otherwise fall to neutral.
    if prev_mode == "processing":
        return "processing"
    return "neutral"


def _clause(text: str, limit: int = 80) -> str:
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(",.;:")
    return (cut or s[:limit]) + "…"


_LOOP_RE = re.compile(
    r"\b(should i|whether (?:to|i)|deciding|figure out|not sure (?:if|whether)|"
    r"go(?:ing)? back and forth|on the fence|or should|do i\b.*\?)\b",
    re.IGNORECASE,
)


def _update_loops(loops: list, user_text: str, turn: int) -> list:
    out = [l for l in loops
           if isinstance(l, dict) and (turn - int(l.get("turn", 0))) < LOOP_TTL_TURNS]
    t = (user_text or "").strip()
    is_loop = bool(_LOOP_RE.search(t)) or (t.endswith("?") and len(t) > 12)
    if is_loop:
        snippet = _clause(t)
        if snippet and not any(l.get("text") == snippet for l in out):
            out.append({"text": snippet, "turn": turn})
    return out[-MAX_LOOPS:]


def _extract_entities(texts: list, prev: list) -> list:
    found: list[str] = []

    def _add(e: str):
        if e and e not in found:
            found.append(e)

    # Scan most-recent text first so recency ordering falls out naturally.
    for txt in texts:
        low = (txt or "").lower()
        for key, disp in _LEXICON.items():
            if key in low:
                _add(disp)
        for m in _MODELVER_RE.findall(txt or ""):
            if any(c.isdigit() for c in m) and len(m) <= 24:
                _add(m)
        for m in _ACRONYM_RE.findall(txt or ""):
            if m not in _STOP_ACR:
                _add(m)
        for m in _CAMEL_RE.findall(txt or ""):
            _add(m)
        for m in _DOTPY_RE.findall(txt or ""):
            _add(m)
    # Blend in a little of the previous set so entities don't whiplash, but
    # let the current turn lead.
    for e in prev:
        _add(e)
    # Drop any entity that's a case-insensitive substring of a more specific
    # one already found (e.g. "qwen" when "qwen2.5:32b" is present).
    deduped = [
        e for e in found
        if not any(e.lower() != o.lower() and e.lower() in o.lower() for o in found)
    ]
    return deduped[:MAX_ENTITIES]


# ─── Main entry points ──────────────────────────────────────────────────────

# ─── Flow / Theory-of-Mind signals (Phase 4C) ───────────────────────────────
# Heuristic, hot-path-safe (no LLM). A future LLM read-pass can replace/augment
# _flow_signals without touching the call sites.

_CORRECTION_RE = re.compile(
    r"\b(no,|that'?s not|not what i|i said|that'?s wrong|nope|incorrect|"
    r"you misunderstood|i meant)\b", re.IGNORECASE)
_SUBTEXT_RE = re.compile(
    r"\b(i'?m fine|it'?s fine|don'?t worry about it|never ?mind|nvm|forget it|"
    r"it'?s nothing|doesn'?t matter|whatever)\b", re.IGNORECASE)
_STOP = set("the a an and or but to of in on for is it i you he she we they that "
            "this with as at be do my your me him her them so if not no yes".split())


def _content_words(t: str) -> set:
    return {w for w in re.findall(r"[a-z']+", (t or "").lower())
            if w not in _STOP and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _recent_user_texts(messages, n: int) -> list:
    out = []
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            out.append(m["content"])
            if len(out) >= n:
                break
    return out  # newest first


def _flow_signals(messages, user_text: str) -> dict:
    """Cheap read of conversational shape: engagement, circling, repeated
    correction, and surface subtext ('I'm fine'). Never raises."""
    sig = {"engagement": "med", "circling": False,
           "repeated_correction": False, "subtext": ""}
    try:
        users = _recent_user_texts(messages, 4)  # newest first
        ut = user_text or ""
        if _SUBTEXT_RE.search(ut) and len(ut.strip()) < 40:
            sig["subtext"] = "says it's fine / brushes it off — don't fully take it at face value"
        if sum(1 for u in users[:3] if _CORRECTION_RE.search(u or "")) >= 2:
            sig["repeated_correction"] = True
        if len(users) >= 2 and _jaccard(_content_words(users[0]), _content_words(users[1])) >= 0.6:
            sig["circling"] = True
        if len(users) >= 3:
            lens = sorted(len(u) for u in users[1:4])
            med = lens[len(lens) // 2]
            cur = len(ut)
            if med >= 60 and cur < 0.4 * med:
                sig["engagement"] = "low"
            elif cur > 1.5 * (med or 1):
                sig["engagement"] = "high"
        return sig
    except Exception:
        return sig


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content") or ""
    return ""


def update(messages, read=None) -> dict:
    """Compute + persist dialogue state from a messages list. Never raises.

    `read` is an optional LLM read-pass result (chloe_read.llm_read); when it's
    confident (certainty >= 0.5) it overrides the heuristic mood and enriches the
    flow subtext/engagement."""
    try:
        # Hold the lock across the whole load->mutate->save so a concurrent
        # voice/chat turn can't clobber this one's update (os.replace only makes
        # each write atomic, not the read-modify-write as a whole).
        with locked("dialogue_state"):
            st = _load()
            now = time.time()
            # Fresh session on a long gap.
            if st.get("updated_ts") and (now - float(st["updated_ts"])) > SESSION_GAP_H * 3600:
                st = _blank_state()

            user_text = _last_user_text(messages)
            st["turn"] = int(st.get("turn", 0)) + 1
            turn = st["turn"]

            st["length"] = _read_length(user_text)
            st["mood"] = _read_mood(user_text)

            prev_mode = st.get("mode", "neutral")
            new_mode = _read_mode(user_text, prev_mode)
            if new_mode != prev_mode:
                st["mode"] = new_mode
                st["mode_since_turn"] = turn
            elif not st.get("mode_since_turn"):
                st["mode_since_turn"] = turn

            st["loops"] = _update_loops(st.get("loops", []), user_text, turn)

            # Entities: most-recent user + assistant turns, newest first.
            recent_texts = []
            for m in reversed(messages or []):
                if isinstance(m, dict) and m.get("content"):
                    recent_texts.append(m["content"])
                if len(recent_texts) >= 6:
                    break
            st["entities"] = _extract_entities(recent_texts, st.get("entities", []))
            st["flow"] = _flow_signals(messages, user_text)

            # Mood ownership (Phase 5): the heuristic OWNS mood — one source of
            # truth, no silent override. Two systems writing the same slot was a
            # drift risk. The gated 3B read-pass now only AUGMENTS the signals
            # heuristics genuinely can't do — subtext + engagement. What the
            # read-pass *thought* the mood was is stashed on the side
            # (flow.read_mood) purely so the trace can show heuristic↔llm
            # disagreement; it does not change behavior.
            if isinstance(read, dict) and float(read.get("certainty", 0) or 0) >= 0.5:
                if read.get("subtext"):
                    st["flow"]["subtext"] = read["subtext"]
                if read.get("engagement") in ("high", "med", "low"):
                    st["flow"]["engagement"] = read["engagement"]
                if read.get("mood"):
                    st["flow"]["read_mood"] = read["mood"]

            st["updated_ts"] = now
            _save(st)
            return st
    except Exception:
        return _blank_state()


def format_block(state: dict) -> str:
    """Render a compact system-prompt block from a state dict. Pure; returns
    '' if there's nothing worth saying."""
    try:
        if not isinstance(state, dict):
            return ""
        mode_line = mood_line = length_line = loops_line = ents_line = ""

        mode = state.get("mode", "neutral")
        if mode != "neutral":
            since = int(state.get("turn", 0)) - int(state.get("mode_since_turn", 0)) + 1
            since = max(since, 1)
            guide = _MODE_GUIDE.get(mode, "")
            tail = f" — {guide}" if guide else ""
            mode_line = f"mode: {mode} (since {since} turn{'s' if since != 1 else ''}){tail}"

        mood = state.get("mood", "neutral")
        if mood != "neutral":
            guide = _MOOD_GUIDE.get(mood, "")
            tail = f" — {guide}" if guide else ""
            mood_line = f"read: {mood}{tail}"

        length = state.get("length", "medium")
        if length == "short":
            length_line = "length: he's terse — match his brevity, don't over-answer"
        elif length == "long":
            length_line = "length: he's expansive — you can be too, but don't pad"

        # Eagerness governor: when he's frustrated/tense/sad or wants a crisp
        # answer, explicitly hold the proactive callbacks + teasing. (These
        # moods/mode already produce a substantive line, so this never
        # un-suppresses an otherwise-empty block.)
        flow = state.get("flow", {}) or {}
        flow_lines = []
        if flow.get("repeated_correction"):
            flow_lines.append("you've corrected the same thing 2x+ — stop re-asserting; ask exactly what's off")
        if flow.get("circling"):
            flow_lines.append("you're circling the same point — name it and ask for the real blocker")
        if flow.get("subtext"):
            flow_lines.append("subtext: " + flow["subtext"])
        if flow.get("engagement") == "low":
            flow_lines.append("he's gone quiet/clipped — wrap up or change tack, don't pile on")

        eager_line = ""
        if (mood in ("frustrated", "tense", "sad") or mode == "distilling"
                or flow.get("engagement") == "low" or flow.get("repeated_correction")):
            eager_line = ("hold: skip proactive callbacks and teasing — "
                          "he wants focus, just help")

        loops = [l.get("text", "") for l in state.get("loops", []) if l.get("text")]
        if loops:
            loops_line = "open loops: " + "; ".join(loops)

        ents = state.get("entities", [])
        if ents:
            ents_line = "on the table: " + ", ".join(ents)

        # A bare length hint with no mode/mood/loops/entities isn't worth a
        # block — keeps acks and one-off factual turns clean.
        substantive = [mode_line, mood_line, loops_line, ents_line] + flow_lines
        if not any(substantive):
            return ""
        lines = ([mode_line, mood_line] + flow_lines
                 + [eager_line, length_line, loops_line, ents_line])
        lines = [ln for ln in lines if ln]
        header = ("\n\n## Session state (your running read — behave it, "
                  "never announce it)\n")
        return header + "\n".join(lines) + "\n"
    except Exception:
        return ""


def current() -> dict:
    """Return the persisted dialogue state (best-effort). Read-only — for
    observability (chloe_trace) so it can report this turn's mood/mode/flow
    without re-deriving them. Never raises."""
    try:
        return _load()
    except Exception:
        return _blank_state()


def block_for(messages, read=None) -> str:
    """Convenience for the chat/voice handlers: update state from the convo and
    return the rendered block. `read` is an optional LLM read-pass result.
    Never raises; '' on any failure."""
    try:
        return format_block(update(messages, read=read))
    except Exception:
        return ""
