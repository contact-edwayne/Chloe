"""tts_lexicon — spoken-pronunciation overrides for Chloe's TTS path.

Problem this solves: respelling a word in Chloe's *text* reply does nothing to
how the speech engine says it (Kokoro / edge-tts / ElevenLabs read the real
word via grapheme-to-phoneme). The only reliable, engine-agnostic way to make
her *say* a word a specific way is to substitute a phonetic respelling into the
SPOKEN copy only, before synthesis — leaving the displayed text + history
untouched (same contract as the emoji strip in jarvis._clean_for_tts).

Usage (in jarvis.py):
    import tts_lexicon
    spoken = tts_lexicon.apply(spoken)   # right before kokoro.create() / etc.

The lexicon lives in `tts_lexicon.json` next to this module — a flat
{ "written form": "spoken respelling" } map. It hot-reloads on mtime change,
so Ed can tune a respelling by ear and hear it on the next sentence with NO
restart. Matching is case-insensitive and whole-word (Unicode word boundaries,
so accented forms like "Pokémon" match). Replacement preserves surrounding
text and punctuation; only the matched word is swapped.

Seed entry: Pokémon -> "poh kee mawn" (Ed's pronunciation, 2026-06-01). Spaces
are deliberate — they force the g2p engine to treat each chunk as its own
token so it can't re-mangle a hyphenated blob. Tune the value by ear.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

_LEXICON_PATH = Path(__file__).resolve().parent / "tts_lexicon.json"

_SEED: dict[str, str] = {
    "pokémon": "poh kee mawn",
    "pokemon": "poh kee mawn",
}

_lock = threading.Lock()
_cache: dict[str, str] = {}
_compiled: "re.Pattern[str] | None" = None
_lookup: dict[str, str] = {}
_mtime: float = -1.0


def _ensure_file() -> None:
    """Create the lexicon file with the seed map if it doesn't exist yet."""
    if not _LEXICON_PATH.exists():
        try:
            _LEXICON_PATH.write_text(
                json.dumps(_SEED, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass


def _compile(entries: dict[str, str]) -> None:
    """Rebuild the matching regex + case-folded lookup from `entries`."""
    global _compiled, _lookup
    _lookup = {k.lower(): v for k, v in entries.items() if k and v}
    if not _lookup:
        _compiled = None
        return
    # Longest keys first so multi-word phrases win over their prefixes.
    keys = sorted(_lookup.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    # \b is Unicode-aware on str patterns, so "pokémon" gets clean boundaries.
    _compiled = re.compile(r"\b(" + alt + r")\b", re.IGNORECASE)


def _refresh() -> None:
    """Load the lexicon from disk if the file changed since last read."""
    global _cache, _mtime
    _ensure_file()
    try:
        m = _LEXICON_PATH.stat().st_mtime
    except OSError:
        m = -1.0
    if m == _mtime and _compiled is not None:
        return
    entries = dict(_SEED)
    try:
        raw = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            entries = {str(k): str(v) for k, v in raw.items()}
    except Exception:
        # Corrupt/missing file → fall back to seed, don't crash the voice path.
        entries = dict(_SEED)
    _cache = entries
    _mtime = m
    _compile(entries)


def apply(text: str) -> str:
    """Return `text` with every lexicon word replaced by its spoken respelling.

    Safe to call on every sentence: cheap, hot-reloads on file change, and a
    no-op (returns input unchanged) when the lexicon is empty or text is falsy.
    Never raises — the voice path must not break on a bad lexicon.
    """
    if not text:
        return text
    try:
        with _lock:
            _refresh()
            pat, look = _compiled, _lookup
        if pat is None:
            return text
        return pat.sub(lambda m: look.get(m.group(0).lower(), m.group(0)), text)
    except Exception:
        return text


def add_entry(written: str, spoken: str) -> dict[str, str]:
    """Add/update a pronunciation override and persist it. Returns the full map."""
    with _lock:
        _refresh()
        entries = dict(_cache)
        entries[written] = spoken
        _LEXICON_PATH.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        global _mtime
        _mtime = -1.0  # force reload on next apply()
    return entries


def entries() -> dict[str, str]:
    """Current lexicon (read-only snapshot)."""
    with _lock:
        _refresh()
        return dict(_cache)


if __name__ == "__main__":
    # Smoke test
    samples = [
        "do you want to play Pokémon or chess?",
        "POKEMON is great",
        "my pokemons are ready",   # 'pokemons' must NOT match (word boundary)
        "",
    ]
    for s in samples:
        print(repr(s), "->", repr(apply(s)))
