"""Brain wiring for jarvis.py — adapter + singleton + command intercept.

Drop this file at C:\\Chloe\\brain_wiring.py. Then jarvis.py needs two edits:

  1. Add at the top with the other imports:
       from brain_wiring import BRAIN, try_handle_brain_command

  2. Inside handle_chat(), right after the _try_handle_remember block
     (search for `ack = _try_handle_remember(_last_user)`), add:

       brain_reply = await asyncio.to_thread(try_handle_brain_command, _last_user)
       if brain_reply is not None:
           _push_history("user", _last_user, modality="chat")
           _push_history("assistant", brain_reply, modality="chat")
           await _ws_send(websocket, {"type": "start"})
           await _ws_send(websocket, {"type": "delta", "text": brain_reply})
           await _ws_send(websocket, {"type": "done"})
           if not data.get("no_tts"):
               hud_server.broadcast_sync("speaking")
               try:
                   await _reply_audio_or_speak(brain_reply, data, label="chat-brain")
               except Exception as e:
                   print(f"[chloe] chat TTS error on brain reply: {e}")
               finally:
                   hud_server.broadcast_sync("idle")
           return

  That's it. The existing _memory system stays as-is — Brain is additive,
  not a replacement.

Why a separate module instead of inlining into jarvis.py:
  - keeps the brain LLM calls stateless (no _voice_history pollution)
  - no tool-calling baggage on brain's heavy synthesis prompts
  - independent client instance — failures here don't break voice path
"""

import os
import re
from pathlib import Path

from brain import Brain
from ollama_keepalive import get_keep_alive as _get_ollama_keep_alive


# ─── Config ─────────────────────────────────────────────────────────────────
# CRITICAL: env vars are read LAZILY at call time, NOT at import time.
# jarvis.py imports brain_wiring before it loads .env via dotenv, so eager
# os.environ.get() at module load reads empty strings. Don't change this
# without re-checking that load order.
BRAIN_ROOT = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")

# _get_groq, MODEL_HEAVY, and the `from groq import Groq` this module used
# to carry were removed 2026-09-01 (dead code removal, stage e) -- Groq is
# fully retired here, _heavy_call and _search_call go straight to Ollama/
# Brave now (see their own docstrings), and nothing in this file called
# _get_groq anymore.


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2:3b").strip()


# ─── LLM adapters ───────────────────────────────────────────────────────────
# 2026-08-31: Groq is fully retired here. MODEL_HEAVY (llama-3.3-70b-
# versatile) already 404s on this account (moved to an enterprise-only
# tier); compound-mini 413s on the free-tier 8000 TPM cap. Both routed
# straight to their Ollama fallback anyway once broken, so _get_groq is
# left defined (dead code, pending a later cleanup pass) but no longer
# called from either adapter below.
def _heavy_call(prompt: str) -> str:
    """One-shot Ollama completion — no history, no tool-calling. For
    ingest, lint, and page synthesis. Groq (llama-3.3-70b-versatile) used
    to be the primary here; it's retired, so this goes straight to
    _light_call now."""
    return _light_call(prompt)


_BRAVE_MAX_QUERY_CHARS = 400  # confirmed live 2026-08-31: HTTP 422 above this
_BRAVE_MAX_QUERY_WORDS = 50   # confirmed live 2026-08-31 (separately, via
                              # _augment_search_query): HTTP 422 above this too


def _no_retrieval_notice(reason: str) -> str:
    """Visible marker prepended to any page body that came from
    _heavy_call (no web search performed), so it can never look
    identical to a properly-researched page. Same transparency rule as
    /wiki_write's citations handling: an honest 'nothing found/nothing
    happened' beats a page that silently reads as grounded when it
    isn't (Ed, 2026-09-01: "a page generated with no retrieval looks
    identical to one generated with it")."""
    return (f"> ⚠ **No web search was performed for this page** ({reason}). "
            f"This content is unverified model output, not grounded in "
            f"retrieved sources — treat any claim or figure here with "
            f"extra skepticism until independently checked.\n\n")


def _search_call(prompt: str, *, topic: str = "") -> dict:
    """Brave search + Ollama synthesis, replacing the old Groq compound-mini
    web-search call. Reuses jarvis.py's shared _brave_search_core -- the
    same retrieval+synthesis engine the voice and chat paths already use --
    instead of a second implementation (Ed, 2026-08-31: "the voice path
    already does this — reuse that code rather than writing a second
    implementation"). Used by /wiki_write to research a topic from scratch,
    and by chloe_jobs.py's daily jobs.

    `topic` is the short natural-language query sent to Brave; `prompt`
    (the full instructions -- structure, style, word count) becomes the
    synthesis step's length/style/structure guidance, never the search
    query itself. `topic` is REQUIRED in practice: falling back to
    `prompt` when the caller forgets it is exactly the 2026-09-01 bug --
    job_daily_topic_rotation and job_daily_critical_thinking_exercise
    both called this with no `topic` kwarg, so `query` silently became
    the full 3000+ char prompt, Brave 422'd on its query-length cap every
    single time, and both jobs had been writing fully ungrounded pages
    since the Groq migration with no visible sign anything was wrong --
    "likely a large share of the fabricated citations we found." Kept as
    a fallback (not required) rather than raising, since a malformed or
    missing topic should degrade to _heavy_call with an explicit notice
    (see below), not crash the job outright.

    Query length is guarded here, not left to Brave's own 422: an
    over-limit query is a CALLER bug, not something to silently swallow.
    Truncated to something searchable, logged loudly (this should be
    visible in every job's log output, not just discoverable by reading
    a stack trace), and still attempted -- better than refusing to search
    at all over a length violation the caller should fix.

    Returns {"text": str, "results": list, "retrieved": bool} -- always
    all three keys. `results` is empty ([]) and `retrieved` is False
    whenever the reply came from _heavy_call (no web search happened,
    nothing to cite) -- in that case `text` is ALSO prefixed with an
    explicit "no web search was performed" notice (_no_retrieval_notice)
    so the page can never silently look identical to a properly-
    researched one. This replaces the old silent fallback, which is what
    let both broken jobs above run for an unknown span with no visible
    failure signal.

    Lazy-imports jarvis to avoid a circular import: jarvis.py imports
    brain_wiring at module load, so importing jarvis back at call time
    (rather than at the top of this file) just returns the
    already-fully-loaded module.
    """
    def _no_search(reason: str) -> dict:
        text = _no_retrieval_notice(reason) + (_heavy_call(prompt) or "")
        return {"text": text, "results": [], "retrieved": False}

    query = (topic or prompt or "").strip()
    if not query:
        return _no_search("no topic or prompt given")

    query_words = query.split()
    if len(query) > _BRAVE_MAX_QUERY_CHARS or len(query_words) > _BRAVE_MAX_QUERY_WORDS:
        truncated = " ".join(query_words[:_BRAVE_MAX_QUERY_WORDS])[:_BRAVE_MAX_QUERY_CHARS]
        print(f"[brain] SEARCH QUERY TOO LONG ({len(query)} chars / "
              f"{len(query_words)} words, Brave's cap is "
              f"{_BRAVE_MAX_QUERY_CHARS} chars / {_BRAVE_MAX_QUERY_WORDS} "
              f"words) -- this is a caller bug (a `topic` wasn't passed, "
              f"or the prompt leaked into the query). Truncating to "
              f"search anyway rather than silently falling back. "
              f"Original: {query[:120]!r}...", flush=True)
        query = truncated
        if not query:
            return _no_search("query was empty after truncation")

    try:
        import jarvis
        result = jarvis._brave_search_core(
            query,
            max_tokens=1800,
            length_instruction=prompt,
            wants_citations=True,
            # /wiki_write's topic is a deliberate standalone research
            # query, not a conversational follow-up -- the short-query
            # augmentation heuristic (<=7 words) otherwise splices in
            # whatever _voice_history happens to contain, corrupting the
            # research on a page that gets written and embedded
            # permanently. Confirmed live 2026-08-31 on topic "silver".
            augment=False,
        )
    except Exception as e:
        print(f"[brain] Brave search core failed: {type(e).__name__}: {e} — "
              f"falling back to heavy (no web search)", flush=True)
        return _no_search(f"search core raised {type(e).__name__}: {e}")
    if not result["retrieved"]:
        print(f"[brain] Brave search (topic={query!r}) found nothing — "
              f"falling back to heavy (no web search)", flush=True)
        return _no_search("Brave returned no results for the search topic")
    if result["synthesis_failed"] or not result["text"]:
        print(f"[brain] Brave search (topic={query!r}) retrieved results "
              f"but synthesis failed — falling back to heavy (no web "
              f"search)", flush=True)
        return _no_search("results were retrieved but synthesis of them failed")
    return {"text": result["text"], "results": result["results"], "retrieved": True}


def _light_call(prompt: str, num_predict: int = 1500) -> str:
    """One-shot Ollama completion. For query selection, fact extraction,
    and any other light synthesis.

    num_predict caps output tokens (default 1500). Background jobs that were
    moved off Groq to preserve quota pass a higher cap (~3500) for long
    syntheses (meta-review, cross-domain, etc.)."""
    # num_ctx — Ollama defaults to 2048 tokens unless set explicitly. Large
    # prompts (cross-domain synthesis, friday meta-review, anything pulling
    # multiple wiki pages) get SILENTLY TRUNCATED at 2048, producing
    # degenerate output that downstream code can't distinguish from a real
    # failure. Default 8192 here (comfortable for qwen2.5:32b on 7900 XTX);
    # override via CHLOE_OLLAMA_CTX for bigger contexts. qwen2.5:32b
    # supports 32k natively; llama3.2:3b supports 128k.
    try:
        num_ctx = int(os.environ.get("CHLOE_OLLAMA_CTX", "8192"))
    except (ValueError, TypeError):
        num_ctx = 8192
    try:
        import requests
        r = requests.post(
            f"{_ollama_url()}/api/chat",
            json={
                "model":      _ollama_model(),
                "messages":   [{"role": "user", "content": prompt}],
                "stream":     False,
                "keep_alive": _get_ollama_keep_alive(),
                "options":  {
                    "temperature": 0.3,
                    "num_predict": num_predict,
                    "num_ctx":     num_ctx,
                },
            },
            # 300s (not 120s) to cover qwen2.5:32b cold-reload (~85s) on top
            # of actual inference time. Hit on 2026-05-11: Groq locked out +
            # qwen unloaded between calls = two-way fallback timed out.
            timeout=300,
        )
        if r.status_code != 200:
            print(f"[brain] light HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return ""
        return (r.json().get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"[brain] light (ollama) failed: {type(e).__name__}: {e}", flush=True)
        return ""


def chloe_llm_call(prompt: str, mode: str) -> str:
    if mode == "heavy":
        return _heavy_call(prompt)
    return _light_call(prompt)


# ─── Singleton ──────────────────────────────────────────────────────────────


def _format_dry_run(r: dict) -> str:
    """Format Brain.ingest(dry_run=True) result as a chat-friendly preview."""
    ent_status = r.get('entities_status', [])
    con_status = r.get('concepts_status', [])
    ent_new = [s for s, st in ent_status if st == 'CREATE']
    ent_upd = [s for s, st in ent_status if st == 'UPDATE']
    con_new = [s for s, st in con_status if st == 'CREATE']
    con_upd = [s for s, st in con_status if st == 'UPDATE']

    out = [f"**DRY RUN** — `{r['slug']}` (nothing written)\n"]
    tldr = r.get('tldr', '')
    if tldr:
        out.append(f"**TLDR:** {tldr}\n")

    if ent_new:
        out.append(f"**Would CREATE {len(ent_new)} entit{'y' if len(ent_new)==1 else 'ies'}:** "
                   + ", ".join(ent_new))
    if ent_upd:
        out.append(f"**Would UPDATE {len(ent_upd)} entit{'y' if len(ent_upd)==1 else 'ies'}:** "
                   + ", ".join(ent_upd))
    if con_new:
        out.append(f"**Would CREATE {len(con_new)} concept{'' if len(con_new)==1 else 's'}:** "
                   + ", ".join(con_new))
    if con_upd:
        out.append(f"**Would UPDATE {len(con_upd)} concept{'' if len(con_upd)==1 else 's'}:** "
                   + ", ".join(con_upd))
    if not (ent_status or con_status):
        out.append("_No entities or concepts would be extracted._")

    key_points = r.get('key_points', [])
    if key_points:
        out.append("\n**Key points** (would land in source page):")
        for p in key_points[:10]:
            out.append(f"  - {p}")

    out.append("\n_Review the list above. If it looks polluted (random people,\n"
               "browser tabs, hallucinated entities), do NOT run without --dry-run._\n"
               "Run `/ingest <filename>` (no flag) to commit.")
    return "\n".join(out)


BRAIN = Brain(root=BRAIN_ROOT, llm_call=chloe_llm_call)



# ============================================================================
# /ingest_screen -- capture screen, save as source, run brain ingest pipeline
# ============================================================================
# Verbose-dump vision prompt: extractor needs substance to find entities.
INGEST_SCREEN_PROMPT = (
    "Transcribe everything visible on this screen as completely as possible. "
    "This is going to be ingested into a knowledge base, so favor fidelity "
    "over summary.\n\n"
    "Include:\n"
    "- The app or website name (and URL if visible) at the top.\n"
    "- All readable text quoted VERBATIM where possible: headings, body text, "
    "error messages, code snippets, button labels, file paths, URLs, "
    "tooltips, status bars.\n"
    "- The structure: which panel/tab/window is active, what sections exist, "
    "which item is selected or focused.\n"
    "- Visible UI elements (toolbars, sidebars, menus, dialogs).\n"
    "- Any tables, lists, or structured data, transcribe their content.\n"
    "Be exhaustive but factual. Do NOT interpret intent or speculate about "
    "purpose. Output as plain markdown with section headings; no preamble."
)


def _validate_slug(slug: str) -> str:
    """Return error string if slug is invalid, empty string if OK."""
    if not slug:
        return "slug is required"
    if any(c in slug for c in ("/", "\\", "..", " ")):
        return "slug must not contain spaces, slashes, or '..'"
    if not all(c.isalnum() or c in "_-" for c in slug):
        return "slug must be snake_case (letters, digits, underscore, dash)"
    if len(slug) > 80:
        return "slug too long (max 80 chars)"
    return ""


def handle_ingest_screen(slug: str, delay: int = 0) -> str:
    """Capture screen, save as markdown source, run brain ingest pipeline.

    Args:
        slug: snake_case identifier; becomes the source filename + wiki title.
        delay: seconds to wait before capture (0-30). Lets the user alt-tab
               to the target window after typing the command.

    On success returns the same shape as /ingest. On failure returns an
    error string.
    """
    import datetime as _dt
    import time as _time

    slug = (slug or "").strip()
    err = _validate_slug(slug)
    if err:
        return (f"`/ingest_screen` failed: {err}\n\n"
                f"Usage: `/ingest_screen [--delay N] <slug>` "
                f"(e.g. `/ingest_screen --delay 5 groq_console_quotas`)")

    # Honor delay before any capture so user can switch focus.
    if delay > 0:
        delay = min(int(delay), 30)
        print(f"[ingest_screen] waiting {delay}s before capture", flush=True)
        _time.sleep(delay)

    # Privacy gates
    if os.environ.get("CHLOE_VISION_DISABLED", "").strip() == "1":
        return ("Vision is disabled (CHLOE_VISION_DISABLED=1). "
                "Unset to capture.")

    try:
        from screen_vision import (
            get_frontmost_app, is_blocked, capture_screen, describe_screen,
        )
    except ImportError as e:
        return f"`/ingest_screen` unavailable: {e}"

    app = get_frontmost_app()
    if app.get("ok") and is_blocked(app):
        return (f"Skipped — frontmost app matches blocklist token "
                f"`{is_blocked(app)}`. Focus a different window and try again.")

    cap = capture_screen()
    if not cap.get("ok"):
        return f"Capture failed: {cap.get('error','?')}"

    desc = describe_screen(cap["png"], prompt=INGEST_SCREEN_PROMPT)
    if not desc.get("ok"):
        return f"Vision call failed: {desc.get('error','?')}"
    body = (desc.get("text") or "").strip()
    if not body:
        return "Vision returned empty description — not saving source."

    # Compose the source file with a metadata header
    a = cap.get("app") or {}
    title = (a.get("title") or "").strip()
    exe = (a.get("exe") or "").strip()
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    mon = cap.get("monitor") or {}
    mon_label = mon.get("name") or f"{mon.get('width','?')}x{mon.get('height','?')}"

    # Title for the wiki source page comes from the slug (Brain.ingest uses
    # src.stem.replace('_',' ').title()), so we don't have to add it ourselves.
    # We DO include human-readable provenance at the top of the source body.
    file_text = (
        f"---\n"
        f"captured_at: {ts}\n"
        f"capture_app_title: {title}\n"
        f"capture_app_exe: {exe}\n"
        f"capture_monitor: {mon_label}\n"
        f"capture_method: /ingest_screen\n"
        f"---\n\n"
        f"# {slug.replace('_',' ').title()}\n\n"
        f"_Captured {ts} from {exe or title or 'screen'}._\n\n"
        f"---\n\n"
        f"{body}\n"
    )

    raw_path = BRAIN.raw_dir / f"{slug}.md"
    overwrote = raw_path.exists()
    try:
        raw_path.write_text(file_text, encoding="utf-8")
    except Exception as e:
        return f"Failed to write source file: {type(e).__name__}: {e}"

    # Run the existing ingest pipeline
    try:
        r = BRAIN.ingest(f"{slug}.md")
    except Exception as e:
        import traceback; traceback.print_exc()
        return (f"Source saved to `{raw_path}` "
                f"({len(file_text)} bytes) but ingest failed: "
                f"{type(e).__name__}: {e}\n\n"
                f"You can retry with `/ingest {slug}.md` once fixed.")

    over_note = " (overwrote existing)" if overwrote else ""
    return (f"Captured + ingested **{r['slug']}**{over_note}.\n\n"
            f"  **TLDR:** {r['tldr']}\n"
            f"  **Source:** `{raw_path.name}` ({len(file_text)} bytes)\n"
            f"  **Touched:** {len(r['entities_touched'])} entities, "
            f"{len(r['concepts_touched'])} concepts.")



# ============================================================================
# /wiki_write -- topic-to-page autonomous research and write
# ============================================================================
# Take a free-form topic, research via Brave search + Ollama synthesis
# (_search_call), write a comprehensive markdown article into raw/, then run
# BRAIN.ingest to extract entities and concepts as usual. The user's input is
# just a topic string - everything else is autonomous.
#
# Pairs with the existing wiki creation paths:
#   /add           - you write the body, Chloe files it
#   /ingest        - Chloe reads a file you prepared
#   /ingest_screen - Chloe captures + transcribes your current screen
#   /wiki_write    - Chloe researches the topic herself (this command)

WIKI_WRITE_PROMPT = (
    "Research the topic \"{topic}\" using web search where helpful. Write a "
    "comprehensive but focused markdown wiki page about it.\n\n"
    "Open with a 1-2 sentence TL;DR that captures the essence.\n\n"
    "Use exactly these six section headings, in this order, copied "
    "character-for-character with nothing added to them (no parenthetical, "
    "no extra words, no punctuation beyond what's shown) -- skip the "
    "'Current status' one per the note below, but never add words to any "
    "of the other five:\n"
    "## Background or context\n"
    "## Key concepts or mechanics\n"
    "## Applications or examples\n"
    "## Current status\n"
    "## Related ideas\n"
    "## Caveats / limitations\n\n"
    "What belongs under each of those six headings (this paragraph is "
    "instructions for you, not text to include anywhere in your output):\n"
    "Background or context is what the topic is and where it comes from. "
    "Key concepts or mechanics is how it works and its main components. "
    "Applications or examples is when and where it's used. Current status "
    "is the ONE exception to 'use all six headings' -- include it, with "
    "its heading copied exactly as shown above, ONLY if the search results "
    "contain current or dated data: a price, a statistic, a recent "
    "development, a specific figure with a date attached. If the results "
    "didn't have anything current to report, skip this heading entirely -- "
    "do not write it and pad it with filler, and do not skip it just "
    "because current data doesn't fit neatly under one of the other "
    "headings. Related ideas is other concepts a reader interested in this "
    "topic would care about. Caveats or limitations covers those, if any "
    "apply.\n\n"
    "Style:\n"
    "- Plain markdown only. Use ## for section headings.\n"
    "- Be specific and fact-grounded. Concrete examples beat abstract description.\n"
    "- If the search results include a specific current number (a price, a "
    "rate, a statistic), use it -- don't default to generic textbook "
    "description when the results gave you something concrete and current.\n"
    "- Plain English. No filler ('In conclusion', 'It is important to note').\n"
    "- Aim for 400-800 words depending on topic depth. Don't pad.\n"
    "- Do NOT add a top-level # heading - the wiki ingest pipeline creates the "
    "title from the slug.\n"
    "- Do NOT write a sources, citations, or references section, and do NOT "
    "invent, guess at, or format any URL yourself -- that section is added "
    "automatically from the actual search results after your output. "
    "Anything you write under a citations-like heading would be fabricated, "
    "since you don't have the real source list.\n"
    "- Start your output directly with the TL;DR paragraph."
)


# Stopwords dropped during slugification — pad without adding distinctness.
# Added 2026-05-17 to fix the truncation pattern that produced slugs like
# `..._the_wheel_p` from "covered call options strategy mechanics when to
# use tax implications the wheel pattern".
_SLUG_STOPWORDS = frozenset({
    # Articles + common connectives
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
    "with", "by", "from", "into", "as", "vs", "via",
    # Question words
    "when", "where", "what", "who", "why", "how", "which",
    # Generic copula
    "is", "are", "was", "were", "be", "been", "being",
    # Topic-tag fluff that doesn't disambiguate a topic
    "use", "uses", "using", "used", "usage",
    "mechanics", "basic", "basics", "intro", "introduction",
    "guide", "overview", "explanation", "explained", "explain",
    "implications", "things", "stuff", "etc",
    # Cross-reference fluff
    "vs.", "versus", "compared",
})


def _slugify_topic(topic: str) -> str:
    """Convert a free-form topic into a slug-validator-safe filename.

    Examples:
        "Kelly criterion"        -> "kelly_criterion"
        "Lebron James (NBA)"     -> "lebron_james_nba"
        "  Polyphasic Sleep!! "  -> "polyphasic_sleep"
        "x86-64 assembly"        -> "x86-64_assembly"
        "covered call options strategy — mechanics, when to use, tax
            implications, the wheel pattern"
            -> "covered_call_options_strategy_tax_wheel_pattern"

    Slug must satisfy _validate_slug: alnum + underscore + dash, <= 80
    chars, no spaces/slashes/dots. Drops stopwords + truncates at token
    boundary when capped (rather than mid-word).
    """
    import re as _re
    # Lowercase + collapse non-alnum-non-dash to single space (so we can
    # tokenize). Keep dashes as part of tokens — they're valid in slugs.
    s = topic.lower().strip()
    s = _re.sub(r"[^a-z0-9-]+", " ", s).strip()
    tokens = [t for t in s.split() if t and t not in _SLUG_STOPWORDS]
    if not tokens:
        # Fallback: nothing left after stopword filter. Use the legacy
        # path so we never return empty.
        legacy = _re.sub(r"[^a-z0-9-]+", "_", topic.lower().strip())
        legacy = _re.sub(r"_+", "_", legacy).strip("_")
        return legacy[:80].rstrip("_")
    # Greedy fit: accumulate tokens until adding the next would exceed 80.
    out = []
    used = 0
    for t in tokens:
        extra = len(t) + (1 if out else 0)  # +1 for joining underscore
        if used + extra > 80:
            break
        out.append(t)
        used += extra
    if not out:
        # First token alone exceeds 80 — truncate it.
        return tokens[0][:80].rstrip("_-")
    return "_".join(out)


WIKI_SYNTH_PROMPT = (
    "You are Chloe, synthesizing a wiki page about \"{topic}\" using "
    "Edward's existing brain as source material. NO web search — only "
    "the pages quoted below. Write a comprehensive but focused markdown "
    "wiki page.\n\n"
    "Required structure:\n"
    "- Open with a 1-2 sentence TL;DR that captures the essence.\n"
    "- ## What the brain already covers (synthesize across the cited pages)\n"
    "- ## Key concepts or mechanics (how the pieces fit together)\n"
    "- ## Connections (which existing entities/concepts link to this — "
    "cite as [[page_name]])\n"
    "- ## Gaps the brain doesn't cover (be explicit — what's missing from "
    "Edward's existing material that a `/wiki_write` web-research pass "
    "would need to fill)\n\n"
    "Style:\n"
    "- Plain markdown only. Use ## for section headings.\n"
    "- Cite source pages inline as [[page_name]] (use the path stems "
    "below; don't invent links). If a claim isn't in the brain, don't "
    "make it — surface it as a gap instead.\n"
    "- Plain English. No filler.\n"
    "- 300-700 words. Don't pad to hit a target.\n"
    "- Do NOT add a top-level # heading - the ingest pipeline writes the "
    "title from the slug.\n"
    "- Start your output directly with the TL;DR paragraph.\n\n"
    "--- RELEVANT BRAIN PAGES ---\n"
    "{relevant_pages}\n"
    "--- END BRAIN PAGES ---\n"
)


# ─── /wiki_interview: Q&A → page ─────────────────────────────────────────────
#
# Two-slash workflow:
#   /wiki_interview <topic>      → Chloe generates 5 questions, returns
#                                  them as a numbered list, stashes
#                                  state in a module-level dict.
#   <user answers all 5 in one chat turn>
#   /wiki_interview_done         → reads the most-recent non-slash user
#                                  turn from chloe_memory, synthesizes
#                                  a wiki page from Q + A pairs, ingests.
#
# Single-user single-interview state — no concurrency concern (Ed is the
# only user; only one interview can be open at a time).

_INTERVIEW_STATE: dict = {}  # {"topic": str, "questions": [str], "started_at": float}


_INTERVIEW_QUESTIONS_PROMPT = (
    "You are Chloe, interviewing Edward to fill a wiki page on the topic "
    "\"{topic}\". Generate exactly 5 questions that, if Edward answered "
    "them thoughtfully, would produce a durable reference page from his "
    "perspective.\n\n"
    "Question shape rules:\n"
    "- Open-ended (no yes/no questions).\n"
    "- Mix of factual ('what does X mean to you?'), experiential ('when "
    "have you used X?'), and opinion ('what gets X wrong in most "
    "explanations?').\n"
    "- Specific, not abstract. Avoid 'what is X' style — assume the "
    "wiki framework already captures definitions; you're after Edward's "
    "lived take.\n"
    "- Each question stands alone — Edward will answer all 5 in one "
    "message.\n\n"
    "Output ONLY the 5 questions as a numbered list (`1. ...` through "
    "`5. ...`). No preamble, no commentary."
)


_INTERVIEW_SYNTH_PROMPT = (
    "You are Chloe, synthesizing a wiki page from a 5-question interview "
    "with Edward on the topic \"{topic}\". Write a markdown wiki page that "
    "treats Edward's answers as the primary source. Quote him sparingly; "
    "synthesize into flowing prose.\n\n"
    "Structure:\n"
    "- TL;DR in 1-2 sentences\n"
    "- ## Edward's take (3-5 short paragraphs synthesizing answers)\n"
    "- ## Concrete examples (specific stories/instances Edward gave)\n"
    "- ## Where the brain should grow this (gaps Edward's answers imply — "
    "what would round out a fuller picture)\n\n"
    "Style:\n"
    "- Plain markdown, ## section headers, no # title (ingest pipeline "
    "writes the title from the slug).\n"
    "- Do not invent facts Edward didn't give. If an answer was thin, "
    "say so rather than padding.\n"
    "- Plain English. No filler.\n"
    "- 300-700 words.\n\n"
    "--- INTERVIEW ---\n{qa_pairs}\n--- END INTERVIEW ---"
)


def handle_wiki_interview(topic: str) -> str:
    """Start a Q&A → wiki interview. Generates 5 questions, returns them.

    The user replies to all 5 in a single chat turn, then runs
    ``/wiki_interview_done`` to synthesize the page.
    """
    import time as _time
    topic = (topic or "").strip().strip('"').strip("'")
    if not topic:
        return ("Usage: `/wiki_interview <topic>`\n\n"
                "Chloe will ask you 5 questions about the topic. Answer "
                "all 5 in your next chat message, then run "
                "`/wiki_interview_done` to write the wiki page from your "
                "answers.")
    if len(topic) > 200:
        return f"Topic too long ({len(topic)} chars; max 200)."

    slug = _slugify_topic(topic)
    err = _validate_slug(slug)
    if err:
        return f"Couldn't make a valid slug from `{topic}`: {err}"

    print(f"[wiki_interview] generating questions for: {topic!r}",
          flush=True)
    raw = _heavy_call(_INTERVIEW_QUESTIONS_PROMPT.format(topic=topic))
    if not raw or not raw.strip():
        return ("LLM returned no questions. Try again, or rephrase the "
                "topic.")

    # Parse numbered list. Accept "1. q", "1) q", "1: q".
    questions: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^\s*(\d+)[.)\]:]?\s+(.+?)\s*$", line)
        if m:
            qtext = m.group(2).strip()
            if qtext:
                questions.append(qtext)
    if len(questions) < 5:
        return (f"Could only parse {len(questions)} questions from the "
                f"LLM output. Try again. Raw output was:\n\n```\n{raw[:600]}"
                f"\n```")
    questions = questions[:5]

    _INTERVIEW_STATE.clear()
    _INTERVIEW_STATE.update({
        "topic": topic,
        "slug": slug,
        "questions": questions,
        "started_at": _time.time(),
    })

    qlist = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return (f"**Interview started: _{topic}_**\n\n"
            f"Answer all 5 below in your **next message** (one combined "
            f"reply is fine — number your answers if you can). Then run "
            f"`/wiki_interview_done` to write the wiki page.\n\n"
            f"{qlist}")


def handle_wiki_interview_done() -> str:
    """Finalize the active interview: pull answers from chloe_memory,
    synthesize page, ingest. Clears state on success."""
    import datetime as _dt
    if not _INTERVIEW_STATE:
        return ("No active interview. Start one with "
                "`/wiki_interview <topic>`.")

    topic = _INTERVIEW_STATE["topic"]
    slug = _INTERVIEW_STATE["slug"]
    questions = _INTERVIEW_STATE["questions"]
    started_at = _INTERVIEW_STATE.get("started_at", 0)

    # Find the most-recent USER turn AFTER the interview started, that
    # isn't itself a slash command. That's the answer payload.
    try:
        from jarvis import _memory  # type: ignore
    except Exception as e:
        return f"/wiki_interview_done: memory unavailable ({e})"
    if _memory is None:
        return "/wiki_interview_done: memory not initialized."

    answer_text = ""
    for row in reversed(_memory.recent_turns(n=20)):
        if row["role"] != "user":
            continue
        if row["ts"] < started_at:
            break
        content = (row["content"] or "").strip()
        if not content or content.startswith("/"):
            continue
        answer_text = content
        break

    if not answer_text:
        return ("Couldn't find an answer message after the interview "
                "started. Type your combined answers as your next chat "
                "message, then re-run `/wiki_interview_done`.")

    qa_pairs = (
        "\n\n".join(f"Q{i+1}. {q}" for i, q in enumerate(questions))
        + "\n\n--- Edward's combined answers ---\n"
        + answer_text
    )

    print(f"[wiki_interview] synthesizing page for {topic!r}", flush=True)
    body = _heavy_call(_INTERVIEW_SYNTH_PROMPT.format(
        topic=topic, qa_pairs=qa_pairs))
    if not body or not body.strip():
        return ("Synthesis returned empty. State preserved — try "
                "`/wiki_interview_done` again, or `/wiki_interview "
                "<topic>` to start fresh.")

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    raw_path = BRAIN.raw_dir / f"{slug}.md"
    file_text = (
        f"---\n"
        f"generated_at: {ts}\n"
        f"generated_via: /wiki_interview\n"
        f"requested_topic: {topic}\n"
        f"interview_questions: {len(questions)}\n"
        f"---\n\n"
        f"# {slug.replace('_', ' ').title()}\n\n"
        f"_Synthesized by Chloe from a 5-question interview with Edward "
        f"on {ts}._\n\n"
        f"---\n\n"
        f"{body.strip()}\n"
    )

    overwrote = raw_path.exists()
    try:
        raw_path.write_text(file_text, encoding="utf-8")
    except Exception as e:
        return f"Failed to write source file: {type(e).__name__}: {e}"

    try:
        r = BRAIN.ingest(f"{slug}.md")
    except Exception as e:
        import traceback; traceback.print_exc()
        return (f"Source saved but ingest failed: {type(e).__name__}: {e}\n"
                f"Retry with `/ingest {slug}.md`.")

    _INTERVIEW_STATE.clear()
    over_note = " (overwrote existing source)" if overwrote else ""
    return (f"Interview synthesized + ingested **{r['slug']}**{over_note}.\n\n"
            f"  **Topic:** {topic}\n"
            f"  **TLDR:** {r['tldr']}\n"
            f"  **Source:** `{raw_path.name}` ({len(file_text)} bytes)\n"
            f"  **Touched:** {len(r['entities_touched'])} entities, "
            f"{len(r['concepts_touched'])} concepts.")


def handle_wiki_synth(topic: str, dry_run: bool = False) -> str:
    """Topic-to-page pipeline using ONLY the existing brain — no web.

    Companion to /wiki_write: when Ed wants a recap-style page that
    pulls his existing notes into a single synthesized view (no fresh
    facts from the web), this is the slash. Useful for "what does the
    brain already know about <topic>" deliverables that should land as
    durable wiki pages rather than chat answers.

    Pipeline:
    1. Validate topic, generate slug (same _slugify_topic as /wiki_write).
    2. Pull relevant brain pages via keyword-select + LLM page-pick
       fallback (same as queue_processor.gather_relevant_pages).
    3. Call _heavy_call with WIKI_SYNTH_PROMPT (no web search — heavy
       path stays on Groq llama-3.3-70b or Ollama qwen).
    4. Wrap result with provenance frontmatter.
    5. Write to BRAIN.raw_dir.
    6. Run BRAIN.ingest to extract entities/concepts + auto-link.
    """
    import datetime as _dt
    topic = (topic or "").strip().strip('"').strip("'")
    if not topic:
        return ("Usage: `/wiki_synth [--dry-run] <topic>`\n"
                "Example: `/wiki_synth covered call mechanics`\n\n"
                "Chloe will synthesize a page from existing brain material "
                "(no web search). For fresh web research, use `/wiki_write`.")

    if len(topic) > 200:
        return (f"Topic too long ({len(topic)} chars; max 200). "
                f"Try something more focused.")

    slug = _slugify_topic(topic)
    err = _validate_slug(slug)
    if err:
        return (f"Couldn't make a valid slug from `{topic}`: {err}\n\n"
                f"Try a topic with letters, digits, spaces, or dashes.")

    raw_path = BRAIN.raw_dir / f"{slug}.md"

    # Gather relevant pages from the existing brain.
    try:
        from queue_processor import gather_relevant_pages
        relevant_pages = gather_relevant_pages(BRAIN, topic, max_pages=8)
    except Exception as e:
        return (f"/wiki_synth: couldn't gather brain context "
                f"({type(e).__name__}: {e}). Brain may be empty or the "
                f"wiki index is missing.")

    if dry_run:
        page_chars = len(relevant_pages)
        return (f"**DRY RUN** - nothing written.\n\n"
                f"  **Topic:** {topic}\n"
                f"  **Slug:** {slug}\n"
                f"  **Would write to:** `{raw_path}`\n"
                f"  **Brain context gathered:** {page_chars} chars\n"
                f"  **Would call:** `_heavy_call(<WIKI_SYNTH_PROMPT for "
                f"{topic!r}>)` then `BRAIN.ingest('{slug}.md')`\n\n"
                f"Run `/wiki_synth {topic}` (without --dry-run) to commit.")

    print(f"[wiki_synth] synthesizing: {topic!r} (slug={slug})", flush=True)
    body = _heavy_call(WIKI_SYNTH_PROMPT.format(
        topic=topic, relevant_pages=relevant_pages))

    if not body or not body.strip():
        return (f"Synthesis returned empty content for `{topic}`. The brain "
                f"may have nothing on this topic — try `/wiki_write {topic}` "
                f"to pull fresh content from the web instead.")

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    file_text = (
        f"---\n"
        f"generated_at: {ts}\n"
        f"generated_via: /wiki_synth\n"
        f"requested_topic: {topic}\n"
        f"---\n\n"
        f"# {slug.replace('_', ' ').title()}\n\n"
        f"_Synthesized by Chloe from existing brain pages on {ts}._\n"
        f"_Original topic input: {topic!r}._\n\n"
        f"---\n\n"
        f"{body.strip()}\n"
    )

    overwrote = raw_path.exists()
    try:
        raw_path.write_text(file_text, encoding="utf-8")
    except Exception as e:
        return f"Failed to write source file: {type(e).__name__}: {e}"

    try:
        r = BRAIN.ingest(f"{slug}.md")
    except Exception as e:
        import traceback; traceback.print_exc()
        return (f"Source saved to `{raw_path}` ({len(file_text)} bytes) but "
                f"ingest failed: {type(e).__name__}: {e}\n\n"
                f"Retry with `/ingest {slug}.md` once the underlying issue "
                f"is resolved.")

    over_note = " (overwrote existing source)" if overwrote else ""
    return (f"Synthesized + ingested **{r['slug']}**{over_note}.\n\n"
            f"  **Topic:** {topic}\n"
            f"  **TLDR:** {r['tldr']}\n"
            f"  **Source:** `{raw_path.name}` ({len(file_text)} bytes)\n"
            f"  **Touched:** {len(r['entities_touched'])} entities, "
            f"{len(r['concepts_touched'])} concepts.")


def handle_wiki_write(topic: str, dry_run: bool = False) -> str:
    """Topic-to-page pipeline.

    1. Validate topic, generate slug.
    2. Call _search_call with WIKI_WRITE_PROMPT (Brave search + Ollama
       synthesis) -- returns {"text", "results"}.
    3. Wrap the text with provenance frontmatter + a real "## Citations"
       section built from `results` (never the model's own output).
    4. Write to BRAIN.raw_dir.
    5. Run BRAIN.ingest to extract entities/concepts.
    6. Return formatted status (same shape as /ingest_screen).
    """
    import datetime as _dt
    topic = (topic or "").strip().strip('"').strip("'")
    if not topic:
        return ("Usage: `/wiki_write [--dry-run] <topic>`\n"
                "Example: `/wiki_write Kelly criterion`\n\n"
                "Chloe will research the topic via web search, write a wiki "
                "page, and ingest it into the entity/concept extraction "
                "pipeline.")

    if len(topic) > 200:
        return (f"Topic too long ({len(topic)} chars; max 200). "
                f"Try something more focused.")

    slug = _slugify_topic(topic)
    err = _validate_slug(slug)
    if err:
        return (f"Couldn't make a valid slug from `{topic}`: {err}\n\n"
                f"Try a topic with letters, digits, spaces, or dashes.")

    raw_path = BRAIN.raw_dir / f"{slug}.md"

    if dry_run:
        return (f"**DRY RUN** - nothing written.\n\n"
                f"  **Topic:** {topic}\n"
                f"  **Slug:** {slug}\n"
                f"  **Would write to:** `{raw_path}`\n"
                f"  **Would call:** `_search_call(<WIKI_WRITE_PROMPT for "
                f"{topic!r}>)` then `BRAIN.ingest('{slug}.md')`\n\n"
                f"Run `/wiki_write {topic}` (without --dry-run) to commit.")

    print(f"[wiki_write] researching: {topic!r} (slug={slug})", flush=True)
    search_result = _search_call(WIKI_WRITE_PROMPT.format(topic=topic), topic=topic)
    body = search_result.get("text", "")
    results = search_result.get("results") or []

    if not body or not body.strip():
        return (f"Research returned empty content for `{topic}`. Possible "
                f"causes: Brave found no results, Ollama synthesis timed "
                f"out, or the local model declined the topic. "
                f"Nothing was written. Try again or use `/ingest_screen` "
                f"with the topic's reference page open.")

    # Real citations from the actual Brave results, same treatment
    # jarvis.py's _persist_brave_to_wiki gives voice/chat search pages.
    # NOT the model's own output -- WIKI_WRITE_PROMPT now explicitly
    # forbids it from writing a sources/citations section itself, because
    # when it was left to do that (2026-08-31, confirmed live on topic
    # "silver") it fabricated one: a self-referential [[wikilink]] to its
    # own slug and an invented `/wiki_write/silver` URL that was never a
    # real source. Same fabrication class the Brave migration was meant
    # to eliminate, just moved from inline [N] markers to frontmatter.
    cites = []
    urls = []
    for i, res in enumerate(results, 1):
        title = (res.get("title") or "").strip().replace("\n", " ")
        url = (res.get("url") or "").strip()
        domain = (res.get("domain") or "").strip()
        if title and url:
            cites.append(f"{i}. [{title}]({url}) — {domain}")
        elif url:
            cites.append(f"{i}. {url}")
        if url:
            urls.append(f"  - {url}")
    urls_block = "\n".join(urls) if urls else "  []"
    cites_block = "\n".join(cites) if cites else "_(no citations returned)_"

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    file_text = (
        f"---\n"
        f"generated_at: {ts}\n"
        f"generated_via: /wiki_write\n"
        f"requested_topic: {topic}\n"
        f"source_urls:\n{urls_block}\n"
        f"---\n\n"
        f"# {slug.replace('_', ' ').title()}\n\n"
        f"_Generated by Chloe via web research on {ts}._\n"
        f"_Original topic input: {topic!r}._\n\n"
        f"---\n\n"
        f"{body.strip()}\n\n"
        f"## Citations\n\n"
        f"{cites_block}\n"
    )

    overwrote = raw_path.exists()
    try:
        raw_path.write_text(file_text, encoding="utf-8")
    except Exception as e:
        return f"Failed to write source file: {type(e).__name__}: {e}"

    try:
        r = BRAIN.ingest(f"{slug}.md")
    except Exception as e:
        import traceback; traceback.print_exc()
        return (f"Source saved to `{raw_path}` ({len(file_text)} bytes) but "
                f"ingest failed: {type(e).__name__}: {e}\n\n"
                f"Retry with `/ingest {slug}.md` once the underlying issue "
                f"is resolved.")

    over_note = " (overwrote existing source)" if overwrote else ""
    return (f"Wrote + ingested **{r['slug']}**{over_note}.\n\n"
            f"  **Topic:** {topic}\n"
            f"  **TLDR:** {r['tldr']}\n"
            f"  **Source:** `{raw_path.name}` ({len(file_text)} bytes)\n"
            f"  **Touched:** {len(r['entities_touched'])} entities, "
            f"{len(r['concepts_touched'])} concepts.")


# ============================================================================
# Auto-fact extraction (fires on every non-command chat message)
# ============================================================================
# Goal: when Edward says "I work at Amazon DSP logistics" in normal chat, save
# that as a durable fact without him needing to type /fact. Reuses the same
# fact_extract_and_add pipeline the /fact command uses.
#
# Cheap filters skip the LLM call entirely on messages that obviously aren't
# facts (commands, questions, very short utterances). Anything that passes
# the filter goes through a thread so the chat reply path isn't blocked.
#
# Disable: CHLOE_AUTO_FACT=0 in .env.

import threading as _threading
import time as _time

# Concurrency cap: at most one extraction in flight at a time. Prevents
# pile-up if the user types a flurry of messages.
_AUTO_FACT_RUNNING = _threading.Lock()

# Real liveness signal for brain_http.py's health_full (2026-09-03): was
# hardcoded to None there ("not currently tracked") -- a permanent null
# regardless of whether the pipeline was ever actually running, so it
# could never have caught the voice-path gap this same audit found. Set
# every time _auto_fact_worker actually executes (an attempt, not just a
# successful save -- "did the pipeline run" is the liveness question,
# not "did it find a fact this particular time").
_auto_fact_last_run_ts: float | None = None


def get_auto_fact_last_run_ts() -> float | None:
    """Unix epoch seconds of the last auto-fact extraction attempt, or
    None if it has never run in this process."""
    return _auto_fact_last_run_ts

# Patterns that strongly imply "this is a question or command, not a fact"
# and aren't worth burning an LLM call on.
_QUESTION_PREFIXES = (
    "what ", "who ", "where ", "when ", "why ", "how ", "is ", "are ", "do ",
    "does ", "did ", "can ", "could ", "would ", "will ", "should ",
    "play ", "open ", "show ", "remind ", "set ", "stop ", "pause ",
    "search ", "look up ", "tell me ",
)


def _should_skip_auto_fact(msg: str) -> bool:
    """Cheap pre-LLM filter. Return True to skip extraction."""
    if os.environ.get("CHLOE_AUTO_FACT", "1").strip() == "0":
        return True
    s = (msg or "").strip()
    if not s:
        return True
    if s.startswith("/"):
        return True  # slash commands are handled separately
    if s.startswith(("(", "[", "*")):
        return True  # quoted/parenthetical asides
    if len(s) < 20:
        return True  # too short to carry a durable fact
    if "?" in s and len(s) < 80:
        return True  # short questions
    low = s.lower().lstrip()
    if any(low.startswith(p) for p in _QUESTION_PREFIXES):
        return True
    return False


def _auto_fact_worker(msg: str):
    """Run on a daemon thread. Extract and save fact if present, else exit."""
    global _auto_fact_last_run_ts
    if not _AUTO_FACT_RUNNING.acquire(blocking=False):
        return  # another extraction already in flight; drop this one
    try:
        _auto_fact_last_run_ts = _time.time()
        try:
            slug = BRAIN.fact_extract_and_add(msg)
            if slug:
                print(f"[auto-fact] saved: {slug}", flush=True)
        except Exception as e:
            print(f"[auto-fact] extract failed: {type(e).__name__}: {e}",
                  flush=True)
    finally:
        _AUTO_FACT_RUNNING.release()


def maybe_auto_extract(msg: str) -> None:
    """Fire-and-forget background extractor. Always returns None.

    Public (2026-09-03, was _maybe_auto_extract): this used to be called
    only from try_handle_brain_command below, i.e. only reachable via
    jarvis.py's CHAT text path. Voice (PTT + wake-word) never routed
    through here at all, so auto-fact-extraction silently never ran for
    anything Ed only ever said out loud -- confirmed live: facts/ had
    gone stale for weeks despite normal voice usage, while
    fact_extract_and_add itself worked fine when called directly. Now
    also called directly from jarvis.py's two voice-turn handlers, same
    position in the dispatch order as here (after the explicit
    "remember: <fact>" short-circuit, before any intent dispatcher that
    might claim and return early) so voice and chat behave identically."""
    if _should_skip_auto_fact(msg):
        return
    t = _threading.Thread(target=_auto_fact_worker, args=(msg,),
                          name="chloe-auto-fact", daemon=True)
    t.start()


# ─── /summarize_old auto-cadence ───────────────────────────────────────────
#
# Pillar 4 follow-up (from chloe_handoff 2026-05-17): wrap the existing
# /summarize_old slash in a daemon thread so memory rollup happens on its
# own once unsummarized turn count climbs past CHLOE_SUMMARIZE_THRESHOLD.
# Opt-in via CHLOE_SUMMARIZE_AUTO=1 — defaults off until soaked.

_SUMMARIZE_AUTO_STARTED = False
_SUMMARIZE_AUTO_LOCK = _threading.Lock()


def _summarize_autopilot_worker():
    """Background loop. Sleeps an hour between checks, fires handle_summarize_old
    when the unsummarized-turn count crosses the threshold."""
    import time as _time
    import random as _random
    # Initial settle delay so we don't fire during boot before _memory is up.
    _time.sleep(120)
    while True:
        try:
            # _memory lives in jarvis; lazy-import so brain_wiring stays
            # importable for tests that don't boot the full stack.
            try:
                from jarvis import _memory  # type: ignore
            except Exception:
                _memory = None  # noqa: F841
            if _memory is None:
                _time.sleep(60)
                continue
            try:
                threshold = int(os.environ.get(
                    "CHLOE_SUMMARIZE_THRESHOLD", "50"))
            except ValueError:
                threshold = 50
            unsummed = _memory.unsummarized_count()
            if unsummed >= threshold:
                # Single in-flight guard so an overrun summarize call can't
                # stack up if a previous one is still inside Groq.
                if _SUMMARIZE_AUTO_LOCK.acquire(blocking=False):
                    try:
                        print(f"[summarize-auto] {unsummed} unsummarized "
                              f"(threshold {threshold}); rolling up.")
                        reply = handle_summarize_old("")
                        first_line = reply.splitlines()[0] if reply else "(empty)"
                        print(f"[summarize-auto] {first_line}")
                    except Exception as e:
                        print(f"[summarize-auto] rollup failed: "
                              f"{type(e).__name__}: {e}")
                    finally:
                        _SUMMARIZE_AUTO_LOCK.release()
        except Exception as e:
            print(f"[summarize-auto] loop error: {type(e).__name__}: {e}")
        # 60 min ± up to 5 min jitter so we don't hit the wire at :00 every hour.
        _time.sleep(3600 + _random.randint(-300, 300))


def maybe_start_summarize_autopilot() -> bool:
    """Spawn the autopilot daemon if CHLOE_SUMMARIZE_AUTO=1. Idempotent.

    Called from jarvis.py near other boot daemons (warm_ollama, wiki_watcher).
    Returns True if started this call, False otherwise.
    """
    global _SUMMARIZE_AUTO_STARTED
    if _SUMMARIZE_AUTO_STARTED:
        return False
    if os.environ.get("CHLOE_SUMMARIZE_AUTO", "0").strip() != "1":
        return False
    t = _threading.Thread(target=_summarize_autopilot_worker,
                          name="chloe-summarize-auto", daemon=True)
    t.start()
    _SUMMARIZE_AUTO_STARTED = True
    print("[summarize-auto] daemon started (CHLOE_SUMMARIZE_AUTO=1)")
    return True


# ─── Natural-language aliases for slash commands ───────────────────────────

# Anchors for "/wiki_write <topic>". Match shapes like:
#   "write a wiki about <topic>"
#   "write a wiki page on <topic>"
#   "research <topic> for the wiki"
#   "make a wiki entry about <topic>"
#   "do a wiki write on <topic>"
_WIKI_WRITE_NL_RE = re.compile(
    r"""
    ^\s*
    (?:please\s+)?
    (?:
        (?:write|make|create|add|do|draft)\s+
        (?:a\s+|an\s+|the\s+)?
        (?:new\s+)?
        wiki\s+
        (?:page\s+|entry\s+|note\s+|article\s+|writeup\s+|write[-\s]?up\s+)?
        (?:about|on|for|covering|of|re)\s+
        (?P<topic>.+?)
      |
        research\s+(?P<topic2>.+?)\s+
        (?:for|into|to\s+add\s+to)\s+
        (?:the\s+)?wiki
    )
    \s*[.?!]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Anchors for "/wiki <query>" (semantic search). Match shapes like:
#   "search the wiki for <query>"
#   "look up <query> in the wiki"
#   "what's in the wiki about <query>"
#   "find <query> in the wiki"
_WIKI_SEARCH_NL_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:search|find|look\s*up|query)\s+
        (?:the\s+)?wiki\s+(?:for|about|on)\s+
        (?P<q>.+?)
      |
        look\s*up\s+(?P<q2>.+?)\s+in\s+(?:the\s+)?wiki
      |
        find\s+(?P<q3>.+?)\s+in\s+(?:the\s+)?wiki
      |
        (?:what(?:'s|\s+is)?\s+in\s+the\s+wiki\s+about|
            what(?:'s|\s+does)?\s+the\s+wiki\s+(?:say|have)\s+(?:about|on))\s+
        (?P<q4>.+?)
    )
    \s*[.?!]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _maybe_wiki_nl_alias(msg: str) -> "str | None":
    """Translate natural-language wiki phrasing to the slash form.

    Returns the rewritten slash command, or ``None`` if no alias matched.
    Already-slash messages pass through untouched. Empty match groups are
    skipped so the regex stays one expression with alternatives.
    """
    if not msg or msg.startswith("/"):
        return None
    if len(msg) > 400:
        # NL aliases are short user phrasings; don't run regex over essays.
        return None
    m = _WIKI_WRITE_NL_RE.match(msg)
    if m:
        topic = (m.group("topic") or m.group("topic2") or "").strip(" \t.\"'")
        if topic:
            return f"/wiki_write {topic}"
    m = _WIKI_SEARCH_NL_RE.match(msg)
    if m:
        q = (m.group("q") or m.group("q2") or m.group("q3")
             or m.group("q4") or "").strip(" \t.\"'")
        if q:
            return f"/wiki {q}"
    return None


# ─── Command intercept ──────────────────────────────────────────────────────

# ============================================================================
# /ask -- vision-augmented query
# ============================================================================
ASK_DESCRIPTION_PROMPT = (
    "Describe what is on this screen factually and specifically. "
    "Identify the app or website. Quote any visible text that is likely "
    "relevant (error messages, code identifiers, headings, file paths, URLs). "
    "Describe UI state (which tab/panel/window is active, any selections, any "
    "highlighted lines). Do NOT interpret the user's intent or answer any "
    "question. 3-6 sentences. No preamble, no markdown, no bullet points."
)


def handle_ask(question: str) -> str:
    """Vision-augmented query: capture+describe -> wiki select -> heavy answer.
    Falls back to brain-only when capture is blocked or fails."""
    question = (question or "").strip()
    if not question:
        return ("Usage: `/ask <question>`. Captures the current screen and "
                "answers using both what's on screen and your wiki.")

    description = ""
    app_label = ""
    skip_reason = ""

    if os.environ.get("CHLOE_VISION_DISABLED", "").strip() == "1":
        skip_reason = "kill switch on (CHLOE_VISION_DISABLED=1)"
    else:
        try:
            from screen_vision import (
                get_frontmost_app, is_blocked, capture_screen, describe_screen,
            )
        except ImportError as e:
            skip_reason = f"screen_vision unavailable: {e}"
            get_frontmost_app = None

        if not skip_reason:
            app = get_frontmost_app()
            if app.get("ok") and is_blocked(app):
                skip_reason = (f"frontmost app matches blocklist token "
                               f"`{is_blocked(app)}`")
            else:
                cap = capture_screen()
                if not cap.get("ok"):
                    skip_reason = f"capture failed: {cap.get('error','?')}"
                else:
                    desc = describe_screen(cap["png"], prompt=ASK_DESCRIPTION_PROMPT)
                    if not desc.get("ok"):
                        skip_reason = f"vision call failed: {desc.get('error','?')}"
                    else:
                        description = (desc.get("text") or "").strip()
                        a = cap.get("app") or {}
                        exe = (a.get("exe") or "").strip()
                        if exe.lower().endswith(".exe"):
                            exe = exe[:-4]
                        app_label = exe or (a.get("title") or "screen")[:40]

    try:
        index = BRAIN.read("wiki/index.md")
    except (FileNotFoundError, ValueError):
        index = ""

    pages = []
    if index:
        try:
            pages = BRAIN._keyword_select(question, index) or []
        except Exception:
            pages = []
        if not pages:
            try:
                pages = BRAIN._json_call(
                    f"You're answering a question against Chloe's wiki.\n\n"
                    f"Question: {question}\n\n"
                    f"Wiki index:\n---\n{index}\n---\n\n"
                    f'Return a JSON array of up to 5 page paths (relative to wiki/, '
                    f'e.g. "entities/foo.md") most likely to answer this question. '
                    f'Return ONLY the JSON array.',
                    "light",
                ) or []
                if not isinstance(pages, list):
                    pages = []
            except Exception:
                pages = []

    contexts = []
    for p in pages[:5]:
        try:
            contexts.append(f"## {p}\n\n" + BRAIN.read(f"wiki/{p}"))
        except (FileNotFoundError, ValueError):
            continue
    wiki_block = ("\n".join(contexts)
                  if contexts else "(no wiki pages matched the question)")

    if description:
        answer_prompt = (
            "Answer the user's question using STRICTLY two sources:\n"
            "  (1) the SCREEN section (what is currently visible on the user's screen)\n"
            "  (2) the WIKI section (the user's notes)\n"
            "Do NOT use prior knowledge or general world knowledge. "
            "If the answer requires information that is in neither source, "
            "say so explicitly and stop. Cite wiki pages as [[page_name]]. "
            "Refer to the screen as 'on screen' when citing it. Be concise.\n\n"
            f"Question: {question}\n\n"
            f"Screen ({app_label}):\n---\n{description}\n---\n\n"
            f"Wiki:\n---\n{wiki_block}\n---"
        )
    else:
        if not contexts:
            return (f"Vision was skipped ({skip_reason}) and the wiki has no "
                    f"pages relevant to that question. Nothing to answer from.")
        answer_prompt = (
            "Answer using STRICTLY the wiki content below. "
            "Do NOT use prior knowledge. If the wiki is silent on the question "
            "or any part of it, say so explicitly. Cite as [[page_name]].\n\n"
            f"Question: {question}\n\nWiki:\n---\n{wiki_block}\n---"
        )

    try:
        answer = chloe_llm_call(answer_prompt, "heavy")
    except Exception as e:
        return f"/ask failed during answer step: {type(e).__name__}: {e}"

    answer = (answer or "").strip() or "(empty answer from model)"
    if not description and skip_reason:
        answer = f"{answer}\n\n_(vision skipped: {skip_reason})_"
    return answer


_COWORK_SCHEDULED_DIR = Path(os.environ.get(
    "COWORK_SCHEDULED_DIR",
    r"C:\Users\eleew\OneDrive\Documents\Claude\Scheduled"))


def handle_status() -> str:
    """Self-awareness slash command. Returns a markdown snapshot of:
    queue depth, recent brain activity, scheduled-task last-runs, TTS
    engine, Ollama state.

    Each section is wrapped in try/except so one broken source doesn't
    blank the whole status. Bounded — no LLM calls, no slow IO.
    """
    import datetime as _dt
    import re as _re
    import socket
    import urllib.request

    out = []
    today = _dt.datetime.now().strftime("%Y-%m-%d")
    out.append(f"**Chloe status — {today} "
               f"{_dt.datetime.now().strftime('%H:%M')}**")
    out.append("")

    # ─── Queue ──────────────────────────────────────────────────────────
    try:
        queue_dir = BRAIN.root / "queue"
        if queue_dir.exists():
            queue_files = sorted(
                (p for p in queue_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime, reverse=True)
            if queue_files:
                latest = queue_files[0]
                age_s = _dt.datetime.now().timestamp() - latest.stat().st_mtime
                age = (f"{int(age_s/60)}m" if age_s < 3600
                       else f"{int(age_s/3600)}h" if age_s < 86400
                       else f"{int(age_s/86400)}d")
                out.append(f"**Queue:** {len(queue_files)} pending "
                           f"(most recent: `{latest.name}`, {age} old)")
            else:
                out.append("**Queue:** empty")
        else:
            out.append("**Queue:** no dir")
    except Exception as e:
        out.append(f"**Queue:** error ({type(e).__name__})")

    # ─── Brain activity (log.md tail) ───────────────────────────────────
    try:
        log_path = BRAIN.wiki_dir / "log.md"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            entries = _re.findall(r"^## \[([\d\- :]+)\] (\w+) \| (.+?)$",
                                  log_text, _re.M)
            last_lint = next(((ts, summary) for ts, op, summary
                              in reversed(entries) if op == "lint"),
                             None)
            recent_ingests = [(ts, summary) for ts, op, summary
                              in entries if op == "ingest"][-3:]
            if last_lint:
                out.append(f"**Last lint:** {last_lint[0]} "
                           f"({last_lint[1]})")
            if recent_ingests:
                names = ", ".join(s[:30] for _, s in recent_ingests)
                out.append(f"**Recent ingests ({len(recent_ingests)}):** "
                           f"{names}")
        else:
            out.append("**Brain log:** missing")
    except Exception as e:
        out.append(f"**Brain log:** error ({type(e).__name__})")

    # ─── Wiki state ─────────────────────────────────────────────────────
    try:
        from wiki_embedding import get_store
        store = get_store()
        out.append(f"**Wiki:** {store.count_embedded()} embedded pages "
                   f"(of {store.count_pages()} total)")
    except Exception as e:
        out.append(f"**Wiki:** error ({type(e).__name__})")

    # ─── Today's web searches ──────────────────────────────────────────
    try:
        sources_dir = BRAIN.wiki_dir / "sources"
        if sources_dir.exists():
            today_searches = list(
                sources_dir.glob(f"web_*_{today}.md"))
            out.append(f"**Web searches today:** {len(today_searches)}")
    except Exception as e:
        out.append(f"**Web searches:** error ({type(e).__name__})")

    # ─── Cowork scheduled tasks ─────────────────────────────────────────
    try:
        if _COWORK_SCHEDULED_DIR.exists():
            tasks = sorted(p for p in _COWORK_SCHEDULED_DIR.iterdir()
                           if p.is_dir())
            if tasks:
                out.append("")
                out.append(f"**Cowork scheduled tasks ({len(tasks)}):**")
                for t in tasks:
                    skill = t / "SKILL.md"
                    sched = "?"
                    if skill.exists():
                        body = skill.read_text(encoding="utf-8",
                                               errors="replace")[:2000]
                        m = _re.search(r"^cronExpression:\s*['\"]?(.+?)['\"]?\s*$",
                                       body, _re.M)
                        if m:
                            sched = m.group(1).strip()
                    out.append(f"- `{t.name}` ({sched})")
    except Exception as e:
        out.append(f"**Cowork tasks:** error ({type(e).__name__})")

    # ─── Runtime: TTS engine + Ollama ───────────────────────────────────
    try:
        out.append("")
        out.append("**Runtime:**")
        if os.environ.get("USE_ELEVENLABS", "0").strip() == "1":
            tts = f"ElevenLabs"
        elif os.environ.get("USE_KOKORO", "1").strip() == "1":
            voice = os.environ.get("KOKORO_VOICE", "af_heart")
            speed = os.environ.get("KOKORO_SPEED", "1.0")
            tts = f"Kokoro ({voice}, {speed}x)"
        else:
            tts = f"edge-tts ({os.environ.get('EDGE_TTS_VOICE', '?')})"
        out.append(f"- TTS: {tts}")

        ollama_url = os.environ.get("OLLAMA_URL",
                                    "http://localhost:11434").rstrip("/")
        try:
            socket.setdefaulttimeout(1.0)
            with urllib.request.urlopen(f"{ollama_url}/api/tags",
                                        timeout=1.0) as r:
                ok = (r.status == 200)
        except Exception:
            ok = False
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")
        keep = _get_ollama_keep_alive()
        out.append(f"- Ollama: {'reachable' if ok else 'UNREACHABLE'} "
                   f"({model}, keep_alive={keep})")
        mode = os.environ.get("CHLOE_MODE", "home")
        out.append(f"- Mode: {mode}")
    except Exception as e:
        out.append(f"**Runtime:** error ({type(e).__name__})")

    return "\n".join(out)


def handle_summarize_old(args: str) -> str:
    """Roll up old conversation turns into a compact wiki summary page.

    Pillar 4 of memory autopilot (2026-05-17): when too much raw history
    accumulates, compress the oldest unsummarized turns via Groq heavy
    into a single `wiki/episodic/conversation_summary_<date>_<idrange>.md`.
    wiki_watcher auto-embeds the page within ~2s so future recall surfaces
    the summary instead of scattered raw turns. Original rows stay in the
    DB but get `summarized=1` so search_turns skips them.

    Usage:
      /summarize_old              - run if unsummarized count >= threshold
      /summarize_old --dry-run    - preview prompt + range without writing
      /summarize_old <N>          - force-summarize the oldest N turns
                                    regardless of threshold

    Env knobs:
      CHLOE_SUMMARIZE_THRESHOLD (default 50) - trigger threshold
      CHLOE_SUMMARIZE_BATCH     (default 30) - turns per rollup
    """
    import datetime as _dt
    dry_run = "--dry-run" in args
    args_clean = args.replace("--dry-run", "").strip()
    batch_override = int(args_clean) if args_clean.isdigit() else None

    try:
        from jarvis import _memory  # type: ignore
    except Exception as e:
        return f"/summarize_old unavailable: {e}"
    if _memory is None:
        return "/summarize_old: memory not initialized yet"

    try:
        threshold = int(os.environ.get("CHLOE_SUMMARIZE_THRESHOLD", "50"))
    except ValueError:
        threshold = 50
    try:
        batch_size = batch_override or int(
            os.environ.get("CHLOE_SUMMARIZE_BATCH", "30"))
    except ValueError:
        batch_size = 30

    unsummed = _memory.unsummarized_count()
    if batch_override is None and unsummed < threshold:
        return (f"Only {unsummed} unsummarized turns — below threshold "
                f"({threshold}). Run `/summarize_old {batch_size}` to force, "
                f"or lower CHLOE_SUMMARIZE_THRESHOLD.")

    rows = _memory.oldest_unsummarized_turns(limit=batch_size)
    if not rows:
        return "No unsummarized turns to roll up."

    id_start, id_end = rows[0]["id"], rows[-1]["id"]
    ts_start = _dt.datetime.fromtimestamp(rows[0]["ts"]).strftime(
        "%Y-%m-%d %H:%M")
    ts_end = _dt.datetime.fromtimestamp(rows[-1]["ts"]).strftime(
        "%Y-%m-%d %H:%M")

    transcript_lines = []
    for r in rows:
        ts = _dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
        transcript_lines.append(f"[{ts}] {r['role']}: {r['content']}")
    transcript = "\n".join(transcript_lines)

    prompt = (
        f"Below is a sequence of {len(rows)} conversation turns between "
        f"Edward and Chloe (Edward's voice + chat assistant). Compress them "
        f"into a compact narrative summary that preserves:\n"
        f"- Topics discussed (especially recurring ones)\n"
        f"- Decisions or commitments Edward made\n"
        f"- Persona-relevant statements (preferences, opinions, biographical)\n"
        f"- Open questions or unresolved threads\n"
        f"- Anything worth remembering about Edward from this window\n\n"
        f"Skip pure transactional exchanges (light-switch commands, weather "
        f"queries) unless they reveal a pattern. Aim for 200-400 words. "
        f"Write in Chloe's voice — lowercase, contractions, prose with "
        f"em-dashes, no bullet lists, no clinical headers. Output ONLY the "
        f"summary body — no frontmatter, no markdown title.\n\n"
        f"--- TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
    )

    if dry_run:
        preview = "\n".join(transcript_lines[:3])
        return (f"**Dry run** — would summarize {len(rows)} turns "
                f"(ids {id_start}-{id_end}, {ts_start} → {ts_end}). "
                f"Prompt size: {len(prompt)} chars.\n\n"
                f"First 3 lines:\n```\n{preview}\n```")

    try:
        summary_body = chloe_llm_call(prompt, "heavy").strip()
    except Exception as e:
        return f"Summarization failed: {type(e).__name__}: {e}"
    if not summary_body:
        return "Summarization returned empty — aborting (turns not marked)."

    today = _dt.date.today().strftime("%Y-%m-%d")
    episodic_dir = BRAIN.wiki_dir / "episodic"
    episodic_dir.mkdir(parents=True, exist_ok=True)
    slug = f"conversation_summary_{today}_{id_start:06d}-{id_end:06d}"
    path = episodic_dir / f"{slug}.md"

    frontmatter = (
        f"---\n"
        f"title: Conversation summary {ts_start} to {ts_end}\n"
        f"type: episodic\n"
        f"source_type: conversation_rollup\n"
        f"created: {today}\n"
        f"turn_id_start: {id_start}\n"
        f"turn_id_end: {id_end}\n"
        f"turns_compressed: {len(rows)}\n"
        f"generated_via: /summarize_old\n"
        f"---\n\n"
    )
    body = (
        f"# Conversation summary, {ts_start} to {ts_end}\n\n"
        f"{summary_body}\n"
    )
    path.write_text(frontmatter + body, encoding="utf-8")

    marked = _memory.mark_summarized([r["id"] for r in rows])
    remaining = _memory.unsummarized_count()

    preview = summary_body[:300] + ("..." if len(summary_body) > 300 else "")
    return (f"Rolled up {len(rows)} turns (ids {id_start}-{id_end}) → "
            f"`wiki/episodic/{slug}.md`. Marked {marked} summarized. "
            f"{remaining} unsummarized turns remain.\n\n"
            f"Summary preview:\n> {preview}")


def handle_capabilities(args: str) -> str:
    """Self-analysis: enumerate slash commands, MCP tools, scheduled jobs,
    env knobs, and modules. Bounded — no LLM calls, no external IO.

    Usage:
      /capabilities                 - full markdown report
      /capabilities slashes         - just slash commands
      /capabilities tools           - just MCP tools
      /capabilities jobs            - just scheduled jobs + health
      /capabilities env             - just env knobs
      /capabilities modules         - just module list
      /capabilities --json          - structured payload (for MCP callers)
    """
    import chloe_capabilities
    import json

    parts = (args or "").strip().split()
    want_json = "--json" in parts
    parts = [p for p in parts if p != "--json"]
    section = parts[0].lower() if parts else None

    if want_json:
        if section is None:
            return json.dumps(chloe_capabilities.summary(),
                              indent=2, default=str)
        # Section-specific JSON paths.
        getter = {
            "slashes": chloe_capabilities.list_slash_commands,
            "tools":   chloe_capabilities.list_mcp_tools,
            "jobs":    chloe_capabilities.list_jobs,
            "env":     chloe_capabilities.list_env_knobs,
            "modules": chloe_capabilities.list_modules,
        }.get(section)
        if getter is None:
            return f"Unknown section: {section!r}. Try slashes|tools|jobs|env|modules."
        return json.dumps(getter(), indent=2, default=str)

    if section is None:
        return chloe_capabilities.format_summary_markdown()

    # Section-specific markdown.
    if section == "slashes":
        items = chloe_capabilities.list_slash_commands()
        out = [f"# Slash commands ({len(items)})"]
        for i in items:
            line = f"- `{i['name']}`"
            if i.get("summary"):
                line += f" — {i['summary']}"
            out.append(line)
        return "\n".join(out)
    if section == "tools":
        items = chloe_capabilities.list_mcp_tools()
        out = [f"# MCP tools ({len(items)})"]
        for t in items:
            line = f"- `mcp__chloe__{t['name']}` — `{t['signature']}`"
            if t.get("summary"):
                line += f"\n  {t['summary']}"
            out.append(line)
        return "\n".join(out)
    if section == "jobs":
        items = chloe_capabilities.list_jobs()
        out = [f"# Scheduled jobs ({len(items)})"]
        for j in items:
            if j.get("error"):
                out.append(f"- _error: {j['error']}_")
                continue
            sched = j.get("schedule") or "?"
            health = j.get("health") or "?"
            age = j.get("age_hours")
            age_str = (f"{age:.1f}h ago" if isinstance(age, (int, float))
                       else "never run")
            result = j.get("last_result") or ""
            line = f"- `{j['name']}` — {sched} · **{health}** · {age_str}"
            if result:
                line += f"\n  _{result[:120]}_"
            out.append(line)
        return "\n".join(out)
    if section == "env":
        items = chloe_capabilities.list_env_knobs()
        out = [f"# Env knobs ({len(items)})"]
        for k in items:
            default = k.get("default", "")
            if len(default) > 80:
                default = default[:77] + "..."
            out.append(f"- `{k['name']}` = {default or '<no default>'}")
        return "\n".join(out)
    if section == "modules":
        items = chloe_capabilities.list_modules()
        out = [f"# Modules ({len(items)})"]
        for m in items:
            line = (f"- `{m['path']}` — {m['line_count']} lines, "
                    f"{m['function_count']} top-level fn(s)")
            if m.get("summary"):
                line += f" — {m['summary']}"
            out.append(line)
        return "\n".join(out)
    return (f"Unknown section: {section!r}. "
            f"Try: slashes, tools, jobs, env, modules. "
            f"Or `/capabilities` (no arg) for the full report.")


def handle_explain(args: str) -> str:
    """Ast-introspect any module under jarvis/. Returns module docstring,
    imports, top-level constants, function signatures + first-line
    docstrings, classes + method lists.

    Usage:
      /explain <module>          - markdown report
      /explain <module> --json   - structured payload
    """
    import chloe_capabilities
    import json

    parts = (args or "").strip().split()
    want_json = "--json" in parts
    parts = [p for p in parts if p != "--json"]
    if not parts:
        return ("Usage: `/explain <module>` "
                "(e.g. `/explain brain_wiring`). "
                "List modules with `/capabilities modules`.")
    name = parts[0]
    try:
        d = chloe_capabilities.describe_module(name)
    except FileNotFoundError as e:
        return f"Unknown module: {name!r}. {e}"
    if want_json:
        return json.dumps(d, indent=2, default=str)
    return chloe_capabilities.format_module_markdown(d)


def handle_apply_proposal(args: str) -> str:
    """Apply a code proposal from `proposals/code_<date>_<slug>.md`.

    Tier 1 self-modification (2026-05-19). Chloe (or any Cowork job) writes
    a markdown proposal describing a change to her own source. Ed reviews it
    and triggers application via this slash. Safety rails (path whitelist,
    ast.parse for .py, timestamped backup, max 5 per session) live in
    `chloe_proposals.apply_proposal`.

    Usage:
      /apply_proposal <slug>             - apply
      /apply_proposal <slug> --dry-run   - preview without writing
      /apply_proposal --list             - list pending proposals
    """
    import chloe_proposals

    parts = (args or "").strip().split()
    dry_run = False
    slug = None
    list_mode = False
    list_status = None
    for tok in parts:
        if tok in ("--dry-run", "--dryrun", "-n"):
            dry_run = True
        elif tok == "--list":
            list_mode = True
        elif tok.startswith("--status="):
            list_mode = True
            list_status = tok.split("=", 1)[1].strip().lower() or None
        elif slug is None:
            slug = tok

    if list_mode:
        rows = chloe_proposals.list_proposals(status=list_status)
        if not rows:
            return ("No code proposals found. They live at "
                    f"`{chloe_proposals.proposals_dir()}\\code_*.md`.")
        state = chloe_proposals.session_state()
        out = [f"**Code proposals** ({len(rows)} total, "
               f"{state['remaining']}/{state['max_per_session']} "
               f"applies left this session):\n"]
        for r in rows[:20]:
            mark = {"applied": "[x]", "reverted": "[~]",
                    "pending": "[ ]"}.get(r["status"], "[?]")
            out.append(f"  {mark} **{r['slug']}** — {r['title']}  "
                       f"({r['kind']} → `{r['target']}`)")
        return "\n".join(out)

    if not slug:
        return ("Usage: `/apply_proposal <slug> [--dry-run]` "
                "or `/apply_proposal --list`. "
                "Proposals live at "
                f"`{chloe_proposals.proposals_dir()}\\code_*.md`.")

    r = chloe_proposals.apply_proposal(slug, dry_run=dry_run)
    if r.get("ok"):
        return r.get("message") or f"Applied `{r.get('slug', slug)}`."
    return f"Apply failed: {r.get('error', 'unknown error')}"


def handle_revert_proposal(args: str) -> str:
    """Revert a previously-applied code proposal from its timestamped backup.

    Companion to /apply_proposal. Restores the target file's bytes from the
    `.bak.proposal_<slug>_<stamp>` snapshot and stamps the proposal
    frontmatter as `status: reverted`.

    Usage:
      /revert_proposal <slug>
    """
    import chloe_proposals
    slug = (args or "").strip().split()[0] if args else None
    if not slug:
        return "Usage: `/revert_proposal <slug>`"
    r = chloe_proposals.revert_proposal(slug)
    if r.get("ok"):
        return r.get("message") or f"Reverted `{r.get('slug', slug)}`."
    return f"Revert failed: {r.get('error', 'unknown error')}"


def handle_autonomous(args: str) -> str:
    """Stage-4 autonomous self-modification control.

    The autonomous proposer (`chloe_jobs.job_autonomous_fix_recurring_errors`)
    scans logs for recurring tracebacks and drafts fix proposals. Apply
    behavior is gated by the `enabled` flag + watchdog rate limits.

    Usage:
      /autonomous                 - show current state + recent history
      /autonomous on              - enable auto-apply (default: off)
      /autonomous off             - disable auto-apply (proposer still
                                    writes proposals; no auto-apply)
      /autonomous freeze <mins>   - block applies for N minutes
      /autonomous unfreeze        - clear the freeze
      /autonomous reset           - clear consecutive_failures lock
      /autonomous history         - last 20 watchdog events
      /autonomous run-now         - fire the proposer manually (writes
                                    digest + proposals; auto-applies
                                    only if state.enabled AND gate open)
    """
    import chloe_watchdog
    import datetime as _dt
    parts = (args or "").strip().split()

    # No args → status
    if not parts:
        from chloe_jobs import _read_autonomous_state
        s = _read_autonomous_state()
        wd = chloe_watchdog.status()
        now = _dt.datetime.now().timestamp()
        fz = s.get("frozen_until", 0.0)
        fz_str = (f"frozen for {int((fz - now)/60)} more min"
                  if fz > now else "not frozen")
        out = ["**Stage-4 autonomous status:**"]
        out.append(f"- Enabled: **{'ON' if s.get('enabled') else 'OFF'}**")
        out.append(f"- Freeze: {fz_str}")
        out.append(f"- Applied today: "
                   f"{wd['applies_today']} / {wd['max_per_day']}")
        out.append(f"- Consecutive failures: "
                   f"{wd['consecutive_failures']} / "
                   f"{wd['max_consecutive_failures']}")
        if wd["consecutive_failures"] >= wd["max_consecutive_failures"]:
            out.append("- **LOCKED.** Use `/autonomous reset` to clear.")
        under_watch = wd.get("under_watch", {})
        if under_watch:
            out.append(f"- Under watch: {', '.join(under_watch.keys())}")
        return "\n".join(out)

    cmd = parts[0].lower()

    if cmd == "on":
        from chloe_jobs import _read_autonomous_state, _write_autonomous_state
        s = _read_autonomous_state()
        s["enabled"] = True
        _write_autonomous_state(s)
        return ("**Stage-4 autonomous ENABLED.** "
                "The proposer will auto-apply fix proposals when "
                "confidence >= 0.85 and the watchdog gate is open. "
                "Daily cap: 2 applies. Use `/autonomous off` to disable.")

    if cmd == "off":
        from chloe_jobs import _read_autonomous_state, _write_autonomous_state
        s = _read_autonomous_state()
        s["enabled"] = False
        _write_autonomous_state(s)
        return ("**Stage-4 autonomous DISABLED.** "
                "Proposer will still write digest + proposals to "
                "`proposals/code_autonomous_*.md` for manual review.")

    if cmd == "freeze" and len(parts) >= 2:
        try:
            mins = int(parts[1])
        except ValueError:
            return f"Bad minutes value: {parts[1]!r}"
        if mins < 1 or mins > 1440:
            return f"Freeze minutes must be 1-1440, got {mins}"
        from chloe_jobs import _read_autonomous_state, _write_autonomous_state
        import time as _t
        s = _read_autonomous_state()
        s["frozen_until"] = _t.time() + (mins * 60)
        _write_autonomous_state(s)
        return f"**Frozen** for {mins} minutes."

    if cmd == "unfreeze":
        from chloe_jobs import _read_autonomous_state, _write_autonomous_state
        s = _read_autonomous_state()
        s["frozen_until"] = 0.0
        _write_autonomous_state(s)
        return "Unfrozen."

    if cmd == "reset":
        r = chloe_watchdog.reset_failures()
        return (f"Reset consecutive_failures counter "
                f"(was {r['prior_failures']}).")

    if cmd == "history":
        rows = chloe_watchdog.history(limit=20)
        if not rows:
            return "No watchdog history yet."
        out = [f"**Watchdog history ({len(rows)} most recent):**"]
        for h in rows:
            out.append(f"- `{h.get('ts_iso','?')}` · "
                       f"**{h.get('action','?')}** · "
                       f"{h.get('outcome','?')} · "
                       f"`{h.get('slug','?')}` · "
                       f"{(h.get('reason','') or '')[:80]}")
        return "\n".join(out)

    if cmd == "run-now":
        try:
            from chloe_jobs import job_autonomous_fix_recurring_errors
            result = job_autonomous_fix_recurring_errors()
            return f"**Autonomous run complete:**\n\n{result}"
        except Exception as e:
            # Log the full traceback to backend.log so we can debug
            # without making Ed grep — chat response stays short.
            import traceback as _tb
            _tb.print_exc()
            return (f"Run failed: {type(e).__name__}: {e}\n"
                    f"_(full traceback in `logs/backend.log`)_")

    return ("Usage: `/autonomous [on | off | freeze <mins> | unfreeze | "
            "reset | history | run-now]`. Empty for status.")


def handle_pending_confirms(args: str) -> str:
    """List Stage-3 pending-confirm slots, or cancel one.

    Usage:
      /pending_confirms                  - list active pending
      /pending_confirms cancel <slug>    - drop one pending
      /pending_confirms cancel-all       - drop every pending
    """
    import chloe_pending_confirms
    parts = (args or "").strip().split()
    if not parts:
        rows = chloe_pending_confirms.pending()
        if not rows:
            return "No pending confirms."
        out = [f"**Pending confirms ({len(rows)}):**"]
        for r in rows:
            ttl_s = int(r["ttl_remaining_s"])
            ttl_str = (f"{ttl_s // 60}m{ttl_s % 60}s"
                       if ttl_s >= 60 else f"{ttl_s}s")
            out.append(f"- `{r['slug']}` — channel `{r['source']}` · "
                       f"{ttl_str} left · {r['summary'][:80]}")
        out.append("\n_Reply yes / yeah / go ahead to apply the most "
                   "recent one in this channel; no / cancel to drop._")
        return "\n".join(out)

    cmd = parts[0].lower()
    if cmd == "cancel-all":
        r = chloe_pending_confirms.cancel("")
        canceled = r.get("canceled", [])
        if not canceled:
            return "Nothing to cancel."
        return f"Canceled {len(canceled)} pending: {', '.join(canceled)}"
    if cmd == "cancel" and len(parts) >= 2:
        slug = parts[1]
        r = chloe_pending_confirms.cancel(slug)
        if r.get("ok"):
            return f"Canceled `{slug}`."
        return f"Cancel failed: {r.get('error', 'unknown')}"
    return ("Usage: `/pending_confirms` "
            "| `/pending_confirms cancel <slug>` "
            "| `/pending_confirms cancel-all`")


def handle_issue_apply_token(args: str) -> str:
    """Mint a Tier-2 confirm-token good for N applies in M minutes.

    Ed types this once per session to authorize a batch of Chloe-driven
    code applies. Chloe (via MCP) or any Cowork job can then call
    `apply_self_patch(slug, token)` without Ed retyping the slash for
    each apply. All Tier-1 safety rails still fire — the token only
    relaxes the "human types the slash at apply time" gate.

    Usage:
      /issue_apply_token                       - 1 apply, 30 min TTL
      /issue_apply_token --applies 3           - 3 applies, 30 min
      /issue_apply_token --minutes 60          - 1 apply, 60 min
      /issue_apply_token --applies 3 --minutes 60
      /issue_apply_token --status              - show active tokens (masked)
      /issue_apply_token --revoke              - drop every active token
    """
    import chloe_proposals
    parts = (args or "").strip().split()
    applies = chloe_proposals.DEFAULT_TOKEN_APPLIES
    minutes = chloe_proposals.DEFAULT_TOKEN_MINUTES
    status_mode = False
    revoke_mode = False
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "--applies" and i + 1 < len(parts):
            try:
                applies = int(parts[i + 1])
            except ValueError:
                return f"Bad --applies value: {parts[i+1]!r} (must be integer)"
            i += 2
            continue
        if tok == "--minutes" and i + 1 < len(parts):
            try:
                minutes = int(parts[i + 1])
            except ValueError:
                return f"Bad --minutes value: {parts[i+1]!r} (must be integer)"
            i += 2
            continue
        if tok == "--status":
            status_mode = True
            i += 1
            continue
        if tok == "--revoke":
            revoke_mode = True
            i += 1
            continue
        return (f"Unknown arg: {tok!r}. "
                f"Usage: `/issue_apply_token [--applies N] [--minutes M] "
                f"| --status | --revoke`")
    if status_mode:
        toks = chloe_proposals.list_tokens()
        if not toks:
            return "No active apply-tokens."
        out = [f"**Active apply-tokens ({len(toks)}):**"]
        for t in toks:
            out.append(f"- `{t['token_id']}` — {t['applies_remaining']} "
                       f"applies left, expires in "
                       f"{t['expires_in_minutes']} min (issued "
                       f"{t['issued_at']})")
        return "\n".join(out)
    if revoke_mode:
        r = chloe_proposals.revoke_tokens()
        return f"Revoked {r.get('revoked', 0)} apply-token(s)."

    r = chloe_proposals.issue_token(applies=applies, minutes=minutes)
    if not r.get("ok"):
        return f"Couldn't issue token: {r.get('error', 'unknown error')}"
    return (f"**Apply-token issued.** Valid for {r['applies']} apply(s), "
            f"expires {r['expires_iso']}.\n\n"
            f"`{r['token']}`\n\n"
            f"Pass this token to `apply_self_patch(slug, token)` via MCP, "
            f"or `apply_proposal_with_token(slug, token)` in code. "
            f"`/issue_apply_token --status` shows remaining quota. "
            f"`/issue_apply_token --revoke` drops every active token.")


def try_handle_brain_command(user_text: str):
    """Synchronous brain command handler. Returns reply string or None.

    Wrap in asyncio.to_thread() when calling from an async context — these
    operations issue HTTP calls and may take a few seconds for ingest/lint.

    Side effect: for non-command chat messages, fires the auto-fact extractor
    on a daemon thread. Always returns None for non-commands; chat path
    continues normally.
    """
    msg = (user_text or "").strip()

    # Natural-language triggers for /wiki_* family. Translate to the slash
    # form before any other routing so the rest of this function still keys
    # off `msg.startswith("/...")`. Conservative anchors — only fires when
    # the verb + "wiki" + a topic are all present, never when "wiki" appears
    # as a passing noun ("the wiki has my notes").
    nl_alias = _maybe_wiki_nl_alias(msg)
    if nl_alias is not None:
        msg = nl_alias

    # Stage 3 self-modification (2026-05-19): voice/chat-confirm per
    # apply. If Ed's message is a non-slash affirmative/negative reply
    # AND there's a pending confirm announced via this channel, resolve
    # it here BEFORE the normal slash dispatch. Source separation
    # ("chat" only matches chat-announced pendings; voice path uses its
    # own resolve() call).
    if msg and not msg.startswith("/"):
        try:
            import chloe_pending_confirms
            resolution = chloe_pending_confirms.resolve(msg, source="chat")
        except Exception as e:
            print(f"[chloe] pending-confirm resolve crashed: "
                  f"{type(e).__name__}: {e}", flush=True)
            resolution = None
        if resolution is not None:
            return resolution.get("reply_text", "(no reply)")

    # Real-time weather (2026-05-26): answer weather questions from a live
    # weather API (weather.py) instead of the generic web-search path. Fires
    # only on an actual weather ask — maybe_weather_reply returns None
    # otherwise, so normal chat continues. Runs before auto-fact so weather
    # turns don't pollute the fact store.
    if msg and not msg.startswith("/"):
        try:
            import weather as _weather
            _wx = _weather.maybe_weather_reply(msg)
        except Exception as e:
            print(f"[chloe] weather check crashed: "
                  f"{type(e).__name__}: {e}", flush=True)
            _wx = None
        if _wx:
            return _wx

    # Auto-fact extraction for plain chat (not slash commands). Fire-and-forget.
    if msg and not msg.startswith("/"):
        maybe_auto_extract(msg)

    if msg.startswith("/weather"):
        place = msg[len("/weather"):].strip() or None
        try:
            import weather as _weather
            return _weather.weather_reply(place)
        except Exception as e:
            return f"Weather lookup failed: {type(e).__name__}: {e}"

    if msg.startswith("/tone"):
        # Manual voice-tone override for deterministic testing of the TTS
        # blends. Normally Chloe picks tones herself from the conversation;
        # this forces one (sticky until changed). Speak a line after to hear it.
        arg = msg[len("/tone"):].strip().lower()
        try:
            import tts_tones
            avail = ", ".join(sorted(tts_tones.PALETTE))
            if arg in ("", "list", "?", "status"):
                return f"voice tones: {avail}\ncurrent: {tts_tones.current_tone()}"
            if arg in ("reset", "off"):
                tts_tones.reset_tone()
                return "voice tone reset to neutral."
            if tts_tones.set_tone(arg):
                return (f"voice tone set to '{arg}' (sticky until you change it). "
                        f"note: PALETTE is import-cached, so new tones need a "
                        f"Chloe restart to actually sound different.")
            return f"unknown tone '{arg}'. available: {avail}"
        except Exception as e:
            return f"tone command failed: {type(e).__name__}: {e}"

    if msg.startswith("/forget"):
        # User-facing privacy control: delete matching memories everywhere
        # Chloe stores them — the SQLite turn log (FTS stays consistent via the
        # AFTER-DELETE trigger) and the typed user-model. Permanent.
        target = msg[len("/forget"):].strip()
        if len(target) < 3:
            return ("usage: /forget <text> — permanently deletes turn-log rows "
                    "and user-model items containing <text> (3+ chars).")
        parts = []
        try:
            import sys as _sys
            _j = _sys.modules.get("jarvis")
            if _j is not None and hasattr(_j, "_memory"):
                mem = _j._memory          # live instance when inside Chloe
            else:                          # standalone (MCP server, scripts)
                from chloe_memory import ChloeMemory as _CM
                _base = os.path.dirname(os.path.abspath(__file__))
                mem = _CM(os.path.join(_base, "chloe_memory.db"),
                          os.path.join(_base, "facts.md"))
            parts.append(f"turn-log rows deleted: {mem.forget(target)}")
        except Exception as e:
            parts.append(f"turn-log forget failed: {type(e).__name__}: {e}")
        try:
            import chloe_ed_profile
            parts.append(f"user-model items removed: "
                         f"{chloe_ed_profile.forget(target)}")
        except Exception as e:
            parts.append(f"user-model forget failed: {type(e).__name__}: {e}")
        return f"forgot {target!r} — " + "; ".join(parts)

    if msg.startswith("/accept_reflection"):
        # Apply the staged reflection into the typed user-model (quarantined).
        try:
            import chloe_ed_profile
            res = chloe_ed_profile.accept_pending()
            if not res:
                return "no pending reflection to accept."
            merged = ", ".join(f"{k}:{n}" for k, n in (res.get("merged") or {}).items())
            return (f"reflection accepted (staged {res.get('accepted_ts')}) — "
                    f"merged {merged or 'nothing'}.")
        except Exception as e:
            return f"accept_reflection failed: {type(e).__name__}: {e}"

    if msg.startswith("/reject_reflection"):
        try:
            import chloe_ed_profile
            return ("pending reflection discarded."
                    if chloe_ed_profile.reject_pending()
                    else "no pending reflection to discard.")
        except Exception as e:
            return f"reject_reflection failed: {type(e).__name__}: {e}"

    if msg.startswith("/reflection"):
        # View the staged reflection. Demoted (Phase 5): reflection no longer
        # auto-merges into the durable model — it stages a proposal you apply
        # with /accept_reflection or drop with /reject_reflection.
        try:
            import chloe_ed_profile
            pend = chloe_ed_profile.pending()
            if not pend or not pend.get("updates"):
                return "no pending reflection."
            lines = [f"pending reflection (staged {pend.get('ts')}):"]
            for slot, items in (pend.get("updates") or {}).items():
                for it in items:
                    lines.append(f"  [{slot}] {it}")
            lines.append("apply: /accept_reflection   discard: /reject_reflection")
            return "\n".join(lines)
        except Exception as e:
            return f"reflection view failed: {type(e).__name__}: {e}"

    if msg.startswith("/debug turn") or msg.startswith("/debugturn"):
        # Full replay of one turn: gloss + per-block token math + the exact
        # assembled system prompt the model saw. "/debug turn 42".
        tail = msg.split("turn", 1)[1].strip() if "turn" in msg else ""
        try:
            import chloe_trace
            if not tail.isdigit():
                return "usage: /debug turn <turn_id>  (see ids via /whathappened)"
            return chloe_trace.format_turn(int(tail))
        except Exception as e:
            return f"debug-turn failed: {type(e).__name__}: {e}"

    if msg.startswith("/whathappened") or msg.startswith("/trace"):
        # Observability: show the decision trace for the last turn(s) — mood
        # read, retrieval, which context blocks the composer dropped, latency.
        # "why did that reply feel off?" answered in one look.
        arg = msg.split(None, 1)
        try:
            n = int(arg[1].strip()) if len(arg) > 1 and arg[1].strip().isdigit() else 1
        except Exception:
            n = 1
        try:
            import chloe_trace
            return chloe_trace.format_last(max(1, min(n, 20)))
        except Exception as e:
            return f"trace lookup failed: {type(e).__name__}: {e}"

    if msg.startswith("/ingest "):
        # Parse args: optional --dry-run flag plus required filename.
        # Flag can appear before or after the filename.
        raw_args = msg[len("/ingest "):].strip().split()
        dry_run = False
        paths = []
        for arg in raw_args:
            if arg in ("--dry-run", "--dryrun", "-n"):
                dry_run = True
            else:
                paths.append(arg)
        if not paths:
            return "Usage: `/ingest [--dry-run] <filename>`"
        path = paths[0]
        try:
            r = BRAIN.ingest(path, dry_run=dry_run)
            if r.get('dry_run'):
                return _format_dry_run(r)
            return (f"Ingested **{r['slug']}**: {r['tldr']}\n\n"
                    f"Touched {len(r['entities_touched'])} entities, "
                    f"{len(r['concepts_touched'])} concepts.")
        except FileNotFoundError as e:
            return (f"Ingest failed: source not found. Place the file in "
                    f"{BRAIN.raw_dir} and pass just the filename. ({e})")
        except Exception as e:
            return f"Ingest failed: {type(e).__name__}: {e}"

    if msg.startswith("/query "):
        return BRAIN.query(msg[len("/query "):].strip())

    if msg == "/lint":
        r = BRAIN.lint()
        return (f"Lint complete. Scanned {r['pages_scanned']} pages. "
                f"{len(r['orphans'])} orphans, "
                f"{len(r['contradictions'])} contradictions surfaced "
                f"to wiki/gaps.md.")

    if msg.startswith("/podcast") or msg == "/podcast":
        # /podcast              — render most recent script
        # /podcast <pattern>    — render most recent script whose filename
        #                         matches <pattern> (e.g. "karpathy" matches
        #                         karpathy_wiki_kb_2026-05-08_214951.md)
        args = msg[len("/podcast"):].strip()
        try:
            # Late imports — Kokoro+numpy are heavy, defer until requested
            from audio_overview import render_script, autoplay
            from pathlib import Path

            scripts = sorted(BRAIN.overviews_dir.glob('*.md'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            if not scripts:
                return ("No overview scripts to render. "
                        "Run `/overview` first to generate one.")

            if args:
                # Substring match against filename
                matches = [p for p in scripts if args.lower() in p.stem.lower()]
                if not matches:
                    return f"No script matching `{args}` in {BRAIN.overviews_dir}"
                script_path = matches[0]
            else:
                script_path = scripts[0]

            r = render_script(str(script_path))
            played = autoplay(r['path'])

            errors_note = ""
            if r['exchanges_failed']:
                errors_note = f"\n  **Failed:** {r['exchanges_failed']} turn(s) skipped"

            playback_note = (" — playing now" if played
                             else " — open the file to listen")
            status_text = (
                f"Rendered audio overview{playback_note}.\n\n"
                f"  **Script:** `{script_path.name}`\n"
                f"  **Audio:** `{r['path']}`\n"
                f"  **Duration:** {r['duration_min']} min\n"
                f"  **Voices:** {r['voice_a']} (host) / {r['voice_b']} (expert)\n"
                f"  **Rendered:** {r['exchanges_rendered']} turns"
                f"{errors_note}"
            )
            # Dict return suppresses the chat-reply TTS — the podcast itself
            # is already playing via os.startfile, so we don\'t want the
            # status text TTS\'d on top of it.
            return {"text": status_text, "no_tts": True}
        except FileNotFoundError as e:
            return f"Podcast render failed: {e}"
        except RuntimeError as e:
            return f"Podcast render failed: {e}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Podcast render failed: {type(e).__name__}: {e}"

    if msg.startswith("/overview") or msg == "/overview":
        # /overview                — script over all sources
        # /overview <slug>         — script over one source (with its linked pages)
        # /overview <slug1> <slug2> — multiple sources
        args = msg[len("/overview"):].strip()
        slugs = args.split() if args else None
        try:
            r = BRAIN.audio_overview_script(source_slugs=slugs)
            return (f"Overview script generated.\n\n"
                    f"  **Sources:** {r['source_count']}\n"
                    f"  **Exchanges:** {r['exchanges']}\n"
                    f"  **Est. length:** ~{r['estimated_minutes']} min when read aloud\n"
                    f"  **Saved to:** `{r['path']}`\n\n"
                    f"Open the file to review the script. If it reads well, "
                    f"we'll add audio rendering next.")
        except (ValueError, RuntimeError) as e:
            return f"Overview failed: {e}"
        except Exception as e:
            return f"Overview failed: {type(e).__name__}: {e}"

    if msg.startswith("/add "):
        # /add <type> <slug> <body...>
        parts = msg[len("/add "):].split(None, 2)
        if len(parts) < 3:
            return ("Usage: `/add <type> <slug> <body>`\n"
                    "  type: `entity` or `concept`\n"
                    "  slug: short snake_case name (e.g. `qmd`, `schema`)\n"
                    "  body: 1-3 sentence description\n\n"
                    "Example: `/add entity qmd Local search engine for markdown "
                    "with BM25 + vector search.`")
        page_type, slug, body = parts
        try:
            r = BRAIN.add_page(page_type, slug, body)
            verb = "Created" if r['action'] == 'created' else "Updated"
            return f"{verb} {r['type']} page **{r['slug']}**."
        except ValueError as e:
            return f"Add failed: {e}"

    if msg.startswith("/fact "):
        slug = BRAIN.fact_extract_and_add(msg[len("/fact "):].strip())
        if slug:
            return f"Fact saved: **{slug}**"
        return "Nothing durable in that statement — try again with a clearer assertion."

    if msg.startswith("/see ambient") or msg == "/see ambient":
        # /see ambient            — show status
        # /see ambient on [N]     — start (or restart) loop at N-minute cadence
        # /see ambient off        — stop loop
        sub = msg[len("/see ambient"):].strip().split()
        try:
            import ambient_vision as av
        except ImportError as e:
            return f"/see ambient unavailable: {e}"

        if not sub:
            s = av.status()
            if not s["running"]:
                return ("Ambient vision is **off**. "
                        "Start with `/see ambient on` "
                        "(or `/see ambient on 5` for 5-minute cadence).")
            return ("Ambient vision is **on**.\n"
                    f"  Cadence: every {s['minutes']:.1f} min\n"
                    f"  Started: {s['started_at']}\n"
                    f"  Ticks: {s['ticks_total']} total, "
                    f"{s['ticks_skipped_blocked']} skipped (blocked), "
                    f"{s['ticks_skipped_disabled']} skipped (disabled), "
                    f"{s['ticks_failed']} failed\n"
                    f"  Last tick: {s['last_tick'] or 'none yet'}\n"
                    f"  Last entry: {s['last_text'] or '-'}")

        cmd = sub[0].lower()
        if cmd == "off":
            r = av.stop()
            if r.get("note") == "not running":
                return "Ambient vision was already off."
            return "Ambient vision **stopped**."
        if cmd == "on":
            try:
                minutes = float(sub[1]) if len(sub) >= 2 else None
            except ValueError:
                return f"Bad cadence `{sub[1]}` - must be a number of minutes."
            try:
                s = av.start(minutes=minutes)
            except Exception as e:
                import traceback; traceback.print_exc()
                return f"Ambient start failed: {type(e).__name__}: {e}"
            return (f"Ambient vision **on** - capturing every "
                    f"{s['minutes']:.1f} min, logging to today's episodic file. "
                    f"Stop with `/see ambient off`.")
        return ("Usage:\n"
                "  `/see ambient`           - status\n"
                "  `/see ambient on [N]`    - start (default 10 min)\n"
                "  `/see ambient off`       - stop")

    if msg.startswith("/see") or msg == "/see":
        prompt = msg[len("/see"):].strip()
        try:
            from screen_vision import see as _see
            r = _see(prompt)
        except ImportError as e:
            return (f"/see unavailable: {e}\n"
                    f"Install with: pip install mss pywin32 pillow")
        except Exception as e:
            import traceback; traceback.print_exc()
            return f"/see crashed: {type(e).__name__}: {e}"

        if not r.get("ok"):
            if r.get("blocked_by"):
                return (f"Skipped - frontmost window matches blocklist token "
                        f"`{r['blocked_by']}`. "
                        f"(Set CHLOE_VISION_BLOCKLIST to change.)")
            return f"/see failed: {r.get('error', 'unknown error')}"

        app = r.get("app") or {}
        title = (app.get("title") or "").strip()
        exe = (app.get("exe") or "").strip()
        header_bits = []
        if title:
            header_bits.append(title[:80])
        if exe and exe.lower() not in title.lower():
            header_bits.append(f"({exe})")
        header = " ".join(header_bits) or "active screen"
        return f"**{header}**\n\n{r['text']}"

    if msg.startswith("/ingest_screen ") or msg == "/ingest_screen":
        # Args: optional `--delay N` flag plus required slug. Either order:
        #   /ingest_screen --delay 5 my_slug
        #   /ingest_screen my_slug --delay 5
        raw_args = msg[len("/ingest_screen"):].strip().split()
        delay = 0
        rest = []
        i = 0
        while i < len(raw_args):
            tok = raw_args[i]
            if tok in ("--delay", "-d") and i + 1 < len(raw_args):
                try:
                    delay = int(raw_args[i + 1])
                except ValueError:
                    return (f"Bad `--delay` value: `{raw_args[i+1]}` "
                            f"(must be an integer 0-30)")
                i += 2
                continue
            rest.append(tok)
            i += 1
        slug = " ".join(rest).strip()
        return handle_ingest_screen(slug, delay=delay)

    if msg.startswith("/ask ") or msg == "/ask":
        return handle_ask(msg[len("/ask"):].strip())

    if msg.startswith("/recall "):
        # Demo-visibility hook over ChloeMemory.search_turns. Shows what
        # the semantic recall layer would surface for a given query —
        # same code path the chat handler uses when looks_like_recall_query
        # fires, just with min_age_hours=0 so today's turns are included.
        text = msg[len("/recall "):].strip()
        if not text:
            return "Usage: `/recall <query>`"
        # Late import — avoids a circular import at module load.
        try:
            from jarvis import _memory  # type: ignore
        except Exception as e:
            return f"/recall unavailable: {e}"
        if _memory is None:
            return "/recall: memory not initialized yet"
        # min_age_hours=0.25 (15 min) skips the trivially-recent self-
        # reference case where the user's just-typed /recall question and
        # Chloe's just-emitted response would otherwise surface as the top
        # hits. Anything older than that is genuinely historical and worth
        # showing. Override by setting min_age_hours in source if needed.
        hits = _memory.search_turns(text, limit=10, min_age_hours=0.25)
        if not hits:
            return f"No matching turns for: _{text}_"
        import datetime as _dt
        now = _dt.datetime.now().timestamp()
        out = [f"**Top recall hits for**: _{text}_  ({len(hits)} of up to 10)\n"]
        for i, h in enumerate(hits, 1):
            age_s = max(0, now - float(h.get('ts', now)))
            if age_s < 60:
                age = f"{int(age_s)}s ago"
            elif age_s < 3600:
                age = f"{int(age_s/60)}m ago"
            elif age_s < 86400:
                age = f"{int(age_s/3600)}h ago"
            else:
                age = f"{int(age_s/86400)}d ago"
            snippet = (h.get('content', '') or '').replace('\n', ' ')
            if len(snippet) > 160:
                snippet = snippet[:157] + '...'
            role = h.get('role', '?')
            modality = h.get('modality', '?')
            out.append(f"{i}. **{role} · {modality} · {age}** — {snippet}")
        return "\n".join(out)

    if msg.startswith("/wiki_write ") or msg == "/wiki_write":
        # Topic-to-page: Chloe researches a topic herself, writes a wiki page,
        # runs the ingest pipeline. See handle_wiki_write for the full flow.
        raw_args = msg[len("/wiki_write"):].strip().split()
        dry_run = False
        rest = []
        for tok in raw_args:
            if tok in ("--dry-run", "--dryrun", "-n"):
                dry_run = True
            else:
                rest.append(tok)
        topic = " ".join(rest).strip()
        return handle_wiki_write(topic, dry_run=dry_run)

    if msg.startswith("/wiki_synth ") or msg == "/wiki_synth":
        # Brain-only synthesis (no web). Companion to /wiki_write.
        raw_args = msg[len("/wiki_synth"):].strip().split()
        dry_run = False
        rest = []
        for tok in raw_args:
            if tok in ("--dry-run", "--dryrun", "-n"):
                dry_run = True
            else:
                rest.append(tok)
        topic = " ".join(rest).strip()
        return handle_wiki_synth(topic, dry_run=dry_run)

    if msg == "/wiki_interview_done":
        return handle_wiki_interview_done()

    if msg.startswith("/wiki_interview ") or msg == "/wiki_interview":
        topic = msg[len("/wiki_interview"):].strip()
        return handle_wiki_interview(topic)

    if msg.startswith("/wiki "):
        # Semantic search over the wiki via WikiEmbeddingStore. Same
        # embedding pipeline (nomic-embed-text) as /recall, just pointed
        # at wiki pages instead of conversation turns. The watcher
        # (wiki_watcher.bat) keeps the store fresh whenever Ed edits
        # a page in Obsidian.
        text = msg[len("/wiki "):].strip()
        if not text:
            return "Usage: `/wiki <query>`"
        try:
            from wiki_embedding import get_store
        except Exception as e:
            return f"/wiki unavailable: {e}"
        try:
            store = get_store()
        except Exception as e:
            return f"/wiki unavailable: {e}"
        if store.count_embedded() == 0:
            return ("/wiki: no embedded pages yet. Run "
                    "`wiki_watcher.bat --once` to backfill.")
        hits = store.search(text, limit=5)
        if not hits:
            return f"No wiki pages matched: _{text}_"
        out = [f"**Top wiki hits for**: _{text}_  "
               f"({len(hits)} of up to 5, "
               f"corpus={store.count_embedded()})\n"]
        for i, h in enumerate(hits, 1):
            title = h.get('title') or h.get('path', '?')
            typ = h.get('type') or '?'
            path = h.get('path', '?')
            score = h.get('score', 0.0)
            snippet = (h.get('snippet') or '').strip()
            if len(snippet) > 160:
                snippet = snippet[:157] + '...'
            out.append(f"{i}. **{title}** · {typ} · "
                       f"score={score:.2f} · `{path}`\n"
                       f"   {snippet}")
        return "\n".join(out)

    if msg == "/status" or msg == "/health":
        return handle_status()

    if msg in ("/brain", "/help brain"):
        return ("**Brain commands:**\n"
                "  `/ingest [--dry-run] <filename>` - ingest from "
                f"`{BRAIN.raw_dir}` (--dry-run previews without writing)\n"
                "  `/query <question>`      - search the wiki\n"
                "  `/recall <query>`        - semantic search over conversation history\n"
                "  `/wiki <query>`          - semantic search over wiki pages\n"
                "  `/web_history [today|week|month]` - list recent Brave search results\n"
                "  `/wiki_write [--dry-run] <topic>` - research a topic via web "
                "search, write a wiki page, ingest it\n"
                "  `/add <type> <slug> <body>` - manually add an entity or concept page\n"
                "  `/overview [slug...]`    - generate 2-voice podcast script from sources\n"
                "  `/podcast [pattern]`     - render the most recent overview script to audio\n"
                "  `/lint`                  - health-check the wiki\n"
                "  `/fact <statement>`      - save a durable user fact\n"
                "  `/see [prompt]`          - describe what's on screen (vision)\n"
                "  `/see ambient on [N]`    - periodic captures to episodic memory\n"
                "  `/ingest_screen [--delay N] <slug>` - capture → save → ingest (delay lets you alt-tab)\n"
                "  `/ask <question>`        - vision-augmented query (screen + wiki)\n"
                "  `/status` / `/health`    - self-awareness snapshot (queue, scheduled tasks, runtime)\n"
                "  `/summarize_old [N|--dry-run]` - roll up old turns into a wiki episodic summary\n"
                "  `/capabilities [section]` - self-analysis snapshot (slashes/tools/jobs/env/modules)\n"
                "  `/explain <module>`      - ast-introspect a jarvis module\n"
                "  `/apply_proposal <slug> [--dry-run|--list]` - apply a code proposal (self-mod)\n"
                "  `/revert_proposal <slug>` - restore a target from its proposal backup\n"
                "  `/issue_apply_token [--applies N] [--minutes M]` - mint a Tier-2 confirm token\n"
                "  `/pending_confirms [cancel <slug>|cancel-all]` - list Stage-3 pending confirms\n"
                "  `/autonomous [on|off|freeze N|reset|history|run-now]` - Stage-4 autonomous self-mod\n"
                "  `/brain`                 - this help message")

    if msg.startswith("/web_history") or msg == "/web_history":
        # List Brave search results persisted to wiki/sources/web_*.md.
        # Window args: today | week | month. Default: all-time, capped at 20.
        # Companion to the persist hook in jarvis._persist_brave_to_wiki —
        # together they make web lookups reviewable + non-volatile.
        arg = msg[len("/web_history"):].strip().lower()
        import datetime as _dt
        import re as _re_wh
        sources_dir = BRAIN.wiki_dir / "sources"
        if not sources_dir.exists():
            return "No web history yet - no `wiki/sources/` directory."
        pages = sorted(sources_dir.glob("web_*.md"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not pages:
            return "No web searches recorded yet."
        now = _dt.datetime.now()
        if arg in ("today", "1d"):
            cutoff = now.replace(hour=0, minute=0,
                                 second=0, microsecond=0).timestamp()
            pages = [p for p in pages if p.stat().st_mtime >= cutoff]
            window_label = "today"
        elif arg in ("week", "7d"):
            cutoff = (now - _dt.timedelta(days=7)).timestamp()
            pages = [p for p in pages if p.stat().st_mtime >= cutoff]
            window_label = "last 7 days"
        elif arg in ("month", "30d"):
            cutoff = (now - _dt.timedelta(days=30)).timestamp()
            pages = [p for p in pages if p.stat().st_mtime >= cutoff]
            window_label = "last 30 days"
        else:
            window_label = "all-time"
        if not pages:
            return f"No web searches in {window_label}."
        pages = pages[:20]
        out = [f"**Web search history** ({window_label}, "
               f"{len(pages)} of up to 20):\n"]
        for i, p in enumerate(pages, 1):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")[:1500]
            except Exception:
                continue
            query = ""
            date = ""
            m = _re_wh.search(r"^query:\s*['\"]?(.+?)['\"]?\s*$",
                              text, _re_wh.M)
            if m:
                query = m.group(1).strip()
            m = _re_wh.search(r"^date:\s*(.+)$", text, _re_wh.M)
            if m:
                date = m.group(1).strip()
            cite_m = _re_wh.search(r"^\d+\.\s+\[(.+?)\]\((.+?)\)",
                                   text, _re_wh.M)
            cite = ""
            if cite_m:
                title = cite_m.group(1)[:50]
                cite = f" - [{title}]({cite_m.group(2)})"
            if not query:
                stem = p.stem.replace("web_", "")
                query = stem.rsplit("_", 1)[0].replace("_", " ")
            out.append(f"{i}. *{date}* - **{query}**{cite}")
        return "\n".join(out)

    if msg.startswith("/summarize_old"):
        # Pillar 4 — compress oldest unsummarized turns into a wiki
        # episodic summary page. See handle_summarize_old docstring.
        return handle_summarize_old(msg[len("/summarize_old"):].strip())

    if msg.startswith("/capabilities") or msg == "/capabilities":
        # Self-analysis: slash commands, MCP tools, scheduled jobs, env
        # knobs, modules. Bounded — no LLM calls. See handle_capabilities.
        return handle_capabilities(msg[len("/capabilities"):].strip())

    if msg.startswith("/explain ") or msg == "/explain":
        # Ast-introspect any module under jarvis/. See handle_explain.
        return handle_explain(msg[len("/explain"):].strip())

    if msg.startswith("/apply_proposal"):
        # Tier 1 self-modification (2026-05-19). Apply a code proposal
        # from proposals/code_<date>_<slug>.md. See handle_apply_proposal.
        return handle_apply_proposal(msg[len("/apply_proposal"):].strip())

    if msg.startswith("/revert_proposal"):
        # Restore a previously-applied proposal's target from its backup.
        return handle_revert_proposal(msg[len("/revert_proposal"):].strip())

    if msg.startswith("/issue_apply_token"):
        # Tier 2 self-modification (2026-05-19 evening). Mint a confirm-
        # token so Chloe / Cowork jobs can apply proposals without Ed
        # retyping the slash. See handle_issue_apply_token.
        return handle_issue_apply_token(
            msg[len("/issue_apply_token"):].strip())

    if msg.startswith("/pending_confirms") or msg == "/pending_confirms":
        # Stage 3 self-modification (2026-05-19): list/cancel pending
        # voice/chat-confirm slots. Resolution itself happens earlier
        # in this function on every non-slash user turn.
        return handle_pending_confirms(
            msg[len("/pending_confirms"):].strip())

    if msg.startswith("/autonomous") or msg == "/autonomous":
        # Stage 4 self-modification (2026-05-19 night): enable/disable
        # the autonomous fix-recurring-errors proposer + watchdog.
        return handle_autonomous(msg[len("/autonomous"):].strip())

    return None
