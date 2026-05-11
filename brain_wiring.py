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
try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment]

from brain import Brain


# ─── Config ─────────────────────────────────────────────────────────────────
# CRITICAL: env vars are read LAZILY at call time, NOT at import time.
# jarvis.py imports brain_wiring before it loads .env via dotenv, so eager
# os.environ.get() at module load reads empty strings and the Groq client
# silently never initializes. Don't change this without re-checking that
# load order. See [brain] log lines for the smoking gun.
BRAIN_ROOT = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")

# Match jarvis.py's MODEL_TEXT — heavy = strongest model on Groq.
MODEL_HEAVY = "llama-3.3-70b-versatile"

# Lazy singletons. Initialized on first call once .env is loaded.
_groq = None
_groq_init_attempted = False


def _get_groq():
    """Lazy Groq client init. Reads GROQ_API_KEY on first call (after .env
    has loaded) and caches the result. Returns None if no key or library."""
    global _groq, _groq_init_attempted
    if _groq is not None:
        return _groq
    if _groq_init_attempted:
        return None  # already tried and failed; don't keep retrying
    _groq_init_attempted = True
    if Groq is None:
        return None
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    _groq = Groq(api_key=key)
    print("[brain] Groq client initialized lazily", flush=True)
    return _groq


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2:3b").strip()


# ─── LLM adapters ───────────────────────────────────────────────────────────
def _heavy_call(prompt: str) -> str:
    """One-shot Groq completion — no history, no tool-calling. For ingest,
    lint, and page synthesis where we want strong reasoning and predictable
    cost.

    Falls back to light if Groq unavailable. Quality drops noticeably but
    keeps the brain operational during quota outages."""
    client = _get_groq()
    if not client:
        print("[brain] no Groq key at call time — heavy ops falling back to light", flush=True)
        return _light_call(prompt)
    try:
        resp = client.with_options(timeout=120.0).chat.completions.create(
            model=MODEL_HEAVY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3500,  # bumped from 2000 to fit longer outputs (overview scripts)
            temperature=0.3,  # synthesis tasks want lower variance than chat
        )
        content = (resp.choices[0].message.content or "").strip()
        # Groq sometimes returns 200 OK with empty content (content filter,
        # truncation to zero tokens, certain quota states). Treat that as a
        # failure and fall back to Ollama instead of silently returning "".
        if not content:
            finish_reason = getattr(resp.choices[0], 'finish_reason', 'unknown')
            print(f"[brain] heavy (groq) returned empty content "
                  f"(finish_reason={finish_reason}) — falling back to light",
                  flush=True)
            return _light_call(prompt)
        return content
    except Exception as e:
        print(f"[brain] heavy (groq) failed: {type(e).__name__}: {e} — "
              f"falling back to light", flush=True)
        return _light_call(prompt)


def _light_call(prompt: str) -> str:
    """One-shot Ollama completion. For query selection, fact extraction,
    and any other light synthesis."""
    try:
        import requests
        r = requests.post(
            f"{_ollama_url()}/api/chat",
            json={
                "model":    _ollama_model(),
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.3, "num_predict": 1500},
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

# Concurrency cap: at most one extraction in flight at a time. Prevents
# pile-up if the user types a flurry of messages.
_AUTO_FACT_RUNNING = _threading.Lock()

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
    if not _AUTO_FACT_RUNNING.acquire(blocking=False):
        return  # another extraction already in flight; drop this one
    try:
        try:
            slug = BRAIN.fact_extract_and_add(msg)
            if slug:
                print(f"[auto-fact] saved: {slug}", flush=True)
        except Exception as e:
            print(f"[auto-fact] extract failed: {type(e).__name__}: {e}",
                  flush=True)
    finally:
        _AUTO_FACT_RUNNING.release()


def _maybe_auto_extract(msg: str) -> None:
    """Fire-and-forget background extractor. Always returns None."""
    if _should_skip_auto_fact(msg):
        return
    t = _threading.Thread(target=_auto_fact_worker, args=(msg,),
                          name="chloe-auto-fact", daemon=True)
    t.start()


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


def try_handle_brain_command(user_text: str):
    """Synchronous brain command handler. Returns reply string or None.

    Wrap in asyncio.to_thread() when calling from an async context — these
    operations issue HTTP calls and may take a few seconds for ingest/lint.

    Side effect: for non-command chat messages, fires the auto-fact extractor
    on a daemon thread. Always returns None for non-commands; chat path
    continues normally.
    """
    msg = (user_text or "").strip()

    # Auto-fact extraction for plain chat (not slash commands). Fire-and-forget.
    if msg and not msg.startswith("/"):
        _maybe_auto_extract(msg)

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

    if msg in ("/brain", "/help brain"):
        return ("**Brain commands:**\n"
                "  `/ingest [--dry-run] <filename>` - ingest from "
                f"`{BRAIN.raw_dir}` (--dry-run previews without writing)\n"
                "  `/query <question>`      - search the wiki\n"
                "  `/add <type> <slug> <body>` - manually add an entity or concept page\n"
                "  `/overview [slug...]`    - generate 2-voice podcast script from sources\n"
                "  `/podcast [pattern]`     - render the most recent overview script to audio\n"
                "  `/lint`                  - health-check the wiki\n"
                "  `/fact <statement>`      - save a durable user fact\n"
                "  `/see [prompt]`          - describe what's on screen (vision)\n"
                "  `/see ambient on [N]`    - periodic captures to episodic memory\n"
                "  `/ingest_screen [--delay N] <slug>` - capture → save → ingest (delay lets you alt-tab)\n"
                "  `/ask <question>`        - vision-augmented query (screen + wiki)\n"
                "  `/brain`                 - this help message")

    return None
