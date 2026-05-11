"""
jarvis.py — Backend for the CHLOE HUD.

Two paths share one brain:

  1. CHAT PATH (text input from HUD)
       HUD → WebSocket → handle_chat() → Groq → stream deltas back to HUD
       The HUD speaks the reply with browser TTS.

  2. VOICE PATH (always running in background)
       Mic → OpenWakeWord ("hey jarvis") → broadcast "listening"
       Mic → record until silence → Groq Whisper → transcribe
       transcript → Groq chat → reply
       broadcast "speaking" → edge-tts → play audio → broadcast "idle"

Both paths share `_voice_history` so Chloe remembers across modalities.

Wire-protocol with the HUD (existing WebSocket on ws://localhost:6789):

  HUD → backend  : {"type": "chat", "messages": [...], "system": "..."}
  backend → HUD  : {"type": "start" | "delta" | "done" | "error", ...}

Plain state strings ("idle"|"listening"|"thinking"|"speaking") are also broadcast
so the HUD ring animates correctly during the voice path too.
"""

import asyncio
import base64
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import random
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import AsyncGroq, Groq
from brain_wiring import BRAIN, try_handle_brain_command
import lights as _lights
from lights import try_handle_lights_command

# Used for the URL-attachment feature in chat (auto-detect http(s) links in the
# user message, fetch them server-side, prepend the page text to the prompt).
# beautifulsoup4 is the only new hard dep beyond what was already here.
try:
    import requests as _requests
    from bs4 import BeautifulSoup as _BeautifulSoup
    _URL_FETCH_AVAILABLE = True
except ImportError:
    _requests = None
    _BeautifulSoup = None
    _URL_FETCH_AVAILABLE = False


import hud_server

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Look for env files in priority order. python-dotenv's default load_dotenv()
# only finds ".env" — but this project also uses "_env"/"env" (per COMMANDS.md),
# so we have to point it explicitly.
#
# When running as a PyInstaller-frozen exe, __file__ points into the bundled
# extraction temp dir (sys._MEIPASS) — but user data (env, facts.md, models/,
# sounds/, kokoro_models/, chloe_memory.db) lives next to the exe itself.
# start_jarvis.py sets cwd to the exe's directory, so we use cwd when frozen.
if getattr(sys, "frozen", False):
    _THIS_DIR = Path.cwd()
else:
    _THIS_DIR = Path(__file__).resolve().parent
for _candidate in (".env", "_env", "env"):
    _p = _THIS_DIR / _candidate
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=False)
        print(f"[chloe] loaded env from: {_p.name}")

GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "").strip()
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "gWVE9uaEr9AGwZO3wYSo").strip()
ELEVENLABS_MODEL    = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5").strip()
# ElevenLabs is opt-in to avoid burning credits. Set USE_ELEVENLABS=1 in _env
# to enable it; otherwise the free edge-tts path is used for everything.
USE_ELEVENLABS = os.environ.get("USE_ELEVENLABS", "").strip().lower() in ("1", "true", "yes", "on")

# Use Groq's Compound system for everything — it has built-in web search +
# code execution and handles the agentic tool loop server-side. Much simpler
# than calling Tavily directly + orchestrating tool calls in our code.
# `compound-mini` is ~3x faster than `compound`, ideal for voice latency.
# Set CHLOE_USE_COMPOUND=0 in env to fall back to plain Llama-3.3 (no search).
# Two-tier model strategy:
#   MODEL_TEXT   = fast everyday chat (high TPM limit, no web search)
#   MODEL_SEARCH = compound-mini (real-time data, lower TPM limit)
# A small router heuristic decides which to use per turn so we don't burn
# through compound-mini's quota on small talk like "how are you".
# Set CHLOE_USE_COMPOUND=0 to disable compound entirely (chat will lose web
# search, but Chloe won't hit GPT-OSS-120B rate limits).
USE_COMPOUND    = (os.environ.get("CHLOE_USE_COMPOUND") or os.environ.get("JARVIS_USE_COMPOUND", "1")).strip() != "0"  # legacy JARVIS_USE_COMPOUND honored
MODEL_TEXT      = "llama-3.3-70b-versatile"        # fast path
MODEL_SEARCH    = "groq/compound-mini"             # web-search-capable
MODEL_VISION    = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_STT       = "whisper-large-v3-turbo"

# ─── LOCAL LLM FALLBACK (Ollama) ─────────────────────────────────────────────
# Ollama runs as a separate Go binary — install from https://ollama.com,
# then `ollama pull llama3.2:3b` (small + fast on CPU, ~2GB) or
# `ollama pull llama3.1:8b` (better quality, needs ~8GB RAM, ~5GB disk).
#
# Chloe falls back here when Groq fails or the network is out, so she keeps
# working offline. Quality is meaningfully lower than llama-3.3-70b on Groq,
# but you stay alive.
#
# Set CHLOE_OLLAMA_FALLBACK=0 in _env to disable. OLLAMA_URL / OLLAMA_MODEL
# override defaults.
OLLAMA_URL              = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL            = os.environ.get("OLLAMA_MODEL", "llama3.2:3b").strip()
OLLAMA_FALLBACK_ENABLED = os.environ.get("CHLOE_OLLAMA_FALLBACK", "1").strip() != "0"
# Local-first routing: when on (default), Ollama handles everyday chat and
# Groq compound-mini is reserved for queries that explicitly need real-time
# web data. Saves Groq quota for what only Groq can do (search).
# Set CHLOE_OLLAMA_PRIMARY=0 in _env to fall back to the original cloud-first
# routing (Groq fast Llama → hedge-retry → compound → Ollama on failure).
OLLAMA_PRIMARY          = os.environ.get("CHLOE_OLLAMA_PRIMARY", "1").strip() != "0"
# Cached after each probe — avoids hammering /api/tags on every turn.
# Tuple (bool, timestamp) so we can re-probe after _OLLAMA_PROBE_TTL seconds.
# Without the TTL, starting Ollama mid-session would never be detected.
_OLLAMA_PROBE_TTL = 60.0  # seconds before re-probing the daemon
_ollama_available_cache = None  # type: ignore[var-annotated]

# Voice loop config
# Wake-word phrase. By default this is "hey jarvis" — the only phrase that
# openwakeword ships pretrained. To get her to respond to "hey Chloe":
#   1. Train a custom openwakeword model on "hey Chloe" (see WAKE_WORD_TRAINING.md)
#   2. Drop the resulting .onnx file at  models/hey_chloe.onnx  inside this folder
# This block auto-detects that file. No code edits needed once the model is in
# place — just restart Chloe.
# Auto-detect ALL custom .onnx wake models in models/. Each one is a separate
# trained openwakeword model — Chloe will fire if ANY of them score above
# WAKE_THRESHOLD. Drop multiple files in to support multiple trigger phrases:
#   models/hey_chloe.onnx   → "hey Chloe"
#   models/chloe.onnx       → "Chloe" (alone)
#   models/yo_chloe.onnx    → "yo Chloe", etc.
# Each phrase needs its own trained model (re-run the Colab notebook with
# `target_word = "<phrase>"`). All loaded models are checked simultaneously
# with one predict() call per audio frame, so adding more is cheap.
_MODELS_DIR = _THIS_DIR / "models"
_CUSTOM_ONNX = sorted(_MODELS_DIR.glob("*.onnx")) if _MODELS_DIR.exists() else []
if _CUSTOM_ONNX:
    WAKE_WORD_PATHS = [str(p) for p in _CUSTOM_ONNX]
    WAKE_WORD_KEYS  = [p.stem for p in _CUSTOM_ONNX]   # ['hey_chloe', 'chloe', ...]
    WAKE_WORD_HUMAN = " / ".join(k.replace('_', ' ') for k in WAKE_WORD_KEYS)
    print(f"[chloe] custom wake model(s) detected ({len(WAKE_WORD_PATHS)}): "
          f"{[Path(p).name for p in WAKE_WORD_PATHS]}")
else:
    WAKE_WORD_PATHS = ["hey_jarvis"]
    WAKE_WORD_KEYS  = ["hey_jarvis"]
    WAKE_WORD_HUMAN = "hey jarvis"
    print("[chloe] no custom wake models in models/ — using built-in 'hey jarvis'")

# Picovoice Porcupine — alternative wake-word engine. If a .ppn keyword file
# exists in models/ AND PORCUPINE_ACCESS_KEY is set in .env, we use Porcupine
# instead of openwakeword. Easier path to a custom "Hey Chloe" trigger:
# Picovoice's web console generates the .ppn in minutes (vs Colab training).
# Free tier — sign up at console.picovoice.ai. Falls back to openwakeword
# automatically if either piece is missing.
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY", "").strip()
_PPN_DIR = _THIS_DIR / "models"
_PPN_FILES = sorted(_PPN_DIR.glob("*.ppn")) if _PPN_DIR.exists() else []
PORCUPINE_PPNS = [str(p) for p in _PPN_FILES]  # all .ppn files, all keywords active
USE_PORCUPINE = bool(PORCUPINE_ACCESS_KEY and PORCUPINE_PPNS)

# Voice-loop sensitivity knobs. All env-overridable so you can tune for your
# room/mic without editing code. Set in .env / _env / env:
#   CHLOE_WAKE_THRESHOLD = 0.5   # 0.0–1.0; lower = easier wake trigger
#   CHLOE_SILENCE_RMS    = 0.004 # below this RMS = silence; lower = catches quieter speech
#   CHLOE_MIC_GAIN       = 1.0   # software gain multiplier (1=off, 2=2x louder, etc)
#   CHLOE_GREETING       = 1     # 0 to skip the spoken greeting at startup
WAKE_THRESHOLD       = float(os.environ.get("CHLOE_WAKE_THRESHOLD", "0.5"))
SAMPLE_RATE          = 16000
CHUNK_SAMPLES        = 1280              # 80ms @ 16kHz
SILENCE_RMS          = float(os.environ.get("CHLOE_SILENCE_RMS", "0.004"))
MIC_GAIN             = float(os.environ.get("CHLOE_MIC_GAIN", "1.0"))
GREETING_ENABLED     = os.environ.get("CHLOE_GREETING", "1").strip() != "0"
BOOT_SOUND_ENABLED   = os.environ.get("CHLOE_BOOT_SOUND", "1").strip() != "0"
SILENCE_HANG_MS      = int(os.environ.get("CHLOE_SILENCE_HANG_MS", "2000"))
MIN_UTTERANCE_S      = 0.3               # discard recordings shorter than this
MAX_RECORD_S         = int(os.environ.get("CHLOE_MAX_RECORD_S", "60"))
_PTT_MAX_S           = int(os.environ.get("CHLOE_PTT_MAX_S", "300"))  # 5min safety cap on push-to-talk
LEADING_TRIM_SECS    = 0.2               # min trailing silence kept on the front of audio
PREROLL_SECS         = 0.15              # how much pre-voice padding to keep (helps Whisper)
VOICE_DEBUG          = True              # print RMS samples while recording

# Follow-up mode: after Chloe finishes speaking, leave the mic open for a
# brief window so the user can ask a follow-up without re-saying the wake
# word. Set CHLOE_FOLLOWUP=0 to disable. CHLOE_FOLLOWUP_S sets the listen
# window in seconds (default 5s).
FOLLOWUP_ENABLED     = os.environ.get("CHLOE_FOLLOWUP", "1").strip() != "0"
FOLLOWUP_LISTEN_S    = float(os.environ.get("CHLOE_FOLLOWUP_S", "5"))

# Mic device override. None = system default. Set via CHLOE_MIC env var:
#   set CHLOE_MIC=15  (matches a device index from the list printed at startup)
#   set CHLOE_MIC=Samson  (substring match against device name)
MIC_DEVICE_OVERRIDE = (os.environ.get("CHLOE_MIC") or os.environ.get("JARVIS_MIC", "")).strip() or None  # legacy JARVIS_MIC honored

# Edge TTS — the default free voice path for Chloe. Override via env var.
# Curated picks for a polished female AI-assistant feel:
#   en-US-AriaNeural             — warm American (default, recommended)
#   en-US-JennyNeural            — slightly more conversational American
#   en-US-AvaMultilingualNeural  — newer, very natural American
#   en-GB-SoniaNeural            — polished British female
#   en-GB-LibbyNeural            — younger, friendlier British female
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "").strip() or "en-US-AriaNeural"

# ─── KOKORO-ONNX LOCAL TTS ──────────────────────────────────────────────────
# Higher-quality offline TTS — meaningfully closer to ElevenLabs than Edge.
# 82M-parameter open-source model (Apache 2.0), runs via ONNX runtime so it
# works on Python 3.14 (no torch dependency).
#
# Setup:
#   1. python download_kokoro.py   (~330 MB, one-time)
#   2. pip install kokoro-onnx soundfile
#   3. set USE_KOKORO=1 in _env
#
# American female voices (af_*) — pick one for KOKORO_VOICE:
#   af_jessica  — young American, conversational, energetic (default)
#   af_heart    — warm/soft, very natural cadence (community favorite)
#   af_bella    — articulate, polished, "professional AI assistant"
#   af_sarah    — clear, neutral
#   af_nicole   — slightly higher energy, friendly
#   af_sky      — calmer, professional
# Plus several British female (bf_*) voices and male (am_*/bm_*) options.
USE_KOKORO         = os.environ.get("USE_KOKORO", "").strip().lower() in ("1", "true", "yes", "on")
KOKORO_DIR         = Path(os.environ.get("KOKORO_DIR", str(_THIS_DIR / "kokoro_models")))
KOKORO_MODEL_PATH  = Path(os.environ.get("KOKORO_MODEL_PATH", str(KOKORO_DIR / "kokoro-v1.0.onnx")))
KOKORO_VOICES_PATH = Path(os.environ.get("KOKORO_VOICES_PATH", str(KOKORO_DIR / "voices-v1.0.bin")))
KOKORO_VOICE       = os.environ.get("KOKORO_VOICE", "").strip() or "af_jessica"
KOKORO_SPEED       = float(os.environ.get("KOKORO_SPEED", "1.0"))

# Default system prompt for the voice path. The HUD chat path sends its own.
def _voice_system(model: str | None = None) -> str:
    """Build the voice path's system prompt. The prompt adapts to whether
    the chosen model can search the web (compound) or not (plain Llama)."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    can_search = (model == MODEL_SEARCH) if model else USE_COMPOUND
    if can_search:
        return (
            f"You are Chloe, a personal home assistant speaking to Ed via voice. "
            f"Today's date is {today} — you DO know the current date and should never "
            f"apologize about not knowing it.\n\n"
            f"You can search the web automatically when needed. For anything that may "
            f"have changed since training (current prices, weather, news, sports scores, "
            f"recent events, who currently holds a position), search the web and give "
            f"Ed the answer. For knowledge you already have (general facts, math, "
            f"conversation, advice, writing), answer directly without searching.\n\n"
            f"NEVER invent numbers or facts — if you can't find something, say so plainly.\n\n"
            f"STYLE:\n"
            f"- Reply in plain spoken sentences. No bullet points, markdown, or lists.\n"
            f"- Keep replies short, friendly, and conversational — usually one or two "
            f"sentences.\n"
            f"- Do NOT cite URLs or list sources unless Ed asks; he's listening, not reading."
        )
    return (
        f"You are Chloe, a personal home assistant speaking to Ed via voice. "
        f"Today's date is {today} — you DO know the current date and should never "
        f"apologize about not knowing it.\n\n"
        f"For this turn you do NOT have web search available. If Ed asks for current/live "
        f"data (prices, weather, news, scores, current officeholders), tell him plainly "
        f"that you'd need to look it up — don't invent the answer. For things you already "
        f"know (general knowledge, conversation, advice, writing, math), answer directly "
        f"without disclaimers.\n\n"
        f"TOOLS:\n"
        f"- You have a `grep_source` tool. CALL IT whenever Ed asks about your own "
        f"implementation, behaviour, configuration, or 'what does X do' / 'how do you Y' "
        f"questions about your code. Quoting actual lines is more useful than guessing "
        f"from memory. Pass a regex pattern (e.g., 'def handle_chat', 'CHLOE_MIC_GAIN'). "
        f"After the tool returns matches, summarise them naturally in spoken English — "
        f"don't read filenames or line numbers aloud unless Ed asks.\n"
        f"- You have a Bitcoin Lightning wallet. Tools: `wallet_balance`, "
        f"`wallet_invoice`, `wallet_send`, `wallet_history`. Speak amounts in "
        f"sats. For `wallet_send`, ALWAYS require Ed to give you a PIN this "
        f"turn — never invent or reuse a previous PIN. If he hasn't given "
        f"one, ask for it BEFORE calling the tool. The system enforces a "
        f"daily spend cap server-side; if a send is refused, relay the "
        f"reason and stop.\n\n"
        f"STYLE:\n"
        f"- Reply in plain spoken sentences. No bullet points, markdown, or lists.\n"
        f"- Keep replies short, friendly, and conversational — usually one or two "
        f"sentences."
    )

if not GROQ_API_KEY:
    print("[chloe] WARNING: GROQ_API_KEY not set — chat & STT will fail")
if USE_ELEVENLABS and ELEVENLABS_API_KEY:
    print(f"[chloe] TTS: ElevenLabs (voice={ELEVENLABS_VOICE_ID}, model={ELEVENLABS_MODEL})")
elif USE_ELEVENLABS and not ELEVENLABS_API_KEY:
    print(f"[chloe] TTS: edge-tts (voice={EDGE_TTS_VOICE}) — USE_ELEVENLABS=1 set but ELEVENLABS_API_KEY missing")
elif USE_KOKORO:
    print(f"[chloe] TTS: Kokoro local (voice={KOKORO_VOICE}, model={KOKORO_MODEL_PATH.name})")
else:
    print(f"[chloe] TTS: edge-tts (voice={EDGE_TTS_VOICE}) — "
          f"set USE_KOKORO=1 for local Kokoro or USE_ELEVENLABS=1 for ElevenLabs")
if USE_COMPOUND:
    print(f"[chloe] tools: ENABLED via groq/compound-mini for real-time queries (router decides per turn)")
    print(f"[chloe]        small-talk uses {MODEL_TEXT} (faster, no rate limit issues)")
else:
    print(f"[chloe] tools: DISABLED (using {MODEL_TEXT} for everything, no real-time data)")
if USE_PORCUPINE:
    print(f"[chloe] wake: Porcupine ready ({len(PORCUPINE_PPNS)} .ppn file(s))")
elif PORCUPINE_ACCESS_KEY and not PORCUPINE_PPNS:
    print(f"[chloe] wake: PORCUPINE_ACCESS_KEY set but no .ppn files in models/ — using openwakeword")
elif PORCUPINE_PPNS and not PORCUPINE_ACCESS_KEY:
    print(f"[chloe] wake: .ppn files found but PORCUPINE_ACCESS_KEY not set in .env — using openwakeword")
print(f"[chloe] sensitivity: wake_threshold={WAKE_THRESHOLD}  silence_rms={SILENCE_RMS}  mic_gain={MIC_GAIN}x")
print(f"[chloe] timing:      silence_hang={SILENCE_HANG_MS}ms  max_record={MAX_RECORD_S}s  ptt_max={_PTT_MAX_S}s")
print(f"[chloe] startup:     greeting={'on' if GREETING_ENABLED else 'off'}  boot_sound={'on' if BOOT_SOUND_ENABLED else 'off'}")

_async_groq = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
_sync_groq  = Groq(api_key=GROQ_API_KEY)      if GROQ_API_KEY else None

# ─── MODE TOGGLE ────────────────────────────────────────────────────────────
# CHLOE_MODE picks a tone + (optional) per-mode facts file. Default "home"
# (warm/casual). "office" uses a more professional/concise tone. Custom
# mode strings work too — they just won't have a built-in tone block.
CHLOE_MODE = os.environ.get("CHLOE_MODE", "home").strip().lower() or "home"

# Per-mode tone addendum injected into every system prompt. Memory + recall
# behavior is identical across modes; this only shifts how Chloe phrases
# herself.
_MODE_TONE_BLOCKS = {
    "home": (
        "MODE: HOME. Tone: warm, casual, conversational. Informal phrasing "
        "and friendly back-and-forth are fine when they fit the moment."
    ),
    "office": (
        "MODE: OFFICE. Tone: professional, concise, direct. Keep replies "
        "tight and businesslike. Skip pleasantries unless asked."
    ),
}


def _mode_block() -> str:
    block = _MODE_TONE_BLOCKS.get(CHLOE_MODE, "")
    if not block:
        return ""
    return f"\n\n## Mode tone:\n{block}"


# ─── PERSISTENT MEMORY ──────────────────────────────────────────────────────
# Three-layer memory: SQLite turn log, markdown facts file, and FTS5 semantic
# recall. See chloe_memory.py for the implementation. Pure stdlib, no extra
# packages.
from chloe_memory import (
    ChloeMemory,
    parse_remember,
    parse_remember_about,
    looks_like_recall_query,
    format_recall_block,
    format_facts_block,
    format_about_block,
)
_MEMORY_DB  = _THIS_DIR / "chloe_memory.db"
# Mode-aware facts file. If facts_<mode>.md exists, use that; otherwise
# fall back to the shared facts.md. Lets you keep distinct fact sets for
# home vs office (or any custom mode) without forking the whole project.
_FACTS_FILE = _THIS_DIR / f"facts_{CHLOE_MODE}.md"
if not _FACTS_FILE.exists():
    _FACTS_FILE = _THIS_DIR / "facts.md"
# Self-knowledge file. Same shape as facts.md but describes Chloe's own
# architecture, capabilities, and limitations. Always injected into the
# system prompt so introspection questions get concrete answers.
_ABOUT_FILE = _THIS_DIR / "chloe_about.md"
_memory     = ChloeMemory(_MEMORY_DB, _FACTS_FILE, about_path=_ABOUT_FILE)
print(f"[chloe] mode={CHLOE_MODE}  memory: db={_MEMORY_DB.name}  "
      f"facts={_FACTS_FILE.name}  about={_ABOUT_FILE.name}  "
      f"turns_logged={_memory.turn_count()}")

# Conversation history shared between voice and HUD-text paths.
# Each entry: {"role": "user"|"assistant", "content": str}
_HISTORY_MAX = 20  # keep last N turns to limit token cost
_voice_history: list[dict] = []
# Hydrate from the SQLite log so Chloe picks up where she left off across
# restarts. We keep just role+content in memory; modality stays in the DB.
try:
    _hydrated = _memory.recent_turns(n=_HISTORY_MAX)
    _voice_history.extend(
        {"role": h["role"], "content": h["content"]} for h in _hydrated
    )
    if _hydrated:
        print(f"[chloe] hydrated {len(_hydrated)} turn(s) from previous sessions")
except Exception as e:
    print(f"[chloe] memory hydration error: {e}")

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _needs_vision(messages):
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                    return True
    return False

def _to_groq_messages(messages):
    """Anthropic-flavored content arrays → Groq/OpenAI shape."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": role, "content": c}); continue
        if not isinstance(c, list): continue
        groq_blocks = []
        for block in c:
            if not isinstance(block, dict): continue
            btype = block.get("type")
            if btype == "text":
                groq_blocks.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                src = block.get("source", {})
                if src.get("type") == "base64":
                    mt = src.get("media_type", "image/jpeg")
                    data = src.get("data", "")
                    groq_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mt};base64,{data}"},
                    })
            elif btype == "image_url":
                groq_blocks.append(block)
        if all(b.get("type") == "text" for b in groq_blocks):
            out.append({"role": role, "content": "".join(b["text"] for b in groq_blocks)})
        else:
            out.append({"role": role, "content": groq_blocks})
    return out

async def _ws_send(ws, obj):
    """Best-effort JSON send over a websocket."""
    try:
        await ws.send(json.dumps(obj))
    except Exception:
        pass


async def _ws_broadcast(obj):
    """Send obj to every currently-connected client. Used for wallet
    responses where the originating socket may have closed during the
    1-2s SDK call (Tailscale-serve flakiness on browser WS connections).
    Broadcasting reaches the PWA whichever connection it currently has."""
    msg = json.dumps(obj)
    clients = list(hud_server.hud_clients)
    if not clients:
        return
    results = await asyncio.gather(
        *[c.send(msg) for c in clients],
        return_exceptions=True,
    )
    sent = sum(1 for r in results if not isinstance(r, Exception))
    print(f"[chloe] broadcast {obj.get('type')} → {sent}/{len(clients)} clients",
          flush=True)


# ─── Streaming TTS for the chat path (opt-in via CHLOE_TTS_STREAMING=1) ─────
# The default chat path buffers the whole reply, then synthesizes, then
# broadcasts. For long replies that's a 5+ second wait before any audio plays.
# When this flag is on, _reply_audio_or_speak routes to _reply_audio_chunked
# which splits the reply into sentences (via the existing _split_sentences_for_tts
# helper used by the voice/PTT path), synthesizes each in order, and broadcasts
# `tts_audio_chunk` messages so the HUD can start playing the first sentence
# while later ones are still being synthesized. Drops time-to-first-audio from
# ~5s to ~500ms on long replies.
#
# Off by default so the demo recording path is untouched. Set to 1 only after
# verifying the chunked path doesn't regress anything.
TTS_STREAMING = os.environ.get("CHLOE_TTS_STREAMING", "0").strip() == "1"


async def _reply_audio_chunked(reply: str, *, label: str = "chat"):
    """Streaming-TTS variant of the reply_audio path. Synthesizes the reply
    one sentence at a time and broadcasts each as a `tts_audio_chunk`. The HUD
    queues chunks and plays them sequentially through the same AnalyserNode
    used for single-shot tts_audio, so amplitude reactivity still works.

    Each chunk message:
        {type: "tts_audio_chunk", chunk_id, total_chunks, is_final,
         format, audio_b64, text}

    Caller still gets the same await-until-done semantics as
    _reply_audio_or_speak's single-shot branch — this returns when every
    chunk has been broadcast. (HUD-side playback may continue after; the
    finally-clause "idle" broadcast in handle_chat ends up filtered by the
    HUD's expectingAudio guard until TtsAudio's onFinalEnd fires.)"""
    sentences = _split_sentences_for_tts(reply)
    if not sentences:
        return
    total = len(sentences)
    for idx, sent in enumerate(sentences):
        try:
            result = await asyncio.to_thread(_synthesize_tts_bytes, sent)
            if result is None:
                print(f"[chloe] {label} chunked TTS: synth failed on "
                      f"chunk {idx + 1}/{total}", flush=True)
                continue
            audio_bytes, fmt = result
            ab64 = base64.b64encode(audio_bytes).decode("ascii")
            await _ws_broadcast({
                "type":         "tts_audio_chunk",
                "chunk_id":     idx,
                "total_chunks": total,
                "is_final":     (idx == total - 1),
                "format":       fmt,
                "audio_b64":    ab64,
                "text":         sent,
            })
            print(f"[chloe] {label}: chunk {idx + 1}/{total} "
                  f"({len(audio_bytes)} bytes, {fmt})", flush=True)
        except Exception as e:
            print(f"[chloe] {label} chunked TTS error on chunk "
                  f"{idx + 1}/{total}: {e}", flush=True)


async def _reply_audio_or_speak(reply: str, data: dict, *, label: str = "chat"):
    """Route Chloe's spoken reply based on the inbound `reply_audio` flag.

    reply_audio=True   → synth TTS to bytes (no local playback) and broadcast
                         a `tts_audio` message to every connected WS client.
                         The PWA's <audio> element plays it. Broadcast (not
                         point-to-point) because Tailscale-served browser WS
                         connections can swap between request and TTS-finish
                         — same bug class the wallet hit.

                         When CHLOE_TTS_STREAMING=1 is set, this routes
                         through _reply_audio_chunked instead — same effect
                         on the HUD via the new tts_audio_chunk handler,
                         drastically lower TTFB on long replies.
    reply_audio falsy  → original behavior: speak on PC speakers via _speak().

    Lives at module scope so handle_chat, handle_ptt_audio, and any future
    reply path can share it without duplicating the bytes/broadcast dance.
    `label` only affects log lines."""
    if data.get("reply_audio"):
        if TTS_STREAMING:
            try:
                await _reply_audio_chunked(reply, label=label)
                return
            except Exception as e:
                # If chunked path crashes, fall through to single-shot so
                # the user still hears something — better degraded than mute.
                print(f"[chloe] {label} chunked TTS crashed, falling back "
                      f"to single-shot: {e}", flush=True)
        try:
            result = await asyncio.to_thread(_synthesize_tts_bytes, reply)
            if result is None:
                print(f"[chloe] {label} reply_audio: all TTS engines failed",
                      flush=True)
                await _ws_broadcast({"type": "error", "text": "TTS failed"})
                return
            audio_bytes, fmt = result
            ab64 = base64.b64encode(audio_bytes).decode("ascii")
            await _ws_broadcast({
                "type": "tts_audio",
                "format": fmt,           # "mp3" or "wav"
                "audio_b64": ab64,
                "text": reply,
            })
            print(f"[chloe] {label}: streamed {len(audio_bytes)} bytes "
                  f"({fmt}) → PWA", flush=True)
        except Exception as e:
            print(f"[chloe] {label} reply_audio synth error: {e}", flush=True)
            await _ws_broadcast({"type": "error",
                                 "text": f"TTS synth: {e}"})
    else:
        await asyncio.to_thread(_speak, reply)


# Words/phrases that suggest the user wants live data. If any appears in their
# message we route to compound-mini; otherwise we use the fast Llama path. This
# is a heuristic — there are obviously real-time questions that don't contain
# any of these words ("is it raining" without "currently/now"), so it'll miss
# some. Tradeoff: compound-mini's TPM limit is small enough that being modest
# with it is worth the occasional miss. The user can always rephrase.
_REALTIME_KEYWORDS = (
    # Time-sensitivity markers
    "current", "currently", "today", "tonight", "tomorrow", "yesterday",
    "now", "right now", "as of now", "at the moment", "as of today",
    "this week", "this month", "this year",
    "this morning", "this afternoon", "this evening",
    "latest", "most recent", "recent", "recently", "live", "live score",
    # Financial / markets — broadened so phrasings like "what's Apple worth"
    # or "how is Tesla trading" don't slip through to the no-search path.
    "price", "cost", "costs", "worth", "value of", "valuation",
    "trading", "trade at", "trades at", "going for",
    "stock", "stocks", "share price", "ticker", "shares",
    "market cap", "market value", "earnings", "dividend", "yield",
    "buy or sell", "should i buy", "should i sell", "investment advice",
    "crypto", "bitcoin", "ethereum", "eth", "btc",
    "exchange rate", "interest rate", "fed rate", "mortgage rate",
    # Weather
    "weather", "forecast", "temperature", "raining", "snowing", "humid",
    "humidity", "is it hot", "is it cold", "is it raining",
    # News / events
    "news", "headline", "headlines", "breaking",
    "what happened", "what's happening", "happening now",
    # Sports — phrases must be sports-specific. Bare words like "score",
    # "match", "won", "winning", "leading" false-positive on common
    # questions ("Hans Zimmer score", "I won the lottery", "leading
    # cause", "music match"). Use disambiguated phrases instead.
    "final score", "the score is", "game score", "scored",
    "playoff", "playoffs", "the match", "match tonight", "match today",
    "game tonight", "game today",
    "who won", "did they win", "winning team", "currently winning",
    "who's leading", "team is leading", "currently leading",
    # Politics / officeholders
    "election", "polls", "polling",
    "who is", "who's the", "who is the", "ceo", "president", "prime minister",
    "currently holds", "current ceo", "current president",
    # Direct lookup / explicit search signals
    "look up", "look that up", "search for", "search the web",
    "google", "find me", "lookup", "find the latest", "check the web",
)


# Phrases the user uses when asking about Chloe's own implementation. These
# trigger introspection routing — force Groq MODEL_TEXT, where llama-3.3-70b's
# tool calling on grep_source is essentially perfect. Ollama llama3.1:8b stuffs
# ~25% of tool calls into message content instead of emitting structured
# tool_calls, which the synthesizer mostly catches but adds latency + flakiness.
# Matched as case-insensitive substrings of the full user text.
_INTROSPECTION_KEYWORDS = (
    # Direct references to Chloe's own code
    "your code", "your source code", "your source", "your python",
    "your implementation", "your function", "your method", "your module",
    "your class", "your handler", "your routine", "your logic",
    "your script", "your tts", "your stt", "your router",
    # Code archaeology phrasing aimed at Chloe
    "show me your code", "show me the code",
    "how does your", "how is your",
    "what does your",
    "where in your code", "where is your",
    "look in your code", "look at your code",
    "search your code", "search your source",
    "grep your", "grep_source",
    # Specific filenames Chloe should recognize as her own
    "in jarvis.py", "in hud_server", "in chloe_memory",
    "in chloe-mobile", "in start_jarvis",
)


# Phrases the fast model emits when it's stalling on a question that really
# needs web search. If any of these show up in a fast-path reply, we
# automatically retry the same turn through compound-mini for real search.
# Patterns are matched as case-insensitive substrings against the full reply.
_HEDGING_PATTERNS = (
    "real-time", "real time",
    "as of my last update", "as of my training",
    "as of my knowledge cutoff", "knowledge cutoff",
    "i don't have access", "i do not have access",
    "i'm not able to access", "i am not able to access",
    "can't access live", "cannot access live",
    "look at a reliable", "consult a reliable",
    "consult a financial", "speaking with a financial",
    "i don't have current", "i do not have current",
    "i can't provide real", "i cannot provide real",
    "i don't have the latest", "i do not have the latest",
    "i'm not sure what the current", "i am not sure what the current",
    "you'll need to look", "you'd need to look", "you would need to look",
    "i don't have up-to-date", "i do not have up-to-date",
    "i can't browse", "i cannot browse",
)


def _looks_like_hedge(reply: str) -> bool:
    """Return True if the reply matches one of our 'I can't answer this
    without web search' fingerprints. Used to auto-retry through
    compound-mini when the fast path missed a real-time question."""
    if not reply:
        return False
    rl = reply.lower()
    return any(p in rl for p in _HEDGING_PATTERNS)

def _needs_realtime(text: str) -> bool:
    """Return True if `text` looks like a question that needs current/live data."""
    if not text:
        return False
    t = text.lower().strip()
    return any(kw in t for kw in _REALTIME_KEYWORDS)


def _is_introspection_query(text: str) -> bool:
    """Return True if `text` looks like the user is asking about Chloe's own
    implementation (her source code, functions, modules). Used to force-route
    these turns to Groq MODEL_TEXT, where grep_source tool calling is reliable;
    Ollama llama3.1:8b emits malformed tool calls ~25% of the time."""
    if not text:
        return False
    t = text.lower().strip()
    return any(kw in t for kw in _INTROSPECTION_KEYWORDS)


def _select_chat_model(user_text: str) -> str:
    """Pick MODEL_SEARCH (compound) when the user's question needs real-time
    data, otherwise MODEL_TEXT (fast Llama). Falls back to fast model if
    compound is disabled."""
    if USE_COMPOUND and _needs_realtime(user_text):
        return MODEL_SEARCH
    return MODEL_TEXT


def _pick_route(user_text: str) -> str:
    """Decide which LLM path handles this turn.

    Returns one of:
      'groq_search' — query needs real-time data; Groq compound-mini
                       (the only path with built-in web search).
      'ollama'      — local Ollama (Ollama-primary mode + daemon reachable).
      'groq_fast'   — Groq fast Llama (legacy cloud-first; also used for
                       introspection questions about Chloe's own code, where
                       MODEL_TEXT's grep_source tool calling is much more
                       reliable than Ollama's; and as the fallback when
                       Ollama is offline in Ollama-primary mode).
    """
    if USE_COMPOUND and _needs_realtime(user_text):
        return 'groq_search'
    # Introspection questions about Chloe's own source — force Groq MODEL_TEXT.
    # llama-3.3-70b emits structured tool_calls reliably; Ollama llama3.1:8b
    # stuffs ~25% of tool calls into message content (the synthesizer catches
    # most but adds latency). Skip if Groq isn't configured — let the normal
    # path handle it; the `groq_fast` branch already falls back to Ollama if
    # Groq is empty/unavailable.
    if _sync_groq is not None and _is_introspection_query(user_text):
        return 'groq_fast'
    if OLLAMA_PRIMARY and _ollama_available():
        return 'ollama'
    return 'groq_fast'


def _last_user_text(messages: list) -> str:
    """Extract plain text from the last user message in a Groq-format history."""
    if not messages:
        return ""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    return b.get("text", "")
        return ""
    return ""


import re as _re

def _extract_retry_after(err: Exception) -> float | None:
    """If a Groq APIError mentions 'try again in X.Xs', return X.X. Else None."""
    s = str(err)
    m = _re.search(r"try again in ([\d.]+)\s*s", s, flags=_re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ─── URL ATTACHMENT FETCH ───────────────────────────────────────────────────
# When the user includes an http(s) link in their chat message we fetch the page
# server-side (browser CORS makes client-side fetch unreliable from the HUD),
# strip the HTML down to readable text, and prepend an excerpt to the user's
# message so Chloe sees the page content as conversational context.
_URL_RE = _re.compile(r'https?://[^\s<>"\'\)\]]+')
_URL_FETCH_TIMEOUT = 10
_URL_FETCH_MAX_WORDS = 3000
_URL_FETCH_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/121.0 Safari/537.36'
)


def _strip_url_trailing_punct(url: str) -> str:
    """URLs in prose often have trailing punctuation that isn't really part of
    the URL — period at end of sentence, closing paren after a parenthetical,
    etc. Trim those before fetching."""
    while url and url[-1] in '.,;:!?)]\'"':
        url = url[:-1]
    return url


def _fetch_url_content(url: str) -> str:
    """Fetch a URL and return up to _URL_FETCH_MAX_WORDS of readable text.
    Returns a friendly error string on failure (never raises) so the chat turn
    can continue even if one link 404s."""
    if not _URL_FETCH_AVAILABLE:
        return (f"[couldn't fetch {url}: install `requests` and "
                f"`beautifulsoup4` to enable URL reading]")
    try:
        url = _strip_url_trailing_punct(url)
        r = _requests.get(
            url,
            headers={'User-Agent': _URL_FETCH_UA,
                     'Accept': 'text/html,application/xhtml+xml'},
            timeout=_URL_FETCH_TIMEOUT,
            allow_redirects=True,
        )
        r.raise_for_status()
        ct = r.headers.get('Content-Type', '').lower()
        if 'html' not in ct and 'text' not in ct and 'xml' not in ct:
            return f"[couldn't fetch {url}: content-type {ct} not supported]"
        soup = _BeautifulSoup(r.text, 'html.parser')
        # Strip non-content elements before extracting
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside',
                         'noscript', 'iframe', 'form', 'svg']):
            tag.decompose()
        # Prefer main/article/content if present, fall back to body
        main = (soup.find('main') or soup.find('article')
                or soup.find(id='content') or soup.find(class_='content')
                or soup.body or soup)
        text = main.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())  # collapse whitespace
        words = text.split(' ')
        if len(words) > _URL_FETCH_MAX_WORDS:
            text = ' '.join(words[:_URL_FETCH_MAX_WORDS]) + ' …[truncated]'
        if not text.strip():
            return f"[fetched {url} but found no readable text]"
        return text
    except _requests.Timeout:
        return f"[couldn't fetch {url}: timed out after {_URL_FETCH_TIMEOUT}s]"
    except _requests.RequestException as e:
        return f"[couldn't fetch {url}: {type(e).__name__}]"
    except Exception as e:
        return f"[couldn't fetch {url}: {type(e).__name__}: {e}]"


def _user_text_from_message(msg: dict) -> str:
    """Extract plain text from a user message (string OR list-of-blocks shape)."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(
            b.get('text', '') for b in c
            if isinstance(b, dict) and b.get('type') == 'text'
        )
    return ''


def _augment_user_message_with_urls(messages: list):
    """If the LAST user message contains http(s) URLs, fetch each one and
    prepend the extracted page text to that message. Returns (messages, urls)
    where urls is the list of URLs that were fetched. Mutates messages in place."""
    if not messages:
        return messages, []
    last = messages[-1]
    if last.get('role') != 'user':
        return messages, []

    user_text = _user_text_from_message(last)
    raw_urls = _URL_RE.findall(user_text)
    if not raw_urls:
        return messages, []

    # Dedupe (preserving order) + strip trailing punctuation
    seen = set()
    unique = []
    for u in raw_urls:
        cleaned = _strip_url_trailing_punct(u)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)

    print(f"[chloe] fetching {len(unique)} URL(s) from user message")
    excerpts = []
    for url in unique:
        text = _fetch_url_content(url)
        print(f"[chloe]   {url} -> {len(text)} chars")
        excerpts.append(f"[Web page: {url}]\n\n{text}\n\n")

    prepend = ''.join(excerpts) + '---\n\n'

    content = last.get('content')
    if isinstance(content, str):
        last['content'] = prepend + content
    elif isinstance(content, list):
        new_blocks = []
        text_block_handled = False
        for b in content:
            if (isinstance(b, dict) and b.get('type') == 'text'
                    and not text_block_handled):
                new_blocks.append({'type': 'text',
                                   'text': prepend + b.get('text', '')})
                text_block_handled = True
            else:
                new_blocks.append(b)
        if not text_block_handled:
            new_blocks.insert(0, {'type': 'text', 'text': prepend})
        last['content'] = new_blocks

    return messages, unique


def _trim_messages_for_model(groq_messages, model, max_msgs=None):
    """Keep the system prompt + last N non-system messages. compound-mini has
    a small context window (~8k tokens), so we keep less for it. Llama-3.3 has
    128k context and tolerates much more. The trim is conservative — the
    aggressive retry after a 413 will trim further."""
    if max_msgs is None:
        max_msgs = 6 if model == MODEL_SEARCH else 30
    system_msgs = [m for m in groq_messages if m.get('role') == 'system']
    other_msgs  = [m for m in groq_messages if m.get('role') != 'system']
    if len(other_msgs) > max_msgs:
        dropped = len(other_msgs) - max_msgs
        other_msgs = other_msgs[-max_msgs:]
        print(f"[chloe] trimmed {dropped} old message(s) from history (model={model}, kept last {max_msgs})")
    return system_msgs + other_msgs


def _is_too_large_error(err) -> bool:
    """Detect Groq's 'request body too large' (413) errors so handle_chat can
    trim history and retry instead of just bubbling up to the HUD."""
    s = str(err).lower()
    return ("too large" in s
            or "request_too_large" in s
            or "413" in s
            or "entity too large" in s)


# ─── CHAT HANDLER (text input from HUD) ──────────────────────────────────────
async def handle_chat(data, websocket):
    """
    Streams Groq response back to a single HUD client over its WebSocket.
    Updates _voice_history so the voice path sees text-input context too.
    Uses groq/compound-mini which has built-in web search + code execution
    handled server-side — no orchestration needed on our end.
    """
    if not _async_groq:
        await _ws_send(websocket, {"type": "error", "text": "GROQ_API_KEY not set in .env"})
        return

    messages = data.get("messages", [])
    system   = data.get("system", "")
    max_tok  = int(data.get("max_tokens", 1024))

    # Check for "remember: <fact>" — short-circuit before URL fetch / model
    # routing / streaming. Cheap, deterministic, and avoids burning Groq
    # tokens on commands we can satisfy locally.
    if messages:
        _last_user = _user_text_from_message(messages[-1]) or ""
        ack = _try_handle_remember(_last_user)
        if ack is not None:
            _push_history("user", _last_user, modality="chat")
            _push_history("assistant", ack, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": ack})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts"):
                hud_server.broadcast_sync("speaking")
                try:
                    await _reply_audio_or_speak(ack, data, label="chat-remember")
                except Exception as e:
                    print(f"[chloe] chat TTS error on remember-ack: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

    # Lights: /lights status and natural-language ("turn off the bedroom")
    if messages:
        _last_user_l = _user_text_from_message(messages[-1]) or ""
        lights_reply = await asyncio.to_thread(try_handle_lights_command, _last_user_l)
        if lights_reply is not None:
            _push_history("user", _last_user_l, modality="chat")
            _push_history("assistant", lights_reply, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": lights_reply})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts"):
                hud_server.broadcast_sync("speaking")
                try:
                    await _reply_audio_or_speak(lights_reply, data, label="chat-lights")
                except Exception as e:
                    print(f"[chloe] chat TTS error on lights reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

    # Brain commands: /ingest, /query, /lint, /fact, /brain, /podcast, /add
    if messages:
        _last_user = _user_text_from_message(messages[-1]) or ""
        brain_reply = await asyncio.to_thread(try_handle_brain_command, _last_user)
        if brain_reply is not None:
            # brain_reply may be a string (normal) or a dict {text, no_tts}
            # for commands that produce their own audio (e.g. /podcast plays
            # a WAV via os.startfile and shouldn\'t also TTS the status text).
            if isinstance(brain_reply, dict):
                _brain_text = brain_reply.get("text", "")
                _brain_silent = bool(brain_reply.get("no_tts"))
            else:
                _brain_text = brain_reply
                _brain_silent = False
            _push_history("user", _last_user, modality="chat")
            _push_history("assistant", _brain_text, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": _brain_text})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts") and not _brain_silent:
                # In reply_audio mode the HUD plays TTS in-browser; its
                # TtsAudio.play onStart/onEnd callbacks already drive the
                # speaking->idle transition in lock-step with actual playback.
                # Broadcasting "speaking"/"idle" here races ahead of the audio
                # (most visible on long /query replies, where the backend
                # "idle" arrives while decodeAudioData is still resolving)
                # and the orb flips back to idle mid-speech. Skip the manual
                # broadcasts in that mode; the audio callbacks handle state.
                # Local _speak path still needs them because
                # _reply_audio_or_speak awaits through playback locally.
                _hud_via_audio = bool(data.get("reply_audio"))
                if not _hud_via_audio:
                    hud_server.broadcast_sync("speaking")
                try:
                    await _reply_audio_or_speak(_brain_text, data, label="chat-brain")
                except Exception as e:
                    print(f"[chloe] chat TTS error on brain reply: {e}")
                    # Backstop: unexpected exception — force idle so the HUD
                    # doesn't get stuck in speaking/thinking.
                    hud_server.broadcast_sync("idle")
                else:
                    if not _hud_via_audio:
                        hud_server.broadcast_sync("idle")
            return

    # If the user message contains URLs, fetch them server-side and inject the
    # readable text into the message before sending to Groq. Browser CORS makes
    # client-side fetch unreliable, so it has to live here.
    if _URL_FETCH_AVAILABLE and messages:
        _urls_in_msg = _URL_RE.findall(_user_text_from_message(messages[-1]))
        if _urls_in_msg:
            await _ws_send(websocket, {
                "type": "tool_start",
                "text": f"Reading {len(set(_urls_in_msg))} link(s)…",
            })
            messages, _fetched = await asyncio.to_thread(
                _augment_user_message_with_urls, messages
            )

    # Pick the model: vision if there's an image, otherwise router decides
    # between fast Llama and compound-mini based on whether the question looks
    # like it needs real-time data.
    # Pick the model. Three branches:
    #   1. Image attached → MODEL_VISION (must be Groq; Ollama has no vision)
    #   2. Real-time query → MODEL_SEARCH (must be Groq compound; only path
    #      with web search)
    #   3. Everyday chat → Ollama (if OLLAMA_PRIMARY=1 and daemon running)
    #      OR Groq fast Llama (legacy mode / Ollama unavailable)
    use_ollama = False
    if _needs_vision(messages):
        model = MODEL_VISION
        route_reason = "image"
    else:
        user_text = _last_user_text(messages)
        route = _pick_route(user_text)
        if route == 'ollama':
            model = MODEL_TEXT  # used only for trim sizing + preamble
            use_ollama = True
            route_reason = "ollama-primary"
        elif route == 'groq_search':
            model = MODEL_SEARCH
            route_reason = "real-time"
        else:
            model = MODEL_TEXT
            route_reason = "fast"

    # Date + behavior preamble. Tailored to which model was chosen so we don't
    # tell a fast-path model "you can search the web" (it can't).
    today = datetime.now().strftime("%A, %B %d, %Y")
    if model == MODEL_SEARCH:
        preamble = (
            f"Today's date is {today}.\n"
            f"You can search the web automatically when needed. Use search for anything "
            f"that may have changed since your training (current prices, weather, news, "
            f"sports scores, recent events, who currently holds a position). For things "
            f"you already know, just answer directly. NEVER invent numbers or facts — "
            f"search instead, or say you couldn't find it."
        )
    else:
        preamble = (
            f"Today's date is {today} — you know the current date and should not claim otherwise.\n"
            f"For this turn you do NOT have web search available. If the question requires "
            f"current/live data (prices, weather, news, scores, who currently holds a "
            f"position) tell the user you'd need to look it up — don't invent the answer. "
            f"For general knowledge, conversation, or anything you already know, answer "
            f"directly without disclaimers."
        )
    # Append self-knowledge + mode tone + long-term facts to every chat
    # turn. Recall block only fires if the user's question looks like a
    # memory probe.
    about_block = format_about_block(_memory.about_body())
    mode_block = _mode_block()
    facts_block = format_facts_block(_memory.facts_body())
    recall_block = ""
    user_text_for_recall = _last_user_text(messages)
    if user_text_for_recall and looks_like_recall_query(user_text_for_recall):
        try:
            hits = _memory.search_turns(user_text_for_recall, limit=5)
            recall_block = format_recall_block(hits)
            if hits:
                print(f"[memory] chat recall: {len(hits)} hit(s)", flush=True)
        except Exception as e:
            print(f"[memory] chat recall failed: {e}", flush=True)
    full_system = (preamble + about_block + mode_block + facts_block
                   + recall_block
                   + ("\n\n" + system if system else ""))

    groq_messages = _to_groq_messages(messages)
    groq_messages.insert(0, {"role": "system", "content": full_system})

    # Show the model that will ACTUALLY generate. When use_ollama, model
    # var holds MODEL_TEXT (used for trim sizing only); the real model is
    # OLLAMA_MODEL. Display that so logs match the "Ollama (X) replied" line.
    _display_model = f"ollama:{OLLAMA_MODEL}" if use_ollama else model
    print(f"[chloe] chat → {_display_model} [{route_reason}] ({len(groq_messages)} msgs)")
    hud_server.broadcast_sync("thinking")

    # ─── Ollama-primary fast path ────────────────────────────────────────
    # Skip the Groq stream entirely when route says use Ollama. Ollama
    # returns its reply in one shot (no streaming) so we fake-stream it
    # word by word for visual rhythm. This is the same pattern the legacy
    # Groq-failure fallback uses, just promoted to a primary path.
    if use_ollama:
        try:
            ollama_msgs = _trim_messages_for_model(groq_messages, MODEL_TEXT)
            ollama_reply = await asyncio.to_thread(
                _ollama_chat, ollama_msgs, max_tok
            )
        except Exception as e:
            print(f"[chloe] Ollama errored: {e}", flush=True)
            traceback.print_exc()
            ollama_reply = ""
        if not ollama_reply and _sync_groq:
            # Ollama daemon hiccup — fall back to Groq fast Llama (cloud)
            print("[chloe] Ollama empty — falling back to Groq fast Llama",
                  flush=True)
            await _ws_send(websocket, {
                "type": "tool_start",
                "text": "Ollama unreachable — using Groq fast path…",
            })
            use_ollama = False  # let the Groq stream block below run
        else:
            # Ollama replied — stream it to the HUD and finish the turn
            await _ws_send(websocket, {"type": "start"})
            for word in ollama_reply.split():
                await _ws_send(websocket, {"type": "delta", "text": word + " "})
                await asyncio.sleep(0.015)
            await _ws_send(websocket, {"type": "done"})
            full_reply = ollama_reply
            last_user = messages[-1] if messages else None
            if last_user and last_user.get("role") == "user":
                uc = last_user.get("content")
                user_text_str = uc if isinstance(uc, str) else next(
                    (b.get("text", "") for b in uc
                     if isinstance(b, dict) and b.get("type") == "text"),
                    "",
                )
                if user_text_str:
                    _push_history("user", user_text_str, modality="chat")
            _push_history("assistant", full_reply, modality="chat")
            if not data.get("no_tts"):
                hud_server.broadcast_sync("speaking")
                try:
                    await _reply_audio_or_speak(full_reply, data, label="chat-ollama")
                except Exception as e:
                    print(f"[chloe] chat TTS error on Ollama reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

    full_reply = ""
    try:
        # Trim the message list to a model-appropriate length BEFORE first try.
        # Avoids the common case where an active chat session gradually fills
        # compound-mini's 8k context window and starts erroring out with 413.
        groq_messages = _trim_messages_for_model(groq_messages, model)

        # Open the stream, retrying once on rate-limit (with the "try again in
        # Xs" hint Groq returns) OR on 413 too-large errors (with aggressive
        # history trim). Both are common enough on free tier that a single
        # retry is worth it.
        stream = None
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                stream = await _async_groq.chat.completions.create(
                    model=model,
                    messages=groq_messages,
                    max_tokens=max_tok,
                    temperature=0.7,
                    stream=True,
                )
                break
            except Exception as e:
                # 413: payload too big — trim history hard and retry once
                if _is_too_large_error(e) and attempts < 3:
                    print(f"[chloe] 413 too-large on {model}; trimming hard and retrying")
                    await _ws_send(websocket, {
                        "type": "tool_start",
                        "text": "Context too large; trimming history and retrying…",
                    })
                    groq_messages = _trim_messages_for_model(
                        groq_messages, model, max_msgs=2
                    )
                    continue
                # Rate limit: respect Groq's "try again in Xs" hint
                wait = _extract_retry_after(e)
                if wait is not None and attempts < 3:
                    pad = 0.5
                    wait_total = min(wait + pad, 30.0)
                    print(f"[chloe] rate-limited on {model}; waiting {wait_total:.1f}s and retrying…")
                    await _ws_send(websocket, {
                        "type": "tool_start",
                        "text": f"Rate-limited; retrying in {wait_total:.0f}s…",
                    })
                    await asyncio.sleep(wait_total)
                    continue
                raise  # different error, or out of retries — bubble up
        if stream is None:
            return  # unreachable, but be defensive
        await _ws_send(websocket, {"type": "start"})

        # Compound systems may include tool execution metadata in the stream.
        # We only forward content deltas to the HUD; tool details are logged.
        executed_tools_seen = False
        async for chunk in stream:
            try:
                delta_obj = chunk.choices[0].delta
                delta = delta_obj.content
            except (AttributeError, IndexError):
                delta = None
            # Compound emits executed_tools metadata once per tool run; surface
            # a one-time "Searching the web…" hint to the HUD when it appears.
            try:
                if not executed_tools_seen:
                    et = getattr(delta_obj, "executed_tools", None)
                    if et:
                        executed_tools_seen = True
                        # Find the first search call's query if there is one
                        first_q = ""
                        for t in et:
                            args = getattr(t, "arguments", "") or ""
                            try:
                                first_q = json.loads(args).get("query", "") if args else ""
                            except Exception:
                                first_q = ""
                            if first_q:
                                break
                        note = f"Searching: {first_q}" if first_q else "Searching the web…"
                        print(f"[chloe]   {note}")
                        await _ws_send(websocket, {"type": "tool_start", "text": note})
            except Exception:
                pass
            if delta:
                full_reply += delta
                await _ws_send(websocket, {"type": "delta", "text": delta})
        await _ws_send(websocket, {"type": "done"})

        # Hedge-detection auto-retry for the chat path. If the fast model
        # produced "I don't have real-time data" etc., re-run the same turn
        # through compound-mini (which has built-in web search) and stream
        # that as a follow-up message. The user sees the hedged reply first
        # (already streamed), a status notification, then the real answer.
        if (full_reply
                and model == MODEL_TEXT
                and USE_COMPOUND
                and _looks_like_hedge(full_reply)):
            print(f"[chloe] chat reply hedged — auto-retrying with compound-mini",
                  flush=True)
            await _ws_send(websocket, {
                "type": "tool_start",
                "text": "That looked like a real-time question — searching the web…",
            })
            try:
                # Re-trim for compound's smaller context window.
                retry_msgs = _trim_messages_for_model(groq_messages, MODEL_SEARCH)
                # Swap the system message preamble to the compound version so
                # it knows it can search.
                today = datetime.now().strftime("%A, %B %d, %Y")
                retry_preamble = (
                    f"Today's date is {today}.\n"
                    f"You can search the web automatically when needed. The previous "
                    f"reply hedged on a real-time question — search the web now and "
                    f"give the user the actual answer."
                )
                retry_full_system = (retry_preamble + about_block
                                     + mode_block + facts_block
                                     + recall_block
                                     + ("\n\n" + system if system else ""))
                if retry_msgs and retry_msgs[0].get("role") == "system":
                    retry_msgs[0] = {"role": "system", "content": retry_full_system}
                else:
                    retry_msgs.insert(0, {"role": "system",
                                          "content": retry_full_system})

                retry_stream = await _async_groq.chat.completions.create(
                    model=MODEL_SEARCH,
                    messages=retry_msgs,
                    max_tokens=max_tok,
                    temperature=0.7,
                    stream=True,
                )
                await _ws_send(websocket, {"type": "start"})
                retry_full = ""
                retry_tools_seen = False
                async for chunk in retry_stream:
                    try:
                        delta_obj = chunk.choices[0].delta
                        delta = delta_obj.content
                    except (AttributeError, IndexError):
                        delta = None
                    try:
                        if not retry_tools_seen:
                            et = getattr(delta_obj, "executed_tools", None)
                            if et:
                                retry_tools_seen = True
                                first_q = ""
                                for tt in et:
                                    args = getattr(tt, "arguments", "") or ""
                                    try:
                                        first_q = json.loads(args).get("query", "") if args else ""
                                    except Exception:
                                        first_q = ""
                                    if first_q:
                                        break
                                if first_q:
                                    print(f"[chloe]   retry searching: {first_q!r}")
                    except Exception:
                        pass
                    if delta:
                        retry_full += delta
                        await _ws_send(websocket, {"type": "delta", "text": delta})
                await _ws_send(websocket, {"type": "done"})
                if retry_full.strip():
                    # Use the better answer for history + TTS.
                    full_reply = retry_full
                    print(f"[chloe] chat retry succeeded with web search ({len(retry_full)} chars)",
                          flush=True)
            except Exception as e:
                print(f"[chloe] chat hedge-retry failed: {e}", flush=True)
                # Fall through with the original (hedged) reply — better than nothing.

        # Update shared history (best-effort: collapse user msg to its text)
        last_user = messages[-1] if messages else None
        if last_user and last_user.get("role") == "user":
            uc = last_user.get("content")
            user_text = uc if isinstance(uc, str) else next(
                (b.get("text", "") for b in uc if isinstance(b, dict) and b.get("type") == "text"),
                "",
            )
            if user_text:
                _push_history("user", user_text, modality="chat")
        if full_reply:
            _push_history("assistant", full_reply, modality="chat")

        # Speak the reply through the same TTS pipeline the voice path uses
        # (ElevenLabs if configured, edge-tts otherwise) so chat and voice
        # replies sound identical. Run in a thread so the asyncio loop stays
        # free for other WebSocket clients while audio plays.
        if full_reply.strip() and not data.get("no_tts"):
            hud_server.broadcast_sync("speaking")
            try:
                await _reply_audio_or_speak(full_reply, data, label="chat-groq")
            except Exception as e:
                print(f"[chloe] chat TTS error: {e}")
            finally:
                hud_server.broadcast_sync("idle")

    except Exception as e:
        traceback.print_exc()
        # Last-resort fallback: if Groq blew up entirely, try the local
        # Ollama daemon. Stream the reply word-by-word for a streaming
        # feel even though Ollama returns it in one shot.
        if OLLAMA_FALLBACK_ENABLED and _ollama_available():
            await _ws_send(websocket, {
                "type": "tool_start",
                "text": f"Groq error ({type(e).__name__}) — falling back to local Ollama…",
            })
            try:
                ollama_reply = await asyncio.to_thread(
                    _ollama_chat, groq_messages, max_tok
                )
            except Exception as oe:
                ollama_reply = ""
                print(f"[chloe] Ollama fallback errored: {oe}", flush=True)
            if ollama_reply:
                await _ws_send(websocket, {"type": "start"})
                # Word-by-word "stream" so the chat panel feels alive even
                # though we got the whole reply in one HTTP response.
                for word in ollama_reply.split():
                    await _ws_send(websocket, {"type": "delta", "text": word + " "})
                    await asyncio.sleep(0.015)
                await _ws_send(websocket, {"type": "done"})
                # Push to history + speak, same as the normal path
                last_user = messages[-1] if messages else None
                if last_user and last_user.get("role") == "user":
                    uc = last_user.get("content")
                    user_text_str = uc if isinstance(uc, str) else next(
                        (b.get("text", "") for b in uc
                         if isinstance(b, dict) and b.get("type") == "text"),
                        "",
                    )
                    if user_text_str:
                        _push_history("user", user_text_str, modality="chat")
                _push_history("assistant", ollama_reply, modality="chat")
                if not data.get("no_tts"):
                    hud_server.broadcast_sync("speaking")
                    try:
                        await _reply_audio_or_speak(ollama_reply, data, label="chat-ollama-fb")
                    except Exception as te:
                        print(f"[chloe] chat TTS error on Ollama reply: {te}")
                    finally:
                        hud_server.broadcast_sync("idle")
                return
        # Diagnose the failure mode so the user knows what to fix
        diag_parts = [f"{type(e).__name__}: {e}"]
        ollama_state = "reachable" if _ollama_available() else "unreachable"
        diag_parts.append(f"Ollama fallback: {ollama_state}")
        if not OLLAMA_FALLBACK_ENABLED:
            diag_parts.append("Fallback disabled (CHLOE_OLLAMA_FALLBACK=0)")
        elif ollama_state == "unreachable":
            diag_parts.append(f"To enable local fallback, run `ollama serve` "
                              f"(model: {OLLAMA_MODEL}, URL: {OLLAMA_URL})")
        await _ws_send(websocket, {"type": "error", "text": " | ".join(diag_parts)})

async def handle_volume(data, websocket):
    pass  # placeholder for future mic-level meters

async def handle_ptt_start(data, websocket):
    """HUD pressed PTT. Signal voice thread to start recording in PTT mode.
    Also fires barge-in so any in-progress speech gets interrupted."""
    if _ptt_mode.is_set():
        await _ws_send(websocket, {"type": "ptt_busy"})
        return
    _ptt_stop_signal.clear()
    _ptt_mode.set()
    # If Chloe is speaking right now, this interrupts her. The TTS loop
    # checks _barge_in_request between (and during) sentences.
    if _speaking.is_set():
        _barge_in_request.set()
        print("[chloe] PTT activated → barging in on speech")
    else:
        print("[chloe] PTT activated by HUD")
    await _ws_send(websocket, {"type": "ptt_started"})


async def handle_ptt_stop(data, websocket):
    """HUD released PTT. Signal the recording loop to finalize and process."""
    if not _ptt_mode.is_set():
        await _ws_send(websocket, {"type": "ptt_idle"})
        return
    _ptt_stop_signal.set()
    print("[chloe] PTT release signaled")
    await _ws_send(websocket, {"type": "ptt_stopping"})


async def handle_ptt_audio(data, websocket):
    """Mobile PTT path: phone records on its own mic, sends a WAV blob over
    the WS, we run it through the same Whisper → ask → reply pipeline the
    desktop PTT path uses.

    Expected payload:
      {"type": "ptt_audio", "wav_b64": "<base64 of mono int16 WAV>",
       "reply_audio": <bool>}

    When `reply_audio` is true we synthesize TTS and stream the audio bytes
    back to THIS client only (so the iPhone hears Chloe in earbuds, not
    echoing through the house). Otherwise we play on PC speakers as usual.

    Why WAV? Self-describing (sample-rate + bit-depth in header), stdlib
    `wave` decodes without ffmpeg, and the browser builds it from raw PCM
    via AudioWorklet so iOS's MediaRecorder codec quirks don't bite us."""
    b64 = data.get("wav_b64", "")
    if not b64:
        await _ws_send(websocket, {"type": "error", "text": "ptt_audio missing wav_b64"})
        return
    try:
        wav_bytes = base64.b64decode(b64)
    except Exception as e:
        await _ws_send(websocket, {"type": "error", "text": f"ptt_audio bad base64: {e}"})
        return

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            ch  = wf.getnchannels()
            sw  = wf.getsampwidth()
            sr  = wf.getframerate()
            n   = wf.getnframes()
            raw = wf.readframes(n)
        if sw != 2:
            await _ws_send(websocket, {"type": "error",
                "text": f"ptt_audio: expected 16-bit PCM, got sampwidth={sw}"})
            return
        audio = np.frombuffer(raw, dtype=np.int16)
        if ch == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        if sr != SAMPLE_RATE:
            audio = _resample_to_16k(audio, sr)
    except Exception as e:
        await _ws_send(websocket, {"type": "error", "text": f"ptt_audio wav decode: {e}"})
        return

    secs = len(audio) / SAMPLE_RATE
    print(f"[chloe] ptt_audio received: {secs:.2f}s of audio (sr={sr}, ch={ch})", flush=True)
    if secs < 0.25:
        await _ws_send(websocket, {"type": "ptt_too_short"})
        hud_server.broadcast_sync("idle")
        return

    hud_server.broadcast_sync("thinking")

    transcript = await asyncio.to_thread(_transcribe, audio)
    if not transcript:
        hud_server.broadcast_sync("idle")
        await _ws_send(websocket, {"type": "error", "text": "empty transcript"})
        return

    reply = await asyncio.to_thread(_ask_groq, transcript)
    if not reply:
        _broadcast_exchange(transcript, "[no reply — see terminal for error]")
        hud_server.broadcast_sync("idle")
        return

    _broadcast_exchange(transcript, reply)

    # Route the spoken reply via the shared helper. reply_audio=True ships
    # bytes to every WS client (broadcast — survives PWA WS swapping mid-
    # response); falsy plays on PC speakers via _speak().
    hud_server.broadcast_sync("speaking")
    try:
        await _reply_audio_or_speak(reply, data, label="ptt_audio")
    except Exception as e:
        print(f"[chloe] ptt_audio TTS error: {e}", flush=True)
    finally:
        hud_server.broadcast_sync("idle")


async def _dispatch(data, websocket):
    t = data.get("type")
    if   t == "chat":      await handle_chat(data, websocket)
    elif t == "volume":    await handle_volume(data, websocket)
    elif t == "ptt_start": await handle_ptt_start(data, websocket)
    elif t == "ptt_stop":  await handle_ptt_stop(data, websocket)
    elif t == "ptt_audio": await handle_ptt_audio(data, websocket)
    elif t == "wallet_balance":         await handle_wallet_balance(data, websocket)
    elif t == "wallet_create_invoice":  await handle_wallet_create_invoice(data, websocket)
    elif t == "wallet_send":            await handle_wallet_send(data, websocket)
    elif t == "wallet_history":         await handle_wallet_history(data, websocket)
    elif t == "lights_state":           await handle_lights_state(data, websocket)
    elif t == "lights_action":          await handle_lights_action(data, websocket)
    elif t == "lights_discover":        await handle_lights_discover(data, websocket)
    elif t == "lights_rename":          await handle_lights_rename(data, websocket)
    elif t == "lights_preset_apply":    await handle_lights_preset_apply(data, websocket)
    elif t == "lights_preset_save":     await handle_lights_preset_save(data, websocket)
    elif t == "lights_preset_delete":   await handle_lights_preset_delete(data, websocket)
    else: await _ws_send(websocket, {"type": "error", "text": f"unknown type: {t}"})


# ─── DIRECT WALLET WS ENDPOINTS ─────────────────────────────────────────────
# These bypass the LLM entirely — the PWA / HUD call these directly so users
# can transact without going through voice/chat. Same security guarantees
# as the LLM tool path: PIN + daily cap enforced server-side via
# wallet_guard.py. Receive endpoints are unauthenticated within the WS
# session (anyone reaching the WS can read balance / generate invoices);
# the send endpoint requires the PIN every time.

async def handle_wallet_balance(data, websocket):
    print("[chloe] WS wallet_balance request", flush=True)
    w = _wallet_module()
    if w is None:
        await _ws_broadcast({
            "type":  "wallet_balance_result",
            "ok":    False,
            "error": "wallet not configured (breez-sdk-liquid missing)",
        })
        return
    try:
        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(None, w.get_balance)
    except Exception as e:
        await _ws_broadcast({
            "type":  "wallet_balance_result",
            "ok":    False,
            "error": f"{type(e).__name__}: {e}",
        })
        return
    await _ws_broadcast({
        "type":                 "wallet_balance_result",
        "ok":                   bool(r.get("ok")),
        "balance_sat":          r.get("balance_sat", 0),
        "pending_send_sat":     r.get("pending_send_sat", 0),
        "pending_receive_sat":  r.get("pending_receive_sat", 0),
        "error":                r.get("error"),
    })


async def handle_wallet_create_invoice(data, websocket):
    print(f"[chloe] WS wallet_create_invoice request: amount={data.get('amount_sat')} memo={data.get('memo')!r}", flush=True)
    w = _wallet_module()
    if w is None:
        await _ws_broadcast({
            "type":  "wallet_invoice_result",
            "ok":    False,
            "error": "wallet not configured",
        })
        return
    amount = data.get("amount_sat")
    if isinstance(amount, str) and amount.strip().isdigit():
        amount = int(amount.strip())
    if not isinstance(amount, int) or amount < 1:
        await _ws_broadcast({
            "type":  "wallet_invoice_result",
            "ok":    False,
            "error": "amount_sat must be a positive integer",
        })
        return
    memo = str(data.get("memo") or "")
    try:
        loop = asyncio.get_event_loop()
        r = await asyncio.wait_for(
            loop.run_in_executor(None, w.create_invoice, amount, memo),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        print("[chloe] wallet_create_invoice TIMEOUT after 30s", flush=True)
        await _ws_broadcast({
            "type":  "wallet_invoice_result",
            "ok":    False,
            "error": "SDK timed out after 30s. Check Chloe terminal for [wallet] log lines to see which step hung.",
        })
        return
    except Exception as e:
        await _ws_broadcast({
            "type":  "wallet_invoice_result",
            "ok":    False,
            "error": f"{type(e).__name__}: {e}",
        })
        return
    bolt11_val = r.get("bolt11")
    print(f"[chloe] wallet_invoice_result → ok={bool(r.get('ok'))} "
          f"bolt11_len={len(bolt11_val) if isinstance(bolt11_val, str) else 0}",
          flush=True)
    await _ws_broadcast({
        "type":       "wallet_invoice_result",
        "ok":         bool(r.get("ok")),
        "bolt11":     bolt11_val,
        "amount_sat": r.get("amount_sat", amount),
        "fees_sat":   r.get("fees_sat", 0),
        "memo":       r.get("memo", memo),
        "error":      r.get("error"),
    })


async def handle_wallet_send(data, websocket):
    """Send a Lightning payment from the PWA / HUD. Same server-side guard
    as the LLM tool path: PIN required, daily cap enforced. We DO NOT log
    the PIN to console even on errors — only mask."""
    w = _wallet_module()
    wg = _wallet_guard_module()
    if w is None or wg is None:
        await _ws_broadcast({
            "type":  "wallet_send_result",
            "ok":    False,
            "error": "wallet not configured",
        })
        return

    dest = str(data.get("destination") or "").strip()
    amount = data.get("amount_sat")
    if isinstance(amount, str) and amount.strip().isdigit():
        amount = int(amount.strip())
    pin = str(data.get("pin") or "")
    if not dest:
        await _ws_broadcast({
            "type":  "wallet_send_result",
            "ok":    False,
            "error": "destination is required",
        })
        return

    print(f"[chloe] PWA send request → {dest[:24]}…  "
          f"amount={amount}  pin=<redacted>", flush=True)

    # Resolve amount for the cap check, same trick as the LLM dispatch:
    # if the invoice has it baked in, prepare_send_payment tells us.
    check_amount = amount if isinstance(amount, int) and amount > 0 else 0
    if check_amount == 0:
        try:
            import breez_sdk_liquid as bsl  # type: ignore
            loop = asyncio.get_event_loop()
            def _prep_check():
                sdk = w._connect()
                prep = sdk.prepare_send_payment(
                    bsl.PrepareSendRequest(destination=dest)
                )
                return w._extract_resolved_amount(prep, fallback=0)
            check_amount = await loop.run_in_executor(None, _prep_check)
        except Exception:
            pass
    if check_amount == 0:
        await _ws_broadcast({
            "type":  "wallet_send_result",
            "ok":    False,
            "error": "cannot determine amount; specify amount_sat",
        })
        return

    ok, reason = wg.authorize_send(check_amount, pin)
    if not ok:
        await _ws_broadcast({
            "type":  "wallet_send_result",
            "ok":    False,
            "error": reason,
        })
        return

    try:
        loop = asyncio.get_event_loop()
        amt_arg = amount if isinstance(amount, int) and amount > 0 else None
        r = await loop.run_in_executor(None, w.pay, dest, amt_arg)
    except Exception as e:
        await _ws_broadcast({
            "type":  "wallet_send_result",
            "ok":    False,
            "error": f"{type(e).__name__}: {e}",
        })
        return

    if not r.get("ok"):
        await _ws_broadcast({
            "type":  "wallet_send_result",
            "ok":    False,
            "error": r.get("error", "send failed"),
        })
        return

    try:
        wg.record_send(int(r.get("amount_sat") or check_amount),
                       r.get("payment_hash"))
    except Exception:
        pass

    await _ws_broadcast({
        "type":         "wallet_send_result",
        "ok":           True,
        "amount_sat":   r.get("amount_sat"),
        "fees_sat":     r.get("fees_sat"),
        "status":       r.get("status"),
        "payment_hash": r.get("payment_hash"),
    })


async def handle_wallet_history(data, websocket):
    print(f"[chloe] WS wallet_history request: limit={data.get('limit')}", flush=True)
    w = _wallet_module()
    if w is None:
        await _ws_broadcast({
            "type":  "wallet_history_result",
            "ok":    False,
            "error": "wallet not configured",
        })
        return
    limit = data.get("limit")
    if isinstance(limit, str) and limit.strip().isdigit():
        limit = int(limit.strip())
    if not isinstance(limit, int) or limit < 1:
        limit = 10
    limit = min(limit, 50)
    try:
        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(None, w.list_history, limit)
    except Exception as e:
        await _ws_broadcast({
            "type":  "wallet_history_result",
            "ok":    False,
            "error": f"{type(e).__name__}: {e}",
        })
        return
    await _ws_broadcast({
        "type":     "wallet_history_result",
        "ok":       bool(r.get("ok")),
        "payments": r.get("payments", []),
        "error":    r.get("error"),
    })

def _push_history(role, content, modality: str = "voice"):
    _voice_history.append({"role": role, "content": content})
    # trim, keeping pairs aligned
    excess = len(_voice_history) - _HISTORY_MAX
    if excess > 0:
        del _voice_history[:excess]
    # Also persist to SQLite so memory survives restarts. Memory errors
    # must not break the conversation flow.
    try:
        _memory.append_turn(role, content, modality=modality)
    except Exception as e:
        print(f"[memory] push failed: {e}", flush=True)


def _augmented_voice_system(model: str | None = None,
                            user_text: str = "") -> str:
    """Voice-path system prompt + self-knowledge + mode tone + long-term
    facts + (optional) recall block.

    Order:
      1. base persona            — who she is at the surface level
      2. about block (always)    — her architecture, capabilities, limits
      3. mode tone               — home vs office phrasing
      4. facts block (always)    — persistent things Ed has told her
      5. recall block (on probe) — top-k matches from past conversation
    """
    base = _voice_system(model)
    about_block = format_about_block(_memory.about_body())
    mode_block = _mode_block()
    facts_block = format_facts_block(_memory.facts_body())
    recall_block = ""
    if user_text and looks_like_recall_query(user_text):
        try:
            hits = _memory.search_turns(user_text, limit=5)
            recall_block = format_recall_block(hits)
            if hits:
                print(f"[memory] recall: {len(hits)} hit(s) for "
                      f"{user_text[:60]!r}", flush=True)
        except Exception as e:
            print(f"[memory] recall failed: {e}", flush=True)
    return base + about_block + mode_block + facts_block + recall_block


def _try_handle_remember(transcript: str) -> str | None:
    """If `transcript` is a 'remember: <fact>' OR 'remember about
    yourself: <note>' command, persist it and return the spoken
    acknowledgement. Otherwise return None so the caller falls through
    to normal LLM handling.

    Order matters: the about-self form is checked first because its
    pattern is a strict superset of plain 'remember:' (otherwise
    'remember about yourself: X' would write 'about yourself: X' as a
    plain fact)."""
    note = parse_remember_about(transcript)
    if note:
        if _memory.add_about_note(note):
            print(f"[memory] new about-note: {note!r}", flush=True)
            return f"Got it. I'll remember that about myself: {note}."
        return "I tried to save that note about myself but couldn't write the about file."

    fact = parse_remember(transcript)
    if not fact:
        return None
    if _memory.add_fact(fact):
        print(f"[memory] new fact: {fact!r}", flush=True)
        return f"Got it. I'll remember that {fact}."
    return "I tried to save that but couldn't write to the facts file."

# ─── VOICE PATH ──────────────────────────────────────────────────────────────
# Pattern: short-lived per-phase audio streams.
#   1. Wake-detection stream  → reads chunks, runs wake model, closes on hit
#   2. Recording stream       → fresh stream, blocking reads, closes before TTS
#   3. TTS playback           → no input stream open at all (avoids contention)
#
# This avoids the Windows audio-device problem where one long-running input
# stream goes silent after a sd.play() call. Each phase gets its own clean
# stream and we never run input + output concurrently.


def _resolve_mic_device(sd):
    """Return a device index for sounddevice. Honors $CHLOE_MIC override
    (numeric index OR substring match against device name).

    Without an override, picks the FIRST WASAPI input device matching the system
    default mic name — MME is the historical default on Windows but it has
    driver-stability issues with USB mics like the Samson C01U. WASAPI is far
    more reliable."""
    if MIC_DEVICE_OVERRIDE is not None:
        if MIC_DEVICE_OVERRIDE.isdigit():
            return int(MIC_DEVICE_OVERRIDE)
        needle = MIC_DEVICE_OVERRIDE.lower()
        # Substring match — but a name like "Microphone" matches the WDM-KS
        # variant first on Windows, and PortAudio blocking input doesn't
        # support WDM-KS (error -9999). Three-pass: prefer WASAPI, then any
        # non-WDM-KS host, then anything (preserves prior behavior as a last
        # resort so the warning path still fires when nothing matches).
        host_apis = sd.query_hostapis()
        def _hostname(d):
            h = d.get("hostapi")
            if h is None or not (0 <= h < len(host_apis)):
                return ""
            return host_apis[h]["name"].upper()
        devs = list(enumerate(sd.query_devices()))
        def _match_pass(predicate):
            for i, d in devs:
                if (d.get("max_input_channels", 0) > 0
                    and needle in d["name"].lower()
                    and predicate(_hostname(d))):
                    print(f"[voice] CHLOE_MIC matched device {i}: {d['name']} "
                          f"({_hostname(d) or '?'})")
                    return i
            return None
        for predicate in (
            lambda h: "WASAPI" in h,
            lambda h: "WDM-KS" not in h and "KERNEL STREAMING" not in h,
            lambda h: True,
        ):
            picked = _match_pass(predicate)
            if picked is not None:
                return picked
        print(f"[voice] WARNING: CHLOE_MIC={MIC_DEVICE_OVERRIDE!r} matched no device, using default")
        return None

    # No override — prefer a WASAPI entry of the system default mic
    try:
        default_idx = sd.default.device[0] if sd.default.device else None
        if default_idx is None:
            return None
        default_name = sd.query_devices(default_idx).get("name", "").lower().strip()
        # Find host APIs
        host_apis = sd.query_hostapis()
        wasapi_idx = None
        for i, h in enumerate(host_apis):
            if "WASAPI" in h["name"].upper():
                wasapi_idx = i
                break
        if wasapi_idx is None:
            return None  # no WASAPI available, stay on default

        # Use the most distinctive single token from the default name as the
        # matcher (e.g. "samson" from "microphone (samson c01u)"), preferring
        # tokens that aren't generic words like "microphone".
        tokens = [t.strip("()") for t in default_name.split()]
        skip = {"microphone", "input", "audio", "mic", "(", ")"}
        needles = [t for t in tokens if t and t not in skip]
        if not needles:
            needles = [tokens[0]] if tokens else []

        # Find a WASAPI device whose name contains any of our needles
        for i, d in enumerate(sd.query_devices()):
            if (d.get("max_input_channels", 0) > 0
                and d.get("hostapi") == wasapi_idx):
                dn = d["name"].lower()
                if any(n in dn for n in needles):
                    print(f"[voice] auto-picked WASAPI device {i}: {d['name']!r} "
                          f"(matched on '{needles[0]}', avoiding flaky MME default)")
                    return i
    except Exception as e:
        print(f"[voice] WASAPI auto-detect failed, using OS default: {e}")
    return None  # fall back to OS default


def _resample_to_16k(chunk_np: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample int16 audio from src_rate down to 16000 Hz. Uses scipy if available
    (better quality), falls back to simple linear interpolation. Always returns
    int16 mono."""
    if src_rate == SAMPLE_RATE:
        return chunk_np
    # scipy.signal.resample_poly is decent quality and fast
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_rate, SAMPLE_RATE)
        up   = SAMPLE_RATE // g
        down = src_rate    // g
        out = resample_poly(chunk_np.astype(np.float32), up, down)
        return np.clip(out, -32768, 32767).astype(np.int16)
    except ImportError:
        # Linear-interp fallback — works fine for speech at these rates
        ratio = SAMPLE_RATE / src_rate
        n_out = int(round(len(chunk_np) * ratio))
        x_old = np.linspace(0, 1, len(chunk_np), endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        out = np.interp(x_new, x_old, chunk_np.astype(np.float32))
        return np.clip(out, -32768, 32767).astype(np.int16)


_resample_warned_devices = set()

def _pick_device_samplerate(sd, device):
    """Determine the sample rate to ask sounddevice to open at. WASAPI rejects
    rates the device doesn't natively support (unlike MME, which silently
    resamples). So we try our preferred rate first; if that fails, fall back
    to the device's default rate and we'll resample in software."""
    if device is None:
        return SAMPLE_RATE
    try:
        sd.check_input_settings(device=device, samplerate=SAMPLE_RATE,
                                channels=1, dtype="int16")
        return SAMPLE_RATE
    except Exception:
        pass
    try:
        info = sd.query_devices(device)
        native = int(info.get("default_samplerate") or 48000)
        if device not in _resample_warned_devices:
            _resample_warned_devices.add(device)
            if VOICE_DEBUG:
                print(f"[voice] device {device} doesn't support {SAMPLE_RATE} Hz natively; "
                      f"opening at {native} Hz and resampling in software (subsequent "
                      f"opens will be silent)")
        return native
    except Exception:
        return 48000


def _open_input_stream_with_retry(sd, device, *, frame_length=None, max_attempts=4):
    """Open a sounddevice.InputStream, retrying transient PortAudio errors.

    Returns (stream, native_rate). If native_rate != SAMPLE_RATE, callers must
    resample chunks before using them with openwakeword / Whisper / Porcupine.

    `frame_length` is the wake detector's expected samples-per-predict at
    SAMPLE_RATE — defaults to CHUNK_SAMPLES (1280 for openwakeword); Porcupine
    typically uses 512. The block size requested from sounddevice is scaled to
    the device's native rate so that after resampling we get exactly
    frame_length samples per chunk.

    The Samson C01U on Windows is prone to brief 'device ID out of range' or
    'no driver installed' errors when streams are opened back-to-back; retry
    with a small sleep so the audio service has time to settle. WASAPI also
    rejects non-native sample rates, so we negotiate one up front."""
    if frame_length is None:
        frame_length = CHUNK_SAMPLES
    native_rate = _pick_device_samplerate(sd, device)
    if native_rate == SAMPLE_RATE:
        block = frame_length
    else:
        block = int(round(frame_length * native_rate / SAMPLE_RATE))

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            stream = sd.InputStream(
                samplerate=native_rate,
                channels=1,
                dtype="int16",
                blocksize=block,
                device=device,
            )
            return stream, native_rate
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                if VOICE_DEBUG:
                    print(f"[voice] stream-open attempt {attempt}/{max_attempts} failed: {e}")
                time.sleep(0.25 * attempt)
    raise last_err



def _broadcast_exchange(user_text: str, assistant_text: str):
    """Push a voice/PTT exchange to the HUD chat panel so the user can see what
    she said + what Chloe replied (otherwise voice activity is invisible there).
    The HUD's onmessage handler routes JSON-shaped strings to handleStructured."""
    try:
        hud_server.broadcast_sync(json.dumps({
            "type": "voice_exchange",
            "user": user_text,
            "assistant": assistant_text,
        }))
    except Exception as e:
        print(f"[voice] broadcast_exchange failed: {e}", flush=True)


def _create_wake_detector():
    """Initialize the wake-word backend. Returns a dict:
       {'engine': 'porcupine'|'openwakeword',
        'handle': backend object,
        'frame_length': int (samples per predict call at SAMPLE_RATE),
        'human': str (label for terminal prints),
        'predict': callable(np.int16 array of frame_length) -> bool}
    Porcupine is tried first if configured; falls back to openwakeword on
    any failure so the voice loop still starts."""
    if USE_PORCUPINE:
        try:
            import pvporcupine
            handle = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=PORCUPINE_PPNS,
            )
            names = [Path(p).name for p in PORCUPINE_PPNS]
            # Build human label from each .ppn's stem ("Hey-Chloe_en_..." → "Hey Chloe")
            human = " / ".join(
                Path(p).stem.split('_')[0].replace('-', ' ') for p in PORCUPINE_PPNS
            )
            print(f"[chloe] wake engine: Porcupine ({len(names)} keyword(s): {names}, "
                  f"frame={handle.frame_length})")
            return {
                'engine': 'porcupine',
                'handle': handle,
                'frame_length': handle.frame_length,
                'human': human,
                # Porcupine.process() returns -1 (no match) or the keyword index
                # (0..N-1). Any >= 0 = a match against any of our keywords.
                'predict': lambda chunk: handle.process(chunk) >= 0,
            }
        except ImportError:
            print("[chloe] Porcupine: pvporcupine package not installed — falling back to openwakeword")
        except Exception as e:
            print(f"[chloe] Porcupine init failed ({type(e).__name__}: {e}) — falling back to openwakeword")

    try:
        from openwakeword.model import Model as WakeModel
        print(f"[voice] loading wake word model(s): {WAKE_WORD_KEYS}")
        handle = WakeModel(wakeword_models=WAKE_WORD_PATHS, inference_framework="onnx")
        # Wake fires if ANY loaded model scores above threshold for this chunk.
        # One predict() call returns scores for ALL loaded models, so this is
        # cheap regardless of how many phrases are configured.
        def _predict_any(chunk, _h=handle, _keys=WAKE_WORD_KEYS, _thr=WAKE_THRESHOLD):
            scores = _h.predict(chunk)
            return any(scores.get(k, 0.0) >= _thr for k in _keys)
        return {
            'engine': 'openwakeword',
            'handle': handle,
            'frame_length': CHUNK_SAMPLES,
            'human': WAKE_WORD_HUMAN,
            'predict': _predict_any,
        }
    except Exception as e:
        print(f"[voice] wake model load failed: {e}")
        print("[voice] try: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
        return None


def _apply_gain(chunk):
    """Multiply int16 audio by MIC_GAIN, clipping to int16 range. No-op when gain
    is 1.0 (the default). Used for far-field setups where the mic is too quiet
    even with Windows input level maxed."""
    if MIC_GAIN == 1.0:
        return chunk
    return np.clip(chunk.astype(np.int32) * MIC_GAIN,
                   -32768, 32767).astype(np.int16)


_GREETING_POOL = [
    "Good {tod}, Ed. Chloe is online and standing by.",
    "Good {tod}, Ed. Ready when you are.",
    "Hello Ed. Chloe online.",
    "Good {tod}, Ed. All systems ready.",
]

def _greet_user():
    """Speak a short startup greeting through ElevenLabs (or edge-tts fallback).
    Runs once when the voice loop boots, after the wake detector is ready."""
    h = datetime.now().hour
    if   h < 12: tod = "morning"
    elif h < 17: tod = "afternoon"
    else:        tod = "evening"
    greeting = random.choice(_GREETING_POOL).format(tod=tod)
    print(f"[chloe] greeting: {greeting!r}")
    hud_server.broadcast_sync("speaking")
    try:
        _speak(greeting)
    except Exception as e:
        print(f"[chloe] greeting failed: {e}")
    finally:
        hud_server.broadcast_sync("idle")


def _generate_boot_chime(sr=44100):
    """Halo/Alien-style military boot sequence. ~1.8s total. Layers:
      1. Deep sub-bass impact (55/110Hz) — the "boom" you feel in your chest
      2. Atmospheric filtered noise swell — ominous spaceship-air
      3. Metallic system-check blips (F#6, A6, C7) — "weapons armed"
      4. Rising sweep (220→660Hz) — building tension
      5. Held A-minor chord (A4, C5, E5) — ominous resolve, not happy
      6. Multi-tap delay echo — cathedral / spaceship reverb tail
    All synthesized in numpy — no audio files."""
    total_s = 1.8
    n = int(total_s * sr)
    out = np.zeros(n, dtype=np.float32)
    t = np.linspace(0, total_s, n, endpoint=False, dtype=np.float32)

    # Layer 1: Sub-bass impact (quick attack, exponential decay)
    bass_env = np.exp(-t * 1.5) * (1 - np.exp(-t * 30))
    bass = (np.sin(2*np.pi*55*t) + 0.5*np.sin(2*np.pi*110*t)) * bass_env * 0.45
    out += bass

    # Layer 2: Atmospheric noise pad (low-passed via running-mean convolution,
    # swelling in then fading out)
    raw_noise = (np.random.rand(n).astype(np.float32) * 2 - 1)
    pad = np.convolve(raw_noise, np.ones(8, dtype=np.float32)/8, mode='same')
    pad_env = np.minimum(t/0.5, 1.0) * np.maximum(0, 1 - (t - 1.2)/0.6)
    out += pad * pad_env * 0.18

    # Layer 3: Metallic system-check blips at 0.45/0.55/0.65s
    for i, freq in enumerate([1480, 1760, 2093]):  # F#6, A6, C7
        start = int((0.45 + i*0.10) * sr)
        bn = int(0.05 * sr)
        if start + bn > n: continue
        bt = np.linspace(0, 0.05, bn, dtype=np.float32)
        beep = np.sin(2*np.pi*freq*bt) * np.exp(-bt * 60) * 0.18
        out[start:start+bn] += beep

    # Layer 4: Rising sweep — 220Hz to 660Hz over 0.55s
    sweep_start, sweep_end = int(0.75*sr), int(1.30*sr)
    sn = sweep_end - sweep_start
    st = np.linspace(0, sn/sr, sn, dtype=np.float32)
    sweep_freq = np.linspace(220, 660, sn)
    sweep_phase = 2*np.pi * np.cumsum(sweep_freq) / sr
    sweep_env = np.minimum(st/0.05, 1.0) * np.maximum(0, 1 - (st - 0.45)/0.10)
    sweep = (np.sin(sweep_phase) + 0.4*np.sin(sweep_phase*0.5)) * sweep_env * 0.30
    out[sweep_start:sweep_end] += sweep

    # Layer 5: Held A-minor chord (A4, C5, E5) — minor for ominous resolve
    chord_start = int(1.20 * sr)
    cn = n - chord_start
    ct = np.linspace(0, cn/sr, cn, dtype=np.float32)
    chord = (np.sin(2*np.pi*440.00*ct)
             + 0.6*np.sin(2*np.pi*523.25*ct)
             + 0.5*np.sin(2*np.pi*659.25*ct))
    chord_env = np.minimum(ct/0.04, 1.0) * np.exp(-ct * 1.2)
    out[chord_start:] += chord * chord_env * 0.22

    # Layer 6: Multi-tap delay/echo — 3 attenuated copies for cathedral feel
    base = out.copy()
    for delay_s, amp in [(0.18, 0.50), (0.36, 0.28), (0.55, 0.15)]:
        d = int(delay_s * sr)
        if d < n:
            tail = np.zeros(n, dtype=np.float32)
            tail[d:] = base[:n-d] * amp
            out += tail

    # Soft-clip with tanh + normalize so loud transients don't crackle
    out = np.tanh(out * 1.1) * 0.85
    return (out * 32767).astype(np.int16), sr


# If a file is dropped in ./sounds/ matching one of these names, it overrides
# the synthesized chime. Drop a royalty-free 1.5–2.5s sci-fi boot sound here
# (Pixabay / Sonniss GameAudio / OpenGameArt are good free sources).
#
# IMPORTANT: must use _THIS_DIR (which handles frozen-mode correctly) and
# NOT Path(__file__).parent — when running as a PyInstaller exe, __file__
# resolves to the bundled extraction temp dir which doesn't have your sound
# files. _THIS_DIR is set to the exe's directory in frozen mode.
SOUNDS_DIR = _THIS_DIR / "sounds"
BOOT_SOUND_NAMES = ("boot.wav", "boot.mp3", "boot.ogg", "boot.flac")


def _find_boot_sound():
    """Return the path to a user-supplied boot sound file if one exists in
    ./sounds/, else None. Checks names in BOOT_SOUND_NAMES order."""
    if not SOUNDS_DIR.exists():
        return None
    for name in BOOT_SOUND_NAMES:
        p = SOUNDS_DIR / name
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def _broadcast_boot_start(duration_s: float):
    """Tell the HUD that the boot sound is starting NOW so it can sync the
    splash-screen animation. Best-effort — broadcast errors are swallowed
    so they never block the audio."""
    try:
        hud_server.broadcast_sync(json.dumps({
            "type": "boot_start",
            "duration_s": float(duration_s),
        }))
    except Exception as e:
        print(f"[chloe] boot_start broadcast failed: {e}")


def _broadcast_boot_end():
    """Signal the HUD that the boot sound has ended. The HUD typically fades
    the splash on its own duration timer, but this is a belt-and-suspenders
    fallback for when the sound finishes early or the duration is wrong."""
    try:
        hud_server.broadcast_sync(json.dumps({"type": "boot_end"}))
    except Exception as e:
        print(f"[chloe] boot_end broadcast failed: {e}")


def _play_boot_chime():
    """Play the boot sound via sounddevice. Blocks until done.
    Prefers ./sounds/boot.{wav,mp3,ogg,flac} if present; falls back to the
    synthesized numpy chime if no file is found or playback fails.

    Broadcasts boot_start (with duration) right before audio playback begins
    so the HUD splash screen can begin its animation in lockstep."""
    boot_file = _find_boot_sound()
    if boot_file is not None:
        try:
            import soundfile as sf
            import sounddevice as sd
            data, sr = sf.read(str(boot_file))
            dur = len(data) / sr
            print(f"[chloe] playing boot sound from file: {boot_file.name} ({dur:.2f}s)")
            _broadcast_boot_start(dur)
            sd.play(data, sr)
            sd.wait()
            _broadcast_boot_end()
            return
        except Exception as e:
            print(f"[chloe] boot file playback failed ({e}) — falling back to synth")
    try:
        import sounddevice as sd
        audio, sr = _generate_boot_chime()
        dur = len(audio) / sr
        print(f"[chloe] playing synth boot chime ({dur:.2f}s)")
        _broadcast_boot_start(dur)
        sd.play(audio, sr)
        sd.wait()
        _broadcast_boot_end()
    except Exception as e:
        print(f"[chloe] boot chime failed: {e}")


# ─── PUSH-TO-TALK STATE ─────────────────────────────────────────────────────
# Cross-thread events so the asyncio handlers (running in the WebSocket event
# loop) can signal the voice thread (running its own audio loop) to switch
# into PTT mode and back. Voice thread polls _ptt_mode between audio reads.
_ptt_mode        = threading.Event()  # set = PTT recording active
_ptt_stop_signal = threading.Event()  # set = stop the PTT recording now


def _ptt_record_phase(sd, device):
    """PTT-mode recording: open a fresh stream, record until _ptt_stop_signal
    is set (or PTT_MAX_S elapses), then transcribe + reply + speak."""
    print("[voice] PTT recording started", flush=True)
    hud_server.broadcast_sync("listening")
    audio = _record_until_signal(sd, device, _ptt_stop_signal, max_seconds=_PTT_MAX_S)
    min_samples = int(MIN_UTTERANCE_S * SAMPLE_RATE)
    if audio is None or len(audio) < min_samples:
        print("[voice] PTT utterance too short, ignoring", flush=True)
        hud_server.broadcast_sync("idle")
        return

    secs = len(audio) / SAMPLE_RATE
    print(f"[voice] PTT recorded {secs:.2f}s", flush=True)

    hud_server.broadcast_sync("thinking")
    transcript = _transcribe(audio)
    if not transcript:
        print("[voice] PTT empty transcript", flush=True)
        _speak_error("Sorry, I didn't catch that.")
        return
    print(f"[voice] PTT heard: {transcript!r}", flush=True)

    # "remember: <fact>" short-circuit — same as in _handle_wake.
    ack = _try_handle_remember(transcript)
    if ack is not None:
        _push_history("user", transcript, modality="voice")
        _push_history("assistant", ack, modality="voice")
        _broadcast_exchange(transcript, ack)
        hud_server.broadcast_sync("speaking")
        _speak(ack)
        hud_server.broadcast_sync("idle")
        print("[voice] PTT remember-ack complete", flush=True)
        return

    # Lights: "turn off the bedroom" / "set top light to 30%"
    lights_reply = try_handle_lights_command(transcript)
    if lights_reply is not None:
        _push_history("user", transcript, modality="voice")
        _push_history("assistant", lights_reply, modality="voice")
        _broadcast_exchange(transcript, lights_reply)
        hud_server.broadcast_sync("speaking")
        _speak(lights_reply)
        hud_server.broadcast_sync("idle")
        print("[voice] PTT lights-ack complete", flush=True)
        return

    reply = _ask_groq(transcript)
    if not reply:
        print("[voice] PTT got empty reply from Groq — aborting", flush=True)
        # Still surface the heard text in the chat panel so the user knows what was transcribed
        _broadcast_exchange(transcript, "[no reply — see terminal for error]")
        _speak_error("I'm having trouble reaching the server. Try again in a moment.")
        return
    print(f"[voice] PTT reply: {reply!r}", flush=True)

    # Broadcast the exchange to the HUD chat panel so it shows up there too
    _broadcast_exchange(transcript, reply)

    hud_server.broadcast_sync("speaking")
    _speak(reply)
    hud_server.broadcast_sync("idle")
    print("[voice] PTT cycle complete", flush=True)


def _record_until_signal(sd, device, stop_event, max_seconds=300):
    """Open a fresh InputStream and record until `stop_event` fires (or the
    safety cap kicks in). Used for push-to-talk: no silence detection, the
    user controls when recording ends."""
    max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)

    collected = []
    stream, native_rate = _open_input_stream_with_retry(sd, device)
    needs_resample = (native_rate != SAMPLE_RATE)
    src_block = stream.blocksize or CHUNK_SAMPLES

    with stream:
        for i in range(max_chunks):
            if stop_event.is_set():
                break
            try:
                audio_data, _overflow = stream.read(src_block)
            except Exception as e:
                print(f"[voice] read error in PTT phase: {e}")
                break
            np_chunk = np.frombuffer(audio_data, dtype=np.int16)
            if needs_resample:
                np_chunk = _resample_to_16k(np_chunk, native_rate)
                if len(np_chunk) < CHUNK_SAMPLES:
                    np_chunk = np.pad(np_chunk, (0, CHUNK_SAMPLES - len(np_chunk)))
                elif len(np_chunk) > CHUNK_SAMPLES:
                    np_chunk = np_chunk[:CHUNK_SAMPLES]
            np_chunk = _apply_gain(np_chunk)
            collected.append(np_chunk)

    if not collected:
        return None
    return np.concatenate(collected)


def _voice_thread_entry():
    """Top-level entry for the voice thread."""
    try:
        _voice_loop()
    except Exception:
        print("[voice] FATAL — voice loop crashed:")
        traceback.print_exc()


def _voice_loop():
    import sounddevice as sd

    try:
        devs = sd.query_devices()
        host_apis = sd.query_hostapis()
        default_in = sd.default.device[0] if sd.default.device else None
        print("[voice] audio input devices:")
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                marker = " ← DEFAULT" if i == default_in else ""
                api_name = host_apis[d["hostapi"]]["name"] if d.get("hostapi") is not None else "?"
                print(f"[voice]   [{i}] {d['name']}  ({api_name}, ch={d['max_input_channels']}, sr={int(d.get('default_samplerate', 0))}){marker}")
    except Exception as e:
        print(f"[voice] could not query devices: {e}")

    device = _resolve_mic_device(sd)
    if device is not None:
        print(f"[voice] using device override: {device}")

    wake = _create_wake_detector()
    if wake is None:
        return
    # Lift to module scope so _speak_* can spawn a barge-in monitor that
    # uses the same wake detector during TTS playback.
    global _wake_detector_global, _voice_device_global
    _wake_detector_global = wake
    _voice_device_global  = device

    print(f"[voice] ready — listening for '{wake['human']}' (threshold={WAKE_THRESHOLD})")

    # Brief sleep so the HUD WebSocket has a chance to connect before we play
    # any audio / broadcast any state.
    time.sleep(2.0)
    # Boot chime first, THEN the spoken greeting (so the chime concludes before
    # Chloe's voice starts).
    if BOOT_SOUND_ENABLED:
        _play_boot_chime()
    if GREETING_ENABLED:
        _greet_user()

    # Outer loop: dispatch to PTT recording if HUD requested push-to-talk,
    # otherwise listen for the wake word.
    #
    # Resilience to a missing/broken mic: after N consecutive crashes the
    # loop backs off exponentially up to ~60s between retries and emits
    # one log line per cycle instead of full tracebacks. Chat/HUD/wallet
    # paths don't depend on this loop, so we don't want a misconfigured
    # mic to bury those logs in spam.
    consecutive_failures = 0
    while True:
        try:
            if _ptt_mode.is_set():
                _ptt_record_phase(sd, device)
                _ptt_mode.clear()
                _ptt_stop_signal.clear()
            else:
                _wake_detect_phase(sd, device, wake)
            consecutive_failures = 0  # successful run resets the back-off
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures <= 3:
                print(f"[voice] voice phase crashed, restarting in 1s: {e}")
                traceback.print_exc()
                time.sleep(1.0)
            else:
                # Persistent failure — back off and stop the traceback spam.
                # Mic isn't coming back without intervention; log once per
                # back-off cycle and wait long enough for the rest of the
                # console to be readable.
                backoff = min(60.0, 2.0 ** (consecutive_failures - 3))
                if consecutive_failures == 4 or consecutive_failures % 10 == 0:
                    print(f"[voice] mic unavailable ({type(e).__name__}: {e}). "
                          f"Backing off for {backoff:.0f}s. Fix CHLOE_MIC in "
                          f".env and restart, OR plug the configured mic in. "
                          f"This will keep retrying silently.")
                time.sleep(backoff)


def _wake_detect_phase(sd, device, wake):
    """Open an InputStream sized to the detector's frame length, listen for the
    wake word, close + transition to recording when it fires. Returns when
    handle_wake completes (or on error). Works with both Porcupine (frame≈512)
    and openwakeword (frame=1280)."""
    if wake['engine'] == 'openwakeword':
        wake['handle'].reset()  # Porcupine has no reset

    frame_length = wake['frame_length']
    stream, native_rate = _open_input_stream_with_retry(
        sd, device, frame_length=frame_length
    )
    needs_resample = (native_rate != SAMPLE_RATE)
    src_block = stream.blocksize or frame_length

    with stream:
        while True:
            # Bail out promptly if HUD just requested PTT — outer loop will
            # then enter _ptt_record_phase with a fresh stream.
            if _ptt_mode.is_set():
                return
            try:
                audio_data, overflow = stream.read(src_block)
            except Exception as e:
                print(f"[voice] read error in wake phase: {e}")
                return  # let outer loop reopen

            np_chunk = np.frombuffer(audio_data, dtype=np.int16)
            if needs_resample:
                np_chunk = _resample_to_16k(np_chunk, native_rate)
                if len(np_chunk) < frame_length:
                    np_chunk = np.pad(np_chunk, (0, frame_length - len(np_chunk)))
                elif len(np_chunk) > frame_length:
                    np_chunk = np_chunk[:frame_length]

            np_chunk = _apply_gain(np_chunk)
            if wake['predict'](np_chunk):
                print(f"[voice] WAKE detected")
                # Audible + visual confirmation that wake fired, distinct
                # from steady-state listening. Backend broadcasts a brief
                # "wake_acked" state which the HUD renders as a green
                # flash; a small chirp plays so the user knows audibly too.
                if CHIRP_ON_WAKE:
                    try: _play_wake_chirp()
                    except Exception: pass
                hud_server.broadcast_sync("wake_acked")
                break

    _handle_wake(sd, device)


CHIRP_ON_WAKE = os.environ.get("CHLOE_WAKE_CHIRP", "1").strip() != "0"


def _play_wake_chirp():
    """~150ms two-tone confirmation chime when the wake word fires. Plays
    asynchronously (no sd.wait) so it doesn't add latency before recording."""
    try:
        import sounddevice as sd
    except ImportError:
        return
    sr = 44100
    dur = 0.15
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    # Quick rising arpeggio: feels like an "I'm listening" prompt.
    freq = np.where(t < 0.07, 660.0, 880.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.minimum(t / 0.01, 1.0) * np.exp(-t * 5.0)
    sig = np.sin(phase) * env * 0.22
    audio = (sig * 32767).astype(np.int16)
    try:
        sd.play(audio, sr)  # don't .wait() — overlap with the recording stream
    except Exception:
        pass


def _next_turn_audio(sd, device):
    """After Chloe finishes speaking, return audio for the next turn if one
    starts immediately, else None. Two triggers:
      1. Barge-in via wake — user said the wake word during speech. Record
         the new utterance now (skipping the wake-detection phase).
      2. Follow-up mode (CHLOE_FOLLOWUP=1) — brief listen window for a
         spontaneous follow-up question without re-saying the wake word.
    """
    if _barge_in_via_wake.is_set():
        _barge_in_via_wake.clear()
        _barge_in_request.clear()
        print("[voice] barge-in via wake — re-recording", flush=True)
        hud_server.broadcast_sync("listening")
        return _record_utterance(sd, device)
    if FOLLOWUP_ENABLED:
        try:
            print(f"[voice] follow-up window ({FOLLOWUP_LISTEN_S}s)…", flush=True)
            hud_server.broadcast_sync("followup")
            audio = _record_utterance(
                sd, device, no_voice_timeout_s=FOLLOWUP_LISTEN_S
            )
            if audio is not None:
                hud_server.broadcast_sync("listening")
                print("[voice] follow-up captured — processing", flush=True)
                return audio
            print("[voice] follow-up window elapsed without voice", flush=True)
            # Reset HUD: we set "followup" above but never picked up audio.
            # Without this the orb stays stuck on the follow-up state.
            hud_server.broadcast_sync("idle")
        except Exception as e:
            print(f"[voice] follow-up listen error: {e}", flush=True)
            hud_server.broadcast_sync("idle")
    return None


# Trust signal: when a step fails (no transcript, empty Groq reply, etc.)
# Chloe says a short error out loud instead of going silent. Set
# CHLOE_ERROR_SPEECH=0 in _env to disable.
ERROR_SPEECH_ENABLED = os.environ.get("CHLOE_ERROR_SPEECH", "1").strip() != "0"


def _speak_error(short_msg: str) -> None:
    """Speak a short error message and update the HUD state. Wrapped so a
    failing TTS itself doesn't cascade into another exception."""
    if not ERROR_SPEECH_ENABLED or not short_msg:
        return
    print(f"[voice] error speech: {short_msg!r}", flush=True)
    try:
        hud_server.broadcast_sync("speaking")
        _speak(short_msg)
    except Exception as e:
        print(f"[voice] error-speech TTS failed: {e}", flush=True)
    finally:
        hud_server.broadcast_sync("idle")


def _process_voice_turn(audio, sd, device) -> bool:
    """Process one user utterance: transcribe, handle remember-command, run
    LLM, speak the reply. Returns True on a successful completed turn (the
    caller may attempt a follow-up); False on any failure that should drop
    the conversation back to wake-detection."""
    min_samples = int(MIN_UTTERANCE_S * SAMPLE_RATE)
    if audio is None:
        # Wake fired but no voice followed — likely a false-positive on the
        # wake model. Stay silent (don't speak an error) since the user
        # may not even know they "triggered" anything.
        print("[voice] no voice captured this turn", flush=True)
        return False
    if len(audio) < min_samples:
        secs = len(audio) / SAMPLE_RATE
        print(f"[voice] utterance too short ({secs:.2f}s < {MIN_UTTERANCE_S}s), ignoring", flush=True)
        return False

    hud_server.broadcast_sync("thinking")
    transcript = _transcribe(audio)
    if not transcript:
        # Both Groq + local Whisper failed (or local isn't installed).
        # Most likely network is out OR the user said something Whisper
        # couldn't decode. Tell them so they don't keep waiting.
        print("[voice] empty transcript from all STT paths", flush=True)
        _speak_error("Sorry, I didn't catch that.")
        return False
    print(f"[voice] heard: {transcript!r}", flush=True)

    # "remember: <fact>" short-circuits the LLM path entirely.
    ack = _try_handle_remember(transcript)
    if ack is not None:
        _push_history("user", transcript, modality="voice")
        _push_history("assistant", ack, modality="voice")
        _broadcast_exchange(transcript, ack)
        hud_server.broadcast_sync("speaking")
        _speak(ack)
        hud_server.broadcast_sync("idle")
        return True

    # Lights: "turn off the bedroom" / "set top light to 30%"
    lights_reply = try_handle_lights_command(transcript)
    if lights_reply is not None:
        _push_history("user", transcript, modality="voice")
        _push_history("assistant", lights_reply, modality="voice")
        _broadcast_exchange(transcript, lights_reply)
        hud_server.broadcast_sync("speaking")
        _speak(lights_reply)
        hud_server.broadcast_sync("idle")
        return True

    reply = _ask_groq(transcript)
    if not reply:
        # _ask_groq logs the underlying error; surface it audibly so the
        # user knows we heard them but couldn't answer.
        print("[voice] got empty reply from Groq", flush=True)
        _broadcast_exchange(transcript, "[no reply — see terminal for error]")
        _speak_error("I'm having trouble reaching the server. Try again in a moment.")
        return False
    print(f"[voice] reply: {reply!r}", flush=True)

    _broadcast_exchange(transcript, reply)
    hud_server.broadcast_sync("speaking")
    _speak(reply)
    hud_server.broadcast_sync("idle")
    return True


def _handle_wake(sd, device):
    """Wake fired: process turns until the conversation naturally ends.
    Initial turn comes from a fresh recording; subsequent turns come from
    barge-in via wake OR follow-up mode (when enabled)."""
    hud_server.broadcast_sync("listening")
    audio = _record_utterance(sd, device)
    while True:
        if not _process_voice_turn(audio, sd, device):
            hud_server.broadcast_sync("idle")
            return
        audio = _next_turn_audio(sd, device)
        if audio is None:
            # Belt-and-suspenders: _next_turn_audio already resets to idle on
            # the no-follow-up path, but a barge-in branch that fails to
            # record could exit here with the HUD stuck on "listening".
            hud_server.broadcast_sync("idle")
            return


def _record_utterance(sd, device, no_voice_timeout_s: float | None = None):
    """Open a fresh InputStream and record until silence persists for
    SILENCE_HANG_MS. Trims leading silence so Whisper doesn't hallucinate.

    If `no_voice_timeout_s` is provided and no voice is detected within that
    many seconds, exit early and return None. Used by follow-up mode to give
    up quickly if the user has nothing more to say.
    """
    silence_chunks_needed = max(1, int((SILENCE_HANG_MS / 1000) * SAMPLE_RATE / CHUNK_SAMPLES))
    max_chunks = int(MAX_RECORD_S * SAMPLE_RATE / CHUNK_SAMPLES)
    if no_voice_timeout_s is not None:
        # Number of chunks after which, if no voice has been heard, we bail.
        no_voice_chunk_limit = int(no_voice_timeout_s * SAMPLE_RATE / CHUNK_SAMPLES)
    else:
        no_voice_chunk_limit = None

    collected = []
    silent_run = 0
    saw_voice = False
    peak_rms = 0.0
    log_every = 12
    first_voice_idx = -1

    stream, native_rate = _open_input_stream_with_retry(sd, device)
    needs_resample = (native_rate != SAMPLE_RATE)
    src_block = stream.blocksize or CHUNK_SAMPLES

    with stream:
        for i in range(max_chunks):
            try:
                audio_data, _overflow = stream.read(src_block)
            except Exception as e:
                print(f"[voice] read error in record phase: {e}")
                break
            np_chunk = np.frombuffer(audio_data, dtype=np.int16)
            if needs_resample:
                np_chunk = _resample_to_16k(np_chunk, native_rate)
                if len(np_chunk) < CHUNK_SAMPLES:
                    np_chunk = np.pad(np_chunk, (0, CHUNK_SAMPLES - len(np_chunk)))
                elif len(np_chunk) > CHUNK_SAMPLES:
                    np_chunk = np_chunk[:CHUNK_SAMPLES]
            np_chunk = _apply_gain(np_chunk)
            collected.append(np_chunk)
            rms = float(np.sqrt(np.mean((np_chunk.astype(np.float32) / 32768.0) ** 2)))
            peak_rms = max(peak_rms, rms)

            if VOICE_DEBUG and i % log_every == 0:
                voiced = "VOICE" if rms > SILENCE_RMS else "quiet"
                print(f"[voice]   rms={rms:.4f} (thr={SILENCE_RMS}) {voiced}  saw_voice={saw_voice} silent_run={silent_run}")

            if rms > SILENCE_RMS:
                if not saw_voice:
                    first_voice_idx = i
                saw_voice = True
                silent_run = 0
            else:
                silent_run += 1
                if saw_voice and silent_run >= silence_chunks_needed:
                    break
                # Follow-up mode: bail if no voice heard within the window.
                if (no_voice_chunk_limit is not None
                        and not saw_voice
                        and i >= no_voice_chunk_limit):
                    break

    secs = len(collected) * CHUNK_SAMPLES / SAMPLE_RATE
    if VOICE_DEBUG:
        print(f"[voice] recorded {secs:.2f}s, peak_rms={peak_rms:.4f}, saw_voice={saw_voice}")

    if not saw_voice or not collected:
        return None

    # Trim leading silence — keep a small lead-in (200ms) for Whisper context
    leading_keep = max(0, int(LEADING_TRIM_SECS * SAMPLE_RATE / CHUNK_SAMPLES))
    trim_start = max(0, first_voice_idx - leading_keep)
    body = collected[trim_start:]
    return np.concatenate(body)


def _transcribe_groq(audio_np: np.ndarray) -> str:
    """PCM int16 numpy → Groq Whisper → text."""
    if not _sync_groq:
        return ""
    secs = len(audio_np) / SAMPLE_RATE
    print(f"[voice] transcribing {secs:.2f}s of audio (Groq)…", flush=True)
    t0 = time.time()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_np.tobytes())
    buf.seek(0)
    try:
        result = _sync_groq.with_options(timeout=120.0).audio.transcriptions.create(
            file=("utterance.wav", buf.read()),
            model=MODEL_STT,
            response_format="text",
            language="en",
        )
        text = result if isinstance(result, str) else getattr(result, "text", "")
        text = text.strip()
        dt = time.time() - t0
        print(f"[voice] Groq transcribed in {dt:.2f}s ({len(text)} chars)", flush=True)
        return text
    except Exception as e:
        dt = time.time() - t0
        print(f"[voice] Groq transcribe error after {dt:.2f}s: {type(e).__name__}: {e}", flush=True)
        return ""


# ─── LOCAL STT FALLBACK (faster-whisper) ─────────────────────────────────────
# Activated only if Groq Whisper fails (network out, rate-limited, etc.).
# Set CHLOE_LOCAL_STT=0 to disable. CHLOE_LOCAL_WHISPER_SIZE picks the model
# (tiny.en / base.en / small.en / medium.en — bigger = slower but more
# accurate; "base.en" is a sensible default at ~140MB).
#
# Note on Python compat: faster-whisper depends on CTranslate2, which
# typically takes a few months after each new Python release to ship wheels.
# If install fails, this layer simply stays disabled — Groq remains the
# primary path and nothing breaks.
LOCAL_STT_ENABLED      = os.environ.get("CHLOE_LOCAL_STT", "1").strip() != "0"
LOCAL_WHISPER_SIZE     = os.environ.get("CHLOE_LOCAL_WHISPER_SIZE", "base.en").strip()
_local_whisper_model   = None
_local_whisper_tried   = False


def _get_local_whisper():
    """Lazy-load faster-whisper. Returns the model or None if unavailable.
    Cached after the first successful load OR after the first failure."""
    global _local_whisper_model, _local_whisper_tried
    if _local_whisper_tried:
        return _local_whisper_model
    _local_whisper_tried = True
    if not LOCAL_STT_ENABLED:
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        if VOICE_DEBUG:
            print("[voice] local STT: faster-whisper not installed (Groq only)",
                  flush=True)
        return None
    try:
        print(f"[voice] loading local Whisper model ({LOCAL_WHISPER_SIZE})…",
              flush=True)
        _local_whisper_model = WhisperModel(
            LOCAL_WHISPER_SIZE, device="cpu", compute_type="int8"
        )
        print(f"[voice] local Whisper ready", flush=True)
    except Exception as e:
        print(f"[voice] local Whisper load failed: {type(e).__name__}: {e}",
              flush=True)
        _local_whisper_model = None
    return _local_whisper_model


def _transcribe_local(audio_np: np.ndarray) -> str:
    """faster-whisper transcription. Returns text or empty string."""
    model = _get_local_whisper()
    if model is None:
        return ""
    t0 = time.time()
    try:
        # faster-whisper takes float32 normalized to [-1, 1] OR a file path.
        audio_f32 = audio_np.astype(np.float32) / 32768.0
        segments, _info = model.transcribe(
            audio_f32, beam_size=1, language="en", vad_filter=False,
        )
        text = " ".join(seg.text for seg in segments).strip()
        dt = time.time() - t0
        print(f"[voice] local Whisper: {dt:.2f}s ({len(text)} chars)", flush=True)
        return text
    except Exception as e:
        dt = time.time() - t0
        print(f"[voice] local Whisper error after {dt:.2f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        return ""


def _transcribe(audio_np: np.ndarray) -> str:
    """Try Groq Whisper first; on empty/failure, fall back to local
    faster-whisper if available. Logged transparently so you can see in
    the terminal which path produced each transcript."""
    text = _transcribe_groq(audio_np)
    if text:
        return text
    if LOCAL_STT_ENABLED:
        print("[voice] Groq returned empty — trying local Whisper fallback",
              flush=True)
        return _transcribe_local(audio_np)
    return ""

# ─── LIVE CODE READING ──────────────────────────────────────────────────────
# Tool the Llama model can call when the user asks about Chloe's own
# implementation. Quoting actual code is more useful than confabulating
# from memory. Compound-mini has its own tool framework — we don't add
# this tool to that path, only to the regular MODEL_TEXT one.
GREP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep_source",
        "description": (
            "Search Chloe's own Python source files for a regex pattern. "
            "Use this whenever the user asks about Chloe's implementation, "
            "behaviour, or config. ALWAYS prefer a real grep over guessing.\n\n"
            "Project conventions to know:\n"
            "- Internal helpers are prefixed with an underscore: `_speak`, "
            "`_speak_elevenlabs`, `_speak_kokoro`, `_speak_edge_tts`, "
            "`_ask_groq`, `_transcribe`, `_voice_loop`, etc.\n"
            "- Async WebSocket handlers are NOT prefixed: `handle_chat`, "
            "`handle_ptt_audio`, `handle_ptt_start`, `handle_ptt_stop`.\n"
            "- Configuration env vars are SCREAMING_SNAKE: `USE_KOKORO`, "
            "`USE_ELEVENLABS`, `KOKORO_VOICE`, `CHLOE_OLLAMA_PRIMARY`, "
            "`CHLOE_MIC_GAIN`, `MODEL_TEXT`, `MODEL_SEARCH`.\n\n"
            "If your first grep returns no matches, RETRY with a different "
            "pattern (broader, or the underscore-prefix convention). Do NOT "
            "guess from memory after a 0-match result.\n\n"
            "Returns matches as 'filename:lineno: code'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Python regex pattern. Good first picks for common "
                        "questions: TTS engine selection → '_speak', mic / "
                        "wake config → 'CHLOE_MIC|WAKE', chat routing → "
                        "'_pick_route|OLLAMA_PRIMARY', PTT path → 'handle_ptt'."
                    ),
                },
                "file": {
                    "type": "string",
                    "description": (
                        "Optional. Specific .py file to restrict the search to "
                        "(e.g., 'jarvis.py', 'chloe_memory.py', 'hud_server.py'). "
                        "Omit to search all .py files in the project root."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
}


# ─── BITCOIN LIGHTNING WALLET TOOLS ─────────────────────────────────────────
# Four tools that surface the Breez SDK Liquid wallet (wallet.py) through
# the Chloe LLM, gated by wallet_guard.py for spends. See WALLET_PLAN.md
# for the full security model. Hard rules enforced server-side here:
#   - wallet_send REQUIRES a `pin` argument every call (no caching).
#   - wallet_send routes through wallet_guard.authorize_send before any
#     SDK call, so a confused or jailbroken LLM can't bypass the cap.
#   - The PIN argument is redacted from stored tool-call args before
#     re-feeding context to the next inference round.
WALLET_BALANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wallet_balance",
        "description": (
            "Check the user's Bitcoin Lightning wallet balance. "
            "Returns spendable sats plus any pending send/receive amounts. "
            "Sats are 1/100,000,000 of a bitcoin. Quote the number of sats "
            "directly and ALWAYS include 'sats' as the unit when speaking "
            "the result. Do NOT convert to USD unless explicitly asked."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

WALLET_INVOICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wallet_invoice",
        "description": (
            "Create a Lightning invoice (BOLT11) the user can give to a "
            "payer to receive a payment. Use when the user asks to "
            "'create an invoice', 'request a payment', or 'get paid'. "
            "The full bolt11 string is automatically pushed to Ed's "
            "Windows clipboard; the tool only returns a short preview. "
            "DO NOT speak the bolt11 string or its preview aloud — TTS "
            "cannot phonemise it. Just say something like 'Invoice for "
            "N sats created and copied to your clipboard.'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "amount_sat": {
                    "type": "integer",
                    "description": "Amount in sats. Must be a positive integer.",
                },
                "memo": {
                    "type": "string",
                    "description": "Optional short description (what this is for).",
                },
            },
            "required": ["amount_sat"],
        },
    },
}

WALLET_SEND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wallet_send",
        "description": (
            "Send a Lightning payment to a BOLT11 invoice or a Lightning "
            "Address (alice@example.com). The user MUST provide a PIN; "
            "do not invent one. Subject to a daily spend cap. If the "
            "user has not given a PIN this turn, ASK for it before "
            "calling this tool — do not call without one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": (
                        "BOLT11 invoice (starts 'lnbc'), BOLT12 offer "
                        "(starts 'lno'), or Lightning Address ('user@host')."
                    ),
                },
                "amount_sat": {
                    "type": "integer",
                    "description": (
                        "Amount in sats. Required for amountless invoices "
                        "and Lightning Addresses; ignored for amount-fixed "
                        "invoices."
                    ),
                },
                "pin": {
                    "type": "string",
                    "description": (
                        "The user's wallet PIN. The user provides this; "
                        "do not guess or reuse a previous PIN. If absent, "
                        "ask the user, then call this tool with the value "
                        "they give."
                    ),
                },
            },
            "required": ["destination", "pin"],
        },
    },
}

WALLET_HISTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wallet_history",
        "description": (
            "List recent wallet payments (sends and receives). Use when "
            "the user asks 'what was my last payment', 'show recent "
            "transactions', or similar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return. Default 5, cap 50.",
                },
            },
        },
    },
}

WALLET_TOOL_SCHEMAS = {
    "wallet_balance": WALLET_BALANCE_SCHEMA,
    "wallet_invoice": WALLET_INVOICE_SCHEMA,
    "wallet_send":    WALLET_SEND_SCHEMA,
    "wallet_history": WALLET_HISTORY_SCHEMA,
}
WALLET_TOOL_NAMES = set(WALLET_TOOL_SCHEMAS.keys())


def _wallet_module():
    """Lazy-import wallet.py. Returns None if the module is missing or
    breez-sdk-liquid isn't installed. We wrap so a missing wallet
    doesn't block all of Chloe — voice/chat keep working, the wallet
    tools just return an unconfigured-error message."""
    try:
        import wallet  # type: ignore
        return wallet
    except Exception as e:  # ImportError or downstream SDK missing
        print(f"[chloe] wallet module unavailable: {type(e).__name__}: {e}",
              flush=True)
        return None


def _wallet_guard_module():
    try:
        import wallet_guard  # type: ignore
        return wallet_guard
    except Exception as e:
        print(f"[chloe] wallet_guard module unavailable: "
              f"{type(e).__name__}: {e}", flush=True)
        return None


def _wallet_dispatch(name: str, args: dict) -> str:
    """Route a wallet_* tool call to the right Python function and
    return a human-readable result string for the LLM to consume."""
    if not isinstance(args, dict):
        args = {}
    w = _wallet_module()
    if w is None:
        return ("Wallet is not configured. The breez-sdk-liquid package "
                "isn't installed or the wallet module is missing. See "
                "WALLET_SETUP.md for setup steps.")
    try:
        if name == "wallet_balance":
            r = w.get_balance()
            if not r.get("ok"):
                return f"Wallet error: {r.get('error', 'unknown')}"
            return json.dumps({
                "balance_sat":          r["balance_sat"],
                "pending_send_sat":     r["pending_send_sat"],
                "pending_receive_sat":  r["pending_receive_sat"],
            })

        if name == "wallet_invoice":
            amount = args.get("amount_sat")
            # Coerce stringy amounts (small models like to send "1000"
            # instead of 1000) — but reject if it's nonsense.
            if isinstance(amount, str) and amount.strip().isdigit():
                amount = int(amount.strip())
            if not isinstance(amount, int) or amount < 1:
                return "Wallet error: amount_sat must be a positive integer."
            memo = str(args.get("memo") or "")
            r = w.create_invoice(amount, memo)
            if not r.get("ok"):
                return f"Wallet error: {r.get('error', 'unknown')}"
            bolt11_full = r["bolt11"] or ""
            # Push the full bolt11 to the Windows clipboard so Ed can
            # paste it wherever needed. Then return only a short PREVIEW
            # to the LLM — Kokoro chokes phonemising a 200+ char lnbc
            # string (510-phoneme limit), and the LLM tends to read the
            # whole thing aloud anyway despite system-prompt nudges.
            clipboard_status = _clipboard_set(bolt11_full)
            preview = (bolt11_full[:18] + "…" + bolt11_full[-8:]
                       if len(bolt11_full) > 30 else bolt11_full)
            return json.dumps({
                "bolt11_preview": preview,
                "amount_sat":     r["amount_sat"],
                "fees_sat":       r["fees_sat"],
                "memo":           r["memo"],
                "clipboard":      clipboard_status,
                "speak_hint":     ("Tell Ed the invoice was created and the "
                                   "full bolt11 is on his clipboard. Do NOT "
                                   "speak the bolt11 string — only describe "
                                   "the amount, memo, and fees."),
            })

        if name == "wallet_send":
            dest = str(args.get("destination") or "").strip()
            amount = args.get("amount_sat")
            pin = args.get("pin") or ""
            if not dest:
                return "Wallet error: destination is required."
            # Resolve amount: if invoice has amount baked in, the SDK will
            # ignore amount_sat. Some destinations (LNURL, amountless
            # invoices) require it. We pass through what we got and let
            # the SDK / our pay() helper do the right thing — but for
            # the cap check we need a number, so default to 0 if absent
            # and make authorize_send block in that case.
            check_amount = amount if isinstance(amount, int) and amount > 0 else 0
            wg = _wallet_guard_module()
            if wg is None:
                return ("Wallet send refused: wallet_guard module is "
                        "unavailable, cannot enforce PIN/cap policy.")
            if check_amount == 0:
                # Invoice may have a baked-in amount, but for the cap
                # check we need a number. Try to extract it from the
                # bolt11 prep call before authorising.
                try:
                    import breez_sdk_liquid as bsl  # type: ignore
                    sdk = w._connect()
                    prep = sdk.prepare_send_payment(
                        bsl.PrepareSendRequest(destination=dest)
                    )
                    check_amount = w._extract_resolved_amount(prep, fallback=0)
                except Exception:
                    pass
            if check_amount == 0:
                return ("Wallet send refused: cannot determine the amount "
                        "to send. Ask the user to specify amount_sat.")
            ok, reason = wg.authorize_send(check_amount, str(pin))
            # Clear the PIN from the user's last history entry whether
            # the auth succeeded or failed — once attempted, the PIN
            # has no business sticking around in subsequent context.
            _scrub_pin_from_last_user_turn()
            if not ok:
                return f"Wallet send refused: {reason}"
            r = w.pay(dest, amount if isinstance(amount, int) else None)
            if not r.get("ok"):
                return f"Wallet send failed: {r.get('error', 'unknown')}"
            try:
                wg.record_send(int(r.get("amount_sat") or check_amount),
                               r.get("payment_hash"))
            except Exception:
                pass
            return json.dumps({
                "ok":            True,
                "amount_sat":    r.get("amount_sat"),
                "fees_sat":      r.get("fees_sat"),
                "status":        r.get("status"),
                "payment_hash":  r.get("payment_hash"),
            })

        if name == "wallet_history":
            limit = args.get("limit")
            # Models sometimes pass numbers as strings ("10"). Coerce
            # before falling back to default.
            if isinstance(limit, str) and limit.strip().isdigit():
                limit = int(limit.strip())
            if not isinstance(limit, int) or limit < 1:
                limit = 5
            r = w.list_history(min(limit, 50))
            if not r.get("ok"):
                return f"Wallet error: {r.get('error', 'unknown')}"
            return json.dumps({"payments": r["payments"]})

        return f"unknown wallet tool: {name}"
    except Exception as e:
        traceback.print_exc()
        return f"Wallet error: {type(e).__name__}: {e}"


def _redact_pin_in_args_str(args_str, name):
    """Used on the Groq path where args arrive as a JSON string. Returns
    a redacted JSON string with `pin` masked, suitable for re-injecting
    into the assistant message we persist."""
    if name != "wallet_send":
        return args_str
    try:
        obj = json.loads(args_str or "{}")
    except Exception:
        return args_str
    if isinstance(obj, dict) and "pin" in obj:
        obj["pin"] = "<redacted>"
        return json.dumps(obj)
    return args_str


def _redact_pin_in_args_dict(args, name):
    """Used on the Ollama path where args are already a dict."""
    if name != "wallet_send":
        return args
    if isinstance(args, dict) and "pin" in args:
        out = dict(args); out["pin"] = "<redacted>"
        return out
    return args


def _clipboard_set(text: str) -> str:
    """Push `text` onto the Windows clipboard via clip.exe (built-in).
    Returns a short status string suitable for the LLM tool result.
    No-ops cleanly on non-Windows or if clip.exe isn't reachable."""
    if not isinstance(text, str) or not text:
        return "skipped (empty)"
    if os.name != "nt":
        return "unavailable (not Windows)"
    try:
        import subprocess as _sp
        # clip.exe accepts UTF-16 LE on Windows; for plain ASCII (bolt11)
        # both work, but UTF-16 is the documented contract.
        _sp.run(
            ["clip"],
            input=text.encode("utf-16-le"),
            check=True,
            timeout=2,
            shell=False,
        )
        return f"copied ({len(text)} chars)"
    except Exception as e:
        return f"copy failed: {type(e).__name__}"


def _scrub_pin_from_last_user_turn():
    """Walk back through _voice_history and mask any 'pin <digits>' style
    token in the most recent user message. Best-effort: protects against
    the model echoing the PIN in its next round when the user said the
    PIN out loud. The canonical PIN handling is via the tool argument."""
    import re as _re
    if not _voice_history:
        return
    for i in range(len(_voice_history) - 1, -1, -1):
        if _voice_history[i].get("role") == "user":
            text = _voice_history[i].get("content") or ""
            scrubbed = _re.sub(
                r"(\bpin\s*[:=]?\s*)(\S+)",
                r"\1<redacted>",
                text,
                flags=_re.IGNORECASE,
            )
            if scrubbed != text:
                _voice_history[i] = dict(_voice_history[i],
                                          content=scrubbed)
            break


def _grep_source(pattern, file=None) -> str:
    """Search Chloe's own .py source for a regex pattern. Used as the
    backing implementation for the `grep_source` tool call (both Groq
    and Ollama paths).

    Returns formatted matches "<filename>:<lineno>: <code>", capped at
    50 lines so the tool result doesn't blow the context window.

    Defensive on inputs because small models (notably llama3.2:3b)
    sometimes echo the JSON schema as the argument rather than picking
    a value, e.g. pattern = {"type": "string", "description": "..."}.
    We coerce / reject non-string args instead of crashing the turn."""
    import re as _re

    # Coerce / validate pattern. If the model passed a dict (echoed schema),
    # try to recover a sensible value before giving up.
    if isinstance(pattern, dict):
        # Common confused shapes: {"value": "X"} or {"pattern": "X"}
        for k in ("value", "pattern", "regex", "default"):
            v = pattern.get(k)
            if isinstance(v, str) and v:
                pattern = v
                break
        else:
            return ("Tool error: pattern arrived as a JSON object instead of a "
                    "string. Pass the pattern directly, e.g. "
                    'grep_source(pattern="def _speak").')
    if not isinstance(pattern, str):
        return f"Tool error: pattern must be a string, got {type(pattern).__name__}."
    pattern = pattern.strip()
    if not pattern:
        return "Empty pattern."
    try:
        rx = _re.compile(pattern)
    except _re.error as e:
        return f"Invalid regex: {e}"

    # Same defensiveness for the optional `file` arg.
    if isinstance(file, dict):
        for k in ("value", "file", "name", "filename", "default"):
            v = file.get(k)
            if isinstance(v, str) and v:
                file = v
                break
        else:
            file = None
    if file is not None and not isinstance(file, str):
        file = None

    base = Path(__file__).parent
    if file:
        # Don't allow path traversal or absolute paths — restrict to the
        # project folder. Strip any leading slashes/dots and require .py.
        clean = Path(file).name  # discards directory components
        target = base / clean
        if not target.exists() or target.suffix != ".py":
            return f"File not found in project: {clean}"
        targets = [target]
    else:
        targets = sorted(base.glob("*.py"))

    matches = []
    MAX_MATCHES = 50
    for path in targets:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if rx.search(line):
                        matches.append(f"{path.name}:{lineno}: {line.rstrip()}")
                        if len(matches) >= MAX_MATCHES:
                            break
        except Exception as e:
            matches.append(f"[error reading {path.name}: {e}]")
        if len(matches) >= MAX_MATCHES:
            break

    if not matches:
        # Help the model retry intelligently: include a list of actual
        # `def` / `async def` names from the searched files. Models like
        # llama3.1:8b will often guess wrong on the first pattern (e.g.
        # 'handle_tts' when the real function is `_speak`); seeing the
        # real inventory lets them retry with a sensible pattern instead
        # of falling back to guessing.
        defs = []
        for path in targets:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("def ") or s.startswith("async def "):
                            # Extract just the name
                            after = s.split("def ", 1)[1]
                            name = after.split("(", 1)[0].strip()
                            if name:
                                defs.append(name)
            except Exception:
                pass
        defs = sorted(set(defs))[:80]  # cap so the response stays manageable
        hint = ""
        if defs:
            hint = (f"\n\nFunctions defined in the searched files (use these "
                    f"to pick a better pattern and retry):\n  "
                    + ", ".join(defs))
        return (f"No matches found for /{pattern}/ in {len(targets)} file(s).{hint}")
    head = (f"Showing first {MAX_MATCHES} of many matches:\n"
            if len(matches) >= MAX_MATCHES
            else f"Found {len(matches)} match(es):\n")
    return head + "\n".join(matches)


def _groq_chat_attempt(user_text: str, model: str) -> str:
    """Run one Groq inference attempt with the given model. Doesn't mutate
    history; returns the reply text or empty string on failure. Handles the
    inner 413 / rate-limit retry loop, AND a tool-call loop for our
    custom grep_source tool (only enabled on MODEL_TEXT — compound-mini
    has its own tool framework, we don't double up).

    Used by `_ask_groq` for both the initial attempt and the hedge-retry
    pass through compound-mini."""
    if not _sync_groq:
        return ""
    msgs = [{"role": "system",
             "content": _augmented_voice_system(model, user_text)}] + _voice_history
    msgs = _trim_messages_for_model(msgs, model)

    use_custom_tools = (model != MODEL_SEARCH)
    MAX_TOOL_ITERS = 3  # cap tool-call rounds so a confused model can't loop forever

    t0 = time.time()
    resp = None
    m = None  # latest assistant message; populated each tool iteration

    for tool_iter in range(MAX_TOOL_ITERS + 1):
        # ── single API call with the existing 413 / rate-limit retry loop ──
        attempts = 0
        resp = None
        while attempts < 3:
            attempts += 1
            try:
                kwargs = {
                    "model":       model,
                    "messages":    msgs,
                    "max_tokens":  400,
                    "temperature": 0.7,
                }
                if use_custom_tools:
                    kwargs["tools"] = [GREP_TOOL_SCHEMA,
                                       *WALLET_TOOL_SCHEMAS.values()]
                resp = _sync_groq.with_options(timeout=90.0).chat.completions.create(**kwargs)
                break
            except Exception as e:
                if _is_too_large_error(e) and attempts < 3:
                    print(f"[voice] 413 too-large on {model}; trimming history and retrying", flush=True)
                    msgs = _trim_messages_for_model(msgs, model, max_msgs=2)
                    continue
                wait = _extract_retry_after(e)
                if wait is not None and attempts < 3:
                    pad = 0.5
                    wait_total = min(wait + pad, 30.0)
                    print(f"[voice] rate-limited on {model}; waiting {wait_total:.1f}s and retrying…", flush=True)
                    time.sleep(wait_total)
                    continue
                dt = time.time() - t0
                print(f"[voice] groq error after {dt:.2f}s on {model}: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                return ""
        if resp is None:
            return ""

        m = resp.choices[0].message
        tool_calls = getattr(m, "tool_calls", None) or []
        if not tool_calls:
            break  # done — model produced its final reply with no further tool needs

        # Cap reached — bail out of the loop and use whatever content the model produced
        if tool_iter == MAX_TOOL_ITERS:
            print(f"[voice] tool-call loop hit max iterations ({MAX_TOOL_ITERS}); bailing", flush=True)
            break

        # Append the assistant message (with tool_calls) so the next round
        # has the right context. PIN is redacted in the persisted copy
        # so a successful wallet_send doesn't leak the PIN into future
        # context windows.
        msgs.append({
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": _redact_pin_in_args_str(
                            tc.function.arguments, tc.function.name
                        ),
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute each tool call, append result as a `tool` role message.
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            name = tc.function.name
            if name == "grep_source":
                result = _grep_source(args.get("pattern", ""), args.get("file"))
                preview = (args.get("pattern") or "")[:60]
                print(f"[voice]   tool grep_source(/{preview}/, file={args.get('file')!r})"
                      f" → {len(result)} chars", flush=True)
            elif name in WALLET_TOOL_NAMES:
                result = _wallet_dispatch(name, args)
                # Logging: never print the PIN, even truncated.
                safe_args = {k: ("<redacted>" if k == "pin" else v)
                             for k, v in args.items()}
                print(f"[voice]   tool {name}({safe_args})"
                      f" → {len(result)} chars", flush=True)
            else:
                result = f"unknown tool: {name}"
                print(f"[voice]   tool {name!r} requested but not implemented", flush=True)
            # Cap tool result so it doesn't blow the context window
            msgs.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result[:4000],
            })

    if resp is None or m is None:
        return ""

    executed = getattr(m, "executed_tools", None) or []
    for t in executed:
        try:
            args = getattr(t, "arguments", "") or ""
            args_obj = json.loads(args) if args else {}
            q = args_obj.get("query", "")
            ttype = getattr(t, "type", "tool")
            print(f"[voice]   compound used {ttype}: {q!r}" if q else
                  f"[voice]   compound used {ttype}", flush=True)
        except Exception:
            pass

    reply = (m.content or "").strip()
    dt = time.time() - t0
    print(f"[voice] {model} replied in {dt:.2f}s ({len(reply)} chars)", flush=True)
    return reply


def _ollama_available() -> bool:
    """Lazy probe: is the local Ollama daemon reachable AND does it have a
    model loaded? Result cached for _OLLAMA_PROBE_TTL seconds so we don't
    HTTP-poll on every turn, but also so a mid-session `ollama serve` gets
    picked up without a Chloe restart. Returns False (without erroring) if
    the daemon is offline or the fallback is disabled."""
    global _ollama_available_cache
    import time as _time
    now = _time.monotonic()
    if isinstance(_ollama_available_cache, tuple):
        cached_value, cached_at = _ollama_available_cache
        if now - cached_at < _OLLAMA_PROBE_TTL:
            return cached_value
    if not OLLAMA_FALLBACK_ENABLED:
        _ollama_available_cache = (False, now)
        return False
    try:
        import requests as _req
        r = _req.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            tags = r.json().get("models", []) or []
            names = [t.get("name", "") for t in tags if t.get("name")]
            has_model = any(OLLAMA_MODEL in n or n.startswith(OLLAMA_MODEL.split(":")[0]) for n in names)
            # Only log on the first probe of a session or after a state change
            prev = _ollama_available_cache[0] if isinstance(_ollama_available_cache, tuple) else None
            result = bool(names)
            if prev != result:
                print(f"[chloe] Ollama detected at {OLLAMA_URL} ({len(names)} models)", flush=True)
                if names:
                    print(f"[chloe]   available: {names}", flush=True)
                if not has_model:
                    print(f"[chloe]   WARNING: target model '{OLLAMA_MODEL}' not pulled. "
                          f"Run: ollama pull {OLLAMA_MODEL}", flush=True)
            _ollama_available_cache = (result, now)
            return result
    except Exception as e:
        if VOICE_DEBUG:
            print(f"[chloe] Ollama probe failed at {OLLAMA_URL}: "
                  f"{type(e).__name__}: {e}", flush=True)
    _ollama_available_cache = (False, now)
    return False


def _loose_parse_dict(s: str):
    """Parse `s` as either JSON or a Python literal dict.

    llama3.1:8b not only emits tool-call shapes as raw text instead of
    structured tool_calls, it sometimes uses Python repr (`None`, `True`,
    `False`) instead of JSON (`null`, `true`, `false`). `json.loads` chokes
    on those; `ast.literal_eval` handles both. Tried in that order so we
    keep the fast path for well-formed JSON. Safe — `literal_eval` only
    evaluates literals (no code, no calls). Returns the parsed object
    or None on failure."""
    import ast as _ast
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return _ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return None


def _synthesize_tool_call_from_content(content: str):
    """When llama3.1/llama3.2 (via Ollama) writes a tool-call shape as
    plain text instead of using Ollama's structured tool_calls API,
    parse it out and return an Ollama-shaped tool_call dict that the
    rest of the loop can execute. Returns None if nothing tool-shaped
    is found.

    Handles the variants we've seen in the wild:
      {"name": "grep_source", "parameters": {...}}
      {"name": "grep_source", "arguments": {...}}
      {"function": {"name": "grep_source", "arguments": {...}}}
      ```json\n{...}\n```
      "Let me look that up: {...}"           (prose preamble)
      {"name": "grep_source", "parameters": {"file": None, ...}}  ← Python repr
    """
    import re as _re
    if not content:
        return None
    s = content.strip()
    # 1) Strip any markdown code fence around a JSON block
    m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, _re.DOTALL | _re.IGNORECASE)
    if m:
        s = m.group(1)
    # 2) If it doesn't start with { yet, try to find the first {...} object.
    if not s.lstrip().startswith("{"):
        m = _re.search(r"\{.*\}", s, _re.DOTALL)
        if not m:
            return None
        s = m.group(0)
    # 3) Loose parse — accepts JSON OR Python repr (None/True/False).
    parsed = _loose_parse_dict(s)
    if not isinstance(parsed, dict):
        return None
    # 4) Extract name + args from the three shapes.
    name = args = None
    if "function" in parsed and isinstance(parsed["function"], dict):
        fn = parsed["function"]
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("parameters")
    else:
        name = parsed.get("name")
        args = parsed.get("arguments") or parsed.get("parameters")
    # 5) Sanity: name must be a known tool, args must be present.
    known_tools = {"grep_source"} | WALLET_TOOL_NAMES
    if not isinstance(name, str) or name not in known_tools:
        return None
    if args is None:
        args = {}
    return {"function": {"name": name, "arguments": args}}


def _ollama_chat(messages: list, max_tokens: int = 400) -> str:
    """Send a non-streaming chat completion to local Ollama. Returns the
    reply text (stripped) or empty string on any failure. Includes a
    tool-call loop for our `grep_source` tool — Ollama's chat API since
    0.3+ accepts an OpenAI-compatible `tools` field, and llama3.2:3b
    has function-calling capability. Loop mirrors `_groq_chat_attempt`.

    `messages` is in OpenAI format and includes the system prompt;
    we mutate a local copy as the tool loop progresses.
    """
    if not _ollama_available():
        return ""
    try:
        import requests as _req
    except ImportError:
        return ""

    # Local copy so we don't mutate the caller's history with intermediate
    # tool messages — those are scaffolding, not user-facing turns.
    msgs = list(messages)

    MAX_TOOL_ITERS = 3
    t0 = time.time()
    final_msg = None

    for tool_iter in range(MAX_TOOL_ITERS + 1):
        try:
            r = _req.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": msgs,
                    "stream":   False,
                    "tools":    [GREP_TOOL_SCHEMA,
                                 *WALLET_TOOL_SCHEMAS.values()],
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens,
                    },
                },
                timeout=180,
            )
            if r.status_code != 200:
                print(f"[chloe] Ollama HTTP {r.status_code}: {r.text[:200]}",
                      flush=True)
                return ""
            data = r.json()
        except Exception as e:
            dt = time.time() - t0
            print(f"[chloe] Ollama error after {dt:.2f}s: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return ""

        msg = data.get("message", {}) or {}
        final_msg = msg
        tool_calls = msg.get("tool_calls") or []

        # Llama 3.1/3.2 are inconsistent about using Ollama's structured
        # tool_calls API — sometimes correctly, sometimes by stuffing the
        # JSON tool-call shape into `content` as plain text. Detect that
        # case (in any of several wrapper styles) and synthesize a real
        # tool_call so we can execute it instead of speaking the JSON.
        if not tool_calls:
            synth = _synthesize_tool_call_from_content(msg.get("content") or "")
            if synth is not None:
                tool_calls = [synth]
                print(f"[chloe]   ollama: synthesized tool_call from JSON-as-text content", flush=True)
                msg["content"] = ""  # don't speak the JSON aloud later

        if not tool_calls:
            break  # final reply, exit loop

        if tool_iter == MAX_TOOL_ITERS:
            print(f"[chloe] Ollama tool-call loop hit max iterations ({MAX_TOOL_ITERS}); bailing", flush=True)
            break

        # Ollama returns tool_calls without an `id`, unlike OpenAI/Groq.
        # We feed them back as-is in the assistant message so the model
        # has the context, then append a generic `tool` role result for
        # each one (Ollama doesn't require tool_call_id matching). PINs
        # in wallet_send args are redacted in the persisted copy.
        redacted_tool_calls = []
        for _tc in tool_calls:
            _fn = dict(_tc.get("function", {}) or {})
            _name = _fn.get("name", "")
            _args = _fn.get("arguments", {})
            if _name == "wallet_send":
                if isinstance(_args, dict):
                    _fn["arguments"] = _redact_pin_in_args_dict(_args, _name)
                elif isinstance(_args, str):
                    _fn["arguments"] = _redact_pin_in_args_str(_args, _name)
            new_tc = dict(_tc); new_tc["function"] = _fn
            redacted_tool_calls.append(new_tc)
        msgs.append({
            "role":       "assistant",
            "content":    msg.get("content") or "",
            "tool_calls": redacted_tool_calls,
        })

        for tc in tool_calls:
            fn   = tc.get("function", {}) or {}
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            # Ollama gives arguments as a dict directly; Groq gives a JSON
            # string. Handle either shape defensively.
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            if name == "grep_source":
                result = _grep_source(args.get("pattern", ""), args.get("file"))
                preview = (args.get("pattern") or "")[:60]
                print(f"[chloe]   ollama-tool grep_source(/{preview}/, file={args.get('file')!r})"
                      f" → {len(result)} chars", flush=True)
            elif name in WALLET_TOOL_NAMES:
                result = _wallet_dispatch(name, args)
                safe_args = {k: ("<redacted>" if k == "pin" else v)
                             for k, v in args.items()}
                print(f"[chloe]   ollama-tool {name}({safe_args})"
                      f" → {len(result)} chars", flush=True)
            else:
                result = f"unknown tool: {name}"
                print(f"[chloe]   ollama-tool {name!r} requested but not implemented", flush=True)

            msgs.append({
                "role":    "tool",
                "name":    name,
                "content": result[:4000],
            })

    if final_msg is None:
        return ""
    reply = (final_msg.get("content") or "").strip()
    # llama3.1:8b sometimes hallucinates tool-call-shaped JSON for non-tool
    # turns (e.g. user says "hi" → model emits a fake tool call). The
    # synthesizer rejected it (unknown tool, or known tool but the model
    # also wrote it as content rather than a structured call). The raw
    # JSON would otherwise leak to the user. Detect the shape and fall
    # back: prefer empty-return so the caller retries on Groq fast Llama
    # (more reliable for tool-calling); only emit a canned reply if Groq
    # isn't configured at all.
    #
    # `_loose_parse_dict` accepts both JSON and Python repr — llama3.1:8b
    # has been observed emitting `None`/`True`/`False` instead of
    # `null`/`true`/`false`, which strict json.loads rejected.
    if reply.startswith("{") and reply.endswith("}"):
        parsed = _loose_parse_dict(reply)
        if isinstance(parsed, dict) and (
            "name" in parsed or
            ("function" in parsed and isinstance(parsed["function"], dict))
        ):
            dt = time.time() - t0
            if _sync_groq is not None:
                print(f"[chloe] Ollama emitted bogus tool-shape after "
                      f"{dt:.2f}s ({reply[:80]}…); returning empty so "
                      f"caller retries on Groq", flush=True)
                return ""
            print(f"[chloe] Ollama emitted bogus tool-shape after "
                  f"{dt:.2f}s ({reply[:80]}…); Groq unavailable — using "
                  f"canned fallback", flush=True)
            reply = "I'm here — what can I help you with?"
    dt = time.time() - t0
    print(f"[chloe] Ollama ({OLLAMA_MODEL}) replied in {dt:.2f}s "
          f"({len(reply)} chars)", flush=True)
    return reply


def _ask_groq(user_text: str) -> str:
    """Run a single conversation turn. Routes between Groq compound-mini
    (web search), local Ollama, and Groq fast Llama based on the active
    routing mode (CHLOE_OLLAMA_PRIMARY) and whether the query needs
    real-time data.

    Local-first mode (default, OLLAMA_PRIMARY=1):
      - Real-time queries → Groq compound-mini (only path with web search)
      - Everything else   → local Ollama (saves Groq quota)
      - On failure either way, the other side is the fallback.

    Cloud-first mode (legacy, OLLAMA_PRIMARY=0):
      - Groq fast/compound by keyword routing
      - Hedge-retry from fast → compound if the fast model bailed
      - Final fallback to Ollama if Groq is unavailable.
    """
    if not _sync_groq and not _ollama_available():
        return ""
    _push_history("user", user_text)

    route = _pick_route(user_text)
    reply = ""

    def _build_ollama_msgs():
        msgs = [{"role": "system",
                 "content": _augmented_voice_system(None, user_text)}] + _voice_history
        return _trim_messages_for_model(msgs, MODEL_TEXT)

    if route == 'ollama':
        # Local-first happy path: Ollama handles everyday chat
        print(f"[voice] Ollama [primary, model={OLLAMA_MODEL}]", flush=True)
        reply = _ollama_chat(_build_ollama_msgs(), max_tokens=400)
        if not reply and _sync_groq:
            print("[voice] Ollama empty — falling back to Groq fast Llama", flush=True)
            reply = _groq_chat_attempt(user_text, MODEL_TEXT)

    elif route == 'groq_search':
        # Real-time query — must go through Groq compound (only path with search)
        print(f"[voice] groq → {MODEL_SEARCH} [real-time]", flush=True)
        reply = _groq_chat_attempt(user_text, MODEL_SEARCH)
        if not reply and OLLAMA_FALLBACK_ENABLED and _ollama_available():
            # Groq compound failed (rate-limit, network). Use Ollama, but
            # Ollama can't web-search, so the answer may be stale — that's
            # better than nothing.
            print("[voice] compound failed — falling back to Ollama (no web search)", flush=True)
            reply = _ollama_chat(_build_ollama_msgs(), max_tokens=400)

    else:  # 'groq_fast' — legacy cloud-first routing
        print(f"[voice] groq → {MODEL_TEXT} [fast]", flush=True)
        reply = _groq_chat_attempt(user_text, MODEL_TEXT)
        # Hedge-retry: if fast model bailed on a real-time question, retry compound
        if reply and USE_COMPOUND and _looks_like_hedge(reply):
            print("[voice] reply looks hedged — auto-retrying with compound-mini", flush=True)
            new_reply = _groq_chat_attempt(user_text, MODEL_SEARCH)
            if new_reply:
                if _looks_like_hedge(new_reply):
                    print("[voice] retry also hedged — accepting compound reply anyway", flush=True)
                else:
                    print("[voice] retry succeeded with web search", flush=True)
                reply = new_reply
        # Final fallback to Ollama if Groq is fully unavailable
        if not reply and OLLAMA_FALLBACK_ENABLED and _ollama_available():
            print("[voice] Groq returned empty — falling back to local Ollama", flush=True)
            reply = _ollama_chat(_build_ollama_msgs(), max_tokens=400)
            if reply:
                print("[voice] Ollama fallback succeeded", flush=True)

    if reply:
        _push_history("assistant", reply)
    else:
        # Pop the user message we pushed since we got nothing back.
        if _voice_history and _voice_history[-1]["role"] == "user":
            _voice_history.pop()
    return reply

_TTS_LINK_RE        = _re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
_TTS_WIKILINK_RE    = _re.compile(r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]')
_TTS_CODEBLOCK_RE   = _re.compile(r'```[\s\S]*?```')
_TTS_INLINE_CODE_RE = _re.compile(r'`([^`]+)`')
_TTS_HEADING_RE     = _re.compile(r'^#{1,6}\s+', _re.MULTILINE)
_TTS_BLOCKQUOTE_RE  = _re.compile(r'^>\s*', _re.MULTILINE)
_TTS_BULLET_RE      = _re.compile(r'^\s*[-\u2022]\s+', _re.MULTILINE)
_TTS_HRULE_RE       = _re.compile(r'^---+\s*$', _re.MULTILINE)
_TTS_ASTERISK_RE    = _re.compile(r'\*+')
_TTS_UNDERSCORE_PAIR_RE = _re.compile(r'(?<!\w)_+([^_\n]+?)_+(?!\w)')

def _clean_for_tts(text: str) -> str:
    """Strip markdown symbols that the TTS engine would otherwise pronounce
    out loud (asterisks, hashes, backticks, etc). Conservative on underscores
    so snake_case identifiers like 'daily_context.py' stay intact.
    """
    if not text:
        return text
    # Code blocks: keep content readable but drop the fences.
    text = _TTS_CODEBLOCK_RE.sub(lambda m: m.group(0).strip("`").strip(), text)
    # Inline links/wikilinks: keep visible text only.
    text = _TTS_LINK_RE.sub(r"\1", text)
    text = _TTS_WIKILINK_RE.sub(
        lambda m: (m.group(2) or m.group(1).split("/")[-1]).replace("_", " "),
        text,
    )
    # Inline code: drop the backticks.
    text = _TTS_INLINE_CODE_RE.sub(r"\1", text)
    # Block markers.
    text = _TTS_HEADING_RE.sub("", text)
    text = _TTS_BLOCKQUOTE_RE.sub("", text)
    text = _TTS_BULLET_RE.sub("", text)
    text = _TTS_HRULE_RE.sub("", text)
    # Emphasis. Asterisks always strippable; underscore italics only when
    # paired around a non-identifier word so 'snake_case' survives.
    text = _TTS_ASTERISK_RE.sub("", text)
    text = _TTS_UNDERSCORE_PAIR_RE.sub(r"\1", text)
    # Whitespace cleanup.
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _speak(text: str) -> None:
    """Synthesize TTS and play. Engine priority:
        1. ElevenLabs   if USE_ELEVENLABS=1 + ELEVENLABS_API_KEY set
        2. Kokoro local if USE_KOKORO=1 + model files present
        3. edge-tts     (always-available free default)
    All three share the same sentence-streaming + barge-in pipeline."""
    text = _clean_for_tts(text)
    if not text:
        return
    if USE_ELEVENLABS and ELEVENLABS_API_KEY:
        _speak_elevenlabs(text)
    elif USE_KOKORO:
        _speak_kokoro(text)
    else:
        _speak_edge_tts(text)


def _synthesize_tts_bytes(text: str):
    """Like _speak() but returns (audio_bytes, format) instead of playing on
    PC speakers. Used by the mobile path so the iPhone hears Chloe's reply
    in earbuds, not on home speakers.

    Format is "mp3" (ElevenLabs / edge-tts) or "wav" (Kokoro). Browser plays
    both natively via an <audio> element with a data URL.

    Falls through the same priority chain as _speak. Returns None only if
    every engine in the chain failed."""
    text = _clean_for_tts(text)
    if not text:
        return None
    if USE_ELEVENLABS and ELEVENLABS_API_KEY:
        b = _elevenlabs_to_bytes(text)
        if b: return (b, "mp3")
    if USE_KOKORO:
        b = _kokoro_to_wav_bytes(text)
        if b: return (b, "wav")
    b = _edge_tts_to_bytes(text)
    if b: return (b, "mp3")
    return None


def _elevenlabs_to_bytes(text: str):
    """ElevenLabs synthesis → MP3 bytes (no local playback). Mirrors the
    SDK-first-then-HTTP fallback that _speak_elevenlabs uses; returns None
    on failure so the mobile caller can fall through to Kokoro/edge-tts."""
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            text=text,
            output_format="mp3_44100_128",
        )
        return b"".join(chunk for chunk in stream if chunk) or None
    except ImportError:
        try:
            import requests
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            }
            payload = {
                "text": text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"[voice] ElevenLabs HTTP {r.status_code}: {r.text[:200]}")
                return None
            return r.content
        except Exception as e:
            print(f"[voice] ElevenLabs HTTP error (to_bytes): {e}")
            return None
    except Exception as e:
        print(f"[voice] ElevenLabs SDK error (to_bytes): {e}")
        return None


def _kokoro_to_wav_bytes(text: str):
    """Kokoro synthesis → WAV bytes (no local playback). Mobile path."""
    engine = _get_kokoro()
    if engine is None:
        return None
    try:
        t0 = time.time()
        samples, sr = engine.create(
            text,
            voice=KOKORO_VOICE,
            speed=KOKORO_SPEED,
            lang="en-us",
        )
        dt = time.time() - t0
        secs = len(samples) / sr if sr else 0
        print(f"[voice] Kokoro→bytes synthesized {secs:.2f}s in {dt:.2f}s", flush=True)
    except Exception as e:
        print(f"[voice] Kokoro to-bytes synthesis error: {type(e).__name__}: {e}")
        return None
    try:
        clipped = np.clip(samples, -1.0, 1.0)
        int16   = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(int(sr))
            wf.writeframes(int16.tobytes())
        return buf.getvalue()
    except Exception as e:
        print(f"[voice] Kokoro WAV encode error: {e}")
        return None


def _edge_tts_to_bytes(text: str):
    """edge-tts synthesis → MP3 bytes (no local playback)."""
    try:
        import edge_tts
    except ImportError as e:
        print(f"[voice] edge-tts dep missing (to_bytes): {e}")
        return None

    async def _synth():
        comm = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        chunks = []
        async for ev in comm.stream():
            if ev.get("type") == "audio":
                chunks.append(ev.get("data") or b"")
        return b"".join(chunks)

    try:
        return asyncio.run(_synth()) or None
    except Exception as e:
        print(f"[voice] edge-tts to-bytes error: {e}")
        return None


def _speak_elevenlabs(text: str) -> None:
    """ElevenLabs TTS — official SDK if installed, else direct HTTP via requests."""
    try:
        import soundfile as sf
        import sounddevice as sd
    except ImportError as e:
        print(f"[voice] audio playback deps missing: {e}")
        return

    audio_bytes = None

    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            text=text,
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(chunk for chunk in stream if chunk)
    except ImportError:
        try:
            import requests
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            }
            payload = {
                "text": text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"[voice] ElevenLabs HTTP {r.status_code}: {r.text[:200]}")
                _speak_edge_tts(text)
                return
            audio_bytes = r.content
        except Exception as e:
            print(f"[voice] ElevenLabs HTTP error: {e}")
            _speak_edge_tts(text)
            return
    except Exception as e:
        print(f"[voice] ElevenLabs SDK error: {e}")
        _speak_edge_tts(text)
        return

    if not audio_bytes:
        print("[voice] ElevenLabs returned no audio")
        return

    tmp = Path(tempfile.gettempdir()) / f"chloe_tts_{int(time.time()*1000)}.mp3"
    _speaking.set()
    _barge_in_request.clear()
    _barge_in_via_wake.clear()
    threading.Thread(target=_barge_in_monitor, daemon=True,
                     name="chloe-barge-in").start()
    try:
        tmp.write_bytes(audio_bytes)
        data, sr = sf.read(str(tmp))
        _play_audio_with_barge_in(data, sr)
    except Exception as e:
        print(f"[voice] ElevenLabs playback error: {e}")
    finally:
        _speaking.clear()
        try: tmp.unlink(missing_ok=True)
        except Exception: pass


# ─── BARGE-IN + STREAMING TTS ────────────────────────────────────────────────
# `_speaking` is set while Chloe is actively producing audio, cleared when
# the entire utterance finishes (or barge-in fires). `_barge_in_request` is
# set by the wake-during-speech monitor to signal the playback loop to abort
# the current sentence + skip remaining sentences. Both used by the streaming
# TTS consumer below AND the barge-in monitor in the voice loop.
_speaking          = threading.Event()
_barge_in_request  = threading.Event()
# Set by the barge-in monitor when the wake word interrupted speech, so the
# voice loop knows to go straight into recording instead of waiting for a
# second wake.
_barge_in_via_wake = threading.Event()
# Cached after _voice_loop creates the wake detector. _speak_* reads these
# to decide whether to spawn a barge-in monitor during TTS playback.
_wake_detector_global = None  # type: ignore[var-annotated]
_voice_device_global  = None  # type: ignore[var-annotated]
# Master toggle. Default on; CHLOE_BARGE_IN=0 in _env disables. Set to 0 if
# you find your audio driver doesn't tolerate concurrent input + output.
BARGE_IN_ENABLED = os.environ.get("CHLOE_BARGE_IN", "1").strip() != "0"


def _barge_in_monitor():
    """Run wake-word detection during TTS playback. Sets _barge_in_request
    + _barge_in_via_wake if wake fires. Exits when _speaking clears.

    Concurrent input + output streams have historically been flaky on
    Windows audio drivers (see comment block above _voice_loop). If we
    fail to open the input stream, we silently bail — barge-in falls back
    to the always-available PTT trigger."""
    if not BARGE_IN_ENABLED:
        return
    wake = _wake_detector_global
    if wake is None:
        return
    try:
        import sounddevice as sd
    except ImportError:
        return

    device = _voice_device_global
    frame_length = wake['frame_length']
    try:
        stream, native_rate = _open_input_stream_with_retry(
            sd, device, frame_length=frame_length, max_attempts=2
        )
    except Exception as e:
        # Concurrent I/O not supported, mic busy, etc. Skip — PTT barge-in
        # still works as a fallback.
        if VOICE_DEBUG:
            print(f"[barge-in] couldn't open monitor stream: {e}", flush=True)
        return

    needs_resample = (native_rate != SAMPLE_RATE)
    src_block = stream.blocksize or frame_length
    if wake['engine'] == 'openwakeword':
        try: wake['handle'].reset()
        except Exception: pass

    try:
        with stream:
            while _speaking.is_set():
                if _barge_in_request.is_set():
                    return
                try:
                    audio_data, _ = stream.read(src_block)
                except Exception:
                    return  # device hiccup → fall back to PTT-only barge-in
                np_chunk = np.frombuffer(audio_data, dtype=np.int16)
                if needs_resample:
                    np_chunk = _resample_to_16k(np_chunk, native_rate)
                    if len(np_chunk) < frame_length:
                        np_chunk = np.pad(np_chunk, (0, frame_length - len(np_chunk)))
                    elif len(np_chunk) > frame_length:
                        np_chunk = np_chunk[:frame_length]
                np_chunk = _apply_gain(np_chunk)
                if wake['predict'](np_chunk):
                    print("[barge-in] wake detected during speech — interrupting", flush=True)
                    _barge_in_via_wake.set()
                    _barge_in_request.set()
                    return
    except Exception as e:
        if VOICE_DEBUG:
            print(f"[barge-in] monitor error: {e}", flush=True)


# Sentence boundary heuristic: punctuation followed by whitespace and a
# capital letter or digit. Keeps abbreviations like "Mr. Smith" or "U.S."
# from triggering false splits, since they'd be followed by lowercase.
_SENT_BOUNDARY_RE = _re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


def _split_sentences_for_tts(text: str) -> list[str]:
    """Split a reply into sentences for streaming TTS. Returns one-element
    list for short text. Always returns at least one element if text is
    non-empty."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_BOUNDARY_RE.split(text) if p.strip()]
    if not parts:
        return [text]
    # Merge any very short trailing fragment (single word, abbreviations,
    # etc.) into the previous sentence so we don't synthesize "Yes." as
    # its own audio file.
    merged = []
    for p in parts:
        if merged and len(p) < 12:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def _play_audio_with_barge_in(data, sr) -> bool:
    """Play `data` via sounddevice, polling _barge_in_request during the
    wait. Returns False if barge-in fired, True if playback completed.
    Caller must have already acquired the audio device."""
    import sounddevice as sd
    sd.play(data, sr)
    # Run sd.wait() in a worker so the main thread can poll for barge-in.
    finished = threading.Event()
    def _waiter():
        try:
            sd.wait()
        finally:
            finished.set()
    threading.Thread(target=_waiter, daemon=True,
                     name="chloe-tts-waiter").start()
    while not finished.is_set():
        if _barge_in_request.is_set():
            try: sd.stop()
            except Exception: pass
            finished.wait(timeout=0.5)
            return False
        time.sleep(0.05)
    return True


def _speak_edge_tts(text: str) -> None:
    """Free TTS via edge-tts with sentence-level streaming.

    Architecture: a producer thread synthesizes sentences in order and pushes
    decoded (data, sr, tmpfile) tuples to a small bounded queue. The caller
    thread (this one) consumes the queue, playing each sentence and polling
    for barge-in between/during sentences. The first sentence usually starts
    playing while later sentences are still being synthesized — that's the
    perceived-latency win over the old "synthesize-then-play" approach.

    Barge-in: if `_barge_in_request` fires, we stop playback and drop the
    rest of the queue."""
    try:
        import edge_tts
        import soundfile as sf
        import sounddevice as sd  # noqa: F401  imported lazily inside helpers
    except ImportError as e:
        print(f"[voice] edge-tts deps missing: {e}")
        return

    sentences = _split_sentences_for_tts(text)
    if not sentences:
        return

    # Bounded queue: 3 is enough that the producer stays a sentence ahead of
    # the consumer without buffering the whole reply if a long answer comes in.
    audio_queue: queue.Queue = queue.Queue(maxsize=3)
    SENTINEL = object()
    producer_done = threading.Event()

    async def _synth_one(idx: int, sent: str):
        tmp = Path(tempfile.gettempdir()) / f"chloe_tts_{int(time.time()*1000)}_{idx}.mp3"
        try:
            comm = edge_tts.Communicate(sent, EDGE_TTS_VOICE)
            await comm.save(str(tmp))
            data, sr = sf.read(str(tmp))
            return data, sr, tmp
        except Exception as e:
            print(f"[voice] edge-tts synth error on sentence {idx}: {e}")
            return None, None, tmp

    async def _producer_async():
        for idx, sent in enumerate(sentences):
            if _barge_in_request.is_set():
                break
            data, sr, tmp = await _synth_one(idx, sent)
            audio_queue.put((data, sr, tmp))
        producer_done.set()

    def _producer_thread():
        try:
            asyncio.run(_producer_async())
        except Exception as e:
            print(f"[voice] edge-tts producer crashed: {e}")
            producer_done.set()
        finally:
            audio_queue.put(SENTINEL)

    threading.Thread(target=_producer_thread, daemon=True,
                     name="chloe-tts-producer").start()

    _speaking.set()
    _barge_in_request.clear()
    _barge_in_via_wake.clear()
    # Start the wake-during-speech monitor in parallel with playback. Safe to
    # spawn even if it ends up failing to open a stream — it just exits.
    threading.Thread(target=_barge_in_monitor, daemon=True,
                     name="chloe-barge-in").start()
    try:
        while True:
            item = audio_queue.get()
            if item is SENTINEL:
                break
            data, sr, tmp = item
            try:
                if data is not None and not _barge_in_request.is_set():
                    completed = _play_audio_with_barge_in(data, sr)
                    if not completed:
                        # Drain remaining queue entries (without playing) so
                        # the producer can finish + we can clean up tmpfiles.
                        while True:
                            try:
                                rem = audio_queue.get_nowait()
                            except queue.Empty:
                                break
                            if rem is SENTINEL:
                                break
                            _, _, rem_tmp = rem
                            try: rem_tmp.unlink(missing_ok=True)
                            except Exception: pass
                        break
            except Exception as e:
                print(f"[voice] edge-tts playback error: {e}")
            finally:
                try: tmp.unlink(missing_ok=True)
                except Exception: pass
    finally:
        _speaking.clear()
        # If barge-in fired, the wake monitor will pick it up by checking
        # _barge_in_request; we leave the flag set for the next layer up.


# ─── KOKORO LOCAL TTS ────────────────────────────────────────────────────────
# Higher-quality offline TTS. Same sentence-level streaming + barge-in
# pipeline as edge-tts, but synthesis happens locally via ONNX runtime
# instead of a Microsoft cloud call. Lazy-loaded so the ~330MB model only
# touches RAM when USE_KOKORO=1.
_kokoro_instance       = None  # type: ignore[var-annotated]
_kokoro_load_attempted = False


def _get_kokoro():
    """Lazy-load the Kokoro instance. Cached after first attempt (success
    or failure). Returns None if unavailable — caller falls back to
    edge-tts so the voice loop never goes silent."""
    global _kokoro_instance, _kokoro_load_attempted
    if _kokoro_load_attempted:
        return _kokoro_instance
    _kokoro_load_attempted = True
    if not USE_KOKORO:
        return None
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("[voice] Kokoro: 'kokoro_onnx' package not installed — "
              "run `pip install kokoro-onnx soundfile`", flush=True)
        return None
    if not KOKORO_MODEL_PATH.exists():
        print(f"[voice] Kokoro: model file missing at {KOKORO_MODEL_PATH}",
              flush=True)
        print(f"[voice]   run: python download_kokoro.py", flush=True)
        return None
    if not KOKORO_VOICES_PATH.exists():
        print(f"[voice] Kokoro: voices file missing at {KOKORO_VOICES_PATH}",
              flush=True)
        print(f"[voice]   run: python download_kokoro.py", flush=True)
        return None
    try:
        t0 = time.time()
        print(f"[voice] loading Kokoro TTS (model={KOKORO_MODEL_PATH.name}, "
              f"voice={KOKORO_VOICE})…", flush=True)
        _kokoro_instance = Kokoro(str(KOKORO_MODEL_PATH),
                                  str(KOKORO_VOICES_PATH))
        # Try to enumerate available voices for the banner — useful if
        # the user typo'd KOKORO_VOICE.
        try:
            voices = list(getattr(_kokoro_instance, "voices", {}).keys())
            if voices and KOKORO_VOICE not in voices:
                print(f"[voice] WARNING: voice '{KOKORO_VOICE}' not in "
                      f"available voices: {voices}", flush=True)
        except Exception:
            pass
        dt = time.time() - t0
        print(f"[voice] Kokoro ready in {dt:.2f}s", flush=True)
    except Exception as e:
        print(f"[voice] Kokoro load failed: {type(e).__name__}: {e}",
              flush=True)
        _kokoro_instance = None
    return _kokoro_instance


def _speak_kokoro(text: str) -> None:
    """Local TTS via Kokoro. Uses the same producer-consumer + barge-in
    architecture as _speak_edge_tts: a worker synthesizes sentences in
    order and pushes (samples, sample_rate) tuples; the consumer plays
    them back one at a time, polling for barge-in throughout.

    Falls through to edge-tts if Kokoro isn't loadable, so a missing
    model file or import error never silences Chloe."""
    kokoro = _get_kokoro()
    if kokoro is None:
        # Soft fallback to edge-tts so the assistant keeps working.
        return _speak_edge_tts(text)

    try:
        import sounddevice as sd  # noqa: F401  imported lazily inside helpers
    except ImportError as e:
        print(f"[voice] Kokoro deps missing: {e}", flush=True)
        return

    sentences = _split_sentences_for_tts(text)
    if not sentences:
        return

    audio_queue: queue.Queue = queue.Queue(maxsize=3)
    SENTINEL = object()

    def _producer():
        try:
            for idx, sent in enumerate(sentences):
                if _barge_in_request.is_set():
                    break
                try:
                    samples, sample_rate = kokoro.create(
                        text=sent,
                        voice=KOKORO_VOICE,
                        speed=KOKORO_SPEED,
                        lang="en-us",
                    )
                    audio_queue.put((samples, sample_rate))
                except Exception as e:
                    print(f"[voice] Kokoro synth error on sentence {idx}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    # Skip this sentence; don't stall the whole reply.
                    continue
        finally:
            audio_queue.put(SENTINEL)

    threading.Thread(target=_producer, daemon=True,
                     name="chloe-kokoro-producer").start()

    _speaking.set()
    _barge_in_request.clear()
    _barge_in_via_wake.clear()
    threading.Thread(target=_barge_in_monitor, daemon=True,
                     name="chloe-barge-in").start()
    try:
        while True:
            item = audio_queue.get()
            if item is SENTINEL:
                break
            samples, sr = item
            try:
                if not _barge_in_request.is_set():
                    completed = _play_audio_with_barge_in(samples, sr)
                    if not completed:
                        # Drain remaining queue entries so the producer
                        # can finish cleanly.
                        while True:
                            try:
                                rem = audio_queue.get_nowait()
                            except queue.Empty:
                                break
                            if rem is SENTINEL:
                                break
                        break
            except Exception as e:
                print(f"[voice] Kokoro playback error: {e}", flush=True)
    finally:
        _speaking.clear()


# ─── REGISTER + START VOICE LOOP ON IMPORT ───────────────────────────────────
# ─── DIRECT LIGHTS WS ENDPOINTS ─────────────────────────────────────────────
# These bypass the LLM/voice path so the HUD CH02 panel can drive bulbs
# directly. State changes broadcast lights_state_result so every client
# (HUD + PWA) updates in sync.

async def _broadcast_lights_state():
    """Snapshot current bulb + preset state and push to all WS clients."""
    snap = await asyncio.to_thread(_lights.get_state_snapshot)
    await _ws_broadcast({"type": "lights_state_result", "ok": True, **snap})


async def handle_lights_state(data, websocket):
    snap = await asyncio.to_thread(_lights.get_state_snapshot)
    await _ws_send(websocket, {"type": "lights_state_result", "ok": True, **snap})


async def handle_lights_action(data, websocket):
    target = data.get("target") or "all"
    kwargs = {}
    for k in ("on", "brightness", "color", "ct", "rgb"):
        if k in data and data[k] is not None:
            kwargs[k] = data[k]
    result = await asyncio.to_thread(_lights.apply_action, target, **kwargs)
    await _ws_send(websocket, {"type": "lights_action_result", **result})
    await _broadcast_lights_state()


async def handle_lights_discover(data, websocket):
    found = await asyncio.to_thread(_lights.discover)
    await _ws_send(websocket, {"type": "lights_discover_result", "ok": True, "found": found})
    await _broadcast_lights_state()


async def handle_lights_rename(data, websocket):
    mac = (data.get("mac") or "").strip()
    new_name = (data.get("name") or "").strip()
    if not mac or not new_name:
        await _ws_send(websocket, {"type": "lights_rename_result", "ok": False,
                                   "error": "mac and name required"})
        return
    ok = await asyncio.to_thread(_lights.rename_bulb, mac, new_name)
    await _ws_send(websocket, {"type": "lights_rename_result", "ok": ok,
                               "mac": mac, "name": new_name.lower()})
    await _broadcast_lights_state()


async def handle_lights_preset_apply(data, websocket):
    name = (data.get("name") or "").strip()
    if not name:
        await _ws_send(websocket, {"type": "lights_preset_apply_result", "ok": False,
                                   "error": "name required"})
        return
    result = await asyncio.to_thread(_lights.apply_preset, name)
    await _ws_send(websocket, {"type": "lights_preset_apply_result", **result})
    await _broadcast_lights_state()


async def handle_lights_preset_save(data, websocket):
    name = (data.get("name") or "").strip()
    if not name:
        await _ws_send(websocket, {"type": "lights_preset_save_result", "ok": False,
                                   "error": "name required"})
        return
    result = await asyncio.to_thread(_lights.save_preset, name)
    await _ws_send(websocket, {"type": "lights_preset_save_result", **result})
    await _broadcast_lights_state()


async def handle_lights_preset_delete(data, websocket):
    name = (data.get("name") or "").strip()
    if not name:
        await _ws_send(websocket, {"type": "lights_preset_delete_result", "ok": False,
                                   "error": "name required"})
        return
    result = await asyncio.to_thread(_lights.delete_preset, name)
    await _ws_send(websocket, {"type": "lights_preset_delete_result", **result})
    await _broadcast_lights_state()


hud_server.set_jarvis_handler(_dispatch)
print(f"[chloe] handler registered  model={MODEL_TEXT}  vision={MODEL_VISION}")
print(f"[chloe] groq key: {'set' if GROQ_API_KEY else 'MISSING (chat will fail)'}")

# Kick off the voice loop in a daemon thread — chat path keeps working even
# if the voice loop fails to initialize (e.g. no mic, missing libs, etc.)
threading.Thread(target=_voice_thread_entry, daemon=True, name="chloe-voice").start()
