"""chloe_env_registry.py - Single source of truth for every environment
variable Chloe's codebase reads (action item #10, audit Part 13:
"Central env-var flag registry").

2026-09-06: 146 os.environ.get()/os.getenv() call sites were found
scattered across ~35 files, each with its own inline default and no
single place documenting what exists. Rewriting all 146 call sites to
route through one accessor would be the deeper fix the audit's own
wording implies, but that is a large, multi-file mechanical change
this bridge session cannot live-test (many of these files -- wiki_dedup,
chloe_ed_profile, social_composer, weather, notify, and more -- were
never touched or verified this session). Retrofitting all of them in
one pass risks a silent regression nobody would catch until Ed hit it
live weeks later.

What this module does instead, safely and additively: documents every
flag this codebase currently reads (name, its default as captured
directly from source, and which file(s) read it) as ENV_REGISTRY below
-- generated once from the actual source via a small extraction script,
not hand-typed, so it's traceable rather than guessed -- and provides
unknown_chloe_env_vars(), a boot-time check that catches the exact bug
class this whole session kept finding: a flag that's SET but silently
does nothing because of a typo (e.g. "CHLOE_BARGE_TRESHOLD" instead of
"CHLOE_BARGE_THRESHOLD") or because it was renamed/removed and an old
.env line never got cleaned up. jarvis.py's boot sequence calls this
once at startup and prints a warning for anything it finds -- see the
"unknown env var(s)" print near the other boot-time diagnostics.

Existing call sites are UNCHANGED -- this module does not replace any
of the 146 os.environ.get() calls, only documents and cross-checks
them. Regenerate the data below by running
`python regen_env_registry.py > chloe_env_registry.py` (checked into
this repo alongside this file) whenever a flag is added, removed, or
its default changes -- "no single place to see what exists" was the
audit's actual complaint, and a registry that quietly drifts out of
date recreates the same problem with extra steps.
"""
import os

# name -> {"default": <python literal>, "default_expr": <source text,
#          when the default isn't a plain literal>, "files": (tuple)}
# Generated 2026-09-06 from every os.environ.get()/os.getenv() call
# site in this repo's own *.py files.
ENV_REGISTRY = {
    # --- bench_ollama.py benchmarking flags ---
    "BENCH_MAX_TOKENS": {"default": '120', "files": ('bench_ollama.py',)},
    # --- Other / third-party env vars ---
    "BRAIN_CHAT_MODEL": {"default_expr": 'os.environ.get("OLLAMA_MODEL", "llama3.2:3b"', "files": ('brain_http.py',)},
    # --- Brave Search API flags ---
    "BRAVE_API_KEY": {"default": None, "files": ('search.py',)},
    "BRAVE_SEARCH_API_KEY": {"default": None, "files": ('search.py',)},
    # --- Breez Liquid wallet SDK flags ---
    "BREEZ_API_KEY": {"default": None, "files": ('wallet.py',)},
    # --- Chloe-specific flags ---
    "CHLOE_ARCADE_WATCH_MAX": {"default": '75', "files": ('jarvis.py',)},
    "CHLOE_ARCADE_WATCH_MIN": {"default": '22', "files": ('jarvis.py',)},
    "CHLOE_AUTO_FACT": {"default": '1', "files": ('brain_wiring.py',)},
    "CHLOE_BACKUP_KEEP_WEEKS": {"default": '4', "files": ('backup_chloe.py',)},
    "CHLOE_BACKUP_ROOT": {"default": 'C:\\Users\\eleew\\OneDrive\\ChloeBackups', "files": ('backup_chloe.py',)},
    "CHLOE_BARGE_CONSEC": {"default": '3', "files": ('jarvis.py',)},
    "CHLOE_BARGE_GAIN": {"default": '1.0', "files": ('jarvis.py',)},
    "CHLOE_BARGE_IN": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_BARGE_THRESHOLD": {"default": '0.7', "files": ('jarvis.py',)},
    "CHLOE_BOOT_SOUND": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_BRAIN_ROOT": {"default": 'C:\\Chloe\\brain', "files": ('backup_chloe.py', 'brain_http.py', 'brain_wiring.py', 'chloe_callbacks.py', 'chloe_chess.py', 'chloe_dialogue_state.py', 'chloe_ed_profile.py', 'chloe_jobs.py', 'chloe_lock.py', 'chloe_mcp_server.py', 'chloe_pending_confirms.py', 'chloe_proposals.py', 'chloe_read.py', 'chloe_synopsis.py', 'chloe_tone_guard.py', 'chloe_trace.py', 'chloe_watchdog.py', 'email_client.py', 'google_contacts.py', 'install_chloe_mcp_config.py', 'test_ambient.py', 'test_ask.py', 'test_ingest_screen.py')},
    "CHLOE_BRAVE_PATH": {"default": '', "files": ('youtube_player.py',)},
    "CHLOE_CALLBACK_TTL_MIN": {"default": '120', "files": ('chloe_callbacks.py',)},
    "CHLOE_CONTACTS_CACHE_TTL_HOURS": {"default": '24', "files": ('google_contacts.py',)},
    "CHLOE_CONTACT_ALIAS_TTL_HOURS": {"default": '24', "files": ('email_client.py',)},
    "CHLOE_CTX_LOG": {"default": '1', "files": ('chloe_context.py',)},
    "CHLOE_CTX_RESERVE": {"default": '1200', "files": ('chloe_context.py',)},
    "CHLOE_DATA_STALENESS_DAYS": {"default": '30', "files": ('wiki_dedup.py', 'wiki_embedding.py')},
    "CHLOE_DESKTOP_ROOT": {"default": '', "files": ('desktop_files.py',)},
    "CHLOE_EMBED_MODEL": {"default": 'nomic-embed-text', "files": ('chloe_memory.py', 'jarvis.py', 'wiki_embedding.py')},
    "CHLOE_EMBED_TIMEOUT": {"default": ['10', '5'], "files": ('chloe_memory.py', 'wiki_embedding.py')},
    "CHLOE_ERROR_SPEECH": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_FACT_HEAVY_THRESHOLD": {"default": '200', "files": ('brain.py',)},
    "CHLOE_FOLLOWUP": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_FOLLOWUP_S": {"default": '5', "files": ('jarvis.py',)},
    "CHLOE_GEN1RECOMP_GAME": {"default": None, "files": ('jarvis.py', 'splice_gen1recomp.py', 'splice_gen1recomp_lock.py')},
    "CHLOE_GEN1RECOMP_PATH": {"default_expr": 'str(default', "files": ('jarvis.py', 'splice_gen1recomp.py')},
    "CHLOE_GRAPH_HOST": {"default": '0.0.0.0', "files": ('brain_http.py',)},
    "CHLOE_GRAPH_PORT": {"default": '6790', "files": ('brain_http.py',)},
    "CHLOE_GREETING": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_HEALTH_URL": {"default_expr": 'HEALTH_ENDPOINT_DEFAULT', "files": ('chloe_watchdog.py',)},
    "CHLOE_JOBS_LOCAL": {"default": '1', "files": ('chloe_jobs.py',)},
    "CHLOE_LIGHTS_BACKOFF_CAP_S": {"default": '900', "files": ('lights.py',)},
    "CHLOE_LIGHTS_BACKOFF_START_S": {"default": '60', "files": ('lights.py',)},
    "CHLOE_LINKEDIN_DRAFT_DIR": {"default": 'C:\\Chloe\\secrets\\linkedin_drafts', "files": ('social.py',)},
    "CHLOE_LLM_READ": {"default": '1', "files": ('chloe_read.py',)},
    "CHLOE_LOCAL_STT": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_MAX_RECORD_S": {"default": '60', "files": ('jarvis.py',)},
    "CHLOE_MEMORY_STALE_HOURS": {"default": '6', "files": ('brain_http.py',)},
    "CHLOE_MIC": {"default": None, "files": ('jarvis.py', 'mic_check.py')},
    "CHLOE_MIC_AUTO_FALLBACK": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_MIC_FALLBACK_DEVICE": {"default": 'Microphone', "files": ('jarvis.py',)},
    "CHLOE_MIC_GAIN": {"default": '1.0', "files": ('jarvis.py',)},
    "CHLOE_MIN_PEAK_RMS": {"default": '0.03', "files": ('jarvis.py',)},
    "CHLOE_MIN_UTTERANCE_S": {"default": '1.5', "files": ('jarvis.py',)},
    "CHLOE_MODE": {"default": 'home', "files": ('brain_wiring.py', 'chloe_jobs.py', 'chloe_mcp_server.py', 'install_chloe_mcp_config.py', 'jarvis.py')},
    "CHLOE_NTFY_SERVER": {"default": 'https://ntfy.sh', "files": ('notify.py',)},
    "CHLOE_NTFY_TOPIC": {"default": '', "files": ('notify.py',)},
    "CHLOE_OLLAMA_CTX": {"default": ['16384', '8192'], "files": ('brain_http.py', 'brain_wiring.py', 'chloe_context.py', 'jarvis.py')},
    "CHLOE_OLLAMA_VISION_MODEL": {"default": 'llama3.2-vision', "files": ('jarvis.py', 'screen_vision.py')},
    "CHLOE_PIT_SUPERSEDE_THRESHOLD": {"default": '0.93', "files": ('wiki_dedup.py',)},
    "CHLOE_PROFILE": {"default": '1', "files": ('chloe_ed_profile.py',)},
    "CHLOE_PROFILE_CAP": {"default": '1400', "files": ('chloe_ed_profile.py',)},
    "CHLOE_PROFILE_MODEL": {"default": 'qwen2.5:14b', "files": ('chloe_ed_profile.py',)},
    "CHLOE_PROFILE_TIMEOUT": {"default": '60', "files": ('chloe_ed_profile.py',)},
    "CHLOE_PTT_MAX_S": {"default": '300', "files": ('jarvis.py',)},
    "CHLOE_QUOTE_STALENESS_DAYS": {"default": '3', "files": ('wiki_dedup.py', 'wiki_embedding.py')},
    "CHLOE_READ_EVERY": {"default": '3', "files": ('chloe_read.py',)},
    "CHLOE_READ_MIN_CHARS": {"default": '45', "files": ('chloe_read.py',)},
    "CHLOE_READ_MODEL": {"default": 'llama3.2:3b', "files": ('chloe_read.py',)},
    "CHLOE_READ_TIMEOUT": {"default": '4', "files": ('chloe_read.py',)},
    "CHLOE_REAL_TOKENS": {"default": '1', "files": ('chloe_context.py',)},
    "CHLOE_RECALL_THRESHOLD": {"default": '0.35', "files": ('chloe_memory.py',)},
    "CHLOE_REVIEWS_ROOT": {"default": 'C:\\Chloe\\reviews', "files": ('chloe_mcp_server.py',)},
    "CHLOE_ROMS_DIR": {"default": 'C:\\Chloe\\roms', "files": ('brain_http.py',)},
    "CHLOE_SAVESTATES_DIR": {"default": 'C:\\Chloe\\savestates', "files": ('brain_http.py',)},
    "CHLOE_SEARCH_MODEL": {"default": 'qwen2.5:14b', "files": ('jarvis.py',)},
    "CHLOE_SEARCH_SYNTH_MODEL": {"default": 'llama3.2:3b', "files": ('jarvis.py',)},
    "CHLOE_SEARCH_SYNTH_TIMEOUT_S": {"default": '35', "files": ('jarvis.py',)},
    "CHLOE_SEARCH_WARM_WAIT_S": {"default": '60', "files": ('jarvis.py',)},
    "CHLOE_SECRETS_DIR": {"default": 'C:\\Chloe\\secrets', "files": ('chloe_jobs.py',)},
    "CHLOE_SILENCE_HANG_MS": {"default": '2000', "files": ('jarvis.py',)},
    "CHLOE_SILENCE_RMS": {"default": '0.004', "files": ('jarvis.py',)},
    "CHLOE_SOCIAL_DB": {"default_expr": 'str(HERE / "chloe_social.db"', "files": ('social_db.py',)},
    "CHLOE_SOCIAL_GROQ_MODEL": {"default": 'llama-3.3-70b-versatile', "files": ('social_composer.py',)},
    "CHLOE_SOCIAL_OLLAMA_MODEL": {"default_expr": 'os.environ.get("OLLAMA_MODEL", "qwen2.5:32b"', "files": ('social_composer.py',)},
    "CHLOE_SOCIAL_SECRETS_DIR": {"default": 'C:\\Chloe\\secrets', "files": ('social.py',)},
    "CHLOE_SUMMARIZE_AUTO": {"default": '0', "files": ('brain_wiring.py',)},
    "CHLOE_SUMMARIZE_BATCH": {"default": '30', "files": ('brain_wiring.py',)},
    "CHLOE_SUMMARIZE_THRESHOLD": {"default": '50', "files": ('brain_wiring.py',)},
    "CHLOE_SYNOPSIS": {"default": '1', "files": ('chloe_synopsis.py',)},
    "CHLOE_SYNOPSIS_KEEP": {"default": '14', "files": ('chloe_synopsis.py',)},
    "CHLOE_SYNOPSIS_MIN_OLDER": {"default": '8', "files": ('chloe_synopsis.py',)},
    "CHLOE_SYNOPSIS_MODEL": {"default": 'llama3.2:3b', "files": ('chloe_synopsis.py',)},
    "CHLOE_SYNOPSIS_REFRESH": {"default": '6', "files": ('chloe_synopsis.py',)},
    "CHLOE_SYNOPSIS_TIMEOUT": {"default": '20', "files": ('chloe_synopsis.py',)},
    "CHLOE_TOK_MODEL": {"default": None, "files": ('chloe_context.py',)},
    "CHLOE_TOK_TIMEOUT": {"default": '1.5', "files": ('chloe_context.py',)},
    "CHLOE_TONE_GUARD_DEBUG": {"default": '1', "files": ('chloe_tone_guard.py',)},
    "CHLOE_TOOL_SYNTH_MODEL": {"default_expr": 'SEARCH_SYNTH_MODEL', "files": ('jarvis.py',)},
    "CHLOE_TRACE": {"default": '1', "files": ('chloe_trace.py',)},
    "CHLOE_TRACE_PROMPT": {"default": '1', "files": ('chloe_trace.py',)},
    "CHLOE_TRACE_RING": {"default": '50', "files": ('chloe_trace.py',)},
    "CHLOE_TTS_STREAMING": {"default": '0', "files": ('jarvis.py',)},
    "CHLOE_URL_VERIFY_CACHE": {"default_expr": 'str(Path(__file__', "files": ('url_verify.py',)},
    "CHLOE_URL_VERIFY_CACHE_DAYS": {"default": '7', "files": ('url_verify.py',)},
    "CHLOE_URL_VERIFY_TIMEOUT": {"default": '6', "files": ('url_verify.py',)},
    "CHLOE_VISION_AMBIENT_MINUTES": {"default": '', "files": ('ambient_vision.py',)},
    "CHLOE_VISION_BLOCKLIST": {"default": '', "files": ('screen_vision.py',)},
    "CHLOE_VISION_DISABLED": {"default": '', "files": ('ambient_vision.py', 'brain_wiring.py', 'jarvis.py', 'screen_vision.py')},
    "CHLOE_VOICE_A": {"default": 'af_heart', "files": ('audio_overview.py',)},
    "CHLOE_VOICE_B": {"default": 'am_michael', "files": ('audio_overview.py',)},
    "CHLOE_VOICE_STREAMING": {"default": '0', "files": ('jarvis.py',)},
    "CHLOE_WAKE_CHIRP": {"default": '1', "files": ('jarvis.py',)},
    "CHLOE_WAKE_THRESHOLD": {"default": '0.5', "files": ('jarvis.py',)},
    "CHLOE_WALLET_DAILY_CAP_SAT": {"default": '', "files": ('wallet_guard.py',)},
    "CHLOE_WALLET_NETWORK": {"default": 'mainnet', "files": ('wallet.py',)},
    "CHLOE_WALLET_SECRETS_DIR": {"default": 'C:\\Chloe\\secrets', "files": ('wallet.py', 'wallet_guard.py')},
    "CHLOE_WEATHER_LOCATION": {"default": '', "files": ('weather.py',)},
    "CHLOE_WEATHER_UNITS": {"default": '', "files": ('weather.py',)},
    "CHLOE_WHISPER_MODEL": {"default_expr": 'os.environ.get("CHLOE_LOCAL_WHISPER_SIZE", "base.en"', "files": ('jarvis.py',)},
    "CHLOE_WIKI_DB": {"default_expr": 'str(Path(__file__', "files": ('wiki_embedding.py',)},
    "CHLOE_WIKI_DEDUP_THRESHOLD": {"default": '0.85', "files": ('wiki_dedup.py',)},
    "CHLOE_WIKI_EMBED_CAP": {"default": '2000', "files": ('wiki_embedding.py',)},
    "CHLOE_WIKI_INJECT_THRESHOLD": {"default": '0.5', "files": ('wiki_embedding.py',)},
    "CHLOE_WIKI_PATH_BOOST": {"default": '0.10', "files": ('wiki_embedding.py',)},
    "CHLOE_WIKI_PATH_BOOST_CAP": {"default": '0.20', "files": ('wiki_embedding.py',)},
    "CHLOE_WIKI_ROOT": {"default": 'C:\\Chloe\\brain\\wiki', "files": ('wiki_embedding.py',)},
    "CHLOE_WIKI_THRESHOLD": {"default": '0.4', "files": ('wiki_embedding.py',)},
    "CHLOE_WS_HOST": {"default": '0.0.0.0', "files": ('hud_server.py',)},
    "CHLOE_WS_PORT": {"default": '6789', "files": ('brain_http.py', 'hud_server.py')},
    # --- Cowork/Claude scheduling integration flags ---
    "COWORK_SCHEDULED_DIR": {"default": 'C:\\Users\\eleew\\OneDrive\\Documents\\Claude\\Scheduled', "files": ('brain_wiring.py',)},
    # --- Other / third-party env vars ---
    "EDGE_TTS_VOICE": {"default": ['', '?'], "files": ('brain_wiring.py', 'jarvis.py')},
    # --- ElevenLabs TTS flags ---
    "ELEVENLABS_API_KEY": {"default": '', "files": ('jarvis.py',)},
    "ELEVENLABS_MODEL": {"default": 'eleven_turbo_v2_5', "files": ('jarvis.py',)},
    "ELEVENLABS_VOICE_ID": {"default": 'gWVE9uaEr9AGwZO3wYSo', "files": ('jarvis.py',)},
    # --- Finnhub market-data API flags ---
    "FINNHUB_API_KEY": {"default": '', "files": ('chloe_jobs.py',)},
    # --- Groq API flags (vision fallback only -- chat/voice/STT are local) ---
    "GROQ_API_KEY": {"default": '', "files": ('brain_http.py', 'jarvis.py', 'screen_vision.py', 'social_composer.py', 'test_see.py')},
    # --- Legacy jarvis.py-prefixed flags ---
    "JARVIS_MIC": {"default": '', "files": ('jarvis.py', 'mic_check.py')},
    # --- Kokoro local TTS flags ---
    "KOKORO_DIR": {"default_expr": 'str(_THIS_DIR / "kokoro_models"', "files": ('jarvis.py',)},
    "KOKORO_MODEL_PATH": {"default": '', "default_expr": 'str(KOKORO_DIR / "kokoro-v1.0.onnx"', "files": ('audio_overview.py', 'jarvis.py')},
    "KOKORO_SPEED": {"default": '1.0', "files": ('brain_wiring.py', 'jarvis.py')},
    "KOKORO_VOICE": {"default": ['', 'af_heart'], "files": ('brain_wiring.py', 'jarvis.py')},
    "KOKORO_VOICES_PATH": {"default": '', "default_expr": 'str(KOKORO_DIR / "voices-v1.0.bin"', "files": ('audio_overview.py', 'jarvis.py')},
    # --- Ollama connection/model flags ---
    "OLLAMA_KEEP_ALIVE": {"default": '30m', "files": ('bench_ollama.py', 'jarvis.py', 'ollama_keepalive.py', 'splice_latency.py')},
    "OLLAMA_MODEL": {"default": ['llama3.2:3b', 'qwen2.5:32b'], "files": ('bench_ollama.py', 'brain_wiring.py', 'chloe_context.py', 'jarvis.py')},
    "OLLAMA_URL": {"default": 'http://localhost:11434', "files": ('bench_ollama.py', 'brain_http.py', 'brain_wiring.py', 'chloe_context.py', 'chloe_ed_profile.py', 'chloe_memory.py', 'chloe_read.py', 'chloe_synopsis.py', 'jarvis.py', 'screen_vision.py', 'social_composer.py', 'wiki_embedding.py')},
    # --- Porcupine wake-word flags ---
    "PORCUPINE_ACCESS_KEY": {"default": '', "files": ('jarvis.py',)},
    # --- Other / third-party env vars ---
    "USERNAME": {"default": '', "files": ('social.py', 'wallet.py', 'wallet_guard.py')},
    # --- TTS backend toggles ---
    "USE_ELEVENLABS": {"default": ['', '0'], "files": ('brain_wiring.py', 'jarvis.py')},
    "USE_KOKORO": {"default": ['', '1'], "files": ('brain_wiring.py', 'jarvis.py')},
}

# Prefixes this codebase actually uses -- unknown_chloe_env_vars()
# only flags a set-but-unrecognized var if it starts with one of
# these, so it never warns about unrelated system/OS env vars (PATH,
# HOME, etc.) that just happen to be present in the process.
_KNOWN_PREFIXES = ('CHLOE_', 'OLLAMA_', 'KOKORO_', 'ELEVENLABS_', 'USE_', 'BREEZ_', 'PORCUPINE_', 'GROQ_', 'BRAVE_', 'FINNHUB_', 'JARVIS_', 'BENCH_', 'COWORK_')


def all_known_names() -> frozenset:
    """Every env var name this codebase is known to read."""
    return frozenset(ENV_REGISTRY)


def describe(name: str) -> str:
    """One-line summary of a registered flag: its default (or the
    source expression it's derived from, when it isn't a plain
    literal) and which file(s) read it. Returns an honest "not
    registered" message for an unknown name rather than guessing."""
    info = ENV_REGISTRY.get(name)
    if info is None:
        return f"{name!r} is not in ENV_REGISTRY -- unrecognized or removed."
    files = ", ".join(info["files"])
    if "default" in info:
        default_part = f"default={info['default']!r}"
    elif "default_expr" in info:
        default_part = f"default is computed: {info['default_expr']}"
    else:
        default_part = "no default -- required if read"
    return f"{name}: {default_part} (read in {files})"


def unknown_chloe_env_vars() -> list[str]:
    """Scan the live process environment for any SET variable that
    starts with one of this codebase's own prefixes (_KNOWN_PREFIXES)
    but isn't in ENV_REGISTRY -- the exact bug class this session kept
    finding by hand: a typo ('CHLOE_BARGE_TRESHOLD'), a renamed flag
    whose old .env line never got cleaned up, or a flag added to code
    without ever being added here. Returns an empty list when
    everything set matches something registered. Never raises.
    """
    try:
        known = all_known_names()
        hits = []
        for name in os.environ:
            if name in known:
                continue
            if any(name.startswith(p) for p in _KNOWN_PREFIXES):
                hits.append(name)
        return sorted(hits)
    except Exception:
        return []

