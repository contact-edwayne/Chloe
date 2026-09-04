"""Single source of truth for validating OLLAMA_KEEP_ALIVE.

Root cause of the 2026-08-31 total Ollama outage: OLLAMA_KEEP_ALIVE=-1 in
.env. Ollama's HTTP API accepts `keep_alive` as either a JSON *number*
(seconds; -1 means "never unload") or a duration *string* with a unit
suffix ("30m", "24h", "10s"). It does NOT accept a bare integer typed as a
JSON string -- sending `"keep_alive": "-1"` fails server-side with
`time: missing unit in duration "-1"`. Every call site read the env var as
a plain string and passed it straight through, so one bad value 400'd
every single Ollama call across the app at once: chat, voice, streaming,
search-synth, and embeddings. Made worse by two compounding bugs: the OS-
level env var (30m) was masking the broken .env value until a later,
correct fix (a targeted override for just this var) let it through; and
some call sites' error handling (`requests.HTTPError.__str__`,
`urllib.error.HTTPError.__str__`) doesn't include the response body, so
the actual "missing unit in duration" message was invisible at the call
site that actually failed -- it was only visible because a different,
unrelated call site happened to log its own response body.

This module is now the ONE place OLLAMA_KEEP_ALIVE is read and validated.
Every Ollama call site across the app (jarvis.py, brain_http.py,
brain_wiring.py, screen_vision.py, chloe_embed.py via chloe_memory.py /
wiki_embedding.py) should call get_keep_alive() once at import time and
reuse the result, rather than independently re-reading+re-trusting the
raw env var -- that duplication is exactly why one bad value took down
every call site simultaneously instead of just one.
"""

import os
import re

# Safe fallback when the configured value is malformed. Deliberately
# modest (not "never unload") -- if we can't trust what the user asked
# for, defaulting to "unload after a while" is a much smaller failure
# mode than accidentally pinning something in VRAM forever.
DEFAULT_KEEP_ALIVE = "30m"

_DURATION_RE = re.compile(r"^\d+(\.\d+)?(ns|us|µs|ms|s|m|h)$")


def normalize_keep_alive(raw):
    """Validate/normalize a keep_alive value for Ollama's HTTP API.

    Returns either an int (bare-integer inputs like "-1", "0", "3600" --
    sent as a JSON *number* so Ollama actually parses "never unload"
    correctly) or a duration string with a valid unit suffix ("30m",
    "24h", ...). Anything else -- missing, empty, or malformed -- falls
    back to DEFAULT_KEEP_ALIVE with a loud warning, so a typo degrades to
    "evicts after 30m" instead of "every Ollama call 400s silently."
    """
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_KEEP_ALIVE
    try:
        return int(raw)
    except ValueError:
        pass
    if _DURATION_RE.match(raw):
        return raw
    print(f"[ollama_keepalive] OLLAMA_KEEP_ALIVE={raw!r} is not a valid "
          f"Ollama keep_alive value (need a bare integer like -1, or a "
          f"duration with a unit suffix like 30m / 24h / 10s) — falling "
          f"back to {DEFAULT_KEEP_ALIVE!r} instead of breaking every "
          f"Ollama call.", flush=True)
    return DEFAULT_KEEP_ALIVE


def get_keep_alive():
    """Read OLLAMA_KEEP_ALIVE from the environment and normalize it.

    Call this once per process (module load) and reuse the result --
    every Ollama call site in the app should pass this same value rather
    than re-reading the env var independently.
    """
    return normalize_keep_alive(os.environ.get("OLLAMA_KEEP_ALIVE"))
