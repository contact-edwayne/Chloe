"""brain_http.py - tiny HTTP server for the brain graph view.

Runs on its own thread alongside the WebSocket server. Routes:

    GET  /api/brain/graph        -> JSON {nodes, edges, stats} (v2)
                                    Query: ?include_facts=1&since_ts=<epoch>
    GET  /api/brain/page?p=...   -> {ok, text, frontmatter, path}
    GET  /api/brain/stats        -> light stats (cheap polling fallback for
                                    the brain-stats pane; same shape as
                                    /graph's `stats` block but no node list)
    GET  /api/brain/events       -> Server-Sent Events stream from event_bus
                                    (upserted/deleted/ingested events)
    POST /api/brain/ingest       -> drop-in ingest. Body either:
                                    JSON {url, text, title, dry_run}, or
                                    multipart/form-data with a `file` field.
                                    Returns {slug, tldr, entities_touched,
                                    concepts_touched, similar, raw_path,
                                    dry_run}.
    DELETE /api/brain/ingest?slug=<slug>
                                 -> discard a recent drop-in. Removes
                                    raw/dropin_*<slug>*.md and
                                    wiki/sources/<slug>.md. Entity/concept
                                    pages are NOT reverted (merge-updates
                                    aren't safely reversible) — caller is
                                    notified via response.
    GET  /brain-graph.html       -> the visualization page
    GET  /quick-capture.html     -> URL/text/notes capture page
                                    (paired with desktop bookmarklet +
                                    iOS Shortcut). Forwards to
                                    /api/brain/ingest via JS.

Bind config:
    CHLOE_GRAPH_HOST    default "0.0.0.0" (matches WS for PWA reach)
    CHLOE_GRAPH_PORT    default 6790
"""

from __future__ import annotations

import json
import os
import queue as _queue
import re
import threading
import time
from datetime import datetime
import datetime as _dt  # for _compute_full_health timestamp
from email.parser import BytesParser
from email.policy import default as _email_default
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from ollama_keepalive import get_keep_alive as _get_ollama_keep_alive


HERE = Path(__file__).parent.resolve()

# Load .env in THIS process. brain_http boots on the hud_server thread at t=0,
# before start_jarvis.py imports jarvis.py (~4s later) — and jarvis is what runs
# load_dotenv. When brain_http is its own process it never inherits the parent's
# loaded env at all. Without this, os.environ["OLLAMA_URL"]/OLLAMA_MODEL are
# absent here even though jarvis has them. cwd-independent (Path(__file__)),
# and override=False so a deliberately-set parent env (OLLAMA_URL, CHLOE_GRAPH_*)
# wins. GROQ_API_KEY is loaded too (still used by the vision path elsewhere)
# but /api/brain/chat no longer depends on it — see _handle_chat_post.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=HERE / ".env", override=False)
except Exception:
    pass  # never let env-loading break server import


def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _wiki_dir() -> Path:
    return _brain_root() / "wiki"


# ─── Stage-4 health endpoint ─────────────────────────────────────────────

def _compute_full_health() -> dict:
    """Bounded health snapshot for the watchdog. No LLM calls; each check
    has a clear pass/fail. Critical sub-checks: `ollama_reachable`,
    `memory_db_writable`. Non-critical signal: `wiki_embedded_count`,
    `groq_key_present`, `auto_fact_last_run_ts`, `voice_loop_alive`,
    `ws_connected`."""
    import socket
    import sqlite3
    import urllib.request
    import urllib.error

    issues: list[str] = []
    health: dict = {"checked_at": _dt.datetime.now().isoformat(timespec="seconds")}

    # 1. Ollama
    try:
        ollama_url = os.environ.get("OLLAMA_URL",
                                    "http://localhost:11434").rstrip("/")
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as r:
            ollama_ok = (r.status == 200)
    except Exception as e:
        ollama_ok = False
        issues.append(f"ollama_reachable: {type(e).__name__}: {e}")
    health["ollama_reachable"] = ollama_ok

    # 2. Groq key present (informational only — Groq is retired for chat/
    # STT; still used by the vision fallback, so absence isn't a health
    # issue for /api/brain/chat, only a vision-quality note).
    groq_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
    health["groq_key_present"] = groq_ok

    # 3. Memory DB writable
    # 2026-09-03: this used to just call unsummarized_count() and report
    # True on any successful query -- confirmed via a separate audit that
    # this is a weak test: it proves the connection opens and the schema
    # exists, but not that turns are actually being appended. It would
    # report memory_db_writable: true even against a schema-only DB
    # nothing had ever written a row to. Now also checks that the most
    # recent turn is fresh (within CHLOE_MEMORY_STALE_HOURS, default 6 --
    # generous enough to not false-positive overnight/away-from-Chloe
    # gaps) so a genuinely stalled write path actually fails this check.
    memory_ok = False
    try:
        from jarvis import _memory  # type: ignore
        if _memory is not None:
            _ = _memory.unsummarized_count()  # still exercises a real query
            last_ts = _memory.most_recent_turn_ts()
            stale_hours = float(os.environ.get("CHLOE_MEMORY_STALE_HOURS", "6"))
            if last_ts is None:
                issues.append("memory_db_writable: turns table has no rows "
                             "(DB reachable, but nothing has ever been "
                             "written)")
            else:
                age_hours = (time.time() - last_ts) / 3600.0
                health["memory_last_turn_age_hours"] = round(age_hours, 2)
                if age_hours > stale_hours:
                    issues.append(
                        f"memory_db_writable: most recent turn is "
                        f"{age_hours:.1f}h old (> {stale_hours:.0f}h "
                        f"threshold) -- turn logging may have stalled")
                else:
                    memory_ok = True
    except Exception as e:
        issues.append(f"memory_db_writable: {type(e).__name__}: {e}")
    health["memory_db_writable"] = memory_ok

    # 4. Wiki embed count (non-critical)
    try:
        from wiki_embedding import get_store
        store = get_store()
        health["wiki_embedded_count"] = int(store.count_embedded())
    except Exception as e:
        health["wiki_embedded_count"] = -1
        issues.append(f"wiki_embedded_count: {type(e).__name__}: {e}")

    # 5. WS port reachable on localhost
    ws_ok = False
    try:
        port = int(os.environ.get("CHLOE_WS_PORT", "6789"))
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            ws_ok = True
    except Exception:
        # WS not connectable — could be normal during boot. Don't issue.
        pass
    health["ws_connected"] = ws_ok

    # 6. Voice loop alive — check the daemon thread is in jarvis
    voice_ok = False
    try:
        import threading
        for t in threading.enumerate():
            if t.name.startswith(("chloe-voice", "voice")) and t.is_alive():
                voice_ok = True
                break
    except Exception as e:
        issues.append(f"voice_loop_alive: {type(e).__name__}: {e}")
    health["voice_loop_alive"] = voice_ok

    # 7. Auto-fact daemon (2026-09-03: actually tracked now -- was
    # hardcoded None with a "not currently tracked" comment, which is
    # exactly why a 2026-09-01/09-02 gap in the voice dispatch wiring
    # (auto-fact was chat-only; voice never called it) went unnoticed by
    # this health check for weeks. brain_wiring.maybe_auto_extract sets
    # the underlying timestamp every time an extraction actually runs.
    try:
        from brain_wiring import get_auto_fact_last_run_ts
        _ts = get_auto_fact_last_run_ts()
        health["auto_fact_last_run_ts"] = (
            _dt.datetime.fromtimestamp(_ts).isoformat(timespec="seconds")
            if _ts else None)
    except Exception as e:
        health["auto_fact_last_run_ts"] = None
        issues.append(f"auto_fact_last_run_ts: {type(e).__name__}: {e}")

    # Tally critical-only failures. ws_connected + voice_loop_alive +
    # auto_fact_last_run_ts are signal, not pass/fail.
    critical_keys = ("ollama_reachable", "memory_db_writable")
    checks_failed = sum(1 for k in critical_keys if not health.get(k))
    checks_ok = sum(1 for k in critical_keys if health.get(k))
    health["checks_ok"] = checks_ok
    health["checks_failed"] = checks_failed
    health["issues"] = issues
    return health


# ─── helpers used by the drop-in ingest endpoint ────────────────────────────

_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\n{3,}")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# ─── ROM library (emulator panel) ───────────────────────────────────────────
# ROMs live in CHLOE_ROMS_DIR (default C:\Chloe\roms). The libretro core is
# inferred from the file extension so EmulatorJS can pick the right system.
_ROM_SYSTEMS = {
    ".nes": "nes", ".fds": "nes",
    ".gb": "gb", ".gbc": "gb", ".sgb": "gb",
    ".gba": "gba",
    ".sfc": "snes", ".smc": "snes",
    ".md": "segaMD", ".gen": "segaMD", ".smd": "segaMD", ".bin": "segaMD",
}


def _roms_dir() -> Path:
    return Path(os.environ.get("CHLOE_ROMS_DIR", r"C:\Chloe\roms"))


def _savestates_dir() -> Path:
    return Path(os.environ.get("CHLOE_SAVESTATES_DIR", r"C:\Chloe\savestates"))


def _savestate_slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return s.strip("._") or "game"


# Injected into the emulator panel when served, so we can fix EmulatorJS chrome
# without re-exporting the bundled panel. EmulatorJS builds `.ejs_parent` (its
# root) + `.ejs_menu_bar` (the control bar) inside the panel's #emu element; the
# bar floats mid-screen unless the parent fills the viewport and the bar is
# pinned to the bottom.
_EMU_STYLE_OVERRIDE = """<style id="chloe-emu-overrides">
#emu .ejs_parent { width:100% !important; height:100% !important; position:absolute !important; inset:0 !important; }
#emu .ejs_menu_bar { position:absolute !important; left:0 !important; right:0 !important; bottom:0 !important; top:auto !important; transform:none !important; }
</style>"""
_IMAGE_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp",
                "image/gif", "image/bmp"}

# Optional reverse-proxy mount prefix. The mobile PWA reaches this server over
# Tailscale at a same-origin path (`https://<host>/brain-api/...`) to dodge
# mixed-content + CORS. Some `tailscale serve --set-path` versions strip the
# matched prefix before forwarding, others pass it through — so we normalize it
# off here and the routes below match identically either way.
_MOUNT_PREFIX = "/brain-api"

def _strip_mount(path: str) -> str:
    if path == _MOUNT_PREFIX:
        return "/"
    if path.startswith(_MOUNT_PREFIX + "/"):
        return path[len(_MOUNT_PREFIX):]
    return path

_DROPIN_VISION_PROMPT = (
    "Describe this image so it can serve as a standalone knowledge-base "
    "entry. Identify what is shown (object/scene/document/diagram/screenshot/"
    "chart/photo). Quote any visible text verbatim (titles, labels, captions, "
    "code, UI text, signage). If it's a diagram or chart, name the axes / "
    "nodes / categories. If it's a screenshot, identify the app or website. "
    "Be specific and factual — no interpretation, no speculation about "
    "intent or context the image doesn't show. 4-8 sentences. No preamble, "
    "no bullet points, no markdown headings."
)


def _safe_slug(text: str, maxlen: int = 60) -> str:
    s = _SLUG_RE.sub("_", (text or "dropin").strip().lower()).strip("_")
    return (s or "dropin")[:maxlen]


def _html_to_text(html: str) -> str:
    """Crude HTML->text. Good enough for Brain.ingest which re-extracts
    everything via the LLM. Strips script/style + tags + collapses
    whitespace. Avoids a bs4 dep on the brain_http path."""
    out = _HTML_SCRIPT_STYLE_RE.sub("", html)
    out = _HTML_TAG_RE.sub(" ", out)
    out = out.replace("&nbsp;", " ").replace("&amp;", "&")
    out = out.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    out = "\n".join(line.rstrip() for line in out.splitlines())
    out = _WHITESPACE_RE.sub("\n\n", out)
    return out.strip()


def _extract_html_title(html: str) -> str:
    m = _HTML_TITLE_RE.search(html)
    if not m:
        return ""
    title = _HTML_TAG_RE.sub("", m.group(1)).strip()
    return title[:200]


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """Return {field_name: {'filename': str|None, 'content': bytes,
    'content_type': str}}. Stdlib-only via email.parser."""
    msg_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + body
    msg = BytesParser(policy=_email_default).parsebytes(msg_bytes)
    parts: dict = {}
    if not msg.is_multipart():
        return parts
    for part in msg.iter_parts():
        cd = part.get("Content-Disposition", "")
        name = None
        filename = None
        for token in cd.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token.split("=", 1)[1].strip('"')
            elif token.startswith("filename="):
                filename = token.split("=", 1)[1].strip('"')
        payload = part.get_payload(decode=True)
        if isinstance(payload, str):
            payload = payload.encode("utf-8", errors="replace")
        parts[name or ""] = {
            "filename": filename,
            "content": payload or b"",
            "content_type": (part.get("Content-Type", "") or "").split(";")[0].strip().lower(),
        }
    return parts


def _looks_like_image(filename: str, content_type: str, head_bytes: bytes) -> str:
    """Return image extension (with dot) if input looks like an image, else ''.
    Order: explicit content-type → filename extension → magic bytes."""
    ct = (content_type or "").lower()
    if ct in _IMAGE_MIMES:
        if ct in ("image/jpeg", "image/jpg"):
            return ".jpg"
        return "." + ct.split("/", 1)[1]
    fn = (filename or "").lower()
    for ext in _IMAGE_EXTS:
        if fn.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if head_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if head_bytes[:4] == b"RIFF" and head_bytes[8:12] == b"WEBP":
        return ".webp"
    if head_bytes[:2] == b"BM":
        return ".bmp"
    return ""


def _describe_image_bytes(image_bytes: bytes) -> tuple[str, str]:
    """Run Chloe's vision model on raw image bytes. Returns (description, model).
    Empty description on failure."""
    try:
        from screen_vision import describe_screen  # type: ignore
    except Exception as e:
        return (f"(vision unavailable: {e})", "")
    try:
        r = describe_screen(image_bytes, prompt=_DROPIN_VISION_PROMPT)
    except Exception as e:
        return (f"(vision call crashed: {type(e).__name__}: {e})", "")
    if not r.get("ok"):
        return (f"(vision failed: {r.get('error','?')})", r.get("model", ""))
    return ((r.get("text") or "").strip(), r.get("model", ""))


def _fetch_url(url: str, timeout: float = 15.0) -> tuple[str, str]:
    """Return (text, title). Raises on network failure."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "chloe-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    if "html" in ctype.lower() or "<html" in text[:1000].lower():
        title = _extract_html_title(text) or url
        return _html_to_text(text), title
    # Plain text / markdown — pass through.
    return text, url


def _similar_for(text: str, exclude_id: str | None = None,
                 limit: int = 6) -> list[dict]:
    """Top-k semantically similar existing wiki pages. Best effort."""
    try:
        from wiki_embedding import WikiEmbeddingStore  # type: ignore
    except Exception:
        return []
    try:
        store = WikiEmbeddingStore()
        hits = store.search(text[:2000], limit=limit)
    except Exception:
        return []
    out = []
    for h in hits:
        path = (h.get("path") or "").strip()
        # WikiEmbeddingStore returns paths relative to wiki_root (no .md).
        node_id = path.replace("\\", "/")
        if node_id.endswith(".md"):
            node_id = node_id[:-3]
        if node_id.startswith("wiki/"):
            node_id = node_id[5:]
        if exclude_id and (node_id == exclude_id or node_id.endswith(f"/{exclude_id}")):
            continue
        out.append({
            "id": node_id,
            "score": float(h.get("score", 0.0)),
            "title": h.get("title") or "",
            "type": h.get("type") or "",
        })
    return out


def _publish_safely(event: dict) -> None:
    try:
        from event_bus import publish  # type: ignore
        publish(event)
    except Exception:
        pass


# ─── HTTP handler ───────────────────────────────────────────────────────────

class _GraphHandler(BaseHTTPRequestHandler):
    # Suppress the per-request log noise; we'll print our own one-liner on boot.
    def log_message(self, fmt, *args):
        pass

    # ---- response helpers ----

    def _json(self, status: int, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, status: int, content: str, ctype: str = "text/plain"):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, status: int, path: Path, ctype: str):
        try:
            data = path.read_bytes()
        except Exception as e:
            self._json(500, {"error": f"failed to read {path.name}: {e}"})
            return
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    # ---- CORS preflight ----

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---- GET ----

    def do_GET(self):
        url = urlparse(self.path)
        path = _strip_mount(url.path)
        qs = parse_qs(url.query)

        if path == "/api/brain/graph":
            try:
                from brain_graph import compute_graph
                include_facts = qs.get("include_facts", ["0"])[0] in ("1", "true", "yes")
                since_ts = qs.get("since_ts", [None])[0]
                since = float(since_ts) if since_ts not in (None, "", "0") else None
                g = compute_graph(_wiki_dir(), include_facts=include_facts,
                                  since_ts=since)
                self._json(200, g)
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if path == "/api/brain/stats":
            # Same compute_graph runs but we strip nodes/edges to keep the
            # payload tiny. Used by the brain-stats pane for cheap polling.
            try:
                from brain_graph import compute_graph
                include_facts = qs.get("include_facts", ["0"])[0] in ("1", "true", "yes")
                g = compute_graph(_wiki_dir(), include_facts=include_facts)
                self._json(200, g.get("stats", {}))
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if path == "/api/brain/page":
            rel = (qs.get("p", [""])[0] or "").strip()
            try:
                from brain_graph import read_page
                r = read_page(_wiki_dir(), rel)
                self._json(200, r)
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if path == "/api/brain/events":
            self._stream_events()
            return

        if path == "/api/health/full":
            # Stage-4 watchdog reads this every 30s. Bounded — no LLM
            # calls, no slow IO. Each check has a clear pass/fail with a
            # one-line issues entry on fail.
            self._json(200, _compute_full_health())
            return

        if path == "/api/brain/chat":
            # GET is for browser preflight only — real path is POST.
            self._text(405, "use POST for /api/brain/chat")
            return

        if path == "/api/brain/dropin_image":
            self._serve_dropin_image(qs.get("name", [""])[0] or "")
            return

        if path in ("/", "/brain-graph.html"):
            page = HERE / "brain-graph.html"
            if not page.exists():
                self._text(404, "brain-graph.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/quick-capture.html":
            page = HERE / "quick-capture.html"
            if not page.exists():
                self._text(404, "quick-capture.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/chess.html":
            # Served file is chess_panel.html (the live, edited panel — the
            # original chess.html got locked read-only by the host).
            page = HERE / "chess_panel.html"
            if not page.exists():
                self._text(404, "chess_panel.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/emulator.html":
            # Served file is emulator_panel.html (the re-exported panel with the
            # canvas-sizing + relative-URL fixes; original emulator.html locked).
            # Lean hand-built arcade page: EmulatorJS owns the viewport (its
            # native control bar / fullscreen / settings / gamepad all work),
            # with a Chloe library + talk-back + explicit save/load state.
            page = HERE / "emulator_lite.html"
            if not page.exists():
                self._text(404, "emulator_lite.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/emulator-mobile.html":
            # iPhone-only arcade UI (mockup device shell + custom touch controls,
            # EmulatorJS' native virtual gamepad disabled). The desktop HUD keeps
            # /emulator.html -> emulator_lite.html unchanged.
            page = HERE / "emulator_mobile.html"
            if not page.exists():
                self._text(404, "emulator_mobile.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/emulator-gb.html":
            # iPhone GB/GBC player built on WasmBoy (Canvas2D, no WebGL) to dodge
            # the EmulatorJS iOS WebKit force-close. Standalone test harness;
            # shares the api/roms + roms/<file> endpoints with the EJS arcade.
            page = HERE / "emulator_gb.html"
            if not page.exists():
                self._text(404, "emulator_gb.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/gen1recomp.html":
            # Native Pokemon Gen-1 recompilation launcher panel — replaces the
            # browser ROM emulator on desktop (see hud.html's ARCADE button).
            # gen1recomp.exe runs as its own OS window; this page only
            # launches/tracks it and drives the same watch/chat WS protocol
            # emulator_lite.html used.
            page = HERE / "gen1recomp_panel.html"
            if not page.exists():
                self._text(404, "gen1recomp_panel.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        if path == "/api/gen1recomp/status":
            self._get_gen1recomp_status()
            return

        if path == "/api/roms":
            self._list_roms()
            return

        if path.startswith("/roms/"):
            self._serve_rom(path[len("/roms/"):])
            return

        if path == "/api/savestate":
            game = parse_qs(urlparse(self.path).query).get("game", [""])[0]
            self._get_savestate(game)
            return

        if path == "/tmprom":
            self._get_tmprom()
            return

        if path == "/api/game_memory":
            game = parse_qs(urlparse(self.path).query).get("game", [""])[0]
            self._get_game_memory(game)
            return

        self._text(404, f"unknown route: {path}")

    def _serve_dropin_image(self, name: str):
        """Serve an archived drop-in image by filename. Path-traversal safe."""
        if not name or "/" in name or "\\" in name or ".." in name:
            self._json(400, {"error": "invalid name"})
            return
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if "." + ext not in _IMAGE_EXTS:
            self._json(400, {"error": "not an image filename"})
            return
        img_path = _brain_root() / "raw" / "dropin_images" / name
        if not img_path.exists():
            self._text(404, "image not found")
            return
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        self._file(200, img_path, mime)

    def _list_roms(self):
        """List ROM files in CHLOE_ROMS_DIR with the system inferred from the
        extension. JSON: {roms:[{name,file,system,size}], dir}."""
        d = _roms_dir()
        roms = []
        try:
            if d.exists():
                for p in sorted(d.iterdir()):
                    if not p.is_file():
                        continue
                    sysname = _ROM_SYSTEMS.get(p.suffix.lower())
                    if not sysname:
                        continue
                    roms.append({"name": p.name, "file": p.name,
                                 "system": sysname, "size": p.stat().st_size})
        except Exception as e:
            self._json(500, {"error": str(e)})
            return
        self._json(200, {"roms": roms, "dir": str(d)})

    def _serve_rom(self, name: str):
        """Serve a ROM by filename from CHLOE_ROMS_DIR. Path-traversal safe —
        only plain filenames directly inside the roms dir are allowed."""
        name = unquote(name or "")
        if not name or "/" in name or "\\" in name or ".." in name:
            self._json(400, {"error": "invalid rom name"})
            return
        p = _roms_dir() / name
        if not p.exists() or not p.is_file():
            self._text(404, "rom not found")
            return
        self._file(200, p, "application/octet-stream")

    def _get_savestate(self, game: str):
        """Serve a saved state for `game` (one slot per game). 404 if none."""
        p = _savestates_dir() / (_savestate_slug(game) + ".state")
        if not p.exists() or not p.is_file():
            self._text(404, "no save state")
            return
        self._file(200, p, "application/octet-stream")

    def _post_savestate(self, game: str):
        """Persist a save state (raw bytes in the POST body) for `game`."""
        slug = _savestate_slug(game)
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length > 0 else b""
            if not data:
                self._json(400, {"error": "empty state body"})
                return
            d = _savestates_dir()
            d.mkdir(parents=True, exist_ok=True)
            (d / (slug + ".state")).write_bytes(data)
            self._json(200, {"ok": True, "game": slug, "bytes": len(data)})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _get_game_memory(self, game: str):
        """Serve `brain/games/<slug>.md` as text/markdown. Empty 200 (with a
        sentinel body) if no page exists yet. Mirrors jarvis._arcade_game_slug
        so the panel and the watch loop agree on the slug rule."""
        import re
        from pathlib import Path
        slug = (game or "").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-") or "unknown-game"
        try:
            try:
                from brain_wiring import BRAIN_ROOT
                root = Path(BRAIN_ROOT)
            except Exception:
                root = Path(r"C:\Chloe\brain")
            p = root / "games" / (slug + ".md")
            if not p.exists() or not p.is_file():
                body = ("# " + (game or "Unknown") + "\n\n"
                        "_No memory yet. Turn Watch on and play a session — "
                        "Chloe will write her notes here when you stop._")
                self._text(200, body, "text/markdown; charset=utf-8")
                return
            body = p.read_text(encoding="utf-8", errors="replace")
            # Hard cap so the panel never has to render a huge file.
            self._text(200, body[-20000:], "text/markdown; charset=utf-8")
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _get_tmprom(self):
        """Serve the last file-picker ROM bytes (one transient slot)."""
        p = _savestates_dir() / "_tmprom"
        if not p.exists() or not p.is_file():
            self._text(404, "no tmprom")
            return
        self._file(200, p, "application/octet-stream")

    def _post_arcade_frame(self):
        """Receive an in-iframe canvas PNG from the arcade panel.

        The arcade page periodically grabs `canvas.toDataURL('image/png')` and
        POSTs raw bytes here. The watch loop in jarvis.py prefers this frame
        over mss screen capture so vision sees ONLY game pixels.
        """
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            # 4 MB cap — a typical retro-game canvas PNG is well under 200KB.
            if length <= 0 or length > 4 * 1024 * 1024:
                self._json(400, {"error": "bad content length"})
                return
            data = self.rfile.read(length)
            if not data:
                self._json(400, {"error": "empty body"})
                return
            try:
                import jarvis  # type: ignore
                stored = jarvis._arcade_set_frame(data)
            except Exception as e:
                self._json(500, {"error": f"frame store failed: {e}"})
                return
            self._json(200, {"ok": True, "bytes": stored})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _get_gen1recomp_status(self):
        try:
            import jarvis  # type: ignore
            self._json(200, jarvis._gen1recomp_status())
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_gen1recomp_launch(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length > 0 else b""
            game = ""
            if data:
                try:
                    game = (json.loads(data.decode("utf-8")) or {}).get("game", "")
                except Exception:
                    game = ""
            import jarvis  # type: ignore
            res = jarvis._gen1recomp_launch(game)
            self._json(200 if res.get("ok") else 500, res)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_gen1recomp_stop(self):
        try:
            import jarvis  # type: ignore
            res = jarvis._gen1recomp_stop()
            self._json(200 if res.get("ok") else 500, res)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_tmprom(self):
        """Persist file-picker ROM bytes so a reload-per-game can fetch them."""
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length > 0 else b""
            if not data:
                self._json(400, {"error": "empty rom body"})
                return
            d = _savestates_dir()
            d.mkdir(parents=True, exist_ok=True)
            (d / "_tmprom").write_bytes(data)
            self._json(200, {"ok": True, "bytes": len(data)})
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ---- POST ----

    def do_POST(self):
        url = urlparse(self.path)
        path = _strip_mount(url.path)
        if path == "/api/brain/ingest":
            self._handle_ingest_post()
            return
        if path == "/api/brain/chat":
            self._handle_chat_post()
            return
        if path == "/api/brain/delete":
            self._handle_node_delete_post()
            return
        if path == "/api/brain/ghosts_bulk_ignore":
            self._handle_ghosts_bulk_ignore_post()
            return
        if path == "/api/savestate":
            game = parse_qs(url.query).get("game", [""])[0]
            self._post_savestate(game)
            return
        if path == "/api/tmprom":
            self._post_tmprom()
            return
        if path == "/api/arcade_frame":
            self._post_arcade_frame()
            return
        if path == "/api/gen1recomp/launch":
            self._post_gen1recomp_launch()
            return
        if path == "/api/gen1recomp/stop":
            self._post_gen1recomp_stop()
            return
        self._text(404, f"POST not supported on: {path}")

    # ---- DELETE ----

    def do_DELETE(self):
        url = urlparse(self.path)
        path = _strip_mount(url.path)
        qs = parse_qs(url.query)
        if path == "/api/brain/ingest":
            slug = (qs.get("slug", [""])[0] or "").strip()
            self._handle_ingest_delete(slug)
            return
        self._text(404, f"DELETE not supported on: {path}")

    # ---- SSE stream ----

    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            from event_bus import subscribe, unsubscribe
        except Exception as e:
            self._text(500, f"event_bus missing: {e}")
            return
        q = subscribe()
        try:
            self.wfile.write(
                f'event: hello\ndata: {{"ts": {time.time()}}}\n\n'.encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    evt = q.get(timeout=15.0)
                except _queue.Empty:
                    try:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    continue
                payload = json.dumps(evt, default=str)
                try:
                    self.wfile.write(
                        f"event: {evt.get('type', 'event')}\ndata: {payload}\n\n"
                        .encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
        finally:
            try:
                unsubscribe(q)
            except Exception:
                pass

    # ---- drop-in ingest impl ----

    def _handle_ingest_post(self):
        content_type = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""

        raw_dir = _brain_root() / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        title_hint = ""
        url_hint = ""
        text_body = ""
        dry_run = False

        try:
            if "application/json" in content_type:
                payload = json.loads(body.decode("utf-8") or "{}")
                dry_run = bool(payload.get("dry_run"))
                url_hint = (payload.get("url") or "").strip()
                title_hint = (payload.get("title") or "").strip()
                text_body = payload.get("text") or ""
                if url_hint and not text_body:
                    text_body, fetched_title = _fetch_url(url_hint)
                    title_hint = title_hint or fetched_title
                if not text_body.strip():
                    self._json(400, {"error": "no url, text, or file provided"})
                    return
            elif "multipart/form-data" in content_type:
                parts = _parse_multipart(body, content_type)
                file_part = parts.get("file")
                if not file_part or not file_part.get("content"):
                    self._json(400, {"error": "multipart missing 'file' field"})
                    return
                title_hint = (parts.get("title", {}).get("content") or b"").decode(
                    "utf-8", errors="replace").strip()
                dry_run = bool(
                    (parts.get("dry_run", {}).get("content") or b"")
                    .decode("utf-8", errors="replace").strip()
                    in ("1", "true", "yes"))
                filename = file_part.get("filename") or "upload"
                title_hint = title_hint or filename
                raw_bytes = file_part["content"]
                file_ctype = file_part.get("content_type") or ""

                img_ext = _looks_like_image(filename, file_ctype, raw_bytes[:32])
                if img_ext:
                    # Vision-describe the image, archive bytes, body becomes
                    # the description so the node is text-searchable.
                    image_dir = _brain_root() / "raw" / "dropin_images"
                    image_dir.mkdir(parents=True, exist_ok=True)
                    img_slug = _safe_slug(title_hint or filename or "image")
                    img_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = image_dir / f"{img_slug}_{img_stamp}{img_ext}"
                    try:
                        img_path.write_bytes(raw_bytes)
                    except Exception as e:
                        self._json(500,
                            {"error": f"failed to archive image: {e}"})
                        return
                    description, vmodel = _describe_image_bytes(raw_bytes)
                    if not description.strip():
                        description = ("(vision returned empty description — "
                                       "image archived at "
                                       f"{img_path.name})")
                    text_body = (
                        f"## Image: {title_hint or filename}\n\n"
                        f"{description}\n\n"
                        f"**Source:** uploaded image\n"
                        f"**Archived at:** `{img_path}`\n"
                    )
                    if vmodel:
                        text_body += f"**Vision model:** `{vmodel}`\n"
                    # Stash so the frontmatter writer can tag it.
                    title_hint = title_hint or filename
                    self._dropin_image_meta = {
                        "image_path": str(img_path),
                        "image_filename": img_path.name,
                        "vision_model": vmodel,
                    }
                else:
                    # Best effort: if it's HTML, convert; else treat as text.
                    head = raw_bytes[:200].lower()
                    if b"<html" in head or b"<!doctype" in head:
                        text_body = _html_to_text(
                            raw_bytes.decode("utf-8", errors="replace"))
                    else:
                        text_body = raw_bytes.decode("utf-8", errors="replace")
                    self._dropin_image_meta = None
            else:
                self._json(415, {"error": f"unsupported content-type: {content_type}"})
                return
        except Exception as e:
            self._json(400, {"error": f"bad request: {type(e).__name__}: {e}"})
            return

        slug_root = _safe_slug(title_hint or url_hint or "dropin")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = raw_dir / f"dropin_{slug_root}_{stamp}.md"

        fm_lines = ["---", "source_type: dropped_in",
                    f"dropped_at: {datetime.now().isoformat(timespec='seconds')}"]
        if url_hint:
            fm_lines.append(f"url: {url_hint}")
        if title_hint:
            fm_lines.append(f"title: {title_hint!r}")
        img_meta = getattr(self, "_dropin_image_meta", None)
        if img_meta:
            fm_lines.append("source_subtype: image")
            fm_lines.append(f"image_path: {img_meta['image_path']!r}")
            fm_lines.append(f"image_filename: {img_meta['image_filename']!r}")
            if img_meta.get("vision_model"):
                fm_lines.append(f"vision_model: {img_meta['vision_model']!r}")
        fm_lines.append("---")
        raw_path.write_text("\n".join(fm_lines) + "\n\n" + text_body,
                            encoding="utf-8")

        # Hand to Brain.ingest.
        try:
            from brain_wiring import BRAIN  # type: ignore
        except Exception as e:
            self._json(500, {"error": f"brain_wiring import failed: {e}"})
            return

        try:
            result = BRAIN.ingest(raw_path, dry_run=dry_run)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json(500, {"error": f"ingest failed: {type(e).__name__}: {e}",
                             "raw_path": str(raw_path)})
            return

        # Compute top-k similar existing pages. Best effort.
        exclude_id = None
        if isinstance(result, dict):
            committed_slug = result.get("slug")
            if committed_slug:
                exclude_id = committed_slug
        similar = _similar_for(text_body, exclude_id=exclude_id)

        # Announce on the bus so the UI can animate the new node + edges.
        if not dry_run and isinstance(result, dict):
            sources_id = f"sources/{result.get('slug')}" if result.get("slug") else None
            _publish_safely({
                "type": "ingested",
                "node_id": sources_id,
                "slug": result.get("slug"),
                "entities_touched": result.get("entities_touched", []),
                "concepts_touched": result.get("concepts_touched", []),
                "similar": [s["id"] for s in similar],
            })

        out = dict(result) if isinstance(result, dict) else {"result": result}
        out["raw_path"] = str(raw_path)
        out["dry_run"] = bool(dry_run)
        out["similar"] = similar
        if url_hint:
            out["url"] = url_hint
        if title_hint:
            out["title"] = title_hint
        if img_meta:
            out["image"] = {
                "filename": img_meta["image_filename"],
                "path": img_meta["image_path"],
                "vision_model": img_meta.get("vision_model", ""),
            }
        # Clear handler-instance state so it doesn't leak to the next request
        # (handlers are pooled per-thread).
        self._dropin_image_meta = None
        self._json(200, out)

    def _handle_node_delete_post(self):
        """Delete or ignore a node.

        Body: JSON {id: str, mode: "file"|"ignore"}.
          - mode="file": real node. Resolve to <wiki_dir>/<id>.md, path-
            traversal guard, unlink. Wiki-watcher picks up the deletion
            and re-embeds. Publishes a `deleted` event.
          - mode="ignore": ghost node. Append slug to
            <brain_root>/.ignore_ghosts. compute_graph filters on next
            request. Publishes a `deleted` event.

        Returns {ok, removed_paths, mode, id} or {error, ...}.
        """
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        node_id = (payload.get("id") or "").strip().replace("\\", "/")
        mode = (payload.get("mode") or "").strip().lower()
        if not node_id or ".." in node_id or node_id.startswith("/"):
            self._json(400, {"error": "invalid id"})
            return
        if mode not in ("file", "ignore"):
            self._json(400, {"error": "mode must be 'file' or 'ignore'"})
            return

        brain = _brain_root()
        wiki = brain / "wiki"

        if mode == "file":
            # Resolve to a real .md path; refuse if missing or outside wiki/.
            target = (wiki / (node_id + ".md")).resolve()
            try:
                target.relative_to(wiki.resolve())
            except Exception:
                self._json(400, {"error": "path escapes wiki dir"})
                return
            if not target.exists():
                self._json(404, {"error": f"no such file: {target.name}"})
                return
            try:
                target.unlink()
            except Exception as e:
                self._json(500, {"error": f"unlink failed: {e}"})
                return
            _publish_safely({
                "type": "deleted",
                "node_id": node_id,
                "removed": [str(target)],
                "delete_mode": "file",
            })
            self._json(200, {
                "ok": True, "mode": "file", "id": node_id,
                "removed_paths": [str(target)],
            })
            return

        # mode == "ignore"
        ignore_file = brain / ".ignore_ghosts"
        already = set()
        if ignore_file.exists():
            try:
                already = {
                    l.strip() for l in ignore_file.read_text(
                        encoding="utf-8").splitlines()
                    if l.strip() and not l.strip().startswith("#")
                }
            except Exception:
                already = set()
        was_new = node_id not in already
        try:
            with ignore_file.open("a", encoding="utf-8") as f:
                if was_new:
                    f.write(node_id + "\n")
        except Exception as e:
            self._json(500, {"error": f"append to .ignore_ghosts failed: {e}"})
            return
        _publish_safely({
            "type": "deleted",
            "node_id": node_id,
            "removed": [],
            "delete_mode": "ignore",
            "was_new": was_new,
        })
        self._json(200, {
            "ok": True, "mode": "ignore", "id": node_id,
            "was_new": was_new, "ignore_file": str(ignore_file),
        })

    def _handle_ghosts_bulk_ignore_post(self):
        """Bulk-ignore every ghost node whose degree <= max_degree.

        Body: JSON {max_degree: int} (default 1, clamped 1..5). Computes the
        matching ghost slugs server-side via
        brain_graph.ghost_slugs_below_degree and appends the new ones to
        <brain_root>/.ignore_ghosts in one pass. compute_graph filters them
        on the next request. Idempotent.

        Returns {ok, max_degree, added: [...], added_count, skipped_count}.
        """
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        try:
            max_degree = int(payload.get("max_degree", 1))
        except (TypeError, ValueError):
            self._json(400, {"error": "max_degree must be an int"})
            return
        max_degree = max(1, min(5, max_degree))

        brain = _brain_root()
        try:
            import brain_graph
            slugs = brain_graph.ghost_slugs_below_degree(
                brain / "wiki", max_degree)
        except Exception as e:
            self._json(500, {"error": f"compute ghosts failed: {e}"})
            return

        ignore_file = brain / ".ignore_ghosts"
        already = set()
        if ignore_file.exists():
            try:
                already = {
                    l.strip() for l in ignore_file.read_text(
                        encoding="utf-8").splitlines()
                    if l.strip() and not l.strip().startswith("#")
                }
            except Exception:
                already = set()
        to_add = [s for s in slugs if s not in already]
        try:
            if to_add:
                with ignore_file.open("a", encoding="utf-8") as f:
                    for s in to_add:
                        f.write(s + "\n")
        except Exception as e:
            self._json(500, {"error": f"append to .ignore_ghosts failed: {e}"})
            return

        _publish_safely({
            "type": "deleted",
            "node_id": "",
            "removed": [],
            "delete_mode": "bulk_ignore",
            "added_count": len(to_add),
            "max_degree": max_degree,
        })
        self._json(200, {
            "ok": True, "mode": "bulk_ignore", "max_degree": max_degree,
            "added": to_add, "added_count": len(to_add),
            "skipped_count": len(slugs) - len(to_add),
        })

    def _handle_chat_post(self):
        """Synchronous chat about one or more brain pages.

        Body: JSON {question: str, context_ids: [str, ...]}.
        Returns: JSON {reply: str, model: str, context_ids: [...]} or
        {error: str}.

        Calls local Ollama directly with a page-context system prompt.
        Bypasses Chloe's full chat pipeline (no WS, no TTS, no Brave
        hedge, no memory write) — this is the v0 "ask about this page"
        surface for the brain-graph side panel. Promote to the full
        pipeline if it becomes the primary chat surface.

        2026-08-31: was Groq llama-3.3-70b (hard 500/502 on any Groq
        failure — this endpoint had NO fallback). Groq is fully retired
        account-wide (404s/413s on every model), so this now calls local
        Ollama directly instead, matching the rest of the app.
        """
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        question = (payload.get("question") or "").strip()
        context_ids = payload.get("context_ids") or []
        if not isinstance(context_ids, list):
            context_ids = [context_ids]
        if not question:
            self._json(400, {"error": "question is required"})
            return

        # Build context blob from referenced pages. Cap each page so a
        # giant page doesn't blow the prompt budget.
        from brain_graph import read_page
        wiki = _wiki_dir()
        context_parts: list[str] = []
        for rel in context_ids[:8]:  # bound the context
            try:
                page = read_page(wiki, rel)
            except Exception:
                continue
            if not page.get("ok"):
                continue
            text = (page.get("text") or "")[:4000]
            context_parts.append(f"=== {rel} ===\n{text}")
        context_blob = "\n\n".join(context_parts) if context_parts else \
            "(no page context — answer from general knowledge.)"

        # Pull Chloe's persona shortlist if available, but keep system
        # prompt lean — full chloe_about.md would cost ~5k tokens.
        persona_hint = (
            "You are Chloe, Edward's voice + chat assistant. You're "
            "speaking inside the brain-graph side panel — short, "
            "specific, no preamble, no hedging. The user just clicked "
            "on a page in their personal knowledge wiki and wants to "
            "discuss it. Answer in 1-4 sentences unless asked for "
            "depth."
        )
        # Inject the current Central date/time so the model doesn't confabulate
        # when asked the time here (this v0 surface bypasses jarvis's full_system
        # and its _now_block, lesson #25). Best-effort: never break the reply.
        try:
            from chloe_clock import now_block
            _now = now_block()
        except Exception:
            _now = ""
        system_prompt = (
            f"{persona_hint}{_now}\n\n"
            f"=== PAGE CONTEXT ===\n{context_blob}\n=== END CONTEXT ==="
        )

        # Local Ollama call.
        import urllib.request
        import urllib.error
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        model = os.environ.get("BRAIN_CHAT_MODEL",
                               os.environ.get("OLLAMA_MODEL", "llama3.2:3b")).strip()
        try:
            num_ctx = int(os.environ.get("CHLOE_OLLAMA_CTX", "16384"))
        except (ValueError, TypeError):
            num_ctx = 8192
        req_body = json.dumps({
            "model":      model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream":     False,
            "keep_alive": _get_ollama_keep_alive(),
            "options": {
                "temperature": 0.6,
                "num_predict": 600,
                "num_ctx":     num_ctx,
            },
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{ollama_url}/api/chat", data=req_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180.0) as r:
                data = json.loads(r.read().decode("utf-8"))
            reply = ((data.get("message") or {}).get("content") or "").strip()
        except urllib.error.HTTPError as e:
            # HTTPError.__str__ is just "HTTP Error 400: Bad Request" --
            # it drops the response BODY, which is where Ollama's actual
            # error message lives (e.g. "time: missing unit in duration
            # \"-1\""). Read it explicitly so this doesn't repeat the
            # 2026-08-31 bug where the real cause was invisible at the
            # call site that actually failed.
            try:
                body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                body = "(could not read error body)"
            self._json(502, {"error": f"ollama call failed: HTTP {e.code}: {body} "
                                       f"(model={model!r})"})
            return
        except Exception as e:
            self._json(502, {"error": f"ollama call failed: {type(e).__name__}: {e} "
                                       f"(is `ollama serve` running? model={model!r})"})
            return
        if not reply:
            reply = "(empty reply)"
        self._json(200, {
            "reply": reply,
            "model": model,
            "context_ids": context_ids,
        })

    def _handle_ingest_delete(self, slug: str):
        if not slug or "/" in slug or ".." in slug:
            self._json(400, {"error": "invalid slug"})
            return
        brain = _brain_root()
        removed = []
        not_reverted = []

        source_page = brain / "wiki" / "sources" / f"{slug}.md"
        if source_page.exists():
            try:
                source_page.unlink()
                removed.append(str(source_page))
            except Exception as e:
                self._json(500, {"error": f"failed to remove source page: {e}",
                                 "removed": removed})
                return

        raw_dir = brain / "raw"
        if raw_dir.exists():
            for raw in raw_dir.glob(f"dropin_*{slug}*.md"):
                try:
                    raw.unlink()
                    removed.append(str(raw))
                except Exception:
                    pass

        # Entity/concept pages get merge-updated by Brain.ingest. Reverting
        # is unsafe (we'd need a per-touch diff). Flag this back to caller.
        ent_dir = brain / "wiki" / "entities"
        con_dir = brain / "wiki" / "concepts"
        for d in (ent_dir, con_dir):
            if not d.exists():
                continue
            for page in d.glob("*.md"):
                try:
                    text = page.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if slug in text:
                    not_reverted.append(str(page))

        if removed:
            _publish_safely({
                "type": "deleted",
                "node_id": f"sources/{slug}",
                "removed": removed,
            })

        self._json(200, {
            "ok": bool(removed),
            "removed": removed,
            "not_reverted_entity_concept_pages": not_reverted,
            "note": ("Entity/concept pages that mention this slug are NOT "
                     "reverted — they may have been merge-updated. Listed "
                     "so you can review."),
        })


_server = None
_thread = None


def start(host: str = None, port: int = None) -> dict:
    """Start the HTTP server on a daemon thread. Idempotent.

    Returns {host, port, running}.
    """
    global _server, _thread
    if _server is not None:
        return {"host": _server.server_address[0],
                "port": _server.server_address[1], "running": True}

    host = host or os.environ.get("CHLOE_GRAPH_HOST", "0.0.0.0")
    port = int(port or os.environ.get("CHLOE_GRAPH_PORT", "6790"))

    try:
        _server = ThreadingHTTPServer((host, port), _GraphHandler)
    except OSError as e:
        print(f"[brain_http] failed to bind {host}:{port} - {e}", flush=True)
        _server = None
        return {"host": host, "port": port, "running": False, "error": str(e)}

    def _run():
        try:
            _server.serve_forever()
        except Exception as e:
            print(f"[brain_http] serve_forever crashed: {e}", flush=True)

    _thread = threading.Thread(target=_run, name="chloe-brain-http", daemon=True)
    _thread.start()
    shown = "localhost" if host in ("127.0.0.1", "localhost") else host
    print(f"[brain_http] graph view served at http://{shown}:{port}/brain-graph.html",
          flush=True)
    return {"host": host, "port": port, "running": True}


def stop():
    """Shut down (mostly for tests)."""
    global _server, _thread
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
    if _thread is not None:
        _thread.join(timeout=2.0)
        _thread = None


if __name__ == "__main__":
    info = start()
    if info.get("running"):
        print(f"serving forever on http://{info['host']}:{info['port']}")
        print("ctrl-c to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop()
