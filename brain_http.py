"""brain_http.py - tiny HTTP server for the brain graph view.

Runs on its own thread alongside the WebSocket server. Three routes:

    GET /api/brain/graph        -> JSON {nodes, edges, stats}
    GET /api/brain/page?p=...   -> raw markdown text
    GET /brain-graph.html       -> the visualization page (served from this folder)

Why a separate HTTP server:
    hud_server.py is WebSocket-only. The graph view is served via file:// or
    via a small HTTP listener so the iframe in hud.html can fetch JSON
    without touching jarvis.py or the WS protocol.

Bind config:
    CHLOE_GRAPH_HOST    default "0.0.0.0" (matches WS for PWA reach)
    CHLOE_GRAPH_PORT    default 6790
"""

import json
import os
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs


HERE = Path(__file__).parent.resolve()


def _wiki_dir() -> Path:
    """Resolve the brain wiki directory at request time. Lazy so .env has loaded."""
    root = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
    return root / "wiki"


class _GraphHandler(BaseHTTPRequestHandler):
    # Suppress the per-request log noise; we'll print our own one-liner on boot.
    def log_message(self, fmt, *args):
        pass

    def _json(self, status: int, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
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

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        # ─── /api/brain/graph ───
        if path == "/api/brain/graph":
            try:
                from brain_graph import compute_graph
                g = compute_graph(_wiki_dir())
                self._json(200, g)
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        # ─── /api/brain/page?p=entities/qmd ───
        if path == "/api/brain/page":
            qs = parse_qs(url.query)
            rel = (qs.get("p", [""])[0] or "").strip()
            try:
                from brain_graph import read_page
                r = read_page(_wiki_dir(), rel)
                self._json(200, r)
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        # ─── /brain-graph.html (and / as a redirect) ───
        if path in ("/", "/brain-graph.html"):
            page = HERE / "brain-graph.html"
            if not page.exists():
                self._text(404, "brain-graph.html not found next to brain_http.py")
                return
            self._file(200, page, "text/html; charset=utf-8")
            return

        self._text(404, f"unknown route: {path}")


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
        # Most likely cause: port already in use. Don't crash the boot.
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
                import time; time.sleep(1)
        except KeyboardInterrupt:
            stop()
