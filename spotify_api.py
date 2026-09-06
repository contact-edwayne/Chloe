"""
spotify_api.py -- OAuth + Spotify Web API wrapper for Chloe's music
integration (Ed, 2026-09-06: "create a music player within Chloe that
is integrated with Spotify on my desktop and uses my logged in Spotify
account").

Ed confirmed he's on Spotify's FREE tier. That's a hard, server-side
restriction on the Web API's playback-CONTROL endpoints (POST/PUT
/me/player/play, /pause, /next, /previous, /shuffle, /volume, /seek --
all return 403 "Premium required" for a free account), but everything
else this module does works on any tier: search, reading current
playback (read-only), and full playlist CRUD (create/list/delete/add-
tracks). See spotify_player.py for how actual playback CONTROL is done
instead (OS-level media keys + the spotify: URI handler + a focused
keystroke for shuffle -- none of that goes through this module's
/me/player endpoints).

Mirrors google_contacts.py's OAuth pattern in THIS SAME REPO almost
exactly (that module's own docstring explains the contract in detail;
repeated briefly here):
  - First run: _get_access_token(interactive=True) (only called from the
    --auth CLI below) opens Ed's browser once for consent, runs a tiny
    local http.server to catch the redirect, exchanges the code for an
    access+refresh token pair, and caches it to
    C:\\Chloe\\secrets\\spotify_oauth_token.json.
  - Every call after that: the cached token is loaded and silently
    refreshed via the refresh token if expired. No browser, no prompt,
    no blocking.
  - Every live voice/chat caller uses interactive=False and gets an
    honest None if no usable cached token exists -- never hangs a
    background thread waiting on a consent flow nobody is watching.

One-time manual step ONLY Ed can do (his own Spotify account): register
a free Spotify Developer app at https://developer.spotify.com/dashboard,
add Redirect URI http://127.0.0.1:8888/callback under the app's
settings, then save the Client ID + Client Secret into
C:\\Chloe\\secrets\\spotify_client_secret.json as:
    {"client_id": "...", "client_secret": "..."}
...then run `python spotify_api.py --auth` once.

Public API
----------
search(query, types=("track",), limit=5) -> list[dict]
get_current_playback() -> dict | None          read-only, works on ANY tier
list_playlists(force_refresh=False) -> list[dict]
resolve_playlist(name) -> dict | None          honest miss, never guesses
create_playlist(name, public=False, description="") -> dict | None
delete_playlist(name_or_id) -> bool            actually "unfollow" -- see
                                                docstring on delete_playlist
add_tracks_to_playlist(playlist_id, uris) -> bool
current_user_id() -> str | None

CLI:
    python spotify_api.py --auth
    python spotify_api.py --search "bohemian rhapsody"
    python spotify_api.py --now-playing
    python spotify_api.py --list-playlists
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

import requests

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CLIENT_SECRET_PATH = SECRETS_DIR / "spotify_client_secret.json"
TOKEN_PATH = SECRETS_DIR / "spotify_oauth_token.json"

# Ed must add this EXACT URI under his Spotify Developer app's Redirect
# URIs (Dashboard -> app -> Settings). 127.0.0.1 (not "localhost") is
# Spotify's own requirement for a loopback redirect as of their 2025
# API policy tightening.
REDIRECT_URI = "http://127.0.0.1:8888/callback"
_REDIRECT_PORT = 8888

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"

# Requesting the playback-control scopes too even though Ed's free
# account will get 403 from those specific endpoints today -- harmless
# to request, and this repo won't need a re-consent flow if he ever
# upgrades to Premium.
SCOPES = (
    "playlist-read-private playlist-modify-public playlist-modify-private "
    "user-read-currently-playing user-read-playback-state "
    "user-modify-playback-state"
)

_PLAYLIST_CACHE_TTL_S = 5 * 60  # short -- playlists change more than contacts do
_playlist_cache: dict = {"at": 0.0, "items": None}
_user_id_cache: Optional[str] = None


def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


# --------------------------------------------------------------------------- #
# OAuth                                                                       #
# --------------------------------------------------------------------------- #

def _load_client_secret() -> Optional[tuple[str, str]]:
    if not CLIENT_SECRET_PATH.exists():
        print(f"[spotify_api] {CLIENT_SECRET_PATH} not found -- create a "
              f"Spotify Developer app at "
              f"https://developer.spotify.com/dashboard, add redirect URI "
              f"{REDIRECT_URI}, and save {{\"client_id\":..., "
              f"\"client_secret\":...}} to that path first", flush=True)
        return None
    try:
        data = json.loads(CLIENT_SECRET_PATH.read_text(encoding="utf-8"))
        return data["client_id"], data["client_secret"]
    except Exception as e:
        print(f"[spotify_api] {CLIENT_SECRET_PATH} is unreadable/malformed: "
              f"{e}", flush=True)
        return None


def _load_token() -> Optional[dict]:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[spotify_api] cached token at {TOKEN_PATH} is unreadable "
              f"({e}); ignoring it", flush=True)
        return None


def _save_token(tok: dict) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_PATH.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(tok), encoding="utf-8")
    os.replace(tmp, TOKEN_PATH)


def _refresh_access_token(refresh_token: str) -> Optional[dict]:
    creds = _load_client_secret()
    if creds is None:
        return None
    client_id, client_secret = creds
    try:
        resp = requests.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            auth=(client_id, client_secret),
            timeout=10,
        )
        if not resp.ok:
            print(f"[spotify_api] token refresh failed: {resp.status_code} "
                  f"{resp.text[:500]}", flush=True)
            return None
        data = resp.json()
    except Exception as e:
        print(f"[spotify_api] token refresh failed: {e}", flush=True)
        return None
    tok = {
        "access_token": data["access_token"],
        # Spotify doesn't always rotate the refresh token -- keep the old
        # one if a new one wasn't issued.
        "refresh_token": data.get("refresh_token", refresh_token),
        "expires_at": time.time() + data.get("expires_in", 3600) - 30,
    }
    _save_token(tok)
    return tok


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Catches the one redirect from Spotify's consent screen. Stores the
    authorization `code` (or an error) on the class itself -- this tiny
    server handles exactly one request then the CLI flow shuts it down,
    so there's no concurrency to worry about."""
    result: dict = {}

    def do_GET(self):  # noqa: N802 (stdlib method name)
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if "code" in qs:
            _CallbackHandler.result = {"code": qs["code"][0]}
            body = b"Spotify connected -- you can close this tab and go back to Chloe."
        else:
            _CallbackHandler.result = {"error": qs.get("error", ["unknown"])[0]}
            body = b"Spotify auth failed -- go back to the terminal for details."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence the default stderr logging
        pass


def _run_consent_flow(client_id: str, client_secret: str) -> Optional[dict]:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
    }
    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    print("[spotify_api] opening your browser for one-time Spotify "
          "consent...", flush=True)
    server = http.server.HTTPServer(("127.0.0.1", _REDIRECT_PORT), _CallbackHandler)
    _CallbackHandler.result = {}
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    webbrowser.open(url)
    t.join(timeout=180)
    server.server_close()
    result = _CallbackHandler.result
    if "code" not in result:
        print(f"[spotify_api] consent did not complete: "
              f"{result.get('error', 'timed out waiting for the redirect')}",
              flush=True)
        return None
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": result["code"],
                "redirect_uri": REDIRECT_URI,
            },
            auth=(client_id, client_secret),
            timeout=10,
        )
        if not resp.ok:
            # raise_for_status()'s own message doesn't include the response
            # BODY, which is exactly where Spotify puts the actual reason
            # (invalid_client, invalid_grant, redirect_uri_mismatch, etc.)
            # -- printing it directly makes this self-diagnosing instead of
            # a guessing game over chat.
            print(f"[spotify_api] token exchange failed: {resp.status_code} "
                  f"{resp.text[:500]}", flush=True)
            return None
        data = resp.json()
    except Exception as e:
        print(f"[spotify_api] token exchange failed: {e}", flush=True)
        return None
    tok = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + data.get("expires_in", 3600) - 30,
    }
    _save_token(tok)
    print(f"[spotify_api] consent complete, token cached to {TOKEN_PATH} "
          f"-- future calls won't prompt again", flush=True)
    return tok


def _get_access_token(*, interactive: bool = False) -> Optional[str]:
    """Same contract as google_contacts._get_credentials: returns an
    access token string, refreshing silently if expired, or None (never
    raises, never blocks) if nothing usable exists and `interactive` is
    False. Pass interactive=True ONLY from the --auth CLI entry point."""
    tok = _load_token()
    if tok and tok.get("expires_at", 0) > time.time():
        return tok["access_token"]
    if tok and tok.get("refresh_token"):
        refreshed = _refresh_access_token(tok["refresh_token"])
        if refreshed:
            return refreshed["access_token"]
    if not interactive:
        print("[spotify_api] no valid cached token, and this isn't an "
              "interactive call -- run `python spotify_api.py --auth` "
              "once, interactively, first", flush=True)
        return None
    creds = _load_client_secret()
    if creds is None:
        return None
    consented = _run_consent_flow(*creds)
    return consented["access_token"] if consented else None


# --------------------------------------------------------------------------- #
# Low-level API helpers                                                      #
# --------------------------------------------------------------------------- #

class SpotifyError(Exception):
    """Raised for any non-2xx Spotify response OTHER than the two this
    module treats as expected/recoverable (401 -- handled internally via
    one retry after a token refresh; 403 Premium-required -- returned as
    a normal {"ok": False, "error": "premium_required"} result, not an
    exception, since callers need to give an honest voice reply rather
    than crash)."""


def _request(method: str, path: str, *, params=None, json_body=None,
             _retried: bool = False):
    token = _get_access_token(interactive=False)
    if token is None:
        return None, "not_connected"
    url = path if path.startswith("http") else f"{API_BASE}{path}"
    try:
        resp = requests.request(
            method, url, params=params, json=json_body,
            headers={"Authorization": f"Bearer {token}"}, timeout=10,
        )
    except Exception as e:
        print(f"[spotify_api] {method} {path} network error: {e}", flush=True)
        return None, "network_error"

    if resp.status_code == 401 and not _retried:
        # Token expired between our own expiry check and the call landing
        # (clock skew, or a very short-lived token) -- force one refresh
        # and retry exactly once rather than failing a live voice turn.
        tok = _load_token()
        if tok and tok.get("refresh_token"):
            _refresh_access_token(tok["refresh_token"])
            return _request(method, path, params=params, json_body=json_body,
                             _retried=True)
        return None, "not_connected"

    if resp.status_code == 403:
        # Usually the expected "Premium required" for a playback-control
        # endpoint on Ed's free account (silent by design -- callers give
        # an honest voice reply for that case, not a crash). But GET /me
        # and other non-player endpoints returning 403 is NOT that case --
        # print it either way so an unexpected 403 (e.g. the app's own
        # user-access list in Developer Mode) isn't silently invisible.
        print(f"[spotify_api] {method} {path} -> 403: {resp.text[:300]}", flush=True)
        return None, "premium_required"

    if resp.status_code == 404:
        print(f"[spotify_api] {method} {path} -> 404: {resp.text[:300]}", flush=True)
        return None, "not_found"

    if resp.status_code == 204:  # success, no body (common on PUT/DELETE)
        return {}, None

    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[spotify_api] {method} {path} failed: {e} -- {resp.text[:300]}",
              flush=True)
        return None, "api_error"

    if not resp.content:
        return {}, None
    try:
        return resp.json(), None
    except Exception:
        return {}, None


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def current_user_id() -> Optional[str]:
    global _user_id_cache
    if _user_id_cache:
        return _user_id_cache
    data, err = _request("GET", "/me")
    if err or not data:
        return None
    _user_id_cache = data.get("id")
    return _user_id_cache


def search(query: str, types=("track",), limit: int = 5) -> list:
    """Search Spotify's catalog. Works on ANY account tier (search is
    not a playback-control endpoint). Returns a flat list of
    {"type", "id", "uri", "name", "artists", "album"} dicts across all
    requested types, most-relevant first within each type."""
    if not query or not query.strip():
        return []
    data, err = _request("GET", "/search", params={
        "q": query, "type": ",".join(types), "limit": max(1, min(limit, 50)),
    })
    if err or not data:
        return []
    out = []
    for t in types:
        key = t + "s"  # "track" -> "tracks", "playlist" -> "playlists", etc.
        for item in (data.get(key) or {}).get("items") or []:
            if item is None:  # Spotify sometimes returns nulls for playlists
                continue
            out.append({
                "type": t,
                "id": item.get("id"),
                "uri": item.get("uri"),
                "name": item.get("name"),
                "artists": [a.get("name") for a in item.get("artists") or []],
                "album": (item.get("album") or {}).get("name"),
            })
    return out


def get_current_playback() -> Optional[dict]:
    """Read-only "what's playing right now" -- works on ANY account
    tier, unlike the playback-CONTROL endpoints. Returns None if
    nothing is playing, Chloe isn't connected yet, or the call fails.
    {"is_playing", "track", "artists", "album", "album_art_url",
    "progress_ms", "duration_ms", "uri"}."""
    data, err = _request("GET", "/me/player/currently-playing")
    if err or not data or not data.get("item"):
        return None
    item = data["item"]
    images = (item.get("album") or {}).get("images") or []
    return {
        "is_playing": bool(data.get("is_playing")),
        "track": item.get("name"),
        "artists": [a.get("name") for a in item.get("artists") or []],
        "album": (item.get("album") or {}).get("name"),
        "album_art_url": images[0]["url"] if images else None,
        "progress_ms": data.get("progress_ms"),
        "duration_ms": item.get("duration_ms"),
        "uri": item.get("uri"),
    }


def list_playlists(force_refresh: bool = False) -> list:
    """[{"id", "name", "uri", "tracks_total", "owner_is_me"}, ...].
    Short-TTL cached (playlists change more often than contacts) --
    never raises; an API failure with a cache present returns the
    stale cache rather than an empty list."""
    now = time.time()
    if not force_refresh and _playlist_cache["items"] is not None \
            and now - _playlist_cache["at"] < _PLAYLIST_CACHE_TTL_S:
        return _playlist_cache["items"]

    me = current_user_id()
    items, page_url = [], "/me/playlists?limit=50"
    while page_url:
        data, err = _request("GET", page_url)
        if err or not data:
            if _playlist_cache["items"] is not None:
                return _playlist_cache["items"]  # stale-but-usable
            return []
        for p in data.get("items") or []:
            items.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "uri": p.get("uri"),
                "tracks_total": (p.get("tracks") or {}).get("total"),
                "owner_is_me": (p.get("owner") or {}).get("id") == me,
            })
        nxt = data.get("next")
        page_url = nxt.replace(API_BASE, "") if nxt else None

    _playlist_cache["items"] = items
    _playlist_cache["at"] = now
    return items


def resolve_playlist(name: str) -> Optional[dict]:
    """Case-insensitive exact-then-partial match against Ed's own
    playlists, same honest-miss contract as google_contacts.resolve_google_
    contact -- returns None rather than guessing on no match or an
    ambiguous partial match across multiple different playlists."""
    text = (name or "").strip().lower()
    if not text:
        return None
    playlists = list_playlists()
    exact = [p for p in playlists if (p.get("name") or "").strip().lower() == text]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    partial = [p for p in playlists if text in (p.get("name") or "").lower()]
    if len(partial) == 1:
        return partial[0]
    return None


def create_playlist(name: str, public: bool = False, description: str = "") -> Optional[dict]:
    """Create a new playlist owned by Ed's account. Returns
    {"id", "name", "uri"} on success, None on failure (not connected,
    no name, or the API call failed -- logged either way)."""
    if not name or not name.strip():
        return None
    me = current_user_id()
    if me is None:
        return None
    data, err = _request("POST", f"/users/{me}/playlists", json_body={
        "name": name.strip(), "public": public, "description": description,
    })
    if err or not data:
        print(f"[spotify_api] create_playlist({name!r}) failed: {err}", flush=True)
        return None
    _playlist_cache["items"] = None  # invalidate cache
    return {"id": data.get("id"), "name": data.get("name"), "uri": data.get("uri")}


def delete_playlist(name_or_id: str) -> bool:
    """Spotify's Web API has no true "delete a playlist" operation --
    only "remove the CURRENT USER from the playlist's followers", which
    for a playlist Ed owns has the same practical effect Ed means by
    "delete": it disappears from his library. (DELETE /playlists/{id}/
    followers is the documented mechanism; there's no separate delete
    endpoint at all, even for an owner.) Accepts either a playlist name
    (resolved the same honest-miss way as resolve_playlist) or a raw
    playlist id. Returns True on success, False otherwise (unresolved
    name, not connected, or the API call failed)."""
    entry = None
    if name_or_id and len(name_or_id) == 22 and name_or_id.isalnum():
        entry = {"id": name_or_id}  # looks like a raw Spotify id already
    else:
        entry = resolve_playlist(name_or_id)
    if entry is None:
        return False
    _, err = _request("DELETE", f"/playlists/{entry['id']}/followers")
    if err:
        print(f"[spotify_api] delete_playlist({name_or_id!r}) failed: {err}", flush=True)
        return False
    _playlist_cache["items"] = None
    return True


def add_tracks_to_playlist(playlist_id: str, uris: list) -> bool:
    if not playlist_id or not uris:
        return False
    _, err = _request("POST", f"/playlists/{playlist_id}/tracks",
                       json_body={"uris": uris})
    if err:
        print(f"[spotify_api] add_tracks_to_playlist failed: {err}", flush=True)
        return False
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                        #
# --------------------------------------------------------------------------- #

def _cli_auth() -> int:
    token = _get_access_token(interactive=True)
    if token is None:
        print("auth failed -- see errors above.")
        return 1
    me = current_user_id()
    print(f"Spotify connected -- user id {me!r}. Chloe can now search, "
          f"read what's playing, and manage your playlists. (Playback "
          f"CONTROL -- play/pause/skip/shuffle -- goes through OS-level "
          f"control in spotify_player.py, not this account's API tier.)")
    return 0


def _cli_search(query: str) -> int:
    results = search(query, types=("track",), limit=5)
    if not results:
        print("No results (or not connected -- run --auth first).")
        return 1
    for r in results:
        print(f'{r["name"]} -- {", ".join(r["artists"])} ({r["album"]}) [{r["uri"]}]')
    return 0


def _cli_now_playing() -> int:
    np = get_current_playback()
    if np is None:
        print("Nothing playing (or not connected).")
        return 1
    state = "playing" if np["is_playing"] else "paused"
    print(f'{np["track"]} -- {", ".join(np["artists"])} ({np["album"]}) [{state}]')
    return 0


def _cli_list_playlists() -> int:
    playlists = list_playlists(force_refresh=True)
    if not playlists:
        print("No playlists found (or not connected).")
        return 1
    for p in playlists:
        print(f'{p["name"]} -- {p["tracks_total"]} track(s) [{p["id"]}]')
    return 0


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--auth":
        return _cli_auth()
    if argv[0] == "--search" and len(argv) > 1:
        return _cli_search(" ".join(argv[1:]))
    if argv[0] == "--now-playing":
        return _cli_now_playing()
    if argv[0] == "--list-playlists":
        return _cli_list_playlists()
    print(f"unknown command: {argv[0]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
