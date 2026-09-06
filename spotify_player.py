"""
spotify_player.py -- OS-level playback CONTROL, and now-playing status,
for Ed's Spotify DESKTOP app (not the Web API's /me/player endpoints --
those return 403 "Premium required" on Ed's free-tier account for
EVERY /me/player call, control actions included AND, live-confirmed
2026-09-06, the supposedly read-only GET /me/player/currently-playing
too: "Active premium subscription required for the owner of the app."
Correction to this module's own earlier assumption -- see item 4 below.
Search and playlist CRUD are unaffected (different endpoint family) and
still go through spotify_api.py's Web API wrapper).

Three separate OS-level mechanisms, each picked because it works
regardless of Premium status:

1. Play/pause/next/previous -- simulated GLOBAL MEDIA KEYS (the same
   VK_MEDIA_PLAY_PAUSE / VK_MEDIA_NEXT_TRACK / VK_MEDIA_PREV_TRACK keys
   a physical keyboard's media buttons send). Windows routes these to
   whichever app owns the active System Media Transport Controls
   (SMTC) session -- Spotify's desktop client registers as an SMTC
   session the moment it's playing anything, Premium or not, so this
   works exactly like pressing a physical play/pause key on a
   keyboard. No window focus needed. Sent via ctypes SendInput (the
   modern, recommended Win32 input-injection API -- keybd_event is
   the older/deprecated one).

2. Play a SPECIFIC track/playlist/album -- the `spotify:` URI protocol
   handler, e.g. `spotify:track:<id>` or `spotify:playlist:<id>`,
   launched via os.startfile(). Spotify's installer registers itself
   as the OS handler for this URI scheme; handing it a URI opens (or
   focuses) the desktop app and starts playing that specific item
   directly -- this is a completely different code path from the Web
   API's POST /me/player/play (which is Premium-gated), so it works on
   any tier. spotify_commands.py resolves the URI via spotify_api.search()
   /resolve_playlist() first, then hands the finished URI here.

3. Shuffle toggle -- there is no global media key for shuffle, so this
   is the one operation that needs the Spotify window to actually have
   keyboard focus: find the window (class name "SpotifyMainWindow",
   Spotify's own stable Win32 window class), bring it to the
   foreground, then send Ctrl+S -- Spotify's own documented desktop
   keyboard shortcut for toggling shuffle. Windows can refuse a
   SetForegroundWindow call from a background process it doesn't
   consider "the current foreground app" (its anti-focus-stealing
   protection) -- when that happens this degrades to "best effort,
   logged, doesn't raise", same honest-failure style as
   youtube_player.py's _focus_player(). Ed should notice if a live
   "shuffle" voice command doesn't visibly toggle the shuffle icon and
   can report it -- this is flagged as unverified until he does.

4. get_now_playing() -- title/artist/album/is_playing/position, read
   straight from Windows' own System Media Transport Controls (SMTC)
   session info via the `winsdk` package, instead of Spotify's Web API.
   This is the SAME OS-level session that already makes the global
   media keys above work (Spotify registers one automatically the
   moment it's playing anything), so reading its metadata is no new
   trust boundary -- and critically, it isn't gated by Spotify's own
   account tier or API-access approval AT ALL, since it never talks to
   Spotify's servers. Filters GlobalSystemMediaTransportControlsSession
   Manager's sessions down to the one whose source_app_user_model_id
   names Spotify (a machine can have several media sessions open at
   once -- a browser tab, this same YouTube player, etc.). Does NOT
   provide album art (SMTC exposes a thumbnail as an in-memory stream,
   not a URL) -- spotify_hud.py does a best-effort spotify_api.search()
   lookup for that instead, since search is unaffected by the Premium
   restriction above.

Windows-only end to end (ctypes.windll, pywin32, winsdk) -- pywin32 is
already a requirements.txt dependency (used elsewhere for foreground-
window detection); winsdk is new, added for this module's SMTC access
alone. Every public function degrades to a logged, non-raising
{"ok": False, "error": ...} (or None, for get_now_playing) on a
non-Windows host or if pywin32/winsdk/ctypes access fails, rather than
crashing whatever voice/chat thread called it -- this bridge session
cannot live-test any of this on Ed's actual machine, so defensive-by-
construction matters more than usual here.

Public API
----------
play_pause() -> dict
next_track() -> dict
previous_track() -> dict
play_uri(uri: str) -> dict
toggle_shuffle() -> dict
is_spotify_running() -> bool
get_now_playing() -> dict | None
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

_IS_WINDOWS = sys.platform == "win32"

# Virtual-key codes for the global media keys (winuser.h).
_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_PLAY_PAUSE = 0xB3
_VK_CONTROL = 0x11
_VK_S = 0x53

_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_EXTENDEDKEY = 0x0001

_SPOTIFY_WINDOW_CLASS = "SpotifyMainWindow"


# --------------------------------------------------------------------------- #
# ctypes SendInput plumbing (global media keys + the Ctrl+S shuffle chord)    #
# --------------------------------------------------------------------------- #
# Standard SendInput structures -- this exact layout (with the union'd
# padding field) is the well-known, widely-documented way to inject a
# single keyboard event on 64-bit Windows via user32!SendInput. Kept
# self-contained here (no extra dependency) since pywin32's own
# win32api.keybd_event wraps the OLDER, deprecated keybd_event call,
# not SendInput.

if _IS_WINDOWS:
    import ctypes.wintypes as wintypes

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]

    _INPUT_KEYBOARD = 1

    def _send_key_event(vk: int, key_up: bool, extended: bool = False) -> None:
        flags = (_KEYEVENTF_KEYUP if key_up else 0) | \
                (_KEYEVENTF_EXTENDEDKEY if extended else 0)
        inp = _INPUT(type=_INPUT_KEYBOARD,
                     union=_INPUT_UNION(ki=_KEYBDINPUT(vk, 0, flags, 0, None)))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _press_media_key(vk: int) -> None:
        # Media keys are "extended" keys on a real keyboard -- setting
        # the flag matches what a physical key send looks like and is
        # what SMTC-listening apps expect.
        _send_key_event(vk, key_up=False, extended=True)
        _send_key_event(vk, key_up=True, extended=True)

    def _press_ctrl_s() -> None:
        _send_key_event(_VK_CONTROL, key_up=False)
        _send_key_event(_VK_S, key_up=False)
        _send_key_event(_VK_S, key_up=True)
        _send_key_event(_VK_CONTROL, key_up=True)
else:
    def _press_media_key(vk: int) -> None:
        raise OSError("global media keys are only implemented on Windows")

    def _press_ctrl_s() -> None:
        raise OSError("shuffle toggle is only implemented on Windows")


def _safe(op_name: str, fn) -> dict:
    """Run fn(), returning {"ok": True} on success or a logged, non-
    raising {"ok": False, "error": ...} on any failure -- this module
    must never crash a caller's voice/chat thread (see module
    docstring)."""
    if not _IS_WINDOWS:
        msg = f"{op_name} is only implemented on Windows"
        print(f"[spotify_player] {msg}", flush=True)
        return {"ok": False, "error": msg}
    try:
        fn()
        return {"ok": True}
    except Exception as e:
        print(f"[spotify_player] {op_name} failed: {e}", flush=True)
        return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# Window lookup (shuffle only -- see module docstring)                       #
# --------------------------------------------------------------------------- #

def _find_spotify_hwnd() -> Optional[int]:
    try:
        import win32gui
    except ImportError:
        return None
    hwnd = win32gui.FindWindow(_SPOTIFY_WINDOW_CLASS, None)
    return hwnd if hwnd else None


def is_spotify_running() -> bool:
    """Best-effort check via the window class -- returns False (never
    raises) if pywin32 isn't available or the window isn't found (app
    closed, or minimized to tray in a way that unregisters the window --
    uncommon but possible)."""
    if not _IS_WINDOWS:
        return False
    try:
        return _find_spotify_hwnd() is not None
    except Exception:
        return False


def _focus_spotify_window() -> bool:
    """Best-effort foreground focus for the shuffle keystroke. Windows'
    own anti-focus-stealing protection can refuse a SetForegroundWindow
    call from a background process -- this never raises either way;
    _press_ctrl_s() still fires regardless, since a partial focus can
    sometimes still deliver the keystroke (same honest-best-effort
    pattern as youtube_player.py's _focus_player)."""
    try:
        import win32con
        import win32gui
    except ImportError:
        return False
    hwnd = _find_spotify_hwnd()
    if hwnd is None:
        return False
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as e:
        print(f"[spotify_player] focus of Spotify window missed (continuing "
              f"anyway): {e}", flush=True)
        return False


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def play_pause() -> dict:
    """Toggles play/pause via the global media key -- works whether
    Spotify is currently playing or paused, exactly like a physical
    media-key press. No way to know in advance which state it'll end
    up in without reading current playback first (spotify_api.
    get_current_playback()); callers that need an honest "paused" vs.
    "resumed" reply should check that before and after, same as
    youtube_player.pause()/resume() do for the browser player."""
    return _safe("play_pause", lambda: _press_media_key(_VK_MEDIA_PLAY_PAUSE))


def next_track() -> dict:
    return _safe("next_track", lambda: _press_media_key(_VK_MEDIA_NEXT_TRACK))


def previous_track() -> dict:
    return _safe("previous_track", lambda: _press_media_key(_VK_MEDIA_PREV_TRACK))


def toggle_shuffle() -> dict:
    """Focuses the Spotify window (best-effort) then sends Ctrl+S,
    Spotify's own desktop shortcut for toggling shuffle. UNVERIFIED
    live as of this writing -- flag to Ed if a live "shuffle" command
    doesn't visibly toggle the shuffle icon in the app."""
    def _do():
        _focus_spotify_window()
        _press_ctrl_s()
    return _safe("toggle_shuffle", _do)


def play_uri(uri: str) -> dict:
    """Launch a spotify: URI (spotify:track:<id>, spotify:playlist:<id>,
    spotify:album:<id>) via the OS's registered protocol handler --
    opens/focuses the desktop app and starts playing that specific
    item. This is NOT the Web API's POST /me/player/play (which is
    Premium-gated); it's the same mechanism as clicking a Spotify link
    anywhere else on the system, so it works on any account tier."""
    if not uri or not uri.startswith("spotify:"):
        return {"ok": False, "error": f"not a spotify: URI: {uri!r}"}

    def _do():
        if _IS_WINDOWS:
            os.startfile(uri)  # noqa: this attr only exists on Windows
        else:
            raise OSError("play_uri needs Windows' os.startfile")
    return _safe("play_uri", _do)


# SMTC PlaybackStatus enum values (Windows.Media.Control namespace) --
# hardcoded here rather than imported from winsdk's enum (keeps this
# readable without requiring winsdk to be installed just to read this
# source file / run this module's tests off Windows).
_SMTC_PLAYING = 4


def get_now_playing() -> Optional[dict]:
    """title/artist/album/is_playing/progress_ms/duration_ms for
    whatever Spotify's desktop app currently has loaded, read from
    Windows' own SMTC session info -- NOT Spotify's Web API (see
    module docstring, item 4, for why: that read-only endpoint also
    turned out to require Premium on Ed's account). Returns None off
    Windows, if the `winsdk` package isn't installed, if no SMTC
    session belongs to Spotify right now (app closed or nothing ever
    loaded), or on any lookup failure -- honest miss, never guesses.
    `album_art_url` is always None here (SMTC exposes a thumbnail as an
    in-memory stream, not a fetchable URL) -- spotify_hud.py fills that
    in separately via a search() lookup."""
    if not _IS_WINDOWS:
        return None
    try:
        import asyncio
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as _MediaManager)
    except ImportError as e:
        print(f"[spotify_player] winsdk not installed -- now-playing via "
              f"SMTC disabled (`pip install winsdk`): {e}", flush=True)
        return None

    async def _fetch():
        mgr = await _MediaManager.request_async()
        for session in mgr.get_sessions():
            app_id = (session.source_app_user_model_id or "").lower()
            if "spotify" not in app_id:
                continue
            info = await session.try_get_media_properties_async()
            playback = session.get_playback_info()
            timeline = session.get_timeline_properties()
            return {
                "is_playing": int(playback.playback_status) == _SMTC_PLAYING,
                "track": info.title or None,
                "artists": [info.artist] if info.artist else [],
                "album": info.album_title or None,
                "album_art_url": None,
                "progress_ms": int(timeline.position.total_seconds() * 1000),
                "duration_ms": int(timeline.end_time.total_seconds() * 1000),
            }
        return None

    try:
        return asyncio.run(_fetch())
    except Exception as e:
        print(f"[spotify_player] SMTC now-playing lookup failed: {e}", flush=True)
        return None
