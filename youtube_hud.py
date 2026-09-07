"""
youtube_hud.py -- Feeds the HUD's "now playing" MUSIC panel from Chloe's
existing YouTube player (youtube_player.py's persistent, Playwright-
controlled Brave tab), plus a real-time audio visualizer via a Windows
WASAPI LOOPBACK capture of actual system audio output.

Why this exists (2026-09-07): the MUSIC overlay was originally built
around Spotify (see spotify_hud.py), but Spotify's Web API turned out to
require the APP OWNER'S account to have an active Premium subscription
for essentially everything -- not just playback control, but /me,
/search, and /me/playlists too (a platform-wide policy change from
~Feb-Mar 2026, confirmed live via 403 "Active premium subscription
required for the owner of the app" on all three, 2026-09-06/07). That's
a hard, unfixable-in-code restriction on Ed's free account. Ed asked to
pivot the MUSIC panel to YouTube instead -- which Chloe already has a
mature, working integration for (youtube_player.py/youtube_playlists.py,
built 2026-09-01, well before Spotify): a REAL persistent browser tab
Chloe actually controls, no DRM, no premium gate, real search-and-play
via yt-dlp, real playlist launch. spotify_hud.py/spotify_api.py/
spotify_player.py/spotify_commands.py are left in place, dormant --
harmless, reversible, and still correct code if Ed ever gets Premium --
but this module is what actually drives the MUSIC panel now.

Now-playing data source: youtube_player.get_now_playing() reads title/
channel/is_playing/progress straight from the persistent page's own DOM
(document.title + the real <video> element's .paused/.currentTime/
.duration) -- no network call, no yt-dlp, genuinely live state of
whatever's already loaded in that tab. Album art needs no lookup at all
(unlike Spotify's SMTC path, which has no art URL and had a documented
no-op stub): YouTube's thumbnail CDN serves a public, unauthenticated
image for any video id at a stable, well-known URL pattern
(https://i.ytimg.com/vi/<id>/hqdefault.jpg), so album_art_url is just
built directly from the video id, no search/cache needed.

Visualizer: same WASAPI-loopback + live FFT approach as spotify_hud.py,
duplicated here rather than imported from it -- deliberately, so this
new module can't regress the already-verified-working Spotify capture
path, and vice versa. It captures whatever's actually audible system-
wide, exactly as before: reacts to the YouTube tab's audio specifically
only in the sense that that's what's actually playing when Ed uses this
panel, same honest trade-off spotify_hud.py's own docstring already
documents.

One background thread, started lazily via start() (same lazy-thread
pattern as spotify_hud.py/youtube_player.py's owner thread -- importing
this module must never start capturing audio or touching the browser as
a side effect). IMPORTANT, and easy to get wrong (got it wrong once,
2026-09-07): start()'s poll loop itself must ALSO never be what launches
the YouTube browser -- it only checks youtube_player.is_running() (a
pure status check) and treats "not running yet" as "nothing playing,"
never calling into youtube_player in a way that would trigger its own
lazy browser launch. The browser should only ever open because Ed (or a
voice command) actually asked to play something. Two nested loops:
  - Outer: poll get_now_playing() every _POLL_INTERVAL_S seconds and
    broadcast a "youtube_now_playing" HUD message every tick (not just
    on change -- see _poll_loop, 2026-09-07 fix for progress/visuals
    drifting out of sync over a long track, especially when the
    visualizer itself never manages to start).
  - Inner (only while is_playing is True): open a WASAPI loopback
    stream and broadcast "youtube_visualizer" frames (normalized FFT
    magnitude bins) at roughly _VIZ_FPS per second. Torn down the
    instant is_playing goes False or the poll loop's next tick reports
    something changed.

Defensive by construction throughout (this bridge session cannot
live-test any of this against Ed's real Brave/WASAPI availability):
every failure mode -- sounddevice missing, no WASAPI host API, no
default output device, a mid-stream capture error, the player thread
not being up yet -- is caught, logged once, and degrades to "now-
playing text still updates (or shows nothing), visualizer silently
stays off" rather than crashing jarvis.py's boot thread.

Public API
----------
start() -- idempotent, call once at jarvis.py boot.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import youtube_player
import hud_server

_POLL_INTERVAL_S = 3.0
_VIZ_FPS = 20
_VIZ_BINS = 24
_SAMPLE_RATE = 44100
# One block == one visualizer frame's worth of audio (_SAMPLE_RATE /
# _VIZ_FPS samples). Previously a fixed 1024 (~23.2ms) regardless of
# _VIZ_FPS, which under-consumed audio relative to the ~50ms/frame the
# loop was paced to via a manual sleep -- see _run_visualizer_until_
# stopped for why that caused an unbounded WASAPI capture backlog.
_BLOCK_SIZE = round(_SAMPLE_RATE / _VIZ_FPS)

_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()


def start() -> None:
    """Idempotent -- safe to call more than once (e.g. a HUD reconnect
    path that also wants to make sure the poll loop is alive)."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_poll_loop, name="youtube-hud", daemon=True)
        _thread.start()


def _broadcast(msg: dict) -> None:
    try:
        hud_server.broadcast_sync(json.dumps(msg))
    except Exception as e:
        print(f"[youtube_hud] broadcast failed: {e}", flush=True)


def _get_now_playing() -> Optional[dict]:
    """Never raises -- see module docstring. Also never LAUNCHES the
    browser: checks youtube_player.is_running() first (a pure status
    check) and returns None immediately if it isn't, rather than
    calling get_now_playing() -- which would enqueue a command and
    trigger _ensure_owner_thread() the moment this poll loop's first
    tick runs, opening Brave at every Chloe boot regardless of whether
    Ed ever asked for music. This is the actual fix for that (2026-09-07,
    Ed: "brave browser is coming up on startup of chloe") -- the owner
    thread should only ever start because someone actually requested
    playback (a voice command, or a MUSIC panel action), never as a
    side effect of this poll loop checking in."""
    try:
        if not youtube_player.is_running():
            return None
        return youtube_player.get_now_playing()
    except Exception as e:
        print(f"[youtube_hud] get_now_playing() errored: {e}", flush=True)
        return None


def _safe_ms(seconds) -> Optional[int]:
    """seconds -> milliseconds, or None if seconds is missing/NaN/inf.
    CONFIRMED live crash (Ed's log, 2026-09-07): the raw <video>
    element's .duration (and, in principle, .currentTime) is NaN
    whenever the element has no loaded media metadata -- e.g. exactly
    the moment a playlist auto-advances and the old video has been
    torn down but the new one hasn't loaded metadata yet. That NaN
    survives the JS bridge as a real float('nan') (not None), so
    round(nan) used to throw ValueError here and permanently kill the
    poll thread (see _poll_loop). Logged once per occurrence so a
    persistent NaN (vs. a one-tick blip) is still visible."""
    if seconds is None:
        return None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        print(f"[youtube_hud] non-finite duration/progress reading "
              f"({seconds!r}) from the player -- likely mid-transition "
              f"between tracks, treating as unknown for this tick",
              flush=True)
        return None
    return round(seconds * 1000)


def _build_playing_broadcast(np: dict) -> dict:
    """Build the youtube_now_playing broadcast dict for a currently-
    playing track. Shared by _poll_loop (on state change) and
    _run_visualizer_until_stopped's periodic recheck (so progress_ms/
    duration_ms get re-synced roughly every _POLL_INTERVAL_S instead of
    only once at state change) -- see module docstring, 2026-09-07 fix
    for progress/visuals drifting out of sync over a long track."""
    video_id = np.get("video_id")
    return {
        "type": "youtube_now_playing", "playing": True,
        "video_id": video_id,
        "title": np.get("title"), "channel": np.get("channel"),
        "album_art_url": (
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
            if video_id else None
        ),
        "is_playing": np.get("is_playing"),
        "progress_ms": _safe_ms(np.get("progress_s")),
        "duration_ms": _safe_ms(np.get("duration_s")),
    }


def refresh_now() -> None:
    """Immediately fetch and broadcast now-playing state instead of
    waiting for _poll_loop's next tick. Call this right after a control
    action that changes what's loaded (next/previous/play_playlist/
    search_and_play) -- closes the up-to-_POLL_INTERVAL_S window where
    the HUD's video/album art still shows the track that just ended.
    Safe no-op if the player isn't running (_get_now_playing already
    checks is_running() and never launches the browser itself)."""
    np = _get_now_playing()
    if not np or not np.get("playing"):
        _broadcast({"type": "youtube_now_playing", "playing": False})
    else:
        _broadcast(_build_playing_broadcast(np))


def _poll_loop() -> None:
    print("[youtube_hud] now-playing poll loop started", flush=True)
    while True:
        # Defense in depth, added after a live crash (2026-09-07, see
        # _safe_ms): nothing supervises/restarts this thread, so ANY
        # unhandled exception here used to mean "now-playing updates
        # stop forever for the rest of the process's life" -- silent
        # and easy to mistake for a real playback failure. One bad
        # tick must never take the whole loop down again.
        try:
            np = _get_now_playing()

            if not np or not np.get("playing"):
                _broadcast({"type": "youtube_now_playing", "playing": False})
            else:
                _broadcast(_build_playing_broadcast(np))
        except Exception as e:
            print(f"[youtube_hud] poll tick failed (continuing): {e}",
                  flush=True)

        time.sleep(_POLL_INTERVAL_S)


def _run_visualizer_until_stopped() -> None:
    """Runs the WASAPI-loopback-capture + FFT + broadcast loop until
    playback stops or a capture error occurs. Re-checks
    _get_now_playing() every _POLL_INTERVAL_S (not every frame) so it
    still notices a pause/tab-navigation within one poll interval. See
    spotify_hud.py's identical-shape function for the same reasoning --
    duplicated here on purpose, not shared, per this module's docstring."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        print(f"[youtube_hud] numpy/sounddevice not available -- "
              f"visualizer disabled, now-playing text still works: {e}",
              flush=True)
        time.sleep(_POLL_INTERVAL_S)
        return

    device_index = _find_wasapi_loopback_device(sd)
    if device_index is None:
        print("[youtube_hud] no WASAPI loopback output device found -- "
              "visualizer disabled, now-playing text still works",
              flush=True)
        time.sleep(_POLL_INTERVAL_S)
        return

    last_playback_check = time.time()

    try:
        extra = sd.WasapiSettings(loopback=True)
        with sd.InputStream(device=device_index, channels=2,
                             samplerate=_SAMPLE_RATE, blocksize=_BLOCK_SIZE,
                             dtype="float32", extra_settings=extra) as stream:
            print(f"[youtube_hud] visualizer capture started on device "
                  f"{device_index}", flush=True)
            while True:
                now = time.time()
                if now - last_playback_check > _POLL_INTERVAL_S:
                    last_playback_check = now
                    np_state = _get_now_playing()
                    if not np_state or not np_state.get("is_playing"):
                        print("[youtube_hud] playback stopped -- ending "
                              "visualizer capture", flush=True)
                        return
                    _broadcast(_build_playing_broadcast(np_state))
                try:
                    block, overflowed = stream.read(_BLOCK_SIZE)
                except Exception as e:
                    print(f"[youtube_hud] audio read error, ending "
                          f"visualizer capture: {e}", flush=True)
                    return
                if overflowed:
                    print("[youtube_hud] WASAPI capture overflow "
                          "(frame dropped)", flush=True)
                bins = _fft_bins(block, np)
                _broadcast({"type": "youtube_visualizer", "bins": bins})
    except Exception as e:
        print(f"[youtube_hud] visualizer stream failed to open, "
              f"disabling for this video: {e}", flush=True)
        time.sleep(_POLL_INTERVAL_S)


def _find_wasapi_loopback_device(sd) -> Optional[int]:
    try:
        hostapis = sd.query_hostapis()
        wasapi_idx = next((i for i, h in enumerate(hostapis)
                            if "wasapi" in h["name"].lower()), None)
        if wasapi_idx is None:
            return None
        default_output = hostapis[wasapi_idx].get("default_output_device")
        if default_output is None or default_output < 0:
            return None
        return default_output
    except Exception as e:
        print(f"[youtube_hud] WASAPI device lookup failed: {e}", flush=True)
        return None


def _fft_bins(block, np) -> list:
    """Collapse one audio block into _VIZ_BINS normalized (0..1) log-
    spaced magnitude bins. Identical logic to spotify_hud.py's function
    of the same name -- see that module for the reasoning; duplicated
    here per this module's own docstring."""
    if block.ndim > 1:
        mono = block.mean(axis=1)
    else:
        mono = block
    windowed = mono * np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(windowed))
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    n = len(spectrum)
    edges = np.unique(np.geomspace(1, n, _VIZ_BINS + 1).astype(int))
    bins = []
    for i in range(len(edges) - 1):
        chunk = spectrum[edges[i]:edges[i + 1]]
        bins.append(float(chunk.max()) if len(chunk) else 0.0)
    while len(bins) < _VIZ_BINS:
        bins.append(0.0)
    return [round(b, 3) for b in bins[:_VIZ_BINS]]
