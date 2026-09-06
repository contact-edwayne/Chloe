"""
spotify_hud.py -- Feeds the HUD's new "now playing" panel: track/artist/
album (+ art) via polling spotify_api.get_current_playback(), and a
real-time audio visualizer via a Windows WASAPI LOOPBACK capture of
actual system audio output -- NOT Spotify's own audio-analysis API.

Why local audio capture instead of Spotify's audio-analysis endpoint:
Spotify deprecated Audio Features/Audio Analysis for new API
applications in their November 2024 policy tightening (extended-quota
approval required, not available to a small personal app like this
one) -- and even where available, that endpoint returns a PRECOMPUTED,
per-track analysis, not a live audio stream, so it was never going to
drive a real-time visualizer anyway. Capturing actual system audio
output (WASAPI loopback -- "what you hear", the same mechanism apps
like OBS use for desktop-audio capture) and running a live FFT locally
sidesteps both problems: no API tier dependency at all, and it reacts
to whatever's actually audible in the moment, not just Spotify (video
audio, anything else playing, would show up too -- an accepted, honest
trade-off, not a bug).

One background thread, started lazily via start() (same lazy-thread
pattern as youtube_player.py's owner thread -- importing this module
must never start capturing audio or hitting the network as a side
effect). Two nested loops:
  - Outer: poll get_current_playback() every _POLL_INTERVAL_S seconds.
    Broadcasts a "spotify_now_playing" HUD message only when the
    track/is_playing state actually CHANGES (never spams identical
    broadcasts every poll tick).
  - Inner (only while is_playing is True): open a WASAPI loopback
    stream and broadcast a "spotify_visualizer" HUD message
    (normalized FFT magnitude bins) at roughly _VIZ_FPS per second.
    Torn down the instant is_playing goes False or the poll loop's
    next tick reports something changed -- no audio device stays open
    while nothing's playing.

Defensive by construction throughout (this bridge session cannot
live-test any of this against Ed's real audio hardware / WASAPI
availability): every failure mode -- sounddevice missing, no WASAPI
host API on this machine, no default output device, a mid-stream
capture error -- is caught, logged once, and degrades to "now-playing
text still updates, visualizer silently stays off" rather than
crashing jarvis.py's boot thread.

Public API
----------
start() -- idempotent, call once at jarvis.py boot.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import spotify_api
import hud_server

_POLL_INTERVAL_S = 3.0
_VIZ_FPS = 20
_VIZ_BINS = 24
_SAMPLE_RATE = 44100
_BLOCK_SIZE = 1024

_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()
_last_broadcast_state: Optional[tuple] = None


def start() -> None:
    """Idempotent -- safe to call more than once (e.g. a HUD reconnect
    path that also wants to make sure the poll loop is alive)."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_poll_loop, name="spotify-hud", daemon=True)
        _thread.start()


def _broadcast(msg: dict) -> None:
    # hud_server.broadcast_sync sends whatever it's given straight to
    # each websocket client -- structured messages are JSON-encoded
    # here first (matches _broadcast_exchange's own
    # json.dumps({...}) convention; the bare-string "idle"/"thinking"
    # signals elsewhere in jarvis.py are a different, simpler wire
    # convention this module doesn't use).
    try:
        hud_server.broadcast_sync(json.dumps(msg))
    except Exception as e:
        print(f"[spotify_hud] broadcast failed: {e}", flush=True)


def _poll_loop() -> None:
    global _last_broadcast_state
    print("[spotify_hud] now-playing poll loop started", flush=True)
    while True:
        try:
            np = spotify_api.get_current_playback()
        except Exception as e:
            print(f"[spotify_hud] get_current_playback() errored: {e}", flush=True)
            np = None

        state = (np.get("track"), np.get("artists"), np.get("is_playing")) if np else None
        if state != _last_broadcast_state:
            _last_broadcast_state = state
            if np is None:
                _broadcast({"type": "spotify_now_playing", "playing": False})
            else:
                _broadcast({
                    "type": "spotify_now_playing", "playing": True,
                    "track": np.get("track"), "artists": np.get("artists"),
                    "album": np.get("album"), "album_art_url": np.get("album_art_url"),
                    "is_playing": np.get("is_playing"),
                })

        if np and np.get("is_playing"):
            _run_visualizer_until_stopped()
        else:
            time.sleep(_POLL_INTERVAL_S)


def _run_visualizer_until_stopped() -> None:
    """Runs the WASAPI-loopback-capture + FFT + broadcast loop until
    playback stops or a capture error occurs. Re-checks
    get_current_playback() every _POLL_INTERVAL_S (not every frame --
    that would mean a network round-trip per visualizer frame) so it
    still notices a pause/track-end within one poll interval."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        print(f"[spotify_hud] numpy/sounddevice not available -- "
              f"visualizer disabled, now-playing text still works: {e}",
              flush=True)
        time.sleep(_POLL_INTERVAL_S)
        return

    device_index = _find_wasapi_loopback_device(sd)
    if device_index is None:
        print("[spotify_hud] no WASAPI loopback output device found -- "
              "visualizer disabled, now-playing text still works",
              flush=True)
        time.sleep(_POLL_INTERVAL_S)
        return

    last_playback_check = time.time()
    frame_interval = 1.0 / _VIZ_FPS

    try:
        extra = sd.WasapiSettings(loopback=True)
        with sd.InputStream(device=device_index, channels=2,
                             samplerate=_SAMPLE_RATE, blocksize=_BLOCK_SIZE,
                             dtype="float32", extra_settings=extra) as stream:
            print(f"[spotify_hud] visualizer capture started on device "
                  f"{device_index}", flush=True)
            while True:
                now = time.time()
                if now - last_playback_check > _POLL_INTERVAL_S:
                    last_playback_check = now
                    np_state = spotify_api.get_current_playback()
                    if not np_state or not np_state.get("is_playing"):
                        print("[spotify_hud] playback stopped -- ending "
                              "visualizer capture", flush=True)
                        return
                try:
                    block, _overflow = stream.read(_BLOCK_SIZE)
                except Exception as e:
                    print(f"[spotify_hud] audio read error, ending "
                          f"visualizer capture: {e}", flush=True)
                    return
                bins = _fft_bins(block, np)
                _broadcast({"type": "spotify_visualizer", "bins": bins})
                time.sleep(max(0.0, frame_interval - (time.time() - now)))
    except Exception as e:
        print(f"[spotify_hud] visualizer stream failed to open, "
              f"disabling for this song: {e}", flush=True)
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
        print(f"[spotify_hud] WASAPI device lookup failed: {e}", flush=True)
        return None


def _fft_bins(block, np) -> list:
    """Collapse one audio block into _VIZ_BINS normalized (0..1) log-
    spaced magnitude bins -- a standard "spectrum analyzer" reduction,
    not anything Spotify-specific. Mono-mixes stereo first."""
    if block.ndim > 1:
        mono = block.mean(axis=1)
    else:
        mono = block
    windowed = mono * np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(windowed))
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    # Log-spaced bin edges bias resolution toward bass/mid frequencies,
    # where most visible music energy actually lives, same as any
    # commercial visualizer/EQ display.
    n = len(spectrum)
    edges = np.unique(np.geomspace(1, n, _VIZ_BINS + 1).astype(int))
    bins = []
    for i in range(len(edges) - 1):
        chunk = spectrum[edges[i]:edges[i + 1]]
        bins.append(float(chunk.max()) if len(chunk) else 0.0)
    while len(bins) < _VIZ_BINS:
        bins.append(0.0)
    return [round(b, 3) for b in bins[:_VIZ_BINS]]
