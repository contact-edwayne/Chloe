"""
Persistent browser-controlled YouTube player for Chloe.

Playback goes through actual browser automation of a real YouTube tab
(Ed's choice, 2026-09-01) -- not a local media player -- and launches
non-headless so the tab is actually visible (also Ed's choice). One
dedicated background thread owns the Playwright instance, browser,
context, and the single Page for the life of the jarvis.py process;
every public function here enqueues a command onto that thread rather
than touching Playwright objects directly.

Why a dedicated thread + queue instead of "call Playwright from whichever
thread needs it": Playwright's SYNC API is not thread-safe across
arbitrary threads, and jarvis.py calls into this module from several
different asyncio.to_thread workers (voice thread, PTT thread, chat
handler) -- calling Playwright methods from more than one OS thread
corrupts its internal (greenlet-based) state. So this module never calls
Playwright directly from a caller's thread: every public function
packages (command, args) into a queue.Queue, blocks on a
concurrent.futures.Future for the result, and the ONE owner thread
(_player_loop) is the only code in the whole process that ever touches
the Playwright Page. Plain threading + queue.Queue, no asyncio -- this
is a job queue with exactly one worker, nothing fancier is needed.

Controls use YouTube's own documented player keyboard shortcuts
(Shift+N next, Shift+P previous, k play/pause) rather than clicking CSS
selectors -- selectors break on every YouTube redesign; these shortcuts
are part of YouTube's public player UI and have been stable for years.
A best-effort click (_focus_player(), see fix #4 below) still happens
first, only to give the page keyboard focus / satisfy a user-gesture
requirement -- never load-bearing for which button gets pressed.

Pause/resume state (deviates slightly from a simple manually-tracked
flag): rather than trusting an internally-tracked "_is_paused" bool,
which can silently drift from reality (an ad interrupts, autoplay policy
blocks playback, Ed clicks the tab himself), pause()/resume() read the
actual `<video>` element's `.paused` DOM property live via
page.evaluate() before deciding whether to press "k" -- so pause() is a
no-op (not an accidental resume) if the video is already paused, and
vice versa for resume(). Falls back to pressing the key unconditionally
only if the DOM read fails (e.g. no video element on the current page,
such as the YouTube homepage or about:blank).

stop() navigates to about:blank rather than pausing, so "stop" and
"pause" stay two distinct, honest states -- a paused tab is still
sitting on the video ready to resume; stop actually leaves it. Document
this if you build a voice intent for "stop" later (not requested yet).

Browser + profile (Ed, 2026-09-01): launches Brave specifically (not
Playwright's bundled Chromium -- Brave is itself Chromium-based, so
Playwright drives it the same way) via a DEDICATED automation profile at
C:\Chloe\secrets\brave_profile, not Ed's everyday default Brave
profile. Deliberate: Chromium-family browsers lock their profile
directory against concurrent access, so pointing this at Ed's live
daily-driver profile would break (or get silently blocked) the moment he
also has his regular Brave open -- a real conflict for a background-music
feature that's supposed to run continuously. A separate profile avoids
that entirely and can run alongside his normal browsing. Trade-off: the
FIRST launch is a blank profile, not already signed in -- Ed needs to
sign into Google once in that visible automated window (same one-time
pattern as the YouTube OAuth consent flow); after that it's a persistent
context, so the session is written to disk and reused on every future
launch without asking again. Brave's executable is located via
_find_brave_executable() (checks common Windows install paths, or
CHLOE_BRAVE_PATH env var override); if Brave isn't found at all, this
falls back to Playwright's bundled Chromium (still non-headless, still
functional for skip/pause/etc, just not the profile Ed asked for) rather
than failing outright.

Three follow-up fixes (Ed, 2026-09-01, found via live testing):

1. Ads playing through despite Shields showing "up" and "Aggressive":
   Playwright launches Chromium-family browsers with
   --disable-component-update in its default arg set (keeps automated
   test runs deterministic). Brave's filter lists are themselves
   delivered AS a component via that same updater -- so on a fresh
   profile that's never downloaded them, shields are correctly
   configured but have nothing loaded to block with. Fixed by passing
   ignore_default_args=["--disable-component-update",
   "--disable-background-networking"] to launch_persistent_context so
   Brave can actually fetch its filter lists like a normal install.
   (The lists still have to download once over the network -- give it a
   minute after the first launch under this fix before judging it.)

2. autoplay=1 silently ignored, video sits paused: Chromium blocks
   unmuted autoplay on a domain until it's seen real engagement there;
   a brand-new profile with zero YouTube history doesn't clear that bar.
   Fixed in _dispatch's play_url handler -- after navigating, if the
   video is still paused a beat later, click it. A Playwright click
   dispatches a real synthetic input event, which DOES count as a user
   gesture to Chromium's autoplay policy (unlike calling .play() via
   page.evaluate(), which would still be blocked).

3. Closing the Brave window by hand permanently breaks the feature --
   every command after that failed forever, because _player_loop only
   ever launched the browser once at startup and had no path back if
   that page/context died. Fixed: launch logic is now its own
   _launch_page(pw) helper, callable more than once, and the dispatch
   loop checks page.is_closed() before every command -- on a closed
   page it relaunches automatically (reusing the same profile, so it
   comes back signed in) rather than failing the command outright.

Two more follow-up fixes (Ed, 2026-09-01, round 2 -- live testing again):

4. Skip/pause/resume/autoplay-nudge all clicked the raw <video> element
   with no timeout override and no error handling, so on YouTube's
   current masthead (a "Search or ask a question" box that can sit,
   focused/expanded, over the top of the page while the player is still
   mounting) that click can get intercepted by the search box instead of
   reaching the video -- Playwright then blocks for its full default
   timeout and raises, which crashed the ENTIRE command (that's why
   "skip" silently did nothing: the exception fired before
   keyboard.press("Shift+N") ever ran). Fixed with a new _focus_player()
   helper: force=True (skip Playwright's own "is anything covering this"
   check -- we don't need a perfectly clean click, just something that
   counts as a user gesture), a short timeout, targets #movie_player (the
   whole player container, reliably large once mounted) instead of the
   raw <video> tag (whose computed box can be tiny/misplaced while
   loading), and never raises -- a focus miss is logged and every caller
   still goes on to send its keyboard shortcut regardless, since the
   shortcut can work even without a clean focus click.

5. Relaunching after Ed closes the window by hand could bring back TWO
   tabs of the same playlist: Brave/Chromium treats a window closed out
   from under an automated session as an unclean shutdown and restores
   the previous tab(s) on the next launch, in addition to the fresh tab
   this module then navigates to the new URL -- so the old (pre-close)
   tab and the new one both end up open. Fixed in _launch_page(): after
   launch, wait briefly for any session-restore tab(s) to finish opening,
   then close every page except one, so playback always starts from
   exactly one tab regardless of how the previous session ended.
"""

from __future__ import annotations

import os
import queue
import re
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Optional

_BRAVE_PROFILE_DIR = Path(r"C:\Chloe\secrets\brave_profile")

# Checked in order; first existing path wins. Covers the two common
# Windows install locations (machine-wide vs per-user). Override with
# CHLOE_BRAVE_PATH if Brave lives somewhere else.
_BRAVE_CANDIDATE_PATHS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    os.path.expandvars(
        r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
]


def _find_brave_executable() -> Optional[str]:
    override = os.environ.get("CHLOE_BRAVE_PATH", "").strip()
    if override:
        return override if os.path.isfile(override) else None
    for candidate in _BRAVE_CANDIDATE_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None

_cmd_queue: "queue.Queue[tuple[str, tuple, Future]]" = queue.Queue()
_owner_thread: Optional[threading.Thread] = None
_owner_thread_lock = threading.Lock()

# Keyboard-shortcut commands are near-instant once the browser is up;
# play_url gets a longer budget because it can coincide with the very
# first (cold) browser launch, which takes several seconds on its own
# before page.goto even starts -- same reasoning as _YTDLP_TIMEOUT_S=45
# in youtube_playlists.py for "first real network op after cold start".
_DEFAULT_TIMEOUT_S = 10
_PLAY_URL_TIMEOUT_S = 45

_VIDEO_ID_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})")


# --------------------------------------------------------------------------- #
# Owner thread                                                                #
# --------------------------------------------------------------------------- #

def _ensure_owner_thread() -> None:
    """Start the owner thread on first use. Lazy, not at import time --
    importing this module must never launch a browser as a side effect
    (e.g. a CLI script or test that only imports for its constants)."""
    global _owner_thread
    with _owner_thread_lock:
        if _owner_thread is not None and _owner_thread.is_alive():
            return
        _owner_thread = threading.Thread(
            target=_player_loop, name="youtube-player", daemon=True)
        _owner_thread.start()


def _player_loop() -> None:
    """Runs forever on the dedicated owner thread. Launches Playwright +
    a single non-headless Chromium page ONCE -- persists for the life of
    the process, no relaunch per command (Ed's explicit requirement) --
    then services commands off _cmd_queue until the process exits
    (daemon thread, so no explicit shutdown path is needed)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        print(f"[youtube_player] playwright is not installed -- run "
              f"`pip install playwright` then "
              f"`python -m playwright install chromium`: {e}", flush=True)
        _drain_queue_with_error("playwright is not installed on this machine")
        return

    try:
        pw = sync_playwright().start()
        page = _launch_page(pw)
    except Exception as e:
        print(f"[youtube_player] failed to launch browser: {e}", flush=True)
        _drain_queue_with_error(f"browser launch failed: {e}")
        return
    print("[youtube_player] ready. If this is the profile's first launch "
          "and YouTube shows signed out, sign into Google in this window "
          "once -- the session persists for every future launch.",
          flush=True)

    while True:
        name, args, fut = _cmd_queue.get()
        try:
            if page.is_closed():
                print("[youtube_player] the browser window was closed -- "
                      "relaunching (same profile, should come back signed "
                      "in)...", flush=True)
                page = _launch_page(pw)
            result = _dispatch(page, name, args)
        except Exception as e:
            print(f"[youtube_player] command {name!r} errored: {e}", flush=True)
            fut.set_exception(e)
        else:
            fut.set_result(result)


def _launch_page(pw):
    """Launch Brave (or fall back to bundled Chromium) against the
    dedicated persistent profile and return its page. Callable more than
    once -- used both for the initial startup launch and to recover after
    the window gets closed by hand (see _player_loop)."""
    brave_path = _find_brave_executable()
    if brave_path:
        print(f"[youtube_player] launching Brave ({brave_path}) with the "
              f"dedicated Chloe profile at {_BRAVE_PROFILE_DIR} "
              f"(non-headless)...", flush=True)
    else:
        print("[youtube_player] Brave not found at any known install path "
              "(set CHLOE_BRAVE_PATH to override) -- falling back to "
              "Playwright's bundled Chromium instead", flush=True)
    _BRAVE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    launch_kwargs = {
        "headless": False,
        # Playwright's default args include --disable-component-update,
        # which also blocks Brave's ad/tracker filter lists from ever
        # downloading (they ship via the same component-updater
        # mechanism) -- see module docstring fix #1.
        "ignore_default_args": ["--disable-component-update",
                                 "--disable-background-networking"],
    }
    if brave_path:
        launch_kwargs["executable_path"] = brave_path
    # launch_persistent_context IS the browser+context combined (no
    # separate browser.new_context() step) -- writes its profile to
    # user_data_dir on disk, so the same signed-in session is reused on
    # every future launch instead of starting from blank each time.
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(_BRAVE_PROFILE_DIR), **launch_kwargs)
    # A window closed by hand looks like an unclean shutdown to
    # Brave/Chromium, which can restore the previous tab(s) on the next
    # launch -- asynchronously, just after launch_persistent_context
    # returns. Give that a moment to happen, then collapse down to
    # exactly one page so relaunches never stack up duplicate playlist
    # tabs (see module docstring fix #5).
    try:
        context.wait_for_event("page", timeout=1500)
    except Exception:
        pass
    pages = context.pages
    if pages:
        page = pages[0]
        for extra in pages[1:]:
            try:
                extra.close()
            except Exception:
                pass
    else:
        page = context.new_page()
    return page


def _drain_queue_with_error(msg: str) -> None:
    """If the owner thread can't start Playwright/Chromium at all, fail
    every command that was (or will ever be) enqueued with a clear error
    instead of leaving callers to hang until their own timeout, forever,
    for the rest of the process's life."""
    while True:
        name, args, fut = _cmd_queue.get()
        fut.set_exception(RuntimeError(msg))


def _enqueue(name: str, args: tuple, timeout: float = _DEFAULT_TIMEOUT_S) -> dict:
    """Enqueue a command for the owner thread and block for its result.
    Never touches Playwright itself -- see module docstring."""
    _ensure_owner_thread()
    fut: Future = Future()
    _cmd_queue.put((name, args, fut))
    try:
        return fut.result(timeout=timeout)
    except TimeoutError:
        return {"ok": False, "error": f"{name} timed out after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# Command dispatch -- runs ONLY on the owner thread                           #
# --------------------------------------------------------------------------- #

def _get_paused_state(page) -> Optional[bool]:
    """True/False from the real player <video> element's .paused
    property, or None if it can't be read (no video on the current
    page). Deliberately queries '.html5-main-video' first, not a bare
    'video' selector -- YouTube's own pages (search results, the
    homepage, sidebar recommendations) can have OTHER <video> elements
    on screen too (muted autoplay thumbnail previews), and
    querySelector('video') grabs whichever one appears first in the
    DOM, which is not reliably the actual player. Falls back to a bare
    'video' selector only if the specific class isn't found (e.g. a
    future YouTube redesign renames it)."""
    try:
        return page.evaluate(
            "() => { const v = document.querySelector('.html5-main-video') "
            "|| document.querySelector('video'); "
            "return v ? v.paused : null; }"
        )
    except Exception:
        return None


def _focus_player(page, timeout: int = 4000) -> bool:
    """Best-effort click to give the player keyboard focus (and, for
    play_url, satisfy Chromium's autoplay user-gesture requirement) --
    see module docstring fix #4. force=True so a stray overlapping
    element (YouTube's search box has been seen covering the player
    briefly while it mounts) can't block Playwright's own actionability
    checks, and targets #movie_player (the player container) rather
    than the raw <video> tag, which can have a tiny/misplaced computed
    box while the player is still loading. Never raises -- callers
    should proceed to their keyboard shortcut regardless of the result,
    since the shortcut can still land even without a clean focus click."""
    try:
        page.click("#movie_player", timeout=timeout, force=True)
        return True
    except Exception as e:
        print(f"[youtube_player] player focus click missed (continuing "
              f"anyway): {e}", flush=True)
        return False


def _dispatch(page, name: str, args: tuple) -> dict:
    if name == "play_url":
        (url,) = args
        page.goto(url, timeout=30000)
        print(f"[youtube_player] playing {url}", flush=True)
        # Chromium blocks unmuted autoplay until this profile/domain has
        # real engagement history -- autoplay=1 in the URL isn't enough
        # on a fresh profile. Give the player a moment to mount, then if
        # it's still paused, click it: a Playwright click is a genuine
        # synthetic user-gesture event, which autoplay policy accepts
        # (unlike a .play() call from page.evaluate(), which wouldn't
        # count and would still be blocked). See module docstring fix #2.
        page.wait_for_timeout(1500)
        if _get_paused_state(page) is True:
            if _focus_player(page):
                print("[youtube_player] autoplay was blocked -- clicked "
                      "to start playback", flush=True)
            # Even on a missed focus click, check again -- the click may
            # have landed close enough to still toggle playback via the
            # player's own click-to-play handler.
            if _get_paused_state(page) is True:
                try:
                    page.keyboard.press("k")
                except Exception:
                    pass
        return {"ok": True, "url": page.url}

    if name == "next_track":
        _focus_player(page)  # best-effort; see _focus_player docstring
        page.keyboard.press("Shift+N")
        print("[youtube_player] next track", flush=True)
        return {"ok": True}

    if name == "previous_track":
        _focus_player(page)  # best-effort; see _focus_player docstring
        page.keyboard.press("Shift+P")
        print("[youtube_player] previous track", flush=True)
        return {"ok": True}

    if name == "pause":
        paused = _get_paused_state(page)
        if paused is True:
            return {"ok": True, "already_paused": True}
        _focus_player(page)  # best-effort; see _focus_player docstring
        page.keyboard.press("k")
        print("[youtube_player] paused", flush=True)
        return {"ok": True, "already_paused": False}

    if name == "resume":
        paused = _get_paused_state(page)
        if paused is False:
            return {"ok": True, "already_playing": True}
        _focus_player(page)  # best-effort; see _focus_player docstring
        page.keyboard.press("k")
        print("[youtube_player] resumed", flush=True)
        return {"ok": True, "already_playing": False}

    if name == "stop":
        page.goto("about:blank")
        print("[youtube_player] stopped (navigated to about:blank)", flush=True)
        return {"ok": True}

    if name == "get_current_video_id":
        url = page.url
        m = _VIDEO_ID_RE.search(url)
        return {"ok": True, "video_id": m.group(1) if m else None, "url": url}

    return {"ok": False, "error": f"unknown player command: {name!r}"}


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def play_url(url: str) -> dict:
    """Navigate the persistent page to `url`. Replaces webbrowser.open()
    as the actual playback mechanism -- callers (youtube_playlists.py)
    keep building the URL themselves (autoplay, shuffle's watch_videos
    construction) and only hand the finished URL here."""
    return _enqueue("play_url", (url,), timeout=_PLAY_URL_TIMEOUT_S)


def next_track() -> dict:
    return _enqueue("next_track", ())


def previous_track() -> dict:
    return _enqueue("previous_track", ())


def pause() -> dict:
    return _enqueue("pause", ())


def resume() -> dict:
    return _enqueue("resume", ())


def stop() -> dict:
    """Navigates the page to about:blank. See module docstring for why
    this is distinct from pause()."""
    return _enqueue("stop", ())


def get_current_video_id() -> Optional[str]:
    """Parse `v=<id>` out of the persistent page's current URL. Used by
    the "add current song to playlist" voice intent. Returns None if
    nothing is playing (about:blank, YouTube homepage, etc.) or the
    player thread isn't up."""
    result = _enqueue("get_current_video_id", ())
    return result.get("video_id") if result.get("ok") else None
