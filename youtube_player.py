"""
Persistent browser-controlled YouTube player for Chloe.

Playback goes through actual browser automation of a real YouTube tab
(Ed's choice, 2026-09-01) -- not a local media player -- and launches
non-headless (Chromium's headless audio output is unreliable and would
risk breaking both playback and the WASAPI-loopback visualizer capture)
but STARTS MINIMIZED via an active Win32 call right after launch
(Ed's later choice, 2026-09-07, once the MUSIC panel was live: he wants
Chloe's own player to feel like where the music plays from, not a
browser window popping up on his desktop). Minimizing is what
reliably hides the window (confirmed live); what used to break
playback wasn't the minimize itself, it was Chromium's background-tab
CPU throttling reacting to it -- fixed directly via three Chromium
launch switches (--disable-backgrounding-occluded-windows,
--disable-renderer-backgrounding, --disable-background-timer-
throttling) rather than by trying to hide the window some other way
Chromium wouldn't throttle (two such attempts -- off-screen
positioning, a layered 0-alpha window -- were tried first and each
failed differently). See _hide_browser_window's own docstring for the
full history. One
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
import time
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

def is_running() -> bool:
    """True if the owner thread (and therefore the browser) is already
    up. Never starts anything -- the one safe way for a caller to check
    "is Chloe already playing something" without itself becoming the
    reason the browser launches (see youtube_hud.py's poll loop, which
    must never be the trigger that opens the browser at boot just
    because it's checking for now-playing state)."""
    return _owner_thread is not None and _owner_thread.is_alive()


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


def _hide_browser_window() -> None:
    """Best-effort, never raises: find the automation browser's actual
    OS window and minimize it via the real Win32 API. This exact
    mechanism was already confirmed LIVE to reliably hide the window
    (Ed, 2026-09-07: "progress. no browser popup." after this was first
    tried). What killed playback afterward wasn't the minimize itself --
    it was Chromium's background-tab throttling reacting to the window
    being minimized/occluded. Fixed at the actual source this time (see
    the anti-throttling launch args in _launch_page) instead of trying
    yet another way to hide the window without Chromium ever noticing --
    two of those (off-screen position, a layered 0-alpha window) were
    each tried and each still let playback die or didn't stay hidden.
    The window is found by matching _BRAVE_PROFILE_DIR in a running
    browser process's command line -- unique to Chloe's dedicated
    automation profile, so this can never touch a window that isn't the
    one this module just launched. Every failure mode here (psutil/
    pywin32 missing, no matching process yet, the enumeration itself
    erroring) just leaves the window visible -- exactly the pre-fix
    behavior, never worse."""
    try:
        import psutil
        import win32con
        import win32gui
        import win32process
    except ImportError:
        return

    profile_marker = str(_BRAVE_PROFILE_DIR)
    pids: set = set()
    deadline = time.time() + 4.0
    while time.time() < deadline and not pids:
        for proc in psutil.process_iter(("pid", "name", "cmdline")):
            try:
                name = (proc.info.get("name") or "").lower()
                if "brave" not in name and "chrome" not in name:
                    continue
                cmdline = proc.info.get("cmdline") or []
                if any(profile_marker in arg for arg in cmdline):
                    pids.add(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not pids:
            time.sleep(0.2)

    if not pids:
        print("[youtube_player] couldn't find the automation browser's "
              "own process to minimize its window (it will stay "
              "visible)", flush=True)
        return

    def _minimize_if_ours(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return True
        if pid in pids:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            except Exception:
                pass
        return True

    try:
        win32gui.EnumWindows(_minimize_if_ours, None)
    except Exception as e:
        print(f"[youtube_player] minimizing the automation browser "
              f"window failed (non-fatal, window may still be visible): "
              f"{e}", flush=True)


def _launch_page(pw):
    """Launch Brave (or fall back to bundled Chromium) against the
    dedicated persistent profile and return its page. Callable more than
    once -- used both for the initial startup launch and to recover after
    the window gets closed by hand (see _player_loop)."""
    brave_path = _find_brave_executable()
    if brave_path:
        print(f"[youtube_player] launching Brave ({brave_path}) with the "
              f"dedicated Chloe profile at {_BRAVE_PROFILE_DIR} "
              f"(non-headless, will be minimized post-launch)...",
              flush=True)
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
        # Minimized via _hide_browser_window (confirmed live to
        # actually hide the window -- two other hiding tricks tried
        # after it, off-screen position and a layered 0-alpha window,
        # each failed differently). What kills playback in a minimized/
        # occluded Chromium window is background-tab throttling, so
        # fix THAT directly instead of trying to avoid ever triggering
        # it: these three switches are Chromium's own documented escape
        # hatch for exactly this (kiosk/automation/streaming setups that
        # need a backgrounded tab to keep running at full rate).
        "args": ["--disable-backgrounding-occluded-windows",
                 "--disable-renderer-backgrounding",
                 "--disable-background-timer-throttling"],
    }
    if brave_path:
        launch_kwargs["executable_path"] = brave_path
    # launch_persistent_context IS the browser+context combined (no
    # separate browser.new_context() step) -- writes its profile to
    # user_data_dir on disk, so the same signed-in session is reused on
    # every future launch instead of starting from blank each time.
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(_BRAVE_PROFILE_DIR), **launch_kwargs)
    _hide_browser_window()
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


def _player_api_call(page, method: str) -> bool:
    """Call a method directly on YouTube's own #movie_player element
    (playVideo/pauseVideo/nextVideo/previousVideo) via page.evaluate().
    This is a direct DOM/JS method call, not synthetic input, so it
    works even when the page/window doesn't have OS focus -- unlike a
    keyboard shortcut (page.keyboard.press), which depends on the
    window actually being focused. Since the browser window is now
    deliberately positioned off-screen and never focused (see
    _launch_page), keyboard shortcuts are no longer reliable, and this
    is the primary path; callers fall back to the focus+keyboard
    approach only if this returns False. Returns True if the method
    existed and was called, False otherwise (including on any error --
    never raises)."""
    try:
        return bool(page.evaluate(
            "(m) => { const p = document.getElementById('movie_player'); "
            "if (p && typeof p[m] === 'function') { p[m](); return true; } "
            "return false; }", method))
    except Exception as e:
        print(f"[youtube_player] #movie_player.{method}() call failed: "
              f"{e}", flush=True)
        return False


def _seek_to(page, seconds: float) -> bool:
    """Seek the current video to an absolute position via #movie_player's
    own seekTo(seconds, allowSeekAhead) method -- same direct-DOM-method
    approach as _player_api_call (works regardless of window focus/
    visibility), just with an argument to pass through. allowSeekAhead=
    true so it seeks even into not-yet-buffered data rather than
    clamping to what's already loaded, matching what clicking YouTube's
    own scrub bar does. Returns True if the call was made, False on any
    failure (including no video loaded) -- never raises."""
    try:
        return bool(page.evaluate(
            "([s]) => { const p = document.getElementById('movie_player'); "
            "if (p && typeof p.seekTo === 'function') { "
            "p.seekTo(s, true); return true; } return false; }",
            [seconds]))
    except Exception as e:
        print(f"[youtube_player] seekTo({seconds}) call failed: {e}",
              flush=True)
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
        if not _player_api_call(page, "nextVideo"):
            _focus_player(page)  # best-effort; see _focus_player docstring
            page.keyboard.press("Shift+N")
        print("[youtube_player] next track", flush=True)
        return {"ok": True}

    if name == "previous_track":
        if not _player_api_call(page, "previousVideo"):
            _focus_player(page)  # best-effort; see _focus_player docstring
            page.keyboard.press("Shift+P")
        print("[youtube_player] previous track", flush=True)
        return {"ok": True}

    if name == "pause":
        paused = _get_paused_state(page)
        if paused is True:
            return {"ok": True, "already_paused": True}
        if not _player_api_call(page, "pauseVideo"):
            _focus_player(page)  # best-effort; see _focus_player docstring
            page.keyboard.press("k")
        print("[youtube_player] paused", flush=True)
        return {"ok": True, "already_paused": False}

    if name == "resume":
        paused = _get_paused_state(page)
        if paused is False:
            return {"ok": True, "already_playing": True}
        if not _player_api_call(page, "playVideo"):
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

    if name == "toggle_play_pause":
        # Atomic on the owner thread: read real .paused state and act on
        # it in one dispatch, instead of two separate _enqueue round-
        # trips (which would race against Ed clicking the tab himself
        # between them). Uses the read state to call the exact API
        # method needed (see _player_api_call) rather than a blind
        # toggle key, which is more reliable now that keyboard shortcuts
        # can't be counted on for an unfocused, off-screen window.
        paused = _get_paused_state(page)
        method = "playVideo" if paused else "pauseVideo"
        if not _player_api_call(page, method):
            _focus_player(page)  # best-effort; see _focus_player docstring
            page.keyboard.press("k")
        print(f"[youtube_player] toggled play/pause (was_paused={paused})",
              flush=True)
        return {"ok": True, "was_paused": paused}

    if name == "seek":
        (seconds,) = args
        ok = _seek_to(page, seconds)
        print(f"[youtube_player] seek to {seconds:.1f}s "
              f"({'ok' if ok else 'failed -- no video loaded?'})", flush=True)
        return {"ok": ok, "error": None if ok else "seekTo call failed"}

    if name == "get_now_playing":
        url = page.url
        m = _VIDEO_ID_RE.search(url)
        video_id = m.group(1) if m else None
        if video_id is None:
            return {"ok": True, "playing": False}
        try:
            info = page.evaluate(
                "() => {"
                "  const v = document.querySelector('.html5-main-video') "
                "|| document.querySelector('video');"
                "  const chEl = document.querySelector("
                "'ytd-video-owner-renderer #channel-name a, "
                "ytd-channel-name #text');"
                "  let title = document.title || '';"
                "  if (title.endsWith(' - YouTube')) "
                "title = title.slice(0, -10);"
                "  return {"
                "    title: title,"
                "    channel: chEl ? chEl.textContent.trim() : null,"
                "    paused: v ? v.paused : null,"
                "    current_time: v ? v.currentTime : null,"
                "    duration: v ? v.duration : null"
                "  };"
                "}"
            )
        except Exception as e:
            print(f"[youtube_player] get_now_playing DOM read failed: {e}",
                  flush=True)
            info = {}
        paused = info.get("paused")
        return {
            "ok": True, "playing": True, "video_id": video_id, "url": url,
            "title": info.get("title") or None,
            "channel": info.get("channel"),
            "is_playing": (paused is False) if paused is not None else None,
            "progress_s": info.get("current_time"),
            "duration_s": info.get("duration"),
        }

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


def seek(seconds: float) -> dict:
    """Seek the current video to an absolute position (seconds from the
    start). Used by the HUD's click-to-seek on the progress bar."""
    return _enqueue("seek", (seconds,))


def toggle_play_pause() -> dict:
    """Single toggle for a HUD play/pause button -- reads the real
    .paused state and presses YouTube's own "k" shortcut in one atomic
    dispatch (see _dispatch's "toggle_play_pause" case)."""
    return _enqueue("toggle_play_pause", ())


def get_now_playing() -> Optional[dict]:
    """Live now-playing info read straight from the persistent page's
    own DOM -- title via document.title (stable across YouTube
    redesigns, same reasoning this module already applies to playback
    control), channel via a best-effort selector (soft-fails to None,
    never breaks the rest of the payload), and is_playing/progress/
    duration from the real <video> element, the same source
    _get_paused_state already reads for pause()/resume(). No network
    call, no yt-dlp -- genuinely live state of whatever's already on
    screen, not a cached guess.

    Returns {"playing": False} (not None) when nothing's loaded --
    about:blank, the YouTube homepage, a URL with no video id -- so
    youtube_hud.py's poll loop can tell "connected but idle" apart from
    "player thread isn't even up", which returns None instead (the
    enqueue itself failed)."""
    result = _enqueue("get_now_playing", ())
    if not result.get("ok"):
        return None
    return {k: v for k, v in result.items() if k != "ok"}
