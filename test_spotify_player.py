"""test_spotify_player.py - Regression test for Ed's Spotify integration
(2026-09-06): spotify_player.py's OS-level playback control.

This bridge session has no Windows machine to test the actual ctypes/
pywin32 calls against -- what IS testable here, and what this covers,
is the module's OWN safety contract: every public function must degrade
to a logged, non-raising {"ok": False, "error": ...} rather than crash a
caller's voice/chat thread, both on a non-Windows host (this container)
and when given bad input (play_uri with a non-spotify: string). Live
verification of the actual media-key/window-focus behavior on Ed's real
Windows machine is still pending -- see this module's own docstring.

spotify_player.py is safe to import directly on any platform (the
Windows-only ctypes structures are defined inside an `if _IS_WINDOWS:`
guard at module level).

Run from the jarvis dir:
    python test_spotify_player.py
Exit code 0 on success, non-zero on any failure.
"""
import spotify_player

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def test_not_windows_flag_matches_this_container():
    check("_IS_WINDOWS correctly reflects this non-Windows test "
          "container (sanity check that the rest of this file's "
          "assumptions hold)", spotify_player._IS_WINDOWS is False)


def test_play_pause_degrades_gracefully_off_windows():
    result = spotify_player.play_pause()
    check("play_pause() never raises off Windows -- returns an honest "
          "{ok: False, error: ...} instead",
          result.get("ok") is False and "Windows" in result.get("error", ""),
          result)


def test_next_and_previous_degrade_gracefully():
    for fn, label in ((spotify_player.next_track, "next_track"),
                       (spotify_player.previous_track, "previous_track")):
        result = fn()
        check(f"{label}() never raises off Windows",
              result.get("ok") is False, result)


def test_toggle_shuffle_degrades_gracefully():
    result = spotify_player.toggle_shuffle()
    check("toggle_shuffle() never raises off Windows even though it "
          "chains two operations (focus + keypress)",
          result.get("ok") is False, result)


def test_is_spotify_running_is_false_off_windows():
    check("is_spotify_running() returns False (not an exception) when "
          "there's no Windows window system to query",
          spotify_player.is_spotify_running() is False)


def test_get_now_playing_is_none_off_windows():
    # get_now_playing() (SMTC-based) is this session's replacement for
    # spotify_api.get_current_playback() -- Ed's live testing found the
    # Web API's read-only currently-playing endpoint ALSO requires
    # Premium on his account. Off Windows (this container), it must
    # return None cleanly rather than raise -- same contract as every
    # other function in this module.
    check("get_now_playing() returns None (not an exception) off Windows",
          spotify_player.get_now_playing() is None)


def test_play_uri_rejects_non_spotify_uri_before_touching_the_os():
    result = spotify_player.play_uri("https://open.spotify.com/track/abc")
    check("play_uri() rejects a bare https:// link (not a spotify: URI) "
          "with a clear error instead of trying to launch it",
          result.get("ok") is False and "spotify:" in result.get("error", ""),
          result)


def test_play_uri_rejects_empty_string():
    result = spotify_player.play_uri("")
    check("play_uri('') is an honest rejection, not a crash",
          result.get("ok") is False, result)


def test_play_uri_valid_shape_still_reported_as_windows_only_here():
    result = spotify_player.play_uri("spotify:track:abc123")
    check("play_uri() with a well-formed spotify: URI gets past input "
          "validation and fails only on the (correctly reported) "
          "missing os.startfile on this platform",
          result.get("ok") is False and "Windows" in result.get("error", ""),
          result)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
