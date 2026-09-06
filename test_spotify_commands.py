"""test_spotify_commands.py - Regression test for Ed's Spotify
integration (2026-09-06): spotify_commands.py's intent dispatcher.

The single most important thing this covers is the collision-avoidance
gate (_spotify_is_active): jarvis.py checks Spotify's dispatcher BEFORE
youtube's (see both modules' docstrings for why), and youtube's own
pause/skip/resume/previous handling claims those bare verbs
UNCONDITIONALLY by phrase shape alone -- so if Spotify's dispatcher
claimed them just as unconditionally, it would permanently shadow
youtube's identical existing voice commands for anyone who also uses
the YouTube player, a real regression this test suite exists to catch
before Ed hits it live. Every test monkeypatches spotify_api /
spotify_player (no real Spotify account, no network, no Windows APIs)
-- same isolation approach as test_gmail_categories.py importing
email_client directly.

Run from the jarvis dir:
    python test_spotify_commands.py
Exit code 0 on success, non-zero on any failure.
"""
import spotify_api
import spotify_commands
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


# --------------------------------------------------------------------------- #
# Fixtures: swap every spotify_api / spotify_player call for a canned stub    #
# --------------------------------------------------------------------------- #

_orig = {}


def _stub(module, name, fn):
    _orig.setdefault((module, name), getattr(module, name))
    setattr(module, name, fn)


def _restore_all():
    for (module, name), fn in _orig.items():
        setattr(module, name, fn)
    _orig.clear()


def _set_now_playing(is_playing, running=True):
    # now-playing state comes from spotify_player.get_now_playing() (SMTC),
    # NOT spotify_api.get_current_playback() (the Web API) -- see both
    # modules' docstrings for why: that endpoint also returns 403
    # Premium-required on Ed's free account, live-confirmed 2026-09-06.
    _stub(spotify_player, "get_now_playing",
          lambda: {"is_playing": is_playing, "track": "T", "artists": ["A"],
                    "album": "AL"} if is_playing is not None else None)
    _stub(spotify_player, "is_spotify_running", lambda: running)


def _default_ok_control_stubs():
    _stub(spotify_player, "next_track", lambda: {"ok": True})
    _stub(spotify_player, "previous_track", lambda: {"ok": True})
    _stub(spotify_player, "play_pause", lambda: {"ok": True})
    _stub(spotify_player, "toggle_shuffle", lambda: {"ok": True})
    _stub(spotify_player, "play_uri", lambda uri: {"ok": True})


# --------------------------------------------------------------------------- #
# The critical collision-avoidance gate                                       #
# --------------------------------------------------------------------------- #

def test_bare_pause_unclaimed_when_spotify_not_playing_and_not_running():
    _set_now_playing(is_playing=False, running=False)
    _default_ok_control_stubs()
    reply = spotify_commands.try_handle_spotify_command("pause")
    check("bare 'pause' is left UNCLAIMED (None) when Spotify isn't "
          "playing and isn't even open -- so youtube's own pause "
          "handler (checked right after Spotify's in jarvis.py) still "
          "gets it, exactly as before this module existed",
          reply is None, reply)
    _restore_all()


def test_bare_pause_claimed_when_spotify_is_playing():
    _set_now_playing(is_playing=True)
    _default_ok_control_stubs()
    reply = spotify_commands.try_handle_spotify_command("pause")
    check("bare 'pause' IS claimed when Spotify is actually the thing "
          "playing right now", reply == "Paused.", reply)
    _restore_all()


def test_bare_skip_claimed_when_spotify_app_is_open_even_if_paused():
    # is_playing False, but the app IS open -- _spotify_is_active also
    # accepts "app is running" as a signal, not just "currently playing".
    _set_now_playing(is_playing=False, running=True)
    _default_ok_control_stubs()
    reply = spotify_commands.try_handle_spotify_command("skip")
    check("bare 'skip' is claimed when Spotify is open even if nothing "
          "is actively playing this instant",
          reply == "Skipped.", reply)
    _restore_all()


def test_explicit_spotify_mention_always_claims_regardless_of_state():
    _set_now_playing(is_playing=False, running=False)
    _default_ok_control_stubs()
    reply = spotify_commands.try_handle_spotify_command("pause spotify")
    check("saying \"spotify\" explicitly claims the command even when "
          "Spotify isn't detected as running -- an explicit mention "
          "should never be second-guessed",
          reply == "Paused.", reply)
    _restore_all()


def test_bare_resume_and_previous_also_gated():
    _set_now_playing(is_playing=False, running=False)
    _default_ok_control_stubs()
    for phrase in ("resume", "previous song", "go back a song"):
        reply = spotify_commands.try_handle_spotify_command(phrase)
        check(f"{phrase!r} unclaimed when Spotify inactive (falls "
              f"through to youtube, matching pre-existing behavior)",
              reply is None, reply)
    _restore_all()


# --------------------------------------------------------------------------- #
# Playlist create/delete                                                     #
# --------------------------------------------------------------------------- #

def test_create_playlist_phrasings():
    _stub(spotify_api, "create_playlist",
          lambda name, **kw: {"id": "x", "name": name, "uri": "spotify:playlist:x"})
    for phrase, expected_name in (
        ("create a playlist called road trip", "road trip"),
        ("make a new playlist named workout", "workout"),
        ("create playlist for chill vibes", "chill vibes"),
    ):
        reply = spotify_commands.try_handle_spotify_command(phrase)
        check(f"{phrase!r} creates a playlist named {expected_name!r}",
              reply == f'Created the "{expected_name}" playlist on Spotify.',
              reply)
    _restore_all()


def test_delete_playlist_phrasings():
    _stub(spotify_api, "delete_playlist", lambda name: True)
    for phrase, expected_name in (
        ("delete the road trip playlist", "road trip"),
        ("delete my workout playlist", "workout"),
    ):
        reply = spotify_commands.try_handle_spotify_command(phrase)
        check(f"{phrase!r} deletes the {expected_name!r} playlist",
              reply == f'Deleted the "{expected_name}" playlist from Spotify.',
              reply)
    _restore_all()


def test_delete_playlist_unresolved_is_honest():
    _stub(spotify_api, "delete_playlist", lambda name: False)
    reply = spotify_commands.try_handle_spotify_command("delete the xyz playlist")
    check("an unresolved delete-playlist name gets an honest can't-find "
          "reply, not a false success",
          "Couldn't find" in reply, reply)
    _restore_all()


# --------------------------------------------------------------------------- #
# Play / search-and-play on Spotify                                          #
# --------------------------------------------------------------------------- #

def test_play_on_spotify_searches_and_plays():
    _stub(spotify_api, "search", lambda q, **kw: [{
        "name": "Thunderstruck", "artists": ["AC/DC"], "uri": "spotify:track:1"}])
    _stub(spotify_player, "play_uri", lambda uri: {"ok": True})
    reply = spotify_commands.try_handle_spotify_command("play thunderstruck on spotify")
    check("'play X on spotify' searches and plays the top result",
          reply == "Playing Thunderstruck by AC/DC on Spotify.", reply)
    _restore_all()


def test_play_on_spotify_no_results_is_honest():
    _stub(spotify_api, "search", lambda q, **kw: [])
    reply = spotify_commands.try_handle_spotify_command("play qwxyz123 on spotify")
    check("no search results gets an honest can't-find reply, not a "
          "hallucinated 'Playing...' claim",
          "Couldn't find" in reply, reply)
    _restore_all()


def test_bare_play_without_spotify_mention_is_unclaimed():
    # Critical: "play <song>" with NO "spotify" anywhere must stay
    # youtube's territory -- this is the other half of the collision
    # gate (the create/delete/pause side is tested above).
    reply = spotify_commands.try_handle_spotify_command("play thunderstruck")
    check("'play X' with no 'spotify' mention is left unclaimed so "
          "youtube's existing catch-all search-and-play still gets it",
          reply is None, reply)


def test_now_playing_query():
    _stub(spotify_player, "get_now_playing", lambda: {
        "is_playing": True, "track": "Song", "artists": ["Artist"], "album": "Album"})
    reply = spotify_commands.try_handle_spotify_command("what song is this")
    check("'what song is this' reports the real current track",
          reply == '"Song" by Artist, from Album.', reply)
    _restore_all()


def test_now_playing_query_when_nothing_playing():
    _stub(spotify_player, "get_now_playing", lambda: None)
    reply = spotify_commands.try_handle_spotify_command("what's playing")
    check("now-playing query when nothing's playing is honest, not a "
          "hallucinated track name",
          reply == "Nothing's playing on Spotify right now.", reply)
    _restore_all()


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
