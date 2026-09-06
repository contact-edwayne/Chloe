"""
spotify_commands.py -- Voice/chat intent dispatcher for Ed's Spotify
integration (Ed, 2026-09-06). Mirrors youtube_playlists.py's
try_handle_youtube_command contract and style exactly: one function,
try_handle_spotify_command(text), returns a voice-friendly reply string
for any recognized Spotify command, or None to mean "unclaimed, fall
through to whatever's checked next" (jarvis.py's own dispatch chain,
same as lights/local_media/email-confirm/youtube).

Playback actually happens two different ways depending on the intent
(see spotify_api.py and spotify_player.py's own docstrings for the full
free-tier-vs-Premium reasoning):
  - play/pause/skip/previous/shuffle -- OS-level control
    (spotify_player.py): global media keys, the spotify: URI handler,
    and a focused Ctrl+S for shuffle. Works on Ed's free account.
  - search, now-playing, playlist create/delete -- the real Web API
    (spotify_api.py). Also works on free tier (none of these are
    playback-control endpoints).

Ordering in jarvis.py's dispatch chain (all three call sites): checked
BEFORE youtube, same reasoning as local_media -- youtube's own "play
<anything>" fallback treats any unresolved name as a live YouTube
search-and-play, so "play thunderstruck on spotify" would get searched
on YOUTUBE (with "on spotify" as part of the literal query) if youtube's
catch-all ran first. try_handle_spotify_command only claims text that
either explicitly says "spotify" or -- for the bare
pause/resume/skip/previous/shuffle/stop verbs youtube ALSO claims
unconditionally by phrase alone -- checks that Spotify is actually the
one playing right now first (see _spotify_is_active below), so an
ordinary "pause" while watching a YouTube video still reaches youtube's
own pause handler unchanged whenever nothing's actually playing on
Spotify.

Search-and-play fallback (mirrors youtube's own "always do something
real" choice, Ed's explicit preference there): "play <name> on spotify"
that doesn't resolve to one of Ed's saved playlists is treated as a live
track search-and-play for `name`, rather than silently doing nothing or
(worse) letting the chat LLM hallucinate a "Playing ..." reply with
nothing actually playing.

CLI:
    python spotify_commands.py "play bohemian rhapsody on spotify"
    python spotify_commands.py "pause"
    python spotify_commands.py "create a playlist called road trip"
"""

from __future__ import annotations

import re
import sys
from typing import Optional

import spotify_api
import spotify_player

_WAKE_PREFIX_RE = re.compile(r"^\s*(?:hey\s+)?chloe[,:]?\s*", re.IGNORECASE)
_MUSIC_WORD_RE = re.compile(r"\b(?:song|track|music|playback|spotify)\b")
_SPOTIFY_WORD_RE = re.compile(r"\bspotify\b")


def _clean(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = _WAKE_PREFIX_RE.sub("", raw)
    return raw.rstrip(" .!?")


def _spotify_is_active(raw: str) -> bool:
    """True if this utterance should be read as a Spotify command even
    though it doesn't literally say "spotify" -- either it names
    song/track/music generically AND Spotify is the thing actually
    playing right now, or the Spotify desktop app is at least open
    (covers "pause" said the instant before jarvis's own now-playing
    poll has caught up). Honest-miss by design: if this returns False,
    the bare pause/skip/etc. verb below is left UNCLAIMED so it falls
    through to youtube's own identical-shaped check, unchanged from
    before this module existed."""
    if _SPOTIFY_WORD_RE.search(raw):
        return True
    np = spotify_api.get_current_playback()
    if np is not None and np.get("is_playing"):
        return True
    return spotify_player.is_spotify_running()


# --------------------------------------------------------------------------- #
# Player-control verb gates (same shapes/reasoning as youtube_playlists.py's,   #
# minus the exact-phrase whitelists -- those exist there so a BARE "pause"     #
# claims unconditionally; here every bare verb is additionally gated by       #
# _spotify_is_active so it doesn't shadow youtube's own identical bare verbs) #
# --------------------------------------------------------------------------- #

_SKIP_VERB_RE = re.compile(r"\b(?:skip|next)\b")
_PREVIOUS_VERB_RE = re.compile(r"\b(?:previous|last)\b|\bgo\s+back\b")
_PAUSE_VERB_RE = re.compile(r"\bpause\b")
_RESUME_VERB_RE = re.compile(r"\b(?:resume|unpause)\b")
_SHUFFLE_VERB_RE = re.compile(r"\bshuffle\b")
_PLAY_PREFIX_RE = re.compile(
    r"^(?:play|put on|start|listen to)\s+(.+)$")


def _is_skip(raw: str) -> bool:
    exact = raw in ("skip", "next", "skip this", "skip it", "skip that",
                     "next one", "next song", "next track", "play the next one")
    return exact or bool(_SKIP_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


def _is_previous(raw: str) -> bool:
    return bool(_PREVIOUS_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


def _is_pause(raw: str) -> bool:
    if raw in ("pause", "pause it", "pause spotify", "pause the music"):
        return True
    if _PLAY_PREFIX_RE.match(raw):
        return False
    return bool(_PAUSE_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


def _is_resume(raw: str) -> bool:
    if raw in ("resume", "unpause", "resume spotify", "resume the music"):
        return True
    if _PLAY_PREFIX_RE.match(raw):
        return False
    return bool(_RESUME_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


def _is_shuffle_toggle(raw: str) -> bool:
    # Deliberately does NOT require _MUSIC_WORD_RE -- "shuffle" alone
    # isn't a word youtube_playlists.py (or anything else in this repo)
    # claims as a standalone toggle command (its own shuffle handling is
    # only ever part of a play-intent, e.g. "play X on shuffle"), so
    # there's no bare-word collision to guard against the way
    # skip/pause/resume need one.
    if not _SHUFFLE_VERB_RE.search(raw):
        return False
    # But "play <name> on shuffle" (a YouTube- or Spotify-flavored PLAY
    # command) must NOT be swallowed here as a toggle -- that's handled
    # by the play/search branch below, checked first in the dispatcher.
    return not _PLAY_PREFIX_RE.match(raw)


# --------------------------------------------------------------------------- #
# Playlist create/delete -- unambiguous, no collision risk with anything      #
# else in this repo (youtube_playlists.py's playlist registration is a CLI-   #
# only name->URL mapping, never a voice-triggered "create X" command).       #
# --------------------------------------------------------------------------- #

_CREATE_PLAYLIST_RE = re.compile(
    r"\b(?:create|make|start)\s+(?:a\s+|the\s+)?(?:new\s+)?playlist\b"
    r"\s*(?:called|named|for|titled)?\s*(.*)$")
_DELETE_PLAYLIST_RE = re.compile(
    r"\bdelete\s+(?:the\s+|my\s+)?(.+?)\s+playlist\b|"
    r"\bdelete\s+(?:the\s+|my\s+)?playlist\s+(?:called|named)?\s*(.+)$")


def _parse_create_playlist(raw: str) -> Optional[str]:
    m = _CREATE_PLAYLIST_RE.search(raw)
    if not m:
        return None
    name = m.group(1).strip(" .!?\"'")
    return name or None


def _parse_delete_playlist(raw: str) -> Optional[str]:
    m = _DELETE_PLAYLIST_RE.search(raw)
    if not m:
        return None
    name = (m.group(1) or m.group(2) or "").strip(" .!?\"'")
    return name or None


# --------------------------------------------------------------------------- #
# Play / search -- requires an explicit "spotify" mention so bare "play X"    #
# keeps going to youtube's own catch-all unchanged (see module docstring).   #
# --------------------------------------------------------------------------- #

_ON_SPOTIFY_RE = re.compile(r"\s*(?:on|in|via|through)\s+spotify\s*$|^\s*spotify\s+")
_SHUFFLE_PHRASE_RE = re.compile(r"\b(?:on|in)?\s*shuffle(?:d)?\b")
_TRAILING_PLAYLIST_RE = re.compile(r"\s+playlist\s*$")


def _parse_play_on_spotify(raw: str) -> Optional[tuple[str, bool]]:
    """("<name>", shuffle) if this is a "play X on spotify" (or "spotify
    play X") shape, else None. `name` may be "" (e.g. bare "play music
    on spotify") -- caller treats an empty name as resume-or-toggle
    rather than a search, same honest-empty handling as the rest of
    this module."""
    if not _SPOTIFY_WORD_RE.search(raw):
        return None
    m = _PLAY_PREFIX_RE.match(raw)
    if not m:
        return None
    body = m.group(1)
    if not _ON_SPOTIFY_RE.search(" " + body) and not raw.startswith("spotify"):
        return None
    shuffle = bool(_SHUFFLE_PHRASE_RE.search(body))
    body = _SHUFFLE_PHRASE_RE.sub(" ", body)
    body = _ON_SPOTIFY_RE.sub("", " " + body).strip()
    body = _TRAILING_PLAYLIST_RE.sub("", body).strip()
    if body in ("music", "something", "some music", "a song", ""):
        body = ""
    return (body, shuffle)


_NOW_PLAYING_RE = re.compile(
    r"\bwhat(?:'s| is)\s+(?:this|playing)\b|"
    r"\bwhat\s+song\s+is\s+this\b|"
    r"\bwhat\s+am\s+i\s+listening\s+to\b|"
    r"\bwhat's\s+on\s+spotify\b")


# --------------------------------------------------------------------------- #
# Handlers                                                                    #
# --------------------------------------------------------------------------- #

def _handle_create_playlist(name: str) -> str:
    entry = spotify_api.create_playlist(name)
    if entry is None:
        return (f'Couldn\'t create a playlist called "{name}" -- Spotify '
                f"isn't connected yet, or the request failed.")
    return f'Created the "{entry["name"]}" playlist on Spotify.'


def _handle_delete_playlist(name: str) -> str:
    ok = spotify_api.delete_playlist(name)
    if not ok:
        return (f'Couldn\'t find a playlist called "{name}" to delete -- '
                f"or Spotify isn't connected yet.")
    return f'Deleted the "{name}" playlist from Spotify.'


def _handle_play(name: str, shuffle: bool) -> str:
    if not name:
        # No target named -- treat as a plain resume/toggle via the
        # media key, but check current state first so the reply is
        # honest rather than guessing "Resumed" when it was already
        # playing (mirrors youtube_player's is-playing-aware pause()).
        np = spotify_api.get_current_playback()
        if np and np.get("is_playing"):
            return "Spotify's already playing."
        result = spotify_player.play_pause()
        if not result.get("ok"):
            return f"Couldn't resume Spotify: {result.get('error')}."
        return "Resumed on Spotify."

    if shuffle:
        # A saved-playlist-by-name shuffle: resolve then queue via URI.
        entry = spotify_api.resolve_playlist(name)
        if entry is not None:
            result = spotify_player.play_uri(entry["uri"])
            if not result.get("ok"):
                return f"Couldn't play {entry['name']}: {result.get('error')}."
            # Spotify has no URI-launch param for "start shuffled" --
            # toggle shuffle right after handing off playback. Honest
            # trade-off: there's a brief window where the first track
            # plays in saved order before shuffle takes effect.
            spotify_player.toggle_shuffle()
            return f"Playing {entry['name']} on Spotify, shuffled."
        # Not a saved playlist -- fall through to search below, same as
        # youtube's search-and-play fallback (shuffle is meaningless for
        # a single-track search result, so it's dropped here).

    results = spotify_api.search(name, types=("track",), limit=1)
    if not results:
        return f'Couldn\'t find "{name}" on Spotify.'
    track = results[0]
    result = spotify_player.play_uri(track["uri"])
    if not result.get("ok"):
        return f"Couldn't play {track['name']}: {result.get('error')}."
    artists = ", ".join(track["artists"])
    return f"Playing {track['name']} by {artists} on Spotify."


def _handle_now_playing() -> str:
    np = spotify_api.get_current_playback()
    if np is None:
        return "Nothing's playing on Spotify right now."
    artists = ", ".join(np["artists"])
    state = "" if np["is_playing"] else " (paused)"
    return f'"{np["track"]}" by {artists}, from {np["album"]}{state}.'


def _format_control(result: dict, verb: str, success: str) -> str:
    if not result.get("ok"):
        return f"Couldn't {verb} on Spotify: {result.get('error', 'unknown error')}."
    return success


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #

def try_handle_spotify_command(text: str) -> Optional[str]:
    """Returns a voice-friendly reply for any recognized Spotify
    command, else None (unclaimed). See module docstring for ordering
    and the free-tier-vs-Premium split between spotify_api.py and
    spotify_player.py."""
    if not text:
        return None
    raw = _clean(text)

    # Playlist CRUD first -- most specific, zero collision risk.
    created = _parse_create_playlist(raw)
    if created is not None:
        return _handle_create_playlist(created)
    deleted = _parse_delete_playlist(raw)
    if deleted is not None:
        return _handle_delete_playlist(deleted)

    if _NOW_PLAYING_RE.search(raw):
        return _handle_now_playing()

    # Explicit "play X on spotify" / "spotify play X" -- checked before
    # the bare player-control verbs so "spotify play the next one" plays
    # a literal search for "the next one" rather than being misread as
    # a skip command (mirrors youtube's own play-prefix-vs-skip
    # ordering note).
    played = _parse_play_on_spotify(raw)
    if played is not None:
        name, shuffle = played
        return _handle_play(name, shuffle)

    if _is_shuffle_toggle(raw) and spotify_player.is_spotify_running():
        return _format_control(spotify_player.toggle_shuffle(),
                                "toggle shuffle", "Shuffle toggled.")

    # Bare pause/resume/skip/previous -- gated behind _spotify_is_active
    # so these never shadow youtube's identical bare-verb handling when
    # nothing's actually playing on Spotify (see module docstring).
    if _is_skip(raw) or _is_previous(raw) or _is_pause(raw) or _is_resume(raw):
        if not _spotify_is_active(raw):
            return None
        if _is_skip(raw):
            return _format_control(spotify_player.next_track(), "skip", "Skipped.")
        if _is_previous(raw):
            return _format_control(spotify_player.previous_track(),
                                    "go back", "Playing the previous song.")
        if _is_pause(raw):
            return _format_control(spotify_player.play_pause(), "pause", "Paused.")
        if _is_resume(raw):
            return _format_control(spotify_player.play_pause(), "resume", "Resumed.")

    return None


# --------------------------------------------------------------------------- #
# CLI                                                                        #
# --------------------------------------------------------------------------- #

def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 0
    reply = try_handle_spotify_command(" ".join(argv))
    print(reply if reply is not None else "(unclaimed -- not a recognized Spotify command)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
