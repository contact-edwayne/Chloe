"""
YouTube playlist launcher for Chloe.

Voice-triggered playback of saved YouTube playlists. Playback itself goes
through youtube_player.py -- a persistent, non-headless, Playwright-
controlled browser tab (Ed's explicit choice, 2026-09-01: a real visible
YouTube tab, not a local media player, so skip/previous/pause/resume have
something to actually control). This module owns config, URL-building
(shuffle's watch_videos construction, autoplay), intent parsing, and
voice dispatch; youtube_player.py is the only thing that ever touches the
browser, and youtube_api.py is the only thing that ever touches Ed's real
YouTube account (OAuth, for "add this to <playlist>" -- see that module).

2026-09-01 update: play_playlist() used to call webbrowser.open() as its
whole playback mechanism (a new tab per play, no way to control it
afterward). That's been replaced with youtube_player.play_url() -- same
final URL, but navigated on ONE persistent page so skip/next/pause/resume
work. URL-building here is UNCHANGED (autoplay=1, the shuffle
watch_videos construction, the whole _get_shuffled_video_ids cache);
only the final "make it play" step moved to youtube_player.

Shuffle (Ed, 2026-09-01, revised): YouTube's undocumented `shuffle=1` URL
flag does NOT reliably shuffle a cold-navigated page load -- confirmed by
live testing (playlist opened in playback order despite the flag) and by
web research the same day (no current source describes it as a working
programmatic method; every up-to-date guide points at either manually
clicking YouTube's own Shuffle button or a third-party randomizer). So
shuffle is implemented as OUR OWN randomization instead of asking YouTube
to do it: `yt-dlp --flat-playlist` lists every video id in the saved
playlist (cached per-playlist, see _get_shuffled_video_ids), Python
shuffles that id list, and the shuffled ids are handed to YouTube's
`watch_videos?video_ids=...` endpoint, which builds an ad-hoc queue in
EXACTLY the order given. This needs yt-dlp installed (`pip install
yt-dlp`) -- a local CLI tool, no API key, no account -- to build the id
list; if it's missing or a fetch fails, playback falls back to the plain
playlist URL and the voice reply says so honestly rather than claiming
"shuffled" when it isn't.

Config persisted to C:\\Chloe\\secrets\\youtube_playlists.json --
{"playlists": [{"name": "...", "url": "...", "video_ids": [...],
"video_ids_fetched_at": "..."}]}. video_ids/video_ids_fetched_at are a
cache populated on first shuffle-play (or `--refresh`), not required for
plain (non-shuffle) playback. Names are lowercased on save so lookups are
case-insensitive.

Surfaces:
    - add_playlist(name, url)          -> save/overwrite a mapping, returns the entry
    - list_playlists()                 -> [{name, url}]
    - parse_intent(text)               -> extracted (playlist phrase, shuffle), or None
    - play_playlist(name, shuffle=)    -> resolve + youtube_player.play_url(), result dict
    - search_and_play(query)           -> yt-dlp ytsearch1 + play, result dict
    - try_handle_youtube_command(text) -> voice-friendly reply, or None (dispatcher
                                           for play/shuffle/skip/previous/pause/resume/
                                           search/add-to-playlist)

CLI:
    python youtube_playlists.py --add "workout" "https://youtube.com/playlist?list=..."
    python youtube_playlists.py --list
    python youtube_playlists.py --refresh "workout"   # force-refetch the shuffle id cache
    python youtube_playlists.py "play workout on shuffle"
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import youtube_player
import youtube_api

_VIDEO_ID_CACHE_TTL_S = 6 * 3600   # re-fetch at most every 6h
_YTDLP_TIMEOUT_S = 45
_WATCH_VIDEOS_ID_CAP = 150         # keeps the generated URL well under length limits

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CONFIG_PATH = SECRETS_DIR / "youtube_playlists.json"


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"playlists": []}
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        print(f"[youtube_playlists] config load failed: {e}", file=sys.stderr)
        return {"playlists": []}
    if not isinstance(cfg.get("playlists"), list):
        print(f"[youtube_playlists] 'playlists' must be a list; resetting.",
              file=sys.stderr)
        cfg["playlists"] = []
    return cfg


def _save_config(cfg: dict) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def add_playlist(name: str, url: str) -> dict:
    """Save (or overwrite, matched by name) a playlist mapping. Returns
    the saved {name, url} entry."""
    name = (name or "").strip().lower()
    url = (url or "").strip()
    if not name:
        raise ValueError("playlist name required")
    if not url:
        raise ValueError("playlist url required")
    cfg = _load_config()
    playlists = cfg.setdefault("playlists", [])
    entry = {"name": name, "url": url}
    for i, p in enumerate(playlists):
        if p.get("name") == name:
            playlists[i] = entry
            break
    else:
        playlists.append(entry)
    _save_config(cfg)
    return entry


def list_playlists() -> list[dict]:
    return _load_config().get("playlists", [])


def _resolve_playlist(name: str) -> Optional[dict]:
    """Resolve free-text against configured playlist names. Same
    resolution order as lights._resolve_targets: exact match first, then
    substring fallback. An ambiguous substring match (more than one
    playlist contains the phrase) is resolved a second way -- reverse
    containment, i.e. the query contains the WHOLE playlist name -- before
    giving up; that disambiguates "play my workout playlist" against a
    "workout" entry when another entry happens to be "workout warmup"."""
    playlists = list_playlists()
    n = (name or "").strip().lower()
    if not playlists or not n:
        return None
    for p in playlists:
        if p.get("name") == n:
            return p
    matches = [p for p in playlists if n in (p.get("name") or "")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact_reverse = [p for p in matches if (p.get("name") or "") in n]
        if len(exact_reverse) == 1:
            return exact_reverse[0]
        return None
    return None


# --------------------------------------------------------------------------- #
# Intent parser                                                               #
# --------------------------------------------------------------------------- #

# Recognizes "play <name>", "play <name> on youtube", "play my <name>
# playlist", "put on <name>". Deliberately permissive about the trailing
# shape (playlist / on youtube / bare) -- the leading "play"/"put on" verb
# is what actually distinguishes this from ordinary chat, so the shape
# after it is unpacked in one place below rather than as separate regex
# branches that would need to stay in sync.
_PLAY_PREFIX_RE = re.compile(r"^(?:play|put\s+on)\s+(.+)$")
_LEADING_FILLER_RE = re.compile(r"^(?:my|the)\s+")
_TRAILING_PLAYLIST_RE = re.compile(r"\s+playlist$")
_TRAILING_ON_YOUTUBE_RE = re.compile(r"\s+on\s+youtube$")
_WAKE_PREFIX_RE = re.compile(r"^\s*(?:hey\s+)?chloe[,:]?\s*")
# Matches "shuffle", "shuffled", "on shuffle", "in shuffle" as one unit so
# stripping it never leaves a dangling "on"/"in" behind in the target name.
_SHUFFLE_PHRASE_RE = re.compile(r"\b(?:on|in)?\s*shuffle(?:d)?\b")


def parse_intent(text: str) -> Optional[tuple[str, bool]]:
    """Extract (playlist name/phrase, shuffle) from `text`, or None if this
    isn't a play-a-playlist command shape. Does NOT resolve against
    configured playlists -- see _resolve_playlist / try_handle_youtube_command,
    same split as lights.py's parse_intent (unresolved target) vs
    _resolve_targets (the lookup).

    Shuffle (Ed, 2026-09-01): recognizes "play <name> on shuffle", "play
    <name> shuffled", and "shuffle play <name>" -- the shuffle phrase is
    stripped as one unit (_SHUFFLE_PHRASE_RE) BEFORE the play-prefix match
    so it can appear on either side of the verb without leaving a dangling
    "on"/"in" in the extracted name. Deliberately does NOT support bare
    "shuffle <name>" (no "play"/"put on" verb) -- same conservative
    same-verb-required gating as the rest of this parser."""
    if not text:
        return None
    raw = text.strip().lower()
    raw = _WAKE_PREFIX_RE.sub("", raw)
    raw = raw.rstrip(" .!?")
    shuffle = bool(_SHUFFLE_PHRASE_RE.search(raw))
    raw = _SHUFFLE_PHRASE_RE.sub(" ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    m = _PLAY_PREFIX_RE.match(raw)
    if not m:
        return None
    name = m.group(1).strip()
    name = _LEADING_FILLER_RE.sub("", name)
    name = _TRAILING_PLAYLIST_RE.sub("", name)
    name = _TRAILING_ON_YOUTUBE_RE.sub("", name)
    name = name.strip()
    if not name:
        return None
    return (name, shuffle)


# --------------------------------------------------------------------------- #
# Player-control intents (skip / previous / pause / resume)                   #
# --------------------------------------------------------------------------- #
# All of these share one preprocessing step (_clean_for_dispatch) rather
# than each re-deriving wake-word/punctuation stripping -- parse_intent's
# own stripping (above) stays separate since it also has to handle the
# shuffle-phrase extraction, which none of these need.

_MUSIC_WORD_RE = re.compile(r"\b(?:song|track|music|playback)\b")


def _clean_for_dispatch(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = _WAKE_PREFIX_RE.sub("", raw)
    raw = raw.rstrip(" .!?")
    return raw


# Skip/next (Ed, 2026-09-01): GATED CAREFULLY -- "skip" and "next" are
# common English words in unrelated conversation ("what's next on my
# calendar", "skip the intro"). Claiming a bare "next"/"skip" too eagerly
# is worse than occasionally requiring "next song" instead, so this is
# an exact-match whitelist of known skip-song phrasings (covers Ed's
# listed examples, including bare "skip"/"next" and "play the next one",
# none of which mention song/track/music by name) PLUS a fallback that
# only fires when the text explicitly names song/track/music alongside a
# skip/next verb -- never a bare substring match on "skip"/"next" alone.
_SKIP_VERB_RE = re.compile(r"\b(?:skip|next)\b")
_SKIP_EXACT_PHRASES = {
    "skip", "next", "skip this", "skip it", "skip that",
    "next one", "next song", "next track",
    "skip this song", "skip the song", "skip this track",
    "play the next one", "play the next song", "play next",
}


def _is_skip_command(raw: str) -> bool:
    if raw in _SKIP_EXACT_PHRASES:
        return True
    return bool(_SKIP_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


# Previous (Ed, 2026-09-01): a looser gate than skip is fine here --
# every phrasing Ed listed ("previous song", "go back a song", "last
# song", "previous track") already names song/track explicitly, so a
# bare previous/back verb + a music word is sufficient; no bare-word
# whitelist needed the way skip/next required one.
_PREVIOUS_VERB_RE = re.compile(r"\b(?:previous|last)\b|\bgo\s+back\b")


def _is_previous_command(raw: str) -> bool:
    return bool(_PREVIOUS_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


# Pause: unlike "skip"/"next", "pause" isn't a word that shows up
# leading ordinary unrelated sentences directed at Chloe, so a simple
# leading-word gate covers most phrasings -- "pause", "pause it", "pause
# the music", "pause please" all start with the bare verb. Also falls
# back to a skip-style verb+music-word match (same shape as
# _is_stop_command below) for non-leading phrasings like "can you pause
# the music" -- added alongside the stop fix below after live testing
# showed the same leading-word-only gate silently missing "I'll stop the
# music" for _is_stop_command. Guarded against _PLAY_PREFIX_RE for the
# same reason stop is: "play ... pause ..." should never happen in
# practice, but better safe than shadowing a real play command.
_PAUSE_VERB_RE = re.compile(r"\bpause\b")


def _is_pause_command(raw: str) -> bool:
    if raw == "pause" or raw.startswith("pause "):
        return True
    if _PLAY_PREFIX_RE.match(raw):
        return False
    return bool(_PAUSE_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


# Resume: Ed's explicit requirement -- bare "play" must NOT trigger this
# (parse_intent already treats bare "play" with no target as unclaimed,
# and that has to stay true). Exact-phrase whitelist covers "keep
# playing"/"continue playing" (verbs that would be too generic to gate
# on alone), PLUS a verb+music-word fallback (same shape as pause/stop)
# for non-leading "resume"/"unpause" phrasings -- guarded by
# _PLAY_PREFIX_RE so it can never fire on an actual "play <target>"
# command.
_RESUME_EXACT_PHRASES = {
    "resume", "unpause", "resume it", "resume that",
    "resume the music", "resume music", "resume playback",
    "keep playing", "continue playing", "continue the music",
}
_RESUME_VERB_RE = re.compile(r"\b(?:resume|unpause)\b")


def _is_resume_command(raw: str) -> bool:
    if raw in _RESUME_EXACT_PHRASES:
        return True
    if _PLAY_PREFIX_RE.match(raw):
        return False
    return bool(_RESUME_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


# Stop: never wired to a voice intent before (Ed, 2026-09-01 -- found via
# live testing) -- "stop the music" was falling all the way through to
# the general chat/LLM, which happily replied "Stopped the music." with
# no actual player command ever firing. That's exactly the false-
# confidence failure mode this whole module is built to avoid elsewhere
# (see _format_result's shuffle_note honesty pattern) -- fixed by giving
# stop its own exact-phrase whitelist, same shape as resume's, checked
# alongside it.
#
# Round 2 (still 2026-09-01, same live session): the exact-phrase-only
# version above STILL missed "I'll stop the music" and a garbled "...
# clearly stopped the music" -- both fell through to chat, which
# hallucinated "stopped" again, twice, with nothing actually stopping.
# Added a verb+music-word fallback (mirrors pause/resume, and the
# existing skip/previous pattern) so "stop" doesn't have to be the
# first word. Guarded by _PLAY_PREFIX_RE so "play don't stop the music"
# (a real song title) plays instead of getting misread as a stop
# command.
_STOP_EXACT_PHRASES = {
    "stop", "stop it", "stop that", "stop playing",
    "stop the music", "stop music", "stop playback",
}
_STOP_VERB_RE = re.compile(r"\bstop\b")


def _is_stop_command(raw: str) -> bool:
    if raw in _STOP_EXACT_PHRASES:
        return True
    if _PLAY_PREFIX_RE.match(raw):
        return False
    return bool(_STOP_VERB_RE.search(raw) and _MUSIC_WORD_RE.search(raw))


def _format_player_result(result: dict, success_reply: str, verb: str) -> str:
    if not result.get("ok"):
        return f"Couldn't {verb}: {result.get('error', 'unknown error')}."
    return success_reply


def _format_pause_result(result: dict) -> str:
    if not result.get("ok"):
        return f"Couldn't pause: {result.get('error', 'unknown error')}."
    return "Already paused." if result.get("already_paused") else "Paused."


def _format_resume_result(result: dict) -> str:
    if not result.get("ok"):
        return f"Couldn't resume: {result.get('error', 'unknown error')}."
    return "Already playing." if result.get("already_playing") else "Resumed."


def _format_stop_result(result: dict) -> str:
    if not result.get("ok"):
        return f"Couldn't stop: {result.get('error', 'unknown error')}."
    return "Stopped."


# --------------------------------------------------------------------------- #
# Search and play (read-only yt-dlp search, no API key/auth)                  #
# --------------------------------------------------------------------------- #

_SEARCH_PATTERNS = [
    re.compile(r"^search\s+youtube\s+for\s+(.+)$"),
    re.compile(r"^search\s+for\s+(.+?)(?:\s+and\s+play\s+it)?$"),
    re.compile(r"^find\s+(.+?)\s+on\s+youtube$"),
    # Bare "search <query>" / "search <query> on youtube" -- added after
    # Ed's live test showed Whisper hearing plain "Search Michael
    # Jackson" and "Search Michael Jackson on YouTube" (no "for"), which
    # the three patterns above all miss. Kept LAST in this list (each
    # pattern is tried in order, first match wins) purely so the more
    # specific "search for"/"search youtube for" phrasings above still
    # get their own (identical) handling first; doesn't change behavior
    # either way, just documents that this one is the catch-all.
    re.compile(r"^search\s+(.+?)(?:\s+on\s+youtube)?$"),
]


def _parse_search_query(raw: str) -> Optional[str]:
    for pat in _SEARCH_PATTERNS:
        m = pat.match(raw)
        if m:
            q = m.group(1).strip()
            if q:
                return q
    return None


def search_and_play(query: str) -> dict:
    """yt-dlp `ytsearch1:<query>` for the top result's id/title/uploader,
    then plays it via youtube_player.play_url. Read-only, no API key or
    account needed -- same yt-dlp-scrapes-like-a-signed-out-browser
    pattern as the shuffle id listing above. Returns {"ok": True,
    "video_id", "title", "uploader", "url"} or {"ok": False, "error"}.

    --print field order (verified live, 2026-09-01, not guessed): each
    `--print TEMPLATE` flag emits one line, in the order the flags were
    given, per matched video -- ytsearch1 caps results to exactly one
    video, so `--print "%(id)s" --print "%(title)s" --print
    "%(uploader)s"` always emits exactly 3 lines: id, then title, then
    uploader, on success."""
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "empty search query"}
    print(f"[youtube_playlists] searching youtube for {query!r} via yt-dlp",
          flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--flat-playlist",
             "--print", "%(id)s", "--print", "%(title)s",
             "--print", "%(uploader)s", f"ytsearch1:{query}"],
            capture_output=True, text=True, timeout=_YTDLP_TIMEOUT_S,
        )
    except FileNotFoundError:
        err = "yt-dlp isn't installed -- run `pip install yt-dlp` to enable search"
        print(f"[youtube_playlists] {err}", flush=True)
        return {"ok": False, "error": err}
    except subprocess.TimeoutExpired:
        err = f"yt-dlp timed out after {_YTDLP_TIMEOUT_S}s"
        print(f"[youtube_playlists] {err}", flush=True)
        return {"ok": False, "error": err}

    if proc.returncode != 0:
        err_lines = (proc.stderr or "").strip().splitlines()
        err = f"yt-dlp search failed: {err_lines[-1] if err_lines else 'unknown error'}"
        print(f"[youtube_playlists] {err}", flush=True)
        if err_lines:
            print(f"[youtube_playlists]   stderr tail: {' | '.join(err_lines[-3:])}", flush=True)
        return {"ok": False, "error": err}

    out_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if len(out_lines) < 3:
        err = "yt-dlp returned no search results"
        print(f"[youtube_playlists] {err}", flush=True)
        return {"ok": False, "error": err}
    video_id, title, uploader = out_lines[0], out_lines[1], out_lines[2]
    print(f"[youtube_playlists] top result: {title!r} by {uploader!r} "
          f"({video_id})", flush=True)

    url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
    player_result = youtube_player.play_url(url)
    if not player_result.get("ok"):
        return {"ok": False, "error": player_result.get("error", "playback failed"),
                "video_id": video_id, "title": title, "uploader": uploader}
    return {"ok": True, "video_id": video_id, "title": title,
            "uploader": uploader, "url": url}


def _handle_search_and_play(query: str) -> str:
    result = search_and_play(query)
    if not result.get("ok"):
        return f"Couldn't find or play that: {result.get('error', 'unknown error')}."
    return f"Playing {result['title']} by {result['uploader']}."


# --------------------------------------------------------------------------- #
# Add current song to a real YouTube playlist (OAuth + Data API)              #
# --------------------------------------------------------------------------- #

_ADD_TO_PLAYLIST_RE = re.compile(
    r"^add\s+(?:this|current)(?:\s+song)?\s+to\s+(?:my\s+)?(.+?)(?:\s+playlist)?$"
)
_YOUTUBE_PLAYLIST_ID_RE = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")


def _parse_add_to_playlist(raw: str) -> Optional[str]:
    m = _ADD_TO_PLAYLIST_RE.match(raw)
    if m:
        name = m.group(1).strip()
        return name or None
    return None


def _extract_youtube_playlist_id(url: str) -> Optional[str]:
    m = _YOUTUBE_PLAYLIST_ID_RE.search(url or "")
    return m.group(1) if m else None


def _handle_add_to_playlist(playlist_phrase: str) -> str:
    """Resolves `playlist_phrase` against configured LOCAL playlist names
    (same _resolve_playlist as everything else in this file), extracts
    the real YouTube playlist id from that entry's saved URL, reads the
    currently-playing video id off the persistent player tab, and calls
    the real YouTube Data API to add it -- see youtube_api.py for the
    OAuth details. Never calls the API if nothing's playing or the
    playlist doesn't resolve; only real playlist ids ever reach
    youtube_api, never a local-only name."""
    entry = _resolve_playlist(playlist_phrase)
    if entry is None:
        return f"I don't have a playlist matching {playlist_phrase!r}."
    playlist_id = _extract_youtube_playlist_id(entry["url"])
    if not playlist_id:
        return (f"Couldn't find a YouTube playlist id in "
                f"{entry['name'].title()}'s saved URL.")
    video_id = youtube_player.get_current_video_id()
    if not video_id:
        return "Nothing's currently playing, so there's nothing to add."
    result = youtube_api.add_video_to_playlist(playlist_id, video_id)
    if not result.get("ok"):
        return f"Couldn't add that: {result.get('error', 'unknown error')}."
    return f"Added to {entry['name'].title()}."


# --------------------------------------------------------------------------- #
# Shuffle: real per-video id list, not YouTube's unreliable URL flag          #
# --------------------------------------------------------------------------- #

def _fetch_playlist_video_ids(url: str) -> tuple[list[str], Optional[str]]:
    """Shell out to yt-dlp to list every video id in `url`. --flat-playlist
    skips per-video metadata, so this is a fast listing call, not a media
    fetch/download. Returns (ids, error) -- error is None on success, and
    ids is always a list (empty on failure). No YouTube API key needed --
    yt-dlp scrapes the same way a signed-out browser would."""
    # Invoke via `sys.executable -m yt_dlp` rather than a bare "yt-dlp"
    # command: pip installs the yt-dlp.exe launcher script into that
    # interpreter's Scripts/ dir, which is very often NOT on PATH (pip
    # warns about exactly this on install) -- `python -m yt_dlp` needs
    # nothing on PATH, only that the package is importable by whichever
    # interpreter is running this file, which is guaranteed since that's
    # the same interpreter `pip install yt-dlp` was run under.
    print(f"[youtube_playlists] fetching video ids via yt-dlp: {url}", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--print", "id", url],
            capture_output=True, text=True, timeout=_YTDLP_TIMEOUT_S,
        )
    except FileNotFoundError:
        err = "yt-dlp isn't installed -- run `pip install yt-dlp` to enable real shuffle"
        print(f"[youtube_playlists] {err}", flush=True)
        return [], err
    except subprocess.TimeoutExpired:
        err = f"yt-dlp timed out after {_YTDLP_TIMEOUT_S}s"
        print(f"[youtube_playlists] {err}", flush=True)
        return [], err
    if proc.returncode != 0:
        err_lines = (proc.stderr or "").strip().splitlines()
        err = f"yt-dlp failed: {err_lines[-1] if err_lines else 'unknown error'}"
        print(f"[youtube_playlists] {err}", flush=True)
        if err_lines:
            print(f"[youtube_playlists]   stderr tail: {' | '.join(err_lines[-3:])}", flush=True)
        return [], err
    ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not ids:
        err = "yt-dlp returned no videos for this playlist"
        print(f"[youtube_playlists] {err}", flush=True)
        return [], err
    print(f"[youtube_playlists] yt-dlp found {len(ids)} video(s)", flush=True)
    return ids, None


def _store_video_ids(playlist_name: str, ids: list[str]) -> None:
    """Persist a freshly-fetched id list onto the matching config entry."""
    cfg = _load_config()
    for p in cfg.get("playlists", []):
        if p.get("name") == playlist_name:
            p["video_ids"] = ids
            p["video_ids_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            break
    _save_config(cfg)


def _get_shuffled_video_ids(entry: dict) -> tuple[list[str], Optional[str]]:
    """Return a shuffled COPY of `entry`'s video id list, refreshing the
    cache via yt-dlp if it's missing or older than _VIDEO_ID_CACHE_TTL_S.
    A stale cache is used as a fallback if a refresh attempt fails --
    stale-but-shuffled beats not-shuffled-at-all. Returns ([], error) only
    when there is no cache AND the fetch failed."""
    cached = entry.get("video_ids") or []
    fetched_at = entry.get("video_ids_fetched_at")
    fresh_enough = False
    if cached and fetched_at:
        try:
            age = time.time() - time.mktime(
                time.strptime(fetched_at, "%Y-%m-%dT%H:%M:%S"))
            fresh_enough = age < _VIDEO_ID_CACHE_TTL_S
        except ValueError:
            fresh_enough = False

    if fresh_enough:
        print(f"[youtube_playlists] shuffle: using cached id list for "
              f"{entry['name']!r} ({len(cached)} videos)", flush=True)
        ids = list(cached)
        random.shuffle(ids)
        return ids, None

    print(f"[youtube_playlists] shuffle: cache miss/stale for {entry['name']!r}, "
          f"refreshing", flush=True)
    fetched_ids, error = _fetch_playlist_video_ids(entry["url"])
    if fetched_ids:
        _store_video_ids(entry["name"], fetched_ids)
        ids = list(fetched_ids)
        random.shuffle(ids)
        return ids, None

    if cached:
        print(f"[youtube_playlists] shuffle: refresh failed ({error}); "
              f"falling back to stale cache ({len(cached)} videos)", flush=True)
        ids = list(cached)
        random.shuffle(ids)
        return ids, error   # stale cache used, but surface the refresh error

    print(f"[youtube_playlists] shuffle: no cache and refresh failed ({error}); "
          f"playing unshuffled", flush=True)
    return [], error


# --------------------------------------------------------------------------- #
# Playback                                                                     #
# --------------------------------------------------------------------------- #

def play_playlist(name: str, shuffle: bool = False) -> dict:
    """Resolve `name` against configured playlists and play it via the
    persistent player tab (youtube_player.play_url). Returns {"ok",
    "name", "url", "shuffle", "shuffle_note"?} on success or {"ok":
    False, "error"} on failure. 2026-09-01: used to call
    webbrowser.open() directly (a new tab per play); now delegates the
    actual "make it play" step to youtube_player so skip/previous/pause/
    resume have a single persistent tab to control -- see that module.

    Shuffle (Ed, 2026-09-01, revised -- see module docstring for why):
    builds our own shuffled order via _get_shuffled_video_ids and hands
    it to YouTube's watch_videos endpoint, which plays an arbitrary id
    list in exactly the order given. If that fails (yt-dlp missing, no
    cache, fetch error) we do NOT silently claim "shuffled" while
    actually playing in playlist order -- shuffle_note carries the reason
    through to _format_result so the voice reply is honest about it."""
    entry = _resolve_playlist(name)
    if entry is None:
        return {"ok": False, "error": f"no playlist matching {name!r}"}

    shuffle_note = None
    if shuffle:
        ids, err = _get_shuffled_video_ids(entry)
        if len(ids) >= 2:
            picked = ids[:_WATCH_VIDEOS_ID_CAP]
            url = "https://www.youtube.com/watch_videos?video_ids=" + ",".join(picked)
            if err:
                shuffle_note = f"used a stale cached order ({err})"
        else:
            url = entry["url"]
            shuffle_note = err or "couldn't determine a shuffled order"
    else:
        url = entry["url"]

    if "autoplay=1" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}autoplay=1"

    player_result = youtube_player.play_url(url)
    if not player_result.get("ok"):
        return {"ok": False, "name": entry["name"], "url": url,
                "shuffle": shuffle,
                "error": player_result.get("error", "playback failed")}
    result = {"ok": True, "name": entry["name"], "url": url, "shuffle": shuffle}
    if shuffle_note:
        result["shuffle_note"] = shuffle_note
    return result


def _format_result(result: dict) -> str:
    """Voice-friendly one-line summary, mirrors lights._format_result."""
    if not result.get("ok"):
        return f"Couldn't play that playlist: {result.get('error', 'unknown error')}."
    name = result["name"].title()
    if result.get("shuffle"):
        if result.get("shuffle_note"):
            return f"Playing {name} -- couldn't shuffle it, so it's playing in order."
        return f"Playing {name}, shuffled."
    return f"Playing {name}."


def try_handle_youtube_command(text: str) -> Optional[str]:
    """Dispatcher: returns a voice-friendly reply string if `text` is ANY
    recognized youtube command, else None. Mirrors
    try_handle_lights_command's contract for jarvis.py dispatch (None =
    unclaimed, fall through to normal chat/LLM).

    Checked in this order, and the order matters:
      1. skip/previous/pause/resume/stop -- checked FIRST and via exact-phrase
         whitelists specifically so a phrase like "play the next one"
         gets caught here, not misread by the play/shuffle parser below
         as an attempt to play a (nonexistent) playlist named "the next
         one".
      2. search-for / find-on-youtube
      3. add-this-to-<playlist>
      4. play/shuffle (the original, most permissive intent) -- checked
         LAST for the same reason as #1: it's a catch-all shape ("play
         <anything>") that would otherwise shadow the more specific
         intents above it.

    play/shuffle when the name DOESN'T resolve to a configured playlist
    (Ed, 2026-09-01, round 3 -- changed from the original behavior):
    rather than falling through to the normal chat/LLM path, "play
    <name>" is now treated as a live YouTube search-and-play for
    whatever `name` is -- see the fallback at the bottom of this
    function. Ed's explicit choice, made after watching the chat model
    hallucinate "Playing Michael Jackson's top tracks, enjoy!" for a
    request that never actually played anything (same false-confidence
    failure this module elsewhere goes out of its way to avoid -- see
    _is_stop_command). Accepted trade-off: "play <anything>" now always
    DOES something real (a search), including low-value filler like
    "play some music", instead of sometimes silently doing nothing.
    """
    if not text:
        return None

    raw = _clean_for_dispatch(text)

    if _is_skip_command(raw):
        return _format_player_result(youtube_player.next_track(),
                                     "Skipped.", "skip")
    if _is_previous_command(raw):
        return _format_player_result(youtube_player.previous_track(),
                                     "Playing the previous song.", "go back")
    if _is_pause_command(raw):
        return _format_pause_result(youtube_player.pause())
    if _is_resume_command(raw):
        return _format_resume_result(youtube_player.resume())
    if _is_stop_command(raw):
        return _format_stop_result(youtube_player.stop())

    query = _parse_search_query(raw)
    if query is not None:
        return _handle_search_and_play(query)

    playlist_phrase = _parse_add_to_playlist(raw)
    if playlist_phrase is not None:
        return _handle_add_to_playlist(playlist_phrase)

    parsed = parse_intent(text)
    if not parsed:
        return None
    name, shuffle = parsed
    entry = _resolve_playlist(name)
    if entry is None:
        # Not a saved playlist -- fall back to a real YouTube search for
        # `name` instead of silently doing nothing (see docstring above).
        # `shuffle` is dropped here on purpose: search-and-play always
        # plays a single top result, so there's nothing to shuffle.
        return _handle_search_and_play(name)
    result = play_playlist(entry["name"], shuffle=shuffle)
    return _format_result(result)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _cli_add(argv: list[str]) -> int:
    if len(argv) < 2:
        print('usage: python youtube_playlists.py --add "<name>" "<url>"')
        return 1
    name, url = argv[0], argv[1]
    entry = add_playlist(name, url)
    print(f"saved playlist {entry['name']!r} -> {entry['url']}")
    return 0


def _cli_refresh(argv: list[str]) -> int:
    if not argv:
        print('usage: python youtube_playlists.py --refresh "<name>"')
        return 1
    entry = _resolve_playlist(argv[0])
    if entry is None:
        print(f"no playlist matching {argv[0]!r}")
        return 1
    ids, err = _fetch_playlist_video_ids(entry["url"])
    if not ids:
        print(f"refresh failed: {err}")
        return 1
    _store_video_ids(entry["name"], ids)
    print(f"refreshed {entry['name']!r}: {len(ids)} video ids cached")
    return 0


def _cli_list() -> int:
    playlists = list_playlists()
    if not playlists:
        print("no playlists configured. run: "
              'python youtube_playlists.py --add "name" "url"')
        return 0
    print(f"config: {CONFIG_PATH}")
    print("Playlists:")
    for p in playlists:
        print(f"  {p['name']:<20} {p['url']}")
    return 0


def _cli_command(text: str) -> int:
    reply = try_handle_youtube_command(text)
    if reply is None:
        print(f"not a youtube-playlist command (or no playlist matched): {text!r}")
        return 1
    print(reply)
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "--add":     return _cli_add(argv[1:])
    if cmd == "--list":    return _cli_list()
    if cmd == "--refresh": return _cli_refresh(argv[1:])
    return _cli_command(" ".join(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
