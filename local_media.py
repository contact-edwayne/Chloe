"""
Local video-folder playback for Chloe.

Voice-triggered playback of video files that already live on Ed's disk
(no browser, no network, no account -- unlike youtube_playlists.py this
just hands a real file path to Windows and lets the OS-associated player
open it, since there's no "skip/pause/resume" surface to control for a
one-shot local file the way there is for a persistent YouTube tab).

Ed, 2026-09-01: "I need Chloe to be able to access and play files in my
workout folder... I should be able to say, 'Chloe, play "file name" from
workout folder' and it plays the video." Built as a small config-driven
registry of named folders (mirrors stocks.py's alias file / youtube_
playlists.py's playlist file) rather than hardcoding the one path, so Ed
can add more named folders later the same way he grew the stock-alias
list -- see --add-folder below. Seeded with the one folder he gave:
{"name": "workout", "path": r"C:\\Users\\eleew\\OneDrive\\Desktop\\Workout"}.

Resolution follows the same "honest miss, never guess" contract as
lights._resolve_targets / youtube_playlists._resolve_playlist /
stocks.resolve_ticker: an ambiguous or unmatched folder or file name
returns None rather than picking something and hoping. File matching
gets one extra tier beyond the usual exact/substring/reverse-containment
ladder -- a token-overlap score -- because on-disk filenames often carry
underscores/dashes/numbering ("leg_day_v2.mp4") that won't literally
contain what Ed says out loud ("play leg day"); the overlap tier only
fires when a single file is the clear best match (score >= 0.5 and
strictly ahead of the runner-up), otherwise it still gives up honestly.

Playback uses os.startfile(path) -- Windows-only, but so is the rest of
this codebase (see C:\\Chloe\\secrets\\ paths, venv\\Scripts\\python.exe
throughout). That hands the file to whatever program Windows already
has associated with its extension (Media Player, VLC, etc.) exactly as
if Ed had double-clicked it in Explorer.

Config persisted to C:\\Chloe\\secrets\\local_media.json --
{"folders": [{"name": "...", "path": "..."}]}. Names are lowercased on
save so lookups are case-insensitive, matching every other config file
in this codebase.

Surfaces:
    - add_folder(name, path)                -> save/overwrite a mapping, returns the entry
    - list_folders()                         -> [{name, path}]
    - play_file(folder_name, file_phrase)    -> resolve + os.startfile(), result dict
    - try_handle_local_media_command(text)   -> voice-friendly reply, or None
                                                 (dispatcher contract: None = unclaimed,
                                                 falls through to normal chat)

CLI:
    python local_media.py --add-folder "workout" "C:\\Users\\eleew\\OneDrive\\Desktop\\Workout"
    python local_media.py --list
    python local_media.py --list "workout"          # list video files in that folder
    python local_media.py "play leg day from workout folder"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CONFIG_PATH = SECRETS_DIR / "local_media.json"

_SEED_FOLDERS = {
    "workout": r"C:\Users\eleew\OneDrive\Desktop\Workout",
}

_VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v", ".webm", ".mpg", ".mpeg", ".flv",
}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"folders": [{"name": n, "path": p} for n, p in _SEED_FOLDERS.items()]}
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        print(f"[local_media] config load failed: {e}", file=sys.stderr)
        return {"folders": [{"name": n, "path": p} for n, p in _SEED_FOLDERS.items()]}
    if not isinstance(cfg.get("folders"), list):
        print(f"[local_media] 'folders' must be a list; resetting.", file=sys.stderr)
        cfg["folders"] = []
    have = {f.get("name") for f in cfg["folders"] if isinstance(f, dict)}
    for n, p in _SEED_FOLDERS.items():
        if n not in have:
            cfg["folders"].append({"name": n, "path": p})
    return cfg


def _save_config(cfg: dict) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def add_folder(name: str, path: str) -> dict:
    """Save (or overwrite, matched by name) a named-folder mapping.
    Returns the saved {name, path} entry."""
    name = (name or "").strip().lower()
    path = (path or "").strip()
    if not name:
        raise ValueError("folder name required")
    if not path:
        raise ValueError("folder path required")
    cfg = _load_config()
    folders = cfg.setdefault("folders", [])
    entry = {"name": name, "path": path}
    for i, f in enumerate(folders):
        if f.get("name") == name:
            folders[i] = entry
            break
    else:
        folders.append(entry)
    _save_config(cfg)
    return entry


def list_folders() -> list[dict]:
    return _load_config().get("folders", [])


def _resolve_folder(name: str) -> Optional[dict]:
    """Same resolution ladder as youtube_playlists._resolve_playlist:
    exact match, then substring, then reverse-containment tiebreak,
    else None."""
    folders = list_folders()
    n = (name or "").strip().lower()
    if not folders or not n:
        return None
    for f in folders:
        if f.get("name") == n:
            return f
    matches = [f for f in folders if n in (f.get("name") or "")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact_reverse = [f for f in matches if (f.get("name") or "") in n]
        if len(exact_reverse) == 1:
            return exact_reverse[0]
        return None
    return None


# --------------------------------------------------------------------------- #
# File listing / resolution                                                   #
# --------------------------------------------------------------------------- #

def _list_video_files(folder_path: Path) -> list[Path]:
    try:
        if not folder_path.is_dir():
            return []
        return sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
        )
    except OSError as e:
        print(f"[local_media] couldn't list {folder_path}: {e}", file=sys.stderr)
        return []


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_file(folder_path: Path, phrase: str) -> Optional[Path]:
    """Resolve a spoken file phrase against the video files actually
    sitting in `folder_path`. Ladder: exact normalized-stem match,
    substring, reverse-containment tiebreak, then a token-overlap score
    that only fires when one file is the unambiguous best match --
    see module docstring for why the extra tier exists. Never guesses;
    returns None on anything ambiguous or unmatched."""
    files = _list_video_files(folder_path)
    phrase = (phrase or "").strip()
    if not files or not phrase:
        return None
    q = _normalize(phrase)

    for f in files:
        if _normalize(f.stem) == q:
            return f

    matches = [f for f in files if q in _normalize(f.stem)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact_reverse = [f for f in matches if _normalize(f.stem) in q]
        if len(exact_reverse) == 1:
            return exact_reverse[0]
        return None

    q_tokens = set(q.split())
    scored: list[tuple[float, Path]] = []
    for f in files:
        f_tokens = set(_normalize(f.stem).split())
        if not f_tokens:
            continue
        overlap = q_tokens & f_tokens
        if not overlap:
            continue
        score = len(overlap) / len(f_tokens)
        scored.append((score, f))
    if scored:
        scored.sort(key=lambda x: -x[0])
        best_score = scored[0][0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.5 and best_score > runner_up:
            return scored[0][1]
    return None


# --------------------------------------------------------------------------- #
# Intent parser                                                               #
# --------------------------------------------------------------------------- #

_WAKE_PREFIX_RE = re.compile(r"^\s*(?:hey\s+)?chloe[,:]?\s*")
# "play <file phrase> from/in [my/the] <folder phrase> [folder]" -- the
# from/in split is what distinguishes this from a plain youtube "play
# <name>" command, so it's required, not optional (unlike the trailing
# "folder" word, which is nice-to-have voice filler). Accepts "in" as
# well as "from" (Ed, 2026-09-01 live test: said "play X in workout
# folder" and it fell through to youtube instead) -- safe to be this
# loose because try_handle_local_media_command only ever CLAIMS the
# command once the folder phrase resolves to something Ed actually
# configured (see that function's docstring), so "play some jazz in my
# car" still falls through to youtube/chat untouched.
_PLAY_FROM_RE = re.compile(
    r"^play\s+(.+?)\s+(?:from|in)\s+(?:my\s+|the\s+)?(.+?)(?:\s+folder)?$"
)
# "play from/in [my/the] <folder phrase> [folder]" -- same shape minus the
# file name (Ed, 2026-09-01 live test: "Play from the workout folder." with
# no file said -- _PLAY_FROM_RE can't match since there's nothing before
# "from", so it fell through to a nonsensical youtube search). Matched only
# as a fallback when _PLAY_FROM_RE misses; parse_intent returns file_phrase
# as None for this shape so the dispatcher can ask which file instead of
# either guessing or leaking the phrase to youtube.
_PLAY_FOLDER_ONLY_RE = re.compile(
    r"^play\s+(?:from|in)\s+(?:my\s+|the\s+)?(.+?)(?:\s+folder)?$"
)
_TRAILING_VIDEO_RE = re.compile(r"\s+video$")


def _clean_for_dispatch(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = _WAKE_PREFIX_RE.sub("", raw)
    raw = raw.rstrip(" .!?")
    return raw


def parse_intent(raw: str) -> Optional[tuple[Optional[str], str]]:
    """Extract (file phrase, folder phrase) from already-cleaned text, or
    None if this isn't a play-from-folder command shape at all. file
    phrase is None when Ed named a (real) folder but no file -- see
    _PLAY_FOLDER_ONLY_RE above -- so the dispatcher can ask which file
    rather than silently doing nothing or leaking the phrase to youtube.
    Does NOT resolve against configured folders or on-disk files -- see
    _resolve_folder / resolve_file for the lookups, same unresolved-
    target/resolve split as every other voice module in this codebase."""
    if not raw:
        return None
    m = _PLAY_FROM_RE.match(raw)
    if m:
        file_phrase = m.group(1).strip()
        folder_phrase = m.group(2).strip()
        file_phrase = _TRAILING_VIDEO_RE.sub("", file_phrase).strip()
        if not file_phrase or not folder_phrase:
            return None
        return (file_phrase, folder_phrase)
    m2 = _PLAY_FOLDER_ONLY_RE.match(raw)
    if m2:
        folder_phrase = m2.group(1).strip()
        if not folder_phrase:
            return None
        return (None, folder_phrase)
    return None


# --------------------------------------------------------------------------- #
# Playback                                                                     #
# --------------------------------------------------------------------------- #

def play_file(folder_phrase: str, file_phrase: str) -> dict:
    """Resolve folder_phrase + file_phrase against config/disk and hand
    the matched file to os.startfile(). Returns {"ok", "folder", "file"}
    on success or {"ok": False, "error", "folder"?} on failure."""
    entry = _resolve_folder(folder_phrase)
    if entry is None:
        return {"ok": False, "error": f"I don't have a folder called {folder_phrase!r} set up"}

    folder_path = Path(entry["path"])
    if not folder_path.is_dir():
        return {"ok": False, "error": f"the {entry['name']} folder doesn't exist on disk",
                "folder": entry["name"]}

    f = resolve_file(folder_path, file_phrase)
    if f is None:
        return {"ok": False,
                "error": f"nothing matching \"{file_phrase}\" in the {entry['name']} folder",
                "folder": entry["name"]}

    try:
        os.startfile(str(f))  # noqa: only exists on Windows -- this codebase is Windows-only.
    except Exception as e:
        return {"ok": False, "error": f"couldn't open {f.name}: {e}",
                "folder": entry["name"], "file": f.name}

    return {"ok": True, "folder": entry["name"], "file": f.name}


def _format_result(result: dict) -> str:
    """Voice-friendly one-line summary, mirrors youtube_playlists._format_result."""
    if not result.get("ok"):
        return f"Couldn't play that: {result.get('error', 'unknown error')}."
    return f"Playing {result['file']} from your {result['folder']} folder."


def try_handle_local_media_command(text: str) -> Optional[str]:
    """Dispatcher: returns a voice-friendly reply string if `text` is a
    recognized "play <file> from <folder> folder" command, else None.
    Mirrors try_handle_lights_command / try_handle_youtube_command's
    contract for jarvis.py dispatch (None = unclaimed, fall through to
    normal chat/LLM).

    Ed, 2026-09-01 (live bug): this is checked BEFORE
    try_handle_youtube_command in jarvis.py (youtube's "play <anything>"
    shape is a catch-all that treats any unresolved playlist name as a
    live YouTube search-and-play -- it would otherwise swallow "play
    cardio abs from workout folder" as a YouTube search before this
    module ever ran, which is exactly what happened live). Being checked
    first means this function must NOT claim every "play X from Y"
    shape -- ordinary YouTube requests like "play some jazz from the
    90s" match that shape too. So it only actually claims the command
    (returns non-None) once folder_phrase resolves to a folder Ed has
    configured; an unrecognized folder name returns None and falls
    through to youtube/chat instead of erroring on a folder we were
    never meant to own. A recognized folder with no matching file DOES
    still claim the command -- see play_file -- since at that point Ed
    clearly meant a local file and an honest "couldn't find it" beats a
    nonsensical YouTube search for the same phrase."""
    if not text:
        return None
    raw = _clean_for_dispatch(text)
    parsed = parse_intent(raw)
    if not parsed:
        return None
    file_phrase, folder_phrase = parsed
    entry = _resolve_folder(folder_phrase)
    if entry is None:
        return None
    if file_phrase is None:
        # Ed named a real folder but no file ("play from the workout
        # folder") -- ask honestly instead of guessing a file or letting
        # this fall through to a youtube search of the bare phrase.
        return (f"Which video from your {entry['name']} folder? Say "
                f"something like \"play <file name> from {entry['name']} "
                f"folder.\"")
    result = play_file(folder_phrase, file_phrase)
    return _format_result(result)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _cli_add_folder(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: local_media.py --add-folder <name> <path>", file=sys.stderr)
        return 1
    entry = add_folder(argv[0], argv[1])
    print(f"Saved folder: {entry['name']} -> {entry['path']}")
    return 0


def _cli_list(argv: list[str]) -> int:
    if argv:
        entry = _resolve_folder(argv[0])
        if entry is None:
            print(f"No folder matching {argv[0]!r}.")
            return 1
        files = _list_video_files(Path(entry["path"]))
        if not files:
            print(f"No video files found in {entry['name']} ({entry['path']}).")
            return 0
        print(f"Videos in {entry['name']} ({entry['path']}):")
        for f in files:
            print(f"  {f.name}")
        return 0
    folders = list_folders()
    if not folders:
        print("No folders configured.")
        return 0
    for f in folders:
        print(f"{f['name']}: {f['path']}")
    return 0


def _cli_command(text: str) -> int:
    reply = try_handle_local_media_command(text)
    if reply is None:
        print("(not recognized as a local-media command)")
        return 1
    print(reply)
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    if argv[0] == "--add-folder":
        return _cli_add_folder(argv[1:])
    if argv[0] == "--list":
        return _cli_list(argv[1:])
    return _cli_command(" ".join(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
