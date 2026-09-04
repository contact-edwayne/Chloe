"""
desktop_files.py — Live folder/file discovery under Ed's Desktop, for
voice-resolved email attachments (and anything else that wants "find a
file Ed's talking about" with no config file involved).

Ed, 2026-09-03: "she should be able to see the folders on my desktop and
recognize which one I'm referring to through voice command and find the
file or photo within that folder." Deliberately different from
local_media.py's named-folder registry (--add-folder into
local_media.json) -- Ed doesn't want to pre-register anything here, he
wants Chloe to look at whatever's actually sitting on the Desktop right
now. Same resolution ladder as local_media._resolve_folder / resolve_file
(and lights._resolve_targets / stocks.resolve_ticker / youtube_playlists.
_resolve_playlist before them) -- exact match, substring, reverse-
containment tiebreak, then (for files) a token-overlap score that only
fires when unambiguous -- just applied to a LIVE directory listing
instead of a config file. Honest miss (None) on anything ambiguous or
unmatched, same contract as every other resolver in this codebase.

Desktop root: Ed's actual Desktop is OneDrive-redirected
(C:\\Users\\eleew\\OneDrive\\Desktop -- confirmed live; it's where
local_media.py's seeded "workout" folder actually lives, and the classic
C:\\Users\\eleew\\Desktop doesn't exist on this machine at all), not the
classic per-profile Desktop. Checked at runtime: OneDrive path preferred
if it exists, classic path as fallback, overridable via
CHLOE_DESKTOP_ROOT (same env-knob pattern as CHLOE_BRAIN_ROOT /
CHLOE_WIKI_ROOT elsewhere in this codebase) for a non-Ed machine or a
future redirect change. Folder/file names starting with "." are skipped
in listings -- OneDrive itself drops sync-status folders like
".tmp.driveupload" directly under Desktop, confirmed live on this
machine, which would otherwise show up as a nonsense voice-matchable
"folder".

Public API
----------
desktop_root() -> Path
list_desktop_folders() -> list[Path]
resolve_folder(phrase) -> Path | None
resolve_file(folder_path, phrase, extensions=None) -> Path | None
check_attachment_size(path) -> str | None   (None = ok to attach)

CLI:
    python desktop_files.py --list
    python desktop_files.py --list "workout"
    python desktop_files.py --resolve-folder "workout"
    python desktop_files.py --resolve-file "workout" "leg day"
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

_ONEDRIVE_DESKTOP = Path(r"C:\Users\eleew\OneDrive\Desktop")
_CLASSIC_DESKTOP = Path(r"C:\Users\eleew\Desktop")

# Gmail's real cap is 25MB; capped lower to leave headroom for MIME
# base64 overhead (~33% larger encoded than raw) plus the rest of the
# message -- a file that reads as "24.9MB, should juuust fit" is exactly
# the kind of thing that silently fails at SMTP time instead.
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

# Example extension set for "a photo" -- callers pass this explicitly
# when they want to scope resolve_file() to images; None (the default)
# matches any file, since "an attachment" isn't only photos.
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp", ".bmp"}


def desktop_root() -> Path:
    """Ed's actual Desktop folder. CHLOE_DESKTOP_ROOT env override wins
    if set; otherwise prefers the OneDrive-redirected Desktop (what's
    actually in use) if it exists, else the classic per-profile Desktop."""
    override = os.environ.get("CHLOE_DESKTOP_ROOT", "").strip()
    if override:
        return Path(override)
    if _ONEDRIVE_DESKTOP.is_dir():
        return _ONEDRIVE_DESKTOP
    return _CLASSIC_DESKTOP


def list_desktop_folders() -> list[Path]:
    """Immediate subdirectories of the desktop root -- not recursive, so
    this stays fast and predictable no matter how deep Ed's folder
    structure gets. Skips dotfile-style entries (OneDrive sync-status
    folders, see module docstring)."""
    root = desktop_root()
    try:
        if not root.is_dir():
            return []
        return sorted(
            p for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    except OSError as e:
        print(f"[desktop_files] couldn't list {root}: {e}", file=sys.stderr)
        return []


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_folder(phrase: str) -> Optional[Path]:
    """Resolve a spoken folder phrase against the LIVE list of Desktop
    subfolders -- exact normalized-name match, substring, reverse-
    containment tiebreak, else honest None (never guesses on an
    ambiguous match). Same ladder as local_media._resolve_folder,
    applied to the current directory listing instead of a config file."""
    folders = list_desktop_folders()
    phrase = (phrase or "").strip()
    if not folders or not phrase:
        return None
    q = _normalize(phrase)

    for f in folders:
        if _normalize(f.name) == q:
            return f

    matches = [f for f in folders if q in _normalize(f.name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact_reverse = [f for f in matches if _normalize(f.name) in q]
        if len(exact_reverse) == 1:
            return exact_reverse[0]
        return None
    return None


def _list_files(folder_path: Path, extensions: Optional[set]) -> list[Path]:
    try:
        if not folder_path.is_dir():
            return []
        files = (p for p in folder_path.iterdir() if p.is_file())
        if extensions:
            files = (p for p in files if p.suffix.lower() in extensions)
        return sorted(files)
    except OSError as e:
        print(f"[desktop_files] couldn't list {folder_path}: {e}", file=sys.stderr)
        return []


def resolve_file(folder_path: Path, phrase: str,
                 extensions: Optional[set] = None) -> Optional[Path]:
    """Resolve a spoken file phrase against the files actually sitting in
    `folder_path` -- one level deep, no recursion into subfolders. Same
    four-tier ladder as local_media.resolve_file: exact normalized-stem
    match, substring, reverse-containment tiebreak, then a token-overlap
    score that only fires when one file is the unambiguous best match
    (score >= 0.5 and strictly ahead of the runner-up) -- on-disk names
    often carry underscores/numbering ("beach_trip_2024.jpg") that won't
    literally contain what Ed says out loud ("the beach photo"), same
    reason local_media needed this tier. `extensions` filters by suffix
    (e.g. PHOTO_EXTS); None (the default) matches any file -- "an
    attachment" isn't only photos. Never guesses; returns None on
    anything ambiguous or unmatched."""
    files = _list_files(folder_path, extensions)
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
    scored: list[tuple] = []
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


def check_attachment_size(path: Path) -> Optional[str]:
    """Returns an honest error string if `path` is too big to email
    (over _MAX_ATTACHMENT_BYTES), else None. Callers should surface this
    directly rather than letting an oversized attachment fail later with
    a confusing SMTP error."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return f"couldn't check {path.name}'s size: {e}"
    if size > _MAX_ATTACHMENT_BYTES:
        mb = size / (1024 * 1024)
        cap_mb = _MAX_ATTACHMENT_BYTES / (1024 * 1024)
        return f"{path.name} is {mb:.1f}MB -- too big to email (cap is {cap_mb:.0f}MB)"
    return None


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _cli_list(argv: list) -> int:
    if argv:
        folder = resolve_folder(argv[0])
        if folder is None:
            print(f"No folder matching {argv[0]!r} under {desktop_root()}.")
            return 1
        files = _list_files(folder, None)
        if not files:
            print(f"No files in {folder}.")
            return 0
        print(f"Files in {folder}:")
        for f in files:
            print(f"  {f.name}")
        return 0
    print(f"Desktop root: {desktop_root()}")
    folders = list_desktop_folders()
    if not folders:
        print("No subfolders found.")
        return 0
    for f in folders:
        print(f"  {f.name}")
    return 0


def _cli_resolve_folder(argv: list) -> int:
    if not argv:
        print("usage: desktop_files.py --resolve-folder <phrase>")
        return 1
    folder = resolve_folder(argv[0])
    print(folder if folder else f"No match for {argv[0]!r}.")
    return 0 if folder else 1


def _cli_resolve_file(argv: list) -> int:
    if len(argv) < 2:
        print("usage: desktop_files.py --resolve-file <folder phrase> <file phrase>")
        return 1
    folder = resolve_folder(argv[0])
    if folder is None:
        print(f"No folder matching {argv[0]!r}.")
        return 1
    f = resolve_file(folder, argv[1])
    print(f if f else f"No file matching {argv[1]!r} in {folder}.")
    return 0 if f else 1


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        return _cli_list(argv[1:])
    if argv[0] == "--resolve-folder":
        return _cli_resolve_folder(argv[1:])
    if argv[0] == "--resolve-file":
        return _cli_resolve_file(argv[1:])
    print("unrecognized arguments; see module docstring for CLI usage")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
