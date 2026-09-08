"""rom_art.py -- best-effort box-art lookup/cache for the Arcade ROM library.

Source: libretro-thumbnails (github.com/libretro-thumbnails), the same
public, no-auth-required box-art set RetroArch/EmulationStation frontends
use -- one repo per system, each with a Named_Boxarts/ folder of
"<Title>.png" files using No-Intro/Redump-style names (region tags in
parens, e.g. "Super Mario 64 (USA).png").

Ed's uploaded ROM filenames won't match those titles exactly, so this does
a fuzzy match (stdlib difflib, no extra dependency) against a per-system
title list and only takes a hit above a confidence floor -- a messy
filename just means no art, never wrong art on a game's tile.

Everything is cached to disk under CHLOE_ROM_ART_DIR (default
C:\\Chloe\\rom_art\\):
  _index/<system>.json      -- that system's title list, refreshed every 30d
  <system>/<romstem>.png    -- the matched cover, once found
  <system>/<romstem>.nomatch -- marks "looked, found nothing confident",
                                 so a bad match isn't re-attempted every load

brain_http.py's _get_rom_art() calls get_art_path() and treats None as
"no art" -- every failure mode here (network down, no repo for this
system, no confident match) degrades to that, never an exception that
would break the ROM library page.
"""
import difflib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

# system key (matches _ROM_SYSTEMS values in brain_http.py) -> repo name
# under github.com/libretro-thumbnails/<repo>
_REPO_BY_SYSTEM = {
    "nes":    "Nintendo_-_Nintendo_Entertainment_System",
    "gb":     "Nintendo_-_Game_Boy_Color",
    "gba":    "Nintendo_-_Game_Boy_Advance",
    "snes":   "Nintendo_-_Super_Nintendo_Entertainment_System",
    "segaMD": "Sega_-_Mega_Drive_-_Genesis",
    "n64":    "Nintendo_-_Nintendo_64",
    "psx":    "Sony_-_PlayStation",
    "ps2":    "Sony_-_PlayStation_2",
}
_BRANCHES = ("master", "main")  # try in order; repos vary
_INDEX_TTL_S = 30 * 24 * 3600    # re-check a system's title list every 30 days
_MATCH_FLOOR = 0.55              # difflib ratio below this = no confident match
_HEADERS = {"User-Agent": "chloe-arcade-rom-art/1.0"}


def _art_dir() -> Path:
    d = Path(os.environ.get("CHLOE_ROM_ART_DIR", r"C:\Chloe\rom_art"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(system: str) -> Path:
    return _art_dir() / "_index" / f"{system}.json"


def _clean_name(name: str) -> str:
    """Strip a filename down to a bare title for matching: no extension, no
    (region)/[tag] annotations, underscores/dots treated as spaces."""
    s = Path(name).stem
    s = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", s)
    s = re.sub(r"[._]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fetch_json(url: str, timeout=8):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_index(system: str) -> list:
    """Titles (no extension) available in `system`'s Named_Boxarts folder,
    from a 30-day-cached local copy when there is one."""
    ip = _index_path(system)
    if ip.exists():
        try:
            cached = json.loads(ip.read_text(encoding="utf-8"))
            if time.time() - cached.get("fetched", 0) < _INDEX_TTL_S:
                return cached.get("titles", [])
        except Exception:
            pass
    repo = _REPO_BY_SYSTEM.get(system)
    titles = []
    if repo:
        for branch in _BRANCHES:
            try:
                url = (f"https://api.github.com/repos/libretro-thumbnails/{repo}"
                       f"/git/trees/{branch}?recursive=1")
                tree = _fetch_json(url)
                for entry in tree.get("tree", []):
                    path = entry.get("path", "")
                    if path.startswith("Named_Boxarts/") and path.lower().endswith(".png"):
                        titles.append(path[len("Named_Boxarts/"):-4])
                if titles:
                    break
            except Exception:
                continue
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text(json.dumps({"fetched": time.time(), "titles": titles}), encoding="utf-8")
    return titles


def _best_match(query: str, titles: list):
    if not titles:
        return None
    best, best_score = None, 0.0
    ql = query.lower()
    for t in titles:
        score = difflib.SequenceMatcher(None, ql, _clean_name(t).lower()).ratio()
        if score > best_score:
            best, best_score = t, score
    if best_score < _MATCH_FLOOR:
        return None
    return best


def get_art_path(system: str, rom_filename: str):
    """Local PNG path for this ROM's box art, fetching + caching on first
    request. None (never an exception) if there's no repo for this system,
    no confident match, or the network lookup failed."""
    if system not in _REPO_BY_SYSTEM:
        return None
    stem = Path(rom_filename).stem
    cache = _art_dir() / system / (stem + ".png")
    if cache.exists():
        return cache
    miss_marker = _art_dir() / system / (stem + ".nomatch")
    if miss_marker.exists():
        return None
    try:
        titles = _fetch_index(system)
        match = _best_match(_clean_name(rom_filename), titles)
        if not match:
            miss_marker.parent.mkdir(parents=True, exist_ok=True)
            miss_marker.write_text("", encoding="utf-8")
            return None
        repo = _REPO_BY_SYSTEM[system]
        img_url = (f"https://raw.githubusercontent.com/libretro-thumbnails/"
                   f"{repo}/master/Named_Boxarts/{quote(match)}.png")
        req = urllib.request.Request(img_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as r:
            data = r.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return cache
    except Exception:
        return None
