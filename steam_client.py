"""steam_client.py — Steam Web API + local launch integration for Chloe.

Read-only Web API calls (owned games, playtime, friends/presence) plus
local game launches via the steam:// URI scheme. Unlike PCSX2/Dolphin/
RPCS3, Chloe doesn't manage a subprocess here — the installed Steam
client owns that; we just hand it a steam://rungameid/<appid> URI and
Windows' own protocol-handler registration (done by the Steam installer)
routes it there.

Config (.env), both required for every function here:
    CHLOE_STEAM_API_KEY   Free key: https://steamcommunity.com/dev/apikey
                          (the "Domain Name" field on that page can be
                          anything, e.g. "localhost" -- it's not checked).
    CHLOE_STEAM_ID64      The account's 17-digit SteamID64. Look it up at
                          https://steamid.io by pasting your profile URL.

Privacy note (see get_friends()'s docstring): friends-list and currently-
playing visibility through the public Web API follow the KEY OWNER's own
Steam privacy settings (Settings > Privacy Settings), not each friend's —
"Friends List" and "Game details" need to be Public for steam_friends to
see anything. This is a real Valve API restriction, not a bug here.

CLI smoke-test:
    python steam_client.py library
    python steam_client.py playtime "Elden Ring"
    python steam_client.py launch "Elden Ring"
    python steam_client.py friends
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

API_BASE = "https://api.steampowered.com"
_TIMEOUT = 10

_PERSONASTATE = {
    0: "Offline", 1: "Online", 2: "Busy", 3: "Away",
    4: "Snooze", 5: "Looking to trade", 6: "Looking to play",
}


def _api_key() -> str:
    k = os.environ.get("CHLOE_STEAM_API_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "No Steam API key set. Get a free one at "
            "https://steamcommunity.com/dev/apikey and put it in .env as "
            "CHLOE_STEAM_API_KEY."
        )
    return k


def _steam_id() -> str:
    sid = os.environ.get("CHLOE_STEAM_ID64", "").strip()
    if not sid or not sid.isdigit():
        raise RuntimeError(
            "No SteamID64 set. Find yours at https://steamid.io (paste your "
            "profile URL) and put it in .env as CHLOE_STEAM_ID64."
        )
    return sid


# ─── Owned games (cached — a personal library doesn't change minute to
# minute, and this call is the heaviest one: it can return 100s of games) ──
_owned_cache: dict[str, Any] = {"ts": 0.0, "games": []}
_OWNED_CACHE_TTL = 3600  # 1h


def get_owned_games(force: bool = False) -> list[dict]:
    now = time.time()
    if not force and _owned_cache["games"] and (now - _owned_cache["ts"]) < _OWNED_CACHE_TTL:
        return _owned_cache["games"]
    r = requests.get(f"{API_BASE}/IPlayerService/GetOwnedGames/v1/", params={
        "key": _api_key(),
        "steamid": _steam_id(),
        "include_appinfo": 1,
        "include_played_free_games": 1,
    }, timeout=_TIMEOUT)
    r.raise_for_status()
    games = r.json().get("response", {}).get("games", []) or []
    _owned_cache["games"] = games
    _owned_cache["ts"] = now
    return games


def get_recently_played(count: int = 10) -> list[dict]:
    r = requests.get(f"{API_BASE}/IPlayerService/GetRecentlyPlayedGames/v1/", params={
        "key": _api_key(), "steamid": _steam_id(), "count": count,
    }, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", {}).get("games", []) or []


def _resolve_game(name: str, games: list[dict] | None = None) -> dict | None:
    """Honest-miss name resolution against the owned-games list: exact ->
    substring (unambiguous only) -> token-overlap >= 0.5 (unambiguous
    only). Never silently guesses between two plausible titles — mirrors
    desktop_files.py's resolution ladder used for email attachments."""
    if games is None:
        games = get_owned_games()
    if not name or not games:
        return None
    needle = name.strip().lower()
    if not needle:
        return None

    for g in games:
        if (g.get("name") or "").strip().lower() == needle:
            return g

    substr = [g for g in games if needle in (g.get("name") or "").lower()]
    if len(substr) == 1:
        return substr[0]

    needle_tokens = set(needle.split())
    best, best_score, tie = None, 0.0, False
    for g in games:
        gtok = set((g.get("name") or "").lower().split())
        if not gtok:
            continue
        overlap = len(needle_tokens & gtok) / max(len(needle_tokens | gtok), 1)
        if overlap > best_score:
            best, best_score, tie = g, overlap, False
        elif overlap == best_score and overlap > 0 and g is not best:
            tie = True
    if best and best_score >= 0.5 and not tie:
        return best
    return None


def game_playtime(name: str) -> dict:
    games = get_owned_games()
    g = _resolve_game(name, games)
    if not g:
        return {"ok": False, "error": f"No owned game matching {name!r} found."}
    minutes = int(g.get("playtime_forever", 0) or 0)
    return {
        "ok": True,
        "name": g.get("name"),
        "hours_total": round(minutes / 60, 1),
        "hours_2weeks": round(int(g.get("playtime_2weeks", 0) or 0) / 60, 1),
    }


def library_summary() -> dict:
    games = get_owned_games()
    total = len(games)
    by_playtime = sorted(games, key=lambda g: g.get("playtime_forever", 0) or 0,
                          reverse=True)
    top = [
        {"name": g.get("name"), "hours": round((g.get("playtime_forever", 0) or 0) / 60, 1)}
        for g in by_playtime[:5] if (g.get("playtime_forever", 0) or 0) > 0
    ]
    try:
        recent = get_recently_played(count=5)
    except Exception:
        recent = []
    recent_fmt = [
        {"name": g.get("name"),
         "hours_2weeks": round((g.get("playtime_2weeks", 0) or 0) / 60, 1)}
        for g in recent
    ]
    return {
        "ok": True,
        "total_games": total,
        "top_played": top,
        "recently_played": recent_fmt,
    }


# ─── Launch ────────────────────────────────────────────────────────────────
def launch_game(name: str) -> dict:
    """Resolve `name` against the owned-games list and hand Steam a
    steam://rungameid/<appid> URI. Steam itself must already be
    installed and its protocol handler registered (true on any normal
    install) — this doesn't need to know where Steam.exe lives, unlike
    the PCSX2/Dolphin/RPCS3 launchers."""
    games = get_owned_games()
    g = _resolve_game(name, games)
    if not g:
        return {"ok": False, "error": (
            f"No owned game matching {name!r} found. Say the exact title "
            f"if it's close, or ask for the library list."
        )}
    appid = g.get("appid")
    uri = f"steam://rungameid/{appid}"
    try:
        if os.name == "nt":
            os.startfile(uri)  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", uri])
    except Exception as e:
        return {"ok": False, "error": f"Failed to launch Steam: {type(e).__name__}: {e}"}
    return {"ok": True, "name": g.get("name"), "appid": appid}


# ─── Arcade panel support (2026-09-08) ──────────────────────────────────────
def launch_appid(appid: int) -> dict:
    """Launch by an already-known appid -- for callers (the arcade panel)
    that already resolved the game and have the exact id, so this skips
    _resolve_game's name-matching ladder entirely. launch_game() above
    (name-based, for voice) delegates the actual launch mechanics to a
    near-identical block; kept separate rather than sharing code because
    the two have different error-shape needs (this one has no "no match"
    case to report)."""
    uri = f"steam://rungameid/{appid}"
    try:
        if os.name == "nt":
            os.startfile(uri)  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", uri])
    except Exception as e:
        return {"ok": False, "error": f"Failed to launch Steam: {type(e).__name__}: {e}"}
    return {"ok": True, "appid": appid}


# ─── Profile + achievements (arcade panel) ────────────────────────────────────
_profile_cache: dict[str, Any] = {"ts": 0.0, "data": None}
_PROFILE_CACHE_TTL = 3600  # 1h


def get_profile() -> dict:
    """Own profile summary (avatar, name, Steam level) for the arcade
    panel's header card."""
    now = time.time()
    if _profile_cache["data"] and (now - _profile_cache["ts"]) < _PROFILE_CACHE_TTL:
        return _profile_cache["data"]
    r = requests.get(f"{API_BASE}/ISteamUser/GetPlayerSummaries/v2/", params={
        "key": _api_key(), "steamids": _steam_id(),
    }, timeout=_TIMEOUT)
    r.raise_for_status()
    players = r.json().get("response", {}).get("players", []) or []
    if not players:
        raise RuntimeError("Steam returned no profile for this SteamID64.")
    p = players[0]
    level = None
    try:
        r2 = requests.get(f"{API_BASE}/IPlayerService/GetSteamLevel/v1/", params={
            "key": _api_key(), "steamid": _steam_id(),
        }, timeout=_TIMEOUT)
        r2.raise_for_status()
        level = r2.json().get("response", {}).get("player_level")
    except Exception:
        pass  # level is a nice-to-have; a profile without it still renders
    data = {
        "name": p.get("personaname"),
        "avatar": p.get("avatarfull") or p.get("avatarmedium") or p.get("avatar"),
        "profile_url": p.get("profileurl"),
        "level": level,
        "status": _PERSONASTATE.get(p.get("personastate", 0), "Unknown"),
    }
    _profile_cache["data"] = data
    _profile_cache["ts"] = now
    return data


_achv_cache: dict[int, dict[str, Any]] = {}
_ACHV_CACHE_TTL = 3600  # 1h


def get_achievements(appid: int) -> dict | None:
    """Achieved/total counts for one game, for the library grid's progress
    bar. Not every game exposes achievements, and Game Details privacy can
    hide them -- both cases return None rather than raising, so one
    unsupported title doesn't break the whole grid."""
    now = time.time()
    cached = _achv_cache.get(appid)
    if cached and (now - cached["ts"]) < _ACHV_CACHE_TTL:
        return cached["data"]
    data = None
    try:
        r = requests.get(f"{API_BASE}/ISteamUserStats/GetPlayerAchievements/v1/", params={
            "key": _api_key(), "steamid": _steam_id(), "appid": appid,
        }, timeout=_TIMEOUT)
        if r.status_code == 200:
            body = r.json().get("playerstats", {})
            achievements = body.get("achievements", []) or [] if body.get("success") else []
            if achievements:
                achieved = sum(1 for a in achievements if a.get("achieved"))
                data = {"achieved": achieved, "total": len(achievements)}
    except Exception:
        data = None
    _achv_cache[appid] = {"ts": now, "data": data}
    return data


# ─── Friends / presence ─────────────────────────────────────────────────────
def get_friends() -> list[dict]:
    """Friend list merged with live status + currently-playing.

    Requires the KEY OWNER's (Ed's) own Steam privacy settings —
    Settings > Privacy Settings > "Friends List" and "Game details" —
    set to Public. This is a Valve Web API restriction: GetFriendList
    and the gameextrainfo field on GetPlayerSummaries both respect the
    profile-owner's privacy, not each friend's. A 401 here almost always
    means Friends List privacy is not Public; returns [] rather than
    raising so a locked-down profile degrades to an honest empty list."""
    r = requests.get(f"{API_BASE}/ISteamUser/GetFriendList/v1/", params={
        "key": _api_key(), "steamid": _steam_id(), "relationship": "friend",
    }, timeout=_TIMEOUT)
    if r.status_code == 401:
        return []
    r.raise_for_status()
    friends = r.json().get("friendslist", {}).get("friends", []) or []
    ids = [f["steamid"] for f in friends if f.get("steamid")]
    if not ids:
        return []

    out = []
    for i in range(0, len(ids), 100):  # GetPlayerSummaries caps at 100 ids/call
        batch = ids[i:i + 100]
        r2 = requests.get(f"{API_BASE}/ISteamUser/GetPlayerSummaries/v2/", params={
            "key": _api_key(), "steamids": ",".join(batch),
        }, timeout=_TIMEOUT)
        r2.raise_for_status()
        for p in r2.json().get("response", {}).get("players", []) or []:
            out.append({
                "name":     p.get("personaname"),
                "status":   _PERSONASTATE.get(p.get("personastate", 0), "Unknown"),
                "playing":  p.get("gameextrainfo"),
                "steamid":  p.get("steamid"),
                "avatar":   p.get("avatarmedium") or p.get("avatar"),
            })
    return out


def friends_summary(friend_name: str | None = None) -> dict:
    try:
        friends = get_friends()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if not friends:
        return {"ok": True, "friends": [], "online_count": 0, "total_friends": 0, "note": (
            "No friends visible. Either the friends list is empty, or Ed's "
            "Steam privacy settings (Friends List / Game details) aren't "
            "set to Public — the public Web API can't see a private list."
        )}

    if friend_name:
        needle = friend_name.strip().lower()
        matches = [f for f in friends if needle in (f["name"] or "").lower()]
        if not matches:
            return {"ok": False, "error": f"No friend matching {friend_name!r} found."}
        if len(matches) > 1:
            exact = [f for f in matches if (f["name"] or "").lower() == needle]
            matches = exact if len(exact) == 1 else matches
            if len(matches) > 1:
                return {"ok": False, "error": (
                    "Multiple friends match " + repr(friend_name) + ": "
                    + ", ".join(f["name"] for f in matches))}
        return {"ok": True, "friend": matches[0]}

    online = [f for f in friends if f["status"] != "Offline"]
    online.sort(key=lambda f: (f["playing"] is None, f["name"] or ""))
    return {
        "ok": True,
        "online_count": len(online),
        "total_friends": len(friends),
        "online": online[:20],
    }


# ─── CLI smoke-test entry point ────────────────────────────────────────────
def _cli(argv: list[str]) -> int:
    if not argv:
        print("usage: python steam_client.py {library|playtime|launch|friends} [arg]")
        return 2
    cmd = argv[0]
    try:
        if cmd == "library":
            print(json.dumps(library_summary(), indent=2))
            return 0
        if cmd == "playtime":
            if len(argv) < 2:
                print("usage: python steam_client.py playtime <game name>")
                return 2
            print(json.dumps(game_playtime(" ".join(argv[1:])), indent=2))
            return 0
        if cmd == "launch":
            if len(argv) < 2:
                print("usage: python steam_client.py launch <game name>")
                return 2
            print(json.dumps(launch_game(" ".join(argv[1:])), indent=2))
            return 0
        if cmd == "friends":
            name = " ".join(argv[1:]) if len(argv) > 1 else None
            print(json.dumps(friends_summary(name), indent=2))
            return 0
        print(f"Unknown command: {cmd}")
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
