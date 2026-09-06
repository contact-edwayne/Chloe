"""test_spotify_api.py - Regression test for Ed's Spotify integration
(2026-09-06): OAuth-adjacent plumbing, playlist resolution, and the
delete_playlist "unfollow, not a real delete" contract.

spotify_api.py is safe to import directly (pure functions + a module-
level `requests` import, no I/O at import time, same as email_client.py
and google_contacts.py). Every test here monkeypatches _request (the
one function that ever touches the network) so this runs with no
Spotify account, no cached token, and no network at all -- same
approach as test_wallet_balance_timing.py's fake SDK object.

Run from the jarvis dir:
    python test_spotify_api.py
Exit code 0 on success, non-zero on any failure.
"""
import spotify_api

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


class _FakeRequest:
    """Drop-in replacement for spotify_api._request that serves canned
    responses keyed by (method, path) instead of hitting the network."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, method, path, params=None, json_body=None, _retried=False):
        self.calls.append((method, path, params, json_body))
        key = (method, path)
        if key in self.table:
            return self.table[key]
        return None, "not_found"


def _patch(monkeypatch_table):
    fake = _FakeRequest(monkeypatch_table)
    spotify_api._request = fake
    spotify_api._playlist_cache["items"] = None
    spotify_api._playlist_cache["at"] = 0.0
    spotify_api._user_id_cache = None
    return fake


_orig_request = spotify_api._request


def _restore():
    spotify_api._request = _orig_request


def test_search_returns_flat_list_across_types():
    _patch({
        ("GET", "/search"): ({
            "tracks": {"items": [
                {"id": "t1", "uri": "spotify:track:t1", "name": "Song A",
                 "artists": [{"name": "Artist A"}], "album": {"name": "Album A"}},
            ]},
        }, None),
    })
    results = spotify_api.search("song a", types=("track",))
    check("search() flattens Spotify's per-type response shape into a "
          "simple list of dicts", results == [{
              "type": "track", "id": "t1", "uri": "spotify:track:t1",
              "name": "Song A", "artists": ["Artist A"], "album": "Album A",
          }], results)
    _restore()


def test_search_empty_query_short_circuits():
    fake = _patch({})
    results = spotify_api.search("   ")
    check("search('') never makes a network call -- honest empty list",
          results == [] and fake.calls == [], (results, fake.calls))
    _restore()


def test_get_current_playback_none_when_nothing_playing():
    _patch({("GET", "/me/player/currently-playing"): ({}, None)})
    check("get_current_playback() returns None when Spotify's response "
          "has no 'item' (nothing playing)",
          spotify_api.get_current_playback() is None)
    _restore()


def test_get_current_playback_shape():
    _patch({("GET", "/me/player/currently-playing"): ({
        "is_playing": True, "progress_ms": 1000,
        "item": {
            "name": "Track X", "duration_ms": 200000,
            "artists": [{"name": "Artist X"}, {"name": "Artist Y"}],
            "album": {"name": "Album X",
                      "images": [{"url": "http://example/art.jpg"}]},
            "uri": "spotify:track:x",
        },
    }, None)})
    np = spotify_api.get_current_playback()
    check("get_current_playback() extracts track/artists/album/art/"
          "is_playing correctly", np == {
              "is_playing": True, "track": "Track X",
              "artists": ["Artist X", "Artist Y"], "album": "Album X",
              "album_art_url": "http://example/art.jpg",
              "progress_ms": 1000, "duration_ms": 200000,
              "uri": "spotify:track:x",
          }, np)
    _restore()


def test_resolve_playlist_exact_match():
    _patch({("GET", "/me"): ({"id": "me"}, None),
            ("GET", "/me/playlists?limit=50"): ({
                "items": [
                    {"id": "p1", "name": "Road Trip", "uri": "spotify:playlist:p1",
                     "tracks": {"total": 12}, "owner": {"id": "me"}},
                    {"id": "p2", "name": "Workout", "uri": "spotify:playlist:p2",
                     "tracks": {"total": 5}, "owner": {"id": "me"}},
                ], "next": None,
            }, None)})
    entry = spotify_api.resolve_playlist("road trip")
    check("resolve_playlist() matches case-insensitively",
          entry is not None and entry["id"] == "p1", entry)
    _restore()


def test_resolve_playlist_ambiguous_partial_is_honest_miss():
    _patch({("GET", "/me"): ({"id": "me"}, None),
            ("GET", "/me/playlists?limit=50"): ({
                "items": [
                    {"id": "p1", "name": "Road Trip 2024", "uri": "spotify:playlist:p1",
                     "tracks": {"total": 1}, "owner": {"id": "me"}},
                    {"id": "p2", "name": "Road Trip 2025", "uri": "spotify:playlist:p2",
                     "tracks": {"total": 1}, "owner": {"id": "me"}},
                ], "next": None,
            }, None)})
    entry = spotify_api.resolve_playlist("road trip")
    check("resolve_playlist() returns None (not a guess) when a partial "
          "match hits more than one differently-named playlist",
          entry is None, entry)
    _restore()


def test_resolve_playlist_no_match():
    _patch({("GET", "/me"): ({"id": "me"}, None),
            ("GET", "/me/playlists?limit=50"): ({"items": [], "next": None}, None)})
    check("resolve_playlist() on an empty playlist list is an honest miss",
          spotify_api.resolve_playlist("anything") is None)
    _restore()


def test_create_playlist_not_connected_returns_none():
    _patch({("GET", "/me"): (None, "not_connected")})
    check("create_playlist() returns None (not a crash) when there's no "
          "usable cached token yet",
          spotify_api.create_playlist("New Playlist") is None)
    _restore()


def test_create_playlist_success():
    _patch({("GET", "/me"): ({"id": "me"}, None),
            ("POST", "/users/me/playlists"): ({
                "id": "newid", "name": "New Playlist", "uri": "spotify:playlist:newid",
            }, None)})
    entry = spotify_api.create_playlist("New Playlist")
    check("create_playlist() returns the new playlist's id/name/uri",
          entry == {"id": "newid", "name": "New Playlist",
                    "uri": "spotify:playlist:newid"}, entry)
    _restore()


def test_delete_playlist_by_raw_id_skips_name_resolution():
    fake = _patch({("DELETE", "/playlists/1234567890123456789012/followers"): ({}, None)})
    ok = spotify_api.delete_playlist("1234567890123456789012")  # 22 alnum chars
    check("delete_playlist() recognizes a raw 22-char Spotify id and "
          "skips resolve_playlist() entirely (no /me or /me/playlists "
          "call in the log)",
          ok and all(c[1] != "/me" for c in fake.calls), (ok, fake.calls))
    _restore()


def test_delete_playlist_unresolved_name_fails_honestly():
    _patch({("GET", "/me"): ({"id": "me"}, None),
            ("GET", "/me/playlists?limit=50"): ({"items": [], "next": None}, None)})
    check("delete_playlist() on a name that doesn't resolve to any "
          "playlist returns False, not an exception",
          spotify_api.delete_playlist("nonexistent") is False)
    _restore()


def test_add_tracks_requires_both_args():
    fake = _patch({})
    check("add_tracks_to_playlist() with no uris is an honest no-op, "
          "never calls the API",
          spotify_api.add_tracks_to_playlist("pid", []) is False
          and fake.calls == [])
    _restore()


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
