"""
Real YouTube Data API v3 access for Chloe -- currently just "add the
currently-playing video to a real YouTube playlist" (Ed, 2026-09-01:
this must write to his actual YouTube account via the official API, not
a local queue -- unlike youtube_playlists.py's local name->URL config,
which never touches Google's servers).

OAuth (Ed has already created a Desktop-app OAuth client and saved it to
C:\\Chloe\\secrets\\youtube_client_secret.json):
  - First run: _get_credentials() has no cached token, so (only when
    called with interactive=True, i.e. from the --auth CLI below) it
    runs InstalledAppFlow.run_local_server() -- opens Ed's browser ONCE
    for consent -- then caches the resulting token (refresh token
    included) to C:\\Chloe\\secrets\\youtube_oauth_token.json.
  - Every call after that: the cached token is loaded and silently
    refreshed via the refresh token if the access token expired. No
    browser, no prompt, no blocking.

Critical: InstalledAppFlow.run_local_server() BLOCKS waiting for a
browser redirect. If Chloe's background voice thread ever hit that path
with no cached token, it would sit there forever waiting for a consent
flow nobody is watching -- a silent hang, not a crash or a clean error.
So add_video_to_playlist() (and anything else in this module a voice/
chat caller reaches) always calls _get_credentials(interactive=False):
with no usable cached token, that returns None immediately instead of
running the flow, and the caller fails honestly with instructions to run
the one-time auth step. Only the --auth CLI entry point below passes
interactive=True, because that's the one context where a human is
actually watching the terminal/browser.

CLI:
    python youtube_api.py --auth     # one-time interactive consent -- run this first
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CLIENT_SECRET_PATH = SECRETS_DIR / "youtube_client_secret.json"
TOKEN_PATH = SECRETS_DIR / "youtube_oauth_token.json"

# Full read/write scope (not youtube.readonly / youtube.force-ssl) --
# playlistItems.insert needs write access to Ed's playlists.
SCOPES = ["https://www.googleapis.com/auth/youtube"]


def _get_credentials(*, interactive: bool = False):
    """Load cached OAuth credentials, refreshing via the refresh token if
    expired. Returns None (never raises, never blocks) if no usable
    credentials exist and `interactive` is False -- see module docstring
    for why. Pass interactive=True ONLY from the --auth CLI entry point."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        print(f"[youtube_api] google-auth-oauthlib / google-api-python-client "
              f"not installed: {e}", flush=True)
        return None

    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except Exception as e:
            print(f"[youtube_api] cached token at {TOKEN_PATH} is unreadable "
                  f"({e}); ignoring it", flush=True)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
            print("[youtube_api] refreshed cached OAuth token", flush=True)
            return creds
        except Exception as e:
            print(f"[youtube_api] token refresh failed: {e}", flush=True)
            creds = None

    if not interactive:
        print("[youtube_api] no valid cached token, and this isn't an "
              "interactive call -- run `python youtube_api.py --auth` "
              "once, interactively, first", flush=True)
        return None

    if not CLIENT_SECRET_PATH.exists():
        print(f"[youtube_api] {CLIENT_SECRET_PATH} not found -- create an "
              f"OAuth Desktop-app client in Google Cloud Console and save "
              f"the downloaded JSON there first", flush=True)
        return None

    print("[youtube_api] opening your browser for one-time YouTube "
          "consent...", flush=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"[youtube_api] consent complete, token cached to {TOKEN_PATH} "
          f"-- future calls won't prompt again", flush=True)
    return creds


def _build_client(creds):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=creds)


def add_video_to_playlist(youtube_playlist_id: str, video_id: str) -> dict:
    """Insert `video_id` into the real YouTube playlist identified by
    `youtube_playlist_id` via playlistItems.insert. Returns {"ok": True}
    on success or {"ok": False, "error"} on failure -- including "not
    authorized yet" if no cached token exists (never runs the
    interactive consent flow itself; see module docstring)."""
    creds = _get_credentials(interactive=False)
    if creds is None:
        return {"ok": False,
                "error": "not connected to YouTube yet -- run `python "
                         "youtube_api.py --auth` once, interactively, to "
                         "connect your account"}
    try:
        youtube = _build_client(creds)
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": youtube_playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()
    except Exception as e:
        print(f"[youtube_api] add_video_to_playlist({youtube_playlist_id!r}, "
              f"{video_id!r}) failed: {e}", flush=True)
        return {"ok": False, "error": str(e)}
    print(f"[youtube_api] added video {video_id} to playlist "
          f"{youtube_playlist_id}", flush=True)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _cli_auth() -> int:
    creds = _get_credentials(interactive=True)
    if creds is None:
        print("auth failed -- see errors above.")
        return 1
    print("YouTube account connected. Chloe can now add songs to your "
          "real playlists without prompting again.")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--auth":
        return _cli_auth()
    print(f"unknown command: {argv[0]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
