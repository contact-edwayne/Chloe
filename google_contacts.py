"""
google_contacts.py -- Read-only access to Ed's real Google Contacts
(saved Contacts + Gmail-derived "Other contacts") so Chloe can resolve a
spoken name like "Madison Wayne" to an email address without Ed having
pre-registered it in email_contacts.json first.

Ed, 2026-09-06: asked for exactly this after seeing email_client.py's
existing resolve_contact() only checks a small manually-curated local
file. Two ways to get "the contacts I have in Gmail" were possible --
(a) derive a name->email map from mailbox history via the IMAP
credentials email_client.py already has, zero new setup, or (b) the
real Google Contacts list (saved Contacts, plus "Other contacts" --
people auto-collected from Gmail interactions but never explicitly
saved) via the People API. Ed chose (b).

Mirrors youtube_api.py's OAuth pattern exactly -- same libraries
(already in requirements.txt: google-auth-oauthlib, google-api-python-
client, no new pip installs needed if YouTube's already set up), same
SECRETS_DIR convention, same "never block a background voice thread on
an interactive consent flow" contract:

  - First run: _get_credentials() has no cached token, so (only when
    called with interactive=True, i.e. from the --auth CLI below) it
    runs InstalledAppFlow.run_local_server() -- opens Ed's browser ONCE
    for consent -- then caches the resulting token (refresh token
    included) to C:\\Chloe\\secrets\\google_contacts_oauth_token.json.
  - Every call after that: the cached token is loaded and silently
    refreshed via the refresh token if the access token expired. No
    browser, no prompt, no blocking.
  - Every non-CLI caller (resolve_google_contact, used from the voice/
    chat email_draft/email_reply tool path) always calls
    _get_credentials(interactive=False): with no usable cached token,
    that returns None immediately rather than hanging a live turn
    waiting for a consent flow nobody is watching. Same honest-miss
    contract as email_client.resolve_contact -- returns None, never
    guesses.

Read-only scopes only (contacts.readonly + contacts.other.readonly) --
this module never writes to Ed's Contacts, it only looks names up.

Results are cached locally (C:\\Chloe\\brain\\google_contacts_cache.json,
default 24h TTL, CHLOE_CONTACTS_CACHE_TTL_HOURS to override) so a normal
lookup is a local file read, not a live API round-trip mid-voice-turn --
consistent with this session's own latency work elsewhere. A successful
resolution is also written back into email_client's fast local
email_contacts.json (only if that name isn't already saved there) so
the very next lookup for the same person skips this module entirely.

Public API
----------
resolve_google_contact(name) -> str | None    honest miss, never guesses
list_contacts(force_refresh=False) -> list[dict]   [{"name", "email"}, ...]

CLI:
    python google_contacts.py --auth              # one-time interactive consent -- run this first
    python google_contacts.py --list               # sanity-check what got fetched
    python google_contacts.py --resolve "Madison Wayne"
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CLIENT_SECRET_PATH = SECRETS_DIR / "google_contacts_client_secret.json"
TOKEN_PATH = SECRETS_DIR / "google_contacts_oauth_token.json"

# Read-only: saved Contacts, plus the auto-collected "Other contacts"
# bucket (people Ed has emailed/been emailed by but never explicitly
# saved) -- the latter is where a lot of real, unregistered contacts
# actually live, and is exactly Gmail's own compose-autocomplete source.
SCOPES = [
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/contacts.other.readonly",
]

_CACHE_TTL_S = float(os.environ.get("CHLOE_CONTACTS_CACHE_TTL_HOURS", "24")) * 3600


def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _cache_path() -> Path:
    p = _brain_root()
    p.mkdir(parents=True, exist_ok=True)
    return p / "google_contacts_cache.json"


# --------------------------------------------------------------------------- #
# OAuth (mirrors youtube_api.py's _get_credentials exactly)                   #
# --------------------------------------------------------------------------- #

def _get_credentials(*, interactive: bool = False):
    """Load cached OAuth credentials, refreshing via the refresh token if
    expired. Returns None (never raises, never blocks) if no usable
    credentials exist and `interactive` is False. Pass interactive=True
    ONLY from the --auth CLI entry point -- see module docstring."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        print(f"[google_contacts] google-auth-oauthlib / "
              f"google-api-python-client not installed: {e}", flush=True)
        return None

    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except Exception as e:
            print(f"[google_contacts] cached token at {TOKEN_PATH} is "
                  f"unreadable ({e}); ignoring it", flush=True)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
            print("[google_contacts] refreshed cached OAuth token", flush=True)
            return creds
        except Exception as e:
            print(f"[google_contacts] token refresh failed: {e}", flush=True)
            creds = None

    if not interactive:
        print("[google_contacts] no valid cached token, and this isn't an "
              "interactive call -- run `python google_contacts.py --auth` "
              "once, interactively, first", flush=True)
        return None

    if not CLIENT_SECRET_PATH.exists():
        print(f"[google_contacts] {CLIENT_SECRET_PATH} not found -- see "
              f"GOOGLE_CONTACTS_SETUP.md to create/reuse an OAuth "
              f"Desktop-app client and save the downloaded JSON there "
              f"first", flush=True)
        return None

    print("[google_contacts] opening your browser for one-time Google "
          "Contacts consent...", flush=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"[google_contacts] consent complete, token cached to "
          f"{TOKEN_PATH} -- future calls won't prompt again", flush=True)
    return creds


# --------------------------------------------------------------------------- #
# Fetching + caching                                                          #
# --------------------------------------------------------------------------- #

def _person_to_entries(person: dict) -> list:
    """One People API `person` resource -> [{"name", "email"}] (0 or 1
    entries -- a contact with no name we could display isn't usable for
    name-based lookup, and one with no email address isn't usable for
    email_draft at all, so both are silently skipped rather than
    stored as a half-populated, unmatchable row)."""
    emails = person.get("emailAddresses") or []
    if not emails:
        return []
    primary_email = next(
        (e for e in emails if (e.get("metadata") or {}).get("primary")),
        emails[0],
    )
    address = (primary_email.get("value") or "").strip()
    if not address:
        return []

    names = person.get("names") or []
    if not names:
        return []
    primary_name = next(
        (n for n in names if (n.get("metadata") or {}).get("primary")),
        names[0],
    )
    display = (primary_name.get("displayName") or "").strip()
    if not display:
        return []
    return [{"name": display, "email": address}]


def _fetch_all_contacts(creds) -> list:
    """Every {"name", "email"} pair from both saved Contacts
    (people.connections.list) and Gmail-derived Other Contacts
    (people.otherContacts.list), paginated to completion."""
    from googleapiclient.discovery import build
    service = build("people", "v1", credentials=creds)
    out = []

    page_token = None
    while True:
        resp = service.people().connections().list(
            resourceName="people/me",
            personFields="names,emailAddresses",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for person in resp.get("connections", []) or []:
            out.extend(_person_to_entries(person))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    page_token = None
    while True:
        resp = service.otherContacts().list(
            readMask="names,emailAddresses",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for person in resp.get("otherContacts", []) or []:
            out.extend(_person_to_entries(person))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return out


def _load_cache(*, allow_stale: bool = False) -> Optional[list]:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not allow_stale and data.get("expires_at", 0) <= time.time():
        return None
    return data.get("contacts")


def _save_cache(contacts: list) -> None:
    p = _cache_path()
    tmp = p.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps({"contacts": contacts, "expires_at": time.time() + _CACHE_TTL_S}),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def list_contacts(force_refresh: bool = False) -> list:
    """[{"name", "email"}, ...] from cache, refreshing from the People
    API when the cache is missing/stale (or force_refresh=True). Never
    raises: an API failure with no usable cache returns []; an API
    failure WITH a stale cache present returns the stale data rather
    than nothing, since a slightly-out-of-date contact list is still
    more useful than treating every contact as unresolvable."""
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return cached
    creds = _get_credentials(interactive=False)
    if creds is None:
        return _load_cache(allow_stale=True) or []
    try:
        contacts = _fetch_all_contacts(creds)
    except Exception as e:
        print(f"[google_contacts] fetch failed: {type(e).__name__}: {e}",
              flush=True)
        return _load_cache(allow_stale=True) or []
    _save_cache(contacts)
    return contacts


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #

def resolve_google_contact(text: str) -> Optional[str]:
    """Resolve a spoken/typed name to an email address via Ed's real
    Google Contacts. Honest miss -- returns None on no match, on an
    ambiguous match (multiple different people/addresses), or if Chloe
    isn't connected yet (see module docstring) -- never guesses, same
    contract as email_client.resolve_contact."""
    text = (text or "").strip().lower()
    if not text:
        return None
    contacts = list_contacts()
    if not contacts:
        return None

    exact = [c for c in contacts if (c.get("name") or "").strip().lower() == text]
    exact_addrs = {c["email"] for c in exact}
    if len(exact_addrs) == 1:
        return exact[0]["email"]
    if len(exact_addrs) > 1:
        # Multiple different people share this exact name -- can't guess.
        return None

    partial = [
        c for c in contacts
        if text in (c.get("name") or "").lower()
        or (c.get("name") or "").lower() in text
    ]
    partial_addrs = {c["email"] for c in partial}
    if len(partial_addrs) == 1:
        return partial[0]["email"]
    return None


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _cli_auth() -> int:
    creds = _get_credentials(interactive=True)
    if creds is None:
        print("auth failed -- see errors above.")
        return 1
    n = len(list_contacts(force_refresh=True))
    print(f"Google Contacts connected -- {n} named contact(s) with an "
          f"email address fetched and cached. Chloe can now resolve "
          f"names like \"Madison Wayne\" without you registering them "
          f"first.")
    return 0


def _cli_list() -> int:
    contacts = list_contacts()
    if not contacts:
        print("No contacts cached -- run `python google_contacts.py "
              "--auth` first (or check for errors above).")
        return 1
    for c in sorted(contacts, key=lambda c: c["name"].lower()):
        print(f'{c["name"]}: {c["email"]}')
    print(f"\n{len(contacts)} contact(s) total.")
    return 0


def _cli_resolve(name: str) -> int:
    addr = resolve_google_contact(name)
    if addr is None:
        print(f'No unambiguous match for "{name}".')
        return 1
    print(f'"{name}" -> {addr}')
    return 0


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--auth":
        return _cli_auth()
    if argv[0] == "--list":
        return _cli_list()
    if argv[0] == "--resolve" and len(argv) > 1:
        return _cli_resolve(" ".join(argv[1:]))
    print(f"unknown command: {argv[0]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
