"""
Chloe -- social media core.

Phase 1 scope (this file): secrets persistence + Bluesky auth client +
a LinkedIn draft-export stub. No posting, no engagement, no DB. Those
arrive in later phases -- see SOCIAL_MEDIA_PLAN.md.

Design notes
------------
- Flat module to match the rest of the jarvis/ layout (brain.py,
  lights.py, wallet.py, etc). Subsequent phases will add sibling
  modules: social_db.py, social_composer.py, etc.
- Secrets live in C:\Chloe\secrets\social.json, mirroring the
  wallet.py and lights.py pattern. Path is overridable via the
  CHLOE_SOCIAL_SECRETS_DIR env var for testing.
- Bluesky talks raw XRPC over `requests` (already pinned in
  requirements.txt).
- LinkedIn is intentionally draft-only -- see locked decisions in
  SOCIAL_MEDIA_PLAN.md. The exporter writes a markdown file Ed pastes
  by hand. Never attempt API posting on personal profiles.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# --- Paths -----------------------------------------------------------------
SECRETS_DIR = Path(os.environ.get("CHLOE_SOCIAL_SECRETS_DIR", r"C:\Chloe\secrets"))
SOCIAL_SECRETS_PATH = SECRETS_DIR / "social.json"

LINKEDIN_DRAFT_DIR = Path(
    os.environ.get(
        "CHLOE_LINKEDIN_DRAFT_DIR",
        r"C:\Chloe\secrets\linkedin_drafts",
    )
)

BSKY_HOST = "https://bsky.social"

LINKEDIN_LABEL = (
    "\U0001F916 Written by Chloe (my AI assistant), edited and approved by me.\n\n"
)


# --- Secrets I/O ------------------------------------------------------------
def _ensure_secrets_dir() -> None:
    """Create the secrets dir if missing; tighten ACLs on Windows."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        try:
            import subprocess
            user = os.environ.get("USERNAME", "")
            if user:
                subprocess.run(
                    [
                        "icacls",
                        str(SECRETS_DIR),
                        "/inheritance:r",
                        "/grant",
                        f"{user}:(OI)(CI)F",
                    ],
                    capture_output=True,
                    timeout=5,
                )
        except Exception:
            pass


def save_secrets(
    bluesky_handle: str,
    bluesky_app_password: str,
    linkedin_profile_url: Optional[str] = None,
) -> Path:
    """Write the social secrets JSON. Overwrites."""
    _ensure_secrets_dir()
    payload = {
        "bluesky": {
            "handle": bluesky_handle.strip(),
            "app_password": bluesky_app_password.strip(),
        },
        "linkedin": {
            "mode": "draft_only",
            "profile_url": (linkedin_profile_url or "").strip(),
        },
    }
    SOCIAL_SECRETS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SOCIAL_SECRETS_PATH


def load_secrets() -> dict:
    if not SOCIAL_SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Social secrets not found at {SOCIAL_SECRETS_PATH}. "
            "Run `python social_health.py init --handle ... --app-password ...` first."
        )
    return json.loads(SOCIAL_SECRETS_PATH.read_text(encoding="utf-8"))


# --- Bluesky client ---------------------------------------------------------
@dataclass
class BlueskySession:
    did: str
    handle: str
    access_jwt: str
    refresh_jwt: str
    active: Optional[bool]
    created_at: float


class BlueskyAuthError(RuntimeError):
    """createSession or createRecord failed."""


class BlueskyClient:
    """Minimal Bluesky XRPC client. create_session + get_profile + create_post."""

    def __init__(self, handle: str, app_password: str, host: str = BSKY_HOST):
        self._handle = handle
        self._app_password = app_password
        self._host = host.rstrip("/")
        self._session: Optional[BlueskySession] = None
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "Chloe/social/0.2 (+contact-edwayne)"})

    def create_session(self) -> BlueskySession:
        url = f"{self._host}/xrpc/com.atproto.server.createSession"
        try:
            r = self._http.post(
                url,
                json={"identifier": self._handle, "password": self._app_password},
                timeout=15,
            )
        except requests.RequestException as e:
            raise BlueskyAuthError(f"network error contacting {self._host}: {e}") from e

        if r.status_code != 200:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text[:500]}
            raise BlueskyAuthError(f"createSession failed: HTTP {r.status_code} {body}")

        data = r.json()
        sess = BlueskySession(
            did=data["did"],
            handle=data["handle"],
            access_jwt=data["accessJwt"],
            refresh_jwt=data["refreshJwt"],
            active=data.get("active"),
            created_at=time.time(),
        )
        self._session = sess
        return sess

    def refresh_session(self) -> BlueskySession:
        """Refresh JWT, falling back to a fresh createSession on failure."""
        if self._session is None:
            return self.create_session()
        url = f"{self._host}/xrpc/com.atproto.server.refreshSession"
        r = self._http.post(
            url,
            headers={"Authorization": f"Bearer {self._session.refresh_jwt}"},
            timeout=15,
        )
        if r.status_code != 200:
            return self.create_session()
        data = r.json()
        self._session = BlueskySession(
            did=data["did"],
            handle=data["handle"],
            access_jwt=data["accessJwt"],
            refresh_jwt=data["refreshJwt"],
            active=data.get("active"),
            created_at=time.time(),
        )
        return self._session

    def _auth_header(self) -> dict:
        if self._session is None:
            raise BlueskyAuthError("no session -- call create_session() first")
        return {"Authorization": f"Bearer {self._session.access_jwt}"}

    def get_profile(self, actor: Optional[str] = None) -> dict:
        url = f"{self._host}/xrpc/app.bsky.actor.getProfile"
        params = {"actor": actor or (self._session.handle if self._session else self._handle)}
        r = self._http.get(url, params=params, headers=self._auth_header(), timeout=15)
        r.raise_for_status()
        return r.json()

    def create_post(self, text: str) -> dict:
        """POST com.atproto.repo.createRecord with an app.bsky.feed.post.

        Returns the server's response, which contains `uri` and `cid`.
        Builds a session on demand. Re-auths once on 401.
        """
        if not text or not text.strip():
            raise ValueError("create_post: empty text")
        if len(text) > 300:
            raise ValueError(
                f"create_post: text is {len(text)} chars, Bluesky limit is 300"
            )

        if self._session is None:
            self.create_session()

        def _do() -> requests.Response:
            url = f"{self._host}/xrpc/com.atproto.repo.createRecord"
            body = {
                "repo": self._session.did,
                "collection": "app.bsky.feed.post",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": text,
                    "createdAt": _dt.datetime.utcnow()
                    .replace(microsecond=0)
                    .isoformat()
                    + "Z",
                },
            }
            return self._http.post(
                url, json=body, headers=self._auth_header(), timeout=20
            )

        r = _do()
        if r.status_code == 401:
            self.refresh_session()
            r = _do()

        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text[:500]}
            raise BlueskyAuthError(f"createRecord failed: HTTP {r.status_code} {err}")

        data = r.json()
        if "uri" not in data or "cid" not in data:
            raise BlueskyAuthError(f"createRecord unexpected response: {data}")
        return data


def bluesky_from_secrets() -> BlueskyClient:
    secrets = load_secrets()
    bsky = secrets.get("bluesky") or {}
    handle = bsky.get("handle")
    pw = bsky.get("app_password")
    if not handle or not pw:
        raise BlueskyAuthError(
            "secrets file is missing bluesky.handle or bluesky.app_password -- "
            "re-run social_health.py init"
        )
    return BlueskyClient(handle, pw)


# --- LinkedIn draft exporter ------------------------------------------------
def linkedin_draft_path(slug: str) -> Path:
    LINKEDIN_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug)[:80]
    ts = time.strftime("%Y%m%d_%H%M%S")
    return LINKEDIN_DRAFT_DIR / f"{ts}_{safe}.md"


def linkedin_export_draft(slug: str, body: str) -> Path:
    path = linkedin_draft_path(slug)
    path.write_text(LINKEDIN_LABEL + body.rstrip() + "\n", encoding="utf-8")
    return path


def banner() -> str:
    return (
        "chloe/social -- phase 2 (drafts + publish)\n"
        f"  secrets:        {SOCIAL_SECRETS_PATH}\n"
        f"  linkedin out:   {LINKEDIN_DRAFT_DIR}\n"
        f"  bluesky host:   {BSKY_HOST}\n"
    )
