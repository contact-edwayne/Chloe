"""test_contacts_ttl.py - Regression test for F-09 (audit action item,
2026-09-06 Round 7): Google-Contacts-sourced entries in
email_contacts.json now carry a cached_at timestamp and expire after
_ALIAS_CACHE_TTL_S (24h default) so a stale local alias can't keep
winning forever if the real address changes upstream in Google
Contacts. Manually-added contacts (--add-contact, or any call with the
default source="manual") never expire. refresh_contact() is the
explicit escape hatch (--refresh-contact "Name") for forcing a fresh
Google lookup without waiting out the window.

email_client.py is safe to import directly (see
test_email_encoding.py's docstring). This test monkeypatches
CONTACTS_PATH/SECRETS_DIR to a temp file so it never touches Ed's real
C:\\Chloe\\secrets\\email_contacts.json, and stubs the google_contacts
module (via sys.modules) so no real Google API call happens.

Run from the jarvis dir:
    python test_contacts_ttl.py
Exit code 0 on success, non-zero on any failure.
"""
import json
import sys
import tempfile
import time
import types
from pathlib import Path

import email_client

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


def _isolated_contacts(fn):
    """Run `fn()` with email_client's CONTACTS_PATH/SECRETS_DIR pointed
    at a fresh temp directory, then restore the originals. Never touches
    the real C:\\Chloe\\secrets\\email_contacts.json."""
    orig_path = email_client.CONTACTS_PATH
    orig_dir = email_client.SECRETS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        email_client.SECRETS_DIR = tmp_dir
        email_client.CONTACTS_PATH = tmp_dir / "email_contacts.json"
        try:
            fn()
        finally:
            email_client.SECRETS_DIR = orig_dir
            email_client.CONTACTS_PATH = orig_path


def _write_raw_contacts(entries: dict):
    data = {"contacts": entries}
    email_client.CONTACTS_PATH.write_text(json.dumps(data), encoding="utf-8")


def test_manual_contact_never_expires():
    def run():
        email_client.add_contact("ed", "ed@example.com", source="manual")
        # Backdate as if it were somehow ancient -- shouldn't matter for
        # manual entries since cached_at is always None for them, but
        # write it directly to be sure the TTL check can't touch it.
        raw = email_client._load_contacts_raw()
        check("a manually-added contact has cached_at=None (never "
              "expires, regardless of how old the process gets)",
              raw["ed"]["cached_at"] is None, raw["ed"])
        contacts = email_client._load_contacts()
        check("a manually-added contact still resolves normally",
              contacts.get("ed") == "ed@example.com", contacts)
    _isolated_contacts(run)


def test_fresh_google_contact_resolves():
    def run():
        email_client.add_contact("madison", "madison@example.com",
                                  source="google_contacts")
        contacts = email_client._load_contacts()
        check("a freshly-cached google_contacts entry (cached_at=now) "
              "still resolves -- well within the 24h TTL",
              contacts.get("madison") == "madison@example.com", contacts)
    _isolated_contacts(run)


def test_stale_google_contact_expires():
    def run():
        # Write a google_contacts entry cached 25 hours ago -- past the
        # default 24h _ALIAS_CACHE_TTL_S window.
        stale_ts = time.time() - (25 * 3600)
        _write_raw_contacts({
            "madison": {"address": "old-madison@example.com",
                        "source": "google_contacts",
                        "cached_at": stale_ts},
        })
        contacts = email_client._load_contacts()
        check("a google_contacts entry older than the TTL is dropped -- "
              "the F-09 fix -- so a stale cached address can't keep "
              "winning silently forever",
              "madison" not in contacts, contacts)


    _isolated_contacts(run)


def test_legacy_flat_string_entry_never_expires():
    def run():
        # Simulate a contacts file written before this metadata existed
        # -- a bare {name: "address"} string, no source/cached_at at all.
        email_client.CONTACTS_PATH.write_text(
            json.dumps({"contacts": {"grandpa": "grandpa@example.com"}}),
            encoding="utf-8")
        raw = email_client._load_contacts_raw()
        check("a legacy flat-string entry normalizes to "
              "source='manual', cached_at=None",
              raw["grandpa"] == {"address": "grandpa@example.com",
                                  "source": "manual", "cached_at": None},
              raw["grandpa"])
        contacts = email_client._load_contacts()
        check("a legacy entry still resolves -- the TTL fix doesn't "
              "retroactively break contacts saved before it existed",
              contacts.get("grandpa") == "grandpa@example.com", contacts)
    _isolated_contacts(run)


def test_refresh_contact_bypasses_cache_and_rewrites_it():
    def run():
        # Seed a stale entry, then force a refresh -- should overwrite
        # with whatever the (stubbed) Google Contacts lookup returns,
        # bypassing the normal TTL check entirely (refresh is explicit,
        # opt-in, not gated on staleness).
        _write_raw_contacts({
            "madison": {"address": "old-madison@example.com",
                        "source": "google_contacts",
                        "cached_at": time.time()},  # not even stale
        })
        fake_module = types.SimpleNamespace(
            resolve_google_contact=lambda name: "new-madison@example.com")
        sys.modules["google_contacts"] = fake_module
        try:
            result = email_client.refresh_contact("madison")
        finally:
            del sys.modules["google_contacts"]
        check("refresh_contact() returns the freshly-resolved address",
              result == {"ok": True, "name": "madison",
                         "address": "new-madison@example.com"}, result)
        contacts = email_client._load_contacts()
        check("refresh_contact() overwrites the cached address even "
              "though the old entry wasn't stale yet -- it's an "
              "explicit escape hatch, not gated on the TTL",
              contacts.get("madison") == "new-madison@example.com", contacts)
    _isolated_contacts(run)


def test_refresh_contact_honest_miss_when_no_google_match():
    def run():
        fake_module = types.SimpleNamespace(
            resolve_google_contact=lambda name: None)
        sys.modules["google_contacts"] = fake_module
        try:
            result = email_client.refresh_contact("nobody")
        finally:
            del sys.modules["google_contacts"]
        check("refresh_contact() returns an honest failure (not a "
              "fabricated address) when Google Contacts has no match",
              result["ok"] is False, result)
    _isolated_contacts(run)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
