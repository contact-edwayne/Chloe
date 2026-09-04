"""
Chloe -- social health check + secrets bootstrap.

Subcommands:

  python social_health.py init --handle ... --app-password ...
      Writes C:\Chloe\secrets\social.json. Run once.

  python social_health.py check
      Loads secrets, attempts a Bluesky session, prints status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import social  # noqa: E402


def _cmd_init(args: argparse.Namespace) -> int:
    if not args.handle or not args.app_password:
        print("ERR: both --handle and --app-password are required", file=sys.stderr)
        return 3

    path = social.save_secrets(
        bluesky_handle=args.handle,
        bluesky_app_password=args.app_password,
        linkedin_profile_url=args.linkedin_profile or None,
    )
    print(f"OK  wrote {path}")
    print("    bluesky:  handle stored, app_password stored")
    print("    linkedin: mode=draft_only")
    print()
    print("Next:  python social_health.py check")
    return 0


def _cmd_check(_args: argparse.Namespace) -> int:
    print(social.banner())

    try:
        secrets = social.load_secrets()
    except FileNotFoundError as e:
        print(f"ERR  {e}", file=sys.stderr)
        return 2

    bsky_cfg = secrets.get("bluesky") or {}
    handle = bsky_cfg.get("handle", "")
    has_pw = bool(bsky_cfg.get("app_password"))
    print("secrets file OK")
    print(f"  bluesky handle:        {handle or '<missing>'}")
    print(f"  bluesky app_password:  {'<set>' if has_pw else '<missing>'}")
    print(f"  linkedin mode:         {(secrets.get('linkedin') or {}).get('mode', '<missing>')}")
    print()

    print("attempting Bluesky session...")
    try:
        client = social.bluesky_from_secrets()
        session = client.create_session()
    except social.BlueskyAuthError as e:
        print(f"FAIL  Bluesky auth: {e}", file=sys.stderr)
        return 1

    print("OK  Bluesky session established")
    print(f"  handle:  {session.handle}")
    print(f"  did:     {session.did}")
    print(f"  active:  {session.active}")
    print(f"  access:  {session.access_jwt[:18]}...  (held in memory only)")
    print()

    try:
        profile = client.get_profile()
    except Exception as e:
        print(f"WARN profile fetch failed: {e}")
    else:
        print("OK  profile fetched")
        display_name = profile.get("displayName") or ""
        description = profile.get("description") or ""
        desc_lines = description.splitlines()
        desc_first = desc_lines[0][:80] if desc_lines else "(none)"
        print(f"  displayName:  {display_name or '(none)'}")
        print(f"  description:  {desc_first}")
        print(f"  followers:    {profile.get('followersCount', 0)}")
        print(f"  follows:      {profile.get('followsCount', 0)}")
        print(f"  posts:        {profile.get('postsCount', 0)}")
        if not display_name or not description:
            print()
            print("  HINT: profile is incomplete -- open the Bluesky app and set")
            print("        a display name + bio before Session 2 (the bio is what")
            print("        actually labels Chloe as AI to anyone who clicks in).")

    print()
    print("LinkedIn: draft-only mode (no API call -- by design)")
    print()
    print("Phase 1 OK. Phase 2 in place: social_db.py + social_composer.py + chloe_post.py.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="social_health",
        description="Chloe social -- health check + secrets bootstrap",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="write social secrets")
    p_init.add_argument("--handle", required=True, help="Bluesky handle")
    p_init.add_argument("--app-password", required=True, help="Bluesky app password")
    p_init.add_argument("--linkedin-profile", default="", help="(optional) Ed's LinkedIn profile URL")
    p_init.set_defaults(func=_cmd_init)

    p_check = sub.add_parser("check", help="run Bluesky auth round-trip")
    p_check.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    if not args.cmd:
        return _cmd_check(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
