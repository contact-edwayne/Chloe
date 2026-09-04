"""
Chloe — social CLI.

A thin command-line wrapper around social_db + social_composer + social
so Ed can drive the full draft → approve → publish loop from cmd while
the PWA tab is still pending (Session 3). Same modules the WS handlers
in jarvis.py call, so anything that works here will work in the HUD
once that's wired.

Commands
────────
  draft   --trigger <name> --context "<text>" [--platform bluesky|linkedin]
              → asks the composer for a draft, persists it as pending,
                prints the id + body for review.

  list    [--status pending|approved|published|rejected|failed]
          [--platform bluesky|linkedin] [--limit N]
              → table of recent drafts.

  show    --id <N>
              → full row for one draft, including source_trace JSON.

  edit    --id <N> --body "<text>"
              → replaces the draft's body before approve.

  approve --id <N> [--body "<text>"]
              → approves the draft and publishes it. For Bluesky:
                hits createRecord, persists post URI. For LinkedIn:
                writes a draft file in C:\\Chloe\\secrets\\linkedin_drafts\\
                that Ed pastes by hand.

  reject  --id <N> [--reason "<text>"]

Exit codes
──────────
  0  success
  1  composer / publish failure
  2  bad arguments / not found
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import social         # noqa: E402  Bluesky client + secrets I/O
import social_db      # noqa: E402  SQLite drafts DAO
import social_composer  # noqa: E402  persona-driven composer


DAILY_CAPS = {"bluesky": 2, "linkedin": 999}


def _fmt_row(row: dict, *, full: bool = False) -> str:
    out = []
    out.append(f"  id:        {row['id']}")
    out.append(f"  platform:  {row['platform']}")
    out.append(f"  status:    {row['status']}")
    out.append(f"  trigger:   {row.get('source_trigger', '')}")
    body = row.get("final_body") or row.get("body") or ""
    if not full and len(body) > 200:
        body = body[:200] + " …"
    out.append(f"  body:      {body!r}")
    if row.get("rationale"):
        out.append(f"  rationale: {row['rationale']}")
    if row.get("post_uri"):
        out.append(f"  post_uri:  {row['post_uri']}")
    if row.get("reject_reason"):
        out.append(f"  rejected:  {row['reject_reason']}")
    if row.get("fail_reason"):
        out.append(f"  failed:    {row['fail_reason']}")
    if full and row.get("source_trace"):
        out.append(f"  trace:     {json.dumps(row['source_trace'])[:400]}")
    return "\n".join(out)


def cmd_draft(args: argparse.Namespace) -> int:
    platform = args.platform
    trigger = args.trigger
    context = args.context.strip()
    if not context:
        print("ERR: --context cannot be empty", file=sys.stderr)
        return 2

    recent = social_db.recent_published_bodies(platform, n=5)
    try:
        out = social_composer.compose_post(
            platform=platform, trigger=trigger,
            context=context, recent_bodies=recent,
        )
    except social_composer.ComposerError as e:
        print(f"FAIL composer: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    draft_id = social_db.create_draft(
        platform=platform,
        body=out["body"],
        rationale=out["rationale"],
        source_trigger=trigger,
        source_trace={
            "model_used": out["model_used"],
            "latency_ms": out["latency_ms"],
            "context_preview": context[:240],
        },
    )

    print(f"OK draft id={draft_id}  model={out['model_used']}  "
          f"latency={out['latency_ms']}ms  chars={len(out['body'])}")
    print()
    print(textwrap.indent(out["body"], "    "))
    print()
    print(f"rationale: {out['rationale']}")
    print()
    print(f"Next:  python chloe_post.py approve --id {draft_id}")
    print(f"  or:  python chloe_post.py edit    --id {draft_id} --body \"...\"")
    print(f"  or:  python chloe_post.py reject  --id {draft_id} --reason \"...\"")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = social_db.list_drafts(
        status=args.status, platform=args.platform, limit=args.limit
    )
    if not rows:
        print("(no drafts match)")
        return 0
    today_bsky = social_db.todays_published_count("bluesky")
    cap = DAILY_CAPS["bluesky"]
    print(f"Today on Bluesky: {today_bsky}/{cap} published")
    print()
    for row in rows:
        print(_fmt_row(row))
        print("  ---")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        row = social_db.get_draft(args.id)
    except LookupError as e:
        print(f"ERR {e}", file=sys.stderr)
        return 2
    print(_fmt_row(row, full=True))
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    body = args.body.strip()
    if not body:
        print("ERR: --body cannot be empty", file=sys.stderr)
        return 2
    try:
        row = social_db.update_body(args.id, body)
    except LookupError as e:
        print(f"ERR {e}", file=sys.stderr)
        return 2
    print(f"OK draft {args.id} body updated ({len(body)} chars)")
    print()
    print(_fmt_row(row))
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    try:
        row = social_db.reject_draft(args.id, args.reason or "")
    except LookupError as e:
        print(f"ERR {e}", file=sys.stderr)
        return 2
    print(f"OK draft {args.id} rejected")
    print(_fmt_row(row))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    try:
        draft = social_db.get_draft(args.id)
    except LookupError as e:
        print(f"ERR {e}", file=sys.stderr)
        return 2

    platform = draft["platform"]

    # Daily cap
    posted_today = social_db.todays_published_count(platform)
    cap = DAILY_CAPS.get(platform, 2)
    if posted_today >= cap:
        print(
            f"FAIL daily cap reached on {platform} ({posted_today}/{cap}). "
            "Try again tomorrow, or reject and re-draft.",
            file=sys.stderr,
        )
        return 1

    # Optional body override
    override = args.body.strip() if args.body else None

    try:
        social_db.approve_draft(args.id, override)
    except LookupError as e:
        print(f"ERR {e}", file=sys.stderr)
        return 2

    draft = social_db.get_draft(args.id)
    final_body = draft["final_body"]

    if platform == "linkedin":
        path = social.linkedin_export_draft(f"draft-{args.id}", final_body)
        row = social_db.mark_published(
            args.id, post_uri=f"file://{path}", post_cid="linkedin-draft"
        )
        print(f"OK linkedin draft exported to {path}")
        print("    paste it into LinkedIn manually.")
        print()
        print(_fmt_row(row))
        return 0

    # Bluesky publish
    try:
        client = social.bluesky_from_secrets()
        result = client.create_post(final_body)
    except social.BlueskyAuthError as e:
        social_db.mark_failed(args.id, str(e))
        print(f"FAIL Bluesky: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        social_db.mark_failed(args.id, f"{type(e).__name__}: {e}")
        print(f"FAIL {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    row = social_db.mark_published(
        args.id, post_uri=result["uri"], post_cid=result["cid"]
    )

    # Convert at://did:plc:.../app.bsky.feed.post/rkey → bsky.app URL
    uri = result["uri"]
    rkey = uri.rsplit("/", 1)[-1] if "/" in uri else ""
    handle = (social.load_secrets().get("bluesky") or {}).get("handle", "")
    web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey and handle else uri

    print(f"OK published — {web_url}")
    print(f"   uri: {uri}")
    print(f"   cid: {result['cid']}")
    print()
    print(_fmt_row(row))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chloe_post", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("draft", help="ask the composer for a new draft")
    p.add_argument("--platform", default="bluesky", choices=["bluesky", "linkedin"])
    p.add_argument("--trigger", required=True, help="e.g. ship_note, manual, hello_world")
    p.add_argument("--context", required=True, help="what Chloe should draft about")
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("list", help="list drafts (newest first)")
    p.add_argument("--status", default=None)
    p.add_argument("--platform", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one draft in full")
    p.add_argument("--id", type=int, required=True)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("edit", help="replace a pending draft's body")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--body", required=True)
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("reject", help="reject a pending/approved draft")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_reject)

    p = sub.add_parser("approve", help="approve and publish (Bluesky) or export (LinkedIn)")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--body", default="", help="optional override of the draft body")
    p.set_defaults(func=cmd_approve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
