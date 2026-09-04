"""
notify.py — Shared ntfy.sh push-notification sender for Chloe.

Extracted 2026-09-02 from chloe_watchdog._send_alert (which now delegates
here) so any module can push a notification to Ed's phone without
duplicating the ntfy wiring. Same shared-secret-topic pattern
chloe_watchdog established: CHLOE_NTFY_TOPIC is a private, hard-to-guess
string Ed subscribes to in the ntfy app (or https://ntfy.sh/<topic> in a
browser) -- anyone who knows the topic can push to it, so it isn't logged
anywhere and shouldn't be committed. Already configured in this install
(see .env, set up for watchdog alerts) -- this module just lets other
features (the notify_me tool, wallet-send confirmations, ...) reuse the
same pipe instead of each rolling its own HTTP POST.

Public API
----------
send_ntfy(title, body, *, priority="default", tags="") -> bool
    Best-effort push. Returns False (never raises) if CHLOE_NTFY_TOPIC
    is unset or the request fails -- callers should treat delivery as
    fire-and-forget, never a reason to fail the calling command.

Priority: one of "min"|"low"|"default"|"high"|"urgent" (ntfy's scale).
Tags: comma-separated ntfy emoji-tag names, e.g. "warning", "moneybag",
"robot" -- see https://docs.ntfy.sh/publish/#tags-emojis.
"""
from __future__ import annotations

import os
import urllib.request


def send_ntfy(title: str, body: str, *, priority: str = "default",
              tags: str = "") -> bool:
    topic = os.environ.get("CHLOE_NTFY_TOPIC", "").strip()
    if not topic:
        return False
    server = os.environ.get("CHLOE_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    try:
        headers = {
            "Title": title,
            "Priority": priority,
            "User-Agent": "chloe-notify",
        }
        if tags:
            headers["Tags"] = tags
        req = urllib.request.Request(
            f"{server}/{topic}",
            data=(body or "").encode("utf-8"),
            method="POST",
            headers=headers,
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"[notify] send failed: {e}", flush=True)
        return False


def _cli() -> int:
    import sys
    if len(sys.argv) < 3:
        print("usage: python notify.py <title> <body> [priority] [tags]")
        return 1
    title, body = sys.argv[1], sys.argv[2]
    priority = sys.argv[3] if len(sys.argv) > 3 else "default"
    tags = sys.argv[4] if len(sys.argv) > 4 else ""
    ok = send_ntfy(title, body, priority=priority, tags=tags)
    print("sent" if ok else "not sent (CHLOE_NTFY_TOPIC unset, or request failed)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
