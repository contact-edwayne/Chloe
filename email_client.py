"""
email_client.py — Email (read + draft + confirm-gated send) for Chloe.

Ed, 2026-09-02: flagged as a gap after the wallet/social-posting work --
Chloe already drafts LinkedIn posts (social_composer.py) and moves real
money (wallet.py) but has no email surface at all. Built on the same two
house patterns already proven out elsewhere in this codebase:

1. Config-driven contact aliasing (mirrors stocks.py's ticker aliases /
   local_media.py's named folders): C:\\Chloe\\secrets\\email_contacts.json
   maps a lowercased name to an address, so "email John" resolves without
   Ed spelling out an address every time. An unresolved name returns None
   -- "honest miss, never guess" -- same contract as every other resolver
   in this codebase.

2. A hard, code-enforced confirm gate before anything irreversible sends
   -- mirrors wallet_send's PIN requirement and chloe_pending_confirms'
   Stage-3 "announce, then a LITERAL yes/no reply resolves it" state
   machine (though this is its own separate, simpler state file -- see
   below for why it isn't just routed through chloe_pending_confirms).
   `email_draft` is a normal LLM tool (composes + stores a pending draft,
   never sends). Actually sending is NOT an LLM tool at all: it is only
   reachable through try_handle_email_confirm_command, a deterministic
   regex/phrase check run on the raw user text BEFORE the LLM sees it
   (jarvis.py wires this the same way as lights/local_media). This means
   an LLM cannot draft-and-send an email in one hallucinated turn no
   matter what it's asked -- sending requires a real, separate human
   utterance that a human, not a model, decides to make. Ollama's own
   tool loop here runs multiple tool calls per turn (see
   _ollama_chat's MAX_TOOL_ITERS), so if email_send were ALSO an LLM
   tool the model could chain draft -> send inside a single request;
   keeping send out of the tool set entirely closes that off structurally
   rather than relying on prompting.

   Why not just reuse chloe_pending_confirms.py? Its `resolve()` is
   hard-wired to `chloe_proposals.apply_proposal(slug)` on "yes" -- it's
   Stage-3 of the self-modification pipeline specifically, not a generic
   confirm primitive, and bending it to also send emails would couple two
   unrelated safety-critical systems. This module reuses its pure
   `classify_reply()` phrase-matching (yes/no/neither from free text) but
   keeps its own state file and its own resolution logic.

Pending drafts live in C:\\Chloe\\brain\\pending_email.json (state, not a
secret -- same root chloe_pending_confirms.py uses), single most-recent
draft, 10-minute TTL. Credentials are IMAP/SMTP app-password auth (no
OAuth dance) via four env vars -- see _configured() below. Gmail: enable
2-Step Verification, then generate an App Password
(myaccount.google.com/apppasswords) -- a regular account password will
be rejected by Gmail's SMTP/IMAP.

2026-09-02: added reading a message's full body (email_read) and
replying to one (email_reply), on top of the original check/draft-new
flow. Both address a specific email by its 1-based INDEX in the most
recent email_check listing, not a raw IMAP id -- Ed says "read me the
first one" / "reply to that", the model already has the ordinal from
the listing it just showed him, and this module resolves that ordinal
back to the actual message via a short-lived cache (last_email_list.json,
30-min TTL, written every time email_check_tool runs) of that listing's
IMAP UIDs -- UIDs rather than sequence numbers because sequence numbers
shift as the mailbox changes between the check and the follow-up.
`email_read` is a plain LLM tool (read-only). `email_reply` follows the
exact same draft-then-confirm split as email_draft -- it only ever
stores a pending draft, carrying the original message's Message-ID/
References so the reply threads properly; actually sending still only
happens through try_handle_email_confirm_command, never as an LLM tool.
No body-content extraction beyond best-effort (prefers text/plain,
falls back to a tag-stripped text/html) -- good enough to read aloud,
not a MIME renderer.

2026-09-03: added attachments, resolved by voice from live Desktop
folders rather than a pre-registered config file -- Ed: "she should be
able to see the folders on my desktop and recognize which one I'm
referring to through voice command and find the file or photo within
that folder." Resolution is desktop_files.py's job (same honest-miss
ladder as local_media.py, applied to a live directory listing instead
of a config file); this module just calls it from draft_email/
draft_reply when `attachment_folder`/`attachment_file` are both given,
and stores the RESOLVED ABSOLUTE PATH (never the spoken phrase) on the
pending draft. A miss on either the folder or the file is an honest
tool-level error -- it never drafts silently without the attachment
Ed asked for. v1 scope: one attachment per email (matches what Ed
asked for -- "a file", "a photo" -- not a batch of them). _send_smtp
builds a multipart MIME message only when a draft carries an
attachment path; a plain draft still sends as a single MIMEText, same
as before this feature existed.

Public API
----------
resolve_contact(text) -> str | None
add_contact(name, address) -> dict
list_recent(n=5, unread_only=False) -> list[dict]         (raises on IMAP failure; each item carries a "uid")
email_check_tool(n=5, unread_only=False) -> str            (LLM tool target, never raises; refreshes the index cache)
read_email_body(uid) -> dict                                (raises on IMAP failure)
email_read_tool(index) -> str                               (LLM tool target, never raises)
draft_email(to, subject, body, attachment_folder=, attachment_file=, source_text=) -> dict
email_draft_tool(to, subject, body, attachment_folder=, attachment_file=, source_text=) -> str
                                                             (LLM tool target, never raises)
draft_reply(index, body, attachment_folder=, attachment_file=) -> dict
email_reply_tool(index, body, attachment_folder=, attachment_file=) -> str
                                                             (LLM tool target, never raises)
mark_draft_announced() -> bool                              (flips the pending draft's
                                                             announced flag; see below)
try_handle_email_confirm_command(text) -> str | None       (dispatcher contract: None = unclaimed;
                                                             also unclaimed if the pending draft
                                                             was never marked announced)

CLI:
    python email_client.py --add-contact "John Smith" john@example.com
    python email_client.py --check
    python email_client.py --check-unread
    python email_client.py --read 1
    python email_client.py --draft "john@example.com" "Subject" "Body text"
    python email_client.py --reply 1 "Reply body text"
"""
from __future__ import annotations

import email as _email
import email.encoders
import email.mime.application
import email.mime.base
import email.mime.image
import email.mime.multipart
import email.mime.text
import email.utils
import html as _html
import imaplib
import json
import mimetypes
import os
import re
import secrets
import smtplib
import sys
import time
from pathlib import Path
from typing import Optional

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CONTACTS_PATH = SECRETS_DIR / "email_contacts.json"

_DRAFT_TTL_S = 600  # 10 minutes
_LAST_LIST_TTL_S = 1800  # 30 minutes -- long enough to check email, get
                          # distracted, then come back and say "read the
                          # first one" without it going stale silently.


def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _draft_state_path() -> Path:
    p = _brain_root()
    p.mkdir(parents=True, exist_ok=True)
    return p / "pending_email.json"


def _last_list_path() -> Path:
    p = _brain_root()
    p.mkdir(parents=True, exist_ok=True)
    return p / "last_email_list.json"


def _save_last_list(uids: list, folder: str = "INBOX") -> None:
    p = _last_list_path()
    tmp = p.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps({"uids": uids, "folder": folder,
                               "expires_at": time.time() + _LAST_LIST_TTL_S}),
                    encoding="utf-8")
    os.replace(tmp, p)


def _load_last_list():
    """Returns (uids, folder) from the last email_check, or (None, None)
    if there isn't one / it expired. `folder` is the IMAP mailbox that
    list came from (e.g. "INBOX" or "[Gmail]/Trash") -- a follow-up
    email_read/email_reply has to look in the SAME mailbox, not assume
    Inbox, or it'll 404 on a UID that only exists in whatever folder Ed
    last checked."""
    p = _last_list_path()
    if not p.exists():
        return None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not data or data.get("expires_at", 0) <= time.time():
        return None, None
    return data.get("uids"), data.get("folder", "INBOX")


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _configured() -> bool:
    return bool(_env("CHLOE_EMAIL_ADDRESS") and _env("CHLOE_EMAIL_APP_PASSWORD"))


def _address() -> str:
    return _env("CHLOE_EMAIL_ADDRESS")


def _app_password() -> str:
    # Google's UI displays a generated App Password grouped in 4s for
    # readability ("abcd efgh ijkl mnop") but the actual credential has
    # no spaces -- a literal copy-paste including them is a common real-
    # world cause of "authentication failed" that has nothing to do with
    # whether the password itself is right. Strip ALL whitespace, not
    # just the leading/trailing whitespace _env() already handles.
    return _env("CHLOE_EMAIL_APP_PASSWORD").replace(" ", "").replace("\t", "")


def _imap_host() -> str:
    return _env("CHLOE_EMAIL_IMAP_HOST", "imap.gmail.com")


def _smtp_host() -> str:
    return _env("CHLOE_EMAIL_SMTP_HOST", "smtp.gmail.com")


def _smtp_port() -> int:
    try:
        return int(_env("CHLOE_EMAIL_SMTP_PORT", "587"))
    except ValueError:
        return 587


# --------------------------------------------------------------------------- #
# Contacts (same alias-file pattern as stocks.py / local_media.py)            #
# --------------------------------------------------------------------------- #

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_contacts() -> dict:
    if not CONTACTS_PATH.exists():
        return {}
    try:
        return json.loads(CONTACTS_PATH.read_text(encoding="utf-8")).get("contacts", {})
    except Exception:
        return {}


def add_contact(name: str, address: str) -> dict:
    name = (name or "").strip().lower()
    address = (address or "").strip()
    if not name or not _EMAIL_RE.match(address):
        return {"ok": False, "error": "need a name and a valid email address"}
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"contacts": _load_contacts()}
    data["contacts"][name] = address
    CONTACTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True, "name": name, "address": address}


def resolve_contact(text: str, source_text: Optional[str] = None) -> Optional[str]:
    """Resolve `text` to an email address: as-is if it already looks like
    one, else an exact/substring match against saved contacts. Honest
    miss -- returns None rather than guessing, same contract as
    stocks.resolve_ticker / lights._resolve_targets.

    2026-09-05: `text` here is whatever the LLM decided to pass as the
    `to` tool argument, NOT necessarily what Ed actually said. Observed
    live on qwen2.5:14b: told to email someone with no saved contact
    (e.g. "email Maddie"), it sometimes fabricates a plausible-looking
    address ("mcregvulk@gmail.com") and passes THAT as `to` instead of
    passing the name through and letting resolution honestly miss. Such
    a string still satisfies _EMAIL_RE, so it used to sail through
    untouched -- exactly the "honest miss, never guess" contract this
    function exists to enforce, bypassed by construction.

    `source_text` -- the raw user utterance/message for this turn, when
    the caller has it -- closes that gap: a well-formed address is only
    trusted as-is if it actually appears in what the user themselves
    typed or said (case-insensitively). A real address Ed spoke or typed
    will be present verbatim in the transcript; a model-invented one
    will not. Callers that can't supply it (CLI usage, tests) keep the
    old as-is behavior, just with a logged warning -- this is defense in
    depth for the LLM tool path, not a hard requirement everywhere."""
    text = (text or "").strip()
    if not text:
        return None
    if _EMAIL_RE.match(text):
        if source_text is None:
            return text
        if text.lower() in source_text.lower():
            return text
        print(f"[email_client] resolve_contact: {text!r} is well-formed but "
              f"doesn't appear in the user's own text -- treating as an "
              f"unresolved contact rather than trusting a possibly "
              f"fabricated address", flush=True)
        # Fall through to the contact lookup below (almost certainly a
        # miss too, since a fabricated address won't match a saved name)
        # rather than returning early -- same honest-miss ending either way.
    low = text.lower()
    contacts = _load_contacts()
    if low in contacts:
        return contacts[low]
    matches = [addr for name, addr in contacts.items() if name in low or low in name]
    if len(matches) == 1:
        return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Folder/label resolution (Gmail IMAP)                                        #
# --------------------------------------------------------------------------- #
# Ed (2026-09-06): email_check/email_read used to hardcode "INBOX" with no
# way to ask for anything else -- correct as a DEFAULT, wrong as the only
# option. Gmail exposes every label as its own IMAP mailbox under the
# "[Gmail]/" prefix (English UI); these are the standard ones, keyed by
# every phrasing Ed's likely to say.
_FOLDER_ALIASES = {
    "inbox": "INBOX",
    "all mail": "[Gmail]/All Mail",
    "all": "[Gmail]/All Mail",
    "everything": "[Gmail]/All Mail",
    "sent": "[Gmail]/Sent Mail",
    "sent mail": "[Gmail]/Sent Mail",
    "drafts": "[Gmail]/Drafts",
    "spam": "[Gmail]/Spam",
    "junk": "[Gmail]/Spam",
    "trash": "[Gmail]/Trash",
    "deleted": "[Gmail]/Trash",
    "deleted items": "[Gmail]/Trash",
    "starred": "[Gmail]/Starred",
    "important": "[Gmail]/Important",
}
_FOLDER_DISPLAY_NAMES = "Inbox, All Mail, Sent, Drafts, Spam, Trash, Starred, or Important"


def resolve_folder(spoken):
    """Map a spoken/typed folder phrase to (imap_mailbox, display_name).
    None/empty -> defaults to Inbox. Returns None on an unrecognized
    phrase -- honest miss (caller lists the valid names) rather than
    guessing which folder was meant."""
    if not spoken or not str(spoken).strip():
        return ("INBOX", "Inbox")
    key = str(spoken).strip().lower()
    mailbox = _FOLDER_ALIASES.get(key)
    if mailbox is None:
        return None
    display = "Inbox" if mailbox == "INBOX" else mailbox.split("/", 1)[-1]
    return (mailbox, display)


# --------------------------------------------------------------------------- #
# Reading (IMAP)                                                              #
# --------------------------------------------------------------------------- #

def list_recent(n: int = 5, unread_only: bool = False, folder: str = "INBOX") -> list[dict]:
    """Most-recent-first list of {uid, from, subject, date, unread}. Raises
    on any IMAP failure -- email_check_tool is the caller that turns that
    into an honest spoken/chat error instead of crashing the turn. Uses
    IMAP UIDs (not sequence numbers) so a follow-up email_read/email_reply
    can address the same message even if the mailbox has changed since."""
    n = max(1, min(int(n or 5), 25))
    with imaplib.IMAP4_SSL(_imap_host()) as imap:
        try:
            imap.login(_address(), _app_password())
        except imaplib.IMAP4.error as e:
            print(f"[email_client] IMAP login failed for {_address()!r}: {e}",
                  flush=True)
            raise
        imap.select(folder or "INBOX", readonly=True)
        if (folder or "INBOX") == "INBOX":
            # Gmail's web UI splits the Inbox into Primary/Social/
            # Promotions/Updates/Forums tabs -- purely a client-side
            # feature standard IMAP SEARCH knows nothing about. A plain
            # ALL/UNSEEN search against INBOX returns every message with
            # the Inbox label regardless of tab, including auto-sorted
            # notification mail (job alerts, etc.) that Ed doesn't
            # consider "new email" -- confirmed live 2026-09-06, "do I
            # have any new emails?" surfaced an Indeed/LinkedIn alert his
            # own Gmail Primary tab doesn't show. Gmail's IMAP extension
            # (X-GM-RAW) runs an actual Gmail search query, so it can
            # filter to the same Primary category the web UI defaults
            # to -- other folders (Sent/Drafts/Spam/Trash/etc.) don't
            # have categories, so this only applies to plain INBOX reads.
            gm_query = "category:primary" + (" is:unread" if unread_only else "")
            status, data = imap.uid("search", None, "X-GM-RAW", f'"{gm_query}"')
        else:
            crit = "UNSEEN" if unread_only else "ALL"
            status, data = imap.uid("search", None, crit)
        if status != "OK":
            raise RuntimeError(f"IMAP search failed: {status}")
        uids = data[0].split()
        uids = uids[-n:][::-1]  # most recent last in IMAP's numbering
        out = []
        for uid in uids:
            status, msg_data = imap.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            raw_headers = msg_data[0][1]
            flags_blob = str(msg_data[0][0])
            msg = _email.message_from_bytes(raw_headers)
            out.append({
                "uid": uid.decode() if isinstance(uid, bytes) else str(uid),
                "from": msg.get("From", "(unknown)"),
                "subject": msg.get("Subject", "(no subject)"),
                "date": msg.get("Date", ""),
                "unread": "\\Seen" not in flags_blob,
            })
        return out


def email_check_tool(n: int = 5, unread_only: bool = False, folder=None,
                      sender=None, subject=None) -> str:
    if not _configured():
        return ("Email isn't configured yet -- Ed needs to set "
                "CHLOE_EMAIL_ADDRESS and CHLOE_EMAIL_APP_PASSWORD in .env "
                "(a Gmail App Password, not the account password).")
    resolved = resolve_folder(folder)
    if resolved is None:
        return (f'I don\'t know a folder called "{folder}" -- I can check '
                f'{_FOLDER_DISPLAY_NAMES}.')
    mailbox, display = resolved
    try:
        if sender or subject:
            # BUG FIXED 2026-09-06: Ed asked "how many emails do I have
            # from Indeed Apply?" and got an answer that just eyeballed
            # the 5 most recent -- email_check had no way to actually
            # filter by sender, so any count/claim about "from X" was
            # unverified. Reuse the same Gmail search find_uids_by_query
            # already built for email_delete's sender/subject filter.
            query = " ".join(p for p in (
                f'from:"{sender}"' if sender else None,
                f'subject:"{subject}"' if subject else None,
            ) if p)
            msgs = find_uids_by_query(query, folder=mailbox, limit=max(1, min(int(n or 5), 25)))
            if unread_only:
                msgs = [m for m in msgs if m.get("unread")]
        else:
            msgs = list_recent(n=n, unread_only=unread_only, folder=mailbox)
    except Exception as e:
        print(f"[email_client] email_check_tool failed: {type(e).__name__}: {e}",
              flush=True)
        return f"Couldn't check email: {type(e).__name__}: {e}"
    _save_last_list([m["uid"] for m in msgs], folder=mailbox)
    where = "" if mailbox == "INBOX" else f" in {display}"
    if not msgs:
        return f"No unread emails{where}." if unread_only else f"{display} looks empty."
    lines = []
    for i, m in enumerate(msgs, 1):
        tag = "(unread) " if m["unread"] else ""
        lines.append(f"{i}. {tag}From {m['from']}, subject: {m['subject']}")
    header = (f"You have {len(msgs)} unread email(s){where}:" if unread_only
              else f"{len(msgs)} most recent email(s){where}:")
    trailer = ("\n\n(Sender + subject + date only -- no message body was "
               "fetched. To read one aloud or reply to one, refer to it by "
               "its number above -- I can call email_read or email_reply.)")
    return header + "\n" + "\n".join(lines) + trailer


# --------------------------------------------------------------------------- #
# Reading a specific message's body, by ordinal index into the last check     #
# --------------------------------------------------------------------------- #

_BODY_MAX_CHARS = 1500


def _decode_part(part) -> str:
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    except Exception:
        return ""


def _extract_body_text(msg) -> str:
    """Best-effort plain text extraction: prefers text/plain, falls back to
    a tag-stripped text/html. Good enough to read aloud -- not a MIME
    renderer."""
    plain = None
    html_part = None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            if ctype == "text/plain" and plain is None:
                plain = _decode_part(part)
            elif ctype == "text/html" and html_part is None:
                html_part = _decode_part(part)
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            plain = _decode_part(msg)
        elif ctype == "text/html":
            html_part = _decode_part(msg)
    if plain:
        return plain.strip()
    if html_part:
        text = re.sub(r"<[^>]+>", " ", html_part)
        text = _html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return ""


def read_email_body(uid, folder: str = "INBOX") -> dict:
    """Fetch one message's full body by IMAP UID, from `folder` (whatever
    mailbox the UID's own email_check listing came from -- see
    _load_last_list). Raises on IMAP failure -- email_read_tool is the
    caller that turns that into an honest error."""
    with imaplib.IMAP4_SSL(_imap_host()) as imap:
        try:
            imap.login(_address(), _app_password())
        except imaplib.IMAP4.error as e:
            print(f"[email_client] IMAP login failed for {_address()!r}: {e}",
                  flush=True)
            raise
        imap.select(folder or "INBOX", readonly=True)
        status, msg_data = imap.uid("fetch", uid, "(BODY.PEEK[] FLAGS)")
        if status != "OK" or not msg_data or not msg_data[0]:
            raise RuntimeError(f"IMAP fetch failed for uid {uid}")
        raw = msg_data[0][1]
        msg = _email.message_from_bytes(raw)
        body = _extract_body_text(msg)
        truncated = len(body) > _BODY_MAX_CHARS
        if truncated:
            body = body[:_BODY_MAX_CHARS]
        return {
            "from": msg.get("From", "(unknown)"),
            "subject": msg.get("Subject", "(no subject)"),
            "date": msg.get("Date", ""),
            "message_id": msg.get("Message-ID", ""),
            "references": msg.get("References", ""),
            "body": body,
            "truncated": truncated,
        }


def email_read_tool(index) -> str:
    if not _configured():
        return ("Email isn't configured yet -- Ed needs to set "
                "CHLOE_EMAIL_ADDRESS and CHLOE_EMAIL_APP_PASSWORD in .env "
                "(a Gmail App Password, not the account password).")
    uids, list_folder = _load_last_list()
    if not uids:
        return ("I don't have a recent email list -- check the inbox first "
                "(email_check), then ask me to read one by number.")
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return "Which one? Give me a number from the last inbox check."
    if idx < 1 or idx > len(uids):
        return f"I only have {len(uids)} email(s) from the last check -- pick a number in that range."
    uid = uids[idx - 1]
    try:
        msg = read_email_body(uid, folder=list_folder or "INBOX")
    except Exception as e:
        print(f"[email_client] email_read_tool failed: {type(e).__name__}: {e}",
              flush=True)
        return f"Couldn't read that email: {type(e).__name__}: {e}"
    body = msg["body"] or "(empty message)"
    note = " (truncated -- long email)" if msg["truncated"] else ""
    return f'From {msg["from"]}, subject "{msg["subject"]}":\n{body}{note}'


# --------------------------------------------------------------------------- #
# Deleting (move to Trash)                                                    #
# --------------------------------------------------------------------------- #
#
# Ed, 2026-09-06: asked Chloe to delete a batch of Indeed-Apply emails --
# she said she did, but there was no delete tool at all, so the LLM
# (forced to pick from what's available) called email_check(folder=
# 'Trash') and hallucinated "already moved to Trash" from nothing. This
# section is the real thing.
#
# Deliberately NOT behind the same hard confirm-gate as email send
# (see the module docstring above -- send is kept out of the LLM tool set
# entirely). Trashing a message is meaningfully lower-stakes than sending
# one as Ed or moving money: it's recoverable from Gmail's Trash for
# ~30 days, and Ed asked for exactly this ("delete messages on my
# command") as a normal voice capability, not a two-step ritual. The
# safety net here instead is _MAX_BULK_DELETE -- a too-broad filter
# refuses rather than silently trashing a big, unintended batch -- and
# honest-miss when neither indices nor a filter is given, so the model
# can never default to "delete everything."

_MAX_BULK_DELETE = 20  # refuse and ask Ed to narrow the filter beyond this


def _gm_raw_arg(query: str) -> str:
    """Quote+escape `query` as a single IMAP string argument for Gmail's
    X-GM-RAW search extension. Needed here (unlike the plain category:
    filter in list_recent) because sender/subject come from free text and
    may themselves contain a double quote or backslash."""
    escaped = query.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _trash_uid(imap, uid) -> None:
    """Move one message to Trash via Gmail's IMAP extension: adding the
    \\Trash label. Per Gmail's own IMAP docs this is the recommended way
    to delete a message -- Gmail automatically removes it from every
    other folder (including Inbox) once \\Trash is applied, no separate
    expunge needed, and the message stays recoverable from Trash for the
    normal ~30 days rather than being immediately, permanently gone."""
    imap.uid("STORE", uid, "+X-GM-LABELS", "(\\Trash)")


def find_uids_by_query(query: str, folder: str = "INBOX", limit: int = 25) -> list[dict]:
    """Search `folder` for messages matching a Gmail search query (e.g.
    'from:"Indeed Apply"') via X-GM-RAW -- the same engine as the Gmail
    search box. Most-recent-first, capped at `limit`. Same dict shape as
    list_recent (uid/from/subject/date/unread) so email_check_tool can
    use either interchangeably. Raises on IMAP failure, same contract as
    list_recent."""
    with imaplib.IMAP4_SSL(_imap_host()) as imap:
        imap.login(_address(), _app_password())
        imap.select(folder or "INBOX", readonly=True)
        status, data = imap.uid("search", None, "X-GM-RAW", _gm_raw_arg(query))
        if status != "OK":
            raise RuntimeError(f"IMAP search failed: {status}")
        uids = data[0].split()
        uids = uids[-limit:][::-1]
        out = []
        for uid in uids:
            status, msg_data = imap.uid(
                "fetch", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            raw_headers = msg_data[0][1]
            flags_blob = str(msg_data[0][0])
            msg = _email.message_from_bytes(raw_headers)
            out.append({
                "uid": uid.decode() if isinstance(uid, bytes) else str(uid),
                "from": msg.get("From", "(unknown)"),
                "subject": msg.get("Subject", "(no subject)"),
                "date": msg.get("Date", ""),
                "unread": "\\Seen" not in flags_blob,
            })
        return out


def email_delete_tool(indices=None, sender=None, subject=None, folder=None) -> str:
    """Move one or more emails to Trash. Two ways to say which ones:
    - `indices`: 1-based number(s) from the last email_check listing,
      same addressing as email_read_tool/draft_reply.
    - `sender` and/or `subject`: a fresh Gmail search against `folder`
      (default Inbox).
    Honest-miss if neither is given -- never guess "all of them." Capped
    at _MAX_BULK_DELETE matches so a too-broad filter can't silently
    trash a large batch; Ed gets the count and is asked to narrow it."""
    if not _configured():
        return ("Email isn't configured yet -- Ed needs to set "
                "CHLOE_EMAIL_ADDRESS and CHLOE_EMAIL_APP_PASSWORD in .env "
                "(a Gmail App Password, not the account password).")

    targets = []  # list of (uid, from_or_None, subject_or_None)
    list_folder = None

    if indices:
        uids, list_folder = _load_last_list()
        if not uids:
            return ("I don't have a recent email list to pick numbers from "
                    "-- check the inbox first (email_check), or tell me a "
                    "sender or subject to search for instead.")
        idx_list = indices if isinstance(indices, list) else [indices]
        bad = []
        seen = set()
        for raw in idx_list:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                bad.append(raw)
                continue
            if idx < 1 or idx > len(uids) or idx in seen:
                bad.append(raw)
                continue
            seen.add(idx)
            targets.append((uids[idx - 1], None, None))
        if bad:
            return (f"I only have {len(uids)} email(s) from the last check "
                    f"-- {bad!r} isn't a valid number in that range.")
    elif sender or subject:
        resolved = resolve_folder(folder)
        if resolved is None:
            return (f'I don\'t know a folder called "{folder}" -- I can check '
                    f'{_FOLDER_DISPLAY_NAMES}.')
        mailbox, display = resolved
        query = " ".join(p for p in (
            f'from:"{sender}"' if sender else None,
            f'subject:"{subject}"' if subject else None,
        ) if p)
        try:
            matches = find_uids_by_query(query, folder=mailbox,
                                          limit=_MAX_BULK_DELETE + 1)
        except Exception as e:
            print(f"[email_client] email_delete_tool search failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            return f"Couldn't search for those emails: {type(e).__name__}: {e}"
        if not matches:
            return f'No emails matching {query} in {display}.'
        if len(matches) > _MAX_BULK_DELETE:
            return (f'That matches {len(matches)}+ emails in {display} -- '
                    f'more than I\'ll delete in one go ({_MAX_BULK_DELETE} max). '
                    f'Narrow it down (a more specific sender or subject) and '
                    f'try again.')
        list_folder = mailbox
        targets = [(m["uid"], m["from"], m["subject"]) for m in matches]
    else:
        return ("Which emails? Give me a sender or subject to search for, "
                "or refer to numbers from your last email check.")

    trashed = []
    failed = []
    try:
        with imaplib.IMAP4_SSL(_imap_host()) as imap:
            imap.login(_address(), _app_password())
            imap.select(list_folder or "INBOX", readonly=False)
            for uid, frm, subj in targets:
                try:
                    _trash_uid(imap, uid)
                    trashed.append((uid, frm, subj))
                except Exception as e:
                    failed.append((uid, str(e)))
    except Exception as e:
        print(f"[email_client] email_delete_tool IMAP session failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return f"Couldn't delete: {type(e).__name__}: {e}"

    if not trashed:
        return f"Couldn't delete any of those: {failed}"

    n = len(trashed)
    named = [(frm, subj) for _, frm, subj in trashed if frm]
    if named:
        preview = "; ".join(f'"{subj}" from {frm}' for frm, subj in named[:5])
        more = f" and {n - 5} more" if n > 5 else ""
        detail = f": {preview}{more}"
    else:
        detail = ""
    fail_note = f" ({len(failed)} couldn't be deleted)" if failed else ""
    return f"Moved {n} email(s) to Trash{detail}{fail_note}."


# --------------------------------------------------------------------------- #
# Drafting + confirm-gated sending                                            #
# --------------------------------------------------------------------------- #

def _load_pending() -> Optional[dict]:
    p = _draft_state_path()
    if not p.exists():
        return None
    try:
        entry = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not entry or entry.get("expires_at", 0) <= time.time():
        return None
    return entry


def _save_pending(entry: Optional[dict]) -> None:
    p = _draft_state_path()
    if entry is None:
        p.unlink(missing_ok=True)
        return
    tmp = p.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _resolve_attachment(attachment_folder: Optional[str],
                        attachment_file: Optional[str]) -> dict:
    """Resolve (folder phrase, file phrase) to an absolute path via
    desktop_files.py's live honest-miss ladder. Returns {"ok": True,
    "path": Path} or {"ok": False, "error": str}. Requires BOTH phrases
    together -- Ed's own phrasing is "the <file> from my <folder>
    folder", and resolving a bare file phrase against every Desktop
    folder at once would be exactly the kind of guess this codebase's
    resolvers are built to refuse."""
    attachment_folder = (attachment_folder or "").strip()
    attachment_file = (attachment_file or "").strip()
    if not attachment_folder or not attachment_file:
        return {"ok": False,
                "error": "need both a folder and a file/photo description "
                        "to attach something"}
    try:
        import desktop_files
    except ImportError as e:
        return {"ok": False, "error": f"attachment support unavailable: {e}"}
    folder = desktop_files.resolve_folder(attachment_folder)
    if folder is None:
        return {"ok": False,
                "error": f'no folder matching "{attachment_folder}" on the desktop'}
    f = desktop_files.resolve_file(folder, attachment_file)
    if f is None:
        return {"ok": False,
                "error": f'nothing matching "{attachment_file}" in the '
                        f'{folder.name} folder'}
    size_error = desktop_files.check_attachment_size(f)
    if size_error:
        return {"ok": False, "error": size_error}
    return {"ok": True, "path": f}


def draft_email(to: str, subject: str, body: str, *,
                attachment_folder: Optional[str] = None,
                attachment_file: Optional[str] = None,
                source_text: Optional[str] = None) -> dict:
    address = resolve_contact(to, source_text=source_text)
    if not address:
        return {"ok": False, "error": f'no email address for "{to}"',
                "error_kind": "contact"}
    attachment_path = None
    if attachment_folder or attachment_file:
        resolved = _resolve_attachment(attachment_folder, attachment_file)
        if not resolved["ok"]:
            resolved["error_kind"] = "attachment"
            return resolved
        attachment_path = str(resolved["path"])
    entry = {
        "id": secrets.token_hex(3),
        "to": address,
        "to_raw": to,
        "subject": (subject or "").strip() or "(no subject)",
        "body": body or "",
        "attachment_path": attachment_path,
        "created_at": time.time(),
        "expires_at": time.time() + _DRAFT_TTL_S,
        # Flipped by mark_draft_announced() once the "Drafted to X -- say
        # 'send it'" confirmation has actually made it back to Ed.
        # try_handle_email_confirm_command refuses to send while this is
        # False -- see its docstring for the incident that motivated it.
        "announced": False,
    }
    _save_pending(entry)
    return {"ok": True, **entry}


def email_draft_tool(to: str, subject: str, body: str, *,
                     attachment_folder: Optional[str] = None,
                     attachment_file: Optional[str] = None,
                     source_text: Optional[str] = None) -> str:
    if not _configured():
        return ("Email isn't configured yet -- Ed needs to set "
                "CHLOE_EMAIL_ADDRESS and CHLOE_EMAIL_APP_PASSWORD in .env "
                "(a Gmail App Password, not the account password).")
    r = draft_email(to, subject, body,
                    attachment_folder=attachment_folder,
                    attachment_file=attachment_file,
                    source_text=source_text)
    if not r["ok"]:
        if r.get("error_kind") == "contact":
            return (f'{r["error"]} -- give me the address, or add them as a '
                    f'contact first.')
        return f'{r["error"]}.'
    preview = r["body"][:200] + ("..." if len(r["body"]) > 200 else "")
    attach_note = f' Attached: {Path(r["attachment_path"]).name}.' if r.get("attachment_path") else ""
    return (f'Drafted to {r["to"]} — subject: "{r["subject"]}". '
            f'"{preview}"{attach_note} Say "send it" to send, or "cancel" to drop it. '
            f'This will NOT send until you say so.')


def draft_reply(index, body: str, *,
                attachment_folder: Optional[str] = None,
                attachment_file: Optional[str] = None) -> dict:
    """Same draft-then-confirm split as draft_email, but addresses the
    sender of a prior email_check listing by 1-based index and carries
    Message-ID/References so the reply threads properly."""
    uids, list_folder = _load_last_list()
    if not uids:
        return {"ok": False, "error": "no recent email list -- check the inbox first"}
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid index"}
    if idx < 1 or idx > len(uids):
        return {"ok": False, "error": f"only {len(uids)} email(s) in the last check"}
    uid = uids[idx - 1]
    try:
        original = read_email_body(uid, folder=list_folder or "INBOX")
    except Exception as e:
        return {"ok": False, "error": f"couldn't load that email: {type(e).__name__}: {e}"}
    to_addr = email.utils.parseaddr(original["from"])[1]
    if not to_addr:
        return {"ok": False, "error": "couldn't find a reply address on that email"}
    attachment_path = None
    if attachment_folder or attachment_file:
        resolved = _resolve_attachment(attachment_folder, attachment_file)
        if not resolved["ok"]:
            return resolved
        attachment_path = str(resolved["path"])
    subj = original["subject"] or ""
    if not subj.lower().startswith("re:"):
        subj = f"Re: {subj}"
    refs = (original.get("references") or "").strip()
    msg_id = (original.get("message_id") or "").strip()
    references = (refs + " " + msg_id).strip() if refs else msg_id
    entry = {
        "id": secrets.token_hex(3),
        "to": to_addr,
        "to_raw": original["from"],
        "subject": subj,
        "body": body or "",
        "attachment_path": attachment_path,
        "in_reply_to": msg_id or None,
        "references": references or None,
        "created_at": time.time(),
        "expires_at": time.time() + _DRAFT_TTL_S,
        "announced": False,
    }
    _save_pending(entry)
    return {"ok": True, **entry}


def email_reply_tool(index, body: str, *,
                     attachment_folder: Optional[str] = None,
                     attachment_file: Optional[str] = None) -> str:
    if not _configured():
        return ("Email isn't configured yet -- Ed needs to set "
                "CHLOE_EMAIL_ADDRESS and CHLOE_EMAIL_APP_PASSWORD in .env "
                "(a Gmail App Password, not the account password).")
    r = draft_reply(index, body,
                    attachment_folder=attachment_folder,
                    attachment_file=attachment_file)
    if not r["ok"]:
        return f'{r["error"]}.'
    preview = r["body"][:200] + ("..." if len(r["body"]) > 200 else "")
    attach_note = f' Attached: {Path(r["attachment_path"]).name}.' if r.get("attachment_path") else ""
    return (f'Drafted a reply to {r["to"]} — subject: "{r["subject"]}". '
            f'"{preview}"{attach_note} Say "send it" to send, or "cancel" to drop it. '
            f'This will NOT send until you say so.')


def _build_attachment_part(path: Path):
    """MIMEImage for image/*, MIMEApplication for application/*, else a
    generic MIMEBase + base64 fallback (covers audio/video/text/anything
    mimetypes can't narrow down) -- same three-tier approach the stdlib
    email docs use for "attach any file type". Content-Disposition
    carries the original filename so it shows up right in the client."""
    ctype, encoding = mimetypes.guess_type(str(path))
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    data = path.read_bytes()
    if maintype == "image":
        part = email.mime.image.MIMEImage(data, _subtype=subtype)
    elif maintype == "application":
        part = email.mime.application.MIMEApplication(data, _subtype=subtype)
    else:
        part = email.mime.base.MIMEBase(maintype, subtype)
        part.set_payload(data)
        email.encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename=path.name)
    return part


def _send_smtp(to: str, subject: str, body: str, *, in_reply_to: Optional[str] = None,
               references: Optional[str] = None,
               attachment_path: Optional[str] = None) -> dict:
    try:
        # Plain MIMEText when there's no attachment -- unchanged from
        # before this feature existed. Multipart only gets built when a
        # draft actually carries an attachment path (2026-09-03).
        if attachment_path:
            p = Path(attachment_path)
            if not p.is_file():
                return {"ok": False,
                        "error": f"attachment no longer exists on disk: {p.name}"}
            msg = email.mime.multipart.MIMEMultipart()
            msg.attach(email.mime.text.MIMEText(body, "plain", "utf-8"))
            msg.attach(_build_attachment_part(p))
        else:
            msg = email.mime.text.MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = _address()
        msg["To"] = to
        msg["Date"] = email.utils.formatdate(localtime=True)
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if references:
            msg["References"] = references
        with smtplib.SMTP(_smtp_host(), _smtp_port(), timeout=15) as s:
            s.starttls()
            s.login(_address(), _app_password())
            s.send_message(msg)
        return {"ok": True}
    except Exception as e:
        print(f"[email_client] SMTP send failed: {type(e).__name__}: {e}",
              flush=True)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def mark_draft_announced() -> bool:
    """Flip the pending draft's `announced` flag. Call this once the
    "Drafted to X -- say 'send it' to confirm" text has actually reached
    Ed (spoken and/or shown), not merely once it was generated.

    2026-09-05, incident: a draft was created successfully, but the
    Ollama call generating that confirmation text then hit a ReadTimeout
    (180s timeout, ~310s actual) -- Ed never heard or saw it, yet the
    draft stayed live and confirmable for its full 10-minute TTL. An
    unrelated later "yes"-shaped utterance could have sent a
    wrong-recipient email that no human ever reviewed.
    try_handle_email_confirm_command now refuses to act on a draft until
    this has been called, closing that window. Precise wiring would call
    this only after confirmed TTS/display delivery; jarvis.py's
    _ollama_chat calls it as soon as the confirmation-generating request
    returns a non-empty reply without raising/timing out, which covers
    the incident above even though it doesn't yet cover a downstream
    TTS-only failure."""
    pending = _load_pending()
    if pending is None:
        return False
    pending["announced"] = True
    _save_pending(pending)
    return True


def try_handle_email_confirm_command(text: str) -> Optional[str]:
    """Deterministic (non-LLM) confirm/cancel gate for the one pending
    email draft, if any. Returns None (unclaimed) whenever there's no
    pending draft, the draft was never announced to Ed (see
    mark_draft_announced), OR the text isn't a recognizable yes/no -- so
    ordinary conversation never gets swallowed by this. See module
    docstring for why sending lives here instead of as an LLM tool."""
    pending = _load_pending()
    if pending is None:
        return None
    if not pending.get("announced"):
        # Draft exists but Ed was never actually told about it -- don't
        # treat this utterance as a send/cancel decision on it (leave the
        # draft in place; it still expires on its own via the normal
        # TTL). See mark_draft_announced's docstring for why.
        print(f"[email_client] confirm phrase seen but draft {pending.get('id')} "
              f"was never announced to Ed -- ignoring (not sending)",
              flush=True)
        return None
    try:
        import chloe_pending_confirms as _cpc
        decision = _cpc.classify_reply(text)
    except Exception:
        decision = ""
    if not decision:
        return None

    _save_pending(None)  # single-resolution, same as chloe_pending_confirms

    if decision == "no":
        return "Okay, I won't send that."

    if not _configured():
        return "Email isn't configured -- can't actually send this."
    r = _send_smtp(pending["to"], pending["subject"], pending["body"],
                   in_reply_to=pending.get("in_reply_to"),
                   references=pending.get("references"),
                   attachment_path=pending.get("attachment_path"))
    if r["ok"]:
        return f'Sent to {pending["to"]}.'
    return f'That didn\'t send -- {r["error"]}'


def _cli() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 0
    if args[0] == "--add-contact" and len(args) == 3:
        print(add_contact(args[1], args[2]))
        return 0
    if args[0] == "--check":
        print(email_check_tool())
        return 0
    if args[0] == "--check-unread":
        print(email_check_tool(unread_only=True))
        return 0
    if args[0] == "--read" and len(args) == 2:
        print(email_read_tool(args[1]))
        return 0
    if args[0] == "--draft" and len(args) == 4:
        print(email_draft_tool(args[1], args[2], args[3]))
        return 0
    if args[0] == "--reply" and len(args) == 3:
        print(email_reply_tool(args[1], args[2]))
        return 0
    print("unrecognized arguments; see module docstring for CLI usage")
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
