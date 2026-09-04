import sys

PATH = "jarvis.py"
text = open(PATH, "r", encoding="utf-8").read()

edits = []

# 1. Import email confirm handler alongside local_media's import.
edits.append(("import email_client", 
"""from local_media import try_handle_local_media_command
""",
"""from local_media import try_handle_local_media_command
from email_client import try_handle_email_confirm_command
"""))

# 2. Chat dispatch: email confirm block before the chat YouTube block.
edits.append(("chat dispatch: email confirm",
'''                try:
                    await _reply_audio_or_speak(local_media_reply, data, label="chat-local-media")
                except Exception as e:
                    print(f"[chloe] chat TTS error on local-media reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

        # YouTube: "play <playlist>" / "play my <playlist> playlist"
''',
'''                try:
                    await _reply_audio_or_speak(local_media_reply, data, label="chat-local-media")
                except Exception as e:
                    print(f"[chloe] chat TTS error on local-media reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

        # Email send/cancel confirm: only fires when there's an actual pending
        # draft (email_client.draft_email) and the text is a recognizable
        # yes/no -- see email_client.try_handle_email_confirm_command's
        # docstring for why sending is gated here instead of as an LLM tool.
        email_confirm_reply = await asyncio.to_thread(try_handle_email_confirm_command, _last_user_l)
        if email_confirm_reply is not None:
            _push_history("user", _last_user_l, modality="chat")
            _push_history("assistant", email_confirm_reply, modality="chat")
            await _ws_send(websocket, {"type": "start"})
            await _ws_send(websocket, {"type": "delta", "text": email_confirm_reply})
            await _ws_send(websocket, {"type": "done"})
            if not data.get("no_tts"):
                try:
                    await _reply_audio_or_speak(email_confirm_reply, data, label="chat-email-confirm")
                except Exception as e:
                    print(f"[chloe] chat TTS error on email-confirm reply: {e}")
                finally:
                    hud_server.broadcast_sync("idle")
            return

        # YouTube: "play <playlist>" / "play my <playlist> playlist"
'''))

# 3. PTT voice dispatch: email confirm block before PTT YouTube block.
edits.append(("PTT voice dispatch: email confirm",
'''        _speak(local_media_reply)
        hud_server.broadcast_sync("idle")
        print("[voice] PTT local-media-ack complete", flush=True)
        return

    # YouTube: "play <playlist>" / "put on my <playlist> playlist"
''',
'''        _speak(local_media_reply)
        hud_server.broadcast_sync("idle")
        print("[voice] PTT local-media-ack complete", flush=True)
        return

    # Email send/cancel confirm -- only fires when there's a pending draft
    # and the text is a recognizable yes/no. See email_client's docstring.
    email_confirm_reply = try_handle_email_confirm_command(transcript)
    if email_confirm_reply is not None:
        _push_history("user", transcript, modality="voice")
        _push_history("assistant", email_confirm_reply, modality="voice")
        _broadcast_exchange(transcript, email_confirm_reply)
        _speak(email_confirm_reply)
        hud_server.broadcast_sync("idle")
        print("[voice] PTT email-confirm-ack complete", flush=True)
        return

    # YouTube: "play <playlist>" / "put on my <playlist> playlist"
'''))

# 4. Wake-word voice dispatch: email confirm block before wake YouTube block.
edits.append(("wake voice dispatch: email confirm",
'''        _speak(local_media_reply)
        hud_server.broadcast_sync("idle")
        return True

    # YouTube: "play <playlist>" / "put on my <playlist> playlist"
''',
'''        _speak(local_media_reply)
        hud_server.broadcast_sync("idle")
        return True

    # Email send/cancel confirm -- only fires when there's a pending draft
    # and the text is a recognizable yes/no. See email_client's docstring.
    email_confirm_reply = try_handle_email_confirm_command(transcript)
    if email_confirm_reply is not None:
        _push_history("user", transcript, modality="voice")
        _push_history("assistant", email_confirm_reply, modality="voice")
        _broadcast_exchange(transcript, email_confirm_reply)
        _speak(email_confirm_reply)
        hud_server.broadcast_sync("idle")
        return True

    # YouTube: "play <playlist>" / "put on my <playlist> playlist"
'''))

# 5. New tool schemas, inserted after WALLET_TOOL_NAMES definition.
NEW_SCHEMAS = '''WALLET_TOOL_NAMES = set(WALLET_TOOL_SCHEMAS.keys())

# ─── COMPUTATION / NOTIFICATION / EMAIL TOOLS (2026-09-02) ─────────────
# Same "give the model a real tool instead of trusting its output" pattern
# as weather.py/stocks.py, extended to arbitrary computation, plus a push-
# notification channel and a confirm-gated email surface. See code_exec.py,
# notify.py, and email_client.py module docstrings for the full rationale
# and safety design of each. email_send is deliberately NOT a tool here --
# see email_client.py's docstring for why sending only happens via the
# deterministic try_handle_email_confirm_command gate, never an LLM call.
RUN_PYTHON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run a short Python snippet and return its stdout/stderr. "
            "Use this for ANY arithmetic, unit conversion, date math, or "
            "other checkable computation instead of computing it yourself "
            "-- you are unreliable at multi-digit arithmetic, this tool "
            "is not. Also useful for quick data-munging (parsing, "
            "counting, sorting). Print the answer -- only stdout is "
            "returned. Runs sandboxed with a short timeout; file "
            "deletion, subprocess/os.system, and network calls are "
            "blocked and will be refused."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to run. Must print() its result.",
                },
            },
            "required": ["code"],
        },
    },
}

NOTIFY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "notify_me",
        "description": (
            "Push a notification to Ed's phone via ntfy. Use when Ed "
            "explicitly asks to be notified/texted/pinged/alerted about "
            "something ('text my phone that...', 'notify me when you're "
            "done', 'send that to my phone'). Do not use this as a "
            "substitute for a normal spoken reply -- only when Ed asks "
            "for a notification specifically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short notification title."},
                "message": {"type": "string", "description": "Notification body text."},
            },
            "required": ["title", "message"],
        },
    },
}

EMAIL_CHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "email_check",
        "description": (
            "Check Ed's inbox. Use when he asks 'do I have any new "
            "emails', 'check my email', 'any unread messages', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many to list. Default 5, max 25."},
                "unread_only": {"type": "boolean", "description": "True to list only unread."},
            },
        },
    },
}

EMAIL_DRAFT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "email_draft",
        "description": (
            "Draft an email. This ONLY drafts -- it never sends. After "
            "calling this, tell Ed what you drafted and that he needs to "
            "say 'send it' to actually send it, or 'cancel' to drop it. "
            "There is no way for you to send an email directly; sending "
            "only happens if Ed explicitly confirms in a later turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient name or email address. A saved contact "
                        "name works too (e.g. 'John'); an unresolvable "
                        "name will fail -- ask Ed for the address."
                    ),
                },
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

EXTRA_TOOL_SCHEMAS = {
    "run_python":  RUN_PYTHON_SCHEMA,
    "notify_me":   NOTIFY_TOOL_SCHEMA,
    "email_check": EMAIL_CHECK_SCHEMA,
    "email_draft": EMAIL_DRAFT_SCHEMA,
}
EXTRA_TOOL_NAMES = set(EXTRA_TOOL_SCHEMAS.keys())

'''
edits.append(("new tool schemas",
"WALLET_TOOL_NAMES = set(WALLET_TOOL_SCHEMAS.keys())\n\n",
NEW_SCHEMAS))

# 6. Extend _TOOL_DOCS_FOR_PROMPT to include the new schemas.
edits.append(("extend _TOOL_DOCS_FOR_PROMPT",
'''_TOOL_DOCS_FOR_PROMPT = _render_tools_for_prompt(
    [GREP_TOOL_SCHEMA, *WALLET_TOOL_SCHEMAS.values()])''',
'''_TOOL_DOCS_FOR_PROMPT = _render_tools_for_prompt(
    [GREP_TOOL_SCHEMA, *WALLET_TOOL_SCHEMAS.values(), *EXTRA_TOOL_SCHEMAS.values()])'''))

# 7. Extend known_tools set used by the loose-parse fallback.
edits.append(("extend known_tools",
'    known_tools = {"grep_source"} | WALLET_TOOL_NAMES\n',
'    known_tools = {"grep_source"} | WALLET_TOOL_NAMES | EXTRA_TOOL_NAMES\n'))

# 8. Extend the live tool-dispatch elif chain in _ollama_chat's loop.
edits.append(("extend tool dispatch loop",
'''            elif name in WALLET_TOOL_NAMES:
                result = _wallet_dispatch(name, args)
                safe_args = {k: ("<redacted>" if k == "pin" else v)
                             for k, v in args.items()}
                print(f"[chloe]   ollama-tool {name}({safe_args})"
                      f" → {len(result)} chars", flush=True)
            else:
                result = f"unknown tool: {name}"
                print(f"[chloe]   ollama-tool {name!r} requested but not implemented", flush=True)
''',
'''            elif name in WALLET_TOOL_NAMES:
                result = _wallet_dispatch(name, args)
                safe_args = {k: ("<redacted>" if k == "pin" else v)
                             for k, v in args.items()}
                print(f"[chloe]   ollama-tool {name}({safe_args})"
                      f" → {len(result)} chars", flush=True)
            elif name in EXTRA_TOOL_NAMES:
                result = _extra_tool_dispatch(name, args)
                print(f"[chloe]   ollama-tool {name}({args}) → {len(result)} chars", flush=True)
            else:
                result = f"unknown tool: {name}"
                print(f"[chloe]   ollama-tool {name!r} requested but not implemented", flush=True)
'''))

# 9. New _extra_tool_dispatch function, placed right after _wallet_dispatch ends.
edits.append(("add _extra_tool_dispatch",
'''        return f"unknown wallet tool: {name}"
    except Exception as e:
        traceback.print_exc()
        return f"Wallet error: {type(e).__name__}: {e}"


def _redact_pin_in_args_str(args_str, name):''',
'''        return f"unknown wallet tool: {name}"
    except Exception as e:
        traceback.print_exc()
        return f"Wallet error: {type(e).__name__}: {e}"


def _extra_tool_dispatch(name: str, args: dict) -> str:
    """Route run_python / notify_me / email_check / email_draft tool
    calls. Mirrors _wallet_dispatch's shape (lazy-import the backing
    module so a missing dependency degrades to an honest error instead
    of blocking all of Chloe). email_send is deliberately absent -- see
    email_client.py's docstring."""
    if not isinstance(args, dict):
        args = {}
    try:
        if name == "run_python":
            import code_exec
            code = args.get("code") or ""
            if not code.strip():
                return "run_python error: no code given."
            return code_exec.run_python_tool(code)

        if name == "notify_me":
            import notify
            title = str(args.get("title") or "Chloe").strip()
            message = str(args.get("message") or "").strip()
            if not message:
                return "notify_me error: no message given."
            ok = notify.send_ntfy(title, message, tags="robot")
            return ("Notification sent." if ok else
                    "Couldn't send a notification -- CHLOE_NTFY_TOPIC "
                    "may be unset or the ntfy request failed.")

        if name == "email_check":
            import email_client
            n = args.get("n")
            if isinstance(n, str) and n.strip().isdigit():
                n = int(n.strip())
            if not isinstance(n, int) or n < 1:
                n = 5
            unread_only = bool(args.get("unread_only"))
            return email_client.email_check_tool(n=n, unread_only=unread_only)

        if name == "email_draft":
            import email_client
            to = str(args.get("to") or "").strip()
            subject = str(args.get("subject") or "").strip()
            body = str(args.get("body") or "").strip()
            if not to:
                return "email_draft error: no recipient given."
            return email_client.email_draft_tool(to, subject, body)

        return f"unknown tool: {name}"
    except Exception as e:
        traceback.print_exc()
        return f"{name} error: {type(e).__name__}: {e}"


def _redact_pin_in_args_str(args_str, name):'''))

# 10. Push notification on a successful LLM-tool wallet_send.
edits.append(("notify on LLM wallet_send success",
'''            try:
                wg.record_send(int(r.get("amount_sat") or check_amount),
                               r.get("payment_hash"))
            except Exception:
                pass
            return json.dumps({
                "ok":            True,
                "amount_sat":    r.get("amount_sat"),
                "fees_sat":      r.get("fees_sat"),
                "status":        r.get("status"),
                "payment_hash":  r.get("payment_hash"),
            })''',
'''            try:
                wg.record_send(int(r.get("amount_sat") or check_amount),
                               r.get("payment_hash"))
            except Exception:
                pass
            try:
                import notify as _notify
                _notify.send_ntfy(
                    "Chloe: Lightning payment sent",
                    f"{r.get('amount_sat') or check_amount} sats sent "
                    f"(fee {r.get('fees_sat', 0)} sats). Status: {r.get('status')}.",
                    tags="zap")
            except Exception:
                pass
            return json.dumps({
                "ok":            True,
                "amount_sat":    r.get("amount_sat"),
                "fees_sat":      r.get("fees_sat"),
                "status":        r.get("status"),
                "payment_hash":  r.get("payment_hash"),
            })'''))

# 11. Push notification on a successful HUD/PWA wallet_send.
edits.append(("notify on HUD wallet_send success",
'''    try:
        wg.record_send(int(r.get("amount_sat") or check_amount),
                       r.get("payment_hash"))
    except Exception:
        pass

    await _ws_broadcast({
        "type":         "wallet_send_result",
        "ok":           True,''',
'''    try:
        wg.record_send(int(r.get("amount_sat") or check_amount),
                       r.get("payment_hash"))
    except Exception:
        pass
    try:
        import notify as _notify
        _notify.send_ntfy(
            "Chloe: Lightning payment sent",
            f"{r.get('amount_sat') or check_amount} sats sent "
            f"(fee {r.get('fees_sat', 0)} sats). Status: {r.get('status')}.",
            tags="zap")
    except Exception:
        pass

    await _ws_broadcast({
        "type":         "wallet_send_result",
        "ok":           True,'''))

failures = []
for label, old, new in edits:
    n = text.count(old)
    if n != 1:
        failures.append((label, n))
        continue
    text = text.replace(old, new, 1)

if failures:
    print("FAILED edits (label, occurrence_count):")
    for f in failures:
        print("  ", f)
    sys.exit(1)

open(PATH, "w", encoding="utf-8").write(text)
print(f"Applied {len(edits)} edits successfully.")
