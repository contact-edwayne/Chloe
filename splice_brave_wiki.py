"""Splice script — instrument Brave search to persist results as wiki/sources/web_*.md.

Pillar 1 of the memory-autopilot sprint (2026-05-17). Three Brave entry points
get instrumented:
  1. _brave_fallback_search (chat hedge fallback) — line ~597
  2. _brave_voice_synth (voice hedge fallback)    — line ~696
  3. /search slash command handler                 — line ~1265

A helper _persist_brave_to_wiki is inserted before _brave_fallback_search and
called from all three sites in a daemon thread (non-blocking).

Safety per chloe_editing_jarvis_py.md:
  - Backup jarvis.py first.
  - Each splice has a unique multi-line anchor; refuse if anchor not found.
  - ast.parse the result.
  - Tail-diff: last 50 lines of new vs backup must match.
  - Line-delta bound: total growth in [60, 120] lines (helper ~50 + 3 sites).
  - On any failure, restore from backup and exit non-zero.
"""

from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "jarvis.py"
STAMP = datetime.now().strftime("%Y-%m-%d-bravewiki")
BACKUP = ROOT / f"jarvis.py.bak.{STAMP}"

# ── Helper function to insert before `async def _brave_fallback_search` ─────
HELPER = '''def _persist_brave_to_wiki(query, reply, results, source_label="brave_search"):
    """Persist a Brave search result to wiki/sources/web_*.md for future recall.

    Called from all three Brave entry points (chat fallback, voice fallback,
    /search slash command). Runs in a daemon thread so the user-facing reply
    is never blocked on the write.

    The wiki_watcher re-embeds the new page within ~2s, so a later
    semantically-similar question can hit Chloe's own memory via
    `looks_like_wiki_query` retrieval instead of re-fetching Brave.

    Failure-mode: log and swallow — search functionality should not break
    just because the wiki write failed. All three call sites operate
    fire-and-forget.
    """
    import re as _re
    import threading as _t
    from datetime import datetime as _dt

    def _worker():
        try:
            if not query or not reply or not reply.strip():
                return
            try:
                from brain_wiring import BRAIN as _BRAIN
            except Exception as e:
                print(f"[brave→wiki] BRAIN unavailable: {e}", flush=True)
                return
            s = (query or "").lower().strip()
            s = _re.sub(r"[^a-z0-9-]+", "_", s)
            s = _re.sub(r"_+", "_", s).strip("_")
            if len(s) > 60:
                s = s[:60].rstrip("_")
            if not s:
                return
            date_s = _dt.now().strftime("%Y-%m-%d")
            ts = _dt.now().isoformat(timespec="seconds")
            slug = f"web_{s}_{date_s}"
            rel = f"wiki/sources/{slug}.md"
            cites = []
            urls = []
            for i, r in enumerate(results or [], 1):
                title = (r.get("title") or "").strip().replace("\\n", " ")
                url = (r.get("url") or "").strip()
                domain = (r.get("domain") or "").strip()
                if title and url:
                    cites.append(f"{i}. [{title}]({url}) — {domain}")
                elif url:
                    cites.append(f"{i}. {url}")
                if url:
                    urls.append(f"  - {url}")
            urls_block = "\\n".join(urls) if urls else "  []"
            cites_block = "\\n".join(cites) if cites else "_(no citations returned)_"
            body = (
                f"---\\n"
                f"type: source\\n"
                f"source_type: web_search\\n"
                f"query: {query!r}\\n"
                f"date: {date_s}\\n"
                f"generated_at: {ts}\\n"
                f"generated_via: {source_label}\\n"
                f"source_urls:\\n{urls_block}\\n"
                f"---\\n\\n"
                f"# Web search: {query}\\n\\n"
                f"_Synthesized from Brave search on {ts}._\\n\\n"
                f"{reply.strip()}\\n\\n"
                f"## Citations\\n\\n"
                f"{cites_block}\\n"
            )
            _BRAIN.write(rel, body)
            print(f"[brave→wiki] persisted {rel} ({len(body)} bytes)",
                  flush=True)
        except Exception as e:
            print(f"[brave→wiki] persist failed: {e}", flush=True)

    _t.Thread(target=_worker, daemon=True).start()


'''

# ── Anchor 1: insert HELPER before `async def _brave_fallback_search` ───────
HELPER_ANCHOR = "async def _brave_fallback_search(websocket, query, data):"

# ── Anchor 2: chat fallback site — insert call before final `done` send ─────
CHAT_OLD = """    await _ws_send(websocket, {
        "type": "sources",
        "items": [
            {
                "n": i + 1,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "domain": r.get("domain", ""),
            }
            for i, r in enumerate(results)
        ],
    })
    await _ws_send(websocket, {"type": "done"})
    return full_reply"""

CHAT_NEW = """    await _ws_send(websocket, {
        "type": "sources",
        "items": [
            {
                "n": i + 1,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "domain": r.get("domain", ""),
            }
            for i, r in enumerate(results)
        ],
    })
    await _ws_send(websocket, {"type": "done"})
    _persist_brave_to_wiki(query, full_reply, results, "brave_chat_fallback")
    return full_reply"""

# ── Anchor 3: voice synth site — wrap final return with persist ─────────────
VOICE_OLD = """        resp = _sync_groq.with_options(timeout=30.0).chat.completions.create(
            model=MODEL_TEXT,
            messages=[
                {"role": "system", "content": search_system},
                {"role": "user", "content": query},
            ],
            max_tokens=250,
            temperature=0.5,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[voice] brave-fallback synthesis error: {e}", flush=True)
        return \"\""""

VOICE_NEW = """        resp = _sync_groq.with_options(timeout=30.0).chat.completions.create(
            model=MODEL_TEXT,
            messages=[
                {"role": "system", "content": search_system},
                {"role": "user", "content": query},
            ],
            max_tokens=250,
            temperature=0.5,
        )
        _voice_reply = (resp.choices[0].message.content or "").strip()
        if _voice_reply:
            _persist_brave_to_wiki(query, _voice_reply, results, "brave_voice_fallback")
        return _voice_reply
    except Exception as e:
        print(f"[voice] brave-fallback synthesis error: {e}", flush=True)
        return \"\""""

# ── Anchor 4: /search slash command — insert persist after history push ─────
SLASH_OLD = """            await _ws_send(websocket, {"type": "done"})
            if full_search_reply.strip():
                _push_history("assistant", full_search_reply, modality="chat")
            # Speak the synthesized reply, but strip [N] citation markers"""

SLASH_NEW = """            await _ws_send(websocket, {"type": "done"})
            if full_search_reply.strip():
                _push_history("assistant", full_search_reply, modality="chat")
                _persist_brave_to_wiki(_q, full_search_reply, _results, "brave_slash_command")
            # Speak the synthesized reply, but strip [N] citation markers"""


def fail(msg: str, restore: bool = True) -> None:
    print(f"[splice] FAIL: {msg}", file=sys.stderr)
    if restore and BACKUP.exists():
        shutil.copy2(BACKUP, TARGET)
        print(f"[splice] restored {TARGET} from {BACKUP.name}", file=sys.stderr)
    sys.exit(1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        fail(f"{label}: anchor not found")
    if count > 1:
        fail(f"{label}: anchor matched {count} times (expected 1)")
    return text.replace(old, new, 1)


def insert_before(text: str, anchor: str, payload: str, label: str) -> str:
    count = text.count(anchor)
    if count == 0:
        fail(f"{label}: anchor not found")
    if count > 1:
        fail(f"{label}: anchor matched {count} times (expected 1)")
    idx = text.find(anchor)
    return text[:idx] + payload + text[idx:]


def main() -> None:
    if not TARGET.exists():
        fail(f"target missing: {TARGET}", restore=False)

    # Step 1: backup
    shutil.copy2(TARGET, BACKUP)
    print(f"[splice] backup -> {BACKUP.name}")

    src = TARGET.read_text(encoding="utf-8")
    orig_lines = src.count("\n")

    # Step 2: in-memory edits
    new = src
    new = insert_before(new, HELPER_ANCHOR, HELPER, "helper")
    new = replace_once(new, CHAT_OLD, CHAT_NEW, "chat-fallback site")
    new = replace_once(new, VOICE_OLD, VOICE_NEW, "voice-synth site")
    new = replace_once(new, SLASH_OLD, SLASH_NEW, "slash-command site")

    new_lines = new.count("\n")
    delta = new_lines - orig_lines
    if not (60 <= delta <= 130):
        fail(f"line-delta {delta} outside [60, 130]")

    # Step 3: ast.parse
    try:
        ast.parse(new)
    except SyntaxError as e:
        fail(f"ast.parse failed: {e}")

    # Step 4: tail-diff — last 50 lines of new must match last 50 of backup
    bak_tail = src.splitlines()[-50:]
    new_tail = new.splitlines()[-50:]
    if bak_tail != new_tail:
        fail("tail-diff: last 50 lines diverged (edits drifted to EOF)")

    # Step 5: commit
    TARGET.write_text(new, encoding="utf-8")
    print(f"[splice] OK — {delta} lines added")
    print(f"[splice] verify by tailing logs after next restart for "
          f"'[brave→wiki] persisted ...' lines")


if __name__ == "__main__":
    main()
