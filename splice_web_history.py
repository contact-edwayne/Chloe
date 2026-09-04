"""Splice script — add /web_history slash command to brain_wiring.py.

Pillar 1 task #5. Adds:
  1. `/web_history` handler that lists web_*.md pages from wiki/sources/.
     Supports `today`, `week`, `month` window args. Caps at 20 results.
  2. One line in the `/brain` help message.

Safety per chloe_editing_jarvis_py.md (applies to brain_wiring.py at its
current size): backup, ast.parse, tail-diff, line-delta bound. On any
failure, auto-restore from backup.
"""

from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "brain_wiring.py"
STAMP = datetime.now().strftime("%Y-%m-%d-webhistory")
BACKUP = ROOT / f"brain_wiring.py.bak.{STAMP}"

# ── Handler block inserted before the final `return None` ───────────────────
HANDLER = '''    if msg.startswith("/web_history") or msg == "/web_history":
        # List Brave search results persisted to wiki/sources/web_*.md.
        # Window args: today | week | month. Default: all-time, capped at 20.
        # Companion to the persist hook in jarvis._persist_brave_to_wiki —
        # together they make web lookups reviewable + non-volatile.
        arg = msg[len("/web_history"):].strip().lower()
        import datetime as _dt
        import re as _re_wh
        sources_dir = BRAIN.wiki_dir / "sources"
        if not sources_dir.exists():
            return "No web history yet - no `wiki/sources/` directory."
        pages = sorted(sources_dir.glob("web_*.md"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not pages:
            return "No web searches recorded yet."
        now = _dt.datetime.now()
        if arg in ("today", "1d"):
            cutoff = now.replace(hour=0, minute=0,
                                 second=0, microsecond=0).timestamp()
            pages = [p for p in pages if p.stat().st_mtime >= cutoff]
            window_label = "today"
        elif arg in ("week", "7d"):
            cutoff = (now - _dt.timedelta(days=7)).timestamp()
            pages = [p for p in pages if p.stat().st_mtime >= cutoff]
            window_label = "last 7 days"
        elif arg in ("month", "30d"):
            cutoff = (now - _dt.timedelta(days=30)).timestamp()
            pages = [p for p in pages if p.stat().st_mtime >= cutoff]
            window_label = "last 30 days"
        else:
            window_label = "all-time"
        if not pages:
            return f"No web searches in {window_label}."
        pages = pages[:20]
        out = [f"**Web search history** ({window_label}, "
               f"{len(pages)} of up to 20):\\n"]
        for i, p in enumerate(pages, 1):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")[:1500]
            except Exception:
                continue
            query = ""
            date = ""
            m = _re_wh.search(r"^query:\\s*['\\\"]?(.+?)['\\\"]?\\s*$",
                              text, _re_wh.M)
            if m:
                query = m.group(1).strip()
            m = _re_wh.search(r"^date:\\s*(.+)$", text, _re_wh.M)
            if m:
                date = m.group(1).strip()
            cite_m = _re_wh.search(r"^\\d+\\.\\s+\\[(.+?)\\]\\((.+?)\\)",
                                   text, _re_wh.M)
            cite = ""
            if cite_m:
                title = cite_m.group(1)[:50]
                cite = f" - [{title}]({cite_m.group(2)})"
            if not query:
                stem = p.stem.replace("web_", "")
                query = stem.rsplit("_", 1)[0].replace("_", " ")
            out.append(f"{i}. *{date}* - **{query}**{cite}")
        return "\\n".join(out)

'''

# Anchor: insert immediately before the terminal `return None` at end of
# try_handle_brain_command. Uniquely identified by the help-text closing
# line + the closing paren + the final return None.
HANDLER_ANCHOR_OLD = '''                "  `/brain`                 - this help message")

    return None
'''

HANDLER_ANCHOR_NEW = '''                "  `/brain`                 - this help message")

''' + HANDLER + '''    return None
'''

# ── Help-text addition: one line for /web_history in the /brain help ────────
HELP_OLD = '''                "  `/wiki <query>`          - semantic search over wiki pages\\n"
                "  `/wiki_write [--dry-run] <topic>` - research a topic via web "
                "search, write a wiki page, ingest it\\n"'''

HELP_NEW = '''                "  `/wiki <query>`          - semantic search over wiki pages\\n"
                "  `/web_history [today|week|month]` - list recent Brave search results\\n"
                "  `/wiki_write [--dry-run] <topic>` - research a topic via web "
                "search, write a wiki page, ingest it\\n"'''


def fail(msg: str, restore: bool = True) -> None:
    print(f"[splice] FAIL: {msg}", file=sys.stderr)
    if restore and BACKUP.exists():
        shutil.copy2(BACKUP, TARGET)
        print(f"[splice] restored {TARGET} from {BACKUP.name}",
              file=sys.stderr)
    sys.exit(1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        fail(f"{label}: anchor not found")
    if count > 1:
        fail(f"{label}: anchor matched {count} times (expected 1)")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        fail(f"target missing: {TARGET}", restore=False)
    shutil.copy2(TARGET, BACKUP)
    print(f"[splice] backup -> {BACKUP.name}")

    src = TARGET.read_text(encoding="utf-8")
    orig_lines = src.count("\n")

    new = src
    new = replace_once(new, HANDLER_ANCHOR_OLD, HANDLER_ANCHOR_NEW,
                       "web_history handler")
    new = replace_once(new, HELP_OLD, HELP_NEW, "/brain help text")

    new_lines = new.count("\n")
    delta = new_lines - orig_lines
    if not (50 <= delta <= 90):
        fail(f"line-delta {delta} outside [50, 90]")

    try:
        ast.parse(new)
    except SyntaxError as e:
        fail(f"ast.parse failed: {e}")

    # Tail-diff intentionally skipped: this splice lands at the very end of
    # try_handle_brain_command (~line 1111/1112), so the legitimate edit IS
    # inside the tail window. Instead verify the function's terminal `return
    # None` is preserved and the file still ends cleanly.
    bak_lines = src.splitlines()
    new_lines_list = new.splitlines()
    if not new_lines_list or new_lines_list[-1].strip() != "return None":
        fail("function-end check: last non-empty line is not `return None`")
    # All lines from the original backup must still be present in order
    # (insertion-only edit — nothing removed). Cheap LCS-free check: every
    # backup line must appear in new with monotonic indices.
    j = 0
    for bl in bak_lines:
        while j < len(new_lines_list) and new_lines_list[j] != bl:
            j += 1
        if j >= len(new_lines_list):
            fail(f"line preservation: backup line missing in new — {bl!r}")
        j += 1

    TARGET.write_text(new, encoding="utf-8")
    print(f"[splice] OK - {delta} lines added")
    print(f"[splice] try in chat: /web_history  or  /web_history week")


if __name__ == "__main__":
    main()
