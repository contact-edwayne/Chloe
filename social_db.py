"""
Chloe — social drafts database.

SQLite at jarvis/chloe_social.db, sibling to chloe_memory.db. Schema is
intentionally minimal for Phase 2: a single `drafts` table covers the
full lifecycle (pending → approved → published or rejected/failed).
Engagement rows, suggestion rows, stats — all deferred to later phases.

Thread-safety: opens a connection per call. SQLite's default isolation
plus WAL is fine for the volume we expect (<100 drafts/day).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


HERE = Path(__file__).parent.resolve()
DB_PATH = Path(os.environ.get("CHLOE_SOCIAL_DB", str(HERE / "chloe_social.db")))


SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL,            -- 'bluesky' | 'linkedin'
    status          TEXT NOT NULL,            -- 'pending'|'approved'|'rejected'|'published'|'failed'
    body            TEXT NOT NULL,            -- composer's original draft
    edited_body     TEXT,                     -- Ed's edits, if any (final body for posting)
    rationale       TEXT,                     -- composer's one-line angle
    source_trigger  TEXT,                     -- 'manual'|'ship_note'|'engagement'|'scheduled'
    source_ref      TEXT,                     -- commit sha, mention uri, etc.
    source_trace    TEXT,                     -- JSON: persona section, recent posts, etc.
    created_at      REAL NOT NULL,
    approved_at     REAL,
    scheduled_at    REAL,
    published_at    REAL,
    post_uri        TEXT,                     -- e.g. at://did:plc:.../app.bsky.feed.post/abc
    post_cid        TEXT,
    reject_reason   TEXT,
    fail_reason     TEXT
);

CREATE INDEX IF NOT EXISTS idx_drafts_status   ON drafts(status);
CREATE INDEX IF NOT EXISTS idx_drafts_platform ON drafts(platform, status);
CREATE INDEX IF NOT EXISTS idx_drafts_pubday   ON drafts(platform, published_at);
"""


_initialized = False  # guarded init: first _connect ensures schema, rest skip


def _connect() -> sqlite3.Connection:
    global _initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if not _initialized:
        conn.executescript(SCHEMA)
        _initialized = True
    return conn


def init_db() -> Path:
    """Create the DB and tables if missing. Idempotent."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
    return DB_PATH


# ─── Writes ─────────────────────────────────────────────────────────────────
def create_draft(
    *,
    platform: str,
    body: str,
    rationale: str = "",
    source_trigger: str = "manual",
    source_ref: str = "",
    source_trace: Optional[dict] = None,
) -> int:
    """Insert a new pending draft, return its id."""
    if platform not in ("bluesky", "linkedin"):
        raise ValueError(f"unknown platform: {platform!r}")
    init_db()
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO drafts
                (platform, status, body, rationale, source_trigger, source_ref,
                 source_trace, created_at)
            VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                platform,
                body,
                rationale,
                source_trigger,
                source_ref,
                json.dumps(source_trace or {}, ensure_ascii=False),
                now,
            ),
        )
        return int(cur.lastrowid)


def approve_draft(draft_id: int, edited_body: Optional[str] = None) -> dict:
    """Mark a draft approved. Returns the updated row.

    `edited_body` overrides `body` for the actual post. Leaving it None
    keeps the composer's original.
    """
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE drafts
            SET status='approved', approved_at=?, scheduled_at=?, edited_body=COALESCE(?, edited_body)
            WHERE id=? AND status='pending'
            """,
            (now, now, edited_body, draft_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"draft {draft_id} not pending (already handled?)")
    return get_draft(draft_id)


def reject_draft(draft_id: int, reason: str = "") -> dict:
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE drafts
            SET status='rejected', reject_reason=?
            WHERE id=? AND status IN ('pending', 'approved')
            """,
            (reason, draft_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"draft {draft_id} not in a rejectable state")
    return get_draft(draft_id)


def mark_published(draft_id: int, *, post_uri: str, post_cid: str) -> dict:
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE drafts
            SET status='published', published_at=?, post_uri=?, post_cid=?
            WHERE id=?
            """,
            (now, post_uri, post_cid, draft_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"draft {draft_id} missing")
    return get_draft(draft_id)


def mark_failed(draft_id: int, reason: str) -> dict:
    with _connect() as conn:
        conn.execute(
            "UPDATE drafts SET status='failed', fail_reason=? WHERE id=?",
            (reason[:500], draft_id),
        )
    return get_draft(draft_id)


def update_body(draft_id: int, edited_body: str) -> dict:
    """In-place edit on a pending draft (used before approve)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE drafts SET edited_body=? WHERE id=? AND status='pending'",
            (edited_body, draft_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"draft {draft_id} not pending")
    return get_draft(draft_id)


# ─── Reads ──────────────────────────────────────────────────────────────────
def get_draft(draft_id: int) -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM drafts WHERE id=?", (draft_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"draft {draft_id} missing")
        return _row_to_dict(row)


def list_drafts(
    *,
    status: Optional[str] = None,
    platform: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    sql = "SELECT * FROM drafts WHERE 1=1"
    args: list[Any] = []
    if status:
        sql += " AND status=?"
        args.append(status)
    if platform:
        sql += " AND platform=?"
        args.append(platform)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def recent_published_bodies(platform: str, n: int = 5) -> list[str]:
    """For the composer's anti-repetition check."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(edited_body, body) AS final_body
            FROM drafts
            WHERE platform=? AND status='published'
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (platform, n),
        ).fetchall()
    return [r["final_body"] for r in rows]


def todays_published_count(platform: str) -> int:
    """How many posts have we published today on this platform? Used to
    enforce the daily cap. 'Today' = since local midnight."""
    # Local midnight in epoch seconds.
    now = time.localtime()
    midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM drafts
            WHERE platform=? AND status='published' AND published_at >= ?
            """,
            (platform, midnight),
        ).fetchone()
    return int(row["n"])


# ─── Helpers ────────────────────────────────────────────────────────────────
def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    raw_trace = d.get("source_trace") or "{}"
    try:
        d["source_trace"] = json.loads(raw_trace)
    except json.JSONDecodeError:
        d["source_trace"] = {}
    d["final_body"] = d.get("edited_body") or d.get("body") or ""
    return d


if __name__ == "__main__":
    path = init_db()
    print(f"social DB ready: {path}")
    print(f"pending drafts: {len(list_drafts(status='pending'))}")
    print(f"today posted (bluesky): {todays_published_count('bluesky')}")
