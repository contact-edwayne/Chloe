"""Conversation sessions derived from the flat `turns` log in chloe_memory.db.

Sessions are NOT stored as rows — they're computed on the fly by grouping
consecutive turns whose time gap exceeds SESSION_GAP_MIN minutes. A session's
stable identity is its first turn's timestamp (start_ts). Auto-titles are
generated lazily by a caller-supplied `titler` callable (keeps this module
decoupled from the LLM stack) and cached in a `session_titles` table keyed by
start_ts.

Public API (all take the chloe_memory.db path):
    list_sessions(db_path, limit=30, gap_min=30)
        -> [ {start_ts, end_ts, n_turns, modalities, preview, title|None} ]  newest-first
    get_session(db_path, start_ts, gap_min=30)
        -> {start_ts, end_ts, turns:[{id,ts,role,content,modality}]} | None
    ensure_title(db_path, start_ts, titler, gap_min=30, force=False) -> str|None
        generate+cache a title (titler(transcript)->str); snippet fallback on failure.

v1 reads the whole turns table per call — fine for a personal assistant's
history; optimize to a ts-only boundary scan if it ever gets large.
"""

import sqlite3
import time

SESSION_GAP_MIN = 30          # minutes of quiet that starts a new session
_PREVIEW_LEN = 90
_TITLE_MAX = 60


def _connect(db_path):
    # The DB is already in WAL mode (set once by chloe_memory's init); don't
    # re-issue the pragma — it's a needless journal write that also errors on
    # some network/overlay mounts. Reads + the small title write inherit WAL.
    return sqlite3.connect(str(db_path), check_same_thread=False)


def _ensure_title_table(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_titles ("
        " start_ts REAL PRIMARY KEY, title TEXT NOT NULL, made_ts REAL NOT NULL)")


def _group(rows, gap_s):
    """rows: (id, ts, role, content, modality) ascending by ts.
    Yields sessions as lists of turn-dicts."""
    sess = []
    last_ts = None
    for tid, ts, role, content, modality in rows:
        if last_ts is not None and (ts - last_ts) > gap_s:
            if sess:
                yield sess
            sess = []
        sess.append({"id": tid, "ts": ts, "role": role,
                     "content": content, "modality": modality})
        last_ts = ts
    if sess:
        yield sess


def _all_turns(db_path):
    conn = _connect(db_path)
    try:
        return conn.execute(
            "SELECT id, ts, role, content, modality FROM turns ORDER BY ts ASC"
        ).fetchall()
    finally:
        conn.close()


def list_sessions(db_path, limit=30, gap_min=SESSION_GAP_MIN):
    """Newest-first session summaries. `title` is the cached title if present,
    else None (call ensure_title to fill the real one)."""
    gap_s = gap_min * 60
    rows = _all_turns(db_path)
    conn = _connect(db_path)
    try:
        _ensure_title_table(conn)
        titles = dict(conn.execute(
            "SELECT start_ts, title FROM session_titles").fetchall())
    finally:
        conn.close()

    out = []
    for s in _group(rows, gap_s):
        start_ts = s[0]["ts"]
        first_user = next((t["content"] for t in s if t["role"] == "user"),
                          s[0]["content"])
        preview = " ".join((first_user or "").split())[:_PREVIEW_LEN]
        out.append({
            "start_ts": start_ts,
            "end_ts": s[-1]["ts"],
            "n_turns": len(s),
            "modalities": sorted({t["modality"] for t in s}),
            "preview": preview,
            "title": titles.get(start_ts),
        })
    out.reverse()                       # newest first
    return out[:limit]


def get_session(db_path, start_ts, gap_min=SESSION_GAP_MIN, tol=1.0):
    """Return the session whose first turn ts matches start_ts (within tol s)."""
    gap_s = gap_min * 60
    target = float(start_ts)
    for s in _group(_all_turns(db_path), gap_s):
        if abs(s[0]["ts"] - target) <= tol:
            return {"start_ts": s[0]["ts"], "end_ts": s[-1]["ts"], "turns": s}
    return None


def ensure_title(db_path, start_ts, titler=None, gap_min=SESSION_GAP_MIN,
                 force=False):
    """Return a cached title, or generate one via `titler` (a
    callable(transcript_text)->str) and cache it. Falls back to a snippet of
    the first user message if titler is None or fails. Returns None only if the
    session can't be found."""
    start_ts = float(start_ts)
    if not force:
        conn = _connect(db_path)
        try:
            _ensure_title_table(conn)
            row = conn.execute(
                "SELECT title FROM session_titles WHERE start_ts=?",
                (start_ts,)).fetchone()
        finally:
            conn.close()
        if row:
            return row[0]

    sess = get_session(db_path, start_ts, gap_min)
    if not sess:
        return None
    turns = sess["turns"]

    title = None
    if titler:
        transcript = "\n".join(
            f"{t['role']}: {' '.join((t['content'] or '').split())[:200]}"
            for t in turns[:6])
        try:
            title = (titler(transcript) or "").strip().strip('"').strip("'").strip()
            title = " ".join(title.split())[:_TITLE_MAX] if title else None
        except Exception:
            title = None
    if not title:
        first_user = next((t["content"] for t in turns if t["role"] == "user"),
                          turns[0]["content"])
        title = " ".join((first_user or "untitled").split())[:48]

    conn = _connect(db_path)
    try:
        _ensure_title_table(conn)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO session_titles(start_ts, title, made_ts) "
                "VALUES (?,?,?)", (start_ts, title, time.time()))
    finally:
        conn.close()
    return title


def delete_session(db_path, start_ts, gap_min=SESSION_GAP_MIN):
    """Delete every turn in the session starting at start_ts, plus its cached
    title. The turns-table FTS triggers (set up by chloe_memory) keep the
    search index in sync on delete. Returns {ok, deleted}."""
    sess = get_session(db_path, start_ts, gap_min)
    if not sess:
        return {"ok": False, "deleted": 0, "error": "session not found"}
    ids = [t["id"] for t in sess["turns"] if t.get("id") is not None]
    conn = _connect(db_path)
    try:
        _ensure_title_table(conn)
        with conn:
            if ids:
                ph = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM turns WHERE id IN ({ph})", ids)
            conn.execute("DELETE FROM session_titles WHERE start_ts=?",
                         (float(start_ts),))
    finally:
        conn.close()
    return {"ok": True, "deleted": len(ids)}


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "chloe_memory.db"
    for s in list_sessions(db, limit=10):
        from datetime import datetime as _dt
        when = _dt.fromtimestamp(s["start_ts"]).strftime("%Y-%m-%d %H:%M")
        print(f"{when}  {s['n_turns']:>3} turns  {','.join(s['modalities']):<12} "
              f"| {s['title'] or s['preview']}")
