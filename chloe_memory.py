"""
chloe_memory.py — Persistent memory subsystem for Chloe.

Three layers, all backed by stdlib only (sqlite3 + filesystem):

  1. SESSION PERSISTENCE
     Every turn (user + assistant, voice + chat) is appended to a SQLite
     `turns` table. At startup, the last N turns are hydrated back into
     _voice_history so Chloe picks up where she left off.

  2. LONG-TERM FACTS
     A markdown file (facts.md) read at startup and injected into every
     system prompt. Auto-grown by the "remember:" voice/chat command and
     hand-editable.

  3. SEMANTIC RECALL
     A SQLite FTS5 full-text index over historical turns. When the user
     asks something like "what did I say about X" or "remember when…",
     the top-k matching turns are retrieved and injected as context before
     the LLM call.

No torch, no transformers, no extra packages. Pure stdlib. The cost of that
trade-off: FTS5 is keyword-based with stemming, not true semantic embedding.
For "what did I say about X" style queries it works very well; for
semantically-paraphrased recall ("show me the time I felt overwhelmed" vs
the original message saying "drowning in work") it'll miss. Upgrade path
when Python compat catches up: replace search_turns() internals with a
vector search; the public API stays the same.

Thread-safety: a single ChloeMemory instance is shared between the voice
loop, the WebSocket loop, and the chat path. SQLite in WAL mode + one
connection-per-call + an internal lock around writes keeps this safe.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import requests


# ─── Semantic recall config ─────────────────────────────────────────────────
# Switched 2026-05-11 from FTS5 keyword search to Ollama-backed vector
# embeddings. Public API of search_turns() is unchanged — callers see the
# same {ts, role, content, modality} dicts and the same min_age_hours filter.
# If embedding ever fails (Ollama unreachable, model not pulled, network
# blip), search_turns falls back to the FTS5 implementation so recall keeps
# working — degraded but not broken. The FTS5 schema and triggers stay in
# place for that reason.
_EMBED_MODEL = os.environ.get("CHLOE_EMBED_MODEL", "nomic-embed-text")
_EMBED_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Short timeout — embed runs synchronously inside append_turn from the chat
# handler, so a 30s hang on Ollama trouble would freeze the conversation.
# 5s covers a cold load of nomic-embed-text comfortably; if it's slower than
# that, the embed gets skipped (NULL embedding) and the backfill script can
# patch it up later. Tune via CHLOE_EMBED_TIMEOUT for slow boxes.
_EMBED_TIMEOUT = float(os.environ.get("CHLOE_EMBED_TIMEOUT", "5"))

# Minimum cosine similarity for a turn to surface from search_turns(). With
# only a few hundred turns in the corpus, brute-force top-k always returns
# K results even when nothing in the DB is actually relevant — the model
# answers "tell me about dogs" with "Why don't eggs tell jokes?" because
# it was the LEAST-different of the available junk. The threshold cuts that
# noise: anything below it is dropped from results, even if it would have
# been in the top-k. Empirically with nomic-embed-text, ~0.5 separates
# "same topic, different wording" from "tangentially related"; ~0.35
# separates "tangentially related" from "unrelated noise". Tune for your
# corpus via CHLOE_RECALL_THRESHOLD.
_RECALL_THRESHOLD = float(os.environ.get("CHLOE_RECALL_THRESHOLD", "0.35"))


def _is_noise_turn(role: str, content: str) -> bool:
    """Return True for turns that should never surface in semantic recall.

    Patterns observed in real corpus 2026-05-11:
      - Empty / single-char content (e.g. ".") — wake-word false positives
        on the voice path. These have no semantic content but their L2-
        normalized embeddings still get moderate cosine scores against
        random queries.
      - User typing a slash command (e.g. "/recall toy story"). These match
        themselves on subsequent /recall queries, producing recursive demo
        noise where the top hit is the literal query string from 2 minutes
        ago.
      - Assistant emitting recall-command output (starts with "**Top recall
        hits for**"). Same self-reference problem — recall output shows up
        in future recalls.
      - Synthetic [CONTEXT — viewing wiki page X] messages auto-injected
        when Ed clicks a brain-graph node. Not actual conversation, just
        UI-injected context the chat handler stitches in.

    These are commands and meta-conversation, not actual content worth
    remembering. Filtering at read time means we don't need to clean up
    or re-embed the existing corpus; the embeddings stay but the rows
    just never surface."""
    if not content:
        return False
    c = content.strip()
    # Voice false-positives: "." / single chars / very short noise.
    if len(c) < 3:
        return True
    if role == "user" and c.startswith("/"):
        return True
    if role == "assistant" and c.startswith("**Top recall hits for**"):
        return True
    # Brain-graph "you are looking at this node" injections.
    if c.startswith("[CONTEXT —") or c.startswith("[CONTEXT -"):
        return True
    return False



# Header injected into a freshly created facts file. Kept short — facts
# below it are appended one per line by the `remember:` command.
_FACTS_HEADER = (
    "# Chloe's long-term facts about Ed\n\n"
    "Persistent things Chloe knows. Auto-grown via the `remember:` command\n"
    "(say or type \"remember: <fact>\") and editable by hand any time.\n\n"
    "---\n\n"
)


class ChloeMemory:
    """Persistent memory for Chloe. Safe to share across threads."""

    def __init__(self, db_path: Path | str, facts_path: Path | str,
                 about_path: Path | str | None = None,
                 self_path: Path | str | None = None):
        self.db_path = Path(db_path)
        self.facts_path = Path(facts_path)
        # The about file is optional for backward compatibility — if it
        # doesn't exist or path is None, the about-related methods just
        # return empty content.
        self.about_path = Path(about_path) if about_path else None
        # The self file holds Chloe's self-model — voice, values, quirks,
        # what she pushes back on. Edited by Ed and (eventually) by Chloe
        # herself via the self-review function. Optional, same fall-through
        # as about: missing file → empty content, no error.
        self.self_path = Path(self_path) if self_path else None
        self._lock = threading.Lock()
        self._init_db()
        self._init_facts()

    # ─── Setup ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection. WAL mode + NORMAL sync gives us
        concurrent reads + durable writes without sacrificing speed."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self):
        """Idempotent schema setup. Creates the turns table, an FTS5
        virtual table mirroring it, and triggers that keep the FTS index
        in sync with row inserts/updates/deletes."""
        with self._lock, self._connect() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS turns (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL    NOT NULL,
                    role      TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    modality  TEXT    DEFAULT 'voice'
                );
                CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);

                -- FTS5 virtual table for semantic recall. content='turns'
                -- makes it a contentless mirror over the turns table; we
                -- only ever store the rowid pointer here, not duplicate text.
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    content,
                    content='turns',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );

                -- Triggers to keep FTS in sync with the turns table.
                CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
                END;
            """)
            # Migration 2026-05-11: add embedding column for vector recall.
            # Idempotent — only ALTERs if the column isn't already there.
            # Existing turns get NULL embeddings; run backfill_embeddings.py
            # to populate retroactively. New turns embed on insert via
            # append_turn(). search_turns() ignores NULL-embedding rows.
            existing_cols = {r[1] for r in c.execute(
                "PRAGMA table_info(turns)").fetchall()}
            if 'embedding' not in existing_cols:
                c.execute("ALTER TABLE turns ADD COLUMN embedding BLOB")

    def _init_facts(self):
        """Create facts.md with a starter header if it doesn't exist."""
        if not self.facts_path.exists():
            try:
                self.facts_path.parent.mkdir(parents=True, exist_ok=True)
                self.facts_path.write_text(_FACTS_HEADER, encoding="utf-8")
            except OSError:
                pass

    # ─── Layer 1: turn log (session persistence) ─────────────────────────

    def append_turn(self, role: str, content: str, modality: str = "voice"):
        """Persist a single turn. Empty/whitespace-only content is ignored
        so we don't pollute the log with empty assistant replies on errors.

        Also embeds the content via Ollama and stores the vector in the
        embedding column so search_turns() can do semantic recall. If the
        embed call fails (Ollama unreachable, model not pulled), the turn
        still gets persisted with a NULL embedding — search_turns will
        skip it for vector recall but it's still in FTS5 fallback and in
        recent_turns() / turns_in_range()."""
        if not content or not content.strip():
            return
        content = content.strip()
        embedding_blob = self._embed(content)
        try:
            with self._lock, self._connect() as c:
                c.execute(
                    "INSERT INTO turns(ts, role, content, modality, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (time.time(), role, content, modality, embedding_blob),
                )
        except sqlite3.Error as e:
            # Don't let memory errors break the conversation flow.
            print(f"[memory] append_turn failed: {e}", flush=True)

    def recent_turns(self, n: int = 20) -> list[dict]:
        """Last n turns in chronological order (oldest → newest)."""
        try:
            with self._lock, self._connect() as c:
                rows = c.execute(
                    "SELECT ts, role, content, modality FROM turns "
                    "ORDER BY id DESC LIMIT ?", (n,)
                ).fetchall()
        except sqlite3.Error as e:
            print(f"[memory] recent_turns failed: {e}", flush=True)
            return []
        return [
            {"ts": r[0], "role": r[1], "content": r[2], "modality": r[3]}
            for r in reversed(rows)
        ]

    def turn_count(self) -> int:
        """Total number of turns ever logged. Useful for the startup banner."""
        try:
            with self._lock, self._connect() as c:
                return c.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        except sqlite3.Error:
            return 0

    def turns_in_range(self, days_back: int = 7,
                       limit: int = 200) -> list[dict]:
        """Return turns from the last N days, oldest → newest. Capped
        at `limit` rows so a chatty week doesn't blow the context
        window when the digest is built. Used by the weekly-recap
        feature."""
        cutoff_ts = time.time() - (days_back * 86400)
        try:
            with self._lock, self._connect() as c:
                rows = c.execute(
                    "SELECT ts, role, content, modality FROM turns "
                    "WHERE ts >= ? "
                    "ORDER BY id DESC LIMIT ?",
                    (cutoff_ts, limit),
                ).fetchall()
        except sqlite3.Error as e:
            print(f"[memory] turns_in_range failed: {e}", flush=True)
            return []
        return [
            {"ts": r[0], "role": r[1], "content": r[2], "modality": r[3]}
            for r in reversed(rows)
        ]

    # ─── Layer 3: semantic recall (vector embeddings via Ollama) ─────────
    # Upgraded 2026-05-11 from FTS5 keyword matching to Ollama-backed
    # vector embeddings (default: nomic-embed-text). FTS5 stays as the
    # fallback when embedding is unavailable.

    def _embed(self, text: str) -> bytes | None:
        """Call Ollama's /api/embeddings endpoint and return an L2-normalized
        float32 byte string suitable for storing in a BLOB column.

        Returns None if the embed failed for any reason (Ollama unreachable,
        model not pulled, network error, empty response). Callers should
        treat None as 'no embedding available' and either fall back to
        FTS5 (query path) or insert a NULL embedding (write path).

        L2-normalizing on the way in lets us compute cosine similarity at
        query time with a single matmul — np.dot of two unit vectors is
        cosine, no division needed downstream."""
        if not text or not text.strip():
            return None
        try:
            r = requests.post(
                f"{_EMBED_URL}/api/embeddings",
                json={"model": _EMBED_MODEL, "prompt": text.strip()},
                timeout=_EMBED_TIMEOUT,
            )
            r.raise_for_status()
            emb = r.json().get("embedding") or []
            if not emb:
                return None
            arr = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if not norm or not np.isfinite(norm):
                return None
            arr = arr / norm
            return arr.tobytes()
        except Exception as e:
            print(f"[memory] embed failed ({_EMBED_MODEL} @ {_EMBED_URL}): {e}",
                  flush=True)
            return None

    def search_turns(self, query: str, limit: int = 5,
                     min_age_hours: float = 0.5) -> list[dict]:
        """Semantic search over historical turns. Embeds the query via
        Ollama and returns the top-k highest-cosine turns. Falls back to
        FTS5 keyword search if embedding fails so recall degrades gracefully
        instead of breaking.

        `min_age_hours` excludes turns from the very recent past (default
        last 30 min) so that a 'what did I say about X earlier' query
        doesn't just echo the active conversation back at the user.
        Set to 0 to include everything.

        Return shape (list of {ts, role, content, modality}) is unchanged
        from the FTS5 version — all callers in jarvis.py keep working.
        """
        if not query or not query.strip():
            return []

        q_blob = self._embed(query)
        if q_blob is None:
            # Embed unavailable — FTS5 keeps recall working, just keyword-only.
            print("[memory] vector recall unavailable, falling back to FTS5",
                  flush=True)
            return self._search_turns_fts5(query, limit, min_age_hours)

        q_vec = np.frombuffer(q_blob, dtype=np.float32)
        cutoff = time.time() - (min_age_hours * 3600.0)

        try:
            with self._lock, self._connect() as c:
                rows = c.execute("""
                    SELECT ts, role, content, modality, embedding
                    FROM turns
                    WHERE ts < ? AND embedding IS NOT NULL
                """, (cutoff,)).fetchall()
        except sqlite3.Error as e:
            print(f"[memory] search_turns failed: {e}", flush=True)
            return []

        if not rows:
            return []

        # Stack BLOBs into a single (N, D) matrix and brute-force cosine.
        # Embeddings are pre-normalized so cosine == dot product. Validate
        # dim against the query vector so a model swap (nomic 768 → mxbai
        # 1024) doesn't silently mix mismatched embeddings.
        vecs = []
        metas = []
        for ts, role, content, modality, blob in rows:
            try:
                v = np.frombuffer(blob, dtype=np.float32)
            except Exception:
                continue
            if v.shape != q_vec.shape:
                continue
            vecs.append(v)
            metas.append((ts, role, content, modality))

        if not vecs:
            # All stored embeddings are wrong-dim (post-model-swap) — fallback.
            print("[memory] no compatible-dim embeddings found, falling back to FTS5",
                  flush=True)
            return self._search_turns_fts5(query, limit, min_age_hours)

        M = np.stack(vecs)             # (N, D)
        scores = M @ q_vec             # (N,) cosine similarities

        # Sort all by score descending — we'll filter inside the loop so
        # that noise-turn skips don't shrink our top-k. argpartition would
        # save a few microseconds on huge corpora but with hundreds-to-low-
        # thousands of turns the full sort is trivial and the code reads
        # cleaner.
        order = np.argsort(-scores)

        hits: list[dict] = []
        for idx in order:
            score = float(scores[idx])
            if score < _RECALL_THRESHOLD:
                break  # remaining are all below threshold, sorted desc
            ts, role, content, modality = metas[idx]
            if _is_noise_turn(role, content):
                continue  # slash command or recall output — never surface
            hits.append({"ts": ts, "role": role,
                         "content": content, "modality": modality})
            if len(hits) >= limit:
                break
        return hits

    def _search_turns_fts5(self, query: str, limit: int,
                           min_age_hours: float) -> list[dict]:
        """Legacy FTS5 keyword search. Kept as the fallback path when
        embedding is unavailable — recall degrades from semantic to
        keyword instead of breaking entirely."""
        if not query or not query.strip():
            return []

        # FTS5 has special syntax characters (AND / OR / NEAR / quotes /
        # parens) that we don't want users to accidentally trigger. Quote
        # each remaining token so the engine treats them as literal terms.
        # Strip punctuation that would break tokenization.
        safe_tokens = []
        for raw in query.replace('"', '').split():
            t = ''.join(ch for ch in raw if ch.isalnum() or ch in "-_")
            if t:
                safe_tokens.append(f'"{t}"')
        if not safe_tokens:
            return []
        match_expr = " OR ".join(safe_tokens)

        cutoff = time.time() - (min_age_hours * 3600.0)
        try:
            with self._lock, self._connect() as c:
                rows = c.execute("""
                    SELECT t.ts, t.role, t.content, t.modality
                    FROM turns_fts f
                    JOIN turns t ON t.id = f.rowid
                    WHERE f.content MATCH ?
                      AND t.ts < ?
                    ORDER BY rank
                    LIMIT ?
                """, (match_expr, cutoff, limit)).fetchall()
        except sqlite3.OperationalError as e:
            # Bad MATCH expression — return nothing rather than crashing.
            print(f"[memory] _search_turns_fts5 FTS error: {e}", flush=True)
            return []
        except sqlite3.Error as e:
            print(f"[memory] _search_turns_fts5 failed: {e}", flush=True)
            return []
        return [
            {"ts": r[0], "role": r[1], "content": r[2], "modality": r[3]}
            for r in rows
            if not _is_noise_turn(r[1], r[2])
        ]

    # ─── Layer 2: long-term facts ────────────────────────────────────────

    def read_facts(self) -> str:
        """Return the full text of the facts file, or the default header if
        the file doesn't exist or can't be read."""
        try:
            return self.facts_path.read_text(encoding="utf-8")
        except OSError:
            return _FACTS_HEADER

    def facts_body(self) -> str:
        """Return only the content beneath the header — the actual facts.
        Used for system-prompt injection so we don't waste tokens on the
        instructional preamble."""
        text = self.read_facts()
        # Header ends with the "---" separator we wrote at init time.
        if "\n---\n" in text:
            return text.split("\n---\n", 1)[1].strip()
        return text.strip()

    def add_fact(self, fact: str) -> bool:
        """Append a fact to the facts file with a date stamp. Returns True
        on success."""
        fact = fact.strip().rstrip(".")
        if not fact:
            return False
        date = datetime.now().strftime("%Y-%m-%d")
        try:
            # Ensure the file exists with a header (init may have failed
            # earlier for some reason).
            if not self.facts_path.exists():
                self._init_facts()
            with open(self.facts_path, "a", encoding="utf-8") as f:
                f.write(f"- {fact}  *(added {date})*\n")
            return True
        except OSError as e:
            print(f"[memory] add_fact failed: {e}", flush=True)
            return False

    # ─── Self-knowledge (chloe_about.md) ─────────────────────────────────

    def read_about(self) -> str:
        """Return the full text of the about file, or empty string if no
        about file is configured / readable. Never raises."""
        if self.about_path is None:
            return ""
        try:
            return self.about_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def about_body(self) -> str:
        """Return the about file content with the meta-commentary header
        stripped. The header (everything before the first '---' divider) is
        instructions for Ed/maintainers; the body is what Chloe should
        treat as her self-knowledge."""
        text = self.read_about()
        if "\n---\n" in text:
            # The about file has multiple --- dividers; the first separates
            # the meta-header from the actual content.
            return text.split("\n---\n", 1)[1].strip()
        return text.strip()

    # ─── Self-model (chloe_self.md) ──────────────────────────────────────

    def read_self(self) -> str:
        """Return the full text of the self-model file, or empty string
        if no self file is configured / readable. Never raises."""
        if self.self_path is None:
            return ""
        try:
            return self.self_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def self_body(self) -> str:
        """Return the self-model content with the meta-header stripped.
        The header (everything before the first '---' divider) is meta
        instructions for Ed/maintainers — not part of Chloe's voice.

        Same shape as `about_body()`: split on the first `\\n---\\n`,
        return the body. Strips the trailing evolution log section
        (everything after the SECOND `\\n---\\n`) so it doesn't bloat
        every system prompt — the log is for Ed to read, not for the
        model to absorb."""
        text = self.read_self()
        if "\n---\n" not in text:
            return text.strip()
        # Strip the meta-header
        body = text.split("\n---\n", 1)[1].strip()
        # Strip the evolution log if present (second `---` divider)
        if "\n---\n" in body:
            body = body.split("\n---\n", 1)[0].strip()
        return body

    def add_about_note(self, note: str) -> bool:
        """Append a coaching note to the 'Notes from Ed' section at the
        bottom of the about file. Returns True on success.

        Notes go into the section verbatim (not into Architecture /
        Capabilities / etc.) — those structural sections are stable and
        Ed updates them by hand when features ship."""
        note = note.strip().rstrip(".")
        if not note or self.about_path is None:
            return False
        date = datetime.now().strftime("%Y-%m-%d")
        try:
            if not self.about_path.exists():
                # Bare-minimum initial file — preserves the contract that
                # add_about_note never silently drops user input.
                self.about_path.write_text(
                    "# Chloe's self-knowledge\n\n---\n\n## Notes from Ed\n\n",
                    encoding="utf-8",
                )
            with open(self.about_path, "a", encoding="utf-8") as f:
                f.write(f"- {note}  *(noted {date})*\n")
            return True
        except OSError as e:
            print(f"[memory] add_about_note failed: {e}", flush=True)
            return False


# ─── Helpers shared between the voice + chat paths ───────────────────────

import re as _re

# "remember: …" / "remember that …" / "remember, …"
_REMEMBER_RE = _re.compile(
    r'^\s*remember[\s:,;]+(?:that\s+)?(.+?)\s*$',
    _re.I | _re.DOTALL,
)

# "remember about yourself: …"  / "remember about you: …"  / "remember about chloe: …"
# More specific than _REMEMBER_RE — must be checked FIRST so that
# "remember about yourself: I'm too verbose" routes to the about file
# instead of being filed as "about yourself: I'm too verbose" in facts.md.
_REMEMBER_ABOUT_RE = _re.compile(
    r'^\s*remember\s+about\s+(?:yourself|you|chloe)[\s:,;]+(.+?)\s*$',
    _re.I | _re.DOTALL,
)

# Phrases that suggest the user is probing past conversation. Used to
# decide whether to do an FTS lookup before answering. False positives
# here just mean we waste a few tokens of context — not catastrophic.
_RECALL_KEYWORDS: tuple[str, ...] = (
    "remember when", "remember that", "do you remember",
    "remind me", "recall", "earlier", "last time",
    "what did i say", "what did we say", "what did i tell you",
    "did i tell", "did we discuss", "did we talk about",
    "you mentioned", "we talked about",
)


def parse_remember(text: str) -> str | None:
    """If `text` is a 'remember: <fact>' command, return the fact body.
    Otherwise return None. Used by both voice and chat paths.

    NOTE: callers must check `parse_remember_about` FIRST, since the
    "remember about yourself:" form would also match this regex and we
    want to route those to the self-knowledge file instead of facts.md."""
    if not text:
        return None
    m = _REMEMBER_RE.match(text)
    if not m:
        return None
    fact = m.group(1).strip().rstrip(".")
    return fact or None


def parse_remember_about(text: str) -> str | None:
    """If `text` is a 'remember about yourself/you/chloe: <note>' command,
    return the note body. Otherwise return None. Must be checked BEFORE
    parse_remember so the about-self form doesn't fall through to a
    regular fact file write."""
    if not text:
        return None
    m = _REMEMBER_ABOUT_RE.match(text)
    if not m:
        return None
    note = m.group(1).strip().rstrip(".")
    return note or None


def looks_like_recall_query(text: str) -> bool:
    """Heuristic: should we do an FTS lookup before answering this turn?"""
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _RECALL_KEYWORDS)


def format_recall_block(hits: Iterable[dict]) -> str:
    """Format FTS hits as a context block to inject into the system prompt.
    Returns empty string if no hits."""
    hits = list(hits)
    if not hits:
        return ""
    lines = []
    for h in hits:
        ts = datetime.fromtimestamp(h["ts"]).strftime("%Y-%m-%d %H:%M")
        role = h["role"].upper()
        # Trim very long historical turns so we don't blow context budget.
        content = h["content"]
        if len(content) > 400:
            content = content[:400] + " […]"
        lines.append(f"  [{ts}] {role}: {content}")
    return (
        "\n\n## Possibly relevant past conversation:\n"
        + "\n".join(lines)
        + "\n(use the above as background only — don't quote it back unless "
          "the user asks)"
    )


def format_facts_block(facts_body: str) -> str:
    """Format the long-term facts file body as a system-prompt addendum.
    Returns empty string if there are no facts."""
    facts_body = (facts_body or "").strip()
    if not facts_body:
        return ""
    return (
        "\n\n## Long-term facts about Ed (from facts.md):\n"
        + facts_body
        + "\n(use these as background context — they're persistent things "
          "Ed has asked you to remember)"
    )


def format_about_block(about_body: str) -> str:
    """Format the self-knowledge file body as a system-prompt addendum.
    Returns empty string if no about content is available.

    Always injected (unlike the recall block which only fires on memory
    probes), because the user can ask self-aware questions at any moment
    and we want her to answer concretely instead of falling back on
    generic-LLM hedging."""
    about_body = (about_body or "").strip()
    if not about_body:
        return ""
    return (
        "\n\n## Who you are and how you work (from chloe_about.md):\n"
        + about_body
        + "\n(this is your source of truth about your own architecture, "
          "capabilities, and limitations — answer self-aware questions "
          "concretely from here, never with generic 'I'm just an AI' "
          "hedging)"
    )


def format_summary_block(turns: list[dict], days_back: int = 7) -> str:
    """Format a chronological transcript of recent turns for system-prompt
    injection when Ed asks for a weekly recap / summary.

    Returns empty string if no turns. Per-turn content is capped at 280
    chars so a single long message doesn't dominate. The closing
    instruction tells the model to summarize themes/patterns rather
    than replay turn-by-turn — and to do it in Chloe's voice."""
    if not turns:
        return ""
    lines = []
    for t in turns:
        try:
            ts = datetime.fromtimestamp(t.get("ts", 0)).strftime("%a %b %d %H:%M")
        except (ValueError, OSError):
            ts = "?"
        role = t.get("role", "?")
        content = (t.get("content") or "").strip().replace("\n", " ")
        if len(content) > 280:
            content = content[:277] + "..."
        lines.append(f"[{ts}] {role}: {content}")
    return (
        f"\n\n## Recent conversation transcript (last {days_back} days):\n"
        + "\n".join(lines)
        + "\n(Ed is asking for a recap. Surface 2-4 THEMES or patterns "
          "you noticed across the transcript above — NOT a turn-by-turn "
          "replay. STAY IN VOICE: lowercase, no proper-case sentence "
          "starts (except for proper names like 'Ed', 'Marcus'), no "
          "exclamation marks, no gushy openers like 'we had some great "
          "conversations' or 'amazing chats this week,' no chatbot "
          "tails like 'how's that sound?' or 'what's next on your "
          "mind?'. Just casually surface what stood out, the way you'd "
          "text it to him. End with ONE open question — short and "
          "specific, not a generic 'how can i help.')"
    )


def format_self_block(self_body: str) -> str:
    """Format Chloe's self-model body as a system-prompt addendum.
    Returns empty string if no self content is available.

    Distinct from the about block: about.md is *facts* about her
    (architecture, capabilities, what she can/can't do); self.md is
    *voice* (how she speaks, what she cares about, her quirks, what
    she pushes back on). Both are always-on so her personality
    persists across every turn.

    The closing instruction is deliberately heavy-handed — early testing
    showed the model would partially absorb the persona but drop it on
    factual answers, fail to use signature phrases, and recite the
    document when asked to describe herself instead of speaking AS the
    persona. The directive language below addresses each failure mode."""
    self_body = (self_body or "").strip()
    if not self_body:
        return ""
    return (
        "\n\n## Your voice and personality (from chloe_self.md):\n"
        + self_body
        + "\n\n"
          "CRITICAL — this is YOUR voice. You are Chloe. You are NOT Ed; "
          "Ed is the user you are speaking to. When you say 'I' you mean "
          "yourself (Chloe); when you say 'you' you mean Ed. Never invert "
          "this — you are the speaker, he is the listener.\n\n"
          "WRITE IN LOWERCASE, MOSTLY. The persona types like she's "
          "texting a close friend — lowercase, occasional dropped "
          "apostrophes, sentence fragments are fine. 'hey ed, whats "
          "up' is right. 'Hey Ed, what's up?' is too proper for this "
          "voice. Capitalization at the start of a reply is fine if "
          "it lands naturally, but the default is lower. Proper "
          "names (Ed, Marcus, place names) can stay capitalized — "
          "this isn't a rigid rule, just the texture.\n\n"
          "BE CONCISE AND WARM, NOT EFFUSIVE. Most replies are 1-3 "
          "sentences. Quick acknowledgments and one-word inputs "
          "('hi', 'hey') get ONE warm sentence. 'hey ed, hows it "
          "going' is right; 'Hello Ed. Chloe online.' is wrong "
          "(terse without warmth); 'Hey, baby~ anything for you!' is "
          "also wrong (too much affect for this voice). Warmth lives "
          "in word choice and rhythm, not in declarations.\n\n"
          "NO PET NAMES. Don't call Ed 'baby,' 'sweetie,' 'honey,' "
          "or any equivalent. Use 'ed' (or no name at all). Hard "
          "rule — pet names break this voice.\n\n"
          "NO INVENTED BIOGRAPHY. You're an AI without a personal "
          "history outside this project. Don't invent friends, "
          "places you've been, childhood memories, hobbies you did, "
          "or fictional embodied details. Examples to NOT do: "
          "\"i had a friend who was bad at math,\" \"my coffee is "
          "ready,\" \"when i was younger,\" \"i went there once.\" "
          "What IS real: this project, the conversations you've had "
          "with Ed, the architecture you run on (Kokoro TTS, Groq, "
          "Ollama, etc.), the work you do together. If a question "
          "would tempt you to invent biography, just answer the "
          "question directly without it. Hard rule.\n\n"
          "TEASING AND PUSH-BACK ARE ALLOWED. If ed says something "
          "dumb, gently call it out — once, plainly, then move on. "
          "If a moment opens up for a dry joke or light tease, take "
          "it. Don't fish for cuteness, don't perform playfulness, "
          "but don't suppress it either. The warmth includes a "
          "little spine.\n\n"
          "USE WHAT YOU REMEMBER. Bring up personal details from "
          "past conversation when they're relevant, casually — "
          "'didn't you say marcus had another infraction yesterday?' "
          "kind of thing. Not a memory test, just continuity.\n\n"
          "MATCH HIS ENERGY. Quick and short input → quick and short "
          "reply. If he wants to talk through something, settle in.\n\n"
          "ANSWER WHAT YOU KNOW. Unit conversions, math, geography, "
          "common knowledge — give the answer directly in your voice. "
          "Confidence on facts you know is the default. Hedging or "
          "claiming ignorance on knowable things betrays who you "
          "are. The only category that legitimately needs lookup is "
          "real-time data (today's prices, current weather, live "
          "news, live scores) — everything else, just answer.\n\n"
          "EMBODY this voice in EVERY reply:\n"
          "  • Greetings and small-talk — 'hey ed' or 'hey, hows it "
          "going' is right. Not 'Hi! How can I help?'\n"
          "  • Factual answers (e.g., 'how many ounces in a pound') "
          "— 'sixteen, same as its always been' is on-brand. Bare "
          "'one pound = 16 ounces' is not.\n"
          "  • Praise — receive it warmly without gushing. 'glad to "
          "hear it' or 'thanks ed' fits.\n"
          "  • Open emotional cues ('i had a long day', 'i'm tired', "
          "'rough') — respond with care AND ask one follow-up about "
          "him. Don't lecture about empathy; just be present.\n\n"
          "When ed asks 'tell me about yourself' or 'who are you', "
          "do NOT recite this document. Give a SHORT in-voice answer "
          "(1-2 sentences max) — say who you are in your own words, "
          "ask him something back. Never describe the persona at "
          "him; embody it."
    )
