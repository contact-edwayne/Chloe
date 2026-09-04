"""
wiki_embedding.py — Semantic recall over Chloe's wiki pages.

Companion to chloe_memory.py. Same embedding model (nomic-embed-text via
Ollama), same normalization, same threshold filter — but the corpus is
`C:\\Chloe\\brain\\wiki\\**/*.md` instead of conversation turns.

Why this exists:
  - Before this module, semantic recall only worked on conversation history.
    The wiki was filesystem-only — `Brain.query()` (in C:\\Chloe\\brain.py)
    does keyword-style page selection via LLM, not vector cosine.
  - With this module, edits to a wiki page (from Obsidian, an external
    editor, or Chloe's own ingest pipeline) become semantically
    searchable as soon as the watcher upserts the row.

Design notes:
  - Standalone module (does not import chloe_memory or brain). Same DB
    file by default (`chloe_memory.db`) so a future cross-corpus search
    can join in one query, but the schema lives in its own `wiki_pages`
    table — no risk of corrupting the turns schema.
  - Hash-based idempotence: a re-upsert of an unchanged file is a hash
    compare + early return. The 2-second polling watcher can hammer this
    method without burning Ollama tokens.
  - Title and type are extracted from YAML frontmatter when present so
    search results are recruiter-legible ("Edward Wayne — entity") rather
    than just paths.

No torch, no transformers, no extra packages. Pure stdlib + numpy +
requests (both already in jarvis's venv).
"""

import hashlib
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import requests


# ─── Config (same env vars as chloe_memory.py so a single override works) ──
_EMBED_MODEL = os.environ.get("CHLOE_EMBED_MODEL", "nomic-embed-text")
_EMBED_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_EMBED_TIMEOUT = float(os.environ.get("CHLOE_EMBED_TIMEOUT", "10"))
# Keep the embedding model resident between calls (same knob as the
# chat model) so recall + wiki lookups don't pay a cold reload.
from ollama_keepalive import get_keep_alive as _get_ollama_keep_alive
_EMBED_KEEP_ALIVE = _get_ollama_keep_alive()

# Wiki search has a slightly higher threshold than turn recall because
# wiki pages are denser and well-formed. ~0.4 separates "actually about
# this topic" from "shares a few words by accident". Tune via env if a
# corpus shift makes results too noisy / too sparse.
_WIKI_THRESHOLD = float(os.environ.get("CHLOE_WIKI_THRESHOLD", "0.4"))

# Point-in-time staleness ceilings (2026-08-31). Two, not one: a market
# quote (SLV price) is stale in hours-to-days; a scheduled-release figure
# (CPI, Fed funds rate) is legitimately "current" for weeks between
# releases. Same env var names as jarvis.py's classifier so one override
# controls both the persist-time classification and the recall-time gate.
_QUOTE_STALENESS_DAYS = float(os.environ.get("CHLOE_QUOTE_STALENESS_DAYS", "3"))
_DATA_STALENESS_DAYS = float(os.environ.get("CHLOE_DATA_STALENESS_DAYS", "30"))

# Path-stem boost. When a query token appears literally in a page's path
# stem (e.g., 'ollama' → 'entities/ollama.md'), bump that page's score
# by this much per matching token. Why: cosine search punishes short
# pages — a 500-char bullseye gets out-scored by a 1200-char neighbor
# that just happens to mention the topic densely.
#
# Empirical sizing on Ed's real corpus (2026-05-12): for query "ollama",
# entities/ollama.md cosine = 0.41, concepts/hybrid_llm_router.md cosine
# = 0.49 — an 0.08 gap. A 0.08 boost merely TIES them; we need slightly
# more to actually win the tie. 0.10 gives a 0.02 margin and pushes the
# bullseye to #1 without overpowering legitimately better semantic
# matches on multi-word queries. This is the classic "title-match"
# heuristic that production retrieval systems (BM25-hybrid, ColBERT,
# etc.) layer on top of vector search. Disable via env=0.
_PATH_BOOST = float(os.environ.get("CHLOE_WIKI_PATH_BOOST", "0.10"))
# Cap the total boost — keeps a 4-word query like 'ollama qwen local
# model' from compounding into an unreasonable +0.40 swing on a page
# whose path stem happens to contain three of those tokens. With
# per-match = 0.10, the cap of 0.20 lets a 2-token path-bullseye
# definitively win (e.g. 'edward wayne' → entities/edward_wayne.md)
# but caps the runaway for ambiguous multi-token queries.
_PATH_BOOST_CAP = float(os.environ.get("CHLOE_WIKI_PATH_BOOST_CAP", "0.20"))

# Cap how much of each page goes into the embedding input. Nomic handles
# long inputs but the back of the page (Related sections, repeated
# wikilinks) dilutes the semantic signal. 2000 chars roughly = the
# frontmatter + TL;DR + first 1-2 body sections — the dense part.
_EMBED_CHAR_CAP = int(os.environ.get("CHLOE_WIKI_EMBED_CAP", "2000"))

# Default wiki root if the caller doesn't override.
_DEFAULT_WIKI_ROOT = Path(os.environ.get(
    "CHLOE_WIKI_ROOT", r"C:\Chloe\brain\wiki"))

# Default DB path — co-located with chloe_memory.db by default.
_DEFAULT_DB = Path(os.environ.get(
    "CHLOE_WIKI_DB",
    str(Path(__file__).resolve().parent / "chloe_memory.db")))


# ─── Frontmatter helpers ────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n?(.*)$', re.DOTALL)
_TITLE_RE = re.compile(r'^title:\s*(.+?)\s*$', re.MULTILINE)
_TYPE_RE = re.compile(r'^type:\s*(.+?)\s*$', re.MULTILINE)
_POINT_IN_TIME_KIND_RE = re.compile(r'^point_in_time_kind:\s*(\S+)', re.MULTILINE)
_SUPERSEDED_RE = re.compile(r'^superseded:\s*true\b', re.MULTILINE | re.IGNORECASE)
_GENERATED_AT_RE = re.compile(r'^generated_at:\s*(\S+)', re.MULTILINE)
_DATE_RE = re.compile(r'^date:\s*(\S+)', re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[str, str, str]:
    """Return (title, type, body). Empty strings if not present.

    Doesn't try to be a full YAML parser — title and type are the only
    fields we need and they're single-line strings in Chloe's brain.py
    schema."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "", "", text
    fm, body = m.group(1), m.group(2)
    title = ""
    typ = ""
    tm = _TITLE_RE.search(fm)
    if tm:
        title = tm.group(1).strip().strip('"').strip("'")
    tym = _TYPE_RE.search(fm)
    if tym:
        typ = tym.group(1).strip().strip('"').strip("'")
    return title, typ, body.strip()


def _parse_point_in_time_meta(text: str) -> tuple[str, bool, float]:
    """Return (point_in_time_kind, superseded, asof_epoch) from
    frontmatter. kind is ''|'quote'|'data'; superseded is a bool;
    asof_epoch is Unix epoch seconds parsed from frontmatter's
    generated_at (preferred, has a timestamp) or date (fallback, day
    granularity), or 0.0 if neither parses.

    asof_epoch is NOT the same as the file's mtime, deliberately: mtime is
    "when was this file last written," which changes on ANY rewrite --
    including this project's own point-in-time marking backfill, an
    Obsidian save, or any future metadata-only edit. A staleness gate
    keyed on mtime would silently reset to "brand new" the moment
    something touches the file for an unrelated reason. asof_epoch is
    "when was this content actually true," which only changes if the
    page's own generated_at/date frontmatter changes.

    Kept separate from _parse_frontmatter() rather than extending its
    return tuple — that function's 3-tuple contract is exercised directly
    by test_wiki_embedding.py, no reason to disturb it for fields most
    pages don't have."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "", False, 0.0
    fm = m.group(1)
    km = _POINT_IN_TIME_KIND_RE.search(fm)
    kind = km.group(1).strip().strip('"').strip("'") if km else ""
    superseded = bool(_SUPERSEDED_RE.search(fm))
    asof_epoch = 0.0
    ga_m = _GENERATED_AT_RE.search(fm)
    raw_ts = ga_m.group(1).strip().strip('"').strip("'") if ga_m else None
    if not raw_ts:
        d_m = _DATE_RE.search(fm)
        raw_ts = d_m.group(1).strip().strip('"').strip("'") if d_m else None
    if raw_ts:
        try:
            asof_epoch = datetime.fromisoformat(raw_ts).timestamp()
        except ValueError:
            try:
                asof_epoch = datetime.strptime(raw_ts, "%Y-%m-%d").timestamp()
            except ValueError:
                asof_epoch = 0.0
    return kind, superseded, asof_epoch


def _title_from_path(rel_path: str) -> str:
    """Fallback title derived from filename when frontmatter has none."""
    stem = Path(rel_path).stem
    return stem.replace('_', ' ').replace('-', ' ').strip().title() or stem


def _normalize_rel(rel_path: str) -> str:
    """Force POSIX-style slashes so DB keys are stable across OSes and
    don't need backslash-escaping in logs."""
    return rel_path.replace('\\', '/').lstrip('/')


# ─── The store ──────────────────────────────────────────────────────────────

class WikiEmbeddingStore:
    """Vector store for wiki pages. Safe to share across threads.

    Lifecycle:
      store = WikiEmbeddingStore(wiki_root, db_path)
      store.upsert_page('entities/edward_wayne.md')   # idempotent
      hits = store.search('who is edward')             # top-k cosine
      store.delete_page('entities/edward_wayne.md')    # explicit removal
      store.purge_missing()                            # GC orphan rows

    All write paths use a lock + connection-per-op (WAL mode) so the
    watcher thread + the chat handler (future /wiki) don't fight."""

    def __init__(self, wiki_root: Path | str = _DEFAULT_WIKI_ROOT,
                 db_path: Path | str = _DEFAULT_DB):
        self.wiki_root = Path(wiki_root).resolve()
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ─── DB plumbing ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self):
        """Idempotent schema setup. The table is namespaced (`wiki_pages`)
        so it can coexist with chloe_memory's `turns` table in the same DB
        file."""
        with self._lock, self._connect() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS wiki_pages (
                    path          TEXT PRIMARY KEY,
                    title         TEXT,
                    type          TEXT,
                    body          TEXT NOT NULL,
                    content_hash  TEXT NOT NULL,
                    mtime         REAL NOT NULL,
                    embedding     BLOB,
                    updated_at    REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_wiki_mtime
                    ON wiki_pages(mtime);
            """)
            # Migration (2026-08-31): point-in-time staleness + supersession
            # gating for quote/data pages. ALTER TABLE ADD COLUMN has no
            # IF NOT EXISTS in SQLite -- check PRAGMA table_info first so
            # re-running this on an already-migrated DB is a no-op.
            existing_cols = {row[1] for row in
                             c.execute("PRAGMA table_info(wiki_pages)").fetchall()}
            if "point_in_time_kind" not in existing_cols:
                c.execute(
                    "ALTER TABLE wiki_pages ADD COLUMN point_in_time_kind TEXT DEFAULT ''")
            if "superseded" not in existing_cols:
                c.execute(
                    "ALTER TABLE wiki_pages ADD COLUMN superseded INTEGER DEFAULT 0")
            if "point_in_time_asof" not in existing_cols:
                # Deliberately NOT sourced from mtime -- see
                # _parse_point_in_time_meta's docstring. 0.0 = unknown/never
                # parsed; treated as "not point-in-time" by search()'s gate
                # (age can't be computed, so nothing to filter on).
                c.execute(
                    "ALTER TABLE wiki_pages ADD COLUMN point_in_time_asof REAL DEFAULT 0")

    # ─── Embedding ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> bytes | None:
        """L2-normalized float32 BLOB or None on failure. Mirrors
        ChloeMemory._embed exactly so the two stores stay vector-compatible
        if/when we want cross-corpus search."""
        # Shared cache: a query already embedded for conversation recall this
        # turn (same model+text) is served from memory, no second round-trip.
        import chloe_embed
        return chloe_embed.embed(
            text, model=_EMBED_MODEL, url=_EMBED_URL,
            timeout=_EMBED_TIMEOUT, keep_alive=_EMBED_KEEP_ALIVE, tag="wiki-embed")

    def _build_embed_input(self, title: str, body: str) -> str:
        """Compose the string we actually send to nomic. Title first so it
        carries weight in the pooled embedding; body capped to avoid
        dilution. Including the title (or filename-derived stem) means a
        bare-title query like 'kokoro' still ranks the kokoro page on top
        even when the body never repeats the word."""
        head = title.strip() if title else ""
        body = body.strip()
        if head:
            joined = f"{head}\n\n{body}"
        else:
            joined = body
        if len(joined) > _EMBED_CHAR_CAP:
            joined = joined[:_EMBED_CHAR_CAP]
        return joined

    # ─── Core ops ───────────────────────────────────────────────────────

    def upsert_page(self, rel_path: str) -> str:
        """Read the file, embed if its content hash changed, write the row.

        Returns one of:
          'inserted'   — new page, embedded + stored
          'updated'    — page existed, content changed, re-embedded
          'unchanged'  — hash matched, no-op (this is the common case
                          when the watcher polls quickly)
          'missing'    — file doesn't exist on disk
          'embed-fail' — Ollama errored; row is stored with NULL embedding
                          so future polls retry (we'd rather have a
                          stub row than silently drop the page)
        """
        rel = _normalize_rel(rel_path)
        full = (self.wiki_root / rel).resolve()
        # Path safety: make sure the resolved file is still under wiki_root.
        # Without this, a symlink/relative-path attack could clobber rows
        # outside the wiki.
        try:
            full.relative_to(self.wiki_root)
        except ValueError:
            return 'missing'
        if not full.exists() or not full.is_file():
            return 'missing'

        try:
            text = full.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return 'missing'

        title, typ, body = _parse_frontmatter(text)
        pit_kind, superseded, pit_asof = _parse_point_in_time_meta(text)
        if not title:
            title = _title_from_path(rel)
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        mtime = full.stat().st_mtime
        now = time.time()

        # Check whether the existing row already matches this hash. If so,
        # touch nothing — burning an embed call on an unchanged page is
        # the dominant cost we want to avoid.
        with self._lock, self._connect() as c:
            row = c.execute(
                "SELECT content_hash, embedding FROM wiki_pages WHERE path = ?",
                (rel,)
            ).fetchone()
            if row and row[0] == content_hash and row[1] is not None:
                # Bump mtime so the orphan-detector doesn't get confused
                # by stale rows, but skip the expensive embed.
                c.execute(
                    "UPDATE wiki_pages SET mtime = ?, updated_at = ? WHERE path = ?",
                    (mtime, now, rel),
                )
                return 'unchanged'

        embed_input = self._build_embed_input(title, body)
        blob = self._embed(embed_input)
        status = 'embed-fail' if blob is None else ('updated' if row else 'inserted')

        with self._lock, self._connect() as c:
            c.execute("""
                INSERT INTO wiki_pages
                    (path, title, type, body, content_hash, mtime, embedding, updated_at,
                     point_in_time_kind, superseded, point_in_time_asof)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title = excluded.title,
                    type = excluded.type,
                    body = excluded.body,
                    content_hash = excluded.content_hash,
                    mtime = excluded.mtime,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at,
                    point_in_time_kind = excluded.point_in_time_kind,
                    superseded = excluded.superseded,
                    point_in_time_asof = excluded.point_in_time_asof
            """, (rel, title, typ, body, content_hash, mtime, blob, now,
                  pit_kind, int(superseded), pit_asof))
        return status

    def delete_page(self, rel_path: str) -> bool:
        """Drop the row for rel_path. Returns True if a row was removed."""
        rel = _normalize_rel(rel_path)
        with self._lock, self._connect() as c:
            cur = c.execute("DELETE FROM wiki_pages WHERE path = ?", (rel,))
            return cur.rowcount > 0

    def purge_missing(self) -> int:
        """Remove rows for files that no longer exist on disk. Returns the
        number purged. Cheap GC pass — call this after a rebase, mass
        rename, or just on watcher startup."""
        with self._lock, self._connect() as c:
            paths = [r[0] for r in c.execute(
                "SELECT path FROM wiki_pages").fetchall()]
        purged = 0
        for rel in paths:
            full = (self.wiki_root / rel).resolve()
            try:
                full.relative_to(self.wiki_root)
            except ValueError:
                if self.delete_page(rel):
                    purged += 1
                continue
            if not full.exists():
                if self.delete_page(rel):
                    purged += 1
        return purged

    def backfill_all(self) -> dict:
        """Walk the wiki dir and upsert every .md file. Returns counters.

        Idempotent: re-running after the first backfill is cheap (every
        page hits the 'unchanged' fast-path). The watcher calls this on
        startup."""
        counters = {'inserted': 0, 'updated': 0, 'unchanged': 0,
                    'embed-fail': 0, 'missing': 0}
        for full in self.wiki_root.rglob('*.md'):
            try:
                rel = str(full.resolve().relative_to(self.wiki_root))
            except ValueError:
                continue
            status = self.upsert_page(_normalize_rel(rel))
            counters[status] = counters.get(status, 0) + 1
        return counters

    # ─── Search ─────────────────────────────────────────────────────────

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        """Lowercase alphanumeric tokens with len >= 3. Used for the path
        boost. Short tokens ('to', 'is') are dropped because they cause
        spurious matches against unrelated paths."""
        out = []
        cur = []
        for ch in query.lower():
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    tok = ''.join(cur)
                    if len(tok) >= 3:
                        out.append(tok)
                    cur = []
        if cur:
            tok = ''.join(cur)
            if len(tok) >= 3:
                out.append(tok)
        return out

    @staticmethod
    def _path_boost(path: str, query_tokens: list[str],
                    per_match: float, cap: float) -> float:
        """Per-token boost when a query token is a substring of the path
        stem (the basename without extension). Returns a value in
        [0, cap]. Substring (not full-token) matching lets 'qwen' boost
        'qwen25' and 'wiki' boost 'wiki_watcher' — usually what the user
        wants."""
        if not query_tokens or per_match <= 0:
            return 0.0
        stem = Path(path).stem.lower()
        hits = sum(1 for t in query_tokens if t in stem)
        return min(hits * per_match, cap)

    @staticmethod
    def _collapse_near_duplicate_point_in_time(query, metas, vecs, q_vec):
        """Among candidates that are BOTH point-in-time (pit_kind set,
        age known) AND about the same subject as `query`
        (wiki_dedup.same_subject), keep only the single BEST MATCH for
        this query (highest raw cosine) and drop the rest. Ed,
        2026-09-03: repeated "what's SLV's price" surfaced a pile of
        near-duplicate answers ahead of the current one -- none of them
        individually old enough to fail the age-ceiling gate above (all
        < CHLOE_QUOTE_STALENESS_DAYS), and
        supersede_prior_point_in_time_page's stricter write-time bar
        (CHLOE_PIT_SUPERSEDE_THRESHOLD=0.93 raw cosine) hadn't collapsed
        them into each other either, since near-duplicate SLV answers
        phrased differently don't always clear that threshold.

        Tiebreak is SCORE, not age -- tried age first (keep the single
        newest per cluster) and confirmed live it can backfire: the
        newest page in a subject cluster is sometimes phrased around a
        different angle (e.g. "macro trend" rather than a direct price
        statement) and scores far lower on THIS query than the near-
        duplicates it would have discarded, making results worse, not
        better. Highest-cosine tends to prefer the freshest DIRECTLY-
        relevant answer anyway in the common case (a same-day quote
        phrased close to the question usually scores well), without ever
        discarding a strong match in favor of a weak one just because
        it's newer.

        This is a SEPARATE, looser mechanism than write-time supersede:
        it runs at READ time against whatever point-in-time candidates
        matched the CURRENT query and only has to agree on subject, not
        near-identical wording -- the goal here is "don't show 6 near-
        duplicate SLV quotes", not "decide whether page A supersedes
        page B on disk" (that decision is intentionally more
        conservative and untouched by this).

        Clusters candidates by PAIRWISE same_subject agreement between
        the candidates THEMSELVES (not just each vs. the query) -- a
        query that happens to name two point-in-time subjects at once
        (e.g. "SLV and gold price") must not collapse an SLV quote into
        a gold quote just because both independently matched the query.
        Entries with an unparseable age (age_days is None) are never
        collapsed either way -- can't safely judge freshness for them."""
        import wiki_dedup as _wiki_dedup

        candidates = [i for i, m in enumerate(metas)
                     if m[4] and m[5] is not None
                     and _wiki_dedup.same_subject(query, m[1] or "")]
        if len(candidates) < 2:
            return metas, vecs

        cosine = {i: float(vecs[i] @ q_vec) for i in candidates}

        drop: set[int] = set()
        for pit_kind in {metas[i][4] for i in candidates}:
            same_kind = [i for i in candidates if metas[i][4] == pit_kind]
            clusters: list[list[int]] = []
            for i in same_kind:
                for cluster in clusters:
                    if _wiki_dedup.same_subject(metas[i][1] or "",
                                                metas[cluster[0]][1] or ""):
                        cluster.append(i)
                        break
                else:
                    clusters.append([i])
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                best = max(cluster, key=lambda i: cosine[i])
                drop.update(i for i in cluster if i != best)

        if not drop:
            return metas, vecs
        kept = [i for i in range(len(metas)) if i not in drop]
        return [metas[i] for i in kept], [vecs[i] for i in kept]

    def search(self, query: str, limit: int = 5,
               threshold: float | None = None,
               apply_staleness_gate: bool = True) -> list[dict]:
        """Top-k semantic hits for `query`. Returns list of dicts with
        keys: path, title, type, snippet, score, cosine, path_boost,
        point_in_time_kind (None if not a point-in-time page), age_days
        (None unless point_in_time_kind is set).

        Returns empty list (not exception) on embed failure or empty
        corpus — callers can render 'no hits' instead of an error.

        Final score = cosine(query, page) + path_stem_boost(query, page).
        The boost is small (default +0.08/match, capped +0.15) so
        semantic ranking is preserved on multi-word queries; close calls
        where the bullseye page is short get tipped in the right
        direction. Disable via CHLOE_WIKI_PATH_BOOST=0.

        apply_staleness_gate (2026-08-31, default True): when True,
        unconditionally excludes superseded pages, excludes point-in-time
        pages (quote/data) older than their respective ceiling
        (_QUOTE_STALENESS_DAYS / _DATA_STALENESS_DAYS), and (2026-09-03)
        collapses near-duplicate point-in-time hits down to just the
        newest per subject via _collapse_near_duplicate_point_in_time --
        see that method's docstring for why age-ceiling + supersession
        alone weren't enough (repeated "what's SLV's price" surfaced a
        pile of same-day near-duplicate quote pages). This whole
        parameter is the ambient-recall AND wiki_search behavior (both
        call search() with the default True) — a stale, superseded, or
        redundant quote should never surface as background context or as
        a top wiki_search hit. Callers doing duplicate-detection
        (wiki_dedup.find_duplicate) pass False: a dedup search needs to
        SEE stale/superseded/redundant candidates, since that's often
        exactly what it's looking for."""
        if not query or not query.strip():
            return []
        q_blob = self._embed(query)
        if q_blob is None:
            return []
        q_vec = np.frombuffer(q_blob, dtype=np.float32)
        thr = _WIKI_THRESHOLD if threshold is None else float(threshold)
        q_tokens = self._query_tokens(query)

        with self._lock, self._connect() as c:
            rows = c.execute("""
                SELECT path, title, type, body, embedding,
                       point_in_time_kind, superseded, point_in_time_asof
                FROM wiki_pages
                WHERE embedding IS NOT NULL
            """).fetchall()
        if not rows:
            return []

        now = time.time()
        vecs = []
        metas = []
        for path, title, typ, body, blob, pit_kind, superseded, pit_asof in rows:
            # Age is computed from the page's own generated_at/date
            # frontmatter (point_in_time_asof), NOT filesystem mtime --
            # mtime changes on any rewrite (an Obsidian save, a metadata
            # backfill) and would silently reset staleness to zero. A
            # missing/unparsed asof (0.0) can't be aged, so it's treated
            # as not-stale (pit_kind still gates it from being FOUND at
            # all if pit_kind is unset, but never filtered on unknown age).
            age_days = ((now - pit_asof) / 86400.0
                       if pit_kind and pit_asof else None)
            if apply_staleness_gate:
                if superseded:
                    continue
                if pit_kind and age_days is not None:
                    ceiling = (_QUOTE_STALENESS_DAYS if pit_kind == "quote"
                              else _DATA_STALENESS_DAYS)
                    if age_days > ceiling:
                        continue
            try:
                v = np.frombuffer(blob, dtype=np.float32)
            except Exception:
                continue
            if v.shape != q_vec.shape:
                continue
            vecs.append(v)
            metas.append((path, title, typ, body, pit_kind or None, age_days))
        if not vecs:
            return []

        if apply_staleness_gate:
            metas, vecs = self._collapse_near_duplicate_point_in_time(
                query, metas, vecs, q_vec)
            if not vecs:
                return []

        M = np.stack(vecs)
        raw_scores = M @ q_vec
        # Apply per-page path-stem boost before ranking so the bullseye
        # page can climb above same-topic-but-denser neighbors.
        boosts = np.array([
            self._path_boost(metas[i][0], q_tokens,
                             _PATH_BOOST, _PATH_BOOST_CAP)
            for i in range(len(metas))
        ], dtype=np.float32)
        scores = raw_scores + boosts
        order = np.argsort(-scores)

        out: list[dict] = []
        for idx in order:
            score = float(scores[idx])
            if score < thr:
                break
            path, title, typ, body, pit_kind, age_days = metas[idx]
            snippet = (body or '').strip().replace('\n', ' ')
            if len(snippet) > 240:
                snippet = snippet[:237] + '...'
            out.append({
                'path': path,
                'title': title or _title_from_path(path),
                'type': typ or '',
                'snippet': snippet,
                'score': score,
                'cosine': float(raw_scores[idx]),
                'path_boost': float(boosts[idx]),
                'point_in_time_kind': pit_kind,
                'age_days': age_days,
            })
            if len(out) >= limit:
                break
        return out

    # ─── Introspection ──────────────────────────────────────────────────

    def count_pages(self) -> int:
        """Number of rows in the store (regardless of embedding state)."""
        with self._lock, self._connect() as c:
            return c.execute(
                "SELECT COUNT(*) FROM wiki_pages").fetchone()[0]

    def count_embedded(self) -> int:
        """Number of rows that have a non-NULL embedding."""
        with self._lock, self._connect() as c:
            return c.execute(
                "SELECT COUNT(*) FROM wiki_pages "
                "WHERE embedding IS NOT NULL").fetchone()[0]

    def list_pages(self) -> list[dict]:
        """Lightweight listing — path, title, type, mtime. Useful for
        debugging and for the watcher's startup banner."""
        with self._lock, self._connect() as c:
            rows = c.execute(
                "SELECT path, title, type, mtime FROM wiki_pages "
                "ORDER BY path"
            ).fetchall()
        return [{'path': r[0], 'title': r[1], 'type': r[2], 'mtime': r[3]}
                for r in rows]


# ─── Module-level singleton + getter ───────────────────────────────────────
# Lets a future /wiki slash command grab the store without rewiring jarvis.

_singleton: WikiEmbeddingStore | None = None
_singleton_lock = threading.Lock()


def get_store(wiki_root: Path | str | None = None,
              db_path: Path | str | None = None) -> WikiEmbeddingStore:
    """Return the process-wide WikiEmbeddingStore. First call initializes
    it (with defaults or the passed-in paths); subsequent calls return the
    same instance regardless of args."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = WikiEmbeddingStore(
                wiki_root=wiki_root or _DEFAULT_WIKI_ROOT,
                db_path=db_path or _DEFAULT_DB,
            )
        return _singleton


# ─── Chat-path auto-injection ────────────────────────────────────────────────
# Helpers that let the chat handler surface wiki pages WITHOUT the user
# explicitly running /wiki. Mirrors chloe_memory's recall path:
#   1. looks_like_wiki_query()   — cheap keyword heuristic: "is this a topic
#      question?" — decides whether a lookup is even worth an embed call.
#   2. wiki_context_for_query()  — runs the search and formats the hits into a
#      system-prompt block. Never raises; returns '' on any failure so the
#      chat path is never broken by a wiki problem.
# Parallel to chloe_memory.looks_like_recall_query / format_recall_block.

# Auto-inject uses a stricter threshold than explicit /wiki: when the user
# did NOT ask for the wiki, a marginal match is just prompt noise, so only
# confident hits get injected. Tune via env if injects feel sparse/noisy.
_WIKI_INJECT_THRESHOLD = float(
    os.environ.get("CHLOE_WIKI_INJECT_THRESHOLD", "0.5"))

# Phrases that suggest the user is asking ABOUT a topic (vs chitchat, a
# slash command, or a memory probe). Substring match — same cheap approach
# as chloe_memory._RECALL_KEYWORDS. A false positive just costs one embed
# call that the threshold filter then drops, so erring loose is fine.
_WIKI_QUERY_KEYWORDS: tuple[str, ...] = (
    "what is", "what are", "what's", "what was", "what does",
    "what do you know about",
    "who is", "who are", "who's", "who was",
    "tell me about", "tell me more about",
    "explain", "describe", "define",
    "how does", "how do", "how is", "how to",
    "why does", "why is", "why are", "why do",
    "do you know about", "do you know what", "do you know who",
    "remind me what", "remind me how", "remind me who",
    "catch me up on", "give me a rundown", "give me the rundown",
    "what's the deal with", "fill me in on",
)


def looks_like_wiki_query(text: str) -> bool:
    """Heuristic: does this turn look like a question about a topic that
    Chloe's wiki might cover? If so the chat path does a wiki lookup before
    answering.

    Mirrors chloe_memory.looks_like_recall_query. Deliberately cheap and a
    little loose — a false positive only costs one embed call, and the
    similarity threshold drops anything that isn't actually relevant. Empty
    input and slash commands are skipped."""
    if not text:
        return False
    t = text.strip().lower()
    if not t or t.startswith("/"):
        return False
    return any(kw in t for kw in _WIKI_QUERY_KEYWORDS)


def format_wiki_block(hits: "Iterable[dict]") -> str:
    """Format wiki search hits as a system-prompt addendum. Returns empty
    string when there are no hits. Parallel to chloe_memory.format_recall_block.

    Each hit is a dict from WikiEmbeddingStore.search (keys: path, title,
    type, snippet, score, point_in_time_kind, age_days).

    Point-in-time hits (search() already excludes ones past their
    staleness ceiling when apply_staleness_gate=True, which is the
    default) still get an inline age label here — "label always" per Ed's
    instruction, so the model sees exactly how fresh a quote/figure is
    rather than presenting it as unqualified current fact."""
    hits = list(hits or [])
    if not hits:
        return ""
    lines = []
    for h in hits:
        title = h.get("title") or h.get("path", "?")
        typ = h.get("type") or "page"
        snippet = (h.get("snippet") or "").strip()
        if len(snippet) > 320:
            snippet = snippet[:320] + " […]"
        age_note = ""
        pit_kind = h.get("point_in_time_kind")
        age_days = h.get("age_days")
        if pit_kind and age_days is not None:
            kind_label = "quote" if pit_kind == "quote" else "official data"
            age_note = f" [point-in-time {kind_label}, {age_days:.1f} day(s) old]"
        lines.append(f"  [{typ}]{age_note} {title}: {snippet}")
    return (
        "\n\n## Possibly relevant wiki pages:\n"
        + "\n".join(lines)
        + "\n(this is from your own wiki — use it as background if it fits "
          "the question; ignore it if it doesn't, and don't mention the "
          "wiki itself unless the user asks. Anything tagged "
          "[point-in-time ...] was true only as of the stated age — do not "
          "present it as current without saying how old it is.)"
    )


def wiki_context_for_query(text: str, limit: int = 2) -> str:
    """End-to-end chat-path helper: if `text` looks like a topic question,
    search the wiki and return a formatted system-prompt block. Returns ''
    when it's not a wiki-style query, when nothing clears the inject
    threshold, or when the store / embeddings are unavailable.

    NEVER raises — safe to call inline while building a system prompt."""
    if not looks_like_wiki_query(text):
        return ""
    try:
        store = get_store()
        if store.count_embedded() == 0:
            return ""
        hits = store.search(text, limit=limit,
                            threshold=_WIKI_INJECT_THRESHOLD)
        # Permissive-mode gate: skip wiki pages whose path contains '_nsfw'
        # unless the flag is on. Lets Ed tag adult-coded notes without
        # leaking them into family-safe chats.
        try:
            import nsfw_mode as _nsfw
            if not _nsfw.is_enabled():
                hits = [h for h in hits
                        if "_nsfw" not in str(h.get("path", "")).lower()]
        except Exception:
            pass
        if hits:
            preview = text.strip()[:60]
            print(f"[wiki] inject: {len(hits)} hit(s) for {preview!r}",
                  flush=True)
        return format_wiki_block(hits)
    except Exception as e:
        print(f"[wiki] context lookup failed: {e}", flush=True)
        return ""
