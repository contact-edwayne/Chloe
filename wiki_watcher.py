"""
wiki_watcher.py — Keep WikiEmbeddingStore in sync with the filesystem.

Polling-based file watcher for C:\\Chloe\\brain\\wiki\\**/*.md. When you
edit a wiki page in Obsidian (or any editor) and save, this process picks
up the mtime change within a couple seconds and re-embeds the page so
`/wiki <query>` reflects your edits.

Why polling instead of watchdog:
  - Zero new pip deps. watchdog isn't in jarvis's venv and Ed prefers
    not adding install steps to the demo path.
  - Wiki is ~40 files. A 2-second polling pass is `os.stat * 40` —
    sub-millisecond on a local SSD.
  - Polling sidesteps Windows-specific edge cases (ReadDirectoryChangesW
    can miss events on network shares, partial saves, etc).

Usage:
    python wiki_watcher.py                # backfill + watch forever
    python wiki_watcher.py --once         # backfill + exit
    python wiki_watcher.py --interval 5   # poll every 5 seconds
    python wiki_watcher.py --rebuild      # force re-embed everything

Logs are stdout-only; cmd users can tee to a file if they want history.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from wiki_embedding import WikiEmbeddingStore, _DEFAULT_WIKI_ROOT, _DEFAULT_DB


# ─── Helpers ────────────────────────────────────────────────────────────────

def _now_str() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[wiki-watch {_now_str()}] {msg}", flush=True)


def _snapshot(wiki_root: Path) -> dict[str, float]:
    """Return {posix_rel_path: mtime} for every .md under wiki_root.

    Cheap — stat per file. Posix-style paths keep DB keys stable."""
    snap: dict[str, float] = {}
    for full in wiki_root.rglob('*.md'):
        try:
            rel = str(full.resolve().relative_to(wiki_root))
        except ValueError:
            continue
        rel = rel.replace('\\', '/')
        try:
            snap[rel] = full.stat().st_mtime
        except OSError:
            continue
    return snap


# ─── Main loop ──────────────────────────────────────────────────────────────

class WikiWatcher:
    """Polling watcher. One per process; no concurrency needed."""

    def __init__(self, store: WikiEmbeddingStore, interval: float = 2.0):
        self.store = store
        self.interval = float(interval)
        self.running = True
        # Internal snapshot — last-seen filesystem state. Initialized in
        # run() before the first poll so the first iteration doesn't fire
        # "everything changed" for every page already in the corpus.
        self._snap: dict[str, float] = {}

    def stop(self, *_):
        """Signal handler for graceful Ctrl-C. Sets the flag; the loop
        exits at the top of the next iteration."""
        if self.running:
            _log("shutting down — finishing current poll")
            self.running = False

    def backfill(self, rebuild: bool = False) -> None:
        """Walk the wiki dir and embed every page that isn't already in
        the store (or every page when rebuild=True)."""
        if rebuild:
            _log("--rebuild: deleting all wiki_pages rows for a clean re-embed")
            # purge by re-creating: delete each known page
            for row in self.store.list_pages():
                self.store.delete_page(row['path'])
        _log(f"backfill starting (wiki_root={self.store.wiki_root})")
        t0 = time.time()
        counters = self.store.backfill_all()
        elapsed = time.time() - t0
        _log(f"backfill done in {elapsed:.1f}s: {counters}")
        purged = self.store.purge_missing()
        if purged:
            _log(f"purged {purged} orphaned row(s) for deleted files")

    def _diff(self, old: dict[str, float],
              new: dict[str, float]) -> tuple[list[str], list[str]]:
        """Return (changed_or_added, removed). 'Changed' = new mtime > old
        mtime; 'added' = path in new but not old; 'removed' = path in old
        but not new."""
        changed: list[str] = []
        removed: list[str] = []
        for rel, mt in new.items():
            if rel not in old or mt > old[rel] + 1e-6:
                changed.append(rel)
        for rel in old:
            if rel not in new:
                removed.append(rel)
        return changed, removed

    def _apply(self, changed: list[str], removed: list[str]) -> None:
        """Run upsert/delete for the deltas the diff produced.

        Also publishes events on event_bus so brain_http.py's SSE stream
        can drive the brain-graph UI's recently-edited pulse + ticker.
        Best-effort: a missing/broken event_bus shouldn't break the
        watcher loop."""
        try:
            from event_bus import publish as _publish  # type: ignore
        except Exception:
            _publish = None  # type: ignore

        def _emit(evt: dict) -> None:
            if _publish is None:
                return
            try:
                _publish(evt)
            except Exception:
                pass

        for rel in changed:
            status = self.store.upsert_page(rel)
            if status in ('inserted', 'updated', 'embed-fail'):
                _log(f"  {status:10s} {rel}")
                # Node id in compute_graph drops the .md suffix.
                node_id = rel[:-3] if rel.endswith('.md') else rel
                _emit({
                    'type': 'upserted',
                    'node_id': node_id,
                    'status': status,
                })
            # 'unchanged' / 'missing' are noise — skip
        for rel in removed:
            ok = self.store.delete_page(rel)
            if ok:
                _log(f"  deleted    {rel}")
                node_id = rel[:-3] if rel.endswith('.md') else rel
                _emit({'type': 'deleted', 'node_id': node_id})

    def run(self, rebuild: bool = False, once: bool = False) -> int:
        """Backfill, then (unless --once) loop polling for changes.

        Returns 0 on clean exit (Ctrl-C or --once completion), non-zero
        on an unhandled error."""
        try:
            self.backfill(rebuild=rebuild)
        except Exception as e:
            _log(f"backfill crashed: {e!r}")
            return 1

        # Seed the snapshot from current state so the first poll doesn't
        # mass-fire "changed" for everything we just embedded.
        self._snap = _snapshot(self.store.wiki_root)

        if once:
            return 0

        _log(f"watching {len(self._snap)} file(s) — Ctrl-C to stop "
             f"(poll interval {self.interval}s)")

        signal.signal(signal.SIGINT, self.stop)
        try:
            signal.signal(signal.SIGTERM, self.stop)
        except (AttributeError, ValueError):
            # SIGTERM not available on some Windows configs; SIGINT is enough.
            pass

        while self.running:
            try:
                time.sleep(self.interval)
                if not self.running:
                    break
                new = _snapshot(self.store.wiki_root)
                changed, removed = self._diff(self._snap, new)
                if changed or removed:
                    self._apply(changed, removed)
                self._snap = new
            except Exception as e:
                _log(f"poll iteration crashed (continuing): {e!r}")
                time.sleep(self.interval)

        _log("clean exit")
        return 0


# ─── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch Chloe's wiki and keep its embeddings fresh.")
    p.add_argument('--wiki-root', type=Path, default=_DEFAULT_WIKI_ROOT,
                   help=f"Wiki root to watch (default: {_DEFAULT_WIKI_ROOT})")
    p.add_argument('--db', type=Path, default=_DEFAULT_DB,
                   help=f"SQLite path for wiki_pages (default: {_DEFAULT_DB})")
    p.add_argument('--interval', type=float, default=2.0,
                   help="Poll interval in seconds (default: 2.0)")
    p.add_argument('--once', action='store_true',
                   help="Backfill and exit; don't watch.")
    p.add_argument('--rebuild', action='store_true',
                   help="Delete all wiki_pages rows then re-embed everything.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.wiki_root.exists():
        _log(f"wiki root does not exist: {args.wiki_root}")
        return 1
    store = WikiEmbeddingStore(wiki_root=args.wiki_root, db_path=args.db)
    _log(f"store ready (db={args.db}); "
         f"pages_known={store.count_pages()} "
         f"embedded={store.count_embedded()}")
    watcher = WikiWatcher(store, interval=args.interval)
    return watcher.run(rebuild=args.rebuild, once=args.once)


if __name__ == '__main__':
    sys.exit(main())
