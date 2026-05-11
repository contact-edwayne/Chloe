"""One-shot backfill: embed every existing turn that doesn't yet have a
vector. Re-runnable — safe to re-run after a model swap (delete the column
data first if you want a full re-embed) or after long offline stretches.

Walks `turns` rows where embedding IS NULL, calls ChloeMemory._embed on
each content field, and UPDATEs the row. Reports progress every 50 rows
and bails cleanly if Ollama goes unreachable mid-run.

    python backfill_embeddings.py            # embed all NULL-embedding rows
    python backfill_embeddings.py --limit 200  # do at most 200 (for testing)
    python backfill_embeddings.py --rebuild  # re-embed EVERY row, not just NULLs
"""

import sqlite3
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DB_PATH = THIS_DIR / "chloe_memory.db"


def main():
    rebuild = "--rebuild" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        try:
            limit = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("--limit needs an integer")
            return 1

    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        return 1

    # Instantiate ChloeMemory so the schema migration runs (adds the
    # embedding column if it's missing). Then we drop down to raw sqlite3
    # for the iteration — ChloeMemory's append_turn would write NEW rows,
    # we want to UPDATE existing rows in place.
    from chloe_memory import ChloeMemory
    facts = THIS_DIR / "facts.md"
    mem = ChloeMemory(DB_PATH, facts)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode = WAL")

    if rebuild:
        target_clause = ""
    else:
        target_clause = "WHERE embedding IS NULL"

    total = conn.execute(
        f"SELECT COUNT(*) FROM turns {target_clause}"
    ).fetchone()[0]
    if total == 0:
        print("Nothing to backfill — all rows already have embeddings.")
        return 0

    if limit is not None and limit < total:
        print(f"Backfilling {limit} of {total} eligible row(s)")
    else:
        print(f"Backfilling {total} row(s)")

    sql = f"SELECT id, content FROM turns {target_clause} ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    rows = conn.execute(sql).fetchall()

    done = 0
    failed = 0
    start = time.time()
    for row_id, content in rows:
        blob = mem._embed(content)
        if blob is None:
            failed += 1
            if failed >= 5 and done == 0:
                print("Embed failed 5x in a row with zero successes — "
                      "bailing out. Check `ollama list` and confirm "
                      "nomic-embed-text is pulled.")
                break
            continue
        try:
            with conn:
                conn.execute(
                    "UPDATE turns SET embedding = ? WHERE id = ?",
                    (blob, row_id),
                )
        except sqlite3.Error as e:
            print(f"  row {row_id} UPDATE failed: {e}")
            failed += 1
            continue
        done += 1
        if done % 50 == 0:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            print(f"  {done}/{len(rows)}  ({rate:.1f} turns/s, "
                  f"{failed} failed so far)")

    elapsed = time.time() - start
    print()
    print(f"done: {done} embedded, {failed} failed, "
          f"{elapsed:.1f}s elapsed ({done/elapsed:.1f} turns/s)"
          if elapsed else f"done: {done} embedded, {failed} failed")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
