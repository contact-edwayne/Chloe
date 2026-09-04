"""
tools/wiki_dedup_report.py — READ-ONLY duplicate-cluster report for Chloe's
wiki. Writes nothing except its own dated report file. Does not touch any
page under C:\\Chloe\\brain\\wiki\\, does not call find_duplicate() in a
write path (that hook isn't wired up yet), and does not merge anything.

Three independent signals, reported separately so the threshold's actual
behavior on the real corpus is visible before anything is automated:

  1. VERSION-SUFFIX FAMILIES — exact filename match modulo a trailing
     _v2/_v3/... suffix (e.g. macro_yield_curve_inversion_mechanics +
     _v2/_v3/_v4). Certain: these are byte-identical-topic files by
     construction (chloe_jobs._write_brain's auto-suffix-on-collision).

  2. CANONICAL-KEY CLUSTERS — pages whose wiki_dedup.canonical_slug()
     output matches exactly (same token SET after stopword-strip +
     singularize + sort). High precision, zero Ollama calls. Scoped to
     concepts/ + entities/ only (sources/daily/etc are dated records, not
     topic pages, and weren't part of the reported duplication pattern).
     Keyed and rendered by FULL PATH (not bare filename stem) -- a stem
     can legitimately exist in both concepts/ and entities/ as two
     different real files (e.g. concepts/research_wheel_strategy.md vs
     entities/research_wheel_strategy.md both exist), and collapsing on
     stem alone would misrender them as duplicate rows of the same file.

  3. COSINE CLUSTERS — pairwise cosine similarity (raw embedding dot
     product, NOT the path-boosted search() score) above
     CHLOE_WIKI_DEDUP_THRESHOLD (default 0.85). Reuses embeddings already
     in WikiEmbeddingStore -- no new Ollama calls.

     IMPORTANT CAVEAT ON CLUSTERING METHOD: naive transitive closure
     (union-find over every above-threshold pair) is single-linkage
     clustering -- if A~B and B~C both clear the threshold, A and C end
     up in the same reported cluster even if A~C is nowhere near it. On
     this corpus that produces a "mega-cluster" chaining together
     genuinely unrelated pages. So each cluster below reports its MINIMUM
     internal pairwise similarity alongside its size -- a cluster whose
     min-pairwise is far below the threshold is a chain artifact, not a
     real duplicate group, and should not be treated as one. This is a
     REPORTING-time issue only: the eventual find_duplicate() write-path
     hook (not yet wired) checks one NEW topic against the existing
     corpus directly (nearest-neighbor), never clusters transitively, so
     it isn't exposed to this failure mode.

Usage:
    python tools/wiki_dedup_report.py
    python tools/wiki_dedup_report.py --threshold 0.85 --also-try 0.88,0.90,0.92
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from wiki_dedup import canonical_slug, STOPWORDS  # noqa: E402
from wiki_embedding import WikiEmbeddingStore, _DEFAULT_WIKI_ROOT, _DEFAULT_DB  # noqa: E402

DEFAULT_THRESHOLD = float(os.environ.get("CHLOE_WIKI_DEDUP_THRESHOLD", "0.85"))

# Scope the semantic + canonical-key passes to actual topic pages.
# sources/ are dated ingest records (one per raw document), daily/ are
# dated logs -- neither is the "same topic, different filename" problem
# this is scoped to. Signal 1 (version-suffix) still scans everything,
# since _write_brain's auto-suffix bug isn't scope-limited.
SCOPED_DIRS = ("concepts", "entities")

_VERSION_SUFFIX_RE = re.compile(r"^(.*)_v(\d+)$")


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _fmt_row(path: str, title: str, mtime: float) -> str:
    dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    return f"  - `{path}` — {title or '(no title)'} (updated {dt})"


def _cluster_stats(idxs, sims) -> tuple:
    """Return (min_pairwise, max_pairwise, mean_pairwise) for a cluster's
    induced subgraph, so a chained mega-cluster is visibly distinguishable
    from a tight one."""
    pairs = [float(sims[i, j]) for a, i in enumerate(idxs) for j in idxs[a + 1:]]
    if not pairs:
        return (1.0, 1.0, 1.0)
    return (min(pairs), max(pairs), sum(pairs) / len(pairs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                     help=f"Cosine threshold for the full cluster dump (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--also-try", type=str, default="0.88,0.90,0.92",
                     help="Comma-separated extra thresholds to summarize (cluster count/max size only)")
    ap.add_argument("--wiki-root", type=Path, default=_DEFAULT_WIKI_ROOT)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    args = ap.parse_args()

    store = WikiEmbeddingStore(wiki_root=args.wiki_root, db_path=args.db)

    with store._lock, store._connect() as c:
        rows = c.execute("""
            SELECT path, title, type, embedding, mtime
            FROM wiki_pages
        """).fetchall()

    print(f"Loaded {len(rows)} pages from {args.db}", flush=True)

    # Key EVERYTHING by full relative path from here on -- a bare stem can
    # collide across concepts/ vs entities/ for two genuinely different files.
    by_path = {r[0]: {"title": r[1], "type": r[2], "embedding": r[3], "mtime": r[4]}
               for r in rows}

    # ─── Signal 1: version-suffix families (all dirs) ──────────────────
    stem_to_path = defaultdict(list)  # stem -> [full paths] (handles cross-dir collisions)
    for path in by_path:
        stem_to_path[Path(path).stem].append(path)

    version_families: dict[str, list[str]] = defaultdict(list)
    for stem in stem_to_path:
        m = _VERSION_SUFFIX_RE.match(stem)
        if m:
            version_families[m.group(1)].append(stem)
    for base, members in list(version_families.items()):
        if base in stem_to_path and base not in members:
            members.insert(0, base)

    # ─── Signal 2: canonical-key clusters (concepts/ + entities/, by path) ──
    scoped_paths = [p for p in by_path if p.split("/")[0] in SCOPED_DIRS]
    canon_groups: dict[str, list[str]] = defaultdict(list)
    for path in scoped_paths:
        key = canonical_slug(Path(path).stem)
        if key:
            canon_groups[key].append(path)
    canon_clusters = {k: v for k, v in canon_groups.items() if len(v) > 1}

    # ─── Signal 3: cosine clusters over raw embeddings (by path) ────────
    embed_paths = [p for p in scoped_paths if by_path[p]["embedding"] is not None]
    vecs = [np.frombuffer(by_path[p]["embedding"], dtype=np.float32) for p in embed_paths]
    if vecs:
        M = np.stack(vecs)
        sims = M @ M.T
    else:
        sims = np.zeros((0, 0))
    n = len(embed_paths)

    def cluster_at(threshold: float):
        uf = UnionFind(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if float(sims[i, j]) >= threshold:
                    uf.union(i, j)
        groups = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)
        return {root: idxs for root, idxs in groups.items() if len(idxs) > 1}

    main_clusters = cluster_at(args.threshold)

    extra_thresholds = [float(t) for t in args.also_try.split(",") if t.strip()]
    sensitivity = []
    for t in extra_thresholds:
        cl = cluster_at(t)
        sizes = sorted((len(v) for v in cl.values()), reverse=True)
        sensitivity.append((t, len(cl), sizes[:5]))

    # ─── Write report ────────────────────────────────────────────────────
    today = date.today().isoformat()
    # Deliberately OUTSIDE wiki_root: wiki_watcher polls wiki_root.rglob('*.md')
    # with no exclusion mechanism, so a report file left inside it would get
    # auto-embedded into the recall corpus within ~2s -- polluting the very
    # store this script is investigating.
    reports_dir = args.wiki_root.parent / "dedup_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"dedup_report_{today}.md"
    lines = []
    lines.append(f"# Wiki Dedup Report — {today}")
    lines.append("")
    lines.append("Read-only report. Nothing was merged, edited, or deleted.")
    lines.append("")
    lines.append(f"- Pages loaded: {len(rows)}")
    lines.append(f"- Scoped to concepts/+entities/ for signals 2 & 3: {len(scoped_paths)} pages "
                 f"({len(embed_paths)} with embeddings)")
    lines.append(f"- Cosine threshold (full dump below): **{args.threshold}** "
                 f"(env CHLOE_WIKI_DEDUP_THRESHOLD, default 0.85)")
    lines.append("")
    lines.append("## Stopwords used by canonical_slug() (Signal 2)")
    lines.append("")
    lines.append("Dropped from anywhere in the token list before fingerprinting:")
    lines.append("")
    lines.append("`" + ", ".join(sorted(STOPWORDS)) + "`")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(f"## Signal 1: Version-suffix families ({len(version_families)})")
    lines.append("")
    lines.append("Exact filename match modulo `_v2`/`_v3`/... — these are the "
                 "`_write_brain()` auto-suffix-on-collision artifacts. Highest confidence.")
    lines.append("")
    for base, members in sorted(version_families.items()):
        lines.append(f"**{base}\\*** ({len(members)} file{'s' if len(members) != 1 else ''})")
        for stem in members:
            for path in stem_to_path[stem]:
                meta = by_path[path]
                lines.append(_fmt_row(path, meta["title"], meta["mtime"]))
        lines.append("")

    lines.append(f"## Signal 2: Canonical-key clusters ({len(canon_clusters)})")
    lines.append("")
    lines.append("Same normalized token SET (stopwords stripped, singularized, sorted) — "
                 "these are word-order/plural/prefix variants of the exact same slug. "
                 "concepts/+entities/ only.")
    lines.append("")
    for key, members in sorted(canon_clusters.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"**canonical key: `{key}`** ({len(members)} files)")
        for path in members:
            meta = by_path[path]
            lines.append(_fmt_row(path, meta["title"], meta["mtime"]))
        lines.append("")

    lines.append(f"## Signal 3: Cosine-similarity clusters ({len(main_clusters)}) at threshold {args.threshold}")
    lines.append("")
    lines.append("Pairwise cosine >= threshold on raw embeddings, connected-component "
                 "clustered (single-linkage — see the module docstring's caveat on chaining). "
                 "Each cluster reports (min/mean/max) internal pairwise similarity: a low "
                 "MIN relative to the threshold means this cluster is held together by a "
                 "chain of edges, not mutual similarity, and should NOT be treated as one "
                 "merge candidate — inspect the specific high edges instead.")
    lines.append("")
    for root, idxs in sorted(main_clusters.items(), key=lambda kv: -len(kv[1])):
        members = [embed_paths[i] for i in idxs]
        canon_keys_in_cluster = {canonical_slug(Path(m).stem) for m in members}
        already_caught = any(k in canon_clusters for k in canon_keys_in_cluster if k)
        lo, hi, mean = _cluster_stats(idxs, sims)
        chain_warning = " ⚠ CHAIN ARTIFACT — min far below threshold, do not treat as one group" \
            if lo < args.threshold - 0.10 else ""
        tag = " (overlaps a Signal-2 cluster)" if already_caught else " (cosine-only)"
        lines.append(f"**cluster{tag}** ({len(members)} files) "
                     f"min={lo:.3f} mean={mean:.3f} max={hi:.3f}{chain_warning}")
        for m in members:
            meta = by_path[m]
            lines.append(_fmt_row(m, meta["title"], meta["mtime"]))
        lines.append("")

    lines.append("## Threshold sensitivity (summary only, not full dumps)")
    lines.append("")
    lines.append("| threshold | cluster count | top-5 cluster sizes |")
    lines.append("|---|---|---|")
    main_sizes = sorted((len(v) for v in main_clusters.values()), reverse=True)
    lines.append(f"| {args.threshold} (main) | {len(main_clusters)} | {main_sizes[:5]} |")
    for t, count, sizes in sensitivity:
        lines.append(f"| {t} | {count} | {sizes} |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}", flush=True)
    print(f"  Signal 1 (version families): {len(version_families)}", flush=True)
    print(f"  Signal 2 (canonical-key clusters): {len(canon_clusters)}", flush=True)
    print(f"  Signal 3 (cosine clusters @ {args.threshold}): {len(main_clusters)}, "
          f"max size {main_sizes[0] if main_sizes else 0}", flush=True)
    for t, count, sizes in sensitivity:
        print(f"  Signal 3 @ {t}: {count} clusters, max size {sizes[0] if sizes else 0}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
