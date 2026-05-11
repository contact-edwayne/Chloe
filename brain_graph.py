"""brain_graph.py - extract a force-directed graph from the wiki.

Pure compute, no HTTP. Called by brain_http.py to serve the graph view at
/brain-graph.html. Each markdown file in BRAIN.wiki_dir/** is a node;
[[wikilinks]] inside the page body are edges.

Public API:
    compute_graph()  -> dict {nodes: [...], edges: [...], stats: {...}}
    read_page(rel)   -> str (raw markdown, path-safe scoped to wiki_dir)
"""

import re
from pathlib import Path

# [[link]] or [[link|alt text]]
LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")

# Page kinds we care about (subdirectories under wiki/)
KNOWN_TYPES = ("entities", "concepts", "sources", "comparisons", "explorations")


def _slug_for_link(target: str, current_type: str) -> str:
    """Resolve a [[link]] target into a canonical node id.

    Wikilinks in this brain look like:
        [[qmd]]                   -> bare slug, resolve relative to current type
        [[entities/qmd]]          -> typed path
        [[entities/qmd.md]]       -> typed path with extension
    Returns the canonical id used as node.id (no extension, normalized).
    """
    t = target.strip().replace("\\", "/")
    if t.endswith(".md"):
        t = t[:-3]
    if "/" not in t:
        # Bare slug — assume same type as the page that linked from it
        t = f"{current_type}/{t}"
    return t


def compute_graph(wiki_dir) -> dict:
    """Walk wiki_dir/**/*.md and return {nodes, edges, stats}.

    Nodes:
        id     - 'entities/qmd', 'concepts/rag', etc. (no .md, forward slashes)
        label  - human-friendly slug ('qmd', 'rag')
        type   - parent dir ('entities', 'concepts', 'sources', ...)
        degree - in_degree + out_degree (used for sizing)
        size   - clamped degree, 6..36 px
    Edges:
        source - node.id
        target - node.id (may not exist as a real page; we still emit it as
                 a 'ghost' node so dangling links are visible)
    """
    wiki_dir = Path(wiki_dir)
    if not wiki_dir.exists():
        return {"nodes": [], "edges": [],
                "stats": {"error": f"wiki_dir not found: {wiki_dir}"}}

    nodes_by_id = {}
    edges = []

    # First pass: every .md file is a real node.
    for path in sorted(wiki_dir.rglob("*.md")):
        rel = path.relative_to(wiki_dir).with_suffix("")  # no .md
        rel_id = str(rel).replace("\\", "/")
        # Skip top-level meta files
        if rel_id in ("index", "log", "gaps", "SESSIONS"):
            continue
        parts = rel_id.split("/")
        if len(parts) < 2:
            page_type = "other"
            slug = parts[0]
        else:
            page_type = parts[0]
            slug = "/".join(parts[1:])
        nodes_by_id[rel_id] = {
            "id":     rel_id,
            "label":  slug,
            "type":   page_type,
            "real":   True,
            "in_deg": 0,
            "out_deg": 0,
        }

    # Second pass: parse links to build edges.
    for path in sorted(wiki_dir.rglob("*.md")):
        rel = path.relative_to(wiki_dir).with_suffix("")
        src_id = str(rel).replace("\\", "/")
        if src_id not in nodes_by_id:
            continue
        src_type = nodes_by_id[src_id]["type"]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in LINK_RE.finditer(text):
            tgt = _slug_for_link(m.group(1), src_type)
            if tgt == src_id:
                continue  # ignore self-loops
            # If target doesn't exist as a real file, create a ghost node.
            if tgt not in nodes_by_id:
                parts = tgt.split("/")
                ghost_type = parts[0] if len(parts) > 1 else "ghost"
                nodes_by_id[tgt] = {
                    "id":     tgt,
                    "label":  parts[-1],
                    "type":   ghost_type,
                    "real":   False,
                    "in_deg": 0,
                    "out_deg": 0,
                }
            edges.append({"source": src_id, "target": tgt})
            nodes_by_id[src_id]["out_deg"] += 1
            nodes_by_id[tgt]["in_deg"] += 1

    # Size each node by degree (clamped). Helps visually pop high-connectivity hubs.
    for n in nodes_by_id.values():
        deg = n["in_deg"] + n["out_deg"]
        n["degree"] = deg
        n["size"] = max(6, min(36, 6 + deg * 2))

    nodes = sorted(nodes_by_id.values(), key=lambda n: -n["degree"])
    type_counts = {}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "real_count": sum(1 for n in nodes if n["real"]),
        "ghost_count": sum(1 for n in nodes if not n["real"]),
        "type_counts": type_counts,
    }
    return {"nodes": nodes, "edges": edges, "stats": stats}


def read_page(wiki_dir, rel_id: str) -> dict:
    """Return raw markdown for a node id, path-safe scoped to wiki_dir.

    rel_id must be a slash-delimited path under wiki_dir (e.g. 'entities/qmd').
    Returns {ok, text, path, error?}.
    """
    wiki_dir = Path(wiki_dir).resolve()
    rel_id = (rel_id or "").strip().strip("/")
    # The Claude-designed v2 frontend sends the page request with an explicit
    # ".md" suffix (e.g. "entities/qmd.md"); the v1 frontend sends it without.
    # Strip a trailing ".md" so we don't end up resolving "entities/qmd.md.md".
    if rel_id.lower().endswith(".md"):
        rel_id = rel_id[:-3]
    if not rel_id or ".." in rel_id.split("/"):
        return {"ok": False, "error": "invalid rel_id"}
    candidate = (wiki_dir / (rel_id + ".md")).resolve()
    # Path-traversal guard
    try:
        candidate.relative_to(wiki_dir)
    except ValueError:
        return {"ok": False, "error": "path escapes wiki_dir"}
    if not candidate.exists():
        return {"ok": False, "error": f"page not found: {rel_id}",
                "ghost": True, "path": str(candidate)}
    try:
        text = candidate.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": f"read failed: {type(e).__name__}: {e}"}
    return {"ok": True, "text": text, "path": str(candidate)}


# ─── CLI for quick smoke testing ─────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    from os import environ
    root = Path(environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
    wiki = root / "wiki"
    if "--page" in sys.argv:
        rel = sys.argv[sys.argv.index("--page") + 1]
        print(json.dumps(read_page(wiki, rel), indent=2))
    else:
        g = compute_graph(wiki)
        print(json.dumps(g["stats"], indent=2))
        print(f"sample top-5 nodes by degree:")
        for n in g["nodes"][:5]:
            print(f"  {n['id']:40s} type={n['type']:12s} deg={n['degree']}")
