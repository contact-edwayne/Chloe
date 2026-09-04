"""Smoke tests for brain_graph + brain_http."""
import os
import sys
import json
import time
import tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent))


def _seed_wiki():
    root = Path(tempfile.mkdtemp(prefix="brain_graph_test_"))
    os.environ["CHLOE_BRAIN_ROOT"] = str(root)
    wiki = root / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "sources").mkdir(parents=True)
    (wiki / "entities" / "qmd.md").write_text(
        "# qmd\n\nLocal markdown search engine. See [[concepts/rag]] for context "
        "and [[entities/karpathy]].\n", encoding="utf-8")
    (wiki / "entities" / "karpathy.md").write_text(
        "# Karpathy\n\nProposed the [[concepts/llm_wiki]] pattern.\n",
        encoding="utf-8")
    (wiki / "concepts" / "rag.md").write_text(
        "# RAG\n\nRetrieval-augmented generation. Contrasted with [[concepts/llm_wiki]].\n",
        encoding="utf-8")
    (wiki / "concepts" / "llm_wiki.md").write_text(
        "# LLM Wiki\n\nPattern by [[entities/karpathy]]. Uses [[entities/qmd]] for indexing.\n",
        encoding="utf-8")
    (wiki / "sources" / "karpathy_gist.md").write_text(
        "# Karpathy Gist\n\nMentions [[entities/karpathy]] and [[concepts/llm_wiki]] "
        "and a missing thing [[entities/notebooklm]] (no page yet).\n",
        encoding="utf-8")
    (wiki / "index.md").write_text("# Index\n", encoding="utf-8")
    return root, wiki


def t_compute_graph_shape():
    from brain_graph import compute_graph
    root, wiki = _seed_wiki()
    g = compute_graph(wiki)
    nodes = g["nodes"]
    edges = g["edges"]
    stats = g["stats"]
    ids = {n["id"] for n in nodes}
    # Real pages are present (5 of them — index.md is filtered)
    assert "entities/qmd" in ids
    assert "entities/karpathy" in ids
    assert "concepts/rag" in ids
    assert "concepts/llm_wiki" in ids
    assert "sources/karpathy_gist" in ids
    # Ghost target appears as a node, but real=False
    assert "entities/notebooklm" in ids
    ghost = next(n for n in nodes if n["id"] == "entities/notebooklm")
    assert ghost["real"] is False
    # index.md is filtered out
    assert "index" not in ids
    # Edges include the relations we wrote
    edge_set = {(e["source"], e["target"]) for e in edges}
    assert ("entities/qmd", "concepts/rag") in edge_set
    assert ("entities/qmd", "entities/karpathy") in edge_set
    assert ("concepts/llm_wiki", "entities/qmd") in edge_set
    assert ("sources/karpathy_gist", "entities/notebooklm") in edge_set
    # Stats correct
    assert stats["node_count"] == len(nodes)
    assert stats["edge_count"] == len(edges)
    assert stats["real_count"] == 5
    assert stats["ghost_count"] == 1
    # Nodes sorted by descending degree
    assert nodes[0]["degree"] >= nodes[-1]["degree"]
    print(f"PASS: compute_graph ({stats['node_count']} nodes, "
          f"{stats['edge_count']} edges, {stats['ghost_count']} ghost)")


def t_read_page_path_safe():
    from brain_graph import read_page
    root, wiki = _seed_wiki()
    r = read_page(wiki, "entities/qmd")
    assert r["ok"] and "Local markdown search engine" in r["text"]
    # Path traversal attempts should fail
    bad = read_page(wiki, "../../../etc/passwd")
    assert not bad["ok"]
    bad = read_page(wiki, "")
    assert not bad["ok"]
    # Ghost page
    ghost = read_page(wiki, "entities/notebooklm")
    assert not ghost["ok"]
    assert ghost.get("ghost") is True
    print("PASS: read_page (success + path-safe + ghost)")


def t_http_endpoints():
    """Start brain_http on a free port and hit the endpoints."""
    import brain_http, socket
    # Pick a free port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    info = brain_http.start(host="127.0.0.1", port=port)
    assert info["running"], f"start failed: {info}"
    try:
        time.sleep(0.2)  # let it spin up
        # /api/brain/graph
        with urlopen(f"http://127.0.0.1:{port}/api/brain/graph", timeout=5) as r:
            data = json.loads(r.read())
        assert "nodes" in data and "edges" in data and "stats" in data
        assert data["stats"]["node_count"] >= 5
        # /api/brain/page
        with urlopen(f"http://127.0.0.1:{port}/api/brain/page?p=" +
                     quote("entities/qmd"), timeout=5) as r:
            data = json.loads(r.read())
        assert data["ok"] and "Local markdown search engine" in data["text"]
        # /brain-graph.html
        with urlopen(f"http://127.0.0.1:{port}/brain-graph.html", timeout=5) as r:
            html = r.read().decode("utf-8")
        assert "CHLOE" in html and ("BRAIN" in html or "Brain" in html)
        assert ("vis-network" in html or "__bundler/manifest" in html or "marked" in html)
        print(f"PASS: HTTP endpoints (port {port}, {len(html)}b html)")
    finally:
        brain_http.stop()


def t_ghost_slugs_below_degree():
    from brain_graph import ghost_slugs_below_degree
    root, wiki = _seed_wiki()
    # entities/notebooklm is the only ghost, referenced once -> degree 1.
    deg1 = ghost_slugs_below_degree(wiki, max_degree=1)
    assert deg1 == ["entities/notebooklm"], deg1
    # No ghosts at degree 0.
    deg0 = ghost_slugs_below_degree(wiki, max_degree=0)
    assert deg0 == [], deg0
    print(f"PASS: ghost_slugs_below_degree (deg<=1 -> {deg1})")


def t_http_bulk_ignore():
    """POST /api/brain/ghosts_bulk_ignore prunes low-degree ghosts."""
    import brain_http, socket
    from urllib.request import Request
    root, wiki = _seed_wiki()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    info = brain_http.start(host="127.0.0.1", port=port)
    assert info["running"], f"start failed: {info}"
    try:
        time.sleep(0.2)
        base = f"http://127.0.0.1:{port}"
        with urlopen(f"{base}/api/brain/graph", timeout=5) as r:
            before = json.loads(r.read())
        assert before["stats"]["ghost_count"] == 1, before["stats"]
        req = Request(f"{base}/api/brain/ghosts_bulk_ignore",
                      data=json.dumps({"max_degree": 1}).encode("utf-8"),
                      headers={"Content-Type": "application/json"},
                      method="POST")
        with urlopen(req, timeout=5) as r:
            res = json.loads(r.read())
        assert res["ok"] and res["added_count"] == 1, res
        assert "entities/notebooklm" in res["added"], res
        # Ghost is filtered on the next graph request.
        with urlopen(f"{base}/api/brain/graph", timeout=5) as r:
            after = json.loads(r.read())
        assert after["stats"]["ghost_count"] == 0, after["stats"]
        assert "entities/notebooklm" not in {n["id"] for n in after["nodes"]}
        # Idempotent: a second call adds nothing.
        req2 = Request(f"{base}/api/brain/ghosts_bulk_ignore",
                       data=json.dumps({"max_degree": 1}).encode("utf-8"),
                       headers={"Content-Type": "application/json"},
                       method="POST")
        with urlopen(req2, timeout=5) as r:
            res2 = json.loads(r.read())
        assert res2["added_count"] == 0, res2
        print(f"PASS: bulk ghost-ignore (port {port}, pruned 1 ghost, idempotent)")
    finally:
        brain_http.stop()


def main():
    t_compute_graph_shape()
    t_read_page_path_safe()
    t_http_endpoints()
    t_ghost_slugs_below_degree()
    t_http_bulk_ignore()


if __name__ == "__main__":
    main()
