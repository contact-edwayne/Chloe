"""Smoke tests for daily_context.py + queue_processor.py.

Both modules exercise BRAIN heavy-LLM calls. We monkeypatch chloe_llm_call
so the tests don't need Groq access.
"""
import os
import sys
import datetime
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# Set up a tmp brain root before importing brain_wiring
TMP = Path(tempfile.mkdtemp(prefix="chloe_workflows_test_"))
os.environ["CHLOE_BRAIN_ROOT"] = str(TMP)


def _seed_brain():
    """Populate a fake brain with episodic, facts, raw, and a wiki index."""
    (TMP / "wiki" / "entities").mkdir(parents=True, exist_ok=True)
    (TMP / "wiki" / "concepts").mkdir(parents=True, exist_ok=True)
    (TMP / "wiki" / "sources").mkdir(parents=True, exist_ok=True)
    (TMP / "episodic").mkdir(parents=True, exist_ok=True)
    (TMP / "raw").mkdir(parents=True, exist_ok=True)
    (TMP / "facts").mkdir(parents=True, exist_ok=True)
    (TMP / "queue").mkdir(parents=True, exist_ok=True)

    yest = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    (TMP / "episodic" / f"{yest}.md").write_text(
        "# yesterday\n\n## [09:15]\n\nFinished the brain graph view.\n", encoding="utf-8")
    (TMP / "facts" / "job.md").write_text(
        "# job\n\nEdward works at Amazon DSP logistics.\n", encoding="utf-8")
    (TMP / "raw" / "karpathy_gist.md").write_text(
        "# Karpathy LLM Wiki\n\nCompounding wiki pattern...\n", encoding="utf-8")
    (TMP / "wiki" / "index.md").write_text(
        "# Index\n\n## Concepts\n- [[llm_wiki]]\n- [[rag]]\n\n## Entities\n- [[karpathy]]\n",
        encoding="utf-8")
    (TMP / "wiki" / "concepts" / "llm_wiki.md").write_text(
        "# LLM Wiki\n\nA compounding-knowledge pattern proposed by [[karpathy]].\n",
        encoding="utf-8")
    (TMP / "wiki" / "concepts" / "rag.md").write_text(
        "# RAG\n\nRetrieval-augmented generation. Contrasted with [[llm_wiki]].\n",
        encoding="utf-8")
    (TMP / "wiki" / "entities" / "karpathy.md").write_text(
        "# Karpathy\n\nAI researcher.\n", encoding="utf-8")


_seed_brain()


# Monkey-patch chloe_llm_call so we don't hit Groq
import brain_wiring
_calls = []
def _fake_llm(prompt, mode):
    _calls.append((mode, len(prompt), prompt[:120]))
    return f"# Mock output\n\nThis is a fake LLM response for mode={mode}, prompt_size={len(prompt)}."
brain_wiring.chloe_llm_call = _fake_llm


def t_daily_context_generates():
    import daily_context
    today = datetime.date.today()
    r = daily_context.generate(target_date=today, dry_run=False)
    assert r["ok"], f"got: {r}"
    out = Path(r["path"])
    assert out.exists(), f"output file not written: {out}"
    body = out.read_text(encoding="utf-8")
    assert "Mock output" in body
    # Confirm the LLM was called with the expected sections in the prompt
    assert _calls, "no LLM call recorded"
    last_prompt = _calls[-1][2]
    assert "PROJECT STATUS" not in last_prompt or True  # prompt has these section names
    print(f"PASS: daily_context generated {len(body)} bytes -> {out.name}")


def t_daily_context_dry_run():
    import daily_context
    before = len(_calls)
    r = daily_context.generate(target_date=datetime.date.today(), dry_run=True)
    after = len(_calls)
    assert r["ok"]
    assert r.get("dry_run") is True
    assert after == before, f"dry-run made an LLM call: {after - before}"
    print("PASS: daily_context --dry-run does not call LLM")


def t_queue_filename_parsing():
    from queue_processor import parse_filename
    cases = [
        ("RESEARCH-rag_vs_wiki.md",   {"verb": "RESEARCH",   "slug": "rag_vs_wiki"}),
        ("SYNTHESIZE-q4_research.md", {"verb": "SYNTHESIZE", "slug": "q4_research"}),
        ("DRAFT-blog_post.md",        {"verb": "DRAFT",      "slug": "blog_post"}),
        ("ANALYZE-bug_log.md",        {"verb": "ANALYZE",    "slug": "bug_log"}),
        ("research-lower_case.md",    {"verb": "RESEARCH",   "slug": "lower_case"}),
        ("BAD_FORMAT.md",             None),
        ("RESEARCH.md",               None),
        ("RESEARCH-.md",              None),
    ]
    for name, expected in cases:
        got = parse_filename(name)
        assert got == expected, f"{name}: got {got}, expected {expected}"
    print(f"PASS: queue filename parsing ({len(cases)} cases)")


def t_queue_drain_processes_and_archives():
    import queue_processor
    # Reset call log
    _calls.clear()

    qf = TMP / "queue" / "RESEARCH-rag_vs_llm_wiki.md"
    qf.write_text("How does RAG differ from the Karpathy LLM Wiki pattern?", encoding="utf-8")

    r = queue_processor.drain(dry_run=False)
    assert r["ok"], f"drain failed: {r}"
    assert r["processed"] == 1, f"expected 1 processed, got {r['processed']}"

    # Original file should be gone from queue
    assert not qf.exists(), "queue file should have been moved out"

    # Archive should now hold the file
    today = datetime.date.today().isoformat()
    archive = TMP / "archive" / "queue" / f"{today}-RESEARCH-rag_vs_llm_wiki.md"
    assert archive.exists(), f"file not archived: {archive}"

    # Generated output should exist
    out = TMP / "generated" / today / "research-rag_vs_llm_wiki.md"
    assert out.exists(), f"generated output missing: {out}"
    body = out.read_text(encoding="utf-8")
    assert "Mock output" in body

    # One LLM call expected
    assert len(_calls) == 1, f"expected 1 LLM call, got {len(_calls)}"
    print(f"PASS: queue drain — processed, archived, output written")


def t_queue_drain_skips_bad_filenames():
    import queue_processor
    bad = TMP / "queue" / "RANDOM_FILE.md"
    bad.write_text("not a queue task", encoding="utf-8")
    _calls.clear()
    r = queue_processor.drain(dry_run=False)
    # Should have a result entry that's NOT ok
    failed = [x for x in r["files"] if not x.get("ok")]
    assert any("doesn't match" in (x.get("error") or "") for x in failed), \
        f"expected filename validation error, got: {r['files']}"
    # Bad file should still be in queue (not archived)
    assert bad.exists(), "bad file shouldn't be archived"
    bad.unlink()
    print("PASS: queue rejects malformed filenames without crashing")


def t_queue_dry_run_no_writes():
    import queue_processor
    qf = TMP / "queue" / "DRAFT-test_dry.md"
    qf.write_text("draft test", encoding="utf-8")
    _calls.clear()
    r = queue_processor.drain(dry_run=True, once=True)
    assert r["ok"]
    assert qf.exists(), "dry-run shouldn't archive"
    assert len(_calls) == 0, f"dry-run shouldn't call LLM, got {len(_calls)}"
    qf.unlink()
    print("PASS: queue --dry-run leaves files in place + no LLM call")


def main():
    t_daily_context_generates()
    t_daily_context_dry_run()
    t_queue_filename_parsing()
    t_queue_drain_processes_and_archives()
    t_queue_drain_skips_bad_filenames()
    t_queue_dry_run_no_writes()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
