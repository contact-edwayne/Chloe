"""wiki_maintain.py -- nightly wiki dedup via the Claude Batch API (50% off).

Never writes to the wiki. Finds duplicate clusters (deterministic signals only:
_vN version-suffix families + wiki_dedup.canonical_slug matches in concepts/
and entities/), asks Claude to merge each cluster into one page, validates the
result, and files it as a kind=full proposal under brain/proposals/ so the
existing /apply_proposal and /revert_proposal flow (backups, caps) applies.
The duplicate files themselves are listed in the proposal rationale; delete
them by hand after applying.

Batches can take up to 24h, so the job is two-phase and idempotent:

    python wiki_maintain.py submit [--limit 25] [--dry-run]
    python wiki_maintain.py collect
    python wiki_maintain.py run        # collect pending, then submit (nightly)

Env: ANTHROPIC_API_KEY, CHLOE_BRAIN_ROOT (default C:\\Chloe\\brain),
     CHLOE_WIKI_MAINTAIN_MODEL (default claude-sonnet-5-5)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import datetime as dt
from collections import defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from wiki_dedup import canonical_slug

BRAIN_ROOT = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
WIKI = BRAIN_ROOT / "wiki"
STATE = BRAIN_ROOT / "wiki_maintain_state.json"
MODEL = os.environ.get("CHLOE_WIKI_MAINTAIN_MODEL", "claude-sonnet-5-5")

TOPIC_DIRS = ("concepts", "entities")
MAX_PAGES_PER_CLUSTER = 6
MAX_CHARS_PER_CLUSTER = 60_000
MIN_KEEP_RATIO = 0.6   # merged must retain >= 60% of the largest source page's length

_VERSION_RE = re.compile(r"^(.*)_v\d+$")

SYSTEM = """You merge duplicate pages of a personal markdown wiki into ONE page.
Rules:
- Keep every distinct fact, date, number, source citation and [[wikilink]]. Drop only true repetition.
- Never invent anything. If pages conflict, keep both claims and mark the conflict inline as "(conflict: ...)".
- Keep the keeper page's YAML frontmatter shape (--- ... --- with title:), updating nothing but what the merge requires.
- Keep dated sections/revision history in chronological order.
- Choose as keeper the page whose filename is the canonical, un-suffixed topic name when one exists.
Return ONLY a JSON object: {"keeper": "<one of the given paths exactly>", "merged_markdown": "<full page>", "notes": "<one sentence>"}"""


# ─── state ─────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"batches": {}, "done": []}


def _save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def _client():
    import anthropic
    return anthropic.Anthropic()


# ─── clustering (deterministic, no LLM, no embeddings) ─────────────────────

def find_clusters() -> list[dict]:
    """Each cluster: {id, kind, paths: [wiki-relative posix paths]}."""
    by_version: dict[str, list[str]] = defaultdict(list)
    by_canon: dict[str, list[str]] = defaultdict(list)
    for sub in TOPIC_DIRS:
        d = WIKI / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            rel = f"{sub}/{p.name}"
            m = _VERSION_RE.match(p.stem)
            base = m.group(1) if m else p.stem
            by_version[f"{sub}/{base}"].append(rel)
            ck = canonical_slug(base)
            if ck:
                by_canon[f"{sub}/{ck}"].append(rel)

    cands: dict[frozenset, dict] = {}
    for kind, groups in (("version-suffix", by_version), ("canonical-key", by_canon)):
        for key, paths in groups.items():
            fs = frozenset(paths)
            if len(paths) >= 2 and fs not in cands:
                cid = hashlib.sha1("|".join(sorted(paths)).encode()).hexdigest()[:10]
                cands[fs] = {"id": cid, "kind": kind, "paths": sorted(paths)}
    # Greedy, largest first, no shared pages: a page lands in at most one
    # cluster, so two proposals never target/replace the same file.
    out, taken = [], set()
    for fs, c in sorted(cands.items(), key=lambda kv: (-len(kv[0]), kv[1]["id"])):
        if fs & taken:
            continue
        taken |= fs
        out.append(c)
    return out


def _read_cluster(c: dict) -> dict[str, str] | None:
    pages = {}
    for rel in c["paths"][:MAX_PAGES_PER_CLUSTER]:
        pages[rel] = (WIKI / rel).read_text(encoding="utf-8", errors="replace")
    if sum(len(t) for t in pages.values()) > MAX_CHARS_PER_CLUSTER:
        return None
    return pages


# ─── submit ────────────────────────────────────────────────────────────────

def _user_prompt(pages: dict[str, str]) -> str:
    parts = [f"Merge these {len(pages)} pages.\n"]
    for rel, text in pages.items():
        parts.append(f"=== PATH: {rel} ===\n{text}\n=== END ===\n")
    return "\n".join(parts)


def submit(limit: int, dry_run: bool) -> None:
    state = _load_state()
    pending = {cid for b in state["batches"].values() for cid in b["clusters"]}
    busy = {p for b in state["batches"].values() for c in b["clusters"].values() for p in c["paths"]}
    busy |= {p for c in state.get("done_paths", []) for p in [c]}
    todo = [c for c in find_clusters()
            if c["id"] not in state["done"] and c["id"] not in pending
            and not busy.intersection(c["paths"])]
    runnable, skipped = [], 0
    for c in todo:
        pages = _read_cluster(c)
        if pages is None:
            skipped += 1
            continue
        runnable.append((c, pages))
    runnable = runnable[:limit]
    chars = sum(len(t) for _, p in runnable for t in p.values())
    print(f"clusters found: {len(todo)} new | submitting: {len(runnable)} | "
          f"skipped (too large): {skipped} | ~{chars // 4:,} input tokens")
    if dry_run or not runnable:
        for c, _ in runnable:
            print(f"  [{c['kind']}] {c['id']}: {', '.join(c['paths'])}")
        return

    requests = [{
        "custom_id": f"c_{c['id']}",
        "params": {
            "model": MODEL,
            "max_tokens": 16000,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": _user_prompt(pages)}],
        },
    } for c, pages in runnable]
    batch = _client().messages.batches.create(requests=requests)
    state["batches"][batch.id] = {
        "submitted": dt.datetime.now().isoformat(timespec="seconds"),
        "clusters": {c["id"]: c for c, _ in runnable},
    }
    _save_state(state)
    print(f"submitted batch {batch.id} ({len(requests)} clusters)")


# ─── collect ───────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def validate(result: dict, cluster: dict, pages: dict[str, str]) -> str | None:
    """Return an error string, or None if the merge is acceptable."""
    keeper = result.get("keeper")
    merged = result.get("merged_markdown")
    if keeper not in cluster["paths"]:
        return f"keeper {keeper!r} not in cluster"
    if not isinstance(merged, str) or not merged.startswith("---"):
        return "merged page missing frontmatter"
    head = merged.split("\n---", 1)[0]
    if "title:" not in head or "\n---" not in merged[3:]:
        return "frontmatter lacks title: or closing ---"
    biggest = max(len(t) for t in pages.values())
    if len(merged) < MIN_KEEP_RATIO * biggest:
        return f"merged ({len(merged)} chars) < {MIN_KEEP_RATIO:.0%} of largest source ({biggest})"
    src_links = {l for t in pages.values() for l in re.findall(r"\[\[([^\]|]+)", t)}
    lost = [l for l in src_links if f"[[{l}" not in merged]
    if lost:
        return f"dropped wikilinks: {', '.join(sorted(lost)[:5])}"
    return None


def _file_proposal(result: dict, cluster: dict) -> Path:
    import chloe_proposals as cp
    keeper = result["keeper"]
    dupes = [p for p in cluster["paths"] if p != keeper]
    rationale = (
        f"Merge of {len(cluster['paths'])} duplicate wiki pages ({cluster['kind']} cluster "
        f"`{cluster['id']}`) into `{keeper}`. {result.get('notes', '')}\n\n"
        "After applying, delete these now-redundant files by hand:\n"
        + "\n".join(f"- `wiki/{d}`" for d in dupes)
    )
    return cp.create_proposal(
        target=f"brain/wiki/{keeper}",
        kind="full",
        rationale=rationale,
        body=result["merged_markdown"],
        test_plan="Open the page; confirm facts from each source survive and wikilinks resolve. "
                  "Run lint_weekly.py afterwards to confirm no new orphans.",
        slug=f"wikimerge_{cluster['id']}",
        title=f"Wiki merge: {Path(keeper).stem}",
    )


def collect() -> None:
    state = _load_state()
    client = _client()
    for bid in list(state["batches"]):
        b = client.messages.batches.retrieve(bid)
        if b.processing_status != "ended":
            print(f"batch {bid}: {b.processing_status}")
            continue
        info = state["batches"][bid]["clusters"]
        ok = bad = 0
        for r in client.messages.batches.results(bid):
            cid = r.custom_id.removeprefix("c_")
            cluster = info.get(cid)
            if cluster is None:
                continue
            state["done"].append(cid)
            state.setdefault("done_paths", []).extend(cluster["paths"])
            if r.result.type != "succeeded":
                print(f"  {cid}: {r.result.type}")
                bad += 1
                continue
            try:
                text = "".join(b_.text for b_ in r.result.message.content if b_.type == "text")
                result = _extract_json(text)
                pages = {p: (WIKI / p).read_text(encoding="utf-8", errors="replace")
                         for p in cluster["paths"][:MAX_PAGES_PER_CLUSTER]}
                err = validate(result, cluster, pages)
            except Exception as e:  # bad JSON, missing file, ...
                err = f"{type(e).__name__}: {e}"
            if err:
                print(f"  {cid}: REJECTED -- {err}")
                bad += 1
                continue
            path = _file_proposal(result, cluster)
            print(f"  {cid}: proposal -> {path.name}")
            ok += 1
        del state["batches"][bid]
        print(f"batch {bid}: {ok} proposals, {bad} rejected/failed")
    _save_state(state)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("submit", "collect", "run"):
        print(__doc__)
        return 2
    cmd = argv[0]
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 25
    dry = "--dry-run" in argv
    if cmd in ("collect", "run"):
        collect()
    if cmd in ("submit", "run"):
        submit(limit, dry)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
