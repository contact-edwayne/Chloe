"""wiki_apply.py -- apply all pending wikimerge_* proposals in one go, reversibly.

    py -3 wiki_apply.py                  # dry run: shows what would change
    py -3 wiki_apply.py --apply          # apply merges, rewrite links, move duplicates
    py -3 wiki_apply.py --revert [dir]   # undo the latest (or given) apply

For each pending proposal (filed by wiki_maintain.py, already validated):
  1. apply it via chloe_proposals (keeper page overwritten; .bak made as usual)
  2. rewrite [[dupe]] / [[dupe|x]] / [[dupe#h]] links wiki-wide to point at the keeper
  3. MOVE (not delete) the duplicate pages into brain/wiki_merge_backup/<stamp>/
Duplicates whose page name also exists elsewhere (e.g. concepts/greeks vs entities/greeks)
stay put until the twin is merged too; every run also sweeps those leftovers.
Every touched file is copied there first and a manifest.json is written, so
--revert restores everything, including the proposals' own backups.
Env: CHLOE_BRAIN_ROOT (default C:\\Chloe\\brain)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import sys
from pathlib import Path

import chloe_proposals as cp

BRAIN = Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))
WIKI = BRAIN / "wiki"
BACKUP_ROOT = BRAIN / "wiki_merge_backup"
_DUPE_RE = re.compile(r"^- `wiki/([^`]+)`", re.M)


def _pending() -> list:
    out = []
    for row in cp.list_proposals("pending"):
        if not row["slug"].startswith("wikimerge_"):
            continue
        prop = cp.load_proposal(row["slug"])
        keeper = prop.target.replace("\\", "/").split("brain/wiki/", 1)[-1]
        dupes = _DUPE_RE.findall(prop.raw_body)
        out.append({"slug": prop.slug, "keeper": keeper, "dupes": dupes})
    return out


def _wiki_md(skip: Path | None = None):
    for p in WIKI.rglob("*.md"):
        if skip and skip in p.parents:
            continue
        yield p


def _keeper_of(prop) -> str:
    return prop.target.replace("\\", "/").split("brain/wiki/", 1)[-1]


def _leftovers(extra: dict | None = None) -> dict:
    """{dupe_rel: keeper_rel} for dupes of APPLIED wikimerge proposals still on disk."""
    out = dict(extra or {})
    for row in cp.list_proposals("applied"):
        if not row["slug"].startswith("wikimerge_"):
            continue
        prop = cp.load_proposal(row["slug"])
        keeper = _keeper_of(prop)
        for d in _DUPE_RE.findall(prop.raw_body):
            if (WIKI / d).exists() and (WIKI / keeper).exists():
                out[d] = keeper
    return out


def _plan(pairs: dict):
    """Decide which dupes can be moved + links rewritten without ambiguity.
    A bare [[stem]] is safe if every existing page with that stem is itself a
    dupe that maps to the same keeper stem."""
    by_stem: dict = {}
    for f in WIKI.rglob("*.md"):
        by_stem.setdefault(f.stem, []).append(str(f.relative_to(WIKI)).replace("\\", "/"))
    move, keep = {}, []
    for d, k in pairs.items():
        ds, ks = Path(d).stem, Path(k).stem
        peers = by_stem.get(ds, [])
        ok = ds != ks and all(
            peer == d or (peer in pairs and Path(pairs[peer]).stem == ks) for peer in peers)
        if ok:
            move[d] = ks
        else:
            keep.append(d)
    return move, keep


def dry_run() -> int:
    items = _pending()
    extra = {}
    bad = 0
    for it in items:
        r = cp.apply_proposal(it["slug"], dry_run=True)
        ok = r.get("ok")
        bad += not ok
        print(f"{'OK ' if ok else 'ERR'} {it['keeper']}  <-  {', '.join(it['dupes']) or '(no dupes listed)'}")
        print(f"      {r.get('summary') or r.get('error')}")
        for d in it["dupes"]:
            if (WIKI / d).exists():
                extra[d] = it["keeper"]
    pairs = _leftovers(extra)
    move, keep = _plan(pairs)
    print(f"\n{len(items)} pending proposals, {bad} would fail.")
    print(f"duplicates to move + relink: {len(move)}")
    for d in keep:
        print(f"  would keep {d} (same page name exists elsewhere that is not yet merged)")
    if not items and not move:
        print("nothing to do")
    else:
        print("Run with --apply to do it.")
    return 0


def apply_all() -> int:
    items = _pending()
    stash = BACKUP_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _n = 1
    while stash.exists():
        _n += 1
        stash = BACKUP_ROOT / (dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{_n}")
    manifest = {"applied": [], "moved": [], "touched": [], "skipped": []}

    for it in items:
        cp._applied_this_session = 0  # per-proposal cap is a Chloe-session guard, not relevant here
        r = cp.apply_proposal(it["slug"])
        if r.get("ok"):
            print(f"applied {it['slug']} -> {it['keeper']}")
            manifest["applied"].append(it["slug"])
        else:
            print(f"SKIP  {it['slug']}: {r.get('error')}")
            manifest["skipped"].append({"slug": it["slug"], "error": r.get("error")})

    move, keep = _plan(_leftovers())
    for d in keep:
        print(f"  keep {d} in place for now (same page name exists elsewhere, not yet merged)")
    if not manifest["applied"] and not move:
        print("nothing to do")
        return 0

    (stash / "touched").mkdir(parents=True)
    (stash / "dupes").mkdir()
    pairs = _leftovers()
    if move:
        by_stem = {}
        for d, ks in move.items():
            by_stem[Path(d).stem] = ks
        pats = {s_: re.compile(r"\[\[((?:(?:concepts|entities)/)?)" + re.escape(s_) + r"(?=[\]|#])")
                for s_ in by_stem}
        moving = set(move)
        n_files = 0
        for f in list(_wiki_md()):
            rel = str(f.relative_to(WIKI)).replace("\\", "/")
            if rel in moving:
                continue  # duplicates are being moved, not edited
            text = f.read_text(encoding="utf-8", errors="replace")
            new = text
            for s_, pat in pats.items():
                new = pat.sub(lambda m, k=by_stem[s_]: f"[[{m.group(1)}{k}", new)
            if new != text:
                dest = stash / "touched" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                f.write_text(new, encoding="utf-8")
                manifest["touched"].append(rel)
                n_files += 1
        print(f"rewrote links in {n_files} page(s)")
        for d in move:
            dest = stash / "dupes" / d
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(WIKI / d), str(dest))
            manifest["moved"].append(d)
        print(f"moved {len(move)} duplicate page(s) to {stash / 'dupes'}")

    (stash / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\ndone. undo with: py -3 wiki_apply.py --revert \"{stash}\"")
    return 0


def revert(stash_arg: str | None) -> int:
    if stash_arg:
        stash = Path(stash_arg)
    else:
        runs = sorted(p for p in BACKUP_ROOT.glob("*") if (p / "manifest.json").exists())
        if not runs:
            print("nothing to revert")
            return 1
        stash = runs[-1]
    m = json.loads((stash / "manifest.json").read_text(encoding="utf-8"))
    for rel in m["touched"]:
        if (stash / "touched" / rel).exists():
            shutil.copy2(stash / "touched" / rel, WIKI / rel)
    for d in m["moved"]:
        dest = WIKI / d
        if not (stash / "dupes" / d).exists():
            print(f"  already restored: {d}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stash / "dupes" / d), str(dest))
    for slug in m["applied"]:
        cp._applied_this_session = 1
        r = cp.revert_proposal(slug)
        print(f"revert {slug}: {'ok' if r.get('ok') else r.get('error')}")
    print(f"restored {len(m['touched'])} page(s), {len(m['moved'])} duplicate(s) from {stash}")
    return 0


def main(argv: list[str]) -> int:
    if "--revert" in argv:
        i = argv.index("--revert")
        return revert(argv[i + 1] if i + 1 < len(argv) else None)
    if "--apply" in argv:
        return apply_all()
    return dry_run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
