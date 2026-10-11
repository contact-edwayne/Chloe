"""wiki_apply.py -- apply all pending wikimerge_* proposals in one go, reversibly.

    py -3 wiki_apply.py                  # dry run: shows what would change
    py -3 wiki_apply.py --apply          # apply merges, rewrite links, move duplicates
    py -3 wiki_apply.py --revert [dir]   # undo the latest (or given) apply

For each pending proposal (filed by wiki_maintain.py, already validated):
  1. apply it via chloe_proposals (keeper page overwritten; .bak made as usual)
  2. rewrite [[dupe]] / [[dupe|x]] / [[dupe#h]] links wiki-wide to point at the keeper
  3. MOVE (not delete) the duplicate pages into brain/wiki_merge_backup/<stamp>/
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


def _stem_count(stem: str) -> int:
    return sum(1 for p in WIKI.rglob("*.md") if p.stem == stem)


def dry_run() -> int:
    items = _pending()
    if not items:
        print("no pending wikimerge_ proposals")
        return 0
    bad = 0
    for it in items:
        r = cp.apply_proposal(it["slug"], dry_run=True)
        ok = r.get("ok")
        bad += not ok
        print(f"{'OK ' if ok else 'ERR'} {it['keeper']}  <-  {', '.join(it['dupes']) or '(no dupes listed)'}")
        print(f"      {r.get('summary') or r.get('error')}")
    print(f"\n{len(items)} proposals, {bad} would fail. Run with --apply to do it.")
    return 0


def apply_all() -> int:
    items = _pending()
    if not items:
        print("no pending wikimerge_ proposals")
        return 0
    stash = BACKUP_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (stash / "touched").mkdir(parents=True)
    (stash / "dupes").mkdir()
    manifest = {"applied": [], "moved": [], "touched": [], "skipped": []}

    applied = []
    for it in items:
        cp._applied_this_session = 0  # per-proposal cap is a Chloe-session guard, not relevant here
        r = cp.apply_proposal(it["slug"])
        if r.get("ok"):
            print(f"applied {it['slug']} -> {it['keeper']}")
            manifest["applied"].append(it["slug"])
            applied.append(it)
        else:
            print(f"SKIP  {it['slug']}: {r.get('error')}")
            manifest["skipped"].append({"slug": it["slug"], "error": r.get("error")})

    # link rewrite map: dupe stem -> keeper stem (only unambiguous stems)
    mapping = {}
    for it in applied:
        kstem = Path(it["keeper"]).stem
        for d in it["dupes"]:
            dstem = Path(d).stem
            if dstem == kstem or _stem_count(dstem) != 1 or not (WIKI / d).exists():
                print(f"  keep {d} in place (ambiguous or missing); links to it untouched")
                continue
            mapping[dstem] = (kstem, d)

    if mapping:
        pats = {s: re.compile(r"\[\[((?:(?:concepts|entities)/)?)" + re.escape(s) + r"(?=[\]|#])")
                for s in mapping}
        n_files = 0
        for f in list(_wiki_md(skip=BACKUP_ROOT)):
            rel = f.relative_to(WIKI)
            if any(str(rel).replace("\\", "/") == d for _, d in mapping.values()):
                continue  # duplicates are being moved, not edited
            text = f.read_text(encoding="utf-8", errors="replace")
            new = text
            for s, pat in pats.items():
                new = pat.sub(lambda m, k=mapping[s][0]: f"[[{m.group(1)}{k}", new)
            if new != text:
                dest = stash / "touched" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                f.write_text(new, encoding="utf-8")
                manifest["touched"].append(str(rel).replace("\\", "/"))
                n_files += 1
        print(f"rewrote links in {n_files} page(s)")
        for s, (k, d) in mapping.items():
            src = WIKI / d
            dest = stash / "dupes" / d
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            manifest["moved"].append(d)
        print(f"moved {len(mapping)} duplicate page(s) to {stash / 'dupes'}")

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
        shutil.copy2(stash / "touched" / rel, WIKI / rel)
    for d in m["moved"]:
        dest = WIKI / d
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
