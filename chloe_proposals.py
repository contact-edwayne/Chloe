"""Code-proposal pipeline for Chloe self-modification (Tier 1).

Chloe (or the Friday meta-review, or any Cowork job) writes a markdown file
under `C:\\Chloe\\brain\\proposals\\code_<YYYY-MM-DD>_<slug>.md` describing a
proposed change to her own source. Ed reviews and applies via the
`/apply_proposal <slug>` slash command. Backed by `/revert_proposal <slug>`.

Tier 1 design constraints (from chloe_handoff.md NEXT-SESSION PRIORITY):
  - target path must resolve under jarvis/ or C:\\Chloe\\brain\\
  - .py targets must ast.parse before write
  - timestamped backup so re-applies on the same day don't clobber
    (lesson #13 — the date-only stamp pattern bit us before)
  - max 5 proposals applied without restart (forces verification cycle)
  - kind="diff" (unified diff) or kind="full" (whole-file replacement)

No external deps — just stdlib. Unified-diff applier is hand-rolled and
intentionally strict: hunk context must match exactly or we refuse to
apply that hunk.
"""

from __future__ import annotations

import ast
import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ─── Roots (read lazily to honour test overrides) ──────────────────────────

def _brain_root() -> Path:
    return Path(os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain"))


def _jarvis_root() -> Path:
    # The directory this module lives in IS jarvis/. Anchor off __file__ so
    # tests + relocations don't drift.
    return Path(__file__).resolve().parent


def proposals_dir() -> Path:
    p = _brain_root() / "proposals"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─── Safety policy ─────────────────────────────────────────────────────────

# Path substrings that REFUSE — checked against the resolved absolute path.
_REFUSED_PATH_PARTS: tuple[str, ...] = (
    "__pycache__",
    ".bak.",
    ".git",
    "venv",
    "venv_py314",
    "dist",
    "node_modules",
    ".vscode",
    ".idea",
    "secrets",          # never touch C:\Chloe\secrets via proposals
)

# Cap on applies per Chloe process lifetime. Forces a restart between
# batches so behaviour changes get verified before more queue up.
MAX_APPLIES_PER_SESSION = 5

# Module-level counter, reset on import (i.e. on Chloe restart).
_applied_this_session: int = 0


def _stamp() -> str:
    """Timestamp suitable for filenames. Includes time so same-day re-runs
    don't collide (lesson #13)."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _today() -> str:
    return _dt.date.today().isoformat()


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _resolve_target(target_str: str) -> Path:
    """Resolve a target spec to an absolute Path.

    Accepts:
      - absolute paths
      - jarvis-relative (e.g. "brain_wiring.py")
      - "jarvis/foo.py" / "brain/bar.md" prefix shorthand
    """
    t = (target_str or "").strip()
    if not t:
        raise ValueError("empty target path")

    # Normalize slashes — proposal markdown is human-typed, may be either
    # direction depending on origin.
    t = t.replace("\\", "/")

    p = Path(t)
    if p.is_absolute():
        return p

    # "jarvis/..." prefix → jarvis root
    if t.startswith("jarvis/"):
        return _jarvis_root() / t[len("jarvis/"):]
    # "brain/..." prefix → brain root
    if t.startswith("brain/"):
        return _brain_root() / t[len("brain/"):]
    # Bare filename or relative path defaults to jarvis (most common).
    return _jarvis_root() / t


def _check_target_allowed(target: Path) -> Optional[str]:
    """Return an error string if target is outside the whitelist or matches
    a refused pattern. Returns None when target is allowed."""
    rp = target.resolve()
    if not (_is_under(rp, _jarvis_root()) or _is_under(rp, _brain_root())):
        return (f"target outside whitelist (must be under jarvis/ or "
                f"{_brain_root()}): {rp}")

    rp_str = str(rp).replace("\\", "/").lower()
    for bad in _REFUSED_PATH_PARTS:
        # Match on path-segment boundary to avoid false positives like a file
        # named "venv_notes.md" — require '/' on either side OR end-of-string.
        needle = bad.lower()
        idx = 0
        while True:
            idx = rp_str.find(needle, idx)
            if idx == -1:
                break
            before_ok = idx == 0 or rp_str[idx - 1] == "/"
            after_pos = idx + len(needle)
            after_ok = (after_pos == len(rp_str)
                        or rp_str[after_pos] in ("/", "."))
            if before_ok and after_ok:
                return f"target matches refused pattern '{bad}': {rp}"
            idx += 1
    return None


# ─── Frontmatter parsing ───────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tiny YAML-ish parser — `key: value` pairs only, no nesting.

    Avoids pulling PyYAML as a dep. Sufficient for our proposal frontmatter.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_body, rest = m.group(1), m.group(2)
    fm: dict[str, str] = {}
    for line in fm_body.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()
    return fm, rest


def _dump_frontmatter(fm: dict, body: str) -> str:
    lines = ["---"]
    for k, v in fm.items():
        if v is None or v == "":
            lines.append(f"{k}:")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body


# ─── Section parsing ───────────────────────────────────────────────────────

_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _extract_sections(body: str) -> dict[str, str]:
    """Split a markdown body by `## Heading` into a dict keyed by lowercased
    heading text. Section values include only the body between this heading
    and the next H2 (or EOF), with surrounding whitespace stripped."""
    headings = list(_H2_RE.finditer(body))
    sections: dict[str, str] = {}
    for i, m in enumerate(headings):
        key = m.group(1).strip().lower()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections[key] = body[start:end].strip()
    return sections


def _strip_fenced_code(text: str, lang_hint: str = "") -> str:
    """If text is wrapped in a fenced code block, unwrap it. Tolerates
    optional language hint and trailing newlines."""
    t = text.strip()
    fence_re = re.compile(
        r"\A```(?:" + re.escape(lang_hint) + r"|\w*)\s*\n(.*?)\n```\s*\Z",
        re.DOTALL,
    )
    m = fence_re.match(t)
    if m:
        return m.group(1)
    return t


# ─── Unified-diff applier (minimal, strict) ────────────────────────────────

_HUNK_HEADER_RE = re.compile(
    r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@",
)


@dataclass
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]  # each starts with ' ', '+', '-', or '\\'


def _parse_unified_diff(diff_text: str) -> list[_Hunk]:
    """Parse a unified-diff body (already file-header-stripped or with
    headers — we just skip non-hunk lines until the first @@). Returns
    list of hunks, in order."""
    hunks: list[_Hunk] = []
    lines = diff_text.splitlines()
    i = 0
    current: Optional[_Hunk] = None
    while i < len(lines):
        line = lines[i]
        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            current = _Hunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2)) if m.group(2) else 1,
                new_start=int(m.group(3)),
                new_count=int(m.group(4)) if m.group(4) else 1,
                lines=[],
            )
            i += 1
            continue
        if current is not None:
            if line.startswith(("+", "-", " ", "\\")):
                current.lines.append(line)
            elif line == "":
                # Treat bare empty lines as unchanged-context-empty.
                current.lines.append(" ")
            else:
                # End of hunk body; loop will pick up next @@ or EOF.
                pass
        i += 1
    if current is not None:
        hunks.append(current)
    return hunks


def _apply_hunks(original: str, hunks: list[_Hunk]) -> str:
    """Apply hunks to original text. Strict: hunk context lines + removed
    lines must EXACTLY match the original at the indicated line range.

    Returns the patched text. Raises ValueError on mismatch.
    """
    orig_lines = original.splitlines(keepends=False)
    # Build the new file by walking the original and splicing each hunk.
    result: list[str] = []
    cursor = 0  # 0-based index into orig_lines
    for hi, hunk in enumerate(hunks):
        # Hunks are 1-based; convert to 0-based.
        old_idx = hunk.old_start - 1 if hunk.old_count > 0 else hunk.old_start
        if old_idx < cursor:
            raise ValueError(
                f"hunk {hi + 1} starts at line {hunk.old_start} but cursor "
                f"is already past it at line {cursor + 1}"
            )
        # Copy original lines from cursor up to the hunk start unchanged.
        result.extend(orig_lines[cursor:old_idx])
        cursor = old_idx

        # Walk hunk lines; verify '-'/' ' against original and emit '+'/' '.
        for hl in hunk.lines:
            if not hl:
                continue
            tag, content = hl[0], hl[1:]
            if tag == "\\":
                # "\ No newline at end of file" marker — ignore.
                continue
            if tag == " " or tag == "-":
                if cursor >= len(orig_lines):
                    raise ValueError(
                        f"hunk {hi + 1}: ran off end of file while expecting "
                        f"{'context' if tag == ' ' else 'removal'}: {content!r}"
                    )
                if orig_lines[cursor] != content:
                    raise ValueError(
                        f"hunk {hi + 1} at orig line {cursor + 1}: "
                        f"{'context' if tag == ' ' else 'removal'} mismatch.\n"
                        f"  expected: {content!r}\n"
                        f"  actual:   {orig_lines[cursor]!r}"
                    )
                if tag == " ":
                    result.append(content)
                cursor += 1
            elif tag == "+":
                result.append(content)
            else:
                raise ValueError(f"hunk {hi + 1}: unknown line tag {tag!r}")
    # Trailing unchanged tail.
    result.extend(orig_lines[cursor:])
    # Preserve a trailing newline if the original had one.
    out = "\n".join(result)
    if original.endswith("\n"):
        out += "\n"
    return out


# ─── Proposal loading ──────────────────────────────────────────────────────

_CODE_PROPOSAL_RE = re.compile(r"^code_(\d{4}-\d{2}-\d{2})_(.+)\.md$")


@dataclass
class Proposal:
    path: Path
    slug: str
    date: str
    frontmatter: dict
    sections: dict[str, str]
    raw_body: str

    @property
    def target(self) -> str:
        return self.frontmatter.get("target", "")

    @property
    def kind(self) -> str:
        return self.frontmatter.get("kind", "diff").strip().lower()

    @property
    def status(self) -> str:
        return self.frontmatter.get("status", "pending").strip().lower()


def _find_proposal_path(slug: str) -> Optional[Path]:
    """Find newest `code_*_<slug>.md` in proposals_dir() matching slug.

    Slug match: try exact match on the trailing slug segment first, then
    case-insensitive contains as a courtesy fallback (Chloe is friendly).
    """
    candidates: list[tuple[float, Path]] = []
    for p in proposals_dir().glob("code_*.md"):
        m = _CODE_PROPOSAL_RE.match(p.name)
        if not m:
            continue
        if m.group(2) == slug:
            candidates.append((p.stat().st_mtime, p))
    if not candidates:
        # Fallback: case-insensitive contains.
        for p in proposals_dir().glob("code_*.md"):
            m = _CODE_PROPOSAL_RE.match(p.name)
            if not m:
                continue
            if slug.lower() in m.group(2).lower():
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_proposal(slug: str) -> Proposal:
    path = _find_proposal_path(slug)
    if path is None:
        raise FileNotFoundError(
            f"no proposal matching slug '{slug}' in {proposals_dir()}"
        )
    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    sections = _extract_sections(body)
    m = _CODE_PROPOSAL_RE.match(path.name)
    date = m.group(1) if m else ""
    real_slug = m.group(2) if m else slug
    return Proposal(
        path=path, slug=real_slug, date=date,
        frontmatter=fm, sections=sections, raw_body=body,
    )


def list_proposals(status: Optional[str] = None) -> list[dict]:
    """Return a sorted (newest first) list of code proposal summaries."""
    rows: list[tuple[float, dict]] = []
    for p in proposals_dir().glob("code_*.md"):
        m = _CODE_PROPOSAL_RE.match(p.name)
        if not m:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        fm, body = _parse_frontmatter(text)
        st = (fm.get("status") or "pending").strip().lower()
        if status and st != status.lower():
            continue
        # First H1 as title; fall back to slug.
        title_m = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else m.group(2)
        rows.append((p.stat().st_mtime, {
            "slug": m.group(2),
            "date": m.group(1),
            "path": str(p),
            "target": fm.get("target", ""),
            "kind": (fm.get("kind") or "diff").strip().lower(),
            "status": st,
            "title": title,
        }))
    rows.sort(reverse=True, key=lambda r: r[0])
    return [r[1] for r in rows]


# ─── Compute proposed file contents ───────────────────────────────────────

def _compute_new_body(prop: Proposal, original: str) -> tuple[str, str]:
    """Return (new_body, summary) for the proposed change.

    summary is a short human-readable description used in reply messages.
    """
    kind = prop.kind
    if kind == "full":
        body = prop.sections.get("full file") or prop.sections.get("full")
        if body is None:
            raise ValueError(
                "kind=full but no `## Full File` section found"
            )
        new_body = _strip_fenced_code(body)
        # Normalize: ensure trailing newline if original had one.
        if original.endswith("\n") and not new_body.endswith("\n"):
            new_body += "\n"
        old_lines = original.count("\n")
        new_lines = new_body.count("\n")
        delta = new_lines - old_lines
        sign = "+" if delta >= 0 else ""
        return new_body, (
            f"full-file replacement, {old_lines} → {new_lines} lines "
            f"({sign}{delta})"
        )

    if kind == "diff":
        diff_body = (
            prop.sections.get("diff")
            or prop.sections.get("unified diff")
            or prop.sections.get("patch")
        )
        if diff_body is None:
            raise ValueError(
                "kind=diff but no `## Diff` section found"
            )
        diff_text = _strip_fenced_code(diff_body, "diff")
        hunks = _parse_unified_diff(diff_text)
        if not hunks:
            raise ValueError("no @@ hunks found in diff section")
        new_body = _apply_hunks(original, hunks)
        adds = sum(1 for h in hunks for line in h.lines
                   if line.startswith("+"))
        rems = sum(1 for h in hunks for line in h.lines
                   if line.startswith("-"))
        return new_body, (
            f"unified diff: {len(hunks)} hunk(s), +{adds}/-{rems} lines"
        )

    raise ValueError(f"unknown proposal kind: {kind!r} (expected diff|full)")


# ─── Apply / revert ────────────────────────────────────────────────────────

def _ast_check(target: Path, new_body: str) -> Optional[str]:
    """Return error string on syntax failure; None on success or non-.py."""
    if target.suffix != ".py":
        return None
    try:
        ast.parse(new_body, filename=str(target))
    except SyntaxError as e:
        return f"ast.parse refused — SyntaxError at line {e.lineno}: {e.msg}"
    return None


def _backup_target(target: Path, slug: str) -> Path:
    bak = target.parent / f"{target.name}.bak.proposal_{slug}_{_stamp()}"
    bak.write_bytes(target.read_bytes())
    return bak


def apply_proposal(slug: str, dry_run: bool = False) -> dict:
    """Apply a code proposal.

    Returns a dict with keys:
      ok, slug, target, kind, summary, backup_path (when applied),
      message, error (when failed).
    """
    global _applied_this_session
    try:
        prop = load_proposal(slug)
    except FileNotFoundError as e:
        return {"ok": False, "slug": slug, "error": str(e)}

    if prop.status == "applied":
        return {
            "ok": False, "slug": prop.slug, "target": prop.target,
            "error": (f"proposal already applied at "
                      f"{prop.frontmatter.get('applied_at')}. "
                      f"Revert first with /revert_proposal {prop.slug}."),
        }

    target_str = prop.target
    if not target_str:
        return {"ok": False, "slug": prop.slug,
                "error": "proposal frontmatter missing `target:`"}

    try:
        target = _resolve_target(target_str)
    except ValueError as e:
        return {"ok": False, "slug": prop.slug, "error": str(e)}

    refusal = _check_target_allowed(target)
    if refusal:
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": refusal}

    if not target.exists():
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": f"target does not exist: {target}"}
    if not target.is_file():
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": f"target is not a regular file: {target}"}

    try:
        original = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": ("target is not UTF-8 text — refusing to apply "
                          "(proposals are markdown/source-code only)")}

    try:
        new_body, summary = _compute_new_body(prop, original)
    except ValueError as e:
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": f"{type(e).__name__}: {e}"}

    if new_body == original:
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": "proposed change is a no-op (new body == original)"}

    ast_err = _ast_check(target, new_body)
    if ast_err:
        return {"ok": False, "slug": prop.slug, "target": str(target),
                "error": ast_err}

    if dry_run:
        return {
            "ok": True, "dry_run": True, "slug": prop.slug,
            "target": str(target), "kind": prop.kind, "summary": summary,
            "message": (f"DRY RUN — would apply `{prop.slug}` to "
                        f"`{target}`: {summary}. "
                        f"No file written. Run without --dry-run to commit."),
        }

    if _applied_this_session >= MAX_APPLIES_PER_SESSION:
        return {
            "ok": False, "slug": prop.slug, "target": str(target),
            "error": (f"max {MAX_APPLIES_PER_SESSION} proposals already "
                      f"applied this session. Restart Chloe to verify the "
                      f"current batch before applying more."),
        }

    # Commit. Order: backup → write → frontmatter update → counter.
    bak_path = _backup_target(target, prop.slug)
    target.write_text(new_body, encoding="utf-8")

    fm = dict(prop.frontmatter)
    fm["status"] = "applied"
    fm["applied_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    fm["backup_path"] = str(bak_path)
    prop.path.write_text(_dump_frontmatter(fm, prop.raw_body), encoding="utf-8")

    _applied_this_session += 1

    restart_hint = (
        " Restart Chloe to pick up code changes."
        if str(target).endswith(".py") else ""
    )
    return {
        "ok": True, "slug": prop.slug, "target": str(target),
        "kind": prop.kind, "summary": summary,
        "backup_path": str(bak_path),
        "applied_count": _applied_this_session,
        "message": (f"Applied `{prop.slug}` to `{target}`: {summary}. "
                    f"Backup at `{bak_path.name}`.{restart_hint} "
                    f"Rollback: `/revert_proposal {prop.slug}`."),
    }


def revert_proposal(slug: str) -> dict:
    """Restore a previously-applied proposal's target from its backup."""
    global _applied_this_session
    try:
        prop = load_proposal(slug)
    except FileNotFoundError as e:
        return {"ok": False, "slug": slug, "error": str(e)}

    if prop.status != "applied":
        return {"ok": False, "slug": prop.slug,
                "error": (f"proposal status is `{prop.status}`, not "
                          f"`applied`. Nothing to revert.")}

    bak_str = prop.frontmatter.get("backup_path", "").strip()
    if not bak_str:
        return {"ok": False, "slug": prop.slug,
                "error": "proposal frontmatter missing `backup_path`"}
    bak = Path(bak_str)
    if not bak.exists():
        return {"ok": False, "slug": prop.slug,
                "error": f"backup file not found: {bak}"}

    target_str = prop.target
    target = _resolve_target(target_str)
    refusal = _check_target_allowed(target)
    if refusal:
        # Belt + suspenders — the original apply should have caught this.
        return {"ok": False, "slug": prop.slug, "error": refusal}

    target.write_bytes(bak.read_bytes())

    fm = dict(prop.frontmatter)
    fm["status"] = "reverted"
    fm["reverted_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    prop.path.write_text(_dump_frontmatter(fm, prop.raw_body), encoding="utf-8")

    # Free a session slot — revert means the apply slot is reusable.
    if _applied_this_session > 0:
        _applied_this_session -= 1

    return {
        "ok": True, "slug": prop.slug, "target": str(target),
        "message": (f"Reverted `{prop.slug}`: `{target}` restored from "
                    f"`{bak.name}`. Restart Chloe to pick up the revert "
                    f"if the target is a live .py module."),
    }


# ─── Helpers for proposal authors (Cowork jobs, /apply_proposal callers) ──

def create_proposal(
    *,
    target: str,
    kind: str,
    rationale: str,
    body: str,
    test_plan: str,
    rollback: str = "",
    slug: Optional[str] = None,
    title: Optional[str] = None,
) -> Path:
    """Write a properly-shaped `code_<date>_<slug>.md` proposal file under
    proposals_dir(). Returns the written path.

    `body` is the unified diff (kind="diff") or full-file body (kind="full").
    Fenced code-block wrapping is added automatically.
    """
    kind = kind.strip().lower()
    if kind not in ("diff", "full"):
        raise ValueError(f"kind must be 'diff' or 'full', got {kind!r}")
    if not slug:
        # Derive a default slug from the title or target.
        seed = title or target
        slug = re.sub(r"[^a-z0-9]+", "_", seed.lower()).strip("_")[:60] or "untitled"

    date = _today()
    fm = {
        "target": target.replace("\\", "/"),
        "kind": kind,
        "slug": slug,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "status": "pending",
        "applied_at": "",
        "backup_path": "",
    }

    section_name = "Full File" if kind == "full" else "Diff"
    fence_lang = "" if kind == "full" else "diff"
    body_stripped = body.strip("\n")
    rollback_text = rollback.strip() or f"`/revert_proposal {slug}`"

    md = (
        f"# {title or slug.replace('_', ' ')}\n\n"
        f"## Rationale\n\n{rationale.strip()}\n\n"
        f"## {section_name}\n\n```{fence_lang}\n{body_stripped}\n```\n\n"
        f"## Test plan\n\n{test_plan.strip()}\n\n"
        f"## Rollback\n\n{rollback_text}\n"
    )

    out = proposals_dir() / f"code_{date}_{slug}.md"
    out.write_text(_dump_frontmatter(fm, md), encoding="utf-8")
    return out


def session_state() -> dict:
    """Diagnostic view of the in-process apply counter."""
    return {
        "applied_this_session": _applied_this_session,
        "max_per_session": MAX_APPLIES_PER_SESSION,
        "remaining": MAX_APPLIES_PER_SESSION - _applied_this_session,
        "tokens": [_token_view(t) for t in _ACTIVE_TOKENS],
    }


# ─── Tier 2: confirm-token gated self-apply ────────────────────────────────
#
# Stage 1 (already shipped) requires Ed to type `/apply_proposal <slug>`
# for every apply. Stage 2 lets Ed mint a *confirm token* good for N
# applies in the next M minutes. Chloe (via MCP) or a Cowork job can then
# apply proposals on its own up to the cap — Ed approves a session
# instead of approving each apply.
#
# All Tier-1 safety rails STILL fire (path whitelist, ast.parse,
# timestamped backup, etc.). The token only relaxes the "human types the
# slash at apply time" gate. The session counter (MAX_APPLIES_PER_SESSION)
# still applies, and the token caps act ON TOP of that.

import secrets as _secrets
import threading as _threading
import time as _time


DEFAULT_TOKEN_MINUTES = 30
DEFAULT_TOKEN_APPLIES = 1
MAX_TOKEN_MINUTES = 120
MAX_TOKEN_APPLIES = 5

_TOKEN_LOCK = _threading.Lock()
_ACTIVE_TOKENS: list[dict] = []  # each: {token, applies_remaining, expires_at, issued_at}


def _token_view(t: dict) -> dict:
    """Render a token entry for status output. NEVER returns the raw
    token string — only the first 4 and last 4 chars for identification."""
    raw = t.get("token", "")
    if len(raw) > 8:
        masked = f"{raw[:4]}…{raw[-4:]}"
    else:
        masked = "****"
    now = _time.time()
    expires_in_s = max(0, t.get("expires_at", 0) - now)
    return {
        "token_id":           masked,
        "applies_remaining":  t.get("applies_remaining", 0),
        "expires_in_minutes": round(expires_in_s / 60.0, 1),
        "issued_at":          _dt.datetime.fromtimestamp(
                                  t.get("issued_at", 0)).isoformat(
                                  timespec="seconds")
                              if t.get("issued_at") else "",
    }


def _prune_expired_tokens() -> None:
    """Drop tokens past their TTL or with zero remaining applies."""
    now = _time.time()
    _ACTIVE_TOKENS[:] = [
        t for t in _ACTIVE_TOKENS
        if t.get("expires_at", 0) > now and t.get("applies_remaining", 0) > 0
    ]


def issue_token(applies: int = DEFAULT_TOKEN_APPLIES,
                minutes: int = DEFAULT_TOKEN_MINUTES) -> dict:
    """Mint a confirm-token good for N applies in M minutes.

    Returns:
        {"ok": True, "token": "<hex>", "applies": N, "expires_at": <epoch>,
         "expires_iso": "<iso>"}

    Token is a 16-byte hex string (32 chars). It's the caller's job to
    transmit it back to Chloe (typically a Cowork job stashes it in a
    proposal frontmatter or passes it as an MCP arg).
    """
    if applies < 1 or applies > MAX_TOKEN_APPLIES:
        return {"ok": False,
                "error": f"applies must be 1-{MAX_TOKEN_APPLIES}, got {applies}"}
    if minutes < 1 or minutes > MAX_TOKEN_MINUTES:
        return {"ok": False,
                "error": f"minutes must be 1-{MAX_TOKEN_MINUTES}, got {minutes}"}
    now = _time.time()
    token = _secrets.token_hex(16)  # 32 chars
    entry = {
        "token": token,
        "applies_remaining": int(applies),
        "expires_at": now + (int(minutes) * 60),
        "issued_at": now,
    }
    with _TOKEN_LOCK:
        _prune_expired_tokens()
        _ACTIVE_TOKENS.append(entry)
    return {
        "ok": True,
        "token": token,
        "applies": int(applies),
        "expires_at": entry["expires_at"],
        "expires_iso": _dt.datetime.fromtimestamp(
            entry["expires_at"]).isoformat(timespec="seconds"),
    }


def _consume_token(token: str) -> tuple[bool, str]:
    """Validate + decrement a token. Returns (ok, error_message).

    Constant-time string comparison so a token can't be guessed via
    timing attack (unlikely but cheap).
    """
    if not token or not isinstance(token, str):
        return False, "missing token"
    token = token.strip()
    with _TOKEN_LOCK:
        _prune_expired_tokens()
        for t in _ACTIVE_TOKENS:
            stored = t.get("token", "")
            # secrets.compare_digest is constant-time.
            if len(stored) == len(token) and _secrets.compare_digest(stored, token):
                if t.get("expires_at", 0) <= _time.time():
                    return False, "token expired"
                if t.get("applies_remaining", 0) <= 0:
                    return False, "token already consumed"
                t["applies_remaining"] -= 1
                # Don't prune yet — keep the entry visible in status until
                # next call; _prune_expired_tokens will sweep it next time.
                return True, ""
        return False, "token not recognized"


def revoke_tokens() -> dict:
    """Drop every active token. Use if Ed suspects a token leaked or
    just wants a clean slate."""
    with _TOKEN_LOCK:
        n = len(_ACTIVE_TOKENS)
        _ACTIVE_TOKENS.clear()
    return {"ok": True, "revoked": n}


def list_tokens() -> list[dict]:
    """Return active-token status (masked). Read-only diagnostic."""
    with _TOKEN_LOCK:
        _prune_expired_tokens()
        return [_token_view(t) for t in _ACTIVE_TOKENS]


def apply_proposal_with_token(slug: str, token: str,
                              dry_run: bool = False) -> dict:
    """Tier-2 entry point: apply a proposal using a confirm-token.

    Same pipeline as apply_proposal — path whitelist, ast.parse,
    timestamped backup, session counter — with an ADDITIONAL token check
    upfront. dry_run does NOT consume the token (so callers can preview
    cheaply).
    """
    if dry_run:
        return apply_proposal(slug, dry_run=True)

    ok, err = _consume_token(token)
    if not ok:
        return {"ok": False, "slug": slug,
                "error": f"token rejected: {err}"}

    # Token consumed — proceed with the normal apply. If the apply fails
    # for any reason (path whitelist, ast.parse, etc.), the token slot is
    # already burnt. That's deliberate: a failed apply still counts so a
    # rogue caller can't probe paths cheaply.
    return apply_proposal(slug, dry_run=False)


# ─── CLI entrypoint ────────────────────────────────────────────────────────

def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Chloe code-proposal pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list proposals")
    sp.add_argument("--status", default=None,
                    choices=["pending", "applied", "reverted"])

    sp = sub.add_parser("show", help="dump a parsed proposal")
    sp.add_argument("slug")

    sp = sub.add_parser("apply", help="apply a proposal")
    sp.add_argument("slug")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("revert", help="revert a proposal")
    sp.add_argument("slug")

    args = ap.parse_args(argv)
    if args.cmd == "list":
        rows = list_proposals(status=args.status)
        if not rows:
            print("(no proposals)")
            return 0
        for r in rows:
            print(f"{r['date']}  {r['status']:8s}  {r['slug']}  "
                  f"→ {r['target']}  ({r['kind']})  {r['title']}")
        return 0
    if args.cmd == "show":
        p = load_proposal(args.slug)
        print(f"path: {p.path}")
        print(f"target: {p.target}")
        print(f"kind: {p.kind}")
        print(f"status: {p.status}")
        print(f"sections: {sorted(p.sections.keys())}")
        return 0
    if args.cmd == "apply":
        r = apply_proposal(args.slug, dry_run=args.dry_run)
        print(r.get("message") or r.get("error"))
        return 0 if r.get("ok") else 1
    if args.cmd == "revert":
        r = revert_proposal(args.slug)
        print(r.get("message") or r.get("error"))
        return 0 if r.get("ok") else 1
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
