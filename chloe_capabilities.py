"""Self-analysis surface for Chloe — what slashes, MCP tools, scheduled
jobs, and env knobs exist right now.

Ed asked: "tell me specifics on what she can do." Without this module,
Chloe (and any caller) has to grep the source by hand. With it, the live
surface is one function call away — and ast-based so it stays accurate
when handlers move around.

Public API:
  - summary() -> dict                 — all surface area in one shot
  - list_slash_commands() -> list[dict]
  - list_mcp_tools() -> list[dict]
  - list_jobs() -> list[dict]         — lazy-imports chloe_jobs.state()
  - list_env_knobs() -> list[dict]
  - list_modules() -> list[dict]
  - describe_module(name) -> dict     — ast introspection of one module

All scans live under the directory containing this file (jarvis/). No
external deps — stdlib only.
"""

from __future__ import annotations

import ast
import datetime as _dt
import os
import re
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent


# ─── Discovery: slash commands ────────────────────────────────────────────

# Inside try_handle_brain_command (brain_wiring.py) every slash dispatch
# is either `msg.startswith("/foo ")`, `msg == "/foo"`, or `msg in (...)`.
# A regex over the source captures all three shapes without us re-running
# the function. Anchors on the leading `/`. The slash literal may end on
# a trailing space (`"/ingest "`) — match up to space-or-quote either way.
_SLASH_PATTERN = re.compile(
    r"""msg(?:\s*\.\s*startswith|\s*==|\s+in)\s*"""
    r"""(?:\(\s*)?["'](/[a-zA-Z_][a-zA-Z_0-9]*)(?:\s+|["'])""",
    re.MULTILINE,
)

# Map slash -> handler function name. Captured by walking try_handle_brain_command
# for "return handle_X(..." patterns inside each slash-conditional.
_HANDLER_CALL = re.compile(r"return\s+(handle_[a-zA-Z_0-9]+)\s*\(")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _docstring_first_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _function_first_doc_line(src: str, func_name: str) -> str:
    """Quick path: ast-parse a file and find a top-level function's first
    docstring line. Returns "" if not found."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                return _docstring_first_line(node)
    return ""


def list_slash_commands() -> list[dict]:
    """Return all slash commands wired into brain_wiring.try_handle_brain_command.

    Each item: {name, handler, summary, source_file, source_line}.
    """
    bw_path = HERE / "brain_wiring.py"
    src = _read(bw_path)
    if not src:
        return []
    lines = src.splitlines()

    # Find try_handle_brain_command body bounds.
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    body_start = body_end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "try_handle_brain_command"):
            body_start = node.body[0].lineno - 1 if node.body else node.lineno
            body_end = node.end_lineno or len(lines)
            break
    if body_start is None:
        return []
    body_text = "\n".join(lines[body_start - 1:body_end])

    out: list[dict] = []
    seen: set[str] = set()
    # Walk lines so we can resolve "which handler is called in this block".
    for m in _SLASH_PATTERN.finditer(body_text):
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        # Find the line in body_text + 30 lines ahead for the handler call.
        idx = m.start()
        window = body_text[idx:idx + 800]
        hm = _HANDLER_CALL.search(window)
        handler = hm.group(1) if hm else ""
        # Find the docstring for the handler.
        summary = ""
        if handler:
            summary = _function_first_doc_line(src, handler)
        # Approximate source line (relative to file).
        line_no = body_start + body_text.count("\n", 0, idx)
        out.append({
            "name": name,
            "handler": handler,
            "summary": summary,
            "source_file": "brain_wiring.py",
            "source_line": line_no,
        })
    out.sort(key=lambda d: d["name"])
    return out


# ─── Discovery: MCP tools ─────────────────────────────────────────────────

def list_mcp_tools() -> list[dict]:
    """Return every @mcp.tool() decorated function in chloe_mcp_server.py.

    Each item: {name, signature, summary, source_line}.
    """
    mcp_path = HERE / "chloe_mcp_server.py"
    src = _read(mcp_path)
    if not src:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    out: list[dict] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorated = False
        for dec in node.decorator_list:
            # Match `@mcp.tool()` or `@mcp.tool`.
            if isinstance(dec, ast.Call):
                if (isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "tool"
                        and isinstance(dec.func.value, ast.Name)
                        and dec.func.value.id == "mcp"):
                    decorated = True
            elif (isinstance(dec, ast.Attribute)
                  and dec.attr == "tool"
                  and isinstance(dec.value, ast.Name)
                  and dec.value.id == "mcp"):
                decorated = True
        if not decorated:
            continue
        out.append({
            "name": node.name,
            "signature": _format_signature(node),
            "summary": _docstring_first_line(node),
            "source_line": node.lineno,
        })
    out.sort(key=lambda d: d["name"])
    return out


def _format_signature(node) -> str:
    """Render `def foo(a: int = 1) -> str` as `foo(a: int = 1) -> str`."""
    args = node.args
    parts: list[str] = []

    pos_only = list(args.posonlyargs)
    pos = list(args.args)
    kw_only = list(args.kwonlyargs)

    pos_defaults = list(args.defaults)
    n_no_default = (len(pos_only) + len(pos)) - len(pos_defaults)

    for i, a in enumerate(pos_only + pos):
        s = a.arg
        if a.annotation is not None:
            s += f": {ast.unparse(a.annotation)}"
        # Default lookup
        if i >= n_no_default:
            d = pos_defaults[i - n_no_default]
            s += f" = {ast.unparse(d)}"
        parts.append(s)
        if i == len(pos_only) - 1 and pos_only:
            parts.append("/")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif kw_only:
        parts.append("*")
    for i, a in enumerate(kw_only):
        s = a.arg
        if a.annotation is not None:
            s += f": {ast.unparse(a.annotation)}"
        d = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        if d is not None:
            s += f" = {ast.unparse(d)}"
        parts.append(s)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    sig = f"{node.name}({', '.join(parts)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


# ─── Discovery: scheduled jobs ────────────────────────────────────────────

def list_jobs() -> list[dict]:
    """Return scheduled-job runtime state via chloe_jobs.state().

    Falls back to a static read of JOBS + SCHEDULES if state() can't run
    (e.g. logs/ missing in a sandbox).
    """
    try:
        import chloe_jobs
    except Exception as e:
        return [{"error": f"chloe_jobs import failed: "
                          f"{type(e).__name__}: {e}"}]
    try:
        st = chloe_jobs.state()
        return list(st.get("jobs", []))
    except Exception:
        # Fallback: static registry only.
        try:
            return [
                {"name": name, "schedule": chloe_jobs.SCHEDULES.get(name, "")}
                for name in chloe_jobs.JOBS.keys()
            ]
        except Exception as e:
            return [{"error": str(e)}]


# ─── Discovery: env knobs ─────────────────────────────────────────────────

# Scan all .py files in jarvis/ for os.environ.get("KEY", "default") patterns.
# Captures CHLOE_* / OLLAMA_* / GROQ_* / etc. + their default values.
_ENV_GET_RE = re.compile(
    r"""os\.environ\.get\(\s*
        ["'](?P<key>[A-Z][A-Z0-9_]*)["']
        (?:\s*,\s*(?P<default>[^)]+))?
        \s*\)""",
    re.VERBOSE,
)


def list_env_knobs() -> list[dict]:
    """Return env-var defaults referenced anywhere under jarvis/.

    Each item: {name, default, source_files: [...] }. Deduped by name.
    Defaults are kept as the raw literal expression (so "1" vs "True" vs
    `r"C:\\Chloe\\brain"` all stay verbatim for the reader).
    """
    knobs: dict[str, dict] = {}
    for py in HERE.glob("*.py"):
        if py.name in ("venv_py314",):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _ENV_GET_RE.finditer(text):
            key = m.group("key")
            default = (m.group("default") or "").strip()
            entry = knobs.setdefault(key, {
                "name": key,
                "default": default,
                "source_files": [],
            })
            if py.name not in entry["source_files"]:
                entry["source_files"].append(py.name)
            # Prefer non-empty default if we see one later.
            if default and not entry["default"]:
                entry["default"] = default
    out = sorted(knobs.values(), key=lambda d: d["name"])
    return out


# ─── Discovery: modules ───────────────────────────────────────────────────

# Files to skip in the module listing (vendored / generated / build cruft).
_SKIP_NAMES = {"setup.py", "conftest.py"}
_SKIP_PREFIXES = ("apply_", "splice_", "verify_", "bench_", "backfill_",
                  "register_", "install_", "test_", "lat_")


def list_modules() -> list[dict]:
    """Return Chloe-internal modules under jarvis/ with cheap stats.

    Each item: {name, path, line_count, function_count, has_docstring,
    summary}. Helper/scratchpad files (`apply_*.py`, `splice_*.py`, etc.)
    are filtered out — they're transient scaffolding, not part of the
    durable surface.
    """
    out: list[dict] = []
    for py in HERE.glob("*.py"):
        name = py.name
        if name in _SKIP_NAMES:
            continue
        if any(name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        text = _read(py)
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        fns = [n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        doc = ast.get_docstring(tree) or ""
        summary = ""
        for line in doc.splitlines():
            line = line.strip()
            if line:
                summary = line
                break
        out.append({
            "name": py.stem,
            "path": name,
            "line_count": text.count("\n") + 1,
            "function_count": len(fns),
            "has_docstring": bool(doc),
            "summary": summary,
        })
    out.sort(key=lambda d: d["name"])
    return out


# ─── describe_module: ast introspection of one file ───────────────────────

def describe_module(name: str) -> dict:
    """Return ast-derived surface of `name` (`.py` optional). Includes the
    module docstring, top-level function signatures + first-line
    docstrings, classes + method lists, and CONSTANT_NAME assignments.
    """
    stem = name[:-3] if name.endswith(".py") else name
    candidates = [HERE / f"{stem}.py", HERE / name]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"no module {name!r} under {HERE} "
            f"(tried {[str(p) for p in candidates]})"
        )
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return {
            "name": stem, "path": path.name, "error": f"SyntaxError: {e}",
        }

    functions: list[dict] = []
    classes: list[dict] = []
    constants: list[dict] = []
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "signature": _format_signature(node),
                "doc": _docstring_first_line(node),
                "line": node.lineno,
                "is_private": node.name.startswith("_"),
            })
        elif isinstance(node, ast.ClassDef):
            methods: list[dict] = []
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": m.name,
                        "signature": _format_signature(m),
                        "doc": _docstring_first_line(m),
                    })
            classes.append({
                "name": node.name,
                "doc": _docstring_first_line(node),
                "line": node.lineno,
                "methods": methods,
            })
        elif isinstance(node, ast.Assign):
            # Single-target UPPER_CASE constants.
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.isupper()):
                name_ = node.targets[0].id
                try:
                    value_preview = ast.unparse(node.value)
                except Exception:
                    value_preview = "<unparseable>"
                if len(value_preview) > 120:
                    value_preview = value_preview[:117] + "..."
                constants.append({
                    "name": name_, "value": value_preview, "line": node.lineno,
                })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                imports.append(ast.unparse(node))
            except Exception:
                continue

    return {
        "name": stem,
        "path": path.name,
        "line_count": src.count("\n") + 1,
        "docstring": (ast.get_docstring(tree) or "").strip(),
        "imports": imports,
        "constants": constants,
        "functions": functions,
        "classes": classes,
    }


# ─── Top-level summary ────────────────────────────────────────────────────

def summary() -> dict:
    """All surface area in one structured payload. Cheap — no LLM calls,
    no IO outside the jarvis/ directory."""
    return {
        "slashes": list_slash_commands(),
        "mcp_tools": list_mcp_tools(),
        "jobs": list_jobs(),
        "env_knobs": list_env_knobs(),
        "modules": list_modules(),
        "computed_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


# ─── Pretty-print formatters for chat surface ─────────────────────────────

def format_summary_markdown(s: Optional[dict] = None) -> str:
    """Render summary() as a chat-friendly markdown report."""
    if s is None:
        s = summary()
    out: list[str] = []
    out.append(f"# Chloe capabilities snapshot")
    out.append(f"_Computed {s.get('computed_at', '?')}._\n")

    slashes = s.get("slashes", [])
    out.append(f"## Slash commands ({len(slashes)})")
    for sl in slashes:
        line = f"- `{sl['name']}`"
        if sl.get("summary"):
            line += f" — {sl['summary']}"
        out.append(line)
    out.append("")

    tools = s.get("mcp_tools", [])
    out.append(f"## MCP tools ({len(tools)})")
    for t in tools:
        line = f"- `mcp__chloe__{t['name']}`"
        if t.get("summary"):
            line += f" — {t['summary']}"
        out.append(line)
    out.append("")

    jobs = s.get("jobs", [])
    out.append(f"## Scheduled jobs ({len(jobs)})")
    for j in jobs:
        if j.get("error"):
            out.append(f"- _error: {j['error']}_")
            continue
        sched = j.get("schedule") or "?"
        health = j.get("health") or "?"
        age = j.get("age_hours")
        age_str = f"{age:.1f}h ago" if isinstance(age, (int, float)) else "never"
        out.append(f"- `{j['name']}` — {sched} · {health} · last run {age_str}")
    out.append("")

    knobs = s.get("env_knobs", [])
    out.append(f"## Env knobs ({len(knobs)})")
    chloe_knobs = [k for k in knobs if k["name"].startswith("CHLOE_")]
    other_knobs = [k for k in knobs if not k["name"].startswith("CHLOE_")]
    for k in chloe_knobs + other_knobs:
        default = k.get("default", "")
        # Trim long defaults
        if len(default) > 60:
            default = default[:57] + "..."
        out.append(f"- `{k['name']}` = {default or '<no default>'}")
    out.append("")

    modules = s.get("modules", [])
    out.append(f"## Modules ({len(modules)})")
    for m in modules:
        out.append(f"- `{m['path']}` — {m['line_count']} lines, "
                   f"{m['function_count']} top-level fn(s)"
                   + (f" — {m['summary']}" if m.get("summary") else ""))

    return "\n".join(out)


def format_module_markdown(d: dict) -> str:
    """Render describe_module() as a chat-friendly markdown report."""
    if d.get("error"):
        return f"**{d['name']}**: {d['error']}"
    out: list[str] = []
    out.append(f"# `{d['path']}` ({d['line_count']} lines)")
    if d.get("docstring"):
        first = d["docstring"].splitlines()[0]
        out.append(f"\n_{first}_\n")
    if d.get("imports"):
        sample = d["imports"][:6]
        more = len(d["imports"]) - len(sample)
        out.append(f"## Imports ({len(d['imports'])})")
        for imp in sample:
            out.append(f"- `{imp}`")
        if more > 0:
            out.append(f"- _… {more} more_")
        out.append("")
    if d.get("constants"):
        out.append(f"## Constants ({len(d['constants'])})")
        for c in d["constants"][:20]:
            out.append(f"- `{c['name']}` = `{c['value']}`")
        if len(d["constants"]) > 20:
            out.append(f"- _… {len(d['constants']) - 20} more_")
        out.append("")
    if d.get("functions"):
        public = [f for f in d["functions"] if not f["is_private"]]
        private = [f for f in d["functions"] if f["is_private"]]
        out.append(f"## Functions ({len(d['functions'])} — "
                   f"{len(public)} public, {len(private)} private)")
        for f in public:
            line = f"- `{f['signature']}`"
            if f["doc"]:
                line += f" — {f['doc']}"
            out.append(line)
        if private:
            out.append(f"\n_Private helpers:_ "
                       + ", ".join(f"`{f['name']}`" for f in private[:20])
                       + (f" … +{len(private) - 20}" if len(private) > 20 else ""))
        out.append("")
    if d.get("classes"):
        out.append(f"## Classes ({len(d['classes'])})")
        for c in d["classes"]:
            line = f"- `{c['name']}`"
            if c.get("doc"):
                line += f" — {c['doc']}"
            out.append(line)
            for m in c["methods"][:6]:
                out.append(f"  - `{m['signature']}`")
            if len(c["methods"]) > 6:
                out.append(f"  - _… {len(c['methods']) - 6} more methods_")
    return "\n".join(out)


# ─── CLI ──────────────────────────────────────────────────────────────────

def _cli(argv: list[str]) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Chloe self-analysis surface")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("summary", help="all surface area")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("describe", help="describe one module")
    sp.add_argument("module")
    sp.add_argument("--json", action="store_true")

    sub.add_parser("slashes", help="list slash commands")
    sub.add_parser("tools", help="list MCP tools")
    sub.add_parser("jobs", help="list scheduled jobs")
    sub.add_parser("env", help="list env knobs")
    sub.add_parser("modules", help="list modules")

    args = ap.parse_args(argv)
    if args.cmd == "summary":
        s = summary()
        if args.json:
            print(json.dumps(s, indent=2, default=str))
        else:
            print(format_summary_markdown(s))
        return 0
    if args.cmd == "describe":
        try:
            d = describe_module(args.module)
        except FileNotFoundError as e:
            print(e); return 1
        if args.json:
            print(json.dumps(d, indent=2, default=str))
        else:
            print(format_module_markdown(d))
        return 0
    if args.cmd == "slashes":
        for s in list_slash_commands():
            print(f"{s['name']:24s}  {s.get('handler','-'):28s}  {s['summary']}")
        return 0
    if args.cmd == "tools":
        for t in list_mcp_tools():
            print(f"{t['name']:24s}  {t['signature']}")
        return 0
    if args.cmd == "jobs":
        for j in list_jobs():
            if "error" in j:
                print(j["error"]); continue
            print(f"{j['name']:36s}  {j.get('schedule','?'):20s}  "
                  f"{j.get('health','?')}")
        return 0
    if args.cmd == "env":
        for k in list_env_knobs():
            print(f"{k['name']:36s}  {k['default']}")
        return 0
    if args.cmd == "modules":
        for m in list_modules():
            print(f"{m['path']:32s}  {m['line_count']:>5d} lines  "
                  f"{m['function_count']:>3d} fns")
        return 0
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
