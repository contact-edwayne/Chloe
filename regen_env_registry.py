"""regen_env_registry.py - Regenerates chloe_env_registry.py from source.

Scans every *.py file in this directory for os.environ.get()/os.getenv()
calls and emits a fresh ENV_REGISTRY module to stdout. Run it from the
jarvis directory whenever a flag is added, removed, or its default
changes, and redirect the output over the existing file:

    python regen_env_registry.py > chloe_env_registry.py

This is a plain source scan (regex + ast.literal_eval on the default
argument), not an import of any module, so it's safe to run any time --
it never executes jarvis.py or anything else, just reads text. See
chloe_env_registry.py's own docstring for why this exists (action item
#10, "Central env-var flag registry").
"""
import re
import glob
import ast

NAME_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z][A-Z0-9_]*)["\']\s*(?:,\s*(.*?))?\)',
    re.DOTALL,
)
KNOWN_FALSE_POSITIVES = {"KEY", "X"}

results = {}
for path in sorted(glob.glob("*.py")):
    if path in ("chloe_env_registry.py", "regen_env_registry.py"):
        continue
    try:
        src = open(path, encoding="utf-8").read()
    except Exception:
        continue
    for m in NAME_RE.finditer(src):
        name = m.group(1)
        if name in KNOWN_FALSE_POSITIVES:
            continue
        start = m.start()
        line_start = src.rfind("\n", 0, start) + 1
        line_end = src.find("\n", start)
        line = src[line_start:line_end if line_end != -1 else len(src)]
        col = start - line_start
        if "#" in line[:col]:
            continue
        default_raw = m.group(2)
        literal = None
        has_literal = False
        computed_repr = None
        if default_raw:
            default_raw = default_raw.strip()
            first_line = default_raw.split("\n")[0].rstrip(", ")
            try:
                literal = ast.literal_eval(first_line)
                has_literal = True
            except Exception:
                computed_repr = first_line[:70]
        results.setdefault(name, {"files": set(), "literals": set(), "computed": set(), "no_default": False})
        results[name]["files"].add(path)
        if default_raw is None:
            results[name]["no_default"] = True
        elif has_literal:
            results[name]["literals"].add(literal)
        elif computed_repr:
            results[name]["computed"].add(computed_repr)

# --- Emit chloe_env_registry.py ---------------------------------------

PREFIX_GROUPS = [
    ("CHLOE_", "Chloe-specific flags"),
    ("OLLAMA_", "Ollama connection/model flags"),
    ("KOKORO_", "Kokoro local TTS flags"),
    ("ELEVENLABS_", "ElevenLabs TTS flags"),
    ("USE_", "TTS backend toggles"),
    ("BREEZ_", "Breez Liquid wallet SDK flags"),
    ("PORCUPINE_", "Porcupine wake-word flags"),
    ("GROQ_", "Groq API flags (vision fallback only -- chat/voice/STT are local)"),
    ("BRAVE_", "Brave Search API flags"),
    ("FINNHUB_", "Finnhub market-data API flags"),
    ("JARVIS_", "Legacy jarvis.py-prefixed flags"),
    ("BENCH_", "bench_ollama.py benchmarking flags"),
    ("COWORK_", "Cowork/Claude scheduling integration flags"),
]

def group_for(name):
    for prefix, label in PREFIX_GROUPS:
        if name.startswith(prefix):
            return prefix
    return ""

lines = []
lines.append('"""chloe_env_registry.py - Single source of truth for every environment')
lines.append('variable Chloe\'s codebase reads (action item #10, audit Part 13:')
lines.append('"Central env-var flag registry").')
lines.append("")
lines.append("2026-09-06: 146 os.environ.get()/os.getenv() call sites were found")
lines.append("scattered across ~35 files, each with its own inline default and no")
lines.append("single place documenting what exists. Rewriting all 146 call sites to")
lines.append("route through one accessor would be the deeper fix the audit's own")
lines.append("wording implies, but that is a large, multi-file mechanical change")
lines.append("this bridge session cannot live-test (many of these files -- wiki_dedup,")
lines.append("chloe_ed_profile, social_composer, weather, notify, and more -- were")
lines.append("never touched or verified this session). Retrofitting all of them in")
lines.append("one pass risks a silent regression nobody would catch until Ed hit it")
lines.append("live weeks later.")
lines.append("")
lines.append("What this module does instead, safely and additively: documents every")
lines.append("flag this codebase currently reads (name, its default as captured")
lines.append("directly from source, and which file(s) read it) as ENV_REGISTRY below")
lines.append("-- generated once from the actual source via a small extraction script,")
lines.append("not hand-typed, so it's traceable rather than guessed -- and provides")
lines.append("unknown_chloe_env_vars(), a boot-time check that catches the exact bug")
lines.append("class this whole session kept finding: a flag that's SET but silently")
lines.append('does nothing because of a typo (e.g. "CHLOE_BARGE_TRESHOLD" instead of')
lines.append('"CHLOE_BARGE_THRESHOLD") or because it was renamed/removed and an old')
lines.append(".env line never got cleaned up. jarvis.py's boot sequence calls this")
lines.append("once at startup and prints a warning for anything it finds -- see the")
lines.append('"unknown env var(s)" print near the other boot-time diagnostics.')
lines.append("")
lines.append("Existing call sites are UNCHANGED -- this module does not replace any")
lines.append("of the 146 os.environ.get() calls, only documents and cross-checks")
lines.append("them. Regenerate the data below by running")
lines.append("`python regen_env_registry.py > chloe_env_registry.py` (checked into")
lines.append("this repo alongside this file) whenever a flag is added, removed, or")
lines.append('its default changes -- "no single place to see what exists" was the')
lines.append("audit's actual complaint, and a registry that quietly drifts out of")
lines.append("date recreates the same problem with extra steps.")
lines.append('"""')
lines.append("import os")
lines.append("")
lines.append("# name -> {\"default\": <python literal>, \"default_expr\": <source text,")
lines.append("#          when the default isn't a plain literal>, \"files\": (tuple)}")
lines.append("# Generated 2026-09-06 from every os.environ.get()/os.getenv() call")
lines.append("# site in this repo's own *.py files.")
lines.append("ENV_REGISTRY = {")

def literal_str(name, info):
    lits = info["literals"]
    if info["no_default"] and not lits and not info["computed"]:
        return None  # required, no default anywhere
    if len(lits) == 1:
        return repr(next(iter(lits)))
    if len(lits) > 1:
        return repr(sorted(lits, key=str))  # differing defaults across call sites
    return None

def expr_str(info):
    comp = info["computed"]
    if not comp:
        return None
    return repr(sorted(comp)[0])

current_group = None
for name in sorted(results):
    info = results[name]
    grp = group_for(name)
    if grp != current_group:
        label = dict(PREFIX_GROUPS).get(grp, "Other / third-party env vars")
        lines.append(f"    # --- {label} ---")
        current_group = grp
    files = tuple(sorted(info["files"]))
    default_lit = literal_str(name, info)
    default_expr = expr_str(info)
    entry_parts = [f'"files": {files!r}']
    if default_lit is not None:
        entry_parts.insert(0, f'"default": {default_lit}')
    if default_expr is not None:
        entry_parts.insert(1 if default_lit is not None else 0,
                           f'"default_expr": {default_expr}')
    if default_lit is None and default_expr is None:
        entry_parts.insert(0, '"default": None')
    lines.append(f'    "{name}": {{{", ".join(entry_parts)}}},')

lines.append("}")
lines.append("")
lines.append("# Prefixes this codebase actually uses -- unknown_chloe_env_vars()")
lines.append("# only flags a set-but-unrecognized var if it starts with one of")
lines.append("# these, so it never warns about unrelated system/OS env vars (PATH,")
lines.append("# HOME, etc.) that just happen to be present in the process.")
lines.append(f"_KNOWN_PREFIXES = {tuple(p for p, _ in PREFIX_GROUPS)!r}")
lines.append("")
lines.append("")
lines.append("def all_known_names() -> frozenset:")
lines.append('    """Every env var name this codebase is known to read."""')
lines.append("    return frozenset(ENV_REGISTRY)")
lines.append("")
lines.append("")
lines.append("def describe(name: str) -> str:")
lines.append('    """One-line summary of a registered flag: its default (or the')
lines.append("    source expression it's derived from, when it isn't a plain")
lines.append('    literal) and which file(s) read it. Returns an honest "not')
lines.append('    registered" message for an unknown name rather than guessing."""')
lines.append("    info = ENV_REGISTRY.get(name)")
lines.append("    if info is None:")
lines.append('        return f"{name!r} is not in ENV_REGISTRY -- unrecognized or removed."')
lines.append('    files = ", ".join(info["files"])')
lines.append('    if "default" in info:')
lines.append('        default_part = f"default={info[\'default\']!r}"')
lines.append('    elif "default_expr" in info:')
lines.append('        default_part = f"default is computed: {info[\'default_expr\']}"')
lines.append("    else:")
lines.append('        default_part = "no default -- required if read"')
lines.append('    return f"{name}: {default_part} (read in {files})"')
lines.append("")
lines.append("")
lines.append("def unknown_chloe_env_vars() -> list[str]:")
lines.append('    """Scan the live process environment for any SET variable that')
lines.append("    starts with one of this codebase's own prefixes (_KNOWN_PREFIXES)")
lines.append("    but isn't in ENV_REGISTRY -- the exact bug class this session kept")
lines.append('    finding by hand: a typo (\'CHLOE_BARGE_TRESHOLD\'), a renamed flag')
lines.append("    whose old .env line never got cleaned up, or a flag added to code")
lines.append("    without ever being added here. Returns an empty list when")
lines.append('    everything set matches something registered. Never raises.')
lines.append('    """')
lines.append("    try:")
lines.append("        known = all_known_names()")
lines.append("        hits = []")
lines.append("        for name in os.environ:")
lines.append("            if name in known:")
lines.append("                continue")
lines.append("            if any(name.startswith(p) for p in _KNOWN_PREFIXES):")
lines.append("                hits.append(name)")
lines.append("        return sorted(hits)")
lines.append("    except Exception:")
lines.append("        return []")
lines.append("")
print("\n".join(lines))
