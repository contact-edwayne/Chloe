"""test_env_registry.py - Regression test for action item #10 (audit
Part 13, "Central env-var flag registry").

chloe_env_registry.py documents every env var this codebase's ~35
files read (146 flags, generated from source by regen_env_registry.py,
not hand-typed) and exposes unknown_chloe_env_vars() -- a boot-time
check that flags a SET-but-unrecognized CHLOE_*/OLLAMA_*/etc. env var,
catching the exact silent-typo/stale-flag bug class this session kept
finding by hand (e.g. chloe_ed_profile.py's missing keep_alive field,
Round 10). jarvis.py's boot sequence calls this once at startup.

This is a scoped, additive fix, NOT a rewrite of the 146 existing
os.environ.get() call sites -- see chloe_env_registry.py's own
docstring for why a full migration was judged too risky to attempt
without live-testing every one of those ~35 files. This test covers
what WAS built: the registry's data integrity and the unknown-var
detector's actual behavior.

chloe_env_registry.py is safe to import directly (pure data + os.environ
reads, no I/O, no side effects at import time).

Run from the jarvis dir:
    python test_env_registry.py
Exit code 0 on success, non-zero on any failure.
"""
import os

import chloe_env_registry as reg

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def test_registry_is_populated():
    check("ENV_REGISTRY has a substantial number of entries (this "
          "codebase is known to read 140+ distinct env vars as of "
          "2026-09-06)", len(reg.ENV_REGISTRY) >= 100, len(reg.ENV_REGISTRY))


def test_every_entry_has_files_and_some_default_info():
    bad = []
    for name, info in reg.ENV_REGISTRY.items():
        if "files" not in info or not info["files"]:
            bad.append((name, "missing files"))
            continue
        if not any(k in info for k in ("default", "default_expr")):
            bad.append((name, "no default or default_expr key at all"))
    check("every registered flag has a non-empty files tuple and at "
          "least a default/default_expr key (even if the value is "
          "None, meaning 'required, no fallback')", not bad, bad[:5])


def test_known_flags_from_this_session_are_registered():
    # Spot-check a handful of flags this exact session touched or
    # relied on -- if any of these vanish from the registry, the
    # extraction regressed.
    for name in ("CHLOE_BARGE_THRESHOLD", "OLLAMA_KEEP_ALIVE",
                  "CHLOE_CONTACT_ALIAS_TTL_HOURS", "OLLAMA_URL",
                  "CHLOE_WALLET_NETWORK"):
        check(f"{name} is registered", name in reg.ENV_REGISTRY, name)


def test_describe_known_flag():
    text = reg.describe("CHLOE_BARGE_THRESHOLD")
    check("describe() on a known flag mentions its default and the "
          "file that reads it",
          "0.7" in text and "jarvis.py" in text, text)


def test_describe_unknown_flag_is_honest():
    text = reg.describe("CHLOE_TOTALLY_MADE_UP_FLAG_XYZ")
    check("describe() on an unregistered name says so plainly instead "
          "of fabricating a description",
          "not in ENV_REGISTRY" in text, text)


def test_all_known_names_matches_registry_keys():
    check("all_known_names() is exactly the registry's key set",
          reg.all_known_names() == frozenset(reg.ENV_REGISTRY))


def test_unknown_detector_catches_a_typo():
    marker = "CHLOE_TEST_TYPO_MARKER_DOES_NOT_EXIST"
    assert marker not in reg.ENV_REGISTRY, "test marker collided with a real flag"
    os.environ[marker] = "1"
    try:
        hits = reg.unknown_chloe_env_vars()
        check("a CHLOE_-prefixed env var that's set but not registered "
              "is caught by unknown_chloe_env_vars() -- this is the "
              "exact bug class (a typo'd flag name) this item exists "
              "to catch", marker in hits, hits)
    finally:
        del os.environ[marker]


def test_unknown_detector_ignores_unrelated_system_vars():
    # PATH/HOME etc. are always set in any real process and must never
    # trigger a false "unknown Chloe flag" warning.
    hits = reg.unknown_chloe_env_vars()
    check("ordinary system env vars (PATH) are never flagged -- only "
          "names starting with a prefix this codebase actually uses",
          "PATH" not in hits, hits)


def test_unknown_detector_silent_when_everything_registered():
    # Set a value for a REAL, already-registered flag -- should never
    # show up as "unknown".
    known_name = next(iter(reg.ENV_REGISTRY))
    os.environ[known_name] = os.environ.get(known_name, "") or "x"
    hits = reg.unknown_chloe_env_vars()
    check(f"setting an already-registered flag ({known_name}) never "
          f"triggers a false 'unknown' warning", known_name not in hits, hits)


def test_no_alias_collisions_between_prefix_groups():
    # Sanity check on the generator's own grouping logic: every
    # registered name should be classified into exactly the group its
    # own prefix implies (or "other"), never ambiguous.
    from collections import Counter
    prefix_hits = Counter()
    for name in reg.ENV_REGISTRY:
        matches = [p for p in reg._KNOWN_PREFIXES if name.startswith(p)]
        prefix_hits[len(matches)] += 1
    check("no registered flag name matches more than one known prefix "
          "(e.g. nothing starts with both 'USE_' and 'USER_' in a way "
          "that would double-classify it)",
          all(k <= 1 for k in prefix_hits), dict(prefix_hits))


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
