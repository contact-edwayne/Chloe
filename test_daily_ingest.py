"""test_daily_ingest.py - Sanity tests for daily_ingest.

Uses a fake Brain injected via sys.modules so the test runs without
Groq, Ollama, or touching C:\\Chloe\\brain. Verifies:
  1. _strip_frontmatter handles present/absent/unterminated frontmatter
  2. _slug_for produces snake-case+date slugs that pass BRAIN's validator
  3. Missing daily note -> ok=True with skipped reason, no raw file
  4. Empty/whitespace daily note -> ok=True with skipped reason
  5. Normal note -> ok=True, raw file written with provenance header,
     BRAIN.ingest called with the correct filename, _log called
  6. Note with YAML frontmatter -> frontmatter NOT echoed into wrapped body
  7. Default date is yesterday (no --date flag)
  8. --dry-run -> no raw write, no BRAIN.ingest call
  9. BRAIN.ingest raising -> ok=False, raw file persisted for retry
 10. _parse_args parses --dry-run, --date, combinations

Run from the jarvis dir:
    python test_daily_ingest.py
Exit code 0 on success, non-zero on any failure.
"""

import datetime
import shutil
import sys
import tempfile
import types
from pathlib import Path

# Import daily_ingest BEFORE we install the fake brain_wiring. That's safe
# because daily_ingest only imports brain_wiring lazily inside ingest() -
# our fake just needs to be in sys.modules before each ingest() call.
import daily_ingest as di


# --- Fake Brain ------------------------------------------------------------

class _FakeBrain:
    """Stand-in for brain_wiring.BRAIN. Exposes the same surface the
    daily_ingest module touches: wiki_dir, raw_dir, ingest(), _log()."""
    def __init__(self, wiki_dir, raw_dir):
        self.wiki_dir = wiki_dir
        self.raw_dir = raw_dir
        self.ingest_calls = []   # list of {"filename": str, "dry_run": bool}
        self.log_calls = []      # list of (category, message)
        # Default ingest() return shape - mirrors brain.py's contract
        self.next_result = {
            "tldr": "Fake TLDR for testing.",
            "entities_touched": ["thing_a", "thing_b"],
            "concepts_touched": ["concept_x"],
        }
        self.raise_next = None   # set to an Exception to force ingest() to raise

    def ingest(self, filename, dry_run=False):
        self.ingest_calls.append({"filename": filename, "dry_run": dry_run})
        if self.raise_next:
            raise self.raise_next
        result = dict(self.next_result)
        # Real Brain derives slug from src.stem. Mirror that so the test
        # exercises the slug-construction path in daily_ingest.
        result["slug"] = filename[:-3] if filename.endswith(".md") else filename
        return result

    def _log(self, category, message):
        self.log_calls.append((category, message))


def _install_fake_brain_wiring(wiki, raw):
    """Replace sys.modules['brain_wiring'] with a stub exporting our fake
    BRAIN. daily_ingest's late `from brain_wiring import BRAIN` will then
    pick up this fake instead of trying to construct the real thing
    (which would require Groq + on-disk wiki + ...)."""
    fake = _FakeBrain(wiki, raw)
    mod = types.ModuleType("brain_wiring")
    mod.BRAIN = fake
    sys.modules["brain_wiring"] = mod
    return fake


PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  ({detail})")


def _setup_dirs():
    """Make a tmp tree with wiki/ and raw/ subdirs. Caller is responsible
    for shutil.rmtree on the returned root."""
    tmp = Path(tempfile.mkdtemp(prefix="di_test_"))
    wiki = tmp / "wiki"
    raw = tmp / "raw"
    wiki.mkdir()
    raw.mkdir()
    return tmp, wiki, raw


def _write(p, body):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# --- Unit tests on helpers -------------------------------------------------

def test_strip_frontmatter():
    print("\n_strip_frontmatter:")
    body = "---\ntitle: Hi\ntags: [a,b]\n---\n\nThe body."
    check("standard frontmatter stripped",
          di._strip_frontmatter(body) == "The body.",
          repr(di._strip_frontmatter(body)))

    body = "# Heading\n\nSome content."
    check("no frontmatter returns unchanged",
          di._strip_frontmatter(body) == body)

    check("empty input returns empty",
          di._strip_frontmatter("") == "")

    body = "---\nfoo: bar\nstill in fm with no closer"
    check("unterminated frontmatter left alone",
          di._strip_frontmatter(body) == body)

    # Multi-key frontmatter
    body = "---\na: 1\nb: 2\nc: 3\n---\nbody here"
    check("multi-key frontmatter stripped",
          di._strip_frontmatter(body) == "body here")


def test_slug_for():
    print("\n_slug_for:")
    check("ISO date slug", di._slug_for("2026-05-12") == "daily_2026-05-12")
    s = di._slug_for("2026-05-12")
    check("slug only contains alnum, underscore, dash",
          all(c.isalnum() or c in "_-" for c in s),
          s)
    check("slug <= 80 chars", len(s) <= 80, s)


# --- Integration-ish tests -------------------------------------------------

def test_missing_note():
    print("\nmissing daily note:")
    tmp, wiki, raw = _setup_dirs()
    brain = _install_fake_brain_wiring(wiki, raw)
    try:
        r = di.ingest(target_date=datetime.date(2026, 5, 1))
        check("ok=True", r.get("ok") is True, str(r))
        check("skipped key present", "skipped" in r, str(r))
        check("BRAIN.ingest not called", brain.ingest_calls == [])
        check("no raw file written", not any(raw.iterdir()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_note():
    print("\nempty / whitespace daily note:")
    tmp, wiki, raw = _setup_dirs()
    _write(wiki / "daily" / "2026-05-01.md", "  \n\n")
    brain = _install_fake_brain_wiring(wiki, raw)
    try:
        r = di.ingest(target_date=datetime.date(2026, 5, 1))
        check("ok=True on empty", r.get("ok") is True, str(r))
        check("skipped key present", "skipped" in r, str(r))
        check("BRAIN.ingest not called", brain.ingest_calls == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_normal_ingest():
    print("\nnormal daily note ingest:")
    tmp, wiki, raw = _setup_dirs()
    note_body = ("# 2026-05-01\n\n"
                 "Finished the wiki write-back loop. Met with the team "
                 "about the Q3 roadmap.\n\n"
                 "- ship demo reel\n"
                 "- write up the hiring loop\n")
    _write(wiki / "daily" / "2026-05-01.md", note_body)
    brain = _install_fake_brain_wiring(wiki, raw)
    try:
        r = di.ingest(target_date=datetime.date(2026, 5, 1))
        check("ok=True", r.get("ok") is True, str(r))
        check("slug echoes filename stem",
              r.get("slug") == "daily_2026-05-01",
              str(r))
        check("raw file written",
              (raw / "daily_2026-05-01.md").exists(),
              str(list(raw.iterdir())))

        wrapped = (raw / "daily_2026-05-01.md").read_text(encoding="utf-8")
        check("provenance heading present",
              "# Daily Note - 2026-05-01" in wrapped)
        check("provenance ingested-from line present",
              "Ingested from" in wrapped)
        check("original body present",
              "wiki write-back loop" in wrapped)

        check("BRAIN.ingest called exactly once",
              len(brain.ingest_calls) == 1,
              str(brain.ingest_calls))
        check("BRAIN.ingest filename correct",
              brain.ingest_calls[0]["filename"] == "daily_2026-05-01.md",
              str(brain.ingest_calls))

        check("entities_touched is an int count",
              r.get("entities_touched") == 2,
              str(r))
        check("concepts_touched is an int count",
              r.get("concepts_touched") == 1,
              str(r))

        check("_log called once",
              len(brain.log_calls) == 1,
              str(brain.log_calls))
        check("_log category is daily_ingest",
              brain.log_calls and brain.log_calls[0][0] == "daily_ingest",
              str(brain.log_calls))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_frontmatter_stripped_from_wrapped():
    print("\nfrontmatter NOT echoed into wrapped body:")
    tmp, wiki, raw = _setup_dirs()
    note_body = ("---\n"
                 "tags: [daily, work]\n"
                 "cssclass: daily-note\n"
                 "secret_property: should-not-appear\n"
                 "---\n\n"
                 "# Today's actual content\n\n"
                 "Worked on the wiki write-back loop integration test.\n")
    _write(wiki / "daily" / "2026-05-01.md", note_body)
    brain = _install_fake_brain_wiring(wiki, raw)
    try:
        di.ingest(target_date=datetime.date(2026, 5, 1))
        wrapped = (raw / "daily_2026-05-01.md").read_text(encoding="utf-8")

        check("cssclass property absent from wrapped",
              "cssclass" not in wrapped, wrapped[:300])
        check("secret_property absent from wrapped",
              "secret_property" not in wrapped, wrapped[:300])
        check("body content still present",
              "Worked on the wiki write-back loop" in wrapped)
        # Exactly one `---` separator survives (the one daily_ingest adds
        # between the provenance line and the body). Frontmatter's two
        # `---` lines should have been removed.
        check("only the daily_ingest separator remains (one '---')",
              wrapped.count("---") == 1,
              f"got {wrapped.count('---')} '---' occurrences")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_default_date_is_yesterday():
    print("\ndefault target_date is yesterday:")
    tmp, wiki, raw = _setup_dirs()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    _write(wiki / "daily" / f"{yesterday}.md",
           "Yesterday's note. Wrote up the cabling diagram and the HUD spec.")
    brain = _install_fake_brain_wiring(wiki, raw)
    try:
        r = di.ingest()  # no target_date - defaults to yesterday
        check("ok=True", r.get("ok") is True, str(r))
        check("raw file at yesterday slug",
              (raw / f"daily_{yesterday}.md").exists(),
              str(list(raw.iterdir())))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dry_run_no_writes():
    print("\ndry-run does not write or call BRAIN.ingest:")
    tmp, wiki, raw = _setup_dirs()
    _write(wiki / "daily" / "2026-05-01.md",
           "# Today\n\nReal content here that is long enough to pass MIN_BODY_CHARS.")
    brain = _install_fake_brain_wiring(wiki, raw)
    try:
        r = di.ingest(target_date=datetime.date(2026, 5, 1), dry_run=True)
        check("ok=True", r.get("ok") is True, str(r))
        check("dry_run flag in result", r.get("dry_run") is True, str(r))
        check("raw_path key in result", "raw_path" in r, str(r))
        check("no raw file written",
              not (raw / "daily_2026-05-01.md").exists())
        check("BRAIN.ingest not called", brain.ingest_calls == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_error_surfaced():
    print("\nBRAIN.ingest raising -> ok=False, raw file persisted:")
    tmp, wiki, raw = _setup_dirs()
    _write(wiki / "daily" / "2026-05-01.md",
           "# Today\n\nReal content here that is long enough to pass MIN_BODY_CHARS.")
    brain = _install_fake_brain_wiring(wiki, raw)
    brain.raise_next = RuntimeError("ollama timeout")
    try:
        r = di.ingest(target_date=datetime.date(2026, 5, 1))
        check("ok=False on ingest error", r.get("ok") is False, str(r))
        check("error message includes exception type",
              "RuntimeError" in r.get("error", ""), str(r))
        check("error message includes exception message",
              "ollama timeout" in r.get("error", ""), str(r))
        # Raw file should still have been written - only ingest() failed.
        # That way Ed can retry with `/ingest daily_<date>.md` without
        # re-reading the Obsidian source.
        check("raw file persisted for manual retry",
              (raw / "daily_2026-05-01.md").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- CLI parse tests -------------------------------------------------------

def test_parse_args():
    print("\n_parse_args:")
    t, d = di._parse_args([])
    check("no args -> (None, False)", t is None and d is False)

    t, d = di._parse_args(["--dry-run"])
    check("--dry-run", t is None and d is True)

    t, d = di._parse_args(["-n"])
    check("-n short flag", t is None and d is True)

    t, d = di._parse_args(["--date", "2026-05-09"])
    check("--date YYYY-MM-DD",
          t == datetime.date(2026, 5, 9) and d is False,
          f"got ({t}, {d})")

    t, d = di._parse_args(["--date", "2026-05-09", "-n"])
    check("--date plus -n",
          t == datetime.date(2026, 5, 9) and d is True,
          f"got ({t}, {d})")

    t, d = di._parse_args(["-n", "--date", "2026-01-01"])
    check("flag order independence",
          t == datetime.date(2026, 1, 1) and d is True,
          f"got ({t}, {d})")


# --- main ------------------------------------------------------------------

if __name__ == "__main__":
    test_strip_frontmatter()
    test_slug_for()
    test_missing_note()
    test_empty_note()
    test_normal_ingest()
    test_frontmatter_stripped_from_wrapped()
    test_default_date_is_yesterday()
    test_dry_run_no_writes()
    test_ingest_error_surfaced()
    test_parse_args()

    print(f"\n{'=' * 50}")
    print(f"PASSED: {PASSED}")
    print(f"FAILED: {FAILED}")
    if FAILED:
        sys.exit(1)
