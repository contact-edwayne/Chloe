"""test_email_encoding.py - Regression test for the IMAP UnicodeEncodeError
crash fixed 2026-09-06 (see _gm_raw_arg's own docstring in
email_client.py, "BUG FIXED 2026-09-06p"): Python's stdlib imaplib
encodes the whole IMAP command line as ASCII before sending, so any
non-ASCII character in a sender/subject delete search -- even an
ordinary HTML-email en-dash -- raised UnicodeEncodeError outright.
_gm_raw_arg normalizes common typographic punctuation to ASCII first,
then strips anything left over as a last resort, so the command can
always be sent.

email_client.py is safe to import directly (no live connection happens
at import time, only inside functions that open a real IMAP socket),
unlike jarvis.py -- see test_grounding_and_barge_in.py's docstring for
why that one can't be imported directly.

Run from the jarvis dir:
    python test_email_encoding.py
Exit code 0 on success, non-zero on any failure.
"""
from email_client import _gm_raw_arg

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


def test_result_is_always_pure_ascii():
    # The actual crash: imaplib._command encodes with .encode("ASCII"),
    # so this is the property that matters most -- whatever comes out
    # of _gm_raw_arg must never contain a non-ASCII codepoint.
    tricky = ("Remote Life Insurance Advisor – Full Training — "
              "Sarah’s “best” offer…")
    result = _gm_raw_arg(tricky)
    try:
        result.encode("ascii")
        ok = True
    except UnicodeEncodeError:
        ok = False
    check("_gm_raw_arg's output always encodes cleanly as ASCII (the "
          "exact live crash from imaplib._command)", ok, result)


def test_en_dash_from_the_live_crash_normalizes_to_hyphen():
    result = _gm_raw_arg("Remote Life Insurance Advisor – Full Training")
    check("the exact en-dash from the live crash (\\u2013) becomes '-', "
          "not stripped or left as-is",
          "–" not in result and "-" in result, result)


def test_em_dash_normalizes_to_hyphen():
    result = _gm_raw_arg("a—b")
    check("em-dash (\\u2014) becomes '-'",
          "—" not in result and "-" in result, result)


def test_smart_quotes_normalize_to_straight():
    result = _gm_raw_arg("Sarah’s “best” offer")
    check("smart single/double quotes normalize to straight ASCII quotes "
          "(then get backslash-escaped like any other quote)",
          "’" not in result and "“" not in result
          and "”" not in result, result)


def test_ellipsis_normalizes_to_three_dots():
    result = _gm_raw_arg("wait…")
    check("ellipsis character (\\u2026) becomes '...'",
          "…" not in result and "..." in result, result)


def test_plain_ascii_query_is_unchanged_apart_from_quoting():
    result = _gm_raw_arg("Indeed Apply")
    check("an ordinary ASCII query round-trips as itself, just wrapped "
          "in the IMAP-literal double quotes",
          result == '"Indeed Apply"', result)


def test_embedded_double_quote_is_escaped():
    result = _gm_raw_arg('subject:"urgent"')
    check("an embedded double quote in the query is backslash-escaped, "
          "not left to break the IMAP literal",
          '\\"urgent\\"' in result, result)


def test_embedded_backslash_is_escaped():
    result = _gm_raw_arg("path\\to\\thing")
    check("an embedded backslash is escaped before quotes are (order "
          "matters -- escaping quotes first would double the backslash "
          "the quote-escape itself adds)",
          "\\\\" in result, result)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
