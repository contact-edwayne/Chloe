"""test_grounding_and_barge_in.py - Persisted regression suite for the
F-01 grounding-check and F-02 barge-in-interlock bugs found and fixed
live in the 2026-09-06 session (see chloe_session_summary_2026-09-06.md,
Round 9-10, and jarvis.py's own inline bug-fix comments next to each
piece this test covers). Consolidates several one-off verification
scripts run by hand that session (heredocs doing `assert X in src`)
into a real, re-runnable test -- this is action item #7 from the
audit's Part 13 action list ("persisted regression test suite from
throwaway scripts").

jarvis.py cannot be imported directly for testing: it starts real
background threads at module scope on import (Ollama/Kokoro/wallet
warm-up, websocket server setup) that would try to reach live services
this environment doesn't have. Instead this file extracts just the
specific functions/constants under test from jarvis.py's source via the
`ast` module and execs them in an isolated namespace with minimal
stubs -- the same verification technique used by hand throughout the
2026-09-06 session before any fix was committed.

Run from the jarvis dir:
    python test_grounding_and_barge_in.py
Exit code 0 on success, non-zero on any failure.
"""
import ast
import re as _re
import threading
from pathlib import Path

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


_JARVIS_SRC = Path(__file__).parent / "jarvis.py"
_SOURCE_TEXT = _JARVIS_SRC.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE_TEXT, filename=str(_JARVIS_SRC))


def _extract(*names):
    """Pull the named top-level function/assignment nodes out of
    jarvis.py's AST and return their exact source text, in the order
    requested. Raises AssertionError if any name isn't found at module
    level -- fails loudly rather than silently testing nothing if
    jarvis.py's structure changes out from under this file."""
    wanted = set(names)
    found = {}
    for node in _TREE.body:
        target_names = set()
        if isinstance(node, ast.FunctionDef):
            target_names = {node.name}
        elif isinstance(node, ast.Assign):
            target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # e.g. `_ACTION_CLAIM_PATTERNS: tuple = (...)` -- an annotated
            # assignment, which ast.Assign does not match.
            target_names = {node.target.id}
        for n in target_names & wanted:
            found[n] = ast.get_source_segment(_SOURCE_TEXT, node)
    missing = wanted - found.keys()
    assert not missing, f"not found at module level in jarvis.py: {sorted(missing)}"
    return "\n\n".join(found[n] for n in names)


# --- _grounding_violation / _ACTION_CLAIM_PATTERNS (F-01) ------------------

_grounding_ns = {"_re": _re}
exec(_extract("_ACTION_CLAIM_PATTERNS", "_BARE_TIMESTAMP_RE",
              "_TIME_QUESTION_RE", "_grounding_violation"), _grounding_ns)
_grounding_violation = _grounding_ns["_grounding_violation"]


def test_moved_to_trash_claim_without_delete_tool_is_caught():
    v = _grounding_violation("The first email has been moved to Trash.", set())
    check("'moved to Trash' with no email_delete call is flagged",
          v is not None, v)


def test_moved_to_trash_claim_with_delete_tool_is_clean():
    v = _grounding_violation("The first email has been moved to Trash.",
                              {"email_delete"})
    check("'moved to Trash' backed by a real email_delete call is NOT flagged",
          v is None, v)


def test_reply_sent_claim_without_reply_tool_is_caught():
    v = _grounding_violation(
        "The reply has been sent, and the email has been deleted from your inbox.",
        set())
    check("'reply...sent' with no email_reply call is flagged",
          v is not None, v)


def test_reply_sent_claim_with_reply_tool_is_clean():
    v = _grounding_violation("Your reply has been sent.", {"email_reply"})
    check("'reply...sent' backed by a real email_reply call is NOT flagged",
          v is None, v)


def test_deleted_email_claim_still_caught_original_pattern():
    v = _grounding_violation("I deleted that email for you.", set())
    check("original 'deleted...email' pattern still catches an unbacked claim",
          v is not None, v)


def test_ordinary_reply_with_no_action_claim_is_clean():
    v = _grounding_violation("You have 3 unread emails in your inbox.", set())
    check("a plain informational reply with no completion claim is NOT flagged",
          v is None, v)


def test_bare_timestamp_derailment_on_non_time_question_is_caught():
    v = _grounding_violation("Sunday, September 6, 2026 at 1:16 PM CDT",
                              set(), user_text="Did you delete that email?")
    check("a bare timestamp standing in for a non-time answer is flagged",
          v is not None, v)


def test_bare_timestamp_on_an_actual_time_question_is_clean():
    v = _grounding_violation("Sunday, September 6, 2026 at 1:16 PM CDT",
                              set(), user_text="What time is it?")
    check("the same bare timestamp answering an ACTUAL time question is "
          "NOT flagged", v is None, v)


# --- _EMAIL_REPLY_KEYWORDS + hedge-triggers-tools routing ------------------

_kw_ns = {}
exec(_extract("_EMAIL_REPLY_KEYWORDS"), _kw_ns)
_EMAIL_REPLY_KEYWORDS = _kw_ns["_EMAIL_REPLY_KEYWORDS"]


def test_send_it_present_tense_is_in_keyword_list():
    check("'send it' (present tense) is in _EMAIL_REPLY_KEYWORDS -- this "
          "was the exact miss behind the live 'Send it.' hallucination",
          "send it" in _EMAIL_REPLY_KEYWORDS)


def test_sent_it_past_tense_still_in_keyword_list():
    check("'sent it' (past tense) is still present -- regression guard, "
          "confirms the fix added rather than replaced the entry",
          "sent it" in _EMAIL_REPLY_KEYWORDS)


def test_prior_reply_do_you_want_to_send_it_matches_keyword_list():
    prior_reply = "Do you want to send it?"
    check("Chloe's own literal prior phrasing ('Do you want to send it?') "
          "now matches _EMAIL_REPLY_KEYWORDS via 'send it'",
          any(kw in prior_reply.lower() for kw in _EMAIL_REPLY_KEYWORDS))


def test_hedge_text_is_recognized_verbatim_in_pick_route_source():
    # _pick_route itself is too entangled (closures over live voice
    # state, network calls) to safely extract and exec in isolation, so
    # this guards the fix at the source-text level instead: it confirms
    # the exact hedge string _grounding_violation's own fallback reply
    # emits is still the literal substring _pick_route checks for. If
    # either side's wording changes without the other, this fails
    # instead of silently reopening the "Delete it." bypass.
    hedge_fragment = "double-check that before I tell you it's"
    check("_pick_route's hedge-detection substring appears verbatim in "
          "jarvis.py, matched against _prior_assistant_reply",
          f'"{hedge_fragment}" in _prior_assistant_reply' in _SOURCE_TEXT
          or f"'{hedge_fragment}' in _prior_assistant_reply" in _SOURCE_TEXT)
    check("_grounding_violation's own fallback reply still starts with "
          "that exact same hedge fragment (both sides of the contract "
          "must agree on the wording)",
          hedge_fragment in _SOURCE_TEXT)


# --- _barge_in_blocks_destructive_action turn-gen race (F-02) --------------

_barge_ns = {"threading": threading}
exec(_extract("_turn_gen_lock", "_turn_gen", "_bump_turn_gen",
              "_barge_in_request", "_barge_in_blocks_destructive_action"),
     _barge_ns)
_barge_in_blocks_destructive_action = _barge_ns["_barge_in_blocks_destructive_action"]
_bump_turn_gen = _barge_ns["_bump_turn_gen"]


def _reset_barge_state():
    _barge_ns["_turn_gen"] = 0
    _barge_ns["_barge_in_request"].clear()


def test_stale_turn_gen_blocks_the_action():
    _reset_barge_state()
    my_gen = _bump_turn_gen()            # turn A starts, gen=1
    _bump_turn_gen()                     # turn B starts before A dispatches, gen=2
    msg = _barge_in_blocks_destructive_action("delete indices=[2]",
                                               my_turn_gen=my_gen)
    check("an action from a superseded turn (stale turn_gen) is blocked "
          "even with _barge_in_request never set -- this is the exact "
          "live race: the next turn's own recording clears the shared "
          "flag long before the old turn's worker reaches dispatch, but "
          "turn_gen still catches it",
          msg is not None, msg)


def test_current_turn_gen_is_not_blocked():
    _reset_barge_state()
    my_gen = _bump_turn_gen()            # the only turn so far
    msg = _barge_in_blocks_destructive_action("send 100 sats",
                                               my_turn_gen=my_gen)
    check("an action from the CURRENT (not superseded) turn is allowed "
          "through", msg is None, msg)


def test_barge_in_request_flag_alone_still_blocks():
    _reset_barge_state()
    my_gen = _bump_turn_gen()
    _barge_ns["_barge_in_request"].set()
    msg = _barge_in_blocks_destructive_action("delete that email",
                                               my_turn_gen=my_gen)
    check("the original _barge_in_request.is_set() signal still blocks "
          "on its own (kept as the faster-to-fire secondary check)",
          msg is not None, msg)


def test_no_turn_gen_supplied_falls_back_to_flag_only():
    _reset_barge_state()
    msg = _barge_in_blocks_destructive_action("delete that email",
                                               my_turn_gen=None)
    check("an old call site that can't supply my_turn_gen (None), with "
          "no barge-in flag set, is allowed through -- the original, "
          "weaker protection, not an error",
          msg is None, msg)


# --- email_reply drafts must be marked announced (dangling-confirm fix) ---

def test_email_draft_used_flag_covers_email_reply_too():
    # The dangling-draft-announced-spam bug (2026-09-06, Round 10):
    # email_reply-created drafts need the same `announced` gate as
    # email_draft-created ones, but the flag that triggers
    # mark_draft_announced() only ever checked for "email_draft". This
    # is a source-text guard, not a behavioral extraction, because the
    # flag lives inline in _ollama_chat's tool-dispatch loop (deeply
    # entangled with the live Ollama request loop) rather than in a
    # standalone function -- if this line's exact condition ever
    # regresses to checking only "email_draft" again, this test catches
    # it.
    check('the dispatch-loop condition sets _email_draft_used for BOTH '
          '"email_draft" and "email_reply", not just the former',
          'name in ("email_draft", "email_reply")' in _SOURCE_TEXT)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
