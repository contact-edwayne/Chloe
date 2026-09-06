"""test_system_prompt_consolidation.py - Regression test for action item
#8 (audit Part 13, "Consolidate system prompt (5->1 injection points)").

The original audit (Part 8) found the effective system prompt for a
turn assembled across five independent places: _voice_system(),
_augmented_voice_system(), _mode_block(), _TOOL_DOCS_FOR_PROMPT, and
chloe_persona.py's about.md trim -- "no single function or file can be
read to see 'the' system prompt for a turn." Two of those five
(_mode_block, _TOOL_DOCS_FOR_PROMPT) were already single, shared call
sites; the real, fixable duplication was that _handle_chat_inner (chat)
built its own near-identical date+search-capability preamble inline
instead of sharing _voice_system's logic, and both _handle_chat_inner
and _augmented_voice_system separately built an identical
about/mode/facts 3-line sequence.

This is a pure text-generation refactor -- _build_turn_preamble() and
_persona_mode_facts_blocks() are meant to produce EXACTLY the same
strings the old, separate implementations produced, just from one
shared place. This test is a golden-string snapshot: the expected
strings below were captured directly from jarvis.py's source BEFORE the
refactor (not re-derived from the new code), so a divergence here means
the refactor actually changed behavior, not just moved code.

jarvis.py cannot be imported directly -- see
test_grounding_and_barge_in.py's docstring for why. Both functions
under test are pure (no I/O, no globals besides MODEL_SEARCH and
_central_now()), so they're extracted via ast and exec'd with a frozen
_central_now stub so the "Today's date is ..." line is deterministic.

Run from the jarvis dir:
    python test_system_prompt_consolidation.py
Exit code 0 on success, non-zero on any failure.
"""
import ast
import datetime
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
    wanted = set(names)
    found = {}
    for node in _TREE.body:
        target_names = set()
        if isinstance(node, ast.FunctionDef):
            target_names = {node.name}
        elif isinstance(node, ast.Assign):
            target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = {node.target.id}
        for n in target_names & wanted:
            found[n] = ast.get_source_segment(_SOURCE_TEXT, node)
    missing = wanted - found.keys()
    assert not missing, f"not found at module level in jarvis.py: {sorted(missing)}"
    return "\n\n".join(found[n] for n in names)


_FROZEN_TODAY = "Saturday, September 06, 2026"


class _FrozenDatetime:
    @staticmethod
    def strftime(fmt):
        return _FROZEN_TODAY


def _central_now():
    return _FrozenDatetime()


_ns = {"_central_now": _central_now, "MODEL_SEARCH": "compound-mini"}
exec(_extract("_build_turn_preamble"), _ns)
_build_turn_preamble = _ns["_build_turn_preamble"]


# --- Golden strings, captured verbatim from jarvis.py BEFORE this
# session's consolidation refactor (the original, separate
# _voice_system()/_handle_chat_inner bodies) -----------------------------

_VOICE_SEARCHABLE_GOLDEN = (
    "You are Chloe, a personal home assistant speaking to Ed via voice. "
    "Today's date is Saturday, September 06, 2026 — you DO know the current date and should never "
    "apologize about not knowing it.\n\n"
    "You can search the web automatically when needed. For anything that may "
    "have changed since training (current prices, weather, news, sports scores, "
    "recent events, who currently holds a position), search the web and give "
    "Ed the answer. For knowledge you already have (general facts, math, "
    "conversation, advice, writing), answer directly without searching.\n\n"
    "NEVER invent numbers or facts — if you can't find something, say so plainly.\n\n"
    "STYLE:\n"
    "- Reply in plain spoken sentences. No bullet points, markdown, or lists.\n"
    "- Keep replies short, friendly, and conversational — usually one or two "
    "sentences.\n"
    "- Do NOT cite URLs or list sources unless Ed asks; he's listening, not reading."
)

_VOICE_NO_SEARCH_GOLDEN = (
    "You are Chloe, a personal home assistant speaking to Ed via voice. "
    "Today's date is Saturday, September 06, 2026 — you DO know the current date and should never "
    "apologize about not knowing it.\n\n"
    "For this turn you do NOT have web search available. If Ed asks for current/live "
    "data (prices, weather, news, scores, current officeholders), tell him plainly "
    "that you'd need to look it up — don't invent the answer. For things you already "
    "know (general knowledge, conversation, advice, writing, math), answer directly "
    "without disclaimers.\n\n"
    "TOOLS:\n"
    "- You have a `grep_source` tool. CALL IT whenever Ed asks about your own "
    "implementation, behaviour, configuration, or 'what does X do' / 'how do you Y' "
    "questions about your code. Quoting actual lines is more useful than guessing "
    "from memory. Pass a regex pattern (e.g., 'def handle_chat', 'CHLOE_MIC_GAIN'). "
    "After the tool returns matches, summarise them naturally in spoken English — "
    "don't read filenames or line numbers aloud unless Ed asks.\n"
    "- You have a Bitcoin Lightning wallet. Tools: `wallet_balance`, "
    "`wallet_invoice`, `wallet_send`, `wallet_history`. Speak amounts in "
    "sats. For `wallet_send`, ALWAYS require Ed to give you a PIN this "
    "turn — never invent or reuse a previous PIN. If he hasn't given "
    "one, ask for it BEFORE calling the tool. The system enforces a "
    "daily spend cap server-side; if a send is refused, relay the "
    "reason and stop.\n\n"
    "STYLE:\n"
    "- Reply in plain spoken sentences. No bullet points, markdown, or lists.\n"
    "- Keep replies short, friendly, and conversational — usually one or two "
    "sentences."
)

_CHAT_SEARCH_GOLDEN = (
    "Today's date is Saturday, September 06, 2026.\n"
    "You can search the web automatically when needed. Use search for anything "
    "that may have changed since your training (current prices, weather, news, "
    "sports scores, recent events, who currently holds a position). For things "
    "you already know, just answer directly. NEVER invent numbers or facts — "
    "search instead, or say you couldn't find it."
)

_CHAT_NO_SEARCH_GOLDEN = (
    "Today's date is Saturday, September 06, 2026 — you know the current date and should not claim otherwise.\n"
    "For this turn you do NOT have web search available. If the question requires "
    "current/live data (prices, weather, news, scores, who currently holds a "
    "position) tell the user you'd need to look it up — don't invent the answer. "
    "For general knowledge, conversation, or anything you already know, answer "
    "directly without disclaimers."
)


def test_voice_searchable_branch_matches_pre_refactor_text():
    result = _build_turn_preamble("compound-mini", voice=True)
    check("_build_turn_preamble(voice=True, searchable model) reproduces "
          "the old _voice_system()'s search-capable branch byte-for-byte",
          result == _VOICE_SEARCHABLE_GOLDEN, result)


def test_voice_no_model_defaults_to_searchable():
    # Old _voice_system: can_search = (model == MODEL_SEARCH) if model
    # else True -- model=None must still take the searchable branch.
    result = _build_turn_preamble(None, voice=True)
    check("voice with model=None still defaults to the searchable "
          "branch (can_search defaults True when model is falsy)",
          result == _VOICE_SEARCHABLE_GOLDEN, result)


def test_voice_non_search_branch_matches_pre_refactor_text():
    result = _build_turn_preamble("llama3.2:3b", voice=True)
    check("_build_turn_preamble(voice=True, non-search model) reproduces "
          "the old _voice_system()'s non-search branch byte-for-byte, "
          "including the TOOLS blurb and PIN requirement",
          result == _VOICE_NO_SEARCH_GOLDEN, result)


def test_chat_search_branch_matches_pre_refactor_text():
    result = _build_turn_preamble("compound-mini", voice=False)
    check("_build_turn_preamble(voice=False, MODEL_SEARCH) reproduces "
          "the old _handle_chat_inner inline preamble's search branch "
          "byte-for-byte", result == _CHAT_SEARCH_GOLDEN, result)


def test_chat_non_search_branch_matches_pre_refactor_text():
    result = _build_turn_preamble("llama3.2:3b", voice=False)
    check("_build_turn_preamble(voice=False, non-search model) "
          "reproduces the old _handle_chat_inner inline preamble's "
          "non-search branch byte-for-byte -- notably NO tools blurb, "
          "NO 'You are Chloe' framing, NO spoken-style rules, since "
          "chat gets those from other blocks (about_block/"
          "_TOOL_DOCS_FOR_PROMPT) that this function was never "
          "responsible for",
          result == _CHAT_NO_SEARCH_GOLDEN, result)


def test_chat_and_voice_branches_are_deliberately_different():
    # Guards against someone "helpfully" unifying the wording later and
    # silently changing behavior -- see this file's own module docstring
    # and _build_turn_preamble's in-source comment for why they differ.
    voice_text = _build_turn_preamble("llama3.2:3b", voice=True)
    chat_text = _build_turn_preamble("llama3.2:3b", voice=False)
    check("voice's non-search preamble still contains the TOOLS blurb "
          "chat intentionally does not get here",
          "TOOLS:" in voice_text and "TOOLS:" not in chat_text)
    check("only voice's preamble opens with 'You are Chloe... speaking "
          "to Ed via voice' framing", "speaking to Ed via voice" in voice_text
          and "speaking to Ed via voice" not in chat_text)


# --- _persona_mode_facts_blocks: source-text guard (this one has real
# dependencies -- ChloeMemory, chloe_persona, _mode_block -- that aren't
# worth stubbing just to prove a 3-line extraction moved verbatim; the
# call-site source itself is the thing that matters here) --------------

def test_persona_mode_facts_blocks_function_exists_and_is_shared():
    check("_persona_mode_facts_blocks is defined once at module level",
          _SOURCE_TEXT.count("def _persona_mode_facts_blocks(") == 1)
    check("_handle_chat_inner's chat path calls the shared helper "
          "instead of its own inline about/mode/facts sequence",
          "about_block, mode_block, facts_block = _persona_mode_facts_blocks(\n"
          "        user_text_for_recall, voice=False)" in _SOURCE_TEXT)
    check("_augmented_voice_system's voice path calls the same shared "
          "helper instead of its own copy of that sequence",
          "about_block, mode_block, facts_block = _persona_mode_facts_blocks(\n"
          "        user_text, voice=True)" in _SOURCE_TEXT)


def test_voice_system_is_now_a_thin_wrapper():
    check("_voice_system is now a one-line wrapper around "
          "_build_turn_preamble(voice=True), not a second copy of the "
          "prose", "return _build_turn_preamble(model, voice=True)" in _SOURCE_TEXT)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
