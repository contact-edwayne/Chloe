"""test_pick_route_logging.py - Regression test for action item #6
(route-decision logging for _pick_route()): every branch of _pick_route
now prints a "[chloe] route: ..." line naming the chosen route and WHY
it was chosen, before returning. This is a direct response to the
routing bugs found live in the 2026-09-06 session (the deictic misroute,
the hedge-bypass, the send-it keyword miss) -- all four were only
discoverable by inferring backwards from a wrong downstream reply,
because _pick_route's decision itself was invisible in the log. This
test guards two things: (1) each branch still returns the route it's
supposed to, and (2) each branch actually emits a log line, so a future
edit can't silently drop the print and reopen that same blind spot.

_pick_route itself is extracted from jarvis.py via `ast` (see
test_grounding_and_barge_in.py's docstring for why jarvis.py can't be
imported directly). Its callees (_is_introspection_query,
_needs_realtime, _ollama_available, etc.) are deliberately replaced
with small controllable stubs rather than the real keyword-matching
implementations -- this test is scoped to "does _pick_route select and
log the right branch given a decision," not "are the keyword lists
themselves correct" (those have their own scope elsewhere). Real
_EMAIL_REPLY_KEYWORDS is used as-is in the hedge/deictic branch since
that fix is specifically about the *routing*, not the keyword matching.

Run from the jarvis dir:
    python test_pick_route_logging.py
Exit code 0 on success, non-zero on any failure.
"""
import ast
import contextlib
import io
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


class _StubToggle:
    """Lets a test flip a stub function's return value for one call."""
    def __init__(self, value=False):
        self.value = value


class _NsfwStub:
    def __init__(self):
        self.enabled = False
        self.adult = False

    def is_enabled(self):
        return self.enabled

    def looks_adult(self, text):
        return self.adult


def _build_namespace(*, ollama_up=True, introspection=False, extra_tool=False,
                      deictic=False, self_knowledge=False, realtime=False,
                      info_question=False, warm=True, nsfw=None):
    ns = {
        "threading": threading,
        "_is_introspection_query": lambda t: introspection,
        "_is_extra_tool_query": lambda t: extra_tool,
        "_has_unresolved_deictic": lambda t, w: deictic,
        "_is_self_knowledge_query": lambda t: self_knowledge,
        "_needs_realtime": lambda t: realtime,
        "_looks_like_info_question": lambda t: info_question,
        "_ollama_available": lambda: ollama_up,
        "nsfw_mode": nsfw or _NsfwStub(),
    }
    exec(_extract("_EMAIL_REPLY_KEYWORDS"), ns)
    ns["_voice_history"] = []
    ns["_ollama_primary_warm"] = threading.Event()
    if warm:
        ns["_ollama_primary_warm"].set()
    exec(_extract("_pick_route"), ns)
    return ns


def _call(ns, user_text, prior_assistant_reply=None):
    if prior_assistant_reply is not None:
        ns["_voice_history"].append({"role": "user", "content": "earlier turn"})
        ns["_voice_history"].append({"role": "assistant",
                                      "content": prior_assistant_reply})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        route = ns["_pick_route"](user_text)
    return route, buf.getvalue()


def test_nsfw_override_routes_and_logs():
    ns = _build_namespace(ollama_up=True, nsfw=_NsfwStub())
    ns["nsfw_mode"].enabled = True
    ns["nsfw_mode"].adult = True
    route, log = _call(ns, "some adult-coded request")
    check("nsfw override returns 'ollama'", route == "ollama", route)
    check("nsfw override logs a route line naming the branch",
          "route:" in log and "nsfw" in log.lower(), log)


def test_introspection_routes_and_logs():
    ns = _build_namespace(ollama_up=True, introspection=True)
    route, log = _call(ns, "what's your mic gain set to?")
    check("introspection query returns 'ollama_tools' when Ollama is up",
          route == "ollama_tools", route)
    check("introspection branch logs its reason",
          "route:" in log and "introspection" in log.lower(), log)


def test_introspection_falls_back_when_ollama_down():
    ns = _build_namespace(ollama_up=False, introspection=True)
    route, log = _call(ns, "what's your mic gain set to?")
    check("introspection query falls back to 'local_chat' when Ollama "
          "is unreachable", route == "local_chat", route)


def test_extra_tool_query_routes_and_logs():
    ns = _build_namespace(ollama_up=True, extra_tool=True)
    route, log = _call(ns, "do I have any new emails?")
    check("email/notify/run_python query returns 'ollama_tools'",
          route == "ollama_tools", route)
    check("extra-tool branch logs its reason",
          "route:" in log and "keyword match" in log.lower(), log)


def test_deictic_hedge_reconfirm_routes_and_logs():
    ns = _build_namespace(ollama_up=True, deictic=True)
    hedge = ("I want to double-check that before I tell you it's done -- "
             "can you ask me again in a moment?")
    route, log = _call(ns, "Delete it.", prior_assistant_reply=hedge)
    check("a deictic follow-up to Chloe's own hedge routes to "
          "'ollama_tools' (the F-02/2026-09-06o fix)",
          route == "ollama_tools", route)
    check("the log line names it as a hedge re-confirm, not a generic "
          "email-keyword match",
          "hedge-reply re-confirm" in log, log)


def test_deictic_email_keyword_followup_routes_and_logs():
    ns = _build_namespace(ollama_up=True, deictic=True)
    route, log = _call(ns, "Send it.",
                        prior_assistant_reply="Do you want to send it?")
    check("a deictic follow-up to an email-flavored prior reply routes "
          "to 'ollama_tools' (the send-it fix)", route == "ollama_tools", route)
    check("the log line names it as a deictic email follow-up, not a "
          "hedge re-confirm", "email-flavored prior reply" in log, log)


def test_realtime_keyword_routes_to_local_search_and_logs():
    ns = _build_namespace(realtime=True)
    route, log = _call(ns, "who won the game last night?")
    check("a realtime-keyword match routes to 'local_search'",
          route == "local_search", route)
    check("the log line attributes it to the realtime-keyword match",
          "realtime-keyword match" in log, log)


def test_info_question_routes_to_local_search_and_logs():
    ns = _build_namespace(info_question=True)
    route, log = _call(ns, "how many moons does Jupiter have?")
    check("an info-question match (no realtime keyword) still routes to "
          "'local_search'", route == "local_search", route)
    check("the log line attributes it to looking like an info question",
          "looks like an info question" in log, log)


def test_self_knowledge_query_is_not_sent_to_search():
    # self_knowledge=True should suppress the realtime/info-question
    # branch even if those stubs would also fire, and fall through to
    # the ordinary ollama/local_chat branches instead.
    ns = _build_namespace(self_knowledge=True, realtime=True, warm=True,
                           ollama_up=True)
    route, log = _call(ns, "what have you learned about trading?")
    check("a self-knowledge query is never routed to 'local_search' even "
          "though it also matches a realtime keyword",
          route != "local_search", route)


def test_warming_up_short_circuits_and_logs():
    ns = _build_namespace(warm=False)
    route, log = _call(ns, "anything at all")
    check("while Ollama's boot warm-up is still in flight, route is "
          "'warming_up'", route == "warming_up", route)
    check("the log line names the warm-up guard", "warm" in log.lower(), log)


def test_default_ollama_available_routes_and_logs():
    ns = _build_namespace(ollama_up=True)
    route, log = _call(ns, "tell me a joke")
    check("an otherwise-unmatched turn with Ollama up routes to 'ollama'",
          route == "ollama", route)
    check("the log line names it the default route",
          "default" in log.lower(), log)


def test_default_ollama_unavailable_routes_and_logs():
    ns = _build_namespace(ollama_up=False)
    route, log = _call(ns, "tell me a joke")
    check("an otherwise-unmatched turn with Ollama down routes to "
          "'local_chat'", route == "local_chat", route)
    check("the log line names Ollama as unavailable",
          "unavailable" in log.lower(), log)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
