"""
wiki_dedup.py — Canonical-slug fingerprinting for Chloe's wiki.

Companion to wiki_embedding.py. This module provides the deterministic,
zero-Ollama-call layer of duplicate detection: two topic strings that are
the same words in a different order, number, or wrapped in a low-signal
prefix (research_wheel_strategy vs wheel_strategy) hash to the same
canonical key.

This is intentionally a STRICT, high-precision signal: it requires the
full normalized token SET to match exactly. It will catch true lexical
restructuring (word-order swaps, singular/plural, a stripped stopword
prefix) with no ambiguity and no risk of merging genuinely different
topics that merely share some words (see STOPWORDS docstring below for
the "finance_dividend_early_assignment" vs
"finance_dividend_impact_on_options" case this is deliberately built to
NOT collapse).

It will NOT catch every duplicate on its own — e.g. "trading_wheel"
canonicalizes to {wheel}, not {strategy, wheel}, so it won't set-match
"wheel_strategy" even though both are about the same real-world topic.
That class of duplicate (same topic, genuinely different words) is what
the semantic/embedding-cosine layer in wiki_embedding.WikiEmbeddingStore
is for. The two layers are complementary, not redundant — see
tools/wiki_dedup_report.py for how they're combined.

NOTE: as of 2026-08-31 this module only exposes canonicalization
(canonical_slug / STOPWORDS). The write-path hook (find_duplicate(),
append_dated_revision(), log_dedup_decision()) is deliberately NOT wired
into brain.py / brain_wiring.py / chloe_jobs.py yet — that's a follow-up
step pending review of tools/wiki_dedup_report.py's output.
"""

from __future__ import annotations

import os
import re

# Stopwords dropped from ANYWHERE in the token list (not just a leading
# position) before the canonical key is built. Two categories:
#
#   1. Generic connectors -- only matter if canonicalizing a raw title
#      rather than an already-underscore-joined slug (slugs from
#      _slugify_topic/Brain._slug are already mostly connector-free).
#   2. Corpus-specific low-signal prefixes Ed named explicitly after
#      seeing them cause real duplicate clusters: "research_wheel_strategy"
#      vs "wheel_strategy" only collapse if "research" is fully dropped,
#      not just de-prioritized. Same for "trading_"/"option(s)_".
#
# Deliberately NOT included: any word that is itself topic-disambiguating
# in this corpus. E.g. "dividend", "assignment", "impact", "early" all
# stay -- stripping any of those is what would risk merging
# finance_dividend_early_assignment into finance_dividend_impact_on_options
# (they'd still differ by {early, assignment} vs {impact, options} even
# after stripping stopwords below, which is the point: canonical_slug only
# merges pages whose FULL remaining token set is identical).
STOPWORDS = frozenset({
    # Generic connectors
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
    "with", "by", "from", "into", "as", "vs", "via",
    # Corpus-specific fluff prefixes (named explicitly: research_/trading_/
    # option_/options_). "options" and "option" collapse to the same
    # stopword since finance topics here are options-trading-flavored
    # almost universally -- this is the most aggressive entry and worth
    # double-checking against the report's canonical-key clusters.
    "research", "trading", "option", "options",
})

# Very small heuristic singularizer -- no inflect/nltk in the venv, and a
# full NLP dependency is overkill for slug tokens. Good enough for the
# finance/design/macro/psychology vocabulary in this corpus; not a general
# English singularizer (e.g. irregular plurals like "analyses" -> "analysis"
# aren't handled -- rare enough in these slugs to accept the miss).
def _singularize(tok: str) -> str:
    if len(tok) <= 3:
        return tok  # too short to safely strip (e.g. "vs", "ies" edge cases)
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def canonical_slug(text: str) -> str:
    """Normalize `text` (a slug or a free-form title) into a canonical
    fingerprint: lowercase, tokenize, drop STOPWORDS (anywhere in the
    string), singularize each remaining token, sort, join with '_'.

    Used as an equality key ONLY -- two topics collapse when their full
    canonical_slug() output matches exactly. This is NOT meant to be used
    as the literal filename for a new page (that stays human-readable via
    the existing slugify functions); it's purely a lookup fingerprint.

    Returns '' if nothing survives (caller should treat that as
    "can't fingerprint this one, skip the canonical-key check").
    """
    if not text:
        return ""
    lowered = text.lower().replace("-", "_")
    tokens = _TOKEN_RE.findall(lowered)
    kept = [_singularize(t) for t in tokens if t not in STOPWORDS]
    kept = [t for t in kept if t]
    if not kept:
        return ""
    # De-duplicate repeated tokens (set, not list) so a repeated word
    # doesn't create a spurious mismatch against a single-occurrence variant.
    return "_".join(sorted(set(kept)))


# ─── Subject-token extraction (2026-09-01) ───────────────────────────────────
# Factored here (was jarvis.py-only, built for _augment_search_query's
# short-query augmentation heuristic) so both jarvis.py and this module's
# own point-in-time supersede check share ONE list instead of two drifting
# copies -- the exact copy-drift problem this session already fixed once
# in this same wiki pipeline (dead-code-removal-adjacent lesson, not
# repeating it here). jarvis.py imports these back (`from wiki_dedup
# import has_no_subject`, etc.) rather than keeping its own copy.
#
# Deliberately NOT a general stopword list -- e.g. "silver"/"SLV"/
# "inflation"/"gold" are absent on purpose, since those ARE the subject in
# "what's the current price of SLV" or "gold vs SLV, which is up more".
# QUERY_FRAME_WORDS never identify a subject on their own ("what", "is",
# "the", "now"); QUERY_ASPECT_WORDS name the generic THING being measured,
# not what it's a measurement of ("price"/"rate" alone doesn't say price
# of what).
QUERY_FRAME_WORDS = frozenset({
    "what", "whats", "who", "whos", "when", "whens", "where", "wheres",
    "why", "how", "hows", "which", "is", "isnt", "are", "arent", "was",
    "wasnt", "were", "werent", "the", "a", "an", "of", "for", "on", "in",
    "at", "to", "now", "today", "currently", "current", "right",
    "moment", "this", "thats",
})
QUERY_ASPECT_WORDS = frozenset({
    "price", "prices", "priced", "pricing", "rate", "rates", "level",
    "levels", "value", "values", "cost", "costs", "worth", "number",
    "figure", "total", "amount", "high", "low",
})
_QUERY_WORD_RE = re.compile(r"[a-zA-Z']+")


def _normalize_query_word(w: str) -> str:
    """Blanket apostrophe-stripping ("what's" -> "whats") is right for
    CONTRACTIONS -- it's how they match QUERY_FRAME_WORDS's pre-stripped
    entries -- but wrong for POSSESSIVES on a real subject word: "SLV's"
    blanket-stripped becomes "slvs", which matches nothing (not a frame/
    aspect word, and not equal to "slv" either) -- confirmed live,
    2026-09-03: this silently broke same_subject("what's SLV's price",
    <title containing "slv">) for every query phrased with a possessive,
    which is why wiki_embedding's near-duplicate-quote collapse (added
    the same day) initially did nothing on real "what's SLV's price"-
    style queries. Fix: strip apostrophes for frame/aspect-word matching
    first; only if THAT doesn't resolve to a known frame/aspect word,
    treat a trailing "'s" as a possessive marker and drop it (stem
    alone), so "SLV's" -> "slv", matching a title's plain "slv" token."""
    no_apos = w.replace("'", "")
    if no_apos in QUERY_FRAME_WORDS or no_apos in QUERY_ASPECT_WORDS:
        return no_apos
    if w.endswith("'s"):
        return w[:-2]
    return no_apos


def extract_subject_tokens(text: str) -> frozenset:
    """Return the set of tokens in `text` that remain after stripping
    question-frame words (what/is/the/current/now/...) and generic aspect
    words (price/rate/level/...) -- the tokens that actually identify a
    SPECIFIC subject, as opposed to generic question-asking scaffolding.
    Empty set means `text` has no subject of its own ("what is the
    current price now"). See _normalize_query_word for how contractions
    ("what's" -> "whats") vs possessives ("SLV's" -> "slv") are told
    apart during matching."""
    words = [_normalize_query_word(w)
            for w in _QUERY_WORD_RE.findall((text or "").lower())]
    return frozenset(w for w in words
                     if w not in QUERY_FRAME_WORDS and w not in QUERY_ASPECT_WORDS)


def has_no_subject(text: str) -> bool:
    """True if `text` has no extractable subject (see extract_subject_tokens)."""
    return len(extract_subject_tokens(text)) == 0


def same_subject(text_a: str, text_b: str) -> bool:
    """True if `text_a` and `text_b` share at least one subject token.
    False if EITHER side has no extractable subject at all -- a
    conservative default: no subject signal on either side should never
    default to 'treat as a match', since the specific failure mode this
    guards (point-in-time supersession) is a wrong page silently
    overwriting a right one. "gold" vs "SLV"/"silver" share nothing ->
    False. "SLV" vs "SLV" (or "silver") -> True."""
    tokens_a = extract_subject_tokens(text_a)
    tokens_b = extract_subject_tokens(text_b)
    if not tokens_a or not tokens_b:
        return False
    return bool(tokens_a & tokens_b)


def find_duplicate(query_text, store, *, scoped_dirs=None, threshold=None):
    """Two-layer duplicate check against the existing wiki corpus.

    1. Canonical-key exact match (cheap, deterministic, zero Ollama calls) —
       catches word-order/plural/prefix-stopword variants of the same topic.
    2. Cosine similarity via the embedding store (semantic) — catches
       same-topic pages worded differently enough that canonical_slug()
       can't see it (see the module docstring's "trading_wheel" example).
       Canonical match wins if both fire.

    `store` is a wiki_embedding.WikiEmbeddingStore instance.
    `scoped_dirs`: restrict candidates to these top-level wiki/ subdirs
    (e.g. ("sources",)); None = no restriction.
    `threshold`: RAW-cosine floor for the semantic layer; defaults to
    CHLOE_WIKI_DEDUP_THRESHOLD (0.85) if not given. Callers that need a
    stricter floor (e.g. supersede_prior_point_in_time_page's 0.93) pass
    it explicitly rather than relying on an env var override, so two
    different call sites can run two different thresholds in the same
    process.

    Both match types additionally require same_subject(query_text,
    candidate title) (2026-09-01, Ed: "three call sites implementing
    'duplicate' differently is its own maintenance risk" — this used to
    be bolted on separately by supersede_prior_point_in_time_page; now
    every caller gets it for free). Gate is on RAW cosine, never the
    boosted score store.search() uses internally to shortlist and rank
    candidates: WikiEmbeddingStore's path-boost rewards shared filename-
    scaffolding tokens (date stems, generic aspect words) between pages
    that can be about entirely unrelated subjects — confirmed on real
    point-in-time quote pages (a gold page and an SLV page scored 0.73-
    0.79 raw cosine while clearing an 0.85 BOOSTED-score gate) and
    structurally identical here for concept/entity dedup, just lower-
    stakes since a false match there merges via append_dated_revision
    (annoying, recoverable) rather than supersede (destructive). Always
    calls store.search with threshold=0.0 so its own boosted-score
    ranking never pre-filters a candidate before this function gets to
    check raw cosine itself.

    same_subject() treats no-extractable-subject on either side as
    no-match (conservative default: a missed merge just leaves two pages
    coexisting; a false merge silently absorbs one page's content into
    an unrelated one).

    IMPORTANT: calls store.search(..., apply_staleness_gate=False) — a
    duplicate-finder must be able to see stale/superseded candidates too
    (that's often exactly what it's looking for, e.g. point-in-time
    supersession), so it never applies the ambient-recall staleness gate
    to its own candidate search.

    Returns {'path', 'title', 'match_type': 'canonical'|'cosine',
    'score', 'cosine', 'boosted_score'} for the best match, or None.
    'score' is kept for backward-compat callers and always equals the
    RAW cosine (1.0 for a canonical match). 'boosted_score' is the value
    that actually ranked the candidate in store.search() — None for a
    canonical match, since no embedding search occurs for an exact
    canonical-slug hit.
    """
    import os as _os
    from pathlib import Path as _Path

    thr = threshold if threshold is not None else float(
        _os.environ.get("CHLOE_WIKI_DEDUP_THRESHOLD", "0.85"))

    canon_key = canonical_slug(query_text)
    if canon_key:
        for p in store.list_pages():
            if scoped_dirs and p["path"].split("/")[0] not in scoped_dirs:
                continue
            if canonical_slug(_Path(p["path"]).stem) == canon_key:
                if not same_subject(query_text, p["title"]):
                    continue
                return {"path": p["path"], "title": p["title"],
                        "match_type": "canonical", "score": 1.0,
                        "cosine": 1.0, "boosted_score": None}

    hits = store.search(query_text, limit=5, threshold=0.0,
                        apply_staleness_gate=False)
    for h in hits:
        if scoped_dirs and h["path"].split("/")[0] not in scoped_dirs:
            continue
        if h["cosine"] < thr:
            continue
        if not same_subject(query_text, h["title"]):
            continue
        return {"path": h["path"], "title": h["title"],
                "match_type": "cosine", "score": h["cosine"],
                "cosine": h["cosine"], "boosted_score": h.get("score")}
    return None


def describe_match_score(match: "dict | None") -> str:
    """One-line score description for logs/status strings, showing the
    RAW cosine (what actually gated the match) and, when it differs from
    a plain 1:1 relationship, the BOOSTED score store.search() used to
    rank candidates (2026-09-01: a log line that only showed 'score' used
    to read as if the raw cosine were the gating value when it was
    sometimes up to 0.20 lower than what really gated inclusion -- see
    find_duplicate's docstring). Every match/log/status site that used
    to print `score={match['score']:.3f}` should call this instead, so
    all three write paths (Brain.ingest, chloe_jobs, wiki_dedup's own
    supersede) render duplicate-match scores identically."""
    if not match:
        return "no match"
    cosine = match.get("cosine", match.get("score"))
    boosted = match.get("boosted_score")
    if boosted is None:
        return f"cosine={cosine:.3f}"
    return f"cosine={cosine:.3f}, boosted={boosted:.3f}"


# ─── Point-in-time claim classification (2026-08-31) ────────────────────────
# "The current price of SLV is $60.32" with 5 real citations, none of them
# timestamped, reads as a durable sourced fact on later recall -- same
# failure class as fabricated citations, just with real URLs. Two ceilings,
# not one: a market quote is stale in hours-to-days; a scheduled-release
# figure (CPI, Fed funds rate) is legitimately "current" for weeks between
# releases. Single source of truth -- both jarvis.py's persist path and
# tools/backfill_point_in_time.py import this rather than each keeping
# their own copy of the classifier (the exact duplication problem the
# wheel_strategy wiki cluster came from).
QUOTE_STALENESS_DAYS = float(os.environ.get("CHLOE_QUOTE_STALENESS_DAYS", "3"))
DATA_STALENESS_DAYS = float(os.environ.get("CHLOE_DATA_STALENESS_DAYS", "30"))

_SCHEDULED_DATA_KEYWORDS_RE = re.compile(
    r"\b(?:cpi|consumer price index|inflation rate|fed funds rate|"
    r"federal funds rate|policy rate|interest rate|unemployment rate|"
    r"jobless rate|gdp|ppi|pce|nonfarm payroll|jobs report)\b",
    re.IGNORECASE,
)
_MARKET_QUOTE_KEYWORDS_RE = re.compile(
    r"\b(?:stock|share price|ticker|trading at|closed at|intraday|"
    r"after[- ]?market|pre[- ]?market|spot price)\b",
    re.IGNORECASE,
)
_DOLLAR_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_BPS_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:bps|basis points)\b", re.IGNORECASE)

POINT_IN_TIME_LABELS = {
    "quote": "market quote",
    "data": "official statistic",
}


def classify_point_in_time(text: str) -> str | None:
    """Return 'quote' (market price/ticker -- QUOTE_STALENESS_DAYS ceiling),
    'data' (scheduled-release official figure -- DATA_STALENESS_DAYS
    ceiling), or None (no quantity claim detected, not point-in-time).

    Scheduled-data keywords (CPI, Fed funds rate, ...) win outright when
    present -- these move on a known calendar, not by the minute. A dollar
    amount or market-quote phrasing with no scheduled-data keyword is a
    'quote'. A bare percentage/bps with neither signal is genuinely
    ambiguous -- default to 'quote', the shorter ceiling: a false 'quote'
    classification on borderline text just means it expires sooner than
    strictly necessary, a cheap mistake compared to the reverse.

    Known gap: only matches the literal phrase "federal funds rate" /
    "fed funds rate" -- a natural variant like "the Federal Reserve's
    funds rate" won't match the scheduled-data keyword set and falls
    through to 'quote' instead of 'data'. Same safe-default behavior
    (shorter ceiling on a miss), just not the ideal classification.

    Does NOT catch outcome/framing claims with no quantity at all (e.g.
    "the outlook for silver remains bullish" or "who won Eurovision") --
    those decay for a different reason (stale framing, not a stale
    number) and aren't handled by this function."""
    if not text:
        return None
    if _SCHEDULED_DATA_KEYWORDS_RE.search(text):
        return "data"
    if _DOLLAR_AMOUNT_RE.search(text) or _MARKET_QUOTE_KEYWORDS_RE.search(text):
        return "quote"
    if _PERCENT_RE.search(text) or _BPS_RE.search(text):
        return "quote"
    return None


def build_point_in_time_metadata(text: str, ts: str) -> dict:
    """Classify `text` and build the frontmatter lines + body marker to
    attach if it contains point-in-time content. Shared by jarvis.py's
    _persist_brave_to_wiki (voice/chat search pages) and brain.py's
    ingest() -> _render_source_page (every ingested source, including
    /wiki_write) so both paths mark point-in-time content identically --
    factored out 2026-08-31 so there's one implementation instead of two
    maintained in parallel (the gap Ed flagged: /wiki_write pages asserted
    a spot price with no point_in_time frontmatter, no as-of marker, and
    no staleness ceiling, aging into ambient recall as durable fact).

    Returns {"kind": str|None, "frontmatter": str, "marker": str,
    "ceiling_days": float|None}. `frontmatter` and `marker` are both ""
    when `kind` is None (nothing point-in-time detected) -- always safe
    to splice into a page unconditionally.
    """
    kind = classify_point_in_time(text)
    if not kind:
        return {"kind": None, "frontmatter": "", "marker": "", "ceiling_days": None}
    frontmatter = (
        f"point_in_time: true\n"
        f"point_in_time_kind: {kind}\n"
    )
    kind_label = POINT_IN_TIME_LABELS.get(kind, kind)
    ceiling = QUOTE_STALENESS_DAYS if kind == "quote" else DATA_STALENESS_DAYS
    marker = (
        f"> ⚠ **Point-in-time data ({kind_label}) — valid as of "
        f"{ts}.** Do not treat as current without re-checking. "
        f"(Ambient recall drops this page after "
        f"{ceiling:.0f} day(s).)\n\n"
    )
    return {"kind": kind, "frontmatter": frontmatter, "marker": marker,
            "ceiling_days": ceiling}


def supersede_prior_point_in_time_page(brain, query_text: str, new_slug: str,
                                        *, scoped_dirs=("sources",)) -> None:
    """Find a prior point-in-time page similar to `query_text` and mark it
    superseded, so the newest point-in-time page on a subject replaces
    the previous one instead of coexisting with it -- a revision history
    of SLV prices in ambient recall helps nobody, and two pages a few
    days apart both clearing the staleness ceiling would still
    contradict each other.

    Reserved for pages whose ENTIRE purpose is being a point-in-time
    answer (voice/chat search results via _persist_brave_to_wiki) -- NOT
    called for general ingested sources (/wiki_write, /ingest,
    /ingest_screen), which are typically mixed durable+time-sensitive
    content (e.g. a /wiki_write concept page has background/mechanics
    sections alongside one current-price paragraph); superseding the
    WHOLE page over one aged number would incorrectly flag good, durable
    content as stale too. Those pages still get labeled via
    build_point_in_time_metadata, just never superseded here (Ed,
    2026-08-31: "a concept page about silver that happens to contain a
    current price isn't a pure quote page").

    Gate is CHLOE_PIT_SUPERSEDE_THRESHOLD (default 0.93) checked against
    RAW cosine, AND same_subject(query_text, candidate title) -- both
    deliberately stricter than find_duplicate's own default gate
    (CHLOE_WIKI_DEDUP_THRESHOLD / 0.85), because a false supersede here
    is destructive (overwrites a still-correct page) where a false
    concept-dedup merge is merely an annoying append. Both the raw-
    cosine gate and the same_subject check now live inside
    find_duplicate() itself (2026-09-01 -- previously bolted on here
    separately, which is what Ed flagged as "three call sites
    implementing 'duplicate' differently"); this function only supplies
    the stricter threshold override.

    `brain` is duck-typed: needs .read(rel) and .write(rel, text)
    (Brain.read/Brain.write's actual signature). Best-effort: any
    failure is logged and swallowed, never blocks the caller's own
    write.
    """
    try:
        import os as _os
        import wiki_embedding as _wiki_embedding
        _store = _wiki_embedding.get_store()
        _pit_threshold = float(
            _os.environ.get("CHLOE_PIT_SUPERSEDE_THRESHOLD", "0.93"))
        _match = find_duplicate(query_text, _store, scoped_dirs=scoped_dirs,
                                threshold=_pit_threshold)
        if not _match:
            return
        _cand_rel = f"wiki/{_match['path']}"
        try:
            _cand_text = brain.read(_cand_rel)
        except Exception:
            _cand_text = ""
        _cand_is_pit = bool(re.search(
            r"^point_in_time_kind:\s*\S+", _cand_text, re.MULTILINE))
        _cand_already_superseded = bool(re.search(
            r"^superseded:\s*true\b", _cand_text, re.MULTILINE | re.IGNORECASE))
        if _cand_is_pit and not _cand_already_superseded:
            _fm_m = re.match(r"^(---\n)(.*?)(\n---\n?)(.*)$", _cand_text, re.DOTALL)
            if _fm_m:
                _new_cand_text = (
                    _fm_m.group(1) + _fm_m.group(2)
                    + "\nsuperseded: true"
                    + _fm_m.group(3) + _fm_m.group(4)
                )
                brain.write(_cand_rel, _new_cand_text)
                print(f"[wiki_dedup] superseded {_match['path']} "
                      f"({_match['match_type']}, {describe_match_score(_match)}) "
                      f"-> {new_slug}", flush=True)
    except Exception as e:
        print(f"[wiki_dedup] supersede check failed (non-fatal, writing "
              f"new page anyway): {e}", flush=True)


# ─── Concept-page merge rule (2026-08-31) ────────────────────────────────────
# Distinct from supersede_prior_point_in_time_page above: that's the
# quote-page rule (a voice/chat search page's ENTIRE purpose is being a
# point-in-time answer, so the newest one replaces the previous one wholesale).
# A concept/entity page is durable reference material that different source
# documents touch repeatedly over time, often extracting the same real-world
# topic under slightly different LLM-chosen names ("wheel_strategy" vs
# "trading_wheel", "silver" vs "silver market") -- find_duplicate() is what
# catches that. The merge rule here is APPEND, not replace or supersede: each
# new pass adds a dated revision section on top of what's already there, so a
# reader (or an embedder building recall context) always sees the accumulated
# history, not just whichever pass happened to run most recently.
_FRONTMATTER_RE = re.compile(r'^(---\n)(.*?)(\n---\n?)(.*)$', re.DOTALL)
_UPDATED_FIELD_RE = re.compile(r'^updated:\s*\S+', re.MULTILINE)


def append_dated_revision(existing_text: str, new_body: str, *,
                          date: str, source_label: str) -> str:
    """Merge `new_body` into `existing_text` as a dated revision section,
    rather than overwriting or superseding it. The ORIGINAL content stays
    intact and first; `new_body` (itself a complete page -- frontmatter +
    prose, as _update_page returns) gets its own frontmatter stripped and
    its prose appended under a `## Revision — {date} (via {source_label})`
    heading. Bumps the existing page's `updated:` frontmatter field to
    `date` if that field is present; leaves everything else in the
    original frontmatter untouched (title, tags, sources -- sources is
    expected to be rewritten separately by the caller via
    _rewrite_sources_field, same as any other ingest path).

    Never raises: if `existing_text` doesn't parse as well-formed
    frontmatter, the revision is still appended (just without the
    'updated:' bump), so a malformed existing page can't block a merge
    that would otherwise fix the sprawl.
    """
    new_body_stripped = new_body.strip()
    m = _FRONTMATTER_RE.match(new_body_stripped)
    new_prose = m.group(4).strip() if m else new_body_stripped

    revision_section = (
        f"\n\n## Revision — {date} (via {source_label})\n\n{new_prose}\n"
    )

    fm_m = _FRONTMATTER_RE.match(existing_text.strip())
    if not fm_m:
        return existing_text.rstrip() + revision_section

    opener, fm_body, closer, body = fm_m.groups()
    if _UPDATED_FIELD_RE.search(fm_body):
        fm_body = _UPDATED_FIELD_RE.sub(f'updated: {date}', fm_body, count=1)
    else:
        fm_body = fm_body.rstrip() + f'\nupdated: {date}'
    return opener + fm_body + closer + body.rstrip('\n') + revision_section


def log_dedup_decision(action: str, query_text: str, match: "dict | None",
                       *, target_path: str = "", caller: str = "") -> None:
    """Append one line to logs/wiki_dedup_decisions.log recording what
    happened, so Ed can see what merged into what across every write path
    (voice/chat supersede, /wiki_write, /wiki_synth, Brain.ingest, the
    daily jobs) without digging through each module's own logs.

    `action`: 'appended' (merged into an existing page) | 'new_page' (no
    duplicate found, wrote fresh) | 'superseded' (quote-page replace).
    `match` is find_duplicate()'s return dict, or None for 'new_page'.
    Best-effort: a logging failure here never blocks the caller's actual
    write -- this is a review aid, not a control-flow dependency.
    """
    try:
        import datetime as _dt
        from pathlib import Path as _Path
        log_dir = _Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        match_desc = (
            f"{match['match_type']}:{match['path']} ({describe_match_score(match)})"
            if match else "-"
        )
        line = (f"{ts}\t{caller or '?'}\t{action}\t"
               f"query={query_text[:80]!r}\tmatch={match_desc}\t"
               f"target={target_path or '-'}\n")
        with open(log_dir / "wiki_dedup_decisions.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[wiki_dedup] log_dedup_decision failed (non-fatal): {e}",
              flush=True)
