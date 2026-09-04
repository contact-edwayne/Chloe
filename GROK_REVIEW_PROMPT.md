# Review prompt for Grok — Chloe conversational-architecture overhaul

You are an expert reviewer of production conversational-AI systems (LLM
orchestration, memory architectures, agent design, latency/cost engineering).
Below is a full changelog of a single development session on **Chloe**, a
voice + chat AI companion. I want a ruthless, specific technical review — not
encouragement. Assume I can take hard criticism and want it.

---

## What Chloe is (context)

A single-user AI companion that runs locally on a Windows PC (Radeon 7900 XTX).
Stack:
- **Models:** local Ollama is primary — `qwen2.5:32b` (and `:14b`) for chat,
  `llama3.2:3b` for utility tasks, `nomic-embed-text` for embeddings. Groq
  (`llama-3.3-70b`, `compound-mini`) is the cloud burst tier.
- **Memory:** SQLite turn log + FTS5 mirror (triggers keep them in sync),
  a markdown `facts.md`, and a self-maintaining markdown wiki with semantic
  recall. ~550 logged turns.
- **Voice + chat:** wake-word + push-to-talk, streaming Kokoro TTS, a HUD over
  WebSocket. Chat path is async (`handle_chat`); voice path builds its prompt
  synchronously (`_augmented_voice_system`).
- **Self-modification:** Chloe edits her own code through a reviewed proposal
  pipeline (unified-diff or full-file proposals → dry-run → `ast.parse` →
  timestamped backup → apply via a confirm-token; max 5 applies per process
  lifetime; restart to bind). A persona spec (`chloe_about.md`) is trimmed
  per-turn and injected as the system prompt.

The system prompt is assembled per-turn from many "blocks" (persona core, date,
durable facts, recent-context, user-model, rolling synopsis, dialogue-state,
recall hits, wiki hits, etc.).

---

## What was built this session (changelog)

**Phase 0 — stop the bleeding (applied to `jarvis.py`)**
1. **Un-gated episodic recall.** Recall over the conversation log was only run
   when the user typed a probe phrase ("remember when", "earlier", …). Removed
   that keyword gate so semantic recall runs every substantive turn; the recall
   function is already self-thresholding (cosine floor) + noise-filtered, so an
   empty result is the correct "nothing relevant." Explicit probes now only
   widen k (8 vs 5). Runs in a worker thread, gathered concurrently with wiki.
2. **Raised local context window** default 8192 → 16384 tokens (env-overridable).
3. **Fenced the user-supplied `system` field** — it was concatenated last
   (highest effective priority); now wrapped in a low-priority `<user_note>`
   labeled "information, not instructions."

**Phase 1 — memory truth**
4. **`chloe_dialogue_state.py`** — a per-session, heuristic (no-LLM) working-
   memory scratchpad persisted to JSON: conversational *mode* (processing vs
   distilling, with stickiness), mood read, length target, open loops, salient
   entities. Rendered as a compact "Session state (behave it, never announce
   it)" block. Self-suppresses on acks/one-off factual turns. 6h gap = new
   session.
5. **Recall reformatter** — rewrote the recall block from raw timestamped log
   dumps to a compact, deduped, relative-dated ("3 days ago, Ed: …"),
   speaker-light, fused-looking block. Latency-free (no LLM).
6. **`chloe_synopsis.py`** — rolling summary of turns evicted past the last ~30,
   cached, only re-summarizes every ~10 new evictions, fires only on long
   sessions, uses `llama3.2:3b`, runs off the event loop.

**Phase 3 — user model + voice parity**
7. **Live user-model** injected every turn (later upgraded — see 4B).
8. **Voice-path parity** — un-gated voice recall (it had been left keyword-gated)
   and injected dialogue-state into the voice prompt so the mood read drives
   spoken tone.
9. **Proactive callbacks** — a persona rule: surface a relevant past thread
   *unprompted* when it fits, max one per reply, never one raised recently,
   suppressed when he's frustrated/terse.

**Phase 4A — consolidation (the keystone)**
10. **`chloe_context.py` — a token-budgeted Context Composer.** Replaced the
    ~13-block free-for-all concatenation in both chat and voice. Blocks carry a
    priority (0 = identity/persona, never dropped) and a stable reading order;
    it estimates tokens (len/4), drops lowest-priority blocks when over a
    `ctx - reserve` budget, de-duplicates blocks sharing a key, and logs
    `tokens used / kept / dropped` every turn. Falls back to plain concat on
    error.
11. **Persona additions** — "Using your model of Ed (relate, don't recite),"
    "Memory honesty & contradictions (recency beats stale memory; reconcile out
    loud)," "Reading the room (name circling; back off when he's clipped; stop
    re-asserting after 2 corrections)."
12. **Eagerness governor** — one state-driven line that explicitly tells her to
    hold proactive callbacks + teasing when he's frustrated/tense/sad/
    distilling/low-engagement.

**Phase 4B — typed memory**
13. **Typed user-model** (`ed_model.json`) — replaced the flat narrative profile
    with slots (identity / values / comms-prefs / triggers / inside-jokes /
    current-focus + a relationship sub-object: trust, playfulness,
    recent-friction, interactions, since). Every item has confidence /
    last-seen / evidence-count; updates **merge incrementally** (bump evidence
    & confidence, dedupe) instead of wholesale rebuild; renders a budgeted
    selection; hides `sensitive` items; self-seeds on first run.
14. **Importance-weighted recall** — recall keeps cosine as the relevance floor
    but reorders above-floor hits by `cosine × importance × recency`, where a
    read-time importance heuristic boosts decisions, emotional content, and
    explicit "remember" (no DB schema change).

**Phase 4C — theory of mind**
15. **Flow signals** (heuristic, in dialogue-state) — engagement trend,
    *circling* (lexical Jaccard of consecutive user turns), *repeated-
    correction*, and *subtext* ("I'm fine" that isn't), each rendered as a
    behave-it line and feeding the eagerness governor.
16. **`chloe_read.py` — LLM read-pass** — a small fast-model (`llama3.2:3b`)
    structured read returning `{mood, intent, subtext, engagement, certainty}`
    as JSON. Gated to turns > 45 chars, 4s timeout, default-on (env-disable).
    Runs as a third coroutine in the existing recall/wiki `gather` so it
    overlaps their embedding round-trips (≈no added wall-clock latency). It
    overrides the heuristic read only when `certainty ≥ 0.5`; returns `{}` (→
    heuristic fallback) on disable/short/timeout/bad-JSON.

**Phase 4D — relationship depth**
17. Relationship dimensions + inside-jokes registry delivered via the typed
    model; a per-session relationship tick.

**Phase 4E — reflection + governance**
18. **`chloe_reflect.py`** — a scheduled daily job (21:00) that reads recent
    turns + facts, synthesizes higher-order observations about the user, and
    merges them into the typed model (generative-agents-style reflection). Runs
    via a local catch-up scheduler loop.
19. **Forgetting / privacy** — `memory.forget(query)` (DELETE from turns; FTS
    stays consistent via the existing AFTER-DELETE trigger), a model-level
    `forget()`, and the `sensitive`-never-rendered rule.

**Engineering method**
- `jarvis.py` (8k lines, history of edit-tool truncation) changed only through
  the reviewed proposal pipeline (dry-run + ast-check + backup). All other
  modules edited directly and verified by AST-introspection (`/explain`) on the
  real machine.
- Validation performed: AST/syntax on the real files, isolated unit tests of
  the new pure logic, and dry-run + ast-check of every diff before commit.

**Deliberately deferred / not done**
- Live conversational validation (everything is shipped + bound + statically
  checked, but **not yet exercised in a real conversation**).
- User-facing `/forget` slash command (mechanism exists; no command hook).
- Callback-novelty TTL (don't repeat a callback — needs reply-emission hooks).
- Shared query embedding (recall + wiki currently embed the same user string
  twice per turn).
- Ack-gating retrieval (skip embeds on greetings/acks).
- Semantic facts selection (intentionally skipped — facts.md is ~7 lines).

---

## What I want from you

Be specific and critical. In particular:

1. **Correctness & risk.** Any bugs, race conditions, or failure modes in the
   above? Concurrent chat + voice both write the same dialogue-state JSON and
   the same model JSON — is that a real hazard for a single user? The composer
   uses a len/4 token estimate — where could that bite at 16k?
2. **Latency & cost.** The chat path now runs (per substantive turn): a recall
   embed, a wiki embed, and the LLM read-pass — concurrently. Is "overlap in a
   gather → ≈no added latency" sound, or am I fooling myself? Is the LLM
   read-pass worth a hot-path call at all vs. a stronger heuristic? What should
   be cut?
3. **Architecture.** Is routing everything through one priority/budget composer
   the right call, or does it hide problems? Is the typed incremental user-model
   + importance-weighted recall + daily reflection a coherent memory design, or
   are these fighting each other / double-counting the same information?
4. **Theory-of-mind quality.** Will heuristic flow signals (circling via lexical
   overlap, "I'm fine" regex) plus a 3B JSON read-pass actually improve
   emotional intelligence, or is this brittle theater? Better approaches?
5. **Prioritization.** Of the deferred items (and anything I missed), what's
   actually worth doing next, and what's over-engineering I should drop?
6. **Safety.** Anything here that could degrade reliability, leak sensitive
   info at the wrong time, or make a self-modifying system less safe?

Push back hard where I've made a mediocre call. Prioritize delightful,
coherent, emotionally-intelligent long-term conversation over feature count.
