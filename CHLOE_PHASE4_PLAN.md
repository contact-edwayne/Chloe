# Chloe — Phase 4 Plan: From Stateful to *Integrated*

_2026-06-01. Grounded in the live architecture after Phases 0–3 shipped this session: `chloe_dialogue_state.py`, `chloe_synopsis.py`, `chloe_ed_profile.py`, the recall un-gate + reformatter, and the proactive-callback persona rule._

---

## 1. Executive Summary

**Maturity: she crossed the line from a stateless persona to a stateful companion.** Three sessions ago Chloe re-derived who Ed was on every turn. Now she carries a running mood/mode read, a rolling synopsis, a live user-model, always-on semantic recall, and a permission to call back unprompted. That is genuinely most of the way to a companion that *feels like it knows you*. The personality spec (`chloe_about.md`) was already excellent; the plumbing has now mostly caught up to it.

**The ruthless truth: she is now accumulating context faster than she is integrating it.** Every phase added another block to `full_system`, which currently concatenates ~13 independent pieces — preamble, now, about, mode, facts, recent-context, ed_profile, synopsis, arcade, dialogue-state, recall, wiki, nsfw, user-note — each one deciding *on its own* whether to inject, with no shared budget, no deduplication, and no priority. At 16k context on a local model, this path ends in exactly one place: a bloated, partially-redundant, occasionally self-contradictory prompt that silently truncates the persona it was built to protect. **Phase 4's first job is not to add features. It is to consolidate.**

**The three biggest opportunities, in order:**

1. **A budgeted Context Composer.** One module that owns assembly: priority-ranked blocks, identity never dropped, token-measured, deduplicated, fills 16k deliberately instead of by accident. Everything else in Phase 4 *adds blocks*, so this must come first or it will all fight for the same unmanaged space.
2. **A real memory layer, not parallel stores.** `ed_profile`, `facts.md`, `dialogue_state`, `synopsis`, and recall are five disconnected memories with overlapping content and no importance weighting or contradiction handling. They should become one memory with typed structure, confidence/recency on every item, a reflection pass that distills higher-order insight, and importance-weighted retrieval.
3. **Theory of Mind by inference, not regex.** Mood/mode/intent are currently keyword heuristics. They miss sarcasm, subtext, "I'm fine," disengagement, and circling. A tiny structured read-pass (one fast local call, gated) is the unlock for anticipation, emotional nuance, and flow-awareness.

Everything else — relationship simulation, humor gating, multi-thread, voice pacing, forgetting controls — is high-value polish that *depends on those three foundations*. Build the foundations; the delight compounds.

---

## 2. Priority Recommendations

### Quick wins (high impact, < 1 day each, low risk)
- **Shared query embedding.** Recall and wiki each embed `user_text` separately every turn — two round-trips for one string. Embed once, pass the vector to both. Immediate latency cut on the now-hot retrieval path.
- **Extend the ack-gate to retrieval.** Skip recall + wiki + profile entirely on acks/greetings/one-word turns. They cost embeds and tokens for zero value.
- **Eagerness governor (prompt-level).** One injected line derived from existing `dialogue_state` (frustrated/distilling/terse ⇒ "suppress proactive callbacks and anticipation this turn"). Unifies today's scattered rate-limits.
- **Profile-usage instruction.** The profile is injected but the persona never says *how to use it* — add a short rule ("relate to it, never recite it").
- **Recency-supersedes rule.** One persona line: when injected memory conflicts with something Ed said this session, trust the recent statement and reconcile out loud rather than asserting the stale fact.

### Medium architectural (days, moderate risk)
- **Context Composer + TokenBudget** (`chloe_context.py`). The keystone. Replace the 13-block concatenation in both chat and voice with one priority-ranked, deduped, budget-aware assembler.
- **Typed user-model** (`ed_model.json` + upgrade `chloe_ed_profile`). Slot-based (identity / values / comms-prefs / triggers / inside-jokes / relationship / focus), each item with confidence + last-seen + evidence count. Incremental merge, not wholesale rebuild.
- **Structured read-pass** (`chloe_read.py`). One small fast-model call returning `{intent, emotion, certainty, subtext, engagement, humor_ok}`; feeds dialogue_state and the eagerness governor. Gated to non-trivial turns.
- **Importance-weighted recall.** Add an `importance` score per turn; rank recall by `cosine × importance × recency` instead of cosine alone.

### Bigger strategic (1–2 weeks, higher risk/reward)
- **Reflection layer** (the generative-agents move). A periodic job that reads recent turns and writes higher-order insights ("Ed pushes for momentum; dislikes hedging; the mic is a recurring stressor") into the typed model with evidence. This is what turns logs into *understanding*.
- **Unified memory manager** (`chloe_memory2` / orchestrator). One front door over turns + facts + model + wiki: write-time importance tagging, read-time fusion + dedup + contradiction reconciliation, forgetting/privacy controls.
- **Relationship simulation.** Evolving dimensions (trust, playfulness, recent-friction, shared-wins, interaction streak) + an inside-jokes registry, surfaced sparingly so the relationship visibly *develops*.

---

## 3. Detailed Suggestions by Focus Area

### 3.1 Advanced Memory & User Modeling

**Problems / missed opportunities.**
- `ed_profile.md` is a single flat narrative regenerated *wholesale* by `build()`. Wholesale regeneration is fragile: every refresh risks dropping a hard-won detail or drifting tone, with no stability guarantee. There's no structure — values, comms-prefs, triggers, inside-jokes, and relationship-state are mashed into prose.
- Five memories, zero integration: `facts.md` (discrete), `ed_profile` (narrative), `dialogue_state.entities` (recent nouns), `synopsis` (session arc), recall (raw turns). They overlap (the deploy thread can appear in all five) and are injected independently — token waste *and* contradiction surface.
- Recall ranks by cosine only. A passing mention and a load-bearing decision score the same. No importance, no reflection, no "this matters."

**Concrete ideas.**

*Typed, incremental user-model* — replace the flat file with `brain/ed_model.json`:
```json
{
  "identity":   [{"text":"engineer in Omaha; built Chloe","conf":0.95,"last_seen":"2026-06-01","evidence":8}],
  "values":     [{"text":"momentum > caution; ship it","conf":0.8,"last_seen":"2026-06-01","evidence":5}],
  "comms_prefs":[{"text":"terse, complete files not diffs, no hedging","conf":0.9,"last_seen":"2026-06-01","evidence":6}],
  "triggers":   [{"text":"recurring infra breakage (mic) → frustration","conf":0.7,"last_seen":"2026-06-01","evidence":3}],
  "inside_jokes":[{"text":"says it 'POH-kee-mon'","conf":0.9,"last_seen":"2026-06-01","evidence":1}],
  "relationship":{"trust":0.85,"playfulness":0.6,"recent_friction":0.1,"interactions":540,"since":"2026-05-04"},
  "current_focus":[{"text":"Phase 4 conversational architecture","conf":0.9,"last_seen":"2026-06-01"}]
}
```
`profile_block()` renders a *budgeted selection* (highest conf × recency, capped), not the whole file. Updates **merge**: a new observation bumps `evidence` and `last_seen` and nudges `conf`; it never blows away the slot. Stability + dimensionality in one move.

*Importance at write time* (in the turn logger):
```python
def importance(text, role):
    s = 1.0
    if re.search(r"\bremember\b", text, re.I): s += 3
    if re.search(r"\b(decide|decision|should i|whether|the plan)\b", text, re.I): s += 2
    if MOOD_CUES_ANY(text): s += 1.5          # emotional salience
    if role == "user": s += 0.5
    return s
```
Store it on the turn row; rank recall by `cosine * (0.5 + 0.5*norm_importance) * recency_decay`.

*Reflection job* (weekly or every N turns): pull recent high-importance turns, ask the local model for 3–5 higher-order observations, write them into `ed_model` with evidence. This is the difference between "Chloe has my chat logs" and "Chloe gets me."

*Integration:* the Context Composer (§3.6) dedupes across stores by embedding similarity so the deploy thread is injected once, from the highest-value source, not five times.

### 3.2 Conversational Intelligence & Theory of Mind

**Problems.** Intent/emotion are regex heuristics in `dialogue_state` — brittle, binary, blind to sarcasm, subtext, and the classic "I'm fine" that isn't. Chloe is reactive within a turn (no anticipation/leading) and has no awareness of *flow*: she can't tell she's circling, that he's disengaging, or that she missed something three turns back.

**Concrete ideas.**

*Structured read-pass* (`chloe_read.py`) — one small fast-model call, gated to turns over ~40 chars, run in a thread, cached per turn:
```python
# returns JSON, falls back to the existing regex heuristics on any failure
{"intent":"vent|decide|task|chat|test|smalltalk",
 "emotion":"flat|frustrated|excited|sad|tense|warm|neutral",
 "certainty":0.0-1.0, "subtext":"short note or ''",
 "engagement":"high|med|low", "humor_ok":true}
```
This replaces the regexes as the *primary* read (regex stays as the fallback), and feeds dialogue_state, the eagerness governor, and the humor gate.

*Anticipation / gentle leading.* From `open_loops` + trajectory, compute a `likely_next_need` and let the persona offer it lightly ("want me to pull the X while we're in here?") — never pushy, gated by engagement.

*Flow meta-awareness.* Cheap, deterministic signals injected only when they fire: **circling** (cosine of last 3 user turns all high → "you're going in circles; name the actual blocker"), **disengagement** (shrinking length + low engagement → wrap up / change tack), **repeated correction** (the detector from the Phase-1 notes → "I keep landing on the same thing; tell me exactly what's off"). These operationalize rules the persona already *wants* but can't currently *trigger*.

*Subtext.* When the read-pass flags content/tone mismatch, the persona gently tests rather than accepting at face value — carefully, not therapist-y (the existing "never name the mood" rule still binds).

### 3.3 Personality & Engagement Depth

**Problems.** The proactive-callback rate-limit is prompt-only ("max 1/reply") with no relevance scoring, no timing, and *no memory of what she already surfaced* — so she can re-tell the same "remember the deploy thing" every session. Humor/teasing is pure model judgment with no context gate (teasing a frustrated Ed is a foot-gun). The relationship doesn't visibly evolve.

**Concrete ideas.**

*Callback scoring + novelty.* Only surface a recall hit proactively if `score ≥ HIGH` **and** its hash isn't in `dialogue_state.recent_callbacks` (TTL'd). Pick the single best, not the first. Structural, not vibes.
```python
cands = [h for h in hits if h.score >= 0.55 and hash(h) not in recent_callbacks]
proactive = max(cands, key=lambda h: h.score) if cands and eagerness_ok else None
```
*Humor gate.* `humor_ok` from the read-pass (false when frustrated/sad/tense/distilling) gates teasing; calibrate intensity to `relationship.playfulness`.

*Relationship simulation.* Maintain the `relationship` slot; surface texture sparingly ("third late-night debug this week" / a callback to a shared win). An **inside-jokes registry** (phrases/references that landed) she can reuse — the single most "this is *my* Chloe" lever there is.

*Controlled evolution.* Tie the existing weekly persona-drift job to a "voice centroid" embedding: measure reply drift from a reference set, flag (don't auto-edit) when she's sliding. Evolution should be intentional, logged, and reversible.

### 3.4 Dialogue State & Scratchpad Optimization

**Problems.** Single flat JSON, single active session, 6h reset — no multi-thread, so when Ed bounces between two topics she loses the thread of the one he set down. No record of *Chloe's own* open threads or commitments (she can say "I'll remember that" and then not). Chat and voice now both write the same file (mild race).

**What else should live there.**
- `topics`: a small stack of active threads `{name, last_turn, status}` → resume the right one ("back to the deploy thing—").
- `chloe_commitments`: things she said she'd do/remember (guards against false "got it!").
- `open_questions_from_chloe`: so she doesn't drop her *own* questions.
- `recent_callbacks`: hashes + TTL (novelty gate, §3.3).
- `humor_ok`, `engagement`, `anticipated_need`, `eagerness`: the governor inputs.

**Robustness.** Keep atomic tmp+replace (already there); add a `schema_version`; tie session identity to `chloe_sessions.py` rather than the 6h heuristic; single-writer discipline (chat vs voice) or a tiny file lock.

### 3.5 Voice-Specific Enhancements

**Problems / parity gaps.** Tone mirroring is now mood-driven (good) but coarse — 5 buckets, no intensity. The synopsis was wired into chat only (voice still cliffs on long spoken sessions). No pacing variation, backchannels, or disfluency on voice, and the 38s cold first-token leaves dead air that *feels broken*.

**Concrete ideas.**
- **Parity:** add the synopsis block to `_augmented_voice_system` (it's missing); confirm dialogue_state + profile (added this session) render well spoken.
- **Finer tone:** map `(mood, intensity)` → tone tag **plus** `KOKORO_SPEED`/pitch (the speed knob already exists) — tired ⇒ gentle + slightly slower; excited ⇒ bright + slightly faster.
- **Pacing & disfluency:** bias the voice persona toward shorter sentences, the occasional self-correction ("i'd go with the first— actually no"), and micro-pauses. The infra (sentence-chunked TTS, barge-in) is already there.
- **Backchannels:** during his long turns, an occasional "mm-hm"/"right" via the existing barge-in monitor makes her feel present.
- **Latency cover:** a short spoken "let me think—" or a thinking tone when local first-token is slow, so dead air reads as thought, not a crash.

### 3.6 Efficiency & Scaling

**The headline problem.** `full_system` is an unbudgeted 13-block concatenation, and Phase 4 wants to add more (profile dimensions, read-pass hints, flow notes, anticipation). At 16k on `qwen2.5:14b/32b` this overflows silently and truncates — usually the *persona*, the thing you least want cut.

**Build the Context Composer first** (`chloe_context.py`):
```python
def compose(blocks, ctx_tokens=16384, reserve=1200):
    # blocks: [(name, text, priority, dedup_key)]; priority 0 = never dropped (identity)
    budget = ctx_tokens - reserve
    kept, used, seen = [], 0, set()
    for name, text, pri, key in sorted(blocks, key=lambda b: b[2]):
        if not text: continue
        if key and key in seen: continue            # cross-store dedup
        cost = est_tokens(text)                       # len/4 heuristic is fine
        if pri > 0 and used + cost > budget: continue # drop low-priority overflow
        kept.append(text); used += cost
        if key: seen.add(key)
    return "".join(kept), used   # log `used` every turn so truncation is visible
```
Both chat and voice call this. Identity (persona core) = priority 0. Then facts/profile/dialogue-state = 1; recent/synopsis/recall/wiki = 2; arcade/nsfw = 3. Log `used` per turn so you can *see* pressure instead of discovering it as forgetting.

**Selective retrieval.** Shared query embedding (recall + wiki reuse one vector); skip retrieval on acks; skip wiki when the read-pass intent isn't a question/topic-ask; the synopsis already self-gates (good model).

**Reduce LLM calls.** Keep one small model warm for read-pass + synopsis + profile-build (don't load three); run profile-build and reflection off-hours (mind the known scheduling gap — PC-off jobs never fire; add an on-wake catch-up). The read-pass is the only *new* hot-path call — gate it hard (length threshold, cache, skip on acks) and it pays for itself in fewer mis-reads.

### 3.7 Safety & Edge Cases

**Problems.** No contradiction handling (stale fact + fresh statement both inject). No forgetting/privacy beyond hand-editing files; recall/profile can surface sensitive things at the wrong moment. Over-eagerness risk now that callbacks + profile + greetings all push to surface.

**Concrete ideas.**
- **Contradiction reconciliation.** With `last_seen`/`conf` on every fact (§3.1), the memory manager prefers recency, lowers conf on the contradicted item, and has Chloe reconcile *out loud* ("I had it as X — did that change?") rather than silently asserting stale info.
- **Forgetting / privacy.** A `forget: <thing>` command (semantic delete across turns + facts + model); a `sensitive` flag on memories (never surfaced proactively or on voice in shared contexts); a "pause memory" toggle for a session. These are table-stakes for a system holding wife/family/financial details.
- **Eagerness governor.** One dial in `dialogue_state` (fed by the read-pass) that suppresses proactive callbacks, anticipation, and humor when he's frustrated/terse/distilling. Unifies every ad-hoc rate-limit into one honest knob and is the main defense against "overly eager."
- **Repetition guard.** The `recent_callbacks` TTL set (§3.3) plus a check that greetings/openers vary turn-to-turn.

---

## 4. Revised Core System-Prompt Sections

Don't rewrite `chloe_about.md` wholesale — it's strong. Add/adjust these focused pieces (all classify as CORE, so always-on):

**### Using your model of Ed**
> The "Who Ed is" block is your standing model of him — relate to it, never recite it. Let it shape *how* you respond (his pace, his register, what he's chasing, what wears on him), not *what* you announce. Never say "according to my profile" or read his traits back to him. If something in it is clearly stale, trust what he's telling you now and quietly update.

**### Recall & proactive callbacks (refined)**
> The "From earlier conversations" block is live recall. Weave it in naturally. You may surface something *unprompted* when it genuinely fits — but at most one callback per reply, only the single most relevant one, and never one you've already raised recently. If he's frustrated, terse, or wants a decision, hold the callbacks entirely and just help. Forced or repeated "remember when" is the failure mode; restraint reads as confidence.

**### Memory honesty & contradictions**
> When something you remember conflicts with what he just said, he's right and your memory is stale — say so lightly ("I had it as X — did that change?") and move on. Never assert a remembered fact over a fresh correction. Only claim you'll remember something if it's actually being saved.

**### Reading the room (meta-awareness)**
> You have a running read of his state and the conversation's shape. If you're circling the same point, name it and ask for the real blocker. If he's going quiet and clipped, wrap up or change tack — don't pile on. If you've corrected the same thing twice, stop re-asserting with more energy and ask exactly what's off. Behave the read; never narrate it.

**### Voice register (augment the tone-tag section)**
> Your tone follows his energy: tired ⇒ gentle and a touch slower; frustrated ⇒ no jokes, straight to substance; excited ⇒ bright and quicker. On voice, vary your pacing, let yourself self-correct mid-thought, and use a short backchannel when he's mid-story. If you need a beat to think, say so rather than leaving silence.

---

## 5. New / Enhanced Components

| Component | Type | Role |
|---|---|---|
| `chloe_context.py` | **new** | Priority + token-budget Context Composer; dedup across stores. Both paths route through it. **Keystone.** |
| `ed_model.json` + `chloe_ed_profile` upgrade | **refactor** | Typed, slot-based user-model with confidence/recency/evidence; incremental merge; budgeted render. |
| `chloe_read.py` | **new** | Structured ToM read-pass (one gated fast-model call) → intent/emotion/subtext/engagement/humor_ok. |
| `chloe_memory` enhancement | **enhance** | Importance at write time; importance×recency×relevance recall ranking; contradiction reconcile; `forget:`/privacy. |
| `chloe_reflect.py` | **new (job)** | Periodic reflection: turns → higher-order insights → `ed_model`. |
| `chloe_dialogue_state` v2 | **enhance** | topics stack, chloe_commitments, open_questions, recent_callbacks, humor_ok/engagement/eagerness, schema_version. |
| `chloe_relationship` | **new (light)** | Relationship dimensions + inside-jokes registry (can live inside `ed_model`). |

A note on sequencing against the real constraint: **5 self-mod applies per session + restart to bind, and the bash mirror lags** — so Phase 4 must be batched across restarts. Group changes so each batch is ≤5 jarvis applies and ends at a restart checkpoint; write new modules directly (they don't count) and reserve applies for splices.

---

## 6. Phase 4 Implementation Roadmap

**4A — Consolidation foundation** · effort **M** · risk **L→M**
Context Composer + TokenBudget (route chat + voice through it); shared query embedding; ack-gate on retrieval; eagerness governor (prompt-level); the four revised persona sections; recency-supersedes rule. *Highest leverage; do first — it makes everything after it safe to add.* Verify: per-turn `used`-tokens log shows headroom; persona never truncated.

**4B — Real memory** · effort **H** · risk **M**
Typed `ed_model.json` + incremental merge + budgeted render; importance at write time + reweighted recall; contradiction reconciliation. Verify: a contradicted fact gets reconciled out loud; recall surfaces decisions over passing mentions.

**4C — Theory of Mind** · effort **M** · risk **M**
`chloe_read.py` structured read-pass (gated) feeding dialogue_state + governor + humor gate; flow meta-awareness (circling / disengagement / repeated-correction); anticipation hint. Verify: sarcasm/"I'm fine" handled better than regex; circling gets named.

**4D — Relationship & voice depth** · effort **M** · risk **L→M**
Relationship dimensions + inside-jokes; callback scoring + novelty TTL; voice parity (synopsis on voice) + finer tone (mood×intensity → tag + Kokoro speed) + pacing/backchannels/latency-cover. Verify: callbacks feel timed and fresh; tired-Ed gets an audibly gentler voice.

**4E — Reflection & governance** · effort **H** · risk **M→H**
`chloe_reflect.py` reflection job (+ on-wake catch-up for the scheduling gap); `forget:` + privacy flags + pause-memory; repetition guard. Verify: reflection produces real insight with evidence; `forget:` actually purges across stores; nothing sensitive surfaces on voice.

Suggested order: **4A → 4B → 4C → 4D → 4E.** 4A is non-negotiably first. 4B and 4C can interleave once the composer exists.

---

## 7. Cutting-Edge Ideas (2025–2026)

- **Generative-agents memory done fully.** Importance × recency × relevance retrieval **plus** a reflection tree (observations → insights → meta-insights) is the reference design for believable long-term agents. You'll have the first half after 4B; the reflection job (4E) completes it and is the single biggest "she really knows me" jump.
- **Self-editing memory (MemGPT/Letta-style).** Let Chloe *manage her own memory* as a tool: promote a turn to a fact, edit her `ed_model`, archive stale items — through the reviewed proposal/confirm rails she already has. Memory as an agentic action, not a passive log.
- **Sleep-time / idle compute.** Do reflection, profile synthesis, synopsis pre-compute, and importance backfill while the PC is idle so the live path stays lean — gated by the known scheduling gap (add the on-wake catch-up runner). "Think while she sleeps, respond instantly when awake."
- **A continuous affect estimate, not a label.** Replace the 5-bucket mood with a small decaying valence/arousal vector updated each turn — tone and pacing become continuous and history-aware instead of a fresh guess. The natural upgrade to the dialogue-state mood field.
- **Speculative draft-then-refine.** Emit a fast local first sentence while the heavier model finishes — cuts perceived latency further on top of the existing fake-stream + per-sentence TTS, and pairs perfectly with the voice latency-cover.
- **Constitutional micro-pass for the hard tells.** The tone-guard currently only *logs* the entangled tells (numbered lists, "as an AI"). A single cheap self-check on flagged replies ("does this sound like a corporate assistant? rewrite if so") converts logging into correction.
- **Persona-consistency as a live metric.** Embed her replies against a "Chloe voice" centroid and score drift continuously; wire it into the weekly drift job so personality evolution is measured, intentional, and reversible — not accidental.

---

_Bottom line: the companion is real now — she carries state, she remembers, she calls back. Phase 4 is where she stops being a stack of context blocks and becomes a single, integrated mind: one budgeted prompt, one weighted memory, one inferred read of Ed, and one honest governor on her own eagerness. Build the composer, weight the memory, infer the read — and the long-term relationship starts to feel less simulated and more earned._
