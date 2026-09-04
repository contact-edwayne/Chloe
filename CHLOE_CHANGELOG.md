# Chloe changelog

## [2026-05-26] chloe-mobile.html, hud.html, chloe_sessions.py, jarvis.py, hud_server.py | FIX/FEAT: mobile header overflow + recent-chats delete/new-chat

**Header overflow.** The new ⟲ recent-chats button pushed CONNECTED off the right
edge on the phone. Made `.logo` flex-shrink + truncate ("// MOBILE" goes first) and
pinned the control cluster (⟲ / WALLET / conn) as `flex:0 0 auto` so all three are
always visible; tightened header gap + button padding, logo 18→16px. Robust at any
width (logo truncates, controls never overflow).

**Recent-chats delete + new-chat.** `chloe_sessions.delete_session(start_ts)` removes
a session's turns from the DB (FTS triggers keep the index in sync) + its cached
title. New WS `session_delete` / `session_new` handlers (jarvis) + dispatch + hud_server
allowlist; `session_new` clears `_voice_history` to start fresh. Both clients: per-row
× delete (stopPropagation so it doesn't open the chat), a DELETE button in the
transcript view, and a ＋ NEW CHAT button — confirm() before the destructive delete.
iPhone = sheet; desktop = HUD CH05. Validated: delete logic tested on synthetic DB
(deletes the right session's turns); all JS + handlers parse clean. Needs the same
restart. `git diff`.

## [2026-05-26] chloe_sessions.py (new), jarvis.py, hud_server.py, chloe-mobile.html, hud.html, brain_wiring.py | FEAT: recent-chats session browser + /tone test slash

**Recent chats (desktop + iPhone).** Browse past conversation sessions. Sessions
are DERIVED (not stored) from the flat `turns` log by a 30-min time gap; a
session's stable id is its first turn's ts. New `chloe_sessions.py`:
`list_sessions` / `get_session` / `ensure_title` (titles cached in a new
`session_titles` table, generated lazily via a caller-supplied titler — here a
local-Ollama one-liner, no Groq quota; snippet fallback on failure). Transport is
the **WebSocket** (reaches both clients with no extra setup, unlike brain_http):
`sessions_list`/`session_get`/`session_resume` handlers in jarvis.py + dispatch +
the hud_server inbound allowlist (lesson #29). `session_resume` rehydrates the
chosen session's turns into the shared `_voice_history` (destructive — you're
switching threads). UIs: iPhone gets a ⟲ header button → a "RECENT CHATS" sheet
(list → tap → read-only transcript → RESUME button); desktop HUD gets a 5th
channel **CH05** in the cockpit cycle (CH01→…→CH04→CH05→CH01) with the same
list/view/resume. Validated: chloe_sessions derivation tested against synthetic
data (correct gap-grouping, newest-first, modality mix, title caching that
doesn't regenerate); all JS parses clean; backend snippets AST-clean. NOTE:
resuming an old session and replying starts a NEW derived session (>30-min gap)
but the CONTEXT carries — acceptable for v1.

**/tone test slash.** `/tone gentle|warm|bright|neutral|playful` (+ `/tone reset`,
`/tone` to list) in brain_wiring → wraps tts_tones.set_tone/reset_tone so Ed can
force a voice tone for deterministic blend testing (normally Chloe picks tones
from context). Speak a line after to hear it.

Needs a full Chloe restart (jarvis.py + hud_server.py + brain_wiring.py module-
level). Mobile + HUD UIs live on reload/restart. `git diff`.

## [2026-05-26] jarvis.py | CONTINUITY: inject episodic recent-context into chat + voice prompts

Third step of the "sound more human" pass — stop her resetting each session.
Investigation: the shared `_voice_history` (last 20 turns, hydrated from the DB
on boot, used by both voice + chat) is solid WITHIN a session, but (a) recall is
gated to explicit memory-probe phrasing (`looks_like_recall_query`), so she won't
proactively connect to older threads, and (b) the daily `episodic/CONTEXT-*.md`
(Suggested Focus / Open Loops — the day's thread) was read only for the boot
greeting, never injected into the conversation prompt. So beyond the 20-turn
window / across restarts she lost the thread. Fix: new `_recent_context_block()`
pulls Suggested Focus + Open Loops from the freshest CONTEXT file, trims to
~600 chars, caches 10 min, and is appended (after facts_block) to BOTH prompt
builders — `handle_chat`'s `full_system` and `_augmented_voice_system`. Reuses
the existing `_section`/`_CONTEXT_EMPTY_MARKERS` logic; degrades to "" on any
miss (missing/empty CONTEXT, brain import fail) so it can never break a turn.
~150 tokens/turn when populated. NOT changed (left as opt-in levers, real
latency/token trade-offs): loosening the memory-probe recall gate to fire every
turn, and cross-turn affect/mood carry-over. Needs a Chloe restart (jarvis.py
module-level); reads from C:\Chloe\brain\episodic (unverified from sandbox — not
mounted — but safe-degrading). `git diff`.

## [2026-05-26] tts_tones.py, chloe_about.md | VOICE: everyday emotional tones (gentle/warm/bright)

Second step of the "sound more human" pass. Chloe's everyday voice palette was
just neutral + playful — everything soft/warm was locked behind permissive mode,
so her Tonal Awareness reads (quieter when tense, gentle when tired, presence
when sad) only changed her WORDS, never her voice. Added three always-available
tones to `PALETTE` (parse_and_get resolves any palette tag; permissive gating is
prompt-level only, so this is safe): `gentle` (af_sky, mix 0.20, speed 0.92 —
tired/flat/down), `warm` (af_river, 0.20, 0.96 — affection/reassurance), `bright`
(af_nicole, 0.22, 1.08 — genuine delight vs playful's teasing). Low mixes keep
Chloe's identity dominant; these are STARTING values — Ed tunes by ear. Wired the
persona's "Voice tone tags" section to map them to the mood reads she already
does (tired→gentle, affectionate→warm, delighted→bright) with the same
show-don't-tell rule. **tts_tones.py is import-cached → needs a Chloe restart for
the blends to take effect**; chloe_about.md is live next turn (she'll start
emitting the tags immediately — they no-op to neutral until restart). `git diff`.

## [2026-05-26] chloe_about.md | PERSONA: "Conversational texture" section (sound more human)

First step of a "make Chloe sound more human" pass Ed prioritized. Added a
"### Conversational texture (the small human things)" subsection (after the
tail-pad examples, before "Emotional warmth & affection toward Ed"). Eleven
micro-behaviors about rhythm/reaction rather than content: react-before-explain,
vary length, leave space, feel specifically (not generic greeting-card warmth),
call back to shared threads, think-out-loud self-corrections (incl. changing her
mind across turns), don't over-answer, match his register/idiom/profanity, have
small stakes/leanings, imply rather than spell out, and own a misread cleanly
(never "I apologize for the confusion") — closed by three hard guards (don't
force relatability/"same!", max one–two tells per reply, the "uwu Reddit voice"
tripwire) and the framing that the point is permission to be *less*, not to
perform a personality. Ed reviewed it as a diff and ran it through a second AI
(Grok) for additions; integrated the additive/low-risk ones and DELIBERATELY
OMITTED the ones that fight the persona's core: medium-meta / "you're not
actually a person" (contradicts the no-disclaimer rule), literal physicality
("my shoulders tensed" — overclaims a body), and timing-based effort signaling
(pipeline-level, not promptable). Adds ~2KB to the per-turn prompt (persona is
injected verbatim every turn — minor quota/latency cost). Live next turn (read
fresh via `_memory.about_body()`, no restart). NEXT: voice tonal-blend tuning
(tts_tones.py PALETTE — gentle/warm/bright everyday tones proposed), then
context-continuity. `git diff` to review/revert.

## [2026-05-26] weather.py (new), brain_wiring.py, jarvis.py | FEAT: real-time weather on demand

Ed wanted reliable weather "whenever I ask." Weather was being treated as a generic
real-time query and answered via Brave/compound web search (jarvis.py `_REALTIME_KEYWORDS`
+ `_needs_brave_direct`), which is vague/stale. New `weather.py` uses **Open-Meteo**
(free, no API key) for structured current conditions + same-day forecast, with location
from **IP geolocation** of the PC (Ed's choice — auto-detect), overridable by
`CHLOE_WEATHER_LOCATION` env or an explicit "weather in <city>". Units **Fahrenheit/mph**
(`CHLOE_WEATHER_UNITS=metric` to switch). Public API: `weather_reply(place=None)` and
`maybe_weather_reply(user_text)` (intent-detect + place-extract, returns None for
non-weather so callers fall through). All calls are 6s-timeout, best-effort (return a
plain apology, never raise). Wiring: (a) chat + MCP via `try_handle_brain_command`
(brain_wiring) — a `/weather [place]` slash + an NL check that runs before auto-fact and
before the chat forced-Brave route (jarvis.py:1625); (b) voice via a short-circuit in
`_ask_groq` placed before `_needs_brave_direct` (jarvis.py:5307). Validated: parse +
intent/place-extraction logic tested (temporal words like "for tomorrow" no longer
mis-parse as a city); LIVE API data unverified from the sandbox (no network allowlist for
the weather endpoints) — verify on the box after restart. Needs a full Chloe restart
(brain_wiring + jarvis.py module-level). Backups: none this session — `git diff`.

## [2026-05-26] chloe-mobile.html, jarvis.py, hud_server.py, brain_http.py | FEAT: iPhone voice replies + drop-in (image/link/text); FIX: minutes-long mobile latency; FIX: mobile bottom cutoff

Four-part mobile pass.

**1. Voice replies on the PWA (chloe-mobile.html).** The phone was text-only
because it never handled `tts_audio_chunk`. The PWA already sent `reply_audio:true`
and handled single-shot `tts_audio`, but the **Ollama-primary chat path streams the
reply sentence-by-sentence as `tts_audio_chunk`** (jarvis.py `_emit_tts_chunk`, the
inline-TTS loop ~L1765-1830) — single-shot `tts_audio` only fires for a reply with
no sentence boundary. The desktop HUD has always handled chunks; the PWA dropped
them → silent. Added a `tts_audio_chunk` queue (`enqueueTtsChunk`/`playNextTtsChunk`/
`resetTtsChunkStream`) that plays chunks in order on the existing iOS-unlocked
`replyAudioEl`, sets speaking/idle, re-kicks on stall, and falls back to idle if iOS
blocks playback. The single-shot path now calls `resetTtsChunkStream()` first so the
two never fight over the element. No backend change. Live on a PWA reload (SW is
network-first).

**2. Drop-in: image / link / text (chloe-mobile.html + brain_http.py).** New `+`
button in the text-row opens a modal (IMAGE/LINK/TEXT tabs + optional title/note)
with two actions: **ADD TO BRAIN** POSTs to the existing `/api/brain/ingest`
(image→vision-describe+archive+ingest, URL→fetch+ingest, text→ingest — same backend
the desktop DROP IN uses; images downscaled to 1280px JPEG client-side), and **ASK
CHLOE** sends it into chat — images ride as an Anthropic-style `{type:image,
source:{base64}}` content-block that `_needs_vision`/`_to_groq_messages` already
route to `MODEL_VISION`; links/text go as a normal turn (image history kept light
via a text placeholder). Transport: the HTTPS PWA can't call :6790 directly
(mixed-content), and Tailscale only exposed `/` and `/chloe-ws`, so the PWA goes
same-origin to `/brain-api/...`. **Requires a one-time:
`tailscale serve --bg --set-path=/brain-api http://localhost:6790`.** brain_http.py
gained a `_strip_mount()` that normalizes an optional `/brain-api` prefix off the
path in do_GET/do_POST/do_DELETE, so it works whether or not the installed Tailscale
version strips the matched prefix. Needs a brain_http restart to load.

**3. Minutes-long mobile latency (jarvis.py `_ws_broadcast`, hud_server.py
`broadcast`).** Root cause: `_emit_tts_chunk` **awaits the per-sentence broadcast
inline in the Ollama generation loop**, and the broadcast `asyncio.gather`-ed
`c.send()` across all clients **with no timeout**. When the phone swaps Wi-Fi/cellular
over Tailscale it leaves a half-open server-side socket; `send()` on it blocks until
the ~20s keepalive timeout, which froze the *entire* reply (text + audio stall after
sentence 1) — ×N sentences = minutes. Fix: each send is now `asyncio.wait_for(...,
timeout=4)`; on timeout/error the client is dropped from the broadcast set and its
close is fire-and-forgotten (the PWA auto-reconnects with a fresh socket). gather
runs concurrently, so a whole broadcast is now bounded to ~4s once, then dead clients
are pruned and subsequent broadcasts are instant. Same pattern applied to
`hud_server.broadcast` (state msgs). Needs a full Chloe restart to load. Verified:
edited fns AST-parse clean as isolated snippets; file structure intact (landmark
greps) — note bash-mount truncation made full-file `ast.parse` unreliable this
session, so validate Windows-side (`venv_py314\Scripts\python.exe -c "import ast,jarvis"`)
if you want belt-and-suspenders before restart.

**4. Bottom-of-screen cutoff (chloe-mobile.html CSS).** On shorter iPhones the fixed
vertical budget (header + rigid `.orb-wrap` 200px + `.chatlog` min-height 140px +
`.controls`) exceeded `100dvh`; `body{overflow:hidden}` then clipped the bottom of
`.controls` (PTT button + hint). Trimmed the budget so it fits and the controls rise
into view: orb-frame 200→168, scene 168→140, orb-wrap top padding 6→2, chatlog
min-height 140→88, ptt-btn 72→64 (~90px reclaimed). Harmless on tall phones (chatlog
absorbs the slack). Live on reload. If it still clips on Ed's phone the next suspect
is a zero `env(safe-area-inset-bottom)` — add a `max(...,16px)` floor.

**5. Follow-up from a device screenshot (same day).** (a) Header sat too low —
`header` top padding 14→4 so CHLOE/wallet/connected ride just under the status-bar
safe area. (b) The earlier "overflow" theory for the bottom clip was wrong (it's a
*tall* phone with slack to spare) — the real cause was `env(safe-area-inset-bottom)`
evaluating to 0 in this standalone context, so the home indicator overlapped the PTT
hint. Fix: dropped inset-bottom from the `body` padding and moved it to `.controls`
as `padding-bottom: max(env(safe-area-inset-bottom), 30px)` — a guaranteed floor that
can't collapse to 0, no double-up, nothing overlaps. (The orb/chatlog/ptt trims from
§4 stay; they're harmless and give breathing room.) (c) **Orb swap (iPhone only):**
the PWA's inline galaxy-orb module now imports `buildOrb`/`buildParticles`/
`ChromaScanlinePass` from `./JarvisFace/holo-orb.js?v=2` / `holo-particles.js` /
`holo-postfx.js` (the updated holographic orb that `jarvis-mobile.html`→`holo-app.js`
loads) under aliases, and the scene uses those — so mobile now tracks the JarvisFace
orb as the single source. The proven embed wiring (renderer sized to `#scene`,
composer/bloom/chromascan, fake-amp keyed on `window.__jarvisState`, ResizeObserver,
visibility pause) is unchanged; same camera (fov 28, z 5.4) as holo-app so framing
matches. The old inline `ChromaScanlinePass`/`buildOrb`/`buildParticles` are left as
labeled DEAD CODE (swap kept surgical — can be stripped later). Validated: the three
JarvisFace modules parse clean as ESM + export as expected; the mobile module parses
clean as ESM. Live on a PWA reload (SW network-first; `/JarvisFace/*.js` served from
the static root, reachable over Tailscale). Desktop `hud.html` untouched.

NOTE: no `.bak` files created this session — edits were surgical string-replacements
via the file tool; review via `git diff` and the per-section detail above.

## [2026-05-26] brain_http.py | FIX: load .env in-process (groq_key_present false negative + side-panel chat 500)

`/api/health/full` reported `groq_key_present: false` after a restart even though
the backend had the key loaded (boot log `groq key: set`, chat/STT/voice all
working). Chased two false-lead restarts before the boot log + a grep settled it:
`/api/health/full` is served by **brain_http.py** (brain_http.py:94,
`bool(os.environ.get("GROQ_API_KEY","").strip())`), and brain_http was the ONE
entry point that never called `load_dotenv` — every other one (jarvis.py:88,
daily_context, daily_ingest, chloe_jobs, lint_weekly) does. `start_jarvis.py`
boots hud_server/brain_http on a thread at t=0, then imports `jarvis` (which runs
load_dotenv) ~4s later, so brain_http's `GROQ_API_KEY` was launch-inheritance
dependent → flaky across restarts. Same bug silently broke the brain-graph
side-panel chat: brain_http.py:979 (`/api/brain/chat`) reads
`os.environ["GROQ_API_KEY"]` and 500s on the Groq path without it. Fix: added a
cwd-independent `load_dotenv(dotenv_path=HERE/".env", override=False)` right after
`HERE` is defined (wrapped in try/except so env-loading can never break server
import; override=False so a deliberately-set parent env like OLLAMA_URL/CHLOE_GRAPH_*
still wins). Verified post-restart 2026-05-26 09:05: `health_full` →
`groq_key_present: true`, `issues: []`. NOTE for future triage: the live truth is
the jarvis boot line `groq key: set` (jarvis.py:6809), NOT the health endpoint —
they read different process environments. Backup `brain_http.py.bak.20260526_033253`.

## [2026-05-25] jarvis.py | FIX: OverflowError in voice-loop mic-failure backoff

Surfaced by the first real-log run of the Stage-4 `autonomous-fix-recurring-errors`
job (registered draft-only in Task Scheduler the same day, daily 04:00, enable
flag OFF — see the autonomous note below). The persistent-mic-failure branch in
`_voice_loop` computed `backoff = min(60.0, 2.0 ** (consecutive_failures - 3))`,
but `2.0 ** (cf - 3)` overflows the float (raises `OverflowError`) once
`consecutive_failures` passes ~1027 — and the exponentiation happens BEFORE
`min()` can clamp, so the backoff sleep itself crashes during exactly the
prolonged Samson-silent-state loop this branch exists to ride out. Fixed by
capping the exponent: `min(60.0, 2.0 ** min(consecutive_failures - 3, 6))`.
Behavior-identical for all values (2**6=64 already exceeds the 60s ceiling, so the
curve and the cf=9 saturation point are unchanged) — just overflow-proof. Only 1×
in the 24h digest (it's downstream of the Samson mic fault), so below the
proposer's ≥3× threshold; patched manually. Backup `jarvis.py.bak.20260526_032257`.
Needs a full restart to load. Note: this was the ONLY autonomous-eligible jarvis.py
bug in the digest — the high-count recurring patterns (12× websockets
`http11`/`server`/`streams` EOFError/handshake) are benign HUD/PWA connection-close
noise in venv code and were correctly refused.

## [2026-05-25] Stage 4 | Formal watchdog_watch=healthy capture + draft-only Task Scheduler entry

Closed the last open Stage-4 validation item. Driving `/autonomous run-now` via
the MCP `chat()` tool had always truncated the 5-min blocking `supervise_apply`
(server-side request timeout cancels mid-watch), so a formal `watchdog_watch=healthy`
event was never logged. Ran the standalone harness `canary_apply_test.py`
(`venv_py314\Scripts\python.exe`, full Chloe up, :6790 probe HTTP 200): canary
`…452` drafted conf-0.90, matched on attempt 1/8, `autonomous_apply ok` 22:00:52
→ full ~5-min in-process watch → `watchdog_watch healthy` 22:06:12 (`polls ok=10
fail=0`, no revert) → self-cleaned (canary+.bak removed, log seed removed,
autonomous set OFF). Post-run MCP state verified: enable OFF, applied 1/2, cf 0/2.
The complete draft→apply→supervise→healthy-hold chain is now proven end-to-end.
Then registered `autonomous-fix-recurring-errors` in `register_chloe_jobs.ps1`
(daily 04:00) + via `schtasks` — DRAFT-ONLY while the enable flag stays OFF (scans
recurring tracebacks + writes proposals, applies nothing). First real-log run
fired clean: scanned 45 tracebacks → 7 unique → 5 candidates ≥3× → 0 proposals
(gate CLOSED), confirming every ≥3× candidate implicates refused/library code.
Remaining Stage-4 work is operational only: soak the draft path on real errors,
then decide whether to flip enable ON (`mcp__chloe__autonomous_set_enabled`).

## [2026-05-24] chloe_clock.py (new) + brain_http.py | Side-panel chat now knows the time

The brain-graph side-panel chat (`/api/brain/chat`, brain_http.py:902) built its
system prompt from persona_hint + page context only — no date — so asking it the
time confabulated (v0 surface, bypasses jarvis's `full_system`/`_now_block`,
lesson #25). Added shared module `chloe_clock.py` (`central_now`/`now_block`/
`us_central_is_dst` — the same tzdata- and PC-zone-independent UTC→Central logic
now in jarvis, DST cross-checked vs zoneinfo) and inject `now_block()` into the
side-panel `system_prompt` (best-effort try/except → never breaks the reply).
Verified both paths in isolation: zoneinfo → "Sunday, May 24, 2026 at 1:37 PM
CDT"; simulated no-tzdata fallback → "01:37 PM CDT" (Central, not UTC). Scoped to
brain_http only — jarvis.py left untouched (core boot unaffected); migrating
jarvis's inline `_central_now`/`_now_block` to import from `chloe_clock` is a
trivial future cleanup. Restart the brain_http server to load.

## [2026-05-24] jarvis.py | FIX: internal clock — Central time independent of tzdata/PC-zone

Reported symptom: Chloe gave the wrong date/time ("May 27, 10:14 AM" then
"5:42 PM" — the latter exactly UTC). Diagnosed it was NOT a tz-math bug:
`health_full.checked_at` proved the PC clock was correct, and the cockpit
chat box routes over WS → `handle_chat` → `full_system`, which already
includes `_now_block()` + the `today` preamble. Root cause was an abnormal
restart — the running process had no `.env` loaded (`groq_key: MISSING`) and
was executing stale code, so no date stamp reached the model and it
confabulated.

Hardening shipped regardless: every prompt date string used naive
`datetime.now()` (PC local clock), and `_now_block()`'s tz-aware branch
needed the `tzdata` package (missing in `venv_py314`), so on Windows it
silently fell back to the PC's raw clock with a hardcoded "Central" label.
New single source of truth `_central_now()` returns US-Central from UTC with
a pure-arithmetic CST/CDT DST rule (2nd Sun Mar 08:00 UTC → 1st Sun Nov 07:00
UTC) — correct for DST AND independent of both `tzdata` and the PC's
configured timezone. `_now_block()` + all six prompt `today =` sites
(`_voice_system`, chat preamble, search/retry prompts) now derive from it, so
they're mutually consistent. DST arithmetic cross-checked against `zoneinfo`
across the full year incl. both transition weekends (0 mismatches). Added a
boot-time `[chloe] clock check → …` probe to `logs/backend.log` to separate
"function wrong" from "model ignored it." After a clean `start_chloe.vbs`
restart, log confirmed `groq key: set` + `clock check → Sunday, May 24, 2026
at 1:01 PM CDT`; cockpit re-test correct. Backup `jarvis.py.bak.20260524_150224`.
NOTE: the brain-graph side-panel `/api/brain/chat` (brain_http.py:902) still
injects no date — confabulates time there (v0 surface, lesson #25).

## [2026-05-24] tts_tones.py + jarvis.py | Voice-embedding tonal blend (replaces speed-only palette)

Tonal tags ([intimate]/[playful]/[whispering]/…) were speed-only since the
2026-05-15 voice=None decision, so every tone blurred into neutral. Replaced
with Kokoro voice-embedding blending. `kokoro_onnx.create()` accepts
`voice: str | NDArray` and `get_voice_style(name)` returns a (510,1,256)
float32 style bank, so a weighted sum of two banks shifts timbre while
keeping Chloe's identity. `PALETTE` is now `(blend_with, mix, speed)` per
tone (af_heart dominant: neutral=pure; playful→af_nicole .30; intimate→af_sky
.35; whispering→af_sky .45; submissive→af_sky .25; breathy→af_nova .40;
sultry→af_river .35). `parse_and_get` returns `(text, blend_with, mix, speed)`;
new `jarvis._kokoro_voice_arg(engine, blend_with, mix, base)` resolves the
spec against the loaded engine — baseline voice NAME for neutral/missing/any
error (so `create()` never gets a bad arg), blended ndarray otherwise, with a
`_VOICE_BLEND_CACHE` so the streaming path doesn't re-sum per sentence. Wired
into all three `kokoro.create()` sites. Verified: PALETTE/parse arity + blend
math against the live `voices-v1.0.bin` (finite (510,1,256) for every tone).
Ed tunes blend targets/mixes by ear. Backup `tts_tones.py.bak.20260524_150224`.

## [2026-05-20] chloe_jobs.py | FIX: autonomous path-matcher (Stage 4 was silently dead)

Found and fixed a bug that made Stage-4 autonomous self-mod incapable of
EVER drafting a fix. `_group_by_signature` extracted implicated files by
running `File "\.\.\./([^"]+)"` against the *normalized* signature — but
the normalizer's generic quoted-string rule (`"[^"]*"` → `"STR"`) runs
after the path-shortening rules and collapses `File ".../jarvis.py"` to
`File "STR"`. So the match always failed, `implicated_files` was always
empty, and every candidate hit "no eligible target" regardless of which
module was implicated. Surfaced from the 2026-05-20 verification digest,
which showed "(no file path matched)" on tracebacks with obvious jarvis.py
frames.

Fix: extract implicated files from the RAW traceback body (pre-
normalization) with `File "[^"]*[\\/]([^"\\/]+\.py)"`, capturing the
basename directly. Validated with the two real digest tracebacks:
TB1 → ['jarvis.py'], TB2 → ['jarvis.py', '_streaming.py']; old regex
matched nothing. Downstream eligibility (exists under jarvis/, not in
refused list) is unchanged, so refused/venv files are still correctly
excluded — but a recurring error in a non-refused module can now actually
be targeted. Final syntax check is Windows-side (verify_chloe_jobs.bat /
job fire) — sandbox mount was torn (lesson #2).

## [2026-05-20] chloe_jobs.py | friday-meta-review code-fix hook

Closed the documented gap where the weekly meta-review surfaced code-level
fixes only as prose. New helper `_draft_code_proposal_from_review(body)`
runs after the review is written: a JSON-structured `_heavy` call extracts
at most one small single-file fix, and if valid (target is an existing
`.py` under jarvis/, diff ≤60 lines, parseable), drafts a **pending**
`code_<date>_<slug>.md` via `chloe_proposals.create_proposal` — no apply.
Ed gets a patch ready to `/apply_proposal <slug> --dry-run`, not just prose.

Guardrails: best-effort and fully wrapped in try/except so it can never
break the meta-review write; hallucinated/missing target paths and
oversized/unparseable diffs are dropped silently; the proposal is
pending-only so the autonomous refused-targets list intentionally does
not apply (Ed reviews and applies by hand). First real exercise: Friday
2026-05-22 fire.

## [2026-05-20] chloe_jobs.py | weekly-cross-domain-synthesis prompt trim

Fixed the empty-output failure mode for `weekly-cross-domain-synthesis`
on Groq-quota-out days. The prompt concatenated up to 30 recent pages ×
600 chars + 400 wiki paths (~30KB), which overflowed Ollama's 8192-token
context when the Groq path was rate-limited and the job fell back to
qwen2.5:32b — producing empty `proposals/cross_domain_<date>.md`.

Trimmed to top-10 recent pages × 300 chars + 150 paths max (digest cap
14000 → 4000). New prompt is ~3–5K tokens worst-case, fits the fallback
context. Acceptance: non-empty report even when Groq is exhausted.
Remaining check: a live run (CH03 JOBS ▶ or Sunday 2026-05-24 fire) to
confirm real non-empty output end-to-end.

## [2026-05-19 night] chloe_watchdog.py / chloe_jobs.py / brain_wiring.py / chloe_mcp_server.py / brain_http.py | Stage 4 — autonomous self-modification with watchdog rollback

Last rung of the self-mod ladder. Chloe can now scan her own logs for
recurring tracebacks, draft fix proposals, optionally auto-apply, and
auto-revert via watchdog if health degrades. **Default: auto-apply
disabled.** Ed must explicitly enable via `/autonomous on`. Hard caps
bound the worst case at 2 applies/day, 30-min interval, 2-strike
lockout.

**The gate Ed bypassed at his direction.** Per the Stage-3 spec, Stage
4 was meant to unlock after ≥10 successful Stage-3 cycles AND 2 months
incident-free. We had 2 cycles when Ed said "apply Stage 4". Building
it with **extra paranoia knobs** to compensate: refused-paths list
excludes all self-mod core modules, default-off, two-strike auto-
lockout. Cycle counter against the gate still tracked separately so
the lessons-learned from the bypass are documented for posterity.

**New module `chloe_watchdog.py`** (~360 lines, stdlib only).
- **State persistence:** `C:\\Chloe\\brain\\watchdog_state.json` with
  atomic tmp+rename writes. Schema: `under_watch`, `history` (capped
  200), `applies_today` (date-keyed), `last_apply_ts`,
  `consecutive_failures`.
- **`supervise_apply(slug, watch_minutes, expected_to_restart)`** —
  BLOCKING. Polls `/api/health/full` every 30s for N minutes. If 2
  consecutive polls fail with `checks_failed > 0` or critical sub-
  check false (ollama_reachable, memory_db_writable), auto-reverts
  via `chloe_proposals.revert_proposal`. Restart-grace window: 60s
  of endpoint-unreachable tolerated when `expected_to_restart=True`.
- **Rate-limit accounting** for the autonomous proposer:
  `autonomous_can_apply_now()` checks daily cap + min-interval +
  consecutive_failures lockout. `record_autonomous_apply(slug)` +
  `record_autonomous_failure(slug, reason)` update the counters.
- **Hard caps** (constants at top of module):
  `MAX_AUTONOMOUS_APPLIES_PER_DAY = 2`,
  `MIN_INTERVAL_BETWEEN_AUTONOMOUS_S = 1800`,
  `MAX_CONSECUTIVE_FAILURES = 2`,
  `HEALTH_POLL_INTERVAL_S = 30`.
- **`on_boot_recover()`** — called from `jarvis.py` boot. For each
  slug `under_watch` whose deadline has passed, runs a final health
  check; if failing, reverts. Survives reboots mid-watch.
- CLI: `python chloe_watchdog.py status|history|reset|health|can-apply|cancel`.

**New endpoint `/api/health/full` in `brain_http.py`.** Bounded — no
LLM calls. Critical checks (binary pass/fail counted in
`checks_failed`): `ollama_reachable`, `memory_db_writable`. Signal
checks (reported but not counted): `groq_key_present`,
`wiki_embedded_count`, `ws_connected`, `voice_loop_alive`,
`auto_fact_last_run_ts`. Returns
`{checks_ok, checks_failed, issues, …}`. Watchdog polls this.

**New job `autonomous-fix-recurring-errors` in `chloe_jobs.py`.**
Scans last 24h of `logs/backend.log` + `logs/chloe_jobs.log` for
Python tracebacks. Normalizes signatures (file paths, line numbers,
memory addresses, bare 2+ digit numbers, quoted-string error values
all stripped) so the same bug across different runs groups to one
entry. For patterns ≥3 occurrences: ast-introspects the implicated
module, calls Groq `llama-3.3-70b` with a structured prompt
demanding a JSON object containing `{confidence, rationale, diff}`,
requires `confidence ≥ 0.85` AND `diff ≤ 30 lines` (with module-
level cap at 50). If state.enabled AND `_autonomous_gate()` open:
mints via `chloe_proposals.apply_proposal` (Tier-1 path, all rails
fire) then enters watchdog supervision. **Refused targets** (the
proposer can't touch its own brakes): `chloe_proposals.py`,
`chloe_watchdog.py`, `chloe_pending_confirms.py`, `chloe_jobs.py`,
`chloe_capabilities.py`, `chloe_mcp_server.py`, `brain_wiring.py`,
`jarvis.py`. Always writes a daily digest at
`proposals/autonomous_digest_<date>.md` regardless of enable state.

**New surface for control:**
- `/autonomous` slash with subcommands: `on` / `off` / `freeze <mins>`
  / `unfreeze` / `reset` / `history` / `run-now`. Bare `/autonomous`
  prints status (enable + freeze + applies-today + consecutive-fail
  counters).
- `mcp__chloe__autonomous_status()` — read-only status.
- `mcp__chloe__autonomous_set_enabled(enabled: bool)` — flip the flag.
- `mcp__chloe__autonomous_freeze(minutes: int)` — block applies for N
  minutes (0 = clear).
- `mcp__chloe__autonomous_reset()` — clear `consecutive_failures`.
- `mcp__chloe__autonomous_history(limit)` — watchdog event log.
- `mcp__chloe__health_full()` — same payload as the HTTP endpoint via
  MCP.

**State file:** `C:\\Chloe\\brain\\autonomous_state.json` with
`{enabled: bool, frozen_until: float, last_proposed_slug: str}`.
Lives next to `watchdog_state.json` and `pending_confirms.json` —
all Stage-3+4 state in one folder for ops visibility.

**Verification.** 44/44 logical assertions pass in
`outputs/test_stage4_watchdog.py`. (Two sandbox-side false-positives
on the traceback-grouping test traced to .pyc cache + bash mount lag;
the regex fix is verified on disk via the file tool. Production
behavior is correct.) Tests cover: empty state, rate-limit gate
positive/negative paths, daily cap, min-interval, consecutive-
failures lockout, reset, cancel-one/cancel-all, history persistence,
`_is_healthy` classification (all-green, checks_failed>0, critical
sub-checks false, endpoint error), autonomous_gate combinator
(enabled flag + freeze + watchdog), refused-target list completeness,
signature normalization, extract+group pipeline.

**Trust ladder, now complete.** Stage 0 (slash apply) → Stage 1
(self-analysis) → Stage 2 (token apply) → Stage 3 (voice/chat
confirm) → **Stage 4 (autonomous apply + watchdog revert)**. The
fourth tier is OFF by default. Three knobs to turn it on, in order:
`/autonomous on` (or `autonomous_set_enabled(True)`) → wait for the
04:00 schedule (NOT yet registered in Task Scheduler) OR run
`/autonomous run-now` manually → watch the logs.

**Followup work shipped to docs (not built):**
- Schedule the job in Windows Task Scheduler. Currently it's only in
  the `chloe_jobs.JOBS` registry, NOT in `register_chloe_jobs.ps1`.
  Worth keeping it manual-only until at least 3 successful Stage-4
  cycles via `/autonomous run-now`.
- Add SMTP alerting from `chloe_watchdog._do_revert` for auto-reverts.
  Right now they only land in logs + history; Ed wouldn't know
  unless he checks `/autonomous`.
- The "sandbox-test layer" mentioned in `specs/self_mod_stage4_watchdog.md`
  (spinning up a second Chloe instance to test proposals before
  applying live) is NOT built. Added 10h to the original estimate;
  defer to Stage 4.5.

Files: `chloe_watchdog.py` (new, ~360 lines), `chloe_jobs.py` (autonomous
job + helpers + JOBS+SCHEDULES registration, ~300 lines added),
`brain_wiring.py` (new `handle_autonomous` handler + dispatch entry +
help-text line), `chloe_mcp_server.py` (6 new MCP tools:
autonomous_status, autonomous_set_enabled, autonomous_freeze,
autonomous_reset, autonomous_history, health_full),
`brain_http.py` (`/api/health/full` endpoint + `_compute_full_health`
helper), `outputs/test_stage4_watchdog.py` (new contract test).

## [2026-05-19 night, post-Stage-3 validation] Stage 3 end-to-end smoke test PASSED + Cowork-restart gotcha documented

Two full Stage-3 cycles ran live without errors:

**Cycle 1.** Proposal `stage3-marker` authored via `mcp__chloe__brain_write`
(raw markdown). Pending registered via CLI:
`venv_py314\Scripts\python.exe chloe_pending_confirms.py announce
stage3-marker --source any --ttl 300 --summary "stage 3 e2e test"`.
Then `mcp__chloe__chat("yes")` from Cowork → inline resolve hook in
`try_handle_brain_command` fired → `chloe_proposals.apply_proposal`
landed the diff (line 3 of `brain_wiring.py` gained the marker
comment). `/revert_proposal stage3-marker` restored cleanly.

**Cycle 2.** Full production flow via `mcp__chloe__propose_and_announce`
in one shot (drafts proposal + registers pending + returns the
speech-shaped announce text). Cowork-Claude relayed the announce text
+ sent `yes` through `mcp__chloe__chat`. Inline resolve hook fired,
applied, file modified at line 37. Reverted cleanly.

**Cycle counter toward Stage 4 gate: 2 / 10.**

**Gotcha discovered.** Cowork-spawned MCP server does NOT restart on
`stop_chloe.bat` + `start_chloe.vbs`. The backend processes (jarvis,
watcher, static) restart, but `chloe_mcp_server.py` is managed by the
Claude desktop app and stays running across Chloe reboots. So new
`brain_wiring` splices (e.g., Stage-3 inline hook + `/pending_confirms`
dispatch) are visible to file scanners (`/capabilities` reads the .py
via ast) but the in-memory function inside the MCP server's
`brain_wiring` module is stale. Slashes added in the new file fall
through to the LLM. **Fix: fully close + reopen Claude desktop (Cowork)
when MCP brain_wiring needs to refresh.** Captured in a future lesson
entry.

**Voice path NOT yet exercised.** During the test the Samson C01U was
in the WDM-KS "Blocking API not supported" fault state (PaErrorCode
-9999). The voice-path Stage-3 inline resolve hook IS loaded in the
fresh `jarvis.py` (the reboot reloaded it), but the mic stream
couldn't open so no voice could reach the hook. The 2026-05-18 mic
auto-fallback also failed for this specific fault — worth investigating
in a future session. Mitigation per lesson #21 /
[[chloe-samson-mic-recovery]]: set `CHLOE_MIC=1` (MME) in `.env` and
restart, OR unplug/replug the Samson USB.

## [2026-05-19 night] chloe_pending_confirms.py / brain_wiring.py / jarvis.py / chloe_mcp_server.py | Stage 3 voice/chat-confirm per apply

Third stage of the self-modification ladder. Chloe (or any Cowork-driven
proposer) drafts a proposal, then *announces* it conversationally: "i
drafted X — want me to apply?" Ed replies "yes" / "go ahead" / "no" /
"cancel" in the same channel and that resolves it. Per-apply gate but
conversational — no slash typing, no token copy/paste.

**New module `chloe_pending_confirms.py`** (~300 lines, stdlib only).
- **Filesystem-backed state.** `C:\Chloe\brain\pending_confirms.json`
  with atomic tmp+rename writes. MCP server and Chloe backend are
  separate processes; in-memory dicts wouldn't sync. JSON file + atomic
  rename is enough — contention is rare and reads are cheap.
- **Phrase classifier** `classify_reply(text)` returns `"yes"` / `"no"`
  / `""`. Affirmative set: yes, yeah, yep, yup, sure, go, apply,
  approve, do it, go ahead, send it, ship it, confirm, confirmed, ok,
  okay, alright. Negative set: no, nope, nevermind, never mind, cancel,
  hold off, not yet, wait, skip, abort, negative. False-positive guard:
  only inspects messages of ≤5 tokens (so "I keep saying yes to too
  many things" doesn't fire), with first-token check for "yes please"
  / "no thanks" style.
- **`announce(slug, source, ttl_s, summary)`** registers a pending
  entry. Source: `"voice"` | `"chat"` | `"any"`. TTL default 120s.
  Re-announcing the same slug refreshes TTL. Returns
  `announce_text` — a speech-shaped string the caller relays to Ed.
- **`resolve(user_text, source)`** is the hook the chat/voice handlers
  call on every non-slash user turn. Classifies the reply, finds the
  newest matching pending in the channel, applies via
  `chloe_proposals.apply_proposal(slug)` (Tier-1 path with all safety
  rails) on "yes" or cancels on "no". Returns `{action, slug, result,
  reply_text}` or `None` if no resolution happened.
- **`pending(source=None)`** lists active slots. **`cancel(slug)`**
  drops one or all.
- CLI: `python chloe_pending_confirms.py announce|resolve|list|cancel`.

**Wiring into the chat path** (`brain_wiring.try_handle_brain_command`):
Inserted right after the NL-alias rewrite and BEFORE the auto-fact
extractor. If `msg` is non-slash and `resolve(msg, source="chat")`
returns a resolution, return its `reply_text` directly — short-
circuits all downstream dispatch.

**Wiring into the voice path** (`jarvis._ask_groq`): Inserted right
after the ack-gate and BEFORE the forced-Brave check. Same shape:
`resolve(user_text, source="voice")`. Pushes the resolved reply
through `_push_history("assistant", reply)` so the next-greeting context
sees it.

**Source separation.** A voice-announced pending can ONLY be resolved
by a voice reply. Chat-announced by chat. `"any"` matches both. Avoids
cross-channel ambiguity (Ed says "yes" out loud while a chat-pending is
also active → only the voice-channel pending resolves, if there is
one).

**New slashes / MCP tools:**
- `/pending_confirms` — list active slots (handler signature also
  supports `cancel <slug>` and `cancel-all`).
- `mcp__chloe__propose_and_announce(target, kind, rationale, body,
  test_plan, slug, title, source, ttl_minutes, summary)` — Cowork-side
  one-shot: drafts the proposal AND registers a pending. Returns the
  speech-shaped announce text for the caller to relay verbatim.
- `mcp__chloe__pending_confirms()` — read-only inventory.
- `mcp__chloe__cancel_pending(slug)` — kill switch.

**Verification.** 48/48 sandbox tests pass in
`outputs/test_stage3_pending.py` (12 test groups: phrase
classification incl. trailing-punct + UPPERCASE + long-sentence false-
positive guard; announce / pending list; source separation; resolve via
correct channel; cancel via "no"; ambiguous reply leaves pending in
place; TTL expiry; `source="any"` matches both channels; cancel one +
cancel all; re-announce refreshes TTL; validation of bad source/ttl/
empty slug; newest-first resolution when multiple pendings exist).
`verify_proposals.py` extended with a Stage-6.5 module-import +
contract section + plain-`"yes"` chat-path inline hook test.

**Trust ladder, restated.** Stage 3 (this ship) gates each apply on
Ed's explicit yes. Stage 4 (autonomous watchdog) stays at design-only
in `specs/self_mod_stage4_watchdog.md` until Stage 3 logs ≥10
successful voice/chat-confirm cycles AND Stage 2/3 combined run 2+
months incident-free. The gate criteria from `specs/` apply verbatim.

Files: `chloe_pending_confirms.py` (new), `brain_wiring.py` (1 new
handler + 1 dispatch entry + 1 inline resolve() hook + 1 help-text
line), `jarvis.py` (1 inline resolve() hook in `_ask_groq`),
`chloe_mcp_server.py` (3 new MCP tools), `verify_proposals.py`
(extended), `outputs/test_stage3_pending.py` (new contract test).

## [2026-05-19 late evening] chloe_capabilities.py / chloe_proposals.py / brain_wiring.py / chloe_mcp_server.py | Stage 1 self-analysis + Stage 2 confirm-token Tier-2 self-modification

Two stages of the self-mod ladder layered on top of this morning's Tier-1
ship. Stage 3 (voice-confirm) and Stage 4 (watchdog) deliberately
deferred — design notes shipped in `specs/`.

**Stage 1: self-analysis surface** — new module `chloe_capabilities.py`
(~500 lines, stdlib-only). Public API:
- `summary()` returns dict of `{slashes, mcp_tools, jobs, env_knobs,
  modules, computed_at}` — all ast-derived, accurate by construction.
- `describe_module(name)` ast-introspects any jarvis module: returns
  docstring + imports + UPPER_CASE constants + function signatures with
  first-line docstrings + classes with method lists.
- `list_slash_commands()` scans `brain_wiring.try_handle_brain_command`
  for `msg.startswith("/foo ")` / `msg == "/foo"` / `msg in (...)`
  patterns, resolves each to its handler function, pulls the handler's
  first docstring line as a one-line summary.
- `list_mcp_tools()` walks `chloe_mcp_server.py` for `@mcp.tool()`
  decorated functions, returns signature + first-line docstring.
- `list_jobs()` lazy-imports `chloe_jobs.state()` (the same data the HUD
  CH03 panel renders).
- `list_env_knobs()` regex-scans every `.py` in jarvis/ for
  `os.environ.get("KEY", "default")` patterns, dedupes by key, returns
  default values + source files.
- `list_modules()` enumerates non-scaffolding `.py` files with line
  counts + function counts + docstring summaries.
- `format_summary_markdown()` + `format_module_markdown()` for
  chat-friendly rendering.
- CLI: `python chloe_capabilities.py summary|describe|slashes|tools|
  jobs|env|modules`.

**New slashes in brain_wiring.py:**
- `/capabilities [section] [--json]` — full self-analysis report or one
  section (slashes / tools / jobs / env / modules). Section name is
  positional, `--json` flips to structured payload (for MCP-side use).
- `/explain <module> [--json]` — ast introspection of one module.
- `/brain` help text gained both lines.

**New MCP tools in chloe_mcp_server.py:**
- `mcp__chloe__capabilities(section="")` — wraps the same summary +
  section paths for Cowork.
- `mcp__chloe__explain(module)` — wraps `describe_module()` for Cowork.

**Stage 2: Tier-2 confirm-token self-modification.** Mint a token via
`/issue_apply_token`; Chloe (via MCP) or any Cowork job can then apply
proposals on its own up to the cap. All Tier-1 safety rails STILL fire —
the token only relaxes the "human types the slash at apply time" gate.

`chloe_proposals.py` additions:
- `issue_token(applies=1, minutes=30)` — mints a 32-char hex token.
  Caps: 1-5 applies, 1-120 minutes. Returns
  `{ok, token, applies, expires_at, expires_iso}`.
- `apply_proposal_with_token(slug, token, dry_run=False)` — Tier-2
  entry point. `dry_run` does NOT consume the token. Real apply consumes
  one slot atomically. All Tier-1 safety rails fire after token
  validation.
- `_consume_token(token)` uses `secrets.compare_digest` for constant-
  time comparison. Pruning sweeps expired/exhausted tokens before each
  match.
- `revoke_tokens()` drops every active token (manual kill switch).
- `list_tokens()` returns masked status (`first4…last4` token IDs only).
- `session_state()` extended with `tokens` field.
- Module-level `_ACTIVE_TOKENS` (list of dicts) + `_TOKEN_LOCK`
  (threading.Lock) for concurrent MCP access safety.

**New slash:** `/issue_apply_token` with `--applies N`, `--minutes M`,
`--status` (show active tokens masked), `--revoke` flags. Issuing prints
the raw token once; Ed copies it to the caller that will use it.

**New MCP tools:**
- `mcp__chloe__apply_self_patch(slug, token, dry_run=False)` — Tier-2
  apply entry from Cowork side. Documentation makes clear that all
  Tier-1 rails still fire and that the token relaxes ONLY the slash-
  retyping gate.
- `mcp__chloe__token_status()` — read-only diagnostic, masked tokens +
  session counter.

**Verification.** 36/36 Tier-1 sandbox tests still pass
(`outputs/test_proposals.py`). 26/26 Tier-2 token-contract tests pass
(`outputs/test_tier2_tokens.py` — caps, format, list+mask, consume+
decrement, wrong-token rejection, revoke, expiry). `verify_proposals.py`
extended to cover Stage 1+2 — runs on Windows via
`verify_proposals.bat`. Bash mount staleness (lesson #2) blocked live
runtime tests of the production `chloe_proposals.py` module from the
sandbox (the file is fresh on disk but the Linux mount lags) —
`verify_proposals.bat` next boot is the authoritative check.

**Trust ladder, restated.** Tier 2 (this ship) is gated by tokens that
Ed mints + the existing Tier-1 safety rails. Tier 3 (autonomous self-
mod with watchdog) and the voice/chat-confirm flavor of Stage 3 are
documented in `specs/self_mod_stage3_voice_confirm.md` and
`specs/self_mod_stage4_watchdog.md` but NOT built — they unlock after
Tier 2 logs ≥10 successful token-gated cycles + 2 incident-free months.

Files: `chloe_capabilities.py` (new), `chloe_proposals.py` (Tier-2
additions ~150 lines), `brain_wiring.py` (5 new handler functions + 5
new dispatch entries + help-text additions), `chloe_mcp_server.py` (4
new MCP tools — capabilities, explain, apply_self_patch, token_status),
`verify_proposals.py` (extended), `outputs/test_tier2_tokens.py`
(contract test).

## [2026-05-19 evening] chloe_proposals.py / brain_wiring.py | Tier-1 self-modification: code-proposal pipeline

Chloe (or any Cowork job, or Claude in Cowork via MCP) can now propose
code changes to her own source as markdown files, and Ed can apply them
with a slash command. Closes the #1 NEXT-SESSION PRIORITY from the
handoff. Tier 2 (gated direct self-patch via MCP) and Tier 3
(autonomous self-mod with watchdog rollback) remain deliberately
unbuilt.

New module `chloe_proposals.py` (~600 lines, no external deps):
- `apply_proposal(slug, dry_run)` / `revert_proposal(slug)` — the
  core pipeline. Resolves target path, runs safety checks, computes
  the proposed final body (from `kind: full` whole-file replacement
  OR `kind: diff` unified-diff hunks), `ast.parse`s the result if the
  target is `.py`, writes a timestamped `.bak.proposal_<slug>_<stamp>`
  backup, then commits. Stamps the proposal frontmatter
  `status: applied` + `applied_at` + `backup_path` so re-apply is
  refused and revert is one-shot.
- `list_proposals(status=None)` / `load_proposal(slug)` / `session_state()`.
- `create_proposal(target, kind, rationale, body, test_plan, slug,
  title, rollback)` helper for proposal authors (Friday meta-review,
  Cowork jobs, etc.). Writes a properly-shaped
  `proposals/code_<YYYY-MM-DD>_<slug>.md` with frontmatter +
  `## Rationale` / `## Diff` (or `## Full File`) / `## Test plan` /
  `## Rollback` sections.
- Minimal hand-rolled unified-diff applier. Strict context matching:
  every `-`/' ' context line in a hunk must equal the original at the
  indicated line range, or the apply is refused. Avoids the
  `unidiff`/`patch` dependency.
- Safety rails:
  - Target must resolve under `jarvis/` or `BRAIN_ROOT` (path
    whitelist). Absolute paths outside both are refused.
  - Path-segment blocklist: `__pycache__`, `.bak.`, `.git`, `venv`,
    `venv_py314`, `dist`, `node_modules`, `.vscode`, `.idea`,
    `secrets` (so `C:\Chloe\secrets\` is never reachable via
    proposals even though it's outside the whitelist anyway).
  - `ast.parse` on the proposed final body for `.py` targets.
  - Stamp in backup filename is `YYYYMMDD_HHMMSS` (not date-only,
    lesson #13 applied).
  - Module-level `MAX_APPLIES_PER_SESSION = 5` cap. Forces a restart
    between batches so behaviour changes get verified before more
    queue up.
- CLI: `python chloe_proposals.py list|show|apply|revert <slug>` for
  ops use without going through Chloe.

New slashes in `brain_wiring.py::try_handle_brain_command`:
- `/apply_proposal <slug>` — applies. `--dry-run` previews without
  writing. `--list` (or `--status=applied|reverted|pending`) shows
  the proposal inventory + remaining session apply slots.
- `/revert_proposal <slug>` — restores the target from its
  `.bak.proposal_<slug>_<stamp>` backup, stamps proposal as
  `reverted`, frees a session slot.
- `/brain` help text updated.

Verification:
- 36/36 sandbox tests pass in `outputs/test_proposals.py`: full/diff
  kinds, dry-run vs commit, backup roundtrip, AST refusal, outside-
  whitelist refusal, venv-pattern refusal, missing-slug, no-op,
  list_proposals filters, session-counter cap.
- `verify_proposals.bat` / `verify_proposals.py` for Windows-side
  end-to-end verification including the slash dispatch through
  `try_handle_brain_command`.

Files: `chloe_proposals.py` (new), `brain_wiring.py` (2x splices —
two new handler functions before `try_handle_brain_command`, two
dispatch entries + 2 help-text lines inside it), `verify_proposals.py`
(new), `verify_proposals.bat` (new).

## [2026-05-18 evening] brain_http / brain_graph / wiki_watcher | brain UI v3 backend — drop-in ingest + SSE event stream + v2 graph schema

Lands the backend that the v3 `brain-graph.html` regen will consume. Five
files touched, one new (`event_bus.py`). Frontend regen happens in a
separate Claude.ai single-file artifact session driven by the rewritten
`BRAIN_GRAPH_DESIGN_PROMPT.md`.

New file:
- `event_bus.py` (~100 lines): in-process pub/sub. `subscribe(maxsize)` →
  Queue, `publish(event)` fans out, drops oldest on full subscriber so
  slow consumers don't block fresh events. `last_event()` for diagnostics.

Extended `brain_graph.compute_graph` (v2 schema):
- Per-node: `mtime`, `body_size`, `source_type` (heuristic from
  frontmatter + path: chloe_generated/brave/dropped_in/finance/daily/
  episodic/external/ghost), `orphan` (real AND degree ≤ 1), `fm_type`
  (frontmatter `type:` if present).
- New args: `include_facts=True` mounts `facts/*.md` under a synthetic
  `facts/` prefix; `since_ts=<epoch>` filters nodes by mtime (powers the
  time slider).
- Stats now include: `orphan_count`, `source_counts`, `recent_changes`
  (top 10 by mtime), `embed_coverage` (from WikiEmbeddingStore — best
  effort), `last_event` (from event_bus), `computed_at`.

New `brain_http` routes:
- `POST /api/brain/ingest` — multipart `file` or JSON `{url, text, title,
  dry_run}`. Writes to `raw/dropin_<slug>_<stamp>.md` with frontmatter,
  calls `BRAIN.ingest()`, returns result + top-6 cosine-similar existing
  pages (best-effort via WikiEmbeddingStore.search), publishes `ingested`
  event on bus.
- `DELETE /api/brain/ingest?slug=<slug>` — removes
  `wiki/sources/<slug>.md` + matching `raw/dropin_*<slug>*.md`. Entity/
  concept pages that mention the slug are listed but NOT reverted (merge-
  updates not safely reversible) — surfaced in response.
- `GET /api/brain/events` — Server-Sent Events. Heartbeat every 15s.
  Streams `hello/upserted/deleted/ingested` from event_bus.
- `GET /api/brain/stats` — cheap stats-only payload (no nodes/edges) for
  the brain-stats pane polling fallback.
- `OPTIONS` handler added for CORS preflight.

`wiki_watcher._apply` now publishes `upserted`/`deleted` events on
event_bus for every filesystem-driven change, so the brain-graph UI's
"recently edited" pulse fires in real time when Obsidian saves a page.

Rewrote `BRAIN_GRAPH_DESIGN_PROMPT.md` as v3 — single-file Claude.ai
artifact spec covering the 4 pending bake-ins (FOUT fix, marked side
panel, status pill parity, sphere/depth polish) + 16 new features:
minimap, LOD clustering, type filter chips, 2D/3D toggle, search-to-
focus, layout toggle, orphan highlight, brain-stats pane, drop-in panel
with pre-commit preview, SSE recently-edited pulse, edit-source badge,
watcher ticker, time slider, provenance trail, cluster chat via
parent.postMessage, facts-as-nodes opt-in.

Smoke test was skipped under the bash sandbox — mount staleness (lesson
#2) showed pre-edit `brain_graph.py` / `brain_http.py` mtimes 8+ hours
behind the Windows view confirmed by Read. `event_bus` self-test
(separate new file) passed. End-to-end verification path: restart Chloe,
`curl http://localhost:6790/api/brain/graph | jq '.stats'` should show
the new keys; `curl -N http://localhost:6790/api/brain/events` should
print `event: hello` immediately and stream `upserted` events when a
wiki file is touched in Obsidian.

---

Append-only record of behavioral changes. Each entry uses this format:

```
## [YYYY-MM-DD HH:MM] <area> | <one-line summary>

<optional details, file paths, splice script name, etc.>
```

Areas: `jarvis`, `brain_wiring`, `brain`, `chloe_memory`, `chloe_about`,
`mcp_server`, `scheduled_tasks`, `persona`, `wiki`, `integrations`,
`config`, `docs`.

Convention: every splice script + every meaningful behavior change appends
an entry. The Friday meta-review and the morning brief can scan this file
for recent activity without grepping git or guessing from `.bak` files.

To see recent activity:
```
grep "^## \[" CHLOE_CHANGELOG.md | tail -10
```

To see what changed in a given area:
```
grep "^## \[.*\] brain_wiring " CHLOE_CHANGELOG.md
```

---

## [2026-05-18] multi | Tier 2 push: voice.md wiring, finance, NL aliases, greeting, autopilot, lights groups, mic recovery, /wiki_synth, /wiki_interview, MCP v2.3

A trivial→Tier-2 sweep across the backlog. Each item below shipped in one
pass; brain-graph items (FOUT, marked rendering, polish, status pill
alignment) are blocked on regenerating the bundled artifact from
`BRAIN_GRAPH_DESIGN_PROMPT.md` — not editable as files.

**brain / queue_processor — DRAFT voice plumbing.** `brain.py` now has
`voice_only()` reading `facts/voice.md` directly, and `facts_only()` /
`_full_context()` exclude voice.md via a shared `_FACTS_EXCLUDE` set
(was: voice.md got concatenated into `{facts}` slot, contradicting its
own "don't quote from this file" header). `queue_processor.PROMPT_DRAFT`
gains a separate `{voice}` slot fed from `voice_only()` with explicit
"style only — never quote from this section" guardrail.

**Foundational finance ingest queue.** Five `RESEARCH-*.md` files
dropped into `brain/queue/`: options_greeks, theta_decay_mechanics,
iv_crush, wheel_strategy_full, moneyness_itm_otm_atm. Every-2h
processor will surface them as `generated/<date>/research-*.md` over
the next cycle. Ties to Ed's actual SLV covered-call position.

**brain_wiring — NL aliases for /wiki_*.** `_maybe_wiki_nl_alias`
translates conservative anchors ("write a wiki about X", "research X
for the wiki", "search the wiki for X", "look up X in the wiki",
"what's in the wiki about X") to the slash form before the rest of
the dispatcher fires. 14/14 fixtures pass; 4 false-positive guards
("the wiki has my notes", etc.) all correctly return None.

**jarvis — brain-driven greeting.** `_latest_context_focus()` parses
the newest `episodic/CONTEXT-*.md` for the first non-trivial line under
`## Suggested Focus` (falls back to `## Open Loops`), second-person
rewrites the prose ("Edward should ... his ..." → "you should ... your
..."), caps at 140 chars, and tacks it onto the canned greeting:
"Good morning, Ed. From your latest context — <line>". Empty-CONTEXT
sentinels ("no active project", "no open loops", etc.) fall through to
the original `_GREETING_POOL`.

**brain_wiring — /summarize_old auto-cadence (pillar 4 follow-up).**
Daemon thread `chloe-summarize-auto` runs once per hour ± 5min jitter,
checks `unsummarized_count() >= CHLOE_SUMMARIZE_THRESHOLD`, fires
`handle_summarize_old("")` under a single-in-flight lock. Opt-in via
`CHLOE_SUMMARIZE_AUTO=1` (default off until soak-tested). Boot hook in
jarvis.py near the other warm daemons.

**scheduled_tasks — position-aware finance prompts.** Updated
`chloe-daily-finance-ingest` SKILL.md to parse "Strategies you're
actively running" from the watchlist and add a per-ticker "Strategy
implications" subsection covering distance-to-strike, IV percentile
shifts, theta window vs upcoming events, macro driver state. Hard
constraint: do NOT invent strikes/expirations not in the watchlist —
prompt Ed to populate them instead.

**brain_wiring — /wiki_synth slash.** Companion to /wiki_write. Pulls
relevant brain pages via `queue_processor.gather_relevant_pages`,
synthesizes a wiki page via `_heavy_call` (no web), ingests through the
same raw → entity/concept pipeline. Use for "what does the brain
already know about X" recap pages. Surfaces explicit "## Gaps the brain
doesn't cover" so /wiki_write can fill them next.

**brain_wiring — /wiki_interview + /wiki_interview_done slashes.**
Two-step interactive Q&A → wiki page flow. `/wiki_interview <topic>`
generates 5 open-ended questions via heavy LLM, stores topic + questions
in module-level state, formats numbered list. Ed answers all 5 in one
chat turn, then runs `/wiki_interview_done` which pulls the most-recent
non-slash user turn from `chloe_memory` (since interview start ts),
synthesizes a page treating Ed's answers as primary source, ingests.
Single active interview at a time (single-user system).

**lights — groups (spec T1-T5).** `_load_config` validates an optional
`groups: {name: [members]}` map (drops name-collision/dangling refs
with stderr warning rather than refusing startup — daily-driver
foot-gun avoided). `_resolve_targets` checks exact bulb match → exact
group match → substring; cleaned groups guaranteed not to collide.
`get_state_snapshot` includes `groups` in WS payload. `_format_status`
and `--list` show groups separately. Error message generalized:
"no such bulb or group". 10/10 resolver fixtures pass.
T6 (HUD CH02 Groups row in hud.html) deferred — hud.html is past
Edit-tool threshold; needs anchor-based splice.

**jarvis — mic flakiness partial fix.** Voice loop tracks
`consecutive_failures` per device. On the 2nd consecutive failure (and
once per session), swaps `device` to `CHLOE_MIC_FALLBACK_DEVICE`
(default 1 — MME) and broadcasts an audible "Mic is misbehaving —
falling back to M M E device. If audio still drops, unplug and replug
the Samson." Disable with `CHLOE_MIC_AUTO_FALLBACK=0`. Fallback failures
fall through to the existing exponential backoff (now with a
"(fallback device also failed)" hint in the log).

**chloe_mcp_server — v2.3 (23 tools, +5).** Added `chat(message,
no_memory=False)` — drives the brain slash handler → lights NL → Groq
llama-3.3-70b fallback chain without TTS/streaming/WS scaffolding, for
closed-loop Cowork→Chloe testing. Added `wallet_balance()` and
`wallet_history(limit)` (read-only). Added `queue_status()` for
pending+recent processor output. Added `see(question)` wrapping
`/ask` vision pipeline. About-menu version bumped, categories updated.

**Blocked items (defer to a brain-graph artifact regen session):**
brain graph FOUT, marked rendering in side panel, status pill
consistency with main HUD, brain graph polish from
`BRAIN_GRAPH_DESIGN_PROMPT.md`. All four need `brain-graph.html`
regenerated as a Claude.ai single-file artifact — the body opacity-0
CSS rule is base64/gzipped inside the bundler template.

**Tier 3 items (not in this push):** voice prosody, knowledge
distillation engine, voice biometrics (torch blocker), Home Assistant,
Calendar/Gmail deep, Plaid read-only, native iPhone client, project
layer. Each is multi-hour to multi-week per the handoff — separate
sessions.

## [2026-05-17 22:50] jarvis | Voice-path inline streaming TTS — last Tier 1 item

Voice path was non-streaming: `_ask_groq` waited for the full reply
text, then `_speak(reply)` synthesized it sentence-by-sentence. Worst
case latency = full LLM gen (~10s on qwen2.5:32b for a long reply) +
first-sentence synth (~0.5s). New path: stream Ollama tokens, parse
sentence boundaries on the fly, feed each completed sentence into
the Kokoro consumer immediately. First-audio latency drops to
first-sentence-stream + first-sentence-synth (~2-3s).

**New building blocks:**

- `_iter_ollama_sentences(messages, max_tokens)` — sync generator
  using `requests.post(stream=True)`. Accumulates content deltas,
  yields complete sentences as `_SENT_BOUNDARY_RE` matches, yields
  any tail buffer at end. Tool-call mid-stream → aborts (caller
  falls back to non-streaming `_ollama_chat` which handles tools).
- `_speak_kokoro_stream(sentence_iter)` — Kokoro consumer with
  iterator-fed producer. Same producer-consumer + barge-in pattern
  as `_speak_kokoro`, just reads from an iterator instead of a
  pre-split list. Per-sentence tone-tag parsing preserved. Returns
  full text spoken so the caller can push to history.
- `_voice_stream_ollama_and_speak(messages, max_tokens)` —
  orchestrator. Returns full text on success, None when streaming
  isn't viable (no Kokoro, no Ollama, tool call, empty stream).

**Wiring:**

- `_ask_groq` ollama branch tries the streaming path first when
  `CHLOE_VOICE_STREAMING=1`, falls through to `_ollama_chat` on
  decline. Sets `_spoke_inline` (a module-level `threading.Event`)
  on streaming success.
- `_ask_groq` clears `_spoke_inline` at the top of every call so
  prior-turn state can't leak.
- Hedge fallback: if streaming spoke a hedged reply inline, the
  user already heard it. Brave correction runs as usual; we clear
  `_spoke_inline` so the standard `_speak(brave_reply)` runs and
  the user hears the correction.
- Both voice call sites (`_handle_ptt` @ 3251, voice loop @ 3593)
  guard `_speak(reply)` with `if not _spoke_inline.is_set()` so
  the same reply doesn't get re-spoken.

**Feature flag:** `CHLOE_VOICE_STREAMING` env var (default 0). Opt-in
to test before flipping default. Only the Kokoro engine has a true
streaming path; edge-tts and ElevenLabs fall back to buffer-then-speak
inside `_speak_kokoro_stream` (no regression vs. baseline).

Restart needed.

## [2026-05-17 22:30] jarvis + hud | Voice/text sync polish

Three changes targeting time-to-first-audio and inter-chunk gaps in
streaming TTS.

**1. Clause-split for first 1-2 sentences** (`jarvis._split_sentences_for_tts`).
New `_clause_split_first` helper plus `_CLAUSE_BOUNDARY_RE` /
`_CLAUSE_SPLIT_MIN_CHARS` (60) / `_CLAUSE_SPLIT_MIN_FRAG` (24)
constants. Long opening sentences (>60 chars) get split on the first
viable comma if the head fragment is ≥24 chars. Cap: only the first
2 sentences are eligible — mid-utterance commas should still feel
like natural single-breath phrases. Comma stays on the head so
playback sounds like a soft pause, not a hard cut.

**2. Kokoro pre-warm at startup** (`jarvis._warm_kokoro_model`).
Daemon thread spawned next to the Ollama warm-up thread. Calls
`_get_kokoro()` (idempotent) so the ~1-3s ONNX cold-load is paid
off the boot path rather than mid-reply. Skipped silently if
USE_KOKORO=0 or deps missing.

**3. HUD pre-buffer mid-stream chunks** (`hud.html` TTS module).
Used to push raw `{b64}` into `streamQueue`, decode-then-play only
when the previous chunk's `onended` fired (~10-150ms decode gap each
time). Now pushes `{bufferPromise: _preDecode(b64)}` — decode starts
the instant the chunk arrives, so by the time onended fires the
AudioBuffer is ready. New `_preDecode` + `_playDecoded` helpers
split decode from play. `_playBuffer` rewritten to delegate to
`_playDecoded` after its own decode (single-shot path unchanged).
First chunk (`chunk_id === 0`) is unaffected — nothing to chain
over for it.

Restart needed for jarvis side. HUD picks up on next page reload.

## [2026-05-17 22:15] jarvis | State-broadcast pattern lifted to edge-tts + ElevenLabs

`_speak_kokoro` already broadcasts "speaking" on first audio onset
(post 2026-05-17 voice-sync patch). The other two engines never
broadcast "speaking" at all — when USE_KOKORO=0 or Kokoro fell
through, the HUD pulse animation didn't fire during audio playback.

Lifted the exact pattern to both:

- `_speak_edge_tts` — `_spoke_at_least_once` guard inside the consumer
  loop. First audio item out of the queue → `broadcast_sync("speaking")`
  + flag flipped. `finally`: clear `_speaking` + if flag, broadcast
  "idle". Mirrors Kokoro's per-sentence streaming consumer.
- `_speak_elevenlabs` — non-streaming, audio is fully ready before
  playback. Broadcast "speaking" right before `_play_audio_with_barge_in`,
  flag flipped, "idle" in finally guarded by the flag.

Three engines now have identical state-broadcast semantics. Net effect:
HUD pulse + orb animation align with actual audio onset regardless of
which engine renders, and the existing call-site "idle" broadcasts
remain idempotent.

Restart needed.

## [2026-05-17 22:00] jarvis | Groq stream-iteration 413 — clean log path

Outer `except` in `handle_chat` now checks `_is_too_large_error(e)`
before dumping a traceback. Stream-iteration 413s (cumulative payload
swell mid-stream, or max-tokens overflow on compound tool data) now
log a single line: `[chloe] stream 413 on <model> (cumulative payload
too large mid-stream) — falling back to Ollama`. Functional behavior
unchanged (still falls to Ollama); just kills the log noise. The
mid-stream retry case (recreate with aggressive trim if no delta
streamed yet) is deferred — partial-state handling adds complexity
the handoff didn't ask for.

## [2026-05-17 21:55] chloe_about | Tonal Awareness — no-announce rule tightened

Drift evidence (today's `persona_drift_2026-05-17.md` + handoff backlog)
shows the abstract "DO NOT announce that you've detected a mood" rule
still leaks ("you sound a bit off", "you seem tired"). Replaced with
explicit show-don't-tell framing:

- 9 banned openers listed verbatim (the actual phrasings that have
  slipped through).
- 5 read→behavior mappings (tense/excited/tired/sad/frustrated) with
  explicit "don't say X" callouts.
- 3 before/after pairs.
- Exception case explicitly bounded to invited reads only.

Restart needed (persona file loaded at startup).

## [2026-05-17 21:30] mcp_server + scheduled_tasks | Persona-evolution duplicate-anchor fix

First real run of `chloe-weekly-persona-evolution` (today) proposed
Cornfield Chase (Zimmer) + Yuki Kajiura as new anchors despite both
already being in `chloe_about.md`'s `## Specific favorites`. Root
cause: the task can't reach `chloe_about.md` directly — file lives at
`C:\Users\eleew\Documents\jarvis\chloe_about.md`, outside the brain
dir, so `brain_read` skips it. The skill prompt told the task to
"reason about what's there" via recall + prior proposals, which is
weak.

Fix shipped:

- New MCP tool `mcp__chloe__persona_read()` — returns body of
  chloe_about.md with meta-header stripped. Wraps the existing
  `ChloeMemory.about_body()` accessor.
- Connector bumped to v2.2 (18 tools). `about()` menu updated:
  "Read — conversation + memory (4)" now includes persona_read.
- `chloe-weekly-persona-evolution` SKILL.md updated via
  `update_scheduled_task`: persona_read() listed as the FIRST tool
  with **Call this FIRST** emphasis, process step 1 made explicit,
  every proposal now requires an `**Already in persona?**` line as
  a hard rule. Quality bar: missing that line = invalid proposal.

Net result: next run will catch existing anchors and either drop
the proposal or label it BRIDGE (adds context to existing anchor).
Today's output (`proposals/persona_2026-05-17.md`) stays as-is for
Ed to manually triage — proposals 3 (Kelly criterion) + 4 (Toy
Story) are net-new and likely worth merging; 1 + 2 are duplicates
to skip.

## [2026-05-17 21:00] chloe_memory + brain_wiring | Pillar 4 SHIPPED — conversation summarization

Primitive + manual trigger. Auto-cadence deferred to follow-up.

**chloe_memory.py:**
- `summarized` column added to `turns` (idempotent migration, default 0).
- `unsummarized_count()`, `oldest_unsummarized_turns(limit)`,
  `mark_summarized(ids)` methods.
- `search_turns()` SQL extended with `AND COALESCE(summarized, 0) = 0`
  so rolled-up turns don't double-surface in recall (the compact
  summary picks them up via wiki recall instead).

**brain_wiring.py:**
- `/summarize_old [N|--dry-run]` slash command. Pulls oldest
  unsummarized turns (threshold gate: `CHLOE_SUMMARIZE_THRESHOLD`,
  default 50; batch size: `CHLOE_SUMMARIZE_BATCH`, default 30), sends
  to `chloe_llm_call(prompt, "heavy")` with a compact-narrative
  prompt, writes `wiki/episodic/conversation_summary_<date>_<idstart>-<idend>.md`
  with frontmatter, marks the source turns `summarized=1`.
- wiki_watcher auto-embeds the new episodic page within ~2s — future
  semantic recall surfaces the summary as a wiki hit, replacing the
  scattered raw turns.
- `--dry-run` shows the prompt size + first 3 transcript lines without
  writing or marking, for safe testing.

**Auto-cadence:** deliberately deferred. Manual `/summarize_old` first
to validate output quality. Future options: in-process daemon thread
in jarvis.py, or an MCP tool + Cowork scheduled task. Either is a
follow-up splice once the primitive is proven.

Restart required (chloe_memory + brain_wiring imported at startup).

## [2026-05-17 20:45] brain | Pillar 2 SHIPPED — auto-fact extraction tightened

`brain.py::fact_extract_and_add` rewritten. Three changes:

1. **Few-shot examples** baked into the prompt — 7 worked examples
   demonstrating short, specific, stopword-free `name` choices
   ("father name", "favorite composer", "SLV covered call strategy",
   "networking opinion", "profession", "birthday", "region").
2. **Heavy mode past length threshold** — `CHLOE_FACT_HEAVY_THRESHOLD`
   env var (default 200 chars). Long statements escalate to heavy mode
   for slug-pick quality. Short statements still use light/Ollama.
3. **Ed-fact category namespacing** — LLM now returns a `category`
   field (biographical / preference / opinion / other). When category
   is one of the first three, slug becomes `ed_<category>_<name>` so
   `facts/` clusters by kind. Falls back to original flat slug on
   "other" or missing.

Added `import os` to brain.py. Both `/fact` and the daemon-threaded
auto-fact worker go through this same function — both paths benefit.

Restart required (brain module imported at startup).

## [2026-05-17 20:30] scheduled_tasks | Pillar 3 SHIPPED — weekly persona-evolution proposal pass

`chloe-weekly-persona-evolution` at cron `0 6 * * 0` (Sundays 06:00,
after persona-drift at 05:00). Reads past 7 days of recall + recent
CONTEXT files + prior persona proposals + current facts. Proposes
ADDITIONS to `chloe_about.md` — new favorites, tonal patterns,
knowledge anchors — as drop-in-ready text with ≥2 recall-quote
evidence per item. Writes to `proposals/persona_<date>.md`. Distinct
from persona-drift (which audits existing rules). Approval bar: queue
proposal, never silent merge. Closes pillar 3 of the memory-autopilot
sprint (set 2026-05-17).

## [2026-05-17 19:15] brain_wiring | Startup-blocking fix — added `from pathlib import Path` to module imports

The `_COWORK_SCHEDULED_DIR` line in the `/status` insertion used `Path`
at module level but `Path` was only lazy-imported inside `handle_status()`.
Result: `NameError: name 'Path' is not defined` at line 775 during
`import jarvis`, which made the voice thread die before the greeting
fired (boot-signal race + splash hang). Caught by reading
`logs/backend.log` line 23. Added `from pathlib import Path` at module
imports. Restart needed.

**Lesson:** when adding module-level constants, check ALL identifiers
used resolve at import time (not just function-local imports). Lazy
imports inside functions don't help module-level statements.

## [2026-05-17 19:00] scheduled_tasks | Cowork → Chloe fact-extraction loop

`chloe-daily-cowork-fact-extract` at cron `30 22 * * *` (22:30 nightly,
before the journal stub at 23:00). Reads today's Cowork session
transcripts via session_info MCP, scans Ed's user messages (not
Claude's replies) for fact-shaped statements, dedups against current
facts.md, caps at 5 candidates/day, writes to
`proposals/facts_from_cowork_<today>.md`. Closes the biggest-leverage
context gap: previously anything Ed told Claude in Cowork stayed
invisible to Chloe's auto-fact extractor.

## [2026-05-17 18:55] scheduled_tasks | Weekly persona drift audit task

`chloe-weekly-persona-drift` at cron `0 5 * * 0` (Sundays 05:00, after
the autonomous audit). Compares recent assistant turns against
chloe_about.md banned phrasings + required behaviors. Surfaces drift
(tail-pads, banned phrases slipping through, clinical-list output,
tonal-read leakage) and writes to `proposals/persona_drift_<date>.md`.
Different from pillar 3 (which proposes additions) — this measures
whether existing rules are being honored. Pure audit, no automatic
edits.

## [2026-05-17 18:50] scheduled_tasks | Morning brief extended with fact-proposal queue

`chloe-daily-morning-brief` prompt updated. Now scans recall + activity
for fact-shaped patterns (biographical, preferences, recurring topics,
new relationships/roles). When ≥1 solid candidate exists, writes
`proposals/facts_<today>.md` alongside the brief, capped at 5/day,
deduped against current facts.md. Ed approves by telling Claude in
Cowork or by direct mcp__chloe__add_fact call. Closes the brief's
one-way nature.

## [2026-05-17 18:40] scheduled_tasks | Daily journal stub task

`chloe-daily-journal-stub` at cron `0 23 * * *` (23:00 nightly). If
Ed didn't write `wiki/daily/<today>.md`, synthesize a third-person
stub from observable activity (recall + web_history + morning brief +
finance news + generated/ + new facts). Writes via brain_write
(refuses overwrites — race-safe). Kills the silent-dark daily_ingest
failure mode.

## [2026-05-17 18:35] scheduled_tasks | Weekly autonomous audit task

`chloe-weekly-autonomous-audit` at cron `0 4 * * 0` (Sundays 04:00,
after backup). Cross-system audit: lists each Cowork + Windows pipeline,
classifies HEALTHY/WARN/DARK by last expected output age, surfaces
anomalies + recommended actions. Output: `brain/autonomous_status_<date>.md`.
Catches "queue empty for a week" / "friday review wasn't scheduled"
classes of mystery before Ed notices them.

## [2026-05-17 18:30] brain_wiring | `/status` slash command added

Self-awareness snapshot: queue depth + most-recent queue file age, last
lint, recent ingests, wiki page count, today's web searches, Cowork
scheduled tasks (with cron expressions), TTS engine, Ollama reachability,
mode. Bounded — no LLM calls. Wrapped section-by-section in try/except
so a broken source doesn't blank the whole status. Aliased to `/health`.
Requires Chloe restart.

## [2026-05-17 18:20] scheduled_tasks + integrations | Weekly backup pipeline

`backup_chloe.py` script in jarvis/ — copies brain + secrets + SQLite +
persona files + facts to `C:\\Users\\eleew\\OneDrive\\ChloeBackups\\<date>\\`
with 4-week rotation. Standalone (pure Python, no shell deps). Companion
Cowork scheduled task `chloe-weekly-backup` (cron `0 3 * * 0`, Sundays
03:00) invokes the script via bash.

## [2026-05-17 18:10] mcp_server | `about()` menu tool added — 17 tools total

New read-only tool that returns the categorized tool menu in markdown.
Helps fresh Cowork sessions (and other MCP clients) bootstrap without
schema-discovering all 17 tools. Cowork restart needed to pick up.

## [2026-05-17 18:00] brain_wiring | Slug quality fix in `_slugify_topic`

Dropped stopwords before slugifying; greedy-fit at token boundary when
capped at 80 chars. Fixes the truncation pattern that produced slugs
like `..._the_wheel_p` from today's covered-call wiki write.

Added `_SLUG_STOPWORDS` frozenset (~30 entries: articles, question
words, generic copula, topic-tag fluff like "use/mechanics/intro/
implications"). Validation walk-through:

- Input: `covered call options strategy — mechanics, when to use, tax
  implications, the wheel pattern`
- Old result: `covered_call_options_strategy_mechanics_when_to_use_tax_implications_the_wheel_p`
- New result: `covered_call_options_strategy_tax_wheel_pattern`

Requires Chloe restart to pick up (`brain_wiring.py` imported once at
boot).

## [2026-05-17 17:30] mcp_server | v2.1 shipped — 16 tools total

Added `brain_write`, `finance_watchlist_read`, `lights_status`,
`lights_set`, `lights_command`, `lights_preset`. Connector now covers
all of Chloe's persistent state (read+write) plus direct lights control.
See `chloe_handoff.md` → MCP connector section for the full menu.

## [2026-05-17 17:30] scheduled_tasks | Daily morning brief task created

`chloe-daily-morning-brief` at cron `0 7 * * *`. Pulls episodic CONTEXT
+ recall + web_history + recent generated/, writes a tight brief to
`brain/briefs/morning_brief_<date>.md`. First test run produced 2072
bytes correctly scoped.

## [2026-05-17 17:30] scheduled_tasks | Daily finance ingest task created

`chloe-daily-finance-ingest` at cron `30 7 * * 1-5`. Reads
`jarvis/finance_watchlist.md` (Ed's tickers + themes + strategies),
WebSearches per-ticker news, writes digest to
`wiki/sources/finance_news_<date>.md` via `brain_write`. First test
run produced 6952 bytes covering SLV, WU, TE with macro overlay.

## [2026-05-17 17:30] scheduled_tasks | Friday meta-review task created

`chloe-friday-meta-review` at cron `0 8 * * 5`. Replaces the missing
Cowork task that should have been firing each Friday. Self-contained
prompt uses MCP tools to gather context + writes to
`C:\Chloe\reviews\<date>_meta.md` via `reviews_write`. First test run
produced 5016 bytes.

## [2026-05-17 17:00] jarvis | Ack-gate fix — short-circuit ≤3-token acks

Closes the 2026-05-12 self-identified `grep_source` false-fire bug.
Added `_THANKS_TOKENS`, `_SHORT_ACK_TOKENS`, `_maybe_pick_ack_reply()`
near `_INTROSPECTION_KEYWORDS`. Chat + voice handlers both gated.
Verified: chat `thanks` → `happy to.` no LLM call. Splice:
`splice_ack_gate.py`.

## [2026-05-17 17:00] chloe_about | "When thanked" rule promoted

Added verbatim banned phrasings from 5/12 transcripts as negative
few-shots, plus 4 positive examples. Reference to the ack-gate
inserted so the model knows trivial acks may be handled before
reaching the prompt.

## [2026-05-17 16:30] mcp_server | v2 shipped — 10 tools

Added `brain_read`, `brain_list`, `reviews_read`, `add_fact`,
`wiki_write`, `reviews_write`, `queue_add`. Closes v1 read-only
restriction and the wiki-only coverage gap.

## [2026-05-17 16:00] jarvis | Bug #4 fix — `_needs_brave_direct` classifier

Forces Brave fallback on temporal+result queries that confidently
confabulate. Closes the hedge-detection blind spot the Eurovision
smoke test exposed. Splice: `splice_brave_direct.py`.

## [2026-05-17 15:30] jarvis + brain_wiring | Pillar 1 — search-history → memory

`_persist_brave_to_wiki` auto-writes Brave results to
`wiki/sources/web_*.md`. New `/web_history [today|week|month]` slash
command. Splices: `splice_brave_wiki.py`, `splice_web_history.py`.

## [2026-05-17 12:00] jarvis | Speak-state broadcast moved to audio onset

Pulse animation now syncs to actual sound, not synth-gap silence. Fix
inside `_speak_kokoro` consumer loop (broadcast "speaking" on first
chunk, "idle" in finally). 21 early call-site broadcasts removed.
Splices: `splice_greet_sync.py`, `splice_speak_sync.py`,
`splice_greet_revert.py`.

## [2026-05-17 11:00] docs | Handoff consolidation

`chloe_handoff.md` rewritten from 8 dated `SESSION_HANDOFF_*.md`
files. Now the canonical state doc — dated handoffs are archive-only.
Memory entry `chloe_canonical_state.md` enforces "read handoff first
when working on Chloe."

---

*Entries before 2026-05-17 should be backfilled from `log.md`, git
history, and `.bak.<date>` filenames as time permits.*
