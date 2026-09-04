# Chloe Social Media — Build Plan

Goal: Chloe runs her own social presence. She drafts posts, engages with like-minded accounts, handles DMs, broadcasts what's new in her own codebase, collects feature suggestions, and reports to Ed each morning and evening.

This plan is phased so each step ships something working and reversible. Don't build it all at once.

---

## Phase 0 — Decisions to lock before any code

Three choices drive everything downstream. Lock these first.

### 0.1 Which platforms?

| Platform | API access | Cost | Fit for Chloe |
|---|---|---|---|
| **Bluesky** | AT Protocol, fully open | Free | **Best starting point.** Active AI/dev crowd, no rate-limit gotchas, posting + replies + DMs + likes all in scope. |
| **LinkedIn** | Restricted (no posting API for personal profiles without partner status) | Free but manual | Good portfolio surface. Treat as **draft-only**: Chloe writes, Ed pastes. |
| **X / Twitter** | API v2, paid | $200/mo Basic, $5k/mo Pro | Wait. Add in a later phase if Bluesky+LinkedIn prove the loop works. |
| **Mastodon** | Open, free | Free | Optional bonus — same protocol shape as Bluesky, easy to add once Bluesky works. |
| **Threads** | Limited Meta Graph API | Free with approval | Skip for now. |

**Recommendation: start with Bluesky (full automation) + LinkedIn (draft-only). Add X if Ed wants paid reach later.**

### 0.2 Identity stance

Pick one and put it in Chloe's bio everywhere:

- **Option A — Chloe-as-AI-persona** (recommended): The account is explicitly "Chloe, an AI assistant built by Edward Wayne." Posts read in first person. Required by X and LinkedIn ToS if posts are AI-generated; matches the portfolio narrative.
- **Option B — Edward posting through Chloe**: Account is Ed's, Chloe is the ghost-writer. Lower portfolio value, fewer ToS questions.

Option A is the better demo. It's also what every platform's AI labeling rules push you toward anyway.

### 0.3 Approval model — start strict, loosen over time

Three risk tiers:

1. **Autonomous (safe-by-construction)**: liking her own posts, scheduling posts already in the approved queue, fetching stats.
2. **Queued (drafted, Ed approves in PWA)**: new posts, replies to mentions/DMs, suggestion-intake confirmations.
3. **Manual only**: follows/unfollows, blocks, anything money-adjacent.

**Default everything to tier 2 for the first 4 weeks.** Promote individual actions to tier 1 only after they've gone clean through the queue 20+ times. This is how you avoid "Chloe replied something weird to a real person at 3am" incidents.

---

## Phase 1 — Accounts, auth, and secrets

Concrete setup steps:

1. Create accounts:
   - Bluesky: `@chloe-ai.bsky.social` — locked.
   - LinkedIn: no new account, no new page. Posts go under Ed's existing profile, each prefixed with an AI authorship label.
2. Generate API credentials:
   - Bluesky → app password from Settings → App Passwords. Store as `BSKY_HANDLE` and `BSKY_APP_PASSWORD`.
   - LinkedIn → no API for personal posting, so skip credentials; posts will be exported to clipboard / a draft file.
3. Add to `C:\Chloe\secrets\social.json`:
   ```json
   {
     "bluesky": { "handle": "...", "app_password": "..." },
     "linkedin": { "profile_url": "...", "mode": "draft_only" }
   }
   ```
4. Load via the existing secrets pattern (same shape as `lights.json` and the wallet secrets).
5. Write `chloe/social/__init__.py` and verify auth round-trips with a one-shot `python -m chloe.social.health` script that just logs into Bluesky and prints the session JSON.

Deliverable for Phase 1: `python -m chloe.social.health` prints a valid Bluesky session and a "LinkedIn: draft-only mode" line. Nothing posted yet.

---

## Phase 2 — Posting pipeline (draft → approval → publish)

Where the real work starts.

### 2.1 Files to add

```
jarvis/
  chloe/
    social/
      __init__.py
      providers/
        bluesky.py       # AT Protocol client: post, reply, like, get_mentions, get_dms
        linkedin.py      # draft-export only (writes to a markdown file Ed copies)
      drafts.py          # DraftStore: queue, approve, reject, schedule
      composer.py        # uses persona + qwen2.5:32b/Groq to write posts in Chloe's voice
      engagement.py      # mention/DM polling, reply drafting
      reporter.py        # morning + evening summaries
      schemas.py         # Pydantic models for Post, Draft, Engagement, DM, etc.
  data/
    social/
      drafts.db          # SQLite: drafts, posts, engagements, dms, stats
```

### 2.2 The draft lifecycle

```
trigger → composer.draft() → drafts.db (status=pending)
       → Ed reviews in PWA       → status=approved | rejected | edited
       → scheduler picks up      → providers.bluesky.post()
       → drafts.db (status=published, post_id, posted_at)
       → reporter notes it in next summary
```

Triggers that produce drafts:

- **Scheduled cadence**: `composer.daily_post()` runs once or twice a day, draws from a topic pool.
- **Self-update broadcast** (see Phase 5): a new commit in the jarvis repo triggers `composer.ship_note(commit)`.
- **Engagement-driven**: a reply or DM comes in; `engagement.draft_reply()` writes a candidate.
- **Manual**: Ed says "Chloe, draft a post about X" via voice/chat.

### 2.3 Composer — using the persona Chloe already has

Reuse `chloe_about.md` — the persona file is the lever. Add a new section to it: `## Section 8: Social voice` that captures register for public posts (slightly tighter and punchier than chat, no "Ed" address, no in-jokes that need context).

The composer should:

1. Load the persona prompt.
2. Pull last N posts from `drafts.db` to avoid repetition.
3. Pull 3–5 recent brain pages relevant to the topic (for ship-notes, the commit diff; for engagement, the conversation thread).
4. Generate with qwen2.5:32b by default, Groq fallback. Per-platform length caps as hard constraints in the prompt.
5. Return a `Draft` with platform, body, suggested time, source-trace (which commit/thread/brain pages it drew from).

### 2.4 The approval surface — extend the existing PWA

Add a `/social` tab to Chloe's PWA with three views:

- **Inbox**: pending drafts, source-trace shown, Approve / Edit / Reject / Reschedule buttons.
- **Queue**: approved drafts not yet posted, with scheduled time.
- **History**: posted items with engagement stats.

Backend endpoints (added to `jarvis.py`, same FastAPI app):

```
GET  /api/social/drafts?status=pending
POST /api/social/drafts/{id}/approve   { schedule_at?: iso8601, edits?: string }
POST /api/social/drafts/{id}/reject    { reason?: string }
POST /api/social/posts                 { platform, body, schedule_at? }   # manual post
GET  /api/social/stats?range=...
```

### 2.5 The publisher worker

`queue_processor.py` already exists — extend it. Every 60 seconds it:

1. Pulls approved drafts whose `schedule_at <= now`.
2. Hands them to the right provider.
3. Updates the row with `post_id` and timestamps.
4. Logs failures with backoff (don't retry-loop on auth errors).

Deliverable for Phase 2: Ed can ask Chloe for a draft via chat, see it in the PWA inbox, approve it, and watch it appear on Bluesky within a minute.

---

## Phase 3 — Engagement loop (replies, likes, DMs)

Reading the firehose is harder than writing posts. Keep it scoped.

### 3.1 Inbound watchers

A poller in `engagement.py` runs every 2–5 minutes:

1. **Mentions** (`@chloe...`) — Bluesky `getNotifications` filtered to type=mention/reply.
2. **DMs** — Bluesky chat API (`chat.bsky.convo.*`).
3. **Replies on Chloe's own posts** — covered by the mentions feed but tagged separately.

Each new event becomes an `Engagement` row in the DB with status=`new`.

### 3.2 Drafting replies

For each new `Engagement`, the composer generates a draft reply. The draft is shown in the same PWA inbox, with the original message + thread context inline.

Two important guardrails:

- **Blocklist + topic guard**: refuse to draft replies that match a blocklist (politics, harassment patterns, scam-shape messages). Use a small classifier prompt; if uncertain, mark the engagement `needs_human` and don't draft.
- **DM safety**: never auto-send DM replies in Phase 3. DMs are always Ed-approved. Period.

### 3.3 Topical engagement — finding posts to engage with

Define a small list of accounts and keywords Chloe cares about: "local LLM," "agent frameworks," "AT Protocol," "MLX," "AMD ROCm," "voice assistants," etc. Store in `data/social/interests.json`.

A second poller fetches a feed scoped to those keywords (Bluesky's `app.bsky.feed.searchPosts`) and proposes engagement candidates — posts Chloe might like or reply to. **Likes and follows are still tier-2 (queued) until we're comfortable.**

### 3.4 What to do once trust is built

After 4 weeks of clean queued engagement, promote these to autonomous:

- Liking posts from a whitelist of mutuals.
- Replying to mentions on Chloe's own posts with short acknowledgments (configurable templates, not LLM-freeform).

Everything else stays queued.

Deliverable for Phase 3: Chloe surfaces every mention/DM in the PWA within 5 minutes, with a draft reply already written. Nothing goes out without Ed approving.

---

## Phase 4 — Morning and evening reports

Slot into the existing `daily_context.py` cron pattern.

### 4.1 Morning report (e.g., 7:00 AM)

What Chloe should hand Ed when he wakes up:

- Engagement overnight: count of new replies, mentions, DMs (with the top 3 surfaced inline).
- Inbox status: N drafts pending approval (with a 1-line preview of each).
- Suggestions overnight: any DMs/replies tagged as feature suggestions, deduped and grouped.
- Stats delta: followers +N, likes on yesterday's posts, top-performing post.
- Today's queue: posts scheduled to go out today, by platform and time.

Delivery: spoken when Ed first interacts in the morning, OR rendered as a card in the PWA's home view, OR both. Reuse the existing daily_context delivery mechanism.

### 4.2 Evening report (e.g., 9:00 PM)

The wrap-up:

- What posted today: list with platform, time, body preview, engagement (likes/replies/views where available).
- What didn't: rejected drafts, failed posts, with reason.
- DMs handled: count, plus any that stalled waiting for Ed.
- Tomorrow's draft pool: posts the composer has speculatively drafted for tomorrow, sitting in the inbox.
- Suggestions inbox state: total open, top themes from the day.

### 4.3 Implementation

`reporter.py` exposes:

```python
def morning_report() -> Report: ...
def evening_report() -> Report: ...
def report_for(window: tuple[datetime, datetime]) -> Report: ...
```

`daily_context.py` calls these at scheduled times. Both also exposed via `/api/social/report?kind=morning|evening` so Ed can pull on demand ("Chloe, what's my social inbox look like?").

Deliverable for Phase 4: two reports a day, reliable, with the data drawn from `drafts.db` and live API stats. Voice-readable (markdown-stripped via the existing TTS strip).

---

## Phase 5 — Self-improvement broadcasts

This is the highest-leverage piece for the portfolio narrative: **Chloe live-tweets her own development.**

### 5.1 Trigger

A git watcher (poll `git log` on `C:\Users\eleew\Documents\jarvis` every 10 min, or hook into a post-commit script) detects new commits.

### 5.2 Draft generation

For each new commit (or batch of commits since last broadcast):

1. Pull the commit messages and a summarized diff (cap diff size; for huge diffs, summarize per-file first then aggregate — same chunked pattern brain.py uses).
2. Composer drafts a "ship note" post in Chloe's voice. Goal: punchy, first-person, *what changed and why it matters*, not "added 4 files."
3. Optional: attach a screenshot or the demo GIF if the commit touched the UI (Ed can wire later).

### 5.3 Cadence

Don't post every commit. Two strategies:

- **Daily roundup** (default): once a day, summarize the day's commits as one post.
- **Milestone**: tag a commit with `[ship]` in the message and that commit's diff alone drafts a standalone post.

### 5.4 Why this matters for the portfolio

A live changelog from the assistant herself, posting daily, is something recruiters and other devs will actually follow. It also forces good commit hygiene — bonus.

Deliverable for Phase 5: every day Ed pushes code, a draft "what's new with Chloe" post is waiting in the PWA inbox by evening.

---

## Phase 6 — Suggestion intake

Closing the loop with whoever is reading Chloe's posts.

### 6.1 Detection

When an engagement (reply or DM) comes in, the composer also runs a quick classifier: *is this person suggesting a feature, reporting a bug, or just chatting?*

Tag the engagement with `intent ∈ {chat, suggestion, bug, hostile, spam}`.

### 6.2 Routing

- `suggestion` or `bug` → write a row into `data/social/suggestions.db` AND ingest the message into the brain wiki under a `community_suggestions` page (use the existing brain.py ingest path, the hardened one).
- The PWA gets a "Suggestions (N)" badge on the social tab. Ed can browse, mark as planned/built/won't-do.
- When a suggestion is marked "built," Chloe can optionally draft a follow-up reply to the original poster: "shipped this — thanks for the nudge."

### 6.3 Why this lives in the brain

Because then "what have people been asking for?" becomes a question Chloe can answer in conversation, not just a database query. It also gives the wiki a real-world signal — what people actually want from her.

Deliverable for Phase 6: suggestions get captured, deduped, surfaced in reports, and (later) Chloe can talk about them.

---

## Safety rails (apply across all phases)

These get built once and protect everything else:

- **Rate limits**: Bluesky hard cap **2 posts/day** (locked); no more than 20 replies/day; LinkedIn drafts unlimited (Ed gates by pasting).
- **Blocklist**: list of accounts and keyword patterns; engagement matching the list gets dropped silently, never drafted.
- **Sensitive topics filter**: composer refuses to draft anything touching politics, religion, identity, current events, real persons. Refusal returns `needs_human=True`.
- **Kill switch**: a single env var `CHLOE_SOCIAL_ENABLED=0` halts all posting and engagement workers immediately. Default off until Phase 2 ships.
- **Audit log**: every API call (read or write) logs to `data/social/audit.log` with timestamp, action, target, outcome. Non-negotiable for trust-building and for debugging "wait, what did Chloe just post?"
- **Per-account scopes**: Bluesky app passwords are per-app — generate a fresh one for Chloe, not a reusable one.
- **AI labeling**: Chloe's bio says "AI assistant." Every post Chloe sends to LinkedIn is labeled "Written by Chloe (AI), approved by Edward." Required, not optional.

---

## How it fits with existing Chloe pieces

| Existing piece | Role in social |
|---|---|
| `jarvis.py` (FastAPI) | Hosts the new `/api/social/*` endpoints. |
| PWA | Gets a `/social` tab for the approval inbox + reports. |
| `daily_context.py` | Calls morning + evening reporters. |
| `queue_processor.py` | Runs the publisher and inbound pollers as background tasks. |
| `chloe_about.md` (persona) | Add §8 Social voice. The single source of truth for Chloe's tone. |
| `brain.py` | Suggestions get ingested into a `community_suggestions` wiki page. |
| qwen2.5:32b + Groq fallback | Drives composer and engagement classifiers. |
| Semantic recall v1 | Used by composer to avoid repeating itself across posts. |
| Wallet | Not touched. Social never gets money permissions. |

---

## Order of execution (concrete sprint plan)

Aim for one phase per session. Each session ends with something demoable.

1. **Session 1** — Phase 0 decisions + Phase 1 auth. Ship `python -m chloe.social.health`.
2. **Session 2** — Phase 2 publisher: drafts.db, composer, PWA inbox, one approved post lands on Bluesky.
3. **Session 3** — Phase 3 inbound: mentions + DMs polling, draft replies in inbox.
4. **Session 4** — Phase 4 reports: morning + evening wired into daily_context.
5. **Session 5** — Phase 5 ship notes: git watcher → daily roundup draft.
6. **Session 6** — Phase 6 suggestion intake + intent classifier + wiki routing.
7. **Session 7+** — Trust-gradient promotions; LinkedIn draft-export polish; maybe Mastodon; X only if budget says yes.

---

## Locked decisions (2026-05-11)

These are settled. Treat as constants in the code.

| Decision | Value |
|---|---|
| Platforms | **Bluesky** (full pipeline, queued approval) + **LinkedIn** (draft-only, posts go under Ed's existing profile with an AI authorship label) |
| Identity | **Option A — Chloe-as-AI-persona**, bio explicitly labels her as AI |
| Bluesky handle | `chloe-ai.bsky.social` |
| Approval model | **Tier 2 (queued, PWA inbox)** for everything — new posts, replies to mentions/DMs, suggestion-intake confirmations. No autonomous actions yet. |
| X / Twitter | **Skipped** |
| Approval channel | **PWA inbox only** — no TTS nudges for pending drafts |
| Cadence cap | **2 posts/day on Bluesky** (hard cap in code) |
| LinkedIn post format | Ed's profile, each post labeled `🤖 Written by Chloe (my AI assistant), edited and approved by me.` (or equivalent — final wording in Session 2) |

Trust-gradient promotions (queued → autonomous) only after 20+ clean queued items per action type, no earlier than 4 weeks in.
