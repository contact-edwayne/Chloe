# Stage 3: voice/chat-confirm per apply

**Status:** designed, NOT built. Unlocks after Stage 2 logs ≥5
successful token-gated apply cycles with no regressions.

**Goal:** Chloe announces a code change conversationally ("I drafted a
fix for X — want me to apply?") and Ed confirms by voice or chat
("yes" / "go ahead" / "apply"). The flow is gated per-apply but is
**conversational** rather than mechanical — no slash typing, no token
copy/paste.

This is the most natural surface for autonomy-with-oversight. Each
apply still has explicit Ed approval, but the friction drops to a
single spoken word.

## Surface area to build

### 1. New module `chloe_pending_confirms.py` (~100 lines)

Module-level state keyed by `proposal_slug`:

```python
_PENDING_CONFIRMS: dict[str, dict] = {}  # slug -> {asked_at, ttl_s, source}

def announce(slug: str, source: str = "voice", ttl_s: int = 120) -> dict:
    """Mark a proposal as pending Ed's spoken confirmation. Returns the
    speech-shaped string Chloe should say."""

def resolve(user_text: str) -> Optional[dict]:
    """Called from the chat/voice handler on each non-slash user turn.
    If user_text reads as 'yes' AND there's a pending confirm within TTL,
    apply the proposal. Returns the apply result or None if no resolution
    triggered.

    Affirmative phrases (case-insensitive):
      yes / yeah / yep / yup / sure / go / apply / approve / do it /
      go ahead / send it / ship it
    Negative phrases (cancel pending):
      no / nope / nevermind / cancel / hold off / not yet / wait
    Anything else = leave pending until TTL expires.
    """

def pending() -> list[dict]:
    """List active pending-confirms (slug, asked_at, ttl_remaining_s)."""

def cancel(slug: str = "") -> dict:
    """Cancel a specific slug or all pending if slug=''."""
```

### 2. Wire into voice + chat paths in `jarvis.py`

In both the voice handler (`_ask_groq` path) and the chat handler
(`handle_chat`), AFTER the ack-gate fires but BEFORE the brain-slash
dispatch, call:

```python
confirm_result = chloe_pending_confirms.resolve(_last_user)
if confirm_result is not None:
    # User said yes/no to a pending confirm. Format reply.
    reply = (
        f"applied. {confirm_result['summary']}"
        if confirm_result.get("ok")
        else f"hmm, that didn't take — {confirm_result.get('error')}"
    )
    # Emit reply through normal voice/chat paths
    ...
    return
```

This is a small, surgical insertion — ~15 lines per path.

### 3. New MCP tool `propose_and_announce(target, kind, rationale, body, test_plan, slug, title, source)`

Wraps `chloe_proposals.create_proposal(...)` then calls
`chloe_pending_confirms.announce(slug, source=...)`. Returns the
speech-shaped string. Cowork-Claude or a Cowork job invokes this when
it wants to draft AND surface a proposal in one shot.

### 4. Optional: TTS the announce string

When `source="voice"`, the announce string is sent through the same
TTS pipeline used for chat replies. That way Chloe speaks the
proposal summary out loud.

## Safety design

- **Per-apply gate:** every apply still requires explicit affirmative
  from Ed. No batching. (Stage 2 tokens handle batching for non-
  conversational callers.)
- **TTL:** default 120 seconds. After that the pending confirm is
  silently dropped — prevents stale "yes" responses from triggering
  an old proposal hours later.
- **Source separation:** voice-announced confirms can only be resolved
  by voice replies. Chat-announced confirms can only be resolved by
  chat replies. Avoids cross-channel ambiguity ("yes" to a voice
  prompt accidentally triggering a chat-announced proposal).
- **Negation handling:** explicit no/cancel words drop the pending
  immediately. Ambiguous responses leave the pending in place — Ed
  can disambiguate by saying yes/no later.
- **One pending per slug:** announcing a slug that's already pending
  refreshes the TTL but doesn't queue a second announcement.

## Open questions for the build

1. **Speech recognition fuzziness.** "yes" / "yeah" / "yep" all work.
   What about "alright" or "okay" or "sounds good"? Risk: too liberal
   = false-positive applies. Recommendation: start with the explicit
   short list, log every miss, expand based on real usage.

2. **Multi-proposal disambiguation.** If two proposals are pending,
   "yes" applies which one? Recommendation: announce + apply one at
   a time. If a second proposal is created while one is pending,
   queue it but don't announce until the first resolves.

3. **Should `propose_and_announce` be a Cowork-only MCP tool, or also
   a slash?** Recommendation: MCP-only for now. The voice/chat-confirm
   path is for Chloe-initiated proposals; Ed initiating his own
   proposals via slash already has `/apply_proposal` for direct
   apply.

## Build estimate

~4 hours including TTS plumbing, voice-path wiring, chat-path wiring,
test fixtures for affirmative/negative phrase matching, and update to
`verify_proposals.bat`.

## Trust gate for promotion to Stage 4

Run Stage 3 in production for ≥10 successful voice/chat-confirm
cycles. Watch for:
- False-positive applies (Chloe applied something Ed didn't intend)
- Missed applies (Ed said yes but it didn't take)
- Cross-channel leakage (voice "yes" resolved a chat pending or vice
  versa)

Zero of all three over 10 cycles = ready for Stage 4. Any one of them
= bug-fix the failure mode and reset the counter.
