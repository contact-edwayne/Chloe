# Spec: Lights Groups

Status: SHELVED 2026-05-10 — full Spec / Plan / Tasks artifact produced as a smoke test of the agent-skills spec-driven-development flow. Feature is real and well-scoped but was not implemented because Edward decided not to ship it tonight. To revive: read this file top-to-bottom, then resume at Phase 4 (Implement) starting with task T1.

Open questions resolved with defaults: explicit groups only (no implicit 'all bulbs'); presets apply uniformly across group members; `lights list` shows groups as a separate section.

## Objective

Let Edward control multiple Zengge Magic Home WiFi bulbs at once by
addressing a named group instead of issuing per-bulb commands. Today
Chloe can turn on, dim, recolor, or apply presets to one bulb at a time
by name (`top`, `middle`, etc.). After this feature, a single command —
voice, chat, or CH02 click — applies to every bulb in the named group.

**Who:** Edward, single user.
**Why:** Reduce repetitive multi-bulb commands. The four bulbs naturally
cluster (e.g. desk + ceiling = "office"); addressing the cluster instead
of each bulb matches how he actually thinks about the room.
**Success:** "Chloe, turn on the office" turns on every bulb in the
`office` group in one command. CH02 panel shows a clickable "Groups" row.
Edward can add or change groups by editing `lights.json` and restarting
Chloe.

## Tech Stack

- Python 3.x (existing)
- `lights.py` (existing module at `C:\Users\eleew\Documents\jarvis\lights.py`)
- `lights.json` config at `C:\Chloe\secrets\lights.json`
- HUD frontend: `hud.html` (existing CH02 panel)
- No new dependencies.

## Project Structure

No new files. Changes scoped to existing surface:

```
jarvis\lights.py          → group resolution + apply-to-all logic
jarvis\jarvis.py          → ws handlers may need group awareness (likely no change)
jarvis\hud.html           → CH02 panel adds a Groups row
C:\Chloe\secrets\lights.json → new top-level "groups" dict
```

## Code Style

Match existing `lights.py` patterns:
- snake_case names
- Top-level functions, no class hierarchy (lights.py is module-level)
- Docstrings for every new public function
- Guard with `if not groups: return ...` style early returns

Example shape for the new resolver:

```python
def resolve_targets(name: str, cfg: dict) -> list[str]:
    """Resolve a bulb-or-group name into a flat list of bulb slugs.

    Returns [name] if name matches a bulb, the group's member list if it
    matches a group, or [] if neither (caller decides how to error).
    A group name and a bulb name MUST NOT collide; load_config rejects
    the config at startup if they do.
    """
    bulbs = {b["name"]: b for b in cfg.get("bulbs", [])}
    groups = cfg.get("groups", {})
    if name in bulbs:
        return [name]
    if name in groups:
        return list(groups[name])
    return []
```

## Commands

No new build/test commands. Existing dev loop applies:
- Run Chloe: `python jarvis.py` (from venv)
- Manual smoke test: edit `lights.json`, restart, try `/lights set <group> on` in chat
- Voice test: "Chloe, turn on the <group>"

## Testing Strategy

Chloe has no automated test suite. Verification is manual:

1. **Config validation** — add a group to `lights.json` with a name that
   conflicts with a bulb, confirm Chloe refuses to start with a clear error.
2. **Single-group on/off** — `/lights set office on` flips all members.
3. **Group brightness** — `/lights set office to 50%` sets all members.
4. **Group color** — `/lights set office to cyan` sets all members.
5. **Group preset** — preset application from CH02 panel applies to all
   group members.
6. **Voice path** — "Chloe, turn on the office" works.
7. **Bulb commands still work** — `/lights set top on` still controls just
   the one bulb, regression check.
8. **Unknown name** — `/lights set foo on` returns the existing
   "no such bulb" error, now generalized to "no such bulb or group".

A small `--test-resolver` CLI flag on `lights.py` could exercise
`resolve_targets()` against a few fixtures without hardware. Optional.

## Boundaries

**Always:**
- Resolve group names through the same lowercase + trim normalization
  bulb names use, so case doesn't matter.
- Preserve the existing single-bulb command surface unchanged.
- Validate `lights.json` on load; refuse to start if a group references
  a non-existent bulb or collides with a bulb name.
- Back up `lights.py` before patching (file is ~24KB+, past Edit-tool
  danger zone — use bash + Python splice).

**Ask first:**
- Adding a new top-level command verb to lights.py (we should reuse
  existing verbs, not invent new ones).
- Adding nested groups (groups containing groups). Defer until proven
  necessary.
- Adding a "scene" concept (group + saved per-bulb color/brightness).
  Different feature; spec separately if wanted.

**Never:**
- Mutate `lights.json` from code (this version). Groups are config,
  edited by hand. Auto-mutation comes with the future UI feature, if any.
- Add new dependencies.
- Break the existing per-bulb command API.

## Success Criteria

- [ ] Voice "Chloe, turn on the office" turns on every bulb in the
  `office` group (defined in lights.json as e.g. `{"office": ["top",
  "middle"]}`) within ~3 seconds.
- [ ] Chat `/lights set office to red` sets all group members to red.
- [ ] CH02 panel displays a Groups row above the bulb cards; clicking a
  group chip applies on/off/brightness/color to all members.
- [ ] Per-bulb commands (`/lights set top on`) continue to work unchanged.
- [ ] Config with bulb/group name collision causes Chloe to refuse to
  start with a clear error message naming the conflicting names.
- [ ] Config referencing a non-existent bulb in a group causes Chloe
  to refuse to start with a clear error message.
- [ ] No new env vars, no new dependencies, no new top-level files.

## Open Questions

1. **Should the CH02 panel show "all 4 bulbs" as a built-in implicit group**
   in addition to user-defined groups? Useful for "lights all off" without
   defining an `everything` group. Default: no, keep groups explicit.
2. **When a group applies a preset that has per-bulb assignments, what wins?**
   The current preset model is global-state (all bulbs to same state).
   Should we extend presets to per-bulb-in-group, or keep presets uniform?
   Default: presets stay uniform; per-bulb-in-group is a future feature.
3. **Should `lights list` output groups as a separate section?** Default:
   yes, format as `bulbs: a, b, c, d | groups: office=[a,b], ambient=[c]`.


---

## Plan

### Components

1. **Config loading** — `lights.py: load_config()` (existing) extended to
   read the new `groups` dict from `lights.json`. Validates:
   (a) no group name collides with a bulb name,
   (b) no group references a non-existent bulb,
   (c) `groups` field is optional (missing = empty dict).
   Raises `ValueError` with a clear message on validation failure.

2. **Target resolver** — `lights.py: resolve_targets(name, cfg) -> list[str]`
   (new). Pure function. Returns `[name]` if name is a bulb, member list
   if name is a group, `[]` otherwise. Caller decides how to handle empty.

3. **Action dispatch** — `lights.py: apply_action`, `set_state`,
   `apply_preset` (existing) all gain a thin wrapper that takes a
   target name, resolves it via `resolve_targets`, and applies the
   action to each resolved bulb. Existing per-bulb call signatures
   stay valid (resolver returns `[name]` for bulbs).

4. **Command surface** — `lights.py: parse_intent` and
   `try_handle_lights_command` (existing) updated so the target
   matcher accepts group names alongside bulb names. The error
   message changes from "no such bulb" to "no such bulb or group".

5. **`--list` output** — `lights.py: list_bulbs` (or equivalent
   CLI handler) extends output with a separate "Groups:" section
   below the bulbs section.

6. **WS state payload** — `lights.py: get_state_snapshot` extended
   to include a `groups` key in the broadcast payload. Existing
   clients ignore unknown keys; new HUD code uses them.

7. **CH02 panel — Groups row** — `hud.html`: above the existing
   per-bulb card grid, render a horizontal row of group chips
   (one per group). Click chip → calls existing `set_state` /
   `apply_action` WS messages with the group name as target.

### Implementation order (sequential, dependency-driven)

```
1. Config + validation    ──┐
                            ▼
2. resolve_targets()      ──┐
                            ▼
3. Action dispatch wrap   ──┐
                            ▼
4. Command surface + list ──┐
                            ▼
5. WS state payload       ──┐
                            ▼
6. HUD CH02 Groups row    ──┐
                            ▼
7. End-to-end verify on hardware
```

Each step is committable on its own and leaves the system working.

### Parallel opportunities

None worth taking. Edward is one developer with one Claude. Sequential
is the right pattern at this scope.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| `lights.py` is past Edit-tool threshold (~24KB+) | Backup first (`copy /Y lights.py lights.py.bak.<date>`); use bash + Python splice via `pathlib.write_text`. |
| `hud.html` is well past threshold (123KB) | Same. Splice with anchor-based replace. CSS additions go before `</style>` close, JS additions go in the existing CH02 channel block. |
| Existing PWA clients see new `groups` key in WS payload | Existing chloe-mobile.html uses dict.get with defaults; unknown keys are ignored. Test by leaving phone connected during a backend restart. |
| `parse_intent` fuzzy matching could falsely match a group name | Reuse the existing exact-match-first / fuzzy-fallback pattern. Group names are config-controlled by Edward; he chooses short slugs. |
| Config validation failures kill Chloe on startup | Acceptable per spec boundary ("Always: refuse to start with clear error"). Log the offending names so Edward can fix `lights.json` immediately. |
| JSON migration: existing `lights.json` has no `groups` key | `load_config` treats missing `groups` as `{}`. Zero-impact for existing setup until Edward adds groups. |

### Verification checkpoints

| Phase | Verify with |
|---|---|
| 1. Config | Edit `lights.json` to add a deliberately bad group (collision with bulb name); restart Chloe; confirm startup refuses with named-conflict error. |
| 2. resolver | One-off Python REPL test against the cfg dict: `resolve_targets("top", cfg)` returns `["top"]`; `resolve_targets("office", cfg)` returns the member list; `resolve_targets("nope", cfg)` returns `[]`. |
| 3. dispatch wrap | `/lights set top on` still works (regression). `/lights set office on` flips all members. |
| 4. command surface | `/lights set foo on` (no such name) returns "no such bulb or group". Voice "Chloe, turn on the office" works. |
| 5. list output | `python lights.py --list` shows `Groups:` section below `Bulbs:`. |
| 6. WS payload | Open HUD, click a group chip, all bulbs in the group respond. Per-bulb cards update to reflect new state. |
| 7. end-to-end | Define an office group of 2 bulbs in `lights.json`. Restart. Voice command, chat command, CH02 chip click — all three should flip all members. |

### Out of scope (explicit)

- UI for creating/editing groups (Edward edits `lights.json` by hand)
- Nested groups (group containing another group)
- Per-bulb-in-group state in presets / scenes
- Stateful group toggles ("turn off remembers state, turn on restores")
- Auto-mutation of `lights.json` from code

These are listed in the spec under "Ask first" or "Never" boundaries.

---

**Plan approval gate:** _Approved by Edward 2026-05-10._


---

## Tasks

Implementation is broken into 9 atomic units, dependency-ordered. Each
task is small enough to complete + verify in a single focused session,
and each leaves the system functional if I stop after it.

### T1. Schema + load_config validation

- **Description:** Extend `load_config()` in `lights.py` to read an
  optional `groups` dict from `lights.json`. Validate (a) every group
  name is unique, (b) no group name collides with a bulb name,
  (c) every member referenced exists as a bulb. On any violation,
  raise `ValueError` with a clear message naming the offending entries.
  If `groups` is missing entirely, treat as `{}` and proceed.
- **Acceptance:**
  - `load_config()` returns a dict with a `groups` key (possibly empty).
  - Bad config (name collision, missing bulb reference, duplicate group)
    raises ValueError with a message that names the conflict.
  - Pre-existing `lights.json` files (no `groups` key) still load.
- **Verify:**
  1. `python -c "from lights import load_config; print(load_config())"`
     against current `lights.json` — should succeed, show empty groups.
  2. Edit `lights.json` to add `{"groups": {"top": ["middle"]}}` (name
     collides with a bulb). Same Python command should raise.
  3. Restore good config.
- **Files:** `lights.py` (one function), `lights.json` (test data only,
  reverted after).

### T2. resolve_targets() pure function

- **Description:** Add `resolve_targets(name: str, cfg: dict) -> list[str]`
  to `lights.py`. Returns `[name]` if name is a bulb, the group's member
  list if name is a group, `[]` otherwise. Name match is exact after
  lowercase + strip.
- **Acceptance:**
  - Bulb name returns single-element list.
  - Group name returns full member list.
  - Unknown name returns empty list.
  - Case insensitive.
- **Verify:** Inline Python test fixture:
  ```python
  cfg = {"bulbs": [{"name": "top"}, {"name": "middle"}],
         "groups": {"office": ["top", "middle"]}}
  assert resolve_targets("top", cfg) == ["top"]
  assert resolve_targets("office", cfg) == ["top", "middle"]
  assert resolve_targets("nope", cfg) == []
  assert resolve_targets("OFFICE", cfg) == ["top", "middle"]
  ```
- **Files:** `lights.py`.

### T3. Action dispatch through resolver

- **Description:** Modify `apply_action`, `set_state`, `apply_preset` in
  `lights.py` to accept a target name and route through `resolve_targets`,
  applying the action to each resolved bulb. Per-bulb calls keep working
  because the resolver returns `[name]` for bulbs.
- **Acceptance:**
  - All three functions accept the same argument they did before.
  - When given a bulb name, behavior is unchanged (regression).
  - When given a group name, action applies to every group member.
  - When given an unknown name, returns a clear error (no crash).
- **Verify:**
  1. `/lights set top on` (chat) — top bulb still toggles, regression check.
  2. Define `office` group containing top + middle in `lights.json`.
     `/lights set office on` — both bulbs respond within ~3s.
  3. `/lights set foo on` — returns "no such bulb or group `foo`".
- **Files:** `lights.py`.

### T4. Command surface + parse_intent + lights --list

- **Description:** Update `parse_intent` and `try_handle_lights_command`
  so the target resolver path matches group names too. Update the
  "no such bulb" error to "no such bulb or group". Update `--list`
  CLI output to show groups as a separate section.
- **Acceptance:**
  - Voice/chat "set <group> on" works (already covered by T3 but verify
    end-to-end through dispatcher, not just the underlying function).
  - Error message names "bulb or group".
  - `python lights.py --list` shows two sections: Bulbs, Groups.
- **Verify:**
  1. `/lights set office to cyan` → all members go cyan.
  2. `/lights set fake on` → error mentions "bulb or group".
  3. `python lights.py --list` → output has "Groups:" header.
- **Files:** `lights.py`.

### T5. WS state payload includes groups

- **Description:** Extend `get_state_snapshot()` to include a `groups`
  key in the broadcast payload (e.g. `{"office": ["top", "middle"]}`).
  Existing clients ignore the new key; new HUD code uses it.
- **Acceptance:**
  - WS `lights_state` broadcast payload has a `groups` field.
  - Field is `{}` when no groups defined.
  - Field reflects current `lights.json` after restart.
- **Verify:**
  1. Open browser dev tools on the HUD, watch the lights_state WS
     message — confirm `groups` key present.
  2. Existing chloe-mobile.html still renders bulb cards correctly
     (regression — it doesn't know about groups yet).
- **Files:** `lights.py`.

### T6. HUD CH02 Groups row

- **Description:** Add a horizontal Groups row above the per-bulb card
  grid in CH02. Each group renders as a clickable chip with the group
  name. Click → on/off toggle (matching the per-bulb card pattern).
  Optional long-press / second click → opens the brightness+color
  controls that apply to the whole group. For tonight's scope, click
  toggles on/off; color/brightness controls are an enhancement.
- **Acceptance:**
  - CH02 panel shows a Groups row above bulb cards when groups exist.
  - Hidden / collapsed when no groups exist.
  - Click chip → all member bulbs toggle.
  - Chip visual state reflects "all on" / "all off" / "mixed" of members.
- **Verify:** Open HUD, switch to CH02, click office chip, watch all
  member bulbs respond. Click again → all off.
- **Files:** `hud.html` (CSS + HTML + JS additions, all in CH02 block).

### T7. End-to-end hardware verify

- **Description:** Verify all three command paths work together.
- **Acceptance:** With a real `office` group defined and Chloe running:
  - Chat: `/lights set office on` works.
  - Voice: "Chloe, turn on the office" works.
  - HUD CH02 click works.
  - All three reflect the same state immediately.
- **Verify:** Manual smoke test in front of the bulbs.
- **Files:** None (verification only).

### Dependencies

```
T1 ──→ T2 ──→ T3 ──→ T4 ──→ T5 ──→ T6 ──→ T7
```

Strict sequential. Each task leaves the system functional, so we can
stop and resume between any two tasks without leaving things broken.

### Time estimates (informal)

| Task | Estimate |
|---|---|
| T1 | 10 min |
| T2 | 5 min |
| T3 | 15 min |
| T4 | 10 min |
| T5 | 5 min |
| T6 | 25 min (HUD work is slower) |
| T7 | 5 min (manual) |
| **Total** | **~75 min** |

Plus backup-before-patch overhead for `lights.py` and `hud.html`
(both past Edit-tool danger zone — bash splice required).

---

**Tasks approval gate:** Edward, do these 9 7 tasks look right? Once
approved, I move to Phase 4 (Implement) and execute T1-T7 in order,
verifying each one before moving on per `incremental-implementation`.
