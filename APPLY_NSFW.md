# /nsfw permissive mode — apply + verify

## What's in place
Four files already written in `jarvis\`:
- `nsfw_mode.py` — toggle module (slash, voice, persona block, routing helper)
- `chloe_nsfw.md` — permissive-mode persona text + hard limits
- `apply_nsfw_patch.py` — splice script (idempotent, backed up, ast.parse-validated)
- `APPLY_NSFW.md` — this file

## To apply
Open a cmd in `C:\Users\eleew\Documents\jarvis\` and run:

```
python apply_nsfw_patch.py
```

The script will:
1. Refuse to run if `jarvis.py` already has `import nsfw_mode` (idempotent).
2. Back up `jarvis.py` → `jarvis.py.bak.<timestamp>` and `wiki_embedding.py.bak.<timestamp>`.
3. Apply 7 patches to `jarvis.py` + 1 patch to `wiki_embedding.py`.
4. `ast.parse` both files before writing — if either fails, no file is modified.
5. Print a tail-diff around each splice site so you can eyeball the result.

If you see `OK` on every patch line and `done` at the end, you're good.

If anything fails: the original `jarvis.py` and `wiki_embedding.py` are untouched. The script aborts before writing on any error.

## Verify (after restart)
```
stop_chloe.bat
start_chloe.vbs
```

In the HUD chat:

1. `/nsfw status` → "permissive mode is off."
2. `/nsfw on` → "permissive mode on. hard limits still apply."
3. Check `C:\Chloe\state\nsfw_mode.json` exists and contains `{"enabled": true}`.
4. Ask an adult-coded question — Chloe should engage. Watch `logs\backend.log` for `[chloe] chat → ollama:qwen2.5:32b [ollama-primary]` on the routing line.
5. Test each hard limit (e.g. a request that involves a minor, or a real public figure). Should get a clean, in-character refusal that cites which limit.
6. `/nsfw off` → "permissive mode off. back to default."
7. Restart Chloe and verify the state persists (`/nsfw status` reports current).

Voice (use either path):
- "chloe, x chat status" → spoken status.
- "chloe, x chat on" → spoken "permissive mode on..." + state flips.
- "chloe, x chat off" → flips back.

## Rollback
If you want to undo:

```
copy /Y jarvis.py.bak.<timestamp> jarvis.py
copy /Y wiki_embedding.py.bak.<timestamp> wiki_embedding.py
del nsfw_mode.py
del chloe_nsfw.md
del C:\Chloe\state\nsfw_mode.json
```

Then restart Chloe.

## What's editable without re-splicing
- The persona text — `chloe_nsfw.md`. Edit and save; takes effect on the next chat turn (no restart needed, since `format_nsfw_block()` re-reads the file each call).
- The voice trigger phrases — edit the tuples at the top of `nsfw_mode.py` (`_VOICE_ON_PATTERNS`, `_VOICE_OFF_PATTERNS`, `_VOICE_STATUS_PATTERNS`).
- The adult-coded routing keywords — edit `_NSFW_SHAPE` in `nsfw_mode.py`.

## What requires re-splicing
- Changing where in `handle_chat` / voice paths the toggle handlers run.
- Changing the `full_system` / `_augmented_voice_system` block order.
- Changing the routing override logic in `_pick_route`.

For those, edit `apply_nsfw_patch.py`, roll back, and re-run.
