# TTS tonal shifts — apply + verify

## What this adds
Chloe can shift her TTS voice per scene by emitting a tone tag at the start of a reply: `[intimate]`, `[whispering]`, `[submissive]`, `[sultry]`, `[breathy]`, `[playful]`, or `[neutral]`. The tag is stripped before TTS speaks it. Voice + speed map per-tone via `tts_tones.PALETTE`. Sticky across turns. Auto-resets to neutral when `/nsfw` flips off.

## Order of operations
**Apply `apply_nsfw_patch.py` first.** This script depends on the `import nsfw_mode` line it inserts. The tones script aborts cleanly if NSFW isn't applied.

## Files in place
- `jarvis\tts_tones.py` — palette + parse + sticky state + `/tone` slash handler
- `jarvis\chloe_about.md` — added "Voice tone tags" section (basic palette, always-on)
- `jarvis\chloe_nsfw.md` — added "Voice tone tags (permissive palette)" subsection
- `jarvis\nsfw_mode.py` — hooked: `set_enabled(False)` now calls `tts_tones.reset_tone()`
- `jarvis\apply_tones_patch.py` — splice script

## To apply
```
cd C:\Users\eleew\Documents\jarvis
python apply_tones_patch.py
```

Same safety harness as the NSFW splice: idempotent, backed up, ast.parse-validated, tail-diff output. Aborts on any anchor miss before touching the file.

## Patches applied to jarvis.py
1. `import tts_tones` after `import nsfw_mode`
2. `/tone` chat slash handler (status/reset/<name>)
3. `_kokoro_to_wav_bytes` — parses leading tone tag, uses palette voice + speed
4. `_speak_kokoro` — same, but parses once before the sentence-split loop so all chunks in a reply share the tone

## Verify (after restart)
```
stop_chloe.bat
start_chloe.vbs
```

1. `/tone status` → "tone: neutral"
2. `/tone playful` → "tone set to playful." Next chat reply: should sound quicker, higher-energy (af_nicole at 1.10x).
3. `/tone reset` → back to neutral.
4. `/nsfw on`, then ask something that calls for intimacy in a story. Chloe should emit `[intimate]` at the start — invisible in spoken audio, but voice shifts to af_sky at 0.88x.
5. `/nsfw off` → tone auto-resets to neutral (verify with `/tone status` after).

## Tuning the palette
Open `tts_tones.py` and edit `PALETTE`. Each entry is `(voice_id, speed)`. Voice IDs come from Kokoro's voice catalog (af_bella, af_sky, af_nicole, af_sarah, af_jessica, etc.). Speed is a float, conservatively 0.80–1.20.

Common knobs:
- "Sounds too slow" → bump speed value up (e.g. intimate 0.88 → 0.92)
- "Voice isn't right" → swap voice_id (e.g. try af_nicole for intimate instead of af_sky)
- "Want a new tone" → add a new key with a (voice, speed) tuple AND teach Chloe about it via chloe_about.md or chloe_nsfw.md

No restart needed — `tts_tones.parse_and_get()` reads PALETTE on every call.

## Edge cases handled
- **Markdown-wrapped tag**: `**[intimate]**` parses correctly (regex tolerates bold/italic wrappers).
- **Unknown tag**: stripped from text (so it's never spoken), sticky tone left unchanged. Logged for tuning.
- **Tag mid-sentence**: only leading tags are parsed. Mid-sentence tags are ignored (kept as text) — by design.
- **No tag emitted**: sticky tone from previous turn carries over.
- **Permissive flips off mid-scene**: nsfw_mode.set_enabled(False) calls reset_tone() to snap back to neutral.

## Rollback
```
copy /Y jarvis.py.bak.<ts> jarvis.py
del tts_tones.py apply_tones_patch.py APPLY_TONES.md
```

Reverse the chloe_about.md / chloe_nsfw.md edits by hand (remove the "Voice tone tags" sections).
