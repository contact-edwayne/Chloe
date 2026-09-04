# Chloe — session handoff for Grok review (2026-06-07)

You are reviewing a working session on **Chloe**, Ed's local voice+chat home
assistant (Python on Windows; WebSocket HUD `hud.html` + iPhone PWA
`chloe-mobile.html` over Tailscale; hybrid local-Ollama/Groq LLM). This session
shipped four things. **Most of it could NOT be tested on the target device by
the implementer (no iOS browser in the sandbox), so adversarial review of the
untested paths is the point of this handoff.** Please scrutinize for bugs, race
conditions, and iOS-WebKit gotchas. Cite file + symbol. Push back hard.

Single source of truth = `chloe_handoff.md`. Lessons live in it (notably:
**bash mount of jarvis.py is a TRUNCATED mirror** — trust the editor, verify
Windows-side with `venv_py314\Scripts\python.exe -c "import ast,<mod>"`).

---

## 1. iPhone arcade: EmulatorJS → WasmBoy (the big one)

**Problem:** the iPhone arcade force-closed ~30s into play. We falsified the
prior "lighter EmulatorJS settings" theory by elimination: ruled out autosave
(disabled → still 30s), the HUD WebGL orb (fully torn down), EJS memory knobs
(threads/rewind/shader off), **core footprint** (crashes on gambatte, the
lightest core), and the **PWA container** (a plain Safari tab dies too).
Conclusion: EmulatorJS's RetroArch WebGL stack leaks GPU textures on iOS WebKit
regardless of game/container.

**Fix shipped + Ed-validated on iPhone:** `emulator_mobile.html` converted
in place from EmulatorJS to **WasmBoy** (`wasmboy@0.7.1` UMD,
`dist/wasmboy.wasm.umd.js`, **Canvas2D output, no WebGL**). All device-shell
UI / ROM drawer / chat sheet kept byte-for-byte; only the engine glue swapped.
GB/GBC only (WasmBoy is Game Boy-only; other systems show a "desktop only"
standby message). Standalone reference build `emulator_gb.html` also exists
(route `/emulator-gb.html` in `brain_http.py`).

**Review targets (these are the risky, lightly-tested bits):**
- **Input = synthesized KeyboardEvents.** WasmBoy's documented `setJoypadState`
  lib API was a silent no-op against the core (even after `disableDefaultJoypad`).
  We instead KEEP the default joypad enabled and dispatch synthetic
  `KeyboardEvent`s on `window`+`document` (responsive-gamepad matches on
  `event.code`: arrows=d-pad, KeyX=A, KeyZ=B, Enter=Start, Shift=Select, WASD as
  backups). See `emuInput`/`dispatchKey`/`KEY_CODES`. **Is relying on synthetic
  untrusted KeyboardEvents fragile across WasmBoy/responsive-gamepad versions?
  Any cleaner public API we missed?**
- **Canvas black-screen fix.** A leftover EJS rule `#emu canvas{...!important;
  object-fit:contain}` out-specified `#wbCanvas` → black screen (game ran, audio
  fine). Fixed with `canvas#wbCanvas{...!important}`. **Verify specificity logic
  and that we didn't just mask a deeper render-target issue.**
- **Audio.** `audioAccumulateSamples:false` (in `WB_OPTIONS`) fixed a ~1s delay;
  Ed confirms tight now. **Risk: crackle/underrun on busier scenes — acceptable
  tradeoff or should we trim the buffer differently?**
- **Lifecycle.** `WB.config` is called once (`wbConfigured` guard); `teardown`
  pauses; `loadROM(url)` lets WasmBoy fetch ROM bytes. **Re-entry / multi-game
  switching correctness?**
- **Save model.** WasmBoy auto-persists cartridge RAM (in-game saves) to
  IndexedDB; Save/Load buttons use `WB.saveState`/`getSaveStates`/`loadState`.
  Old EmulatorJS server-side `.state` saves are NOT portable (different core).

`node --check` passes on the inline script; backup
`emulator_mobile.html.bak.20260607_013635`.

## 2. Chat input queue (jarvis.py — APPLIED via proposal, live)

Goal: messages sent while a reply is in flight should queue + process in order,
not drop/race. **Finding:** hud_server's per-connection read loop already
awaits handlers (same-socket serialized); the actual dropper was the HUD client
(`if(busy)return;`). Fixes:
- `hud.html` + `chloe-mobile.html`: allow send-while-busy, dim "queued" marker.
- `jarvis.py`: `handle_chat` is now a **FIFO wrapper** (`_chat_busy` flag +
  `_chat_queue`) around `_handle_chat_inner` (original body unchanged), for
  cross-client ordering (HUD + PWA + arcade panel). `_splice_latest_assistant`
  inserts the newest `_voice_history` assistant turn into a payload that predates
  it. **Review: is the module-global `_chat_busy`/`_chat_queue` safe given
  these are asyncio coroutines on one loop (no true parallelism, but reentrancy)?
  Any path where the queue isn't drained / `_chat_busy` stays stuck on an
  exception? (drain is in a `try/finally`.) Starvation/ordering edge cases?**

## 3. Auto-return-to-listening (jarvis.py — APPLIED, live)

PTT replies now append the wake-path follow-up loop (`_next_turn_audio` +
`_process_voice_turn`, multi-turn) so an immediate follow-up needs no PTT/wake.
Window = `CHLOE_FOLLOWUP_S`. **Review: any double-`_speak` or HUD-state leak
between the PTT path and the borrowed wake-path helpers? Loop exit conditions?**

## 4. iPhone always-listen / hands-free VAD (chloe-mobile.html — client-only)

Ed asked for always-on listening on the iPhone (no PTT). Added an **AUTO**
toggle → RMS voice-activity detector over the existing AudioWorklet Float32
stream (`vadFeed`): 400ms rolling preroll, trigger `RMS≥0.022`
(`localStorage.chloe_vad_thresh`), 1.1s silence ends the utterance, ≥350ms
voiced floor, 30s cap; ships via the existing `ptt_audio` WS path. **Gated on
`window.__chloeState==='idle'` + 600ms tail so it can't self-trigger on her own
TTS** (mic also requests `echoCancellation`), and not during a manual PTT hold
or while a game overlay is open. Persists in localStorage; iOS gesture-gates the
mic so one tap re-arms per launch; foreground/screen-on only.

**Review targets:**
- **Self-trigger / echo:** is `__chloeState==='idle'` + 600ms tail + browser
  echoCancellation enough to stop her TTS (played through `replyAudioEl` on the
  same device, possibly on speaker) from re-triggering capture? This is the
  scariest failure mode (infinite self-conversation). What hardening would you add?
- **NO wake word** — any speech triggers (Madi, TV). Intended per Ed, but is a
  cheap always-on keyword gate worth offering?
- VAD threshold/timing defaults — reasonable for phone-on-desk speech?
- `node --check` passes on all 5 PWA script blocks.

---

## What's verified vs not
- **Verified by Ed on-device:** WasmBoy arcade (render/play>30s/controls/save/
  audio); chat queue + PTT follow-up applied (dry-run clean, restarted, symbols
  present in jarvis.py).
- **NOT yet verified on-device:** always-listen VAD (just shipped); chat-queue
  cross-client race under real concurrent load; PTT follow-up multi-turn feel.

## Open / TODO
- Synopsis cut still gated on a long (>30-turn) + voice session (see handoff A).
- `chloe_memory.py` recall-noise filter needs a restart + Windows ast check.
- Chess iPhone UI (`chess_panel.html` → mobile split) not started.
- Optional: WasmBoy old-save import (WRAM→SRAM reconstruct + IndexedDB inject) —
  Ed declined for now.

**Ask:** prioritize the self-trigger risk in §4 and the concurrency correctness
in §2 — those are the two places an unreviewed bug bites hardest.
