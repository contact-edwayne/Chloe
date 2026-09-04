# Design prompt — Chloe emulator panel (paste into Claude Design)

> Copy everything below the line into Claude Design. It's the visual brief, the
> EmulatorJS integration, and the backend contract the panel must speak. The
> backend (ROM library endpoints) is already built and live — this prompt is
> only for the front-end panel. The result should feel like a sibling of the
> existing Chloe **chess** panel (same cyber-gothic cockpit look).

---

Design a single, self-contained retro-game **emulator panel** — one HTML file
with inline CSS and vanilla JavaScript (no build step, no frameworks, no
localStorage). It embeds into an existing dark "command cockpit" HUD as an
iframe, so it must look native to that world — not like a generic emulator site.

## Aesthetic (match the existing Chloe chess panel)

This is a game panel for **Chloe**, a personal AI companion. Reuse the exact
visual language of her chess panel so the two feel like a set:

- **Palette:** deep near-black backgrounds (#06070b / #0a0b10 / #0d0f16),
  **glowing cyan** primary accent (#22d3ee / #00e5ff), soft **violet/pink**
  secondary accents (#b06cff / #ff6ec7) used sparingly for Chloe's voice and
  highlights.
- **Type:** glowing monospace (JetBrains Mono feel) for labels/telemetry; a
  clean sans (Inter) for body; an italic serif (Cormorant Garamond) for Chloe's
  spoken lines.
- **Texture:** dark glassy panels, hairline cyan borders, faint scanline sheen,
  small corner filigree, clip-path bevelled buttons. Restrained lolita-tech
  elegance. No bland corporate/Material styling.
- **Motion:** smooth and tasteful. A small animated orb in the header (a
  glowing cyan/violet sphere) that gently pulses; subtle glow on active items.

## Layout

A header, a left **library/loader** rail, and a large **emulator viewport** as
the centerpiece.

1. **Header** — a small pulsing orb at top-left, an identity line
   (e.g. `CHLOE // ARCADE` · `EMU CORE` · `LINK STABLE`), and a session/status
   readout. Same header treatment as the chess panel.
2. **System filter** — chips/segments for the supported systems:
   **NES, Game Boy / GBC, GBA, SNES, Genesis**. Selecting one filters the
   library list.
3. **ROM library** — a scrollable list fetched from the backend (see contract).
   Each row: game name, a small system tag, and size. Click a row to load and
   play it. Group or tag by system; show an empty state ("no ROMs — drop files
   in your ROM folder") when the list is empty.
4. **Load from file** — a file-picker button ("LOAD ROM…") to play a local ROM
   that isn't in the library. Accept `.nes .fds .gb .gbc .gba .sfc .smc .md
   .gen .smd .bin`.
5. **Emulator viewport** — where the game renders (the EmulatorJS player mounts
   here). Before a game is chosen, show a tasteful idle/standby state. Controls
   (pause, reset, save state, fullscreen, volume) come from EmulatorJS's own
   in-canvas menu — you don't need to rebuild them, just give the player room.
6. **Chloe voice strip** (reserve a slim area, like the chess "says" strip) —
   leave a styled, normally-empty line where Chloe's commentary will appear
   later. It can stay empty for now; just lay out the space in her accent style.

## Emulator: EmulatorJS integration (this is the engine)

Use **EmulatorJS** (libretro cores compiled to WebAssembly) from its CDN. The
canonical embed: set config globals, then inject the loader script.

```js
// container element the player mounts into (e.g. <div id="emu"></div>)
window.EJS_player      = '#emu';
window.EJS_core        = core;            // see system → core map below
window.EJS_gameUrl     = romUrl;          // backend URL or a blob: URL (file picker)
window.EJS_pathtodata  = 'https://cdn.emulatorjs.org/stable/data/';
window.EJS_startOnLoaded = true;
window.EJS_gameName    = displayName;     // optional, nice to set
// then load the loader (after the globals are set):
const s = document.createElement('script');
s.src = 'https://cdn.emulatorjs.org/stable/data/loader.js';
document.body.appendChild(s);
```

**Viewport sizing — CRITICAL (the last build had audio but a black screen).**
The element EmulatorJS mounts into (`#emu` / `EJS_player`) MUST have a definite,
non-zero width AND height that does **not** derive from its own children. If the
container is auto/content-height, EmulatorJS sizes the canvas to the container
while the container sizes to the canvas → a ResizeObserver feedback loop, the
canvas collapses to ~0px, and you get sound with no picture. Requirements:

- The emulator viewport MUST fill all remaining vertical space: a flex child
  with `flex: 1; min-height: 0` inside a full-height flex column. **Do NOT use
  `aspect-ratio` or a fixed-height box for it** — that shrink-wraps the player to
  the game and docks EmulatorJS's control bar at the game's bottom edge, which
  lands in the MIDDLE of the screen (this is the current bug). Let the game
  letterbox inside the full-height viewport instead.
- The mount element is `position: relative; width: 100%; height: 100%;
  overflow: hidden;` — let EmulatorJS fill it. Never size the container FROM the
  canvas.
- If you attach a ResizeObserver yourself, debounce it (rAF) and never write a
  layout property the observer reads, or you reintroduce the loop.

Sanity check: at typical HUD size, `#emu` must have a clearly non-zero pixel
height *before* `loader.js` runs.

**Control bar at the BOTTOM (not floating mid-screen) — exact cause.**
EmulatorJS's bar is `.ejs_menu_bar`, already `position:absolute; bottom:0` of
its parent `.ejs_parent` (which is `height:100%`). The bar is fine. The bug is
that the emulator AREA isn't full height: the whole panel must fill the iframe's
height top-to-bottom. Right now the layout stalls at fixed floors (the main grid
~620px, the emulator frame `min-height:540px`), so the emulator only reaches
~540px and the bar correctly docks to the bottom of THAT — which is the middle
of a taller screen. FIX: the panel **root must be `100dvh` tall as a flex
column** so the main grid / viewport region grows to fill the full height
(keep the `min-height` as a floor, don't remove it — it prevents the canvas
collapse loop). Verify by hovering at full window size: the bar must sit at the
true bottom edge.

**Save / load state (currently does nothing on click — fix it).** Clicking the
Save State disk icon produces no response. Make in-browser save states actually
work: set `window.EJS_disableDatabases = false` (databases ON) and do NOT set a
sandbox attribute on anything that would block storage. The host browser now
allows localStorage/IndexedDB and downloads, so EmulatorJS' state slots should
persist. Verify end to end: Save State → make an in-game change → Load State
restores it, and a state survives a panel reload. Also override the bar CSS as a
safety net so it pins to the bottom: `.ejs_menu_bar { position: absolute
!important; bottom: 0 !important; top: auto !important; }` (EmulatorJS' control
bar is `.ejs_menu_bar`).

**System → core map** (use exactly these core strings):

| system (from backend / extension) | EJS_core |
|---|---|
| NES (`.nes .fds`) | `nes` |
| Game Boy / GBC (`.gb .gbc .sgb`) | `gb` |
| GBA (`.gba`) | `gba` |
| SNES (`.sfc .smc`) | `snes` |
| Genesis (`.md .gen .smd .bin`) | `segaMD` |

**Switching games:** EmulatorJS is built for one game per instance. To load a
different ROM, fully tear down first: remove the old player container `#emu`
(and the EmulatorJS-injected DOM), create a fresh empty `#emu`, reset the
`EJS_*` globals, then re-inject `loader.js`. Don't try to hot-swap the ROM on a
live instance. (A clean way is to wrap the viewport in a wrapper div and
`wrapper.innerHTML = '<div id="emu"></div>'` before each load.)

**Input:** EmulatorJS handles keyboard and gamepad mapping itself; no work
needed beyond letting its menu open.

## Backend contract (already live — do not invent endpoints)

The panel is served from, and talks to, the **same origin**, so use
**relative URLs only** (no leading slash) — this is required. On the desktop HUD
the panel is served at `http://<host>:6790/…`; on the iPhone PWA it's served
behind an HTTPS reverse-proxy prefix (`https://<host>/brain-api/…`). A
root-relative URL like `/api/roms` breaks under that prefix, but a relative one
like `api/roms` resolves correctly in BOTH. So fetch with `fetch('api/roms')`,
and build the ROM URL with
`new URL('roms/' + encodeURIComponent(file), document.baseURI).href`.

- **List ROMs:** `GET /api/roms` →
  ```json
  { "roms": [ { "name": "Super Mario Bros.nes", "file": "Super Mario Bros.nes",
                "system": "nes", "size": 40960 } ], "dir": "C:\\Chloe\\roms" }
  ```
  `system` is one of `nes | gb | gba | snes | segaMD` — use it directly as
  `EJS_core`. Render `name`, the system tag, and `size` (format as KB/MB).
- **Play a library ROM:** `EJS_gameUrl = new URL('roms/' + encodeURIComponent(file), document.baseURI).href`.
  (Relative `roms/<file>`; the backend streams the ROM bytes there.)
- **Play a picked file:** `EJS_gameUrl = URL.createObjectURL(file)`, and infer
  `EJS_core` from the file extension using the map above.

That's the whole contract — list, then point `EJS_gameUrl` at either a
`/roms/<file>` URL or a blob URL, and set the matching core.

## Talk to Chloe (in-panel chat)

Add a slim **"TALK TO CHLOE"** strip at the bottom of the panel (below the
emulator) — a text input plus a small scrollable log, styled like the chess
panel's talk-back. Open a websocket to the backend:

```js
const wsUrl = (location.protocol === 'https:')
  ? `wss://${location.host}/chloe-ws`
  : `ws://${location.hostname || 'localhost'}:6789`;
```

On submit, send `{ type: 'chat', messages: [ ...short rolling history,
{ role: 'user', content: text } ] }` — do **not** set `reply_audio` (Chloe
speaks the reply aloud on the PC). Stream her reply off the socket: `start`
(begin a line), `delta` (append `msg.text`), `done` (finalize). Show your
message and her streamed reply in the log; keep ~12 messages of history for
context.

**Unify with her watch commentary (important).** Also handle incoming
`{ "type": "game_comment", "text": "…" }` on the SAME socket: append the text to
the SAME chat log as a Chloe message, AND push `{ role: 'assistant', content:
text }` into the same rolling history. Those are Chloe's spoken reactions while
she watches Ed play — they must land in this one chat thread and become part of
its history, so the typed conversation and the live commentary are a single
continuous session, not two separate ones. Ignore other unrecognized message
types.

## Deliverable

One self-contained HTML file (inline CSS + JS) I can drop into the HUD as an
iframe panel, matching the chess panel's aesthetic. Fetch `/api/roms` on load to
populate the library; wire the file picker; mount EmulatorJS per the integration
above. Prioritize the cockpit aesthetic and a clean "pick a game → it plays"
flow. The backend contract is fixed, so build the UI around it.
