# Design prompt — Chloe chess panel (paste into Claude Design)

> Copy everything below the line into Claude Design. It describes the visual
> brief, the interaction spec, and the exact data contract the panel must speak
> so it plugs into Chloe's existing backend with no changes. The backend
> (`chloe_chess.py` + the `game_*` WebSocket handlers) is already built — this
> prompt is only for the front-end panel.

---

Design a single, self-contained chess game panel — one HTML file with inline
CSS and vanilla JavaScript (no build step, no external frameworks, no
localStorage). It will be embedded into an existing dark "command cockpit" HUD,
so it must look native to that world, not like a generic chess site.

## Who it's for and the aesthetic

This is the game panel for **Chloe**, a personal AI companion with a strong
visual identity. The aesthetic is **cyber-gothic-lolita meets sci-fi command
interface**:

- **Palette:** deep near-black backgrounds (#0a0b10 / #0d0f16), **glowing cyan**
  as the primary accent (around #00e5ff / #22d3ee), with soft secondary accents
  of **violet and pink** (#b06cff, #ff6ec7). Use the pink/violet sparingly, for
  highlights and Chloe's own voice.
- **Type:** a glowing monospace / techno font for labels, coordinates, and
  readouts (think cockpit telemetry). A clean sans for Chloe's commentary line.
- **Texture:** dark glassy panels with subtle inner glow and thin 1px cyan
  borders; faint scanline or grid sheen is welcome but must stay subtle.
  Delicate, lolita-flavored detailing — hairline frames, a small ribbon or
  filigree motif at a corner or on the header — without becoming cute-overload.
  Restrained, elegant, high-tech.
- **Motion:** smooth, tasteful. Pieces glide between squares (short ease). The
  last move and the selected square glow. A soft pulse when it's Chloe's turn
  ("Chloe is thinking…"). No bouncy/cartoonish animation.
- **Mood:** confident, sleek, a little dark and romantic. It should feel like
  playing chess against an intelligent presence inside a spaceship cockpit.
  Absolutely avoid bland corporate / Material-default styling.

## What the panel contains

1. **The board** — an 8×8 chess board, coordinate labels (files a–h, ranks
   1–8) in the cockpit font. Light/dark squares should read as a dark, tasteful
   pair (e.g. desaturated slate vs. near-black), NOT classic brown/cream. The
   board can flip so the player's color is always at the bottom.
2. **Pieces** — render from the data (see contract). You may use the provided
   Unicode glyphs, or your own SVG piece set if it suits the aesthetic better
   (a thin, glowing, monochrome-with-accent set would fit beautifully — white
   pieces in cyan/white glow, black pieces in violet/ink).
3. **Chloe's voice strip** — a dedicated area where Chloe's one-line commentary
   (`says`) appears, styled as *her* speaking (her accent color, maybe a small
   avatar/orb). Lowercase, in-character. Empty most of the time; populated at
   game start/end and on notable moments.
4. **Status + result** — a readout for check / checkmate / stalemate / draw,
   and an end-of-game banner ("you win" / "Chloe wins" / "draw") in-aesthetic.
5. **Difficulty indicator** — shows the current level **1–5** and signals that
   it **adapts automatically** to keep games close (e.g. a 5-segment gauge
   labeled "ADAPTIVE"). Read-only; the backend sets it.
6. **Evaluation bar** — a slim vertical or horizontal bar showing who's ahead,
   driven by `eval_cp` (centipawns, + = White ahead). Subtle, optional to label.
7. **Move list** — the running game in SAN (`moves_san`), scrollable, paired by
   move number, cockpit font.
8. **Controls** — "New game" (with a choice of playing **White or Black**) and
   "Resign". Keep them minimal and on-aesthetic.

## Interaction

- **Click-to-move** (primary): click your piece → its legal destination squares
  glow → click a destination to move. Clicking elsewhere/again deselects.
  (Drag-and-drop is a nice-to-have on top, not required.)
- Only allow input when it's the player's turn (`turn === "player"`). While
  Chloe thinks, show the thinking state and ignore board input.
- **Promotion:** if a pawn move reaches the last rank, ask which piece (queen
  default) and send the move with the promotion suffix (e.g. `e7e8q`).
- Highlight the **last move** (both squares) and flag the **king in check**.
- On an illegal/rejected move, gently snap back and (optionally) flash the
  Chloe voice strip with a teasing note — but the backend is the source of
  truth, so always re-render from the state it returns.

## Data contract (must match exactly — do not invent fields)

The panel communicates over a single WebSocket that the host HUD already owns
(reuse `window.ws` if present; otherwise open `ws://${location.hostname}:6789`).
All messages are JSON.

**Send (panel → backend):**

- Start a game: `{"type":"game_new","player_white":true}`  (`false` = play Black)
- Make a move: `{"type":"game_move","move":"e2e4"}`  (UCI; promotion e.g. `"e7e8q"`)
- Resign: `{"type":"game_resign"}`
- Ask for current state (e.g. on panel open): `{"type":"game_state"}`

**Receive (backend → panel):**

- `{"type":"game_state_update", ...state, "says":"<chloe one-liner or empty>", "error":"<optional>"}`
- `{"type":"game_error","error":"<message>"}`  (e.g. no active game, dependency missing)

The `state` object inside `game_state_update` has exactly these fields:

| field | type | meaning |
|---|---|---|
| `fen` | string | full FEN (use this if you render with your own piece set / a chess lib) |
| `grid` | array[8][8] | rows top→bottom = rank 8 → rank 1; each cell is `""` or a Unicode glyph (♔♕♖♗♘♙ white, ♚♛♜♝♞♟ black) |
| `player_color` | `"white"`/`"black"` | the human's color (orient the board so this is at the bottom) |
| `turn` | `"player"`/`"chloe"` | whose move it is now |
| `legal_moves` | string[] | legal moves in UCI, **only populated when it's the player's turn** (use to highlight destinations: filter entries starting with the clicked square) |
| `last_move` | string/null | last move played, UCI (highlight both squares) |
| `moves_san` | string[] | full move history in SAN, for the move list |
| `status` | string | `"ongoing"`, `"check"`, `"checkmate"`, `"stalemate"`, `"draw_material"`, `"draw_repetition"`, `"draw_fifty"`, `"resigned"`, or `"no_game"` |
| `result` | string/null | from the player's POV: `"win"`, `"loss"`, `"draw"`, or `null` while ongoing |
| `in_check` | bool | a king is in check |
| `difficulty` | int 1–5 | current adaptive difficulty |
| `eval_cp` | int | static evaluation in centipawns, + = White ahead |
| `game_over` | bool | true when the game has ended |

Render the position from `grid` (simplest) or from `fen` (if you bring your own
piece set). Always treat each `game_state_update` as the full truth and
re-render from it — don't track game logic client-side beyond what's needed for
selection/highlighting. The backend enforces all rules.

## Deliverable

One self-contained HTML file (inline CSS + JS) that I can drop into the existing
HUD as a panel. Keep it modular enough to live alongside other HUD panels.
Prioritize the aesthetic and the click-to-move feel; the data wiring above is
fixed, so build the UI around it.
