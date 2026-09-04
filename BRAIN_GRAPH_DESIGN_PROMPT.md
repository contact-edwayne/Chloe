# Design Prompt: Chloe Brain Graph Visualization (v3, 2026-05-18)

Paste this entire file into a fresh Claude.ai conversation. The end of the
prompt asks for a single-file HTML artifact you can drop into Chloe's
`jarvis/brain-graph.html`. No prior context required.

If you've designed v2 of this — discard the bake-ins from that session. The
backend has changed (v2 JSON schema, two new endpoints, an SSE stream) and
the feature set is larger. Build fresh against the spec below.

---

## You are designing a visualization for Chloe, a personal AI assistant.

### What Chloe is

Chloe is a desktop voice/chat assistant — a portfolio piece for AI
engineering work and also a daily-driver. Sci-fi HUD aesthetic: cyan-on-
near-black, `Share Tech Mono` for body, `Orbitron` for headings, animated
grid backdrop, glowing rings around a central avatar orb. J.A.R.V.I.S.
mood, not Apple-glassmorphism. Don't use Material, shadcn, rounded-card
patterns.

### What the brain is

Chloe maintains a personal knowledge base built on Andrej Karpathy's
"LLM Wiki" pattern: a folder of markdown files where every ingested source
is broken into entity pages, concept pages, and source pages,
cross-referenced with `[[wikilinks]]`. Pages live under
`wiki/entities/`, `wiki/concepts/`, `wiki/sources/`, `wiki/comparisons/`,
`wiki/explorations/`, plus `daily/<YYYY-MM-DD>.md` daily notes from
Obsidian and `episodic/CONTEXT-*.md` auto-generated context files. There's
also a sibling `facts/*.md` directory the UI can opt into.

Same shape as a human-maintained Obsidian vault — except an LLM
maintains it, with assists from a wiki_watcher polling the filesystem
and re-embedding on save.

### What the visualization is

A graph view of the brain. Every markdown file is a node, every
`[[wikilink]]` is an edge. Opened from a "BRAIN" button on Chloe's main
HUD; shown as a fullscreen overlay inside an iframe; closeable with Esc.

The previous version reached the "suspended-in-cyan-space" aesthetic well
but the brain has been growing fast — daily ingest, autonomous research,
Brave search persistence, finance news, queue processor output. Naive
node-and-edge drawing is starting to read as noise. The v3 design needs to
help with **zooming out**, **filtering**, and **active editing** (drop in
a file or URL → see where it lands in the graph), in addition to looking
like Chloe's HUD.

---

### Current backend (consume this — don't change it)

A small Python HTTP server at `http://localhost:6790` exposes:

#### `GET /api/brain/graph`
Optional query: `?include_facts=1` (mounts `facts/*.md` as nodes under a
`facts/` prefix), `?since_ts=<epoch>` (filters to nodes with mtime ≥ ts —
powers the time slider).

Response:

```json
{
  "nodes": [
    {
      "id":          "entities/qmd",
      "label":       "qmd",
      "type":        "entities",
      "real":        true,
      "in_deg":      3,
      "out_deg":     2,
      "degree":      5,
      "size":        16,
      "mtime":       1747500000.0,
      "body_size":   2840,
      "source_type": "chloe_generated",
      "fm_type":     "",
      "orphan":      false
    }
  ],
  "edges": [{ "source": "entities/qmd", "target": "concepts/rag" }],
  "stats": {
    "node_count":     142,
    "edge_count":     310,
    "real_count":     108,
    "ghost_count":    34,
    "orphan_count":   17,
    "type_counts":    {"entities": 41, "concepts": 38, "sources": 22, "daily": 7, "...": "..."},
    "source_counts":  {"chloe_generated": 79, "brave": 14, "dropped_in": 3, "...": "..."},
    "recent_changes": [{"id": "concepts/iv_crush", "mtime": 1747509999.0, "source_type": "chloe_generated"}],
    "embed_coverage": {"embedded": 108, "total": 108, "ratio": 1.0},
    "last_event":     {"type": "upserted", "node_id": "concepts/iv_crush", "ts": 1747509999.0},
    "computed_at":    1747510020.0,
    "include_facts":  false,
    "since_ts":       null
  }
}
```

Node `type` values: `entities | concepts | sources | comparisons |
explorations | daily | episodic | facts | other | ghost`. Ghost = the
target of a `[[link]]` that has no actual file.

Node `source_type` values (verified against live data 2026-05-18):
`chloe_generated | web_search | dropped_in | finance_news | daily |
episodic | external | ghost | unknown`. Drives badge color. The dominant
two by volume on the live brain are `chloe_generated` (LLM-authored
entity/concept/source pages) and `ghost` (broken `[[wikilinks]]` —
currently ~72% of nodes, a known brain-pollution issue).

#### `GET /api/brain/stats`
Same `stats` block, no node/edge arrays. Use this for cheap polling in the
brain-stats pane (every ~10s) so the full graph payload doesn't get
re-fetched on a heartbeat schedule.

#### `GET /api/brain/page?p=<rel_id>`
Returns `{ok, text, frontmatter, path, mtime}`. `text` is raw markdown.
`frontmatter` is a flat key-value dict parsed from the leading `--- ... ---`
block.

#### `GET /api/brain/events`  (Server-Sent Events)
Long-lived `text/event-stream`. Events:

- `event: hello` — `{ts}` on subscribe.
- `event: upserted` — `{type, node_id, status, ts}` whenever the
  wiki_watcher detects a filesystem change (Obsidian edit, autonomous
  ingest, etc.).
- `event: deleted` — `{type, node_id, ts}`.
- `event: ingested` — `{type, node_id, slug, entities_touched,
  concepts_touched, similar, ts}` whenever a drop-in ingest commits.
- Comment heartbeats (`: heartbeat`) every 15s to keep proxies honest.

Use `new EventSource('/api/brain/events')`. The UI should fan these out to:
- Brief opacity-pulse ring on the affected node ("recently touched").
- Watcher events ticker at the bottom of the brain-stats pane.
- On `ingested`: animate ghost-node materialization + outbound edges to
  the `similar` ids.

#### `POST /api/brain/ingest`
Two body shapes:

JSON:
```json
{
  "url":     "https://example.com/article",
  "text":    "alternative — paste content directly",
  "title":   "optional hint",
  "dry_run": false
}
```

Or `multipart/form-data` with fields `file` (the upload), optional
`title`, optional `dry_run`.

Response:
```json
{
  "slug":              "example_article",
  "tldr":              "...",
  "entities_touched":  ["foo", "bar"],
  "concepts_touched":  ["baz"],
  "similar":           [{"id": "concepts/rag", "score": 0.81, "title": "...", "type": "concepts"}],
  "raw_path":          "C:\\Chloe\\brain\\raw\\dropin_..._20260518_103245.md",
  "url":               "https://example.com/...",
  "title":             "...",
  "dry_run":           false
}
```

For `dry_run: true`, response includes `dry_run: true` and
`entities_status`/`concepts_status` lists of `[slug, CREATE|UPDATE]`
tuples instead of `entities_touched`/`concepts_touched`. The raw file is
still written either way — discard via DELETE if the preview looks wrong.

#### `DELETE /api/brain/ingest?slug=<slug>`
Removes `wiki/sources/<slug>.md` and matching `raw/dropin_*<slug>*.md`.
Returns `{ok, removed, not_reverted_entity_concept_pages, note}`. Entity
and concept pages that mention the slug are listed but NOT reverted —
they may have been merge-updated and aren't safely reversible. Surface
this to the user in the discard confirmation.

---

### Aesthetic goals (preserve from v2)

1. **Suspended in space.** Orbs drift on independent paths. Slow currents,
   not chaos. Like jellyfish or stars.
2. **Glowing avatars, not flat dots.** Each orb is a small sphere: cyan
   core, off-axis highlight, soft halo, subtle pulse. Hubs glow stronger
   than periphery.
3. **Monochrome cyan, not rainbow.** Real pages stay in the cyan family
   (hue ~165–205). Type differentiation comes from very small hue shifts
   plus luminosity. Ghosts are muted blue-gray, recessed.
4. **Importance is brightness.** Higher-degree pages pulse bigger and glow
   brighter. The eye lands on hubs first.
5. **Background has slight motion.** Subtle grid + radial gradient, drifts
   over ~30s. Field feels alive even on a still graph.
6. **HUD-grade chrome.** Topbar with brand mark + stats + legend.
   Frosted-glass panels with cyan borders. Buttons monospace, with hover
   glow. No Material, no rounded everything, no shadcn.

---

### New features for v3 — implement all of these

Order roughly outside-in (chrome → graph → interactions). Take liberties
with positioning as long as nothing collides.

#### A. Zoom-out / navigation

1. **Minimap.** Corner thumbnail (bottom-right is fine). Full graph at
   reduced opacity, viewport rectangle marked. Click-drag the rectangle to
   pan; click anywhere on the minimap to recenter. Updates on graph
   transform changes.

2. **LOD clustering.** Below a zoom threshold, collapse nodes of the same
   `type` that sit within a screen-space radius into a super-node showing
   the count (e.g. "entities ×17"). As the user zooms in, super-nodes
   explode back into their constituents. Use a simple grid-bucketing
   approach — k-means at every frame is too expensive for 60fps. Animate
   the transitions; don't pop.

3. **Type filter chips.** Top-bar pill row showing each `type` with its
   count and source color swatch. Click to toggle visibility. Visible
   types desaturate-fade-out rather than vanish (~250ms) so the user
   doesn't lose orientation. Include a chip for `ghost` and one for
   `orphan` (logical filter, not a type) so cleanup work is one-click
   away.

4. **2D ⇄ 3D toggle.** Top-right corner button. 2D mode is force-directed
   in the plane (faster, easier to grep). 3D uses three.js with
   gentle camera drift. Both modes share the same node/edge data — only
   the layout layer changes.

5. **Search-to-focus.** Top-bar input. As the user types, fuzzy-match
   `label` + `id`; matching nodes brighten, others dim. Hit Enter to
   pan/zoom the camera to the matched-node centroid. Esc clears.

6. **Layout toggle.** Force-directed (default) vs. hierarchical-by-type
   (entities ↑ concepts ↑ sources, downward edges). The hierarchical
   layout is for "how does this concept depend on its sources" reading.
   Animate the rearrangement, ~600ms.

#### B. Brain hygiene + insight

7. **Orphan highlight.** One-click filter: only nodes with `orphan: true`
   are visible. Useful for cleanup — Chloe's brain currently has ~75% of
   nodes as ghost-or-orphan from a 2026-05-10 cleanup pass, and this is
   the surface that lets the user actually act on it.

8. **Brain-stats pane.** Collapsible panel (bottom-left). Reads from
   `/api/brain/stats` every 10s. Shows: total node/edge counts, real vs
   ghost vs orphan, type breakdown, source breakdown, embed coverage
   (e.g. "108/108 embedded"), last event from SSE
   (`"upserted concepts/iv_crush • 12s ago"`). The pane also hosts:

9. **Watcher events ticker.** Last ~5 SSE events as a scrolling line list.
   Newest at top, fade older entries. Clicking an entry pans the camera to
   that node.

#### C. Drop-in ingest panel

10. **Drop-in panel.** Top-right corner (below the layout/2D toggles).
    Two affordances:

    - **Drag-drop file zone** ("Drop a .md, .txt, .html, or URL here").
      Accepts files. Also accepts dragged URLs (read from
      `DataTransfer.getData('text/uri-list')` or `text/plain`).
    - **URL input** + "Ingest" button. Hitting Enter does dry-run first.

    Flow:
    1. POST `/api/brain/ingest` with `dry_run: true` first. Show a
       pre-commit card overlay: would-be slug, tldr, list of
       `entities_status` and `concepts_status` (CREATE = green, UPDATE =
       cyan), and the top-5 `similar` matches as clickable chips.
    2. User clicks "Commit" → POST again with `dry_run: false`. Or
       "Discard" → DELETE `/api/brain/ingest?slug=<slug>` (which actually
       deletes the still-on-disk raw + source page).
    3. On commit, a **ghost node materializes** near the camera (or near
       the highest-scoring `similar` node), animates into place via the
       force layout, and animates inbound edges to the `similar` ids over
       ~800ms. The `ingested` SSE event provides the same payload so the
       same code path runs whether the ingest came from this UI or from
       another client.

    The pre-commit card is the missing GUI for `/ingest --dry-run`.

#### D. Liveness + provenance

11. **Recently-edited pulse.** On every `upserted` SSE event, the affected
    node gets a brief halo ring expansion (~600ms) so the user sees
    Obsidian / autonomous edits land in real time. Throttle to one ring
    per node per second so a backfill burst doesn't strobe.

12. **Edit-source badge.** Each node gets a small badge dot — color from
    `source_type`. Suggested palette (all within the cyan family except
    web_search/dropped_in/finance_news which need to be visually
    distinct):

    | source_type      | badge color   | rationale                       |
    |------------------|---------------|---------------------------------|
    | chloe_generated  | cyan          | Chloe's autonomous work         |
    | web_search       | warm amber    | Brave-fetched web result        |
    | dropped_in       | violet        | User upload via drag-drop UI    |
    | finance_news     | seafoam green | Daily finance digest            |
    | daily            | pale yellow   | Obsidian daily note             |
    | episodic         | warm gray     | CONTEXT files                   |
    | external         | muted teal    | Other sources                   |
    | ghost            | dim gray      | Broken link                     |
    | unknown          | dim gray      | Fallback (shouldn't be common)  |

    Badge sits on the orb's lower-right, ~25% of orb diameter.

13. **Provenance trail.** Clicking a node opens the side panel (markdown
    via `marked`) AND highlights the provenance chain in the graph:
    upstream → the source page(s) referenced in this page's
    `[[wikilinks]]` and frontmatter; downstream → every page whose body
    references this node. Edges in the chain animate flow (dotted line,
    slow march). Non-chain nodes/edges dim to ~30% opacity. Click another
    node to switch; Esc clears.

14. **Time slider.** Bottom-strip slider. Left end = oldest mtime in the
    graph, right end = newest. Sliding left re-fetches
    `/api/brain/graph?since_ts=<epoch>` so the graph shows only nodes
    created/edited from that point onward. Useful for "what's been added
    this week / month". Debounce the fetch (~300ms after release). Show
    the corresponding date label as the user drags.

15. **Cluster chat.** Shift-click multiple nodes to multi-select. A
    floating "Ask about these N pages" button appears. Clicking it sends
    `window.parent.postMessage({type: 'brain_cluster_chat', context_ids:
    [...]})` — the HUD already routes brain-panel chat via this
    mechanism. Don't open your own WS.

#### E. Pre-existing fixes (must land in v3)

16. **FOUT fix.** Don't put `opacity: 0` on `body`. Wrap your app in a
    `<div id="app" class="opacity-0">` and toggle that. The previous regen
    buried the body opacity rule inside a base64-gzipped bundler payload
    that wouldn't apply on initial paint, leading to a flash.

17. **Markdown rendering in side panel via `marked` CDN.** The previous
    regen rendered raw markdown as `<pre>`. Use `marked.parse(text)` with
    a permissive sanitization pass — these pages are first-party.

18. **Status pill consistency.** The bottom-left "status pill" must match
    the main HUD's pill style — same border-radius, same cyan border,
    same typography (`Share Tech Mono`, 11px, letter-spacing tight). The
    HUD pill template:
    ```
    background: rgba(0, 14, 28, 0.85);
    border: 1px solid rgba(0, 220, 255, 0.35);
    border-radius: 4px;
    padding: 6px 10px;
    color: #b5f0ff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    ```

19. **Visual polish from v2 notes:**
    - Orbs as realized spheres: radial gradient (cyan core, off-axis
      highlight at ~25%/30% NW, soft outer halo).
    - More obvious drift — bump amplitude from ~22px to ~40px for hubs,
      keep frequencies low.
    - Layered depth — parallax particle field behind the graph (a few
      hundred faint dots drifting at 30% of graph velocity); slight
      distance-fog gradient toward edges.
    - Edge motion — short cyan flow dashes along edges, ~1px wide,
      opacity pulse from 0.2 → 0.5 → 0.2 over 2s, offset per-edge so
      they're not synchronized. Performance budget: must hold 60fps on a
      300-edge graph.

#### F. Facts as nodes

20. **Opt-in facts layer.** A toggle in the top-bar (label: "facts" with
    a tiny dot). When on, re-fetch
    `/api/brain/graph?include_facts=1`. Facts pages mount under a
    synthetic `facts/` prefix. Style them with a slightly different shape
    (e.g. small cyan square instead of sphere, or a thicker ring) so
    they're visually distinguishable from wiki pages.

---

### Hard constraints

- **Single self-contained HTML file.** All CSS/JS inline or via CDN. No
  build step. No npm.
- **No `localStorage` / `sessionStorage`.** All state in memory.
- **60fps target** on a 250-node, 400-edge graph on modest hardware.
  Lazy-render off-screen orbs.
- **Iframe-safe.** Lives inside hud.html. Communicate with the parent via
  `window.parent.postMessage(...)` only.
- **Allowed CDN libraries:** vis-network, marked, lucide, d3, three.js,
  Chart.js, recharts, Tone.js, Tailwind utility classes (no compiler).
  You may swap vis-network for d3 or three.js if a better look results,
  but click-node-to-read-page-in-side-panel must still work.
- **Backend endpoints are fixed.** Match the schemas above exactly. If
  the backend is unreachable, fall back to baked-in mock data so the
  artifact runs in Claude.ai's preview pane.

### Mock data for offline preview

Bake in ~25 mock nodes with mixed types and source_types, including:
- 2 designated hubs (degree ≥ 12).
- 3 ghosts.
- 4 orphans (real, degree ≤ 1).
- 1 recently-edited node (mtime within the last 60s).
- 1 `dropped_in` source.
- 1 `brave` source.
- 1 `daily` page.

Mock the SSE stream by emitting a synthetic `upserted` event every ~8s on
a random node. Mock the ingest POST by waiting 1.5s and returning a
plausible response (or just open the pre-commit card directly).

---

### What I want from you

A single self-contained HTML artifact that hits all aesthetic goals from
v2 plus implements every numbered feature 1–20 in this prompt. After the
artifact, in 3 short paragraphs:

1. **What you cut or simplified, and why.** (Reality check: this is a
   lot for one file.)
2. **Where the 60fps budget gets tight** and what you did to stay inside
   it. Be specific about the worst-case interaction (e.g. opening the
   minimap while the time slider is animating + SSE is firing).
3. **What you'd build in v4** if I gave you another iteration. Top 3 in
   priority order.

Don't preamble. Don't apologize for size. Get to the design.
