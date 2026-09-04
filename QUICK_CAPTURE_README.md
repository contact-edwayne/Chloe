# Quick capture — drop any URL or note into Chloe's brain

Shipped 2026-05-19 as automation priority #6. Three surfaces, one
backend endpoint.

## Backend

- `GET /quick-capture.html` — the capture page itself, served by
  `brain_http.py` next to `brain-graph.html`. Reads pre-filled values
  from URL params: `?u=` URL, `?t=` title, `?n=` notes.
- `POST /api/brain/ingest` — same endpoint the brain UI's drop-in
  panel uses. JSON body `{url, title, text, dry_run}`. If `text` is
  present, it's ingested directly and `url` becomes provenance only
  (faster + more accurate than backend-side fetching).

## Surface 1: bookmarklet (desktop browsers)

Drag this to your bookmarks bar (or create a new bookmark with the
`javascript:...` line as the URL). Default target is `localhost:6790`
— change `HOST` if Chloe runs on a different machine/port.

```javascript
javascript:(function(){var HOST='http://localhost:6790';var u=encodeURIComponent(location.href);var t=encodeURIComponent(document.title||'');var sel=window.getSelection&&window.getSelection().toString().slice(0,4000);var n=encodeURIComponent(sel||'');window.open(HOST+'/quick-capture.html?u='+u+'&t='+t+(n?'&n='+n:''),'chloeCapture','width=520,height=720');})();
```

Behavior:
- Captures the current tab URL + title.
- If you have text selected on the page, that selection is pre-filled
  in the Notes box → ingested as the body (no backend fetch needed).
- Pops a 520×720 window. `Cmd/Ctrl+Enter` to submit, `Esc` to cancel.
- Window auto-closes 1.5s after a successful ingest.

To install in Chrome/Edge:
1. Right-click bookmarks bar → Add page.
2. Name: "Chloe → brain". URL: paste the whole `javascript:...` line.
3. Click bookmark on any page to capture it.

## Surface 2: iOS Shortcut

Build this once in the Shortcuts app on your iPhone (Tailscale gives
you access to `<tailscale-ip>:6790` from anywhere). Replace
`<tailscale-ip>` with your actual Tailscale IP (visible in the
Tailscale app or in `tailscale status` on the Mac/PC).

Shortcut steps:
1. **Receive: Safari web pages** (Share Sheet input).
2. **Get URLs from input** → store as variable `URL`.
3. **Get name of URL** (with `URL`) → store as `Title`.
4. **Get clipboard** (optional — if you copied a selection before
   sharing, this captures it) → store as `Notes`.
5. **URL encode** each of `URL` / `Title` / `Notes`.
6. **Text** action — build:
   `http://<tailscale-ip>:6790/quick-capture.html?u=[URL]&t=[Title]&n=[Notes]`
7. **Open URLs** (the constructed link). Safari opens the capture
   page with everything pre-filled; tap Ingest.

Pin to Share Sheet. Now: any time you're reading something on iOS,
Share → "Drop to Chloe" → tap Ingest. Done.

## Surface 3: direct visit

Just open `http://localhost:6790/quick-capture.html` (or your
Tailscale IP) in any browser. Type or paste a URL, title, and/or
notes. Submit. No bookmarklet/shortcut needed for occasional use.

## When to use Notes vs URL-only

- **URL only (Notes empty):** backend fetches the page, strips
  HTML→text, ingests. Best for articles where you want the whole
  body in the brain. Slower (1-5s of HTTP fetch), occasionally
  blocked by paywalls/Cloudflare.
- **Notes filled:** backend skips fetching, ingests Notes directly
  with the URL recorded as provenance in the page frontmatter.
  Best when:
  - You highlighted just the important section.
  - The page is paywalled and you copied the body manually.
  - You're capturing a thought/quote that's not really a "page".
  - The page is dynamic JS-rendered (backend can't fetch usefully).

## Storage

Every capture lands in:

- `C:\Chloe\brain\raw\dropin_<slug>_<stamp>.md` (frontmatter +
  body), then
- Decomposed by Chloe's ingest pipeline into
  `C:\Chloe\brain\wiki\sources\<slug>.md` plus any new entity /
  concept pages it extracted, then
- Embedded by `wiki_watcher` within ~2s for recall + wiki search.

## Notes on host config

- Default port: 6790 (`CHLOE_GRAPH_PORT` env override).
- Default bind: `0.0.0.0` (`CHLOE_GRAPH_HOST` env override) — already
  set this way so Tailscale-on-phone reach works.
- The capture page calls `/api/brain/ingest` with a relative path,
  so whatever host serves the page also handles the ingest. No CORS
  fiddling needed.
