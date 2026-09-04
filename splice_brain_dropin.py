"""splice_brain_dropin.py - inject the brain drop-in panel into the bundled
brain-graph.html artifact.

Lesson #23 pattern: parse __bundler/manifest + __bundler/template, decode the
template (JSON-encoded HTML string), inject CSS + HTML + JS before </body>,
re-encode with </script>-escaping, write back. Backup is timestamped so a
re-run on the same day doesn't clobber a clean baseline (lesson #13).

Usage:
    python splice_brain_dropin.py            # patch in place
    python splice_brain_dropin.py --restore  # restore most-recent backup
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
TARGET = HERE / "brain-graph.html"

ANCHOR_INJECTED = "/* CHLOE_DROPIN_INJECTED v2 */"
TEMPLATE_RE = re.compile(
    r'(<script type="__bundler/template">)(.*?)(</script>)', re.DOTALL)


# ───── injected payload (CSS + HTML + JS) ────────────────────────────────────
# Important: any literal "</script>" inside JS strings here MUST be split so it
# doesn't terminate the parent script tag when the template is decoded.

DROPIN_CSS = """
""" + ANCHOR_INJECTED + """
.cdrop-btn {
  position: fixed; bottom: 18px; left: 18px; z-index: 50;
  background: rgba(8,14,28,.92); border: 1px solid var(--rule-strong);
  color: var(--cyan); font: 11px/1 "Share Tech Mono", ui-monospace, monospace;
  letter-spacing: .12em; text-transform: uppercase;
  padding: 8px 12px; cursor: pointer; border-radius: 2px;
  box-shadow: 0 0 18px rgba(88,230,255,.18);
}
.cdrop-btn:hover { background: rgba(20,40,70,.95); color: var(--cyan-bright); }
.cdrop-btn .led {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: var(--cyan); box-shadow: 0 0 8px var(--cyan); margin-right: 8px;
  vertical-align: middle;
}
.cdrop-overlay {
  position: fixed; inset: 0; z-index: 100; display: none;
  background: rgba(2,4,12,.78); backdrop-filter: blur(6px);
  align-items: center; justify-content: center;
}
.cdrop-overlay.open { display: flex; }
.cdrop-modal {
  width: min(640px, 92vw); max-height: 86vh; overflow: auto;
  background: linear-gradient(180deg, rgba(8,14,28,.96), rgba(4,8,18,.96));
  border: 1px solid var(--rule-strong);
  box-shadow: 0 0 60px rgba(88,230,255,.18), 0 0 120px rgba(20,40,80,.4);
  padding: 18px 20px; color: var(--ink); border-radius: 4px;
  font-family: "Share Tech Mono", ui-monospace, monospace;
}
.cdrop-modal h2 {
  margin: 0 0 4px; font-size: 14px; letter-spacing: .2em;
  text-transform: uppercase; color: var(--cyan);
}
.cdrop-modal .sub {
  margin: 0 0 14px; font-size: 11px; color: var(--ink-dim);
  letter-spacing: .08em;
}
.cdrop-tabs {
  display: flex; gap: 0; margin-bottom: 12px;
  border-bottom: 1px solid var(--rule);
}
.cdrop-tab {
  background: transparent; border: 0; color: var(--ink-dim);
  font: 11px/1 "Share Tech Mono", ui-monospace, monospace;
  letter-spacing: .14em; text-transform: uppercase;
  padding: 8px 14px; cursor: pointer; border-bottom: 2px solid transparent;
}
.cdrop-tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.cdrop-tab:hover { color: var(--cyan-bright); }
.cdrop-panel { display: none; }
.cdrop-panel.active { display: block; }
.cdrop-drop {
  border: 1px dashed var(--rule-strong); border-radius: 3px;
  padding: 28px 18px; text-align: center; color: var(--ink-dim);
  font-size: 12px; cursor: pointer; transition: background .15s, color .15s;
}
.cdrop-drop.hover, .cdrop-drop:hover {
  background: rgba(88,230,255,.06); color: var(--cyan-bright);
}
.cdrop-drop b { color: var(--cyan); font-weight: 500; }
.cdrop-input, .cdrop-textarea {
  width: 100%; background: rgba(2,4,12,.6); border: 1px solid var(--rule);
  color: var(--ink); padding: 10px 12px;
  font: 12px/1.5 "Share Tech Mono", ui-monospace, monospace;
  outline: none; border-radius: 2px;
}
.cdrop-input:focus, .cdrop-textarea:focus { border-color: var(--cyan); }
.cdrop-textarea { resize: vertical; min-height: 140px; }
.cdrop-row { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
.cdrop-title-in { flex: 1 1 auto; }
.cdrop-submit {
  background: var(--cyan); color: #02040a; border: 0; padding: 10px 16px;
  font: 11px/1 "Share Tech Mono", ui-monospace, monospace;
  letter-spacing: .14em; text-transform: uppercase; cursor: pointer;
  border-radius: 2px;
}
.cdrop-submit:hover { background: var(--cyan-bright); }
.cdrop-submit[disabled] { opacity: .5; cursor: wait; }
.cdrop-secondary {
  background: transparent; color: var(--ink-dim);
  border: 1px solid var(--rule);
  padding: 9px 14px; font: 11px/1 "Share Tech Mono", ui-monospace, monospace;
  letter-spacing: .14em; text-transform: uppercase; cursor: pointer;
  border-radius: 2px;
}
.cdrop-secondary:hover { color: var(--ink); border-color: var(--rule-strong); }
.cdrop-status {
  margin-top: 12px; padding: 10px 12px; font-size: 11px;
  border: 1px solid var(--rule); border-radius: 2px;
  background: rgba(2,4,12,.5); color: var(--ink-dim); display: none;
  white-space: pre-wrap; max-height: 240px; overflow: auto;
}
.cdrop-status.show { display: block; }
.cdrop-status.ok { border-color: rgba(88,230,255,.5); color: var(--cyan); }
.cdrop-status.err { border-color: rgba(255,120,120,.5); color: #ff9a9a; }
.cdrop-status.working { border-color: rgba(255,179,90,.5); color: var(--warn); }
.cdrop-similar { margin-top: 10px; font-size: 11px; color: var(--ink-dim); }
.cdrop-similar b { color: var(--cyan); font-weight: 500; }
.cdrop-similar ul { margin: 6px 0 0; padding-left: 18px; }
.cdrop-similar li { margin: 2px 0; }
.cdrop-similar a { color: var(--cyan); text-decoration: none; cursor: pointer; }
.cdrop-similar a:hover { color: var(--cyan-bright); }
.cdrop-img-preview {
  margin-top: 10px; display: none; text-align: center;
}
.cdrop-img-preview.show { display: block; }
.cdrop-img-preview img {
  max-width: 100%; max-height: 240px;
  border: 1px solid var(--rule); border-radius: 2px;
}
.cdrop-help {
  margin-top: 14px; font-size: 10px; color: var(--ink-dim);
  letter-spacing: .04em;
}
.cdrop-close {
  position: absolute; top: 14px; right: 18px;
  background: transparent; border: 0; color: var(--ink-dim);
  font: 14px/1 "Share Tech Mono", ui-monospace, monospace;
  cursor: pointer; padding: 4px 8px;
}
.cdrop-close:hover { color: var(--cyan-bright); }
.cdrop-modal { position: relative; }

/* ── existing-UI fixes (v2) ───────────────────────────────────────── */
/* Hub-roster scrollbar: match cockpit palette (was white track + gray thumb). */
#hub-list {
  scrollbar-width: thin;
  scrollbar-color: var(--rule-strong) rgba(2, 4, 12, 0.55);
}
#hub-list::-webkit-scrollbar { width: 6px; }
#hub-list::-webkit-scrollbar-track {
  background: rgba(2, 4, 12, 0.55);
  border-left: 1px solid rgba(88, 230, 255, 0.06);
}
#hub-list::-webkit-scrollbar-thumb {
  background: var(--rule);
  border-radius: 0;
}
#hub-list::-webkit-scrollbar-thumb:hover { background: var(--rule-strong); }
#hub-list::-webkit-scrollbar-corner { background: transparent; }
/* Apply the same treatment globally for any other auto-scroll panels that
   inherited the browser default (filters list, ghost list overflow cases). */
.rightrail ul, .leftrail ul, .panel-body, .tweaks {
  scrollbar-width: thin;
  scrollbar-color: var(--rule-strong) rgba(2, 4, 12, 0.55);
}
.rightrail ul::-webkit-scrollbar,
.leftrail ul::-webkit-scrollbar,
.tweaks::-webkit-scrollbar { width: 6px; }
.rightrail ul::-webkit-scrollbar-track,
.leftrail ul::-webkit-scrollbar-track,
.tweaks::-webkit-scrollbar-track {
  background: rgba(2, 4, 12, 0.55);
}
.rightrail ul::-webkit-scrollbar-thumb,
.leftrail ul::-webkit-scrollbar-thumb,
.tweaks::-webkit-scrollbar-thumb {
  background: var(--rule);
}
.rightrail ul::-webkit-scrollbar-thumb:hover,
.leftrail ul::-webkit-scrollbar-thumb:hover,
.tweaks::-webkit-scrollbar-thumb:hover { background: var(--rule-strong); }

/* Diagnostic STABLE chip: was overflowing the leftrail box. Render it as a
   compact chip so the header fits on one line at any rail width. */
.diagnostic h4 {
  gap: 6px;
  flex-wrap: nowrap;
  min-width: 0;
}
.diagnostic h4 > :first-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 0 1 auto;
}
.diagnostic h4 b {
  flex: 0 0 auto;
  font-size: 9px;
  letter-spacing: 0.14em;
  padding: 2px 6px;
  background: rgba(88, 230, 255, 0.08);
  border: 1px solid var(--rule);
  border-radius: 2px;
  white-space: nowrap;
  color: var(--cyan-bright);
  text-shadow: 0 0 6px rgba(88, 230, 255, 0.35);
}
.diagnostic h4 b[data-state="warn"]   { color: var(--warn);  border-color: rgba(255,179,90,0.5);  background: rgba(255,179,90,0.08); text-shadow: 0 0 6px rgba(255,179,90,0.4); }
.diagnostic h4 b[data-state="error"]  { color: #ff9a9a;      border-color: rgba(255,120,120,0.5); background: rgba(255,120,120,0.08); text-shadow: 0 0 6px rgba(255,120,120,0.4); }
"""

DROPIN_HTML = r"""
<button class="cdrop-btn" id="cdrop-open" title="Drop a file, link, image, or text into Chloe's brain">
  <span class="led"></span>DROP IN
</button>
<div class="cdrop-overlay" id="cdrop-overlay">
  <div class="cdrop-modal" role="dialog" aria-labelledby="cdrop-h">
    <button class="cdrop-close" id="cdrop-close" title="Close (Esc)">&times;</button>
    <h2 id="cdrop-h">Drop into brain</h2>
    <p class="sub">Adds a node to Chloe&rsquo;s wiki. Files become text. Images get vision-described. URLs get fetched. Free text becomes a note.</p>
    <div class="cdrop-tabs" role="tablist">
      <button class="cdrop-tab active" data-tab="file">FILE</button>
      <button class="cdrop-tab" data-tab="url">URL</button>
      <button class="cdrop-tab" data-tab="text">TEXT</button>
      <button class="cdrop-tab" data-tab="image">IMAGE</button>
    </div>

    <div class="cdrop-panel active" data-panel="file">
      <div class="cdrop-drop" id="cdrop-file-drop">
        Drop a file here or <b>click to browse</b>.<br>
        <span style="font-size:10px;opacity:.7">.txt / .md / .html / .json &mdash; ingested as text</span>
      </div>
      <input type="file" id="cdrop-file-input" style="display:none">
      <div class="cdrop-row">
        <input type="text" class="cdrop-input cdrop-title-in" id="cdrop-file-title" placeholder="Title (optional)">
        <button class="cdrop-submit" id="cdrop-file-submit" disabled>Ingest</button>
      </div>
    </div>

    <div class="cdrop-panel" data-panel="url">
      <input type="url" class="cdrop-input" id="cdrop-url-input" placeholder="https://...">
      <div class="cdrop-row">
        <input type="text" class="cdrop-input cdrop-title-in" id="cdrop-url-title" placeholder="Title (optional &mdash; will use page <title>)">
        <button class="cdrop-submit" id="cdrop-url-submit">Fetch &amp; ingest</button>
      </div>
    </div>

    <div class="cdrop-panel" data-panel="text">
      <textarea class="cdrop-textarea" id="cdrop-text-input" placeholder="Paste text, notes, an article body, anything..."></textarea>
      <div class="cdrop-row">
        <input type="text" class="cdrop-input cdrop-title-in" id="cdrop-text-title" placeholder="Title (required for text)">
        <button class="cdrop-submit" id="cdrop-text-submit">Ingest</button>
      </div>
    </div>

    <div class="cdrop-panel" data-panel="image">
      <div class="cdrop-drop" id="cdrop-image-drop">
        Drop an image here or <b>click to browse</b>.<br>
        <span style="font-size:10px;opacity:.7">JPG / PNG / WebP / GIF &mdash; vision-described by llama-4-scout</span>
      </div>
      <input type="file" id="cdrop-image-input" accept="image/*" style="display:none">
      <div class="cdrop-img-preview" id="cdrop-image-preview"><img alt="preview"></div>
      <div class="cdrop-row">
        <input type="text" class="cdrop-input cdrop-title-in" id="cdrop-image-title" placeholder="Title (optional)">
        <button class="cdrop-submit" id="cdrop-image-submit" disabled>Describe &amp; ingest</button>
      </div>
    </div>

    <div class="cdrop-status" id="cdrop-status"></div>
    <div class="cdrop-similar" id="cdrop-similar"></div>
    <div class="cdrop-help">Esc to close &middot; New node lands in <code>wiki/sources/</code> within ~2s of ingest.</div>
  </div>
</div>
"""

# JS is wrapped in an IIFE. NOTE: any literal "</" sequence followed by "script"
# inside a string would terminate the parent template script. Split as
# "<" + "/script>" to be safe. We don't currently need this, but the
# convention is preserved for future edits.
DROPIN_JS = r"""
(function(){
  if (window.__cdropMounted) return;
  window.__cdropMounted = true;

  const $ = (id) => document.getElementById(id);
  const overlay = $('cdrop-overlay');
  const status = $('cdrop-status');
  const similar = $('cdrop-similar');

  function setStatus(kind, text) {
    if (!text) {
      status.className = 'cdrop-status';
      status.textContent = '';
      return;
    }
    status.className = 'cdrop-status show ' + kind;
    status.textContent = text;
  }
  function setSimilar(items) {
    if (!items || !items.length) { similar.innerHTML = ''; return; }
    const lis = items.map(s =>
      '<li><a data-node="' + (s.id||'') + '">' + (s.title || s.id || '?') +
      '</a> <span style="opacity:.6">&middot; ' +
      (Number(s.score||0).toFixed(2)) + '</span></li>').join('');
    similar.innerHTML = '<b>Similar pages already in the brain:</b><ul>' + lis + '</ul>';
    similar.querySelectorAll('a[data-node]').forEach(a => {
      a.addEventListener('click', (e) => {
        const id = a.getAttribute('data-node');
        if (window.openNode) { window.openNode(id); }
        else if (window.postMessage) {
          window.parent.postMessage({type:'brain-focus-node', node_id: id}, '*');
        }
        closeModal();
      });
    });
  }

  function openModal() { overlay.classList.add('open'); setStatus('', ''); setSimilar([]); }
  function closeModal() { overlay.classList.remove('open'); }

  $('cdrop-open').addEventListener('click', openModal);
  $('cdrop-close').addEventListener('click', closeModal);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('open')) {
      e.stopPropagation(); closeModal();
    }
  }, true);

  // Tabs
  document.querySelectorAll('.cdrop-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.cdrop-tab').forEach(b => b.classList.toggle('active', b===btn));
      document.querySelectorAll('.cdrop-panel').forEach(p => p.classList.toggle('active', p.dataset.panel===tab));
      setStatus('', ''); setSimilar([]);
    });
  });

  // ─── File tab ─────────────────────────────────────────────────────────
  let fileBlob = null;
  const fileDrop = $('cdrop-file-drop');
  const fileInput = $('cdrop-file-input');
  fileDrop.addEventListener('click', () => fileInput.click());
  fileDrop.addEventListener('dragover', (e) => { e.preventDefault(); fileDrop.classList.add('hover'); });
  fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('hover'));
  fileDrop.addEventListener('drop', (e) => {
    e.preventDefault(); fileDrop.classList.remove('hover');
    if (e.dataTransfer.files[0]) { fileBlob = e.dataTransfer.files[0]; renderFilePick(); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) { fileBlob = fileInput.files[0]; renderFilePick(); }
  });
  function renderFilePick() {
    fileDrop.innerHTML = '<b>' + fileBlob.name + '</b><br>' +
      '<span style="font-size:10px;opacity:.7">' +
      Math.round(fileBlob.size/1024) + ' KB &middot; click to replace</span>';
    $('cdrop-file-submit').disabled = false;
  }
  $('cdrop-file-submit').addEventListener('click', async () => {
    if (!fileBlob) return;
    const title = $('cdrop-file-title').value.trim();
    const fd = new FormData();
    fd.append('file', fileBlob, fileBlob.name);
    if (title) fd.append('title', title);
    await submit(fd, null);
  });

  // ─── URL tab ──────────────────────────────────────────────────────────
  $('cdrop-url-submit').addEventListener('click', async () => {
    const url = $('cdrop-url-input').value.trim();
    if (!url) { setStatus('err', 'enter a URL.'); return; }
    const title = $('cdrop-url-title').value.trim();
    await submit(null, { url: url, title: title || undefined });
  });

  // ─── Text tab ─────────────────────────────────────────────────────────
  $('cdrop-text-submit').addEventListener('click', async () => {
    const text = $('cdrop-text-input').value.trim();
    const title = $('cdrop-text-title').value.trim();
    if (!text) { setStatus('err', 'enter some text.'); return; }
    if (!title) { setStatus('err', 'title is required for free text.'); return; }
    await submit(null, { text: text, title: title });
  });

  // ─── Image tab ────────────────────────────────────────────────────────
  let imageBlob = null;
  const imageDrop = $('cdrop-image-drop');
  const imageInput = $('cdrop-image-input');
  const imagePreview = $('cdrop-image-preview');
  imageDrop.addEventListener('click', () => imageInput.click());
  imageDrop.addEventListener('dragover', (e) => { e.preventDefault(); imageDrop.classList.add('hover'); });
  imageDrop.addEventListener('dragleave', () => imageDrop.classList.remove('hover'));
  imageDrop.addEventListener('drop', (e) => {
    e.preventDefault(); imageDrop.classList.remove('hover');
    if (e.dataTransfer.files[0]) { imageBlob = e.dataTransfer.files[0]; renderImagePick(); }
  });
  imageInput.addEventListener('change', () => {
    if (imageInput.files[0]) { imageBlob = imageInput.files[0]; renderImagePick(); }
  });
  function renderImagePick() {
    imageDrop.innerHTML = '<b>' + imageBlob.name + '</b><br>' +
      '<span style="font-size:10px;opacity:.7">' +
      Math.round(imageBlob.size/1024) + ' KB &middot; click to replace</span>';
    const url = URL.createObjectURL(imageBlob);
    imagePreview.querySelector('img').src = url;
    imagePreview.classList.add('show');
    $('cdrop-image-submit').disabled = false;
  }
  $('cdrop-image-submit').addEventListener('click', async () => {
    if (!imageBlob) return;
    const title = $('cdrop-image-title').value.trim();
    const fd = new FormData();
    fd.append('file', imageBlob, imageBlob.name);
    if (title) fd.append('title', title);
    await submit(fd, null, 'vision-describing image (10-30s)...');
  });

  // ─── Submit ───────────────────────────────────────────────────────────
  async function submit(formData, jsonBody, workingMsg) {
    setStatus('working', workingMsg || 'ingesting...');
    setSimilar([]);
    // Disable all submit buttons.
    const subs = document.querySelectorAll('.cdrop-submit');
    subs.forEach(b => b.disabled = true);
    try {
      const opts = { method: 'POST' };
      if (formData) {
        opts.body = formData;
      } else {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(jsonBody);
      }
      // Same-origin: brain_http serves the page AND the API.
      const r = await fetch('/api/brain/ingest', opts);
      let data;
      try { data = await r.json(); } catch (_) { data = { error: 'non-json response (status ' + r.status + ')' }; }
      if (!r.ok) {
        setStatus('err', 'Ingest failed (' + r.status + '): ' + (data.error || JSON.stringify(data)));
        return;
      }
      const slug = data.slug || data.title || '(unknown)';
      const ents = (data.entities_touched || []).length;
      const cons = (data.concepts_touched || []).length;
      const msg = [
        '✓ ingested as: ' + slug,
        'entities touched: ' + ents,
        'concepts touched: ' + cons,
        data.tldr ? ('\nTL;DR: ' + data.tldr) : '',
        data.image ? ('\nimage: ' + (data.image.filename||'?')) : '',
        '\nraw: ' + (data.raw_path||'?'),
      ].filter(Boolean).join('\n');
      setStatus('ok', msg);
      setSimilar(data.similar || []);
      // Clear file/image picks so user can drop another.
      fileBlob = null; imageBlob = null;
      $('cdrop-file-submit').disabled = true;
      $('cdrop-image-submit').disabled = true;
      fileDrop.innerHTML = 'Drop a file here or <b>click to browse</b>.<br><span style="font-size:10px;opacity:.7">.txt / .md / .html / .json &mdash; ingested as text</span>';
      imageDrop.innerHTML = 'Drop an image here or <b>click to browse</b>.<br><span style="font-size:10px;opacity:.7">JPG / PNG / WebP / GIF &mdash; vision-described by llama-4-scout</span>';
      imagePreview.classList.remove('show');
    } catch (e) {
      setStatus('err', 'network error: ' + (e && e.message ? e.message : e));
    } finally {
      subs.forEach(b => {
        // Re-enable URL/text submit; file/image need a new pick to re-enable.
        if (b.id === 'cdrop-url-submit' || b.id === 'cdrop-text-submit') b.disabled = false;
      });
    }
  }
})();
"""


def patch(html: str) -> str:
    """Apply the injection. Returns patched HTML string (already encoded for
    template re-embed)."""
    inject = (
        "<style>"
        + DROPIN_CSS
        + "</style>\n"
        + DROPIN_HTML
        + "<script>"
        + DROPIN_JS
        + "</script>\n"
    )
    if ANCHOR_INJECTED in html:
        # Already patched — re-inject by removing the old block first.
        # Find the previous injection by anchor + last </script> before </body>.
        # Simpler: just refuse and let --restore handle it.
        raise SystemExit("brain-graph.html already contains the drop-in anchor — "
                         "run --restore first, then re-splice.")
    body_close = html.rfind("</body>")
    if body_close < 0:
        raise SystemExit("no </body> found in template HTML")
    return html[:body_close] + inject + html[body_close:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true",
                    help="restore the most recent .bak file written by this script")
    args = ap.parse_args()

    if not TARGET.exists():
        print(f"missing {TARGET}", file=sys.stderr)
        return 2

    if args.restore:
        baks = sorted(HERE.glob("brain-graph.html.bak.dropin_*"), reverse=True)
        if not baks:
            print("no dropin backup found", file=sys.stderr)
            return 1
        latest = baks[0]
        shutil.copy2(latest, TARGET)
        print(f"restored from {latest.name}")
        return 0

    src = TARGET.read_text(encoding="utf-8")
    m = TEMPLATE_RE.search(src)
    if not m:
        print("__bundler/template script not found", file=sys.stderr)
        return 1
    template_json = m.group(2)
    try:
        html = json.loads(template_json)
    except Exception as e:
        print(f"template JSON decode failed: {e}", file=sys.stderr)
        return 1

    patched_html = patch(html)
    # Re-encode. Then escape any literal </script that would terminate the
    # template's parent <script> early (lesson #23 gotcha).
    new_json = json.dumps(patched_html, ensure_ascii=False)
    new_json = new_json.replace("</", "<\\/")  # be paranoid: every "</" gets escaped.
    new_src = src[:m.start(2)] + new_json + src[m.end(2):]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET.with_suffix(TARGET.suffix + f".bak.dropin_{stamp}")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(new_src, encoding="utf-8")

    delta = len(new_src) - len(src)
    print(f"OK. backup: {bak.name}")
    print(f"size: {len(src):,} -> {len(new_src):,} (+{delta:,} bytes)")
    print("Anchor:", ANCHOR_INJECTED)
    # Sanity: confirm no literal </script> in the embedded JSON.
    raw_template = new_src[m.start(2):m.end(2) + (len(new_src) - len(src))]
    bad = raw_template.count("</script")
    print(f"literal '</script' in embedded template: {bad} (must be 0)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
