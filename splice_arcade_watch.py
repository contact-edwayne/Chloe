"""splice_arcade_watch.py - add Chloe watch-mode wiring + query-param
auto-launch to the bundled emulator.html artifact (the EmulatorJS player
arcade.html actually opens for every system except ps2/gc/ps3).

Lesson #23 pattern (see splice_brain_dropin.py): parse __bundler/manifest +
__bundler/template, decode the template (JSON-encoded HTML string), patch
the decoded HTML with targeted replace_once edits, re-encode with
</script>-escaping, write back. Backup is timestamped.

Root cause this fixes (2026-09-09, "her watching feature on games is not
running"): emulator.html is a self-contained bundled app with its OWN rom
library UI. It never opens a WebSocket to jarvis.py at all, so unlike its
sibling panels (pcsx2_panel.html, dolphin_panel.html, rpcs3_panel.html,
gen1recomp_panel.html, emulator_lite.html, emulator_mobile.html -- all of
which wsSend game_watch_start/stop with the current game name) there has
never been any way to start watch-mode commentary for a game routed
through this player: PSX, Genesis, SNES, NES, GBA, non-recomp N64 --
exactly the systems the arcade ROM library actually plays day to day.
It also ignores the ?g=&sys=&name= query params arcade.html's
mountPlaying() passes when you click a library entry, so clicking a game
always dropped you back into this file's own picker instead of loading it.

Usage:
    python splice_arcade_watch.py            # patch in place
    python splice_arcade_watch.py --restore  # restore most-recent backup
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
TARGET = HERE / "emulator.html"

ANCHOR_INJECTED = "/* CHLOE_ARCADE_WATCH_INJECTED v1 */"
TEMPLATE_RE = re.compile(
    r'(<script type="__bundler/template">)(.*?)(</script>)', re.DOTALL)


def replace_once(html: str, old: str, new: str, label: str) -> str:
    n = html.count(old)
    if n != 1:
        raise SystemExit(f"FAIL {label}: count={n} (expected 1)")
    return html.replace(old, new)


def patch(html: str) -> str:
    if ANCHOR_INJECTED in html:
        raise SystemExit("emulator.html already contains the watch-mode anchor -- "
                          "run --restore first, then re-splice.")

    # 1. CSS -- a subtle violet "watching" state for the new button, next to
    #    the existing .btn.warn rule this file already uses for EJECT.
    old_css = """  .btn:disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }
"""
    new_css = """  .btn:disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }
  """ + ANCHOR_INJECTED + """
  .btn.watch-on {
    color: #fff;
    border-color: var(--violet);
    background: linear-gradient(180deg, rgba(176, 108, 255, 0.22), rgba(176, 108, 255, 0.05));
    box-shadow: 0 0 12px rgba(176, 108, 255, 0.4), inset 0 0 12px rgba(176, 108, 255, 0.15);
  }
  .btn.watch-on .glyph { color: var(--violet-1); }
"""
    html = replace_once(html, old_css, new_css, "css watch-on rule")

    # 2. Button markup next to EJECT.
    old_btns = """        <div class="vp-actions">
          <button class="btn warn" id="btnEject" disabled=""><span class="glyph">⏏</span>EJECT</button>
        </div>"""
    new_btns = """        <div class="vp-actions">
          <button class="btn" id="btnWatch" disabled="" title="Chloe watches your screen and comments live"><span class="glyph">◉</span>CHLOE: WATCH</button>
          <button class="btn warn" id="btnEject" disabled=""><span class="glyph">⏏</span>EJECT</button>
        </div>"""
    html = replace_once(html, old_btns, new_btns, "watch button markup")

    # 3. DOM ref.
    old_ref = "  const btnEject  = $('btnEject');\n"
    new_ref = ("  const btnEject  = $('btnEject');\n"
               "  const btnWatch  = $('btnWatch');\n")
    html = replace_once(html, old_ref, new_ref, "btnWatch DOM ref")

    # 4. State vars, next to the existing activeBlobUrl teardown-owned state.
    old_state = "  let activeBlobUrl = null; // we own this; revoke on teardown\n"
    new_state = (old_state +
        "  // Chloe backend WS (watch-mode commentary) -- same protocol as\n"
        "  // the native-emulator panels (pcsx2_panel.html etc), added here\n"
        "  // because this bundled player never had it (2026-09-09).\n"
        "  let ws = null, curGame = null, watching = false, didWatchReset = false;\n"
        "  function wsUrl() {\n"
        "    return location.protocol === 'https:'\n"
        "      ? `wss://${location.host}/chloe-ws`\n"
        "      : `ws://${location.hostname || 'localhost'}:6789`;\n"
        "  }\n"
        "  function wsSend(o) {\n"
        "    try { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); } catch (_) {}\n"
        "  }\n"
        "  function syncWatchBtn() {\n"
        "    if (!btnWatch) return;\n"
        "    btnWatch.disabled = !curGame;\n"
        "    btnWatch.textContent = watching ? 'CHLOE: WATCHING' : 'CHLOE: WATCH';\n"
        "    btnWatch.classList.toggle('watch-on', watching);\n"
        "  }\n"
        "  function wsConnect() {\n"
        "    try { ws = new WebSocket(wsUrl()); } catch (_) { return; }\n"
        "    ws.addEventListener('open', () => {\n"
        "      // A stale 'watching' from a previous page-life (or a reload\n"
        "      // mid-session) shouldn't silently resume -- force it off once\n"
        "      // per connection, same as the native panels do.\n"
        "      if (!didWatchReset) {\n"
        "        didWatchReset = true;\n"
        "        watching = false;\n"
        "        syncWatchBtn();\n"
        "        wsSend({ type: 'game_watch_stop' });\n"
        "      }\n"
        "    });\n"
        "    ws.addEventListener('close', () => setTimeout(wsConnect, 3000));\n"
        "  }\n")
    html = replace_once(html, old_state, new_state, "watch-mode WS state + helpers")

    # 5. startEmulator(): loading a new cartridge ends any watch session that
    #    belonged to the previous one, and makes the new title watchable.
    old_start = """  function startEmulator(opts) {
    // opts: { core, gameUrl, name, sysLabel }
    teardownEmulator();

    setBusy(true, 'spooling ' + opts.core + ' core');
    nowTitle.textContent = opts.name;"""
    new_start = """  function startEmulator(opts) {
    // opts: { core, gameUrl, name, sysLabel }
    teardownEmulator();
    if (watching) { wsSend({ type: 'game_watch_stop' }); watching = false; }
    curGame = opts.name;
    syncWatchBtn();

    setBusy(true, 'spooling ' + opts.core + ' core');
    nowTitle.textContent = opts.name;"""
    html = replace_once(html, old_start, new_start, "startEmulator watch reset")

    # 6. returnToStandby(): ejecting stops watching too (nothing left to watch).
    old_standby = """  function returnToStandby() {
    teardownEmulator();
    nowTitle.textContent = '— standby —';"""
    new_standby = """  function returnToStandby() {
    teardownEmulator();
    if (watching) { wsSend({ type: 'game_watch_stop' }); watching = false; }
    curGame = null;
    syncWatchBtn();
    nowTitle.textContent = '— standby —';"""
    html = replace_once(html, old_standby, new_standby, "returnToStandby watch reset")

    # 7. Wire the button.
    old_wire = """  btnEject.addEventListener('click', () => {
    setSays("ejected. pick another, beloved.");
    returnToStandby();
  });"""
    new_wire = """  btnEject.addEventListener('click', () => {
    setSays("ejected. pick another, beloved.");
    returnToStandby();
  });
  if (btnWatch) btnWatch.addEventListener('click', () => {
    if (!curGame) return;
    watching = !watching;
    wsSend({ type: watching ? 'game_watch_start' : 'game_watch_stop', game: curGame });
    syncWatchBtn();
  });"""
    html = replace_once(html, old_wire, new_wire, "watch button click wiring")

    # 8. Boot: connect the WS, stop watching on tab close, and -- separate but
    #    related bug -- actually honor the ?g=&sys=&name= query params
    #    arcade.html passes when you click a library entry, instead of always
    #    dropping back into this page's own picker.
    old_boot = """  // pretty session id
  sessionId.textContent = '0X' + Math.random().toString(16).slice(2, 6).toUpperCase() + '-' + Math.random().toString(16).slice(2, 6).toUpperCase();
  loadLibrary();

})();"""
    new_boot = """  // pretty session id
  sessionId.textContent = '0X' + Math.random().toString(16).slice(2, 6).toUpperCase() + '-' + Math.random().toString(16).slice(2, 6).toUpperCase();
  wsConnect();
  window.addEventListener('beforeunload', () => {
    try { wsSend({ type: 'game_watch_stop' }); } catch (_) {}
  });
  // arcade.html's mountPlaying() opens this iframe with ?g=<file>&sys=<key>
  // &name=<display name> for a library click -- honor it instead of always
  // landing on this page's own (separate) picker.
  loadLibrary().then(() => {
    const qp = new URLSearchParams(location.search);
    const qGame = qp.get('g'), qSys = qp.get('sys');
    const sysDef = qSys && SYS[qSys];
    if (qGame && sysDef) {
      const qName = qp.get('name') || qGame;
      state.activeRomKey = qGame;
      renderLibrary();
      setSays('loading ' + qName.toLowerCase() + '… have fun.');
      startEmulator({
        core: sysDef.ejsCore,
        gameUrl: '/roms/' + encodeURIComponent(qGame),
        name: qName,
        sysLabel: sysDef.label,
      });
    }
  });

})();"""
    html = replace_once(html, old_boot, new_boot, "boot: wsConnect + auto-launch")

    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true",
                     help="restore the most recent .bak file written by this script")
    args = ap.parse_args()

    if not TARGET.exists():
        print(f"missing {TARGET}", file=sys.stderr)
        return 2

    if args.restore:
        baks = sorted(HERE.glob("emulator.html.bak.watch_*"), reverse=True)
        if not baks:
            print("no watch-mode backup found", file=sys.stderr)
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
    new_json = json.dumps(patched_html, ensure_ascii=False)
    new_json = new_json.replace("</", "<\\/")  # lesson #23: never leave a literal </ in the embedded JSON.
    new_src = src[:m.start(2)] + new_json + src[m.end(2):]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET.with_suffix(TARGET.suffix + f".bak.watch_{stamp}")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(new_src, encoding="utf-8")

    delta = len(new_src) - len(src)
    print(f"OK. backup: {bak.name}")
    print(f"size: {len(src):,} -> {len(new_src):,} (+{delta:,} bytes)")
    print("Anchor:", ANCHOR_INJECTED)
    raw_template = new_src[m.start(2):m.end(2) + (len(new_src) - len(src))]
    bad = raw_template.count("</script")
    print(f"literal '</script' in embedded template: {bad} (must be 0)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
