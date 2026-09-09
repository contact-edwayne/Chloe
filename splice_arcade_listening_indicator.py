"""splice_arcade_listening_indicator.py - give the game view a real listening
symbol (2026-09-09, "add a listening symbol so I know that I can talk to
her").

Root cause: emulator.html's WS connection only ever SENT messages
(game_watch_start/stop) -- it never registered a 'message' listener at all,
so the backend's plain-string voice-state broadcasts ("idle"/"listening"/
"thinking"/"speaking", same ones hud.html's ring already reacts to) were
silently dropped. There was no way to tell, while looking at the game, that
the wake-word mic had actually opened. Bonus: game_comment broadcasts
(Chloe's live watch commentary) were audio-only in this view too -- wire
them into the existing (currently decorative) saysText line so her words
show as well as sound.

Same __bundler/manifest + __bundler/template splice pattern as
splice_arcade_watch.py / splice_arcade_watch_focus.py. Backup is
timestamped.

Usage:
    python splice_arcade_listening_indicator.py            # patch in place
    python splice_arcade_listening_indicator.py --restore  # restore latest backup
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

ANCHOR_INJECTED = "/* CHLOE_ARCADE_LISTENING_INDICATOR_INJECTED v1 */"
TEMPLATE_RE = re.compile(
    r'(<script type="__bundler/template">)(.*?)(</script>)', re.DOTALL)


def replace_once(html: str, old: str, new: str, label: str) -> str:
    n = html.count(old)
    if n != 1:
        raise SystemExit(f"FAIL {label}: count={n} (expected 1)")
    return html.replace(old, new)


def patch(html: str) -> str:
    if ANCHOR_INJECTED in html:
        raise SystemExit("emulator.html already contains the listening-indicator "
                          "anchor -- run --restore first, then re-splice.")

    # 1. CSS: new orb glow states for listening/speaking, distinct from the
    #    existing .thinking (which is reserved for ROM-load busy state and
    #    also toggles the vpLoading spinner -- reusing it here would pop
    #    that spinner every time the mic opens).
    old_css = """  .orb-wrap.thinking .orb-svg {
    animation: orb-think 1.4s ease-in-out infinite alternate;
  }
  @keyframes orb-think {
    from { filter: drop-shadow(0 0 6px rgba(127, 207, 255, 0.55)); }
    to   { filter: hue-rotate(70deg) drop-shadow(0 0 14px rgba(255, 110, 199, 0.65)); }
  }"""
    new_css = (old_css + "\n" + ANCHOR_INJECTED + "\n" +
        """  .orb-wrap.voice-listening .orb-svg {
    animation: orb-listen 0.9s ease-in-out infinite;
  }
  @keyframes orb-listen {
    0%, 100% { filter: drop-shadow(0 0 6px rgba(127, 207, 255, 0.55)); }
    50%      { filter: drop-shadow(0 0 16px rgba(70, 255, 160, 0.9)); }
  }
  .orb-wrap.voice-speaking .orb-svg {
    animation: orb-speak 0.6s ease-in-out infinite alternate;
  }
  @keyframes orb-speak {
    from { filter: drop-shadow(0 0 8px rgba(127, 207, 255, 0.6)); }
    to   { filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.9)); }
  }""")
    html = replace_once(html, old_css, new_css, "orb voice-state CSS")

    # 2. CSS: says-text turns the same green while listening, so the line
    #    Ed already glances at doubles as a text confirmation.
    old_says_css = """  .says-text {
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 19px;
    color: var(--violet-1);
    line-height: 1.4;
    text-shadow: 0 0 10px rgba(176, 108, 255, 0.25);
    min-height: 1.4em;
    flex: 1;
  }"""
    new_says_css = (old_says_css + "\n" +
        """  .says-text.voice-listening {
    color: #46ffa0;
    text-shadow: 0 0 10px rgba(70, 255, 160, 0.45);
  }""")
    html = replace_once(html, old_says_css, new_says_css, "says-text listening color")

    # 3. JS: setSays now remembers the last real line so listening/idle can
    #    restore it; setVoiceState drives the orb + says-text off backend
    #    voice-state broadcasts.
    old_setsays = """  function setSays(text) {
    saysText.textContent = (text || '').toLowerCase();
  }"""
    new_setsays = """  let _lastRealSays = saysText.textContent;
  function setSays(text) {
    saysText.textContent = (text || '').toLowerCase();
    _lastRealSays = saysText.textContent;
  }
  function setVoiceState(s) {
    orbWrap.classList.remove('voice-listening', 'voice-speaking');
    if (s === 'listening') {
      orbWrap.classList.add('voice-listening');
      saysText.classList.add('voice-listening');
      saysText.textContent = 'listening\\u2026';
    } else if (s === 'speaking') {
      orbWrap.classList.add('voice-speaking');
      saysText.classList.remove('voice-listening');
    } else {
      saysText.classList.remove('voice-listening');
      if (saysText.textContent === 'listening\\u2026') saysText.textContent = _lastRealSays;
    }
  }"""
    html = replace_once(html, old_setsays, new_setsays, "setSays + setVoiceState")

    # 4. JS: actually listen on the WS connection. Previously wsConnect()
    #    only sent (game_watch_start/stop) and reacted to open/close --
    #    'message' was never handled at all.
    old_wsconnect = """  function wsConnect() {
    try { ws = new WebSocket(wsUrl()); } catch (_) { return; }
    ws.addEventListener('open', () => {
      // A stale 'watching' from a previous page-life (or a reload
      // mid-session) shouldn't silently resume -- force it off once
      // per connection, same as the native panels do.
      if (!didWatchReset) {
        didWatchReset = true;
        watching = false;
        syncWatchBtn();
        wsSend({ type: 'game_watch_stop' });
      }
    });
    ws.addEventListener('close', () => setTimeout(wsConnect, 3000));
  }"""
    new_wsconnect = """  function wsConnect() {
    try { ws = new WebSocket(wsUrl()); } catch (_) { return; }
    ws.addEventListener('open', () => {
      // A stale 'watching' from a previous page-life (or a reload
      // mid-session) shouldn't silently resume -- force it off once
      // per connection, same as the native panels do.
      if (!didWatchReset) {
        didWatchReset = true;
        watching = false;
        syncWatchBtn();
        wsSend({ type: 'game_watch_stop' });
      }
    });
    // 2026-09-09: "add a listening symbol so I know that I can talk to her"
    // -- this connection only ever sent; it never reacted to anything the
    // backend broadcast. Mirror the same plain-string voice-state
    // broadcasts hud.html's ring already reacts to, and show her live
    // watch commentary text as it arrives (previously audio-only here).
    ws.addEventListener('message', (ev) => {
      let data = null;
      try { data = JSON.parse(ev.data); } catch (_) {}
      if (data && typeof data === 'object') {
        if (data.type === 'game_comment' && typeof data.text === 'string') {
          setSays(data.text);
        }
        return;
      }
      if (typeof ev.data === 'string') {
        const s = ev.data.trim().toLowerCase();
        if (s === 'idle' || s === 'listening' || s === 'thinking' || s === 'speaking') {
          setVoiceState(s);
        }
      }
    });
    ws.addEventListener('close', () => setTimeout(wsConnect, 3000));
  }"""
    html = replace_once(html, old_wsconnect, new_wsconnect, "wsConnect message listener")

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
        baks = sorted(HERE.glob("emulator.html.bak.listen_*"), reverse=True)
        if not baks:
            print("no listening-indicator backup found", file=sys.stderr)
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
    new_json = new_json.replace("</", "<\\/")
    new_src = src[:m.start(2)] + new_json + src[m.end(2):]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET.with_suffix(TARGET.suffix + f".bak.listen_{stamp}")
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
