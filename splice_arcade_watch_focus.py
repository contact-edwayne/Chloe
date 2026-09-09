"""splice_arcade_watch_focus.py - fix keyboard focus so controls actually
reach the game after a cartridge loads (2026-09-09, "the controller seems
to be disconnected from the game I can't move").

Root cause: startEmulator() now (as of splice_arcade_watch.py) can be
invoked with no real user click inside this iframe's own document -- the
new auto-launch path calls it straight from a `loadLibrary().then()`
callback when arcade.html passes ?g=&sys=&name=. A user's manual library
click established this iframe's document focus as a side effect before;
auto-launch never does, so this frame (three levels deep: hud.html >
arcade.html's #emu-iframe > this page's own iframe > EmulatorJS canvas)
never receives keyboard focus and every keypress goes to a frame above it
instead of the emulator.

Same __bundler/manifest + __bundler/template splice pattern as
splice_arcade_watch.py. Backup is timestamped.

Usage:
    python splice_arcade_watch_focus.py            # patch in place
    python splice_arcade_watch_focus.py --restore  # restore most-recent backup
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

ANCHOR_INJECTED = "/* CHLOE_ARCADE_WATCH_FOCUS_INJECTED v1 */"
TEMPLATE_RE = re.compile(
    r'(<script type="__bundler/template">)(.*?)(</script>)', re.DOTALL)


def replace_once(html: str, old: str, new: str, label: str) -> str:
    n = html.count(old)
    if n != 1:
        raise SystemExit(f"FAIL {label}: count={n} (expected 1)")
    return html.replace(old, new)


def patch(html: str) -> str:
    if ANCHOR_INJECTED in html:
        raise SystemExit("emulator.html already contains the focus-fix anchor -- "
                          "run --restore first, then re-splice.")

    old = """    // Friendly: hide busy when EmulatorJS reports ready, if it exposes a hook.
    window.EJS_ready = function () { setBusy(false); };
    window.EJS_onGameStart = function () { setBusy(false); };"""
    new = ("    " + ANCHOR_INJECTED + "\n"
        "    // A cartridge can start with no real user click inside this\n"
        "    // iframe's own document (the auto-launch path below calls\n"
        "    // startEmulator() straight from a promise callback) -- without\n"
        "    // one, this frame never gets keyboard focus and every keypress\n"
        "    // goes to a frame above it instead of the game (2026-09-09:\n"
        "    // \"controller seems disconnected, can't move\"). Reclaim it\n"
        "    // once the core is actually ready to receive input.\n"
        "    function focusEmulator() {\n"
        "      try { window.focus(); } catch (_) {}\n"
        "      try {\n"
        "        var canvas = emuWrap.querySelector('canvas');\n"
        "        if (canvas) { canvas.tabIndex = -1; canvas.focus({ preventScroll: true }); }\n"
        "      } catch (_) {}\n"
        "    }\n"
        "    // Friendly: hide busy when EmulatorJS reports ready, if it exposes a hook.\n"
        "    window.EJS_ready = function () { setBusy(false); focusEmulator(); };\n"
        "    window.EJS_onGameStart = function () { setBusy(false); focusEmulator(); };")
    html = replace_once(html, old, new, "focus reclaim on EJS ready/start")

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
        baks = sorted(HERE.glob("emulator.html.bak.focus_*"), reverse=True)
        if not baks:
            print("no focus-fix backup found", file=sys.stderr)
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
    bak = TARGET.with_suffix(TARGET.suffix + f".bak.focus_{stamp}")
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
