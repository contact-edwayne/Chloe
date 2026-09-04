"""splice_hud_ch03_jobs.py — add a CH03 JOBS channel to hud.html and the
matching WS handlers to jarvis.py. Idempotent via anchor checks.

Lesson #6: hud.html and jarvis.py are past the Edit-tool comfort
threshold. Splice via Python with ast.parse for jarvis.py and a
literal-string anchor check + tail-diff for hud.html.

Usage:
    python splice_hud_ch03_jobs.py            # apply
    python splice_hud_ch03_jobs.py --restore  # roll back to last backup
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
HUD = HERE / "hud.html"
JARVIS = HERE / "jarvis.py"

HUD_ANCHOR = "/* ─── CH03 JOBS PANEL — added 2026-05-19 ───────────────────────────── */"
JARVIS_ANCHOR = "# ─── CH03 JOBS handlers — added 2026-05-19 ─────────────────────────────"


# ═══════════════════════════════════════════════════════════════════════════
# hud.html splice
# ═══════════════════════════════════════════════════════════════════════════

HUD_CSS = """

""" + HUD_ANCHOR + """
.jobs-panel {
  display: none; flex: 1; overflow-y: auto;
  padding: 16px 20px; flex-direction: column; gap: 12px;
}
.jobs-panel.active { display: flex; }
.jobs-panel::-webkit-scrollbar { width: 6px; }
.jobs-panel::-webkit-scrollbar-thumb { background: #00f7ff44; border-radius: 3px; }
.jobs-panel::-webkit-scrollbar-track { background: transparent; }
.jobs-summary {
  display: flex; gap: 12px; flex-wrap: wrap;
  padding: 10px 14px; border: 1px solid var(--border);
  background: rgba(0, 13, 26, 0.6);
  font-family: 'Share Tech Mono', monospace; font-size: 11px;
  letter-spacing: 2px; text-transform: uppercase;
}
.jobs-summary .pill { color: #00f7ff66; }
.jobs-summary .pill b { color: var(--cyan); font-weight: 500; }
.jobs-summary .pill.ok b { color: #00ff88; }
.jobs-summary .pill.fail b { color: #ff5b5b; }
.jobs-summary .pill.running b { color: #ffb35a; }
.jobs-list { display: flex; flex-direction: column; gap: 6px; }
.job-card {
  border: 1px solid var(--border);
  background: rgba(0, 13, 26, 0.6);
  padding: 10px 12px;
  display: grid;
  grid-template-columns: 8px 1fr auto auto;
  gap: 10px; align-items: center;
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px; color: var(--cyan);
}
.job-card .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #555; box-shadow: 0 0 4px #555;
}
.job-card.healthy   .dot { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
.job-card.stale     .dot { background: #ffb35a; box-shadow: 0 0 6px #ffb35a; }
.job-card.fail      .dot { background: #ff5b5b; box-shadow: 0 0 6px #ff5b5b; }
.job-card.running   .dot { background: #58e6ff; box-shadow: 0 0 8px #58e6ff;
                            animation: chloe-pulse 1.2s ease-in-out infinite; }
.job-card.never_run .dot { background: #6b8c9c; box-shadow: none; }
.job-card .name {
  letter-spacing: 1.5px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.job-card .schedule {
  font-size: 9px; color: #00f7ff66;
  letter-spacing: 2px; text-transform: uppercase;
}
.job-card .age {
  font-size: 9px; color: #00f7ff66;
  letter-spacing: 1px; font-variant-numeric: tabular-nums;
  min-width: 70px; text-align: right;
}
.job-card .run-btn {
  background: transparent; border: 1px solid var(--border2);
  color: var(--cyan); font: 9px/1 'Share Tech Mono', monospace;
  letter-spacing: 2px; text-transform: uppercase;
  padding: 5px 9px; cursor: pointer; border-radius: 2px;
}
.job-card .run-btn:hover { background: rgba(0, 247, 255, 0.1); }
.job-card .run-btn[disabled] { opacity: .4; cursor: wait; }
.job-card .result {
  grid-column: 2 / -1;
  font-size: 10px; color: #00f7ff88;
  letter-spacing: 0; white-space: pre-wrap;
  margin-top: 2px;
  max-height: 60px; overflow: hidden;
}
.jobs-refresh-row {
  display: flex; justify-content: space-between; align-items: center;
  margin-top: 4px;
}
.jobs-refresh-btn {
  background: transparent; border: 1px dashed var(--border2);
  color: #00f7ff77; font: 10px/1 'Share Tech Mono', monospace;
  letter-spacing: 2px; text-transform: uppercase;
  padding: 6px 10px; cursor: pointer;
}
.jobs-refresh-btn:hover { color: var(--cyan); border-color: var(--cyan); }
"""

HUD_PANEL_HTML = """
      <div class="jobs-panel" id="jobs-panel">
        <div class="jobs-summary" id="jobs-summary">
          <span class="pill">loading...</span>
        </div>
        <div class="jobs-list" id="jobs-list">
          <span style="color:#00f7ff44;font-size:10px;padding:4px;">awaiting state...</span>
        </div>
        <div class="jobs-refresh-row">
          <button class="jobs-refresh-btn" onclick="requestJobsState()">↻ refresh</button>
          <span style="color:#00f7ff44;font-size:9px;letter-spacing:1px;" id="jobs-asof">—</span>
        </div>
      </div>
"""

# JS replacement: extend toggleChannel to 3-way and add jobs handlers.
HUD_JS_TOGGLE_OLD = """  let _currentChannel = 'CH01';
  let _lightsState = { bulbs: [], presets: [] };
  let _wheelTargetBulb = null;   // name of the bulb the wheel is editing
  let _wheelColor = '#ffffff';

  function toggleChannel() {
    _currentChannel = (_currentChannel === 'CH01') ? 'CH02' : 'CH01';
    const label = document.getElementById('ch-label');
    const chatLog = document.getElementById('chatlog');
    const lightsPanel = document.getElementById('lights-panel');
    const inputRow = document.querySelector('.chat-input-row');
    const attachments = document.getElementById('attachments');
    if (_currentChannel === 'CH02') {
      if (label) label.textContent = 'CH 02';
      if (chatLog) chatLog.style.display = 'none';
      if (lightsPanel) lightsPanel.classList.add('active');
      if (inputRow) inputRow.style.display = 'none';
      if (attachments) attachments.style.display = 'none';
      requestLightsState();
    } else {
      if (label) label.textContent = 'CH 01';
      if (chatLog) chatLog.style.display = '';
      if (lightsPanel) lightsPanel.classList.remove('active');
      if (inputRow) inputRow.style.display = '';
      if (attachments) attachments.style.display = '';
    }
  }"""

HUD_JS_TOGGLE_NEW = """  let _currentChannel = 'CH01';
  let _lightsState = { bulbs: [], presets: [] };
  let _jobsState = { jobs: [], summary: {} };
  let _jobsPollTimer = null;
  let _wheelTargetBulb = null;   // name of the bulb the wheel is editing
  let _wheelColor = '#ffffff';

  function toggleChannel() {
    // 3-way cycle: CH01 (chat) -> CH02 (lights) -> CH03 (jobs) -> CH01
    if (_currentChannel === 'CH01')      _currentChannel = 'CH02';
    else if (_currentChannel === 'CH02') _currentChannel = 'CH03';
    else                                  _currentChannel = 'CH01';
    _applyChannelView();
  }

  function _applyChannelView() {
    const label = document.getElementById('ch-label');
    const chatLog = document.getElementById('chatlog');
    const lightsPanel = document.getElementById('lights-panel');
    const jobsPanel = document.getElementById('jobs-panel');
    const inputRow = document.querySelector('.chat-input-row');
    const attachments = document.getElementById('attachments');

    // Default: hide all secondary panels + restore chat.
    if (lightsPanel) lightsPanel.classList.remove('active');
    if (jobsPanel)   jobsPanel.classList.remove('active');
    if (chatLog)     chatLog.style.display = '';
    if (inputRow)    inputRow.style.display = '';
    if (attachments) attachments.style.display = '';
    if (_jobsPollTimer) { clearInterval(_jobsPollTimer); _jobsPollTimer = null; }

    if (_currentChannel === 'CH02') {
      if (label) label.textContent = 'CH 02';
      if (chatLog) chatLog.style.display = 'none';
      if (lightsPanel) lightsPanel.classList.add('active');
      if (inputRow) inputRow.style.display = 'none';
      if (attachments) attachments.style.display = 'none';
      requestLightsState();
    } else if (_currentChannel === 'CH03') {
      if (label) label.textContent = 'CH 03';
      if (chatLog) chatLog.style.display = 'none';
      if (jobsPanel) jobsPanel.classList.add('active');
      if (inputRow) inputRow.style.display = 'none';
      if (attachments) attachments.style.display = 'none';
      requestJobsState();
      // Poll every 5s while CH03 is open so running jobs flip to done.
      _jobsPollTimer = setInterval(requestJobsState, 5000);
    } else {
      if (label) label.textContent = 'CH 01';
    }
  }

  // ─── CH03 JOBS ─────────────────────────────────────────────────────────
  function requestJobsState() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'jobs_state' }));
    }
  }

  function runJob(name, btn) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (btn) { btn.disabled = true; btn.textContent = '...'; }
    ws.send(JSON.stringify({ type: 'jobs_run', job: name }));
    // Refresh state shortly after the kick to reflect 'running'.
    setTimeout(requestJobsState, 800);
  }

  function onJobsState(data) {
    _jobsState = { jobs: data.jobs || [], summary: data.summary || {} };
    renderJobs();
    const asof = document.getElementById('jobs-asof');
    if (asof && data.computed_at) {
      const d = new Date(data.computed_at * 1000);
      asof.textContent = 'as of ' + d.toLocaleTimeString('en-US', { hour12: false });
    }
  }

  function _fmtAge(hours) {
    if (hours == null) return '—';
    if (hours < 1)  return Math.round(hours * 60) + 'm';
    if (hours < 48) return hours.toFixed(1) + 'h';
    return Math.round(hours / 24) + 'd';
  }

  function renderJobs() {
    const sum = _jobsState.summary || {};
    const summaryEl = document.getElementById('jobs-summary');
    if (summaryEl) {
      summaryEl.innerHTML =
        '<span class="pill">total <b>' + (sum.total || 0) + '</b></span>' +
        '<span class="pill">today <b>' + (sum.ran_today || 0) + '</b></span>' +
        '<span class="pill ok">ok <b>' + (sum.ok || 0) + '</b></span>' +
        (sum.fail ? '<span class="pill fail">fail <b>' + sum.fail + '</b></span>' : '') +
        (sum.running ? '<span class="pill running">running <b>' + sum.running + '</b></span>' : '');
    }
    const list = document.getElementById('jobs-list');
    if (!list) return;
    if (!_jobsState.jobs.length) {
      list.innerHTML = '<span style="color:#00f7ff44;font-size:10px;padding:4px;">no jobs registered</span>';
      return;
    }
    list.innerHTML = _jobsState.jobs.map(j => {
      const cls = j.health || 'never_run';
      const age = j.health === 'running' ? 'running…' : _fmtAge(j.age_hours);
      const result = j.last_result
        ? '<div class="result">' + escapeHtml(j.last_result) + '</div>'
        : '';
      const btnLabel = j.running ? '...' : '▶ run';
      const btnDisabled = j.running ? 'disabled' : '';
      return '<div class="job-card ' + cls + '">' +
        '<span class="dot"></span>' +
        '<span class="name" title="' + escapeHtml(j.name) + '">' +
          escapeHtml(j.name) + '</span>' +
        '<span class="schedule">' + escapeHtml(j.schedule || '') + '</span>' +
        '<span class="age">' + age + '</span>' +
        '<button class="run-btn" ' + btnDisabled +
          ' onclick="runJob(\\'' + j.name + '\\', this)">' + btnLabel + '</button>' +
        result +
      '</div>';
    }).join('');
  }

  function escapeHtml(s) {
    return (s == null ? '' : String(s))
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }"""

# WS-handler dispatch in the message-router: add jobs_state + jobs_run cases.
HUD_WS_HANDLER_OLD = "    } else if (data.type === 'lights_state_result') {"
HUD_WS_HANDLER_NEW = (
    "    } else if (data.type === 'jobs_state_update') {\n"
    "      onJobsState(data);\n"
    "    } else if (data.type === 'jobs_run_result') {\n"
    "      // Job kicked off — state refresh will show 'running'.\n"
    "      if (data.error) console.warn('[jobs]', data.error);\n"
    "      setTimeout(requestJobsState, 800);\n"
    "    } else if (data.type === 'lights_state_result') {"
)


# ═══════════════════════════════════════════════════════════════════════════
# jarvis.py splice
# ═══════════════════════════════════════════════════════════════════════════

JARVIS_DISPATCH_OLD = (
    '    elif t == "lights_state":           await handle_lights_state(data, websocket)'
)
JARVIS_DISPATCH_NEW = (
    '    elif t == "jobs_state":             await handle_jobs_state(data, websocket)\n'
    '    elif t == "jobs_run":               await handle_jobs_run(data, websocket)\n'
    '    elif t == "lights_state":           await handle_lights_state(data, websocket)'
)

JARVIS_HANDLERS = '''

''' + JARVIS_ANCHOR + '''
# Wraps chloe_jobs.state() and chloe_jobs.run_async() for the HUD CH03 channel.
async def handle_jobs_state(data, websocket):
    """Send a fresh jobs snapshot to the requesting client."""
    try:
        import chloe_jobs
        snap = chloe_jobs.state()
    except Exception as e:
        await _ws_send(websocket, {"type": "jobs_state_update",
                                   "error": f"{type(e).__name__}: {e}",
                                   "jobs": [], "summary": {}})
        return
    await _ws_send(websocket, {"type": "jobs_state_update", **snap})


async def handle_jobs_run(data, websocket):
    """Trigger a single job. Non-blocking — returns immediately. The job
    runs in a background thread; the next jobs_state poll will show its
    state flip from 'running' to 'healthy' (or 'fail') when it finishes."""
    name = (data.get("job") or "").strip()
    if not name:
        await _ws_send(websocket, {"type": "jobs_run_result",
                                   "ok": False, "error": "missing job name"})
        return
    try:
        import chloe_jobs
    except Exception as e:
        await _ws_send(websocket, {"type": "jobs_run_result",
                                   "ok": False, "error": f"import: {e}"})
        return
    if name not in chloe_jobs.JOBS:
        await _ws_send(websocket, {"type": "jobs_run_result",
                                   "ok": False, "error": f"unknown job: {name}"})
        return

    def _on_complete(job_name, result, ok):
        # When the background job finishes, push a fresh state to every
        # connected client so the CH03 panel updates without polling.
        try:
            import asyncio as _asyncio
            import chloe_jobs as _cj
            snap = _cj.state()
            payload = {"type": "jobs_state_update", **snap}
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    _asyncio.run_coroutine_threadsafe(_ws_broadcast(payload), loop)
                else:
                    _asyncio.run(_ws_broadcast(payload))
            except RuntimeError:
                pass
        except Exception:
            pass

    started = chloe_jobs.run_async(name, on_complete=_on_complete)
    if not started:
        await _ws_send(websocket, {"type": "jobs_run_result",
                                   "ok": False, "error": "already running",
                                   "job": name})
        return
    await _ws_send(websocket, {"type": "jobs_run_result",
                               "ok": True, "job": name, "queued": True})
'''


# ═══════════════════════════════════════════════════════════════════════════
# Splice logic
# ═══════════════════════════════════════════════════════════════════════════

def _backup(p: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak.{label}_{stamp}")
    shutil.copy2(p, bak)
    return bak


def _splice_hud(hud_src: str) -> str:
    if HUD_ANCHOR in hud_src:
        raise SystemExit("hud.html already contains CH03 jobs anchor — "
                         "run --restore first")
    out = hud_src

    # 1) CSS: inject right before the very last </style> in <head>. We pick
    # the first </style> that closes the inline UI stylesheet block (the
    # one that contains '.ch-label' which we know is the channel style block).
    css_target_idx = out.find(".ch-label")
    if css_target_idx < 0:
        raise SystemExit("ch-label CSS anchor not found")
    style_close = out.find("</style>", css_target_idx)
    if style_close < 0:
        raise SystemExit("no </style> after .ch-label")
    out = out[:style_close] + HUD_CSS + "\n" + out[style_close:]

    # 2) Panel HTML: inject after the <div class="lights-panel" ...> block
    # closes. Search for </div> closing the lights-panel.
    lp_idx = out.find('id="lights-panel"')
    if lp_idx < 0:
        raise SystemExit("lights-panel HTML anchor not found")
    # Walk forward, count <div>/</div> depth from the start of lights-panel.
    open_idx = out.rfind("<div", 0, lp_idx)
    depth = 0
    i = open_idx
    end = -1
    while i < len(out):
        if out.startswith("<div", i):
            depth += 1
            i += 4
        elif out.startswith("</div>", i):
            depth -= 1
            i += 6
            if depth == 0:
                end = i
                break
        else:
            i += 1
    if end < 0:
        raise SystemExit("could not locate lights-panel closing </div>")
    out = out[:end] + "\n" + HUD_PANEL_HTML + out[end:]

    # 3) JS toggle: replace the lights-only toggleChannel with the 3-way one.
    if HUD_JS_TOGGLE_OLD not in out:
        raise SystemExit("toggleChannel anchor not found — hud.html structure changed?")
    out = out.replace(HUD_JS_TOGGLE_OLD, HUD_JS_TOGGLE_NEW, 1)

    # 4) WS dispatch: inject jobs_* before lights_state_result.
    if HUD_WS_HANDLER_OLD not in out:
        raise SystemExit("WS lights_state_result anchor not found")
    out = out.replace(HUD_WS_HANDLER_OLD, HUD_WS_HANDLER_NEW, 1)
    return out


def _splice_jarvis(src: str) -> str:
    if JARVIS_ANCHOR in src:
        raise SystemExit("jarvis.py already contains CH03 jobs anchor — "
                         "run --restore first")
    if JARVIS_DISPATCH_OLD not in src:
        raise SystemExit("jarvis.py dispatch anchor not found")
    out = src.replace(JARVIS_DISPATCH_OLD, JARVIS_DISPATCH_NEW, 1)

    # Append handler functions at EOF (after the last existing handler).
    out = out.rstrip() + "\n\n" + JARVIS_HANDLERS + "\n"
    return out


def _restore_latest(p: Path, label: str) -> bool:
    baks = sorted(HERE.glob(f"{p.name}.bak.{label}_*"), reverse=True)
    if not baks:
        return False
    shutil.copy2(baks[0], p)
    print(f"restored {p.name} from {baks[0].name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        n = 0
        if _restore_latest(HUD, "ch03"):
            n += 1
        if _restore_latest(JARVIS, "ch03"):
            n += 1
        print(f"restored {n}/2 files")
        return 0 if n else 1

    hud_src = HUD.read_text(encoding="utf-8")
    jarvis_src = JARVIS.read_text(encoding="utf-8")

    print("=== splicing hud.html (CH03 panel)")
    new_hud = _splice_hud(hud_src)
    print(f"   delta: {len(hud_src):,} -> {len(new_hud):,} "
          f"({len(new_hud)-len(hud_src):+,} bytes)")

    print("=== splicing jarvis.py (jobs_state + jobs_run handlers)")
    new_jarvis = _splice_jarvis(jarvis_src)
    print(f"   delta: {len(jarvis_src):,} -> {len(new_jarvis):,} "
          f"({len(new_jarvis)-len(jarvis_src):+,} bytes)")

    # Parse-check jarvis.py — refuse to write if it breaks.
    try:
        ast.parse(new_jarvis)
        print("   jarvis.py ast.parse OK")
    except SyntaxError as e:
        print(f"   jarvis.py would not parse: {e}", file=sys.stderr)
        return 2

    # Backup + write.
    hud_bak = _backup(HUD, "ch03")
    jarvis_bak = _backup(JARVIS, "ch03")
    print(f"   backups: {hud_bak.name}, {jarvis_bak.name}")

    HUD.write_text(new_hud, encoding="utf-8")
    JARVIS.write_text(new_jarvis, encoding="utf-8")

    # Line-preservation sanity check on hud.html — every line that was
    # in the original should still appear in the new (lesson #12 idea).
    orig_lines = set(hud_src.splitlines())
    new_lines = set(new_hud.splitlines())
    missing = orig_lines - new_lines
    # Strip the toggleChannel block we deliberately rewrote.
    expected_drops = set(HUD_JS_TOGGLE_OLD.splitlines())
    spurious = missing - expected_drops
    if spurious:
        print(f"   WARN: {len(spurious)} original line(s) missing from new hud.html "
              "(beyond expected toggleChannel replacement)")
        for L in list(spurious)[:5]:
            print(f"     - {L[:120]}")
    else:
        print("   hud.html line-preservation OK")

    print("\nDone. Restart Chloe to load the WS handlers, then click CH 01 in the")
    print("HUD twice to cycle to CH 03 JOBS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
