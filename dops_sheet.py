"""dops_sheet.py -- fill today's tab of the Maze "HOM1 Daily Sheet" (Google Sheets).

Same inputs as dops_daily.py (DOPS xlsx from Slack + Cortex roster) but writes
into the real sheet: duplicates the Template tab (or, if there is none, the
newest tab) to a new first tab named like "Oct 10", then fills only the cells
the automation owns:

    B1 date, B2 "<Weekday> Dispatch: <name>", wave header times (if they moved)
    slot rows (A.1-A.5 -> 6-10, B.1-B.5 -> 14-18, C.1-C.2 -> 22-23):
        B label, C driver, E route, G dock, H bag.  D Vehicle / F Helper blank.
        Unused slots get B:F cleared.
    J:O Scheduled Stops, P:Q Pick Ups (R time blank), S:T MNR  (from row 3)

Everything else (attendance, notes, injected pkgs, phones, "Brett" row ...) is
left alone. Re-running the same day overwrites the same tab, so you can re-run
after Cortex updates.

    python dops_sheet.py --cortex cortex_roster.txt --dry-run
    python dops_sheet.py --cortex cortex_roster.txt [--dops file] [--date YYYY-MM-DD]
                         [--dispatcher Ed] [--sheet-id ID]

Setup (once):
  pip install gspread
  Auth, either:
    GOOGLE_SERVICE_ACCOUNT_JSON=path\\to\\key.json   (share the sheet with its
        client_email as Editor), or
    nothing -> gspread.oauth(): one-time browser consent, token cached.
  Sheet id: --sheet-id, DOPS_SHEET_ID, or "google_sheet_id" in dops_config.json.

Tip: first run against a copy of the sheet (File > Make a copy) and point
DOPS_SHEET_ID at the copy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

import dops_daily as dd

MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "June",
          7: "July", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
SLOT_ORDER = ["A.1", "A.2", "A.3", "A.4", "A.5", "B.1", "B.2", "B.3", "B.4", "B.5", "C.1", "C.2"]
SLOT_ROWS = {"A.1": 6, "A.2": 7, "A.3": 8, "A.4": 9, "A.5": 10,
             "B.1": 14, "B.2": 15, "B.3": 16, "B.4": 17, "B.5": 18, "C.1": 22, "C.2": 23}
BAG = {lab: i + 1 for i, lab in enumerate(SLOT_ORDER)}
WAVE_HDR_ROW = {"A": 4, "B": 12, "C": 20}
FIRST_DATA_ROW = 3
CLEAR_RIGHT = "J3:T80"


def tab_title(day: dt.date) -> str:
    return f"{MONTHS[day.month]} {day.day}"


def _dock(label: str, cfg: dict):
    letter, _, idx = label.partition(".")
    try:
        return cfg["docks"][letter][int(idx) - 1]
    except (KeyError, IndexError, ValueError):
        return None


def _window(s: dict) -> str:
    return f"{dd._hr12(s['start'])}-{dd._hr12(s['end'])}" if s["start"] and s["end"] else ""


# ─── planning (pure: no Google calls) ──────────────────────────────────────

def plan_cells(dops: dict, roster: list[tuple[str, str]], day: dt.date, cfg: dict,
               dispatcher: str, col_b: list[str] | None = None) -> dict:
    """Returns {"updates": [{"range","values"}...], "flags": [...], "counts": {...}}.
    col_b = current column-B values (for wave-header time checks); None skips that."""
    assigned, names, flags = dd.match_roster(dops, roster, cfg)
    who = lambda route: names.get(route, route)
    updates: list[dict] = []

    updates.append({"range": "B2", "values": [[f"{day.strftime('%A')} Dispatch: {dispatcher}"]]})

    # slots
    by_label = {r["staging"]: r for r in dops["routes"]}
    for label in by_label:
        if label not in SLOT_ROWS:
            flags.append(f"DOPS staging {label} ({by_label[label]['route']}) has no row on the sheet -- not placed")
    for label in SLOT_ORDER:
        row = SLOT_ROWS[label]
        r = by_label.get(label)
        if r is None:
            updates.append({"range": f"B{row}:F{row}", "values": [["", "", "", "", ""]]})
            continue
        dock = _dock(label, cfg)
        name = names.get(r["route"], f"?? {r['route']}")
        updates.append({"range": f"B{row}:H{row}",
                        "values": [[label, name, "", r["route"], "", dock if dock is not None else "", BAG[label]]]})

    # wave header times (only if DOPS disagrees with what the tab says)
    if col_b is not None:
        waves = sorted({r["wave"] for r in dops["routes"]})
        for letter, wtime in zip("ABC", waves):
            hrow = WAVE_HDR_ROW[letter]
            cur = col_b[hrow - 1] if len(col_b) >= hrow else ""
            want = f"{wtime.hour % 12 or 12}:{wtime.minute:02d}"
            m = re.search(r"(\d{1,2}:\d{2})\s*([AP]M)", cur, re.I)
            if m and m.group(1) != want:
                new = cur[:m.start(1)] + want + cur[m.end(1):]
                updates.append({"range": f"B{hrow}", "values": [[new]]})
                flags.append(f"Wave {letter} time changed to {want}: header updated; check RTS and OTD cells by hand")

    # right-hand blocks
    sd, cx, mn = [], [], []
    for route, stops in dops["sd"].items():
        sd += [[s["tba"], who(route), s["order"], _window(s), s["service"], s["city"]] for s in stops] + [[""] * 6]
        if route not in {r["route"] for r in dops["routes"]}:
            flags.append(f"SD has stops for {route}, which is not in DOPS Solution")
    for route, ids in dops["cxl"].items():
        cx += [[t, who(route), ""] for t in ids] + [["", "", ""]]
    for route, ids in dops["mnr"].items():
        mn += [[t, who(route)] for t in ids]
    sd, cx = (sd[:-1] if sd else sd), (cx[:-1] if cx else cx)       # no trailing separator
    for rng, block, width in (("J", sd, "O"), ("P", cx, "R"), ("S", mn, "T")):
        if block:
            updates.append({"range": f"{rng}{FIRST_DATA_ROW}:{width}{FIRST_DATA_ROW + len(block) - 1}", "values": block})
    sd_ids = {s["tba"] for stops in dops["sd"].values() for s in stops}
    for route, ids in dops["mnr"].items():
        flags += [f"{t} appears in both SD and MNR ({route})" for t in ids if t in sd_ids]
    if dops["ex_count"]:
        flags.append(f"EX sheet has {dops['ex_count']} package(s); not placed")
    return {"updates": updates, "flags": flags,
            "counts": {"routes": len(dops["routes"]), "sd": len(sd), "cxl": len(cx), "mnr": len(mn)}}


# ─── Google I/O ────────────────────────────────────────────────────────────

def _client():
    import gspread
    sa = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    return gspread.service_account(filename=sa) if sa else gspread.oauth()


def _http(ss):
    c = ss.client
    return getattr(c, "http_client", c)


def _get_grid(ss, title: str, a1: str, fields: str) -> list:
    r = _http(ss).request("get", f"https://sheets.googleapis.com/v4/spreadsheets/{ss.id}",
                          params={"ranges": f"'{title}'!{a1}", "includeGridData": "true", "fields": fields})
    return r.json()["sheets"][0]["data"][0].get("rowData", [])


def _set_date_header(ss, ws, day: dt.date, flags: list) -> None:
    """B1 holds 'HOM 1 ... Date: m/d' with mixed font sizes. Replace the date and
    re-apply the cell's existing text-format runs so the styling survives."""
    cur = (ws.get("B1") or [[""]])[0][0] if ws.get("B1") else ""
    new = re.sub(r"Date:\s*\d{1,2}/\d{1,2}", f"Date: {day.month}/{day.day}", cur)
    if not cur or new == cur:
        if not cur:
            flags.append("B1 is empty on this tab; date not written")
        return
    try:
        rows = _get_grid(ss, ws.title, "B1", "sheets.data.rowData.values.textFormatRuns")
        runs = (rows[0]["values"][0].get("textFormatRuns") if rows and rows[0].get("values") else None)
        cell = {"userEnteredValue": {"stringValue": new}}
        fields = "userEnteredValue"
        if runs:
            cell["textFormatRuns"] = runs
            fields += ",textFormatRuns"
        ss.batch_update({"requests": [{"updateCells": {
            "rows": [{"values": [cell]}], "fields": fields,
            "start": {"sheetId": ws.id, "rowIndex": 0, "columnIndex": 1}}}]})
    except Exception as e:
        ws.update_acell("B1", new)
        flags.append(f"B1 date written as plain text (font mix may be lost): {type(e).__name__}")


def _band_colors(ss, ws) -> dict | None:
    """Sample band/pick-up/MNR colours from the tab we duplicated (J3:T60)."""
    rows = _get_grid(ss, ws.title, "J3:T60", "sheets.data.rowData.values.userEnteredFormat.backgroundColor")
    def color(row, j):
        try:
            return row["values"][j]["userEnteredFormat"]["backgroundColor"]
        except (KeyError, IndexError, TypeError):
            return None
    distinct = []
    for r in rows:
        c = color(r, 0)
        if c and c not in distinct and any(v < 0.99 for v in c.values()):
            distinct.append(c)
    if not rows or not distinct:
        return None
    return {"bands": [distinct[0], distinct[1] if len(distinct) > 1 else distinct[0]],
            "time": color(rows[0], 3), "pick": color(rows[0], 6), "mnr": color(rows[0], 9)}


def _paint(ws_id: int, c0: int, c1: int, r0: int, r1: int, color: dict | None) -> dict:
    return {"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": r0, "endRowIndex": r1,
                  "startColumnIndex": c0, "endColumnIndex": c1},
        "cell": {"userEnteredFormat": {"backgroundColor": color or {"red": 1, "green": 1, "blue": 1}}},
        "fields": "userEnteredFormat.backgroundColor"}}


def _format_blocks(ss, ws, counts: dict, dops: dict, flags: list) -> None:
    """Row colours for the right-hand blocks follow the new group sizes; rows
    left over from the duplicated tab go back to white. Values are already in."""
    pal = _band_colors(ss, ws)
    if not pal:
        flags.append("Right-block colours not re-applied (couldn't sample the tab)")
        return
    reqs, r = [], FIRST_DATA_ROW - 1                      # 0-based
    g = 0
    for route, stops in dops["sd"].items():
        band = pal["bands"][g % 2]
        n = len(stops)
        reqs.append(_paint(ws.id, 9, 15, r, r + n, band))              # J:O
        reqs.append(_paint(ws.id, 12, 13, r, r + n, pal["time"] or band))  # M (time)
        r += n
        g += 1
        if g < len(dops["sd"]):                                         # separator takes the NEXT group's colour
            reqs.append(_paint(ws.id, 9, 15, r, r + 1, pal["bands"][g % 2]))
            r += 1
    reqs.append(_paint(ws.id, 9, 15, r, 80, None))
    reqs.append(_paint(ws.id, 15, 18, FIRST_DATA_ROW - 1, FIRST_DATA_ROW - 1 + counts["cxl"], pal["pick"]))
    reqs.append(_paint(ws.id, 15, 18, FIRST_DATA_ROW - 1 + counts["cxl"], 80, None))
    reqs.append(_paint(ws.id, 18, 20, FIRST_DATA_ROW - 1, FIRST_DATA_ROW - 1 + counts["mnr"], pal["mnr"]))
    reqs.append(_paint(ws.id, 18, 20, FIRST_DATA_ROW - 1 + counts["mnr"], 80, None))
    ss.batch_update({"requests": reqs})


def run(a) -> int:
    cfg = dd.load_config()
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(dd.TZ).date()
    dispatcher = a.dispatcher or cfg["dispatcher"]
    out_dir = Path(a.out)
    dops_path = Path(a.dops) if a.dops else dd.fetch_dops_from_slack(day, out_dir / "inbox", cfg)
    dops = dd.read_dops(dops_path)
    roster = dd.read_cortex(a.cortex)
    title = tab_title(day)

    if a.dry_run:
        plan = plan_cells(dops, roster, day, cfg, dispatcher)
        print(f"[dry-run] tab {title!r}: {plan['counts']}")
        for u in plan["updates"][:14]:
            print(f"  {u['range']:<8} {u['values'][0][:7]}")
        for f in plan["flags"]:
            print(f"  ! {f}")
        return 0

    sheet_id = a.sheet_id or os.environ.get("DOPS_SHEET_ID") or cfg["google_sheet_id"]
    if not sheet_id:
        raise SystemExit("No sheet id: --sheet-id, DOPS_SHEET_ID, or google_sheet_id in dops_config.json")
    import gspread
    ss = _client().open_by_key(sheet_id)
    extra: list[str] = []
    existed = True
    try:
        ws = ss.worksheet(title)
        print(f"tab {title!r} exists -- overwriting the automated cells")
        if not a.dispatcher:                       # re-run: keep whoever is already named on the tab
            cur = ((ws.get("B2") or [[""]])[0] or [""])[0]
            m = re.search(r"Dispatch:\s*(.+)$", cur)
            if m and m.group(1).strip():
                dispatcher = m.group(1).strip()
    except gspread.WorksheetNotFound:
        existed = False
        try:
            src = ss.worksheet(cfg["template_tab"])
        except gspread.WorksheetNotFound:
            src = ss.get_worksheet(0)
            extra.append(f"No {cfg['template_tab']!r} tab: duplicated {src.title!r}; its attendance/notes carried over -- clear by hand")
        ws = ss.duplicate_sheet(src.id, insert_sheet_index=0, new_sheet_name=title)
        print(f"created tab {title!r} from {src.title!r}")

    col_b = ws.col_values(2)
    plan = plan_cells(dops, roster, day, cfg, dispatcher, col_b)
    plan["flags"] += extra
    ws.batch_clear([CLEAR_RIGHT])
    ws.batch_update(plan["updates"], value_input_option="USER_ENTERED")
    _set_date_header(ss, ws, day, plan["flags"])
    try:
        _format_blocks(ss, ws, plan["counts"], dops, plan["flags"])
    except Exception as e:
        plan["flags"].append(f"Right-block colours not re-applied: {type(e).__name__}: {e}")
    print(f"filled {title!r}: {plan['counts']}")
    for f in plan["flags"]:
        print(f"  ! {f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cortex", required=True)
    ap.add_argument("--dops")
    ap.add_argument("--date")
    ap.add_argument("--dispatcher")
    ap.add_argument("--sheet-id")
    ap.add_argument("--out", default="dops_out")
    ap.add_argument("--dry-run", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
