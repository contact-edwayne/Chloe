"""Central-time helpers — shared source of truth for the date/time stamp
injected into Chloe's prompts.

US Central is computed correctly regardless of the `tzdata` package (missing
in venv_py314 on Windows, which makes zoneinfo's tz database unavailable) AND
regardless of the PC's configured timezone: the fallback derives Central from
UTC with a pure-arithmetic CST/CDT DST rule.

Currently consumed by brain_http.py's side-panel chat. jarvis.py still carries
an equivalent inline copy (_central_now/_now_block) — migrating it to import
from here is a trivial future cleanup.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta


def us_central_is_dst(utc_dt: datetime) -> bool:
    """True if US Central is on daylight time (CDT) at the given UTC instant.

    DST (since 2007): 2nd Sunday of March 08:00 UTC (02:00 CST) →
    1st Sunday of November 07:00 UTC (02:00 CDT)."""
    y = utc_dt.year
    mar1 = datetime(y, 3, 1)
    first_sun_mar = 1 + (6 - mar1.weekday()) % 7         # Mon=0..Sun=6
    dst_start = datetime(y, 3, first_sun_mar + 7, 8, 0)  # 2nd Sun, 08:00 UTC
    nov1 = datetime(y, 11, 1)
    first_sun_nov = 1 + (6 - nov1.weekday()) % 7
    dst_end = datetime(y, 11, first_sun_nov, 7, 0)       # 1st Sun, 07:00 UTC
    naive = utc_dt.replace(tzinfo=None)
    return dst_start <= naive < dst_end


def central_now() -> datetime:
    """US-Central 'now' as a tz-aware datetime. Uses zoneinfo when tzdata is
    present; otherwise computes Central from UTC with a fixed CST/CDT offset
    (DST-correct, independent of the PC's configured timezone). `%Z` yields
    CST/CDT on both paths."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        utc = datetime.now(timezone.utc)
        off = -5 if us_central_is_dst(utc) else -6
        return utc.astimezone(timezone(timedelta(hours=off),
                                       "CDT" if off == -5 else "CST"))


def now_block() -> str:
    """Current date + time in Ed's timezone (US Central), formatted for
    injection into a system prompt so the model always knows what time it is."""
    now = central_now()
    tzname = now.strftime("%Z") or "Central"
    hour12 = now.hour % 12 or 12
    stamp = (f"{now.strftime('%A, %B')} {now.day}, {now.year} at "
             f"{hour12}:{now.strftime('%M %p')} {tzname}")
    return (f"\n\nCURRENT DATE & TIME (Ed's timezone): {stamp}. "
            f"Use this whenever he asks the date or time — state it "
            f"confidently, never guess or say you don't know.")
