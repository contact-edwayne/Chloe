"""Real-time weather for Chloe — Open-Meteo + IP geolocation, no API key.

Public API
----------
weather_reply(place: str | None = None) -> str
    A spoken-friendly current-conditions + short-forecast line. When `place`
    is None the location resolves from CHLOE_WEATHER_LOCATION (if set) else
    IP geolocation of this machine (Ed chose auto-detect-from-PC).

maybe_weather_reply(user_text: str) -> str | None
    Intent-detect a weather question, pull an optional "in <city>" location,
    and return weather_reply(...). Returns None when the text isn't a weather
    ask, so the caller falls through to normal routing.

Units default to Fahrenheit / mph. Set CHLOE_WEATHER_UNITS=metric for C / km/h.

All network calls are short-timeout and best-effort: on any failure the
functions return a plain English apology rather than raising, so a wedged
endpoint can never break the chat/voice turn.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

_TIMEOUT = 6.0
_UA = "Chloe-weather/1.0 (+local home assistant)"

# caches so repeated asks don't re-hit the network
_geo_cache: dict[str, tuple] = {}          # place(lower) -> (lat, lon, label, ts)
_ip_cache: dict = {"data": None, "ts": 0.0}
_IP_TTL = 1800.0                            # re-detect IP location every 30 min
_GEO_TTL = 86400.0

# WMO weather-interpretation codes -> short text
_WMO = {
    0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}

# words that, after "in/for/at/near", are NOT a place (locations, filler, or
# temporal phrases like "forecast for tomorrow")
_NOT_A_PLACE = {
    "here", "there", "outside", "inside", "the morning", "the afternoon",
    "the evening", "the moment", "celsius", "fahrenheit", "general",
    "today", "tomorrow", "tonight", "now", "right now", "yesterday",
    "this week", "this morning", "this afternoon", "this evening",
    "a bit", "later", "real time", "realtime", "a while",
}

# substrings that mark a weather question
_WX_TRIGGERS = (
    "weather", "forecast", "temperature", "how hot", "how cold", "how warm",
    "is it raining", "is it snowing", "is it sunny", "is it cold", "is it hot",
    "is it warm", "rain today", "snowing", "raining", "humidity",
    "wind speed", "how windy", "degrees out", "feels like out", "chance of rain",
)


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _units() -> dict:
    metric = os.environ.get("CHLOE_WEATHER_UNITS", "").strip().lower() in (
        "metric", "c", "celsius", "si")
    if metric:
        return {"temperature_unit": "celsius", "wind_speed_unit": "kmh",
                "tl": "°C", "wl": "km/h"}
    return {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "tl": "°F", "wl": "mph"}


def _ip_location():
    """(lat, lon, label) for this machine via IP geolocation, or None."""
    now = time.time()
    if _ip_cache["data"] and now - _ip_cache["ts"] < _IP_TTL:
        return _ip_cache["data"]
    providers = (
        ("https://ipapi.co/json/",
         lambda d: (d.get("latitude"), d.get("longitude"),
                    ", ".join(x for x in (d.get("city"),
                              d.get("region_code") or d.get("region")) if x)
                    or d.get("city") or "your area")),
        ("http://ip-api.com/json/",
         lambda d: (d.get("lat"), d.get("lon"),
                    ", ".join(x for x in (d.get("city"), d.get("region")) if x)
                    or "your area")),
    )
    for url, parse in providers:
        try:
            d = _get_json(url)
            lat, lon, label = parse(d)
            if lat is not None and lon is not None:
                res = (float(lat), float(lon), label)
                _ip_cache["data"], _ip_cache["ts"] = res, now
                return res
        except Exception:
            continue
    return None


def _geocode(place: str):
    """(lat, lon, label) for a named place via Open-Meteo geocoding, or None."""
    key = place.strip().lower()
    if not key:
        return None
    hit = _geo_cache.get(key)
    if hit and time.time() - hit[3] < _GEO_TTL:
        return hit[:3]
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search?" + \
            urllib.parse.urlencode({"name": place, "count": 1,
                                    "language": "en", "format": "json"})
        d = _get_json(url)
        results = d.get("results") or []
        if not results:
            return None
        r0 = results[0]
        label = ", ".join(x for x in (r0.get("name"), r0.get("admin1"),
                          r0.get("country_code")) if x)
        out = (float(r0["latitude"]), float(r0["longitude"]), label or place)
        _geo_cache[key] = (*out, time.time())
        return out
    except Exception:
        return None


def _resolve_location(place):
    """Resolve to (lat, lon, label). place arg wins, then env, then IP."""
    if place:
        return _geocode(place)            # None => caller reports "couldn't find"
    env = os.environ.get("CHLOE_WEATHER_LOCATION", "").strip()
    if env:
        g = _geocode(env)
        if g:
            return g
    return _ip_location()


def _fetch(lat: float, lon: float):
    u = _units()
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "weather_code,wind_speed_10m,is_day"),
        "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                  "precipitation_probability_max"),
        "temperature_unit": u["temperature_unit"],
        "wind_speed_unit": u["wind_speed_unit"],
        "timezone": "auto", "forecast_days": 1,
    })
    return _get_json(url), u


def weather_reply(place: str | None = None) -> str:
    loc = _resolve_location(place)
    if loc is None:
        if place:
            return f"I couldn't find a place called {place}."
        return ("I couldn't pin down your location for the weather right now — "
                "try asking with a city, like \"weather in Austin.\"")
    lat, lon, label = loc
    try:
        data, u = _fetch(lat, lon)
    except Exception:
        return "I couldn't reach the weather service just now — try again in a moment."

    cur = data.get("current") or {}
    daily = data.get("daily") or {}
    tl, wl = u["tl"], u["wl"]

    try:
        temp = round(cur["temperature_2m"])
    except Exception:
        return "The weather service gave me something I couldn't read — try again shortly."
    feels = round(cur.get("apparent_temperature", temp))
    wind = round(cur.get("wind_speed_10m", 0) or 0)
    cond = _WMO.get(int(cur.get("weather_code", -1)), "")

    head = f"Right now in {label} it's {temp}{tl}"
    if cond:
        head += f" with {cond}"
    extras = []
    if abs(feels - temp) >= 3:
        extras.append(f"feels like {feels}{tl}")
    if wind:
        extras.append(f"wind {wind} {wl}")
    if extras:
        head += " (" + ", ".join(extras) + ")"
    sentence = head + "."

    try:
        hi = round(daily["temperature_2m_max"][0])
        lo = round(daily["temperature_2m_min"][0])
        seg = f" Today: high of {hi}{tl}, low of {lo}{tl}"
        pop = (daily.get("precipitation_probability_max") or [None])[0]
        if pop is not None:
            seg += f", {pop}% chance of precipitation"
        sentence += seg + "."
    except Exception:
        pass
    return sentence


def _extract_place(text: str):
    """Pull a place out of '... in/for/at/near <place>'. None if absent."""
    m = re.search(
        r"\b(?:in|for|at|near)\s+([A-Za-z][A-Za-z .'\-]*?)"
        r"(?:\s+(?:today|tomorrow|right now|now|this week|tonight|please|"
        r"currently))?[\?\.!,]*\s*$",
        text, re.I)
    if not m:
        return None
    place = m.group(1).strip(" .,?!").strip()
    if not place or place.lower() in _NOT_A_PLACE:
        return None
    return place


def maybe_weather_reply(user_text: str):
    """Return a weather reply if `user_text` is a weather question, else None.

    BUG FIXED 2026-09-06: plain substring matching against _WX_TRIGGERS
    let "raining" match inside any unrelated word that happens to
    contain it -- confirmed live: "...full training?" (an email-reading
    request, nothing to do with weather) matched "raining" inside
    "training" and got answered with the local forecast instead of
    routing to the actual question. Trigger phrases are now matched on
    word boundaries via regex so a trigger only fires as a whole word/
    phrase, never as merely a contiguous substring of a longer,
    unrelated word."""
    if not user_text:
        return None
    low = user_text.lower()
    if not any(re.search(r'\b' + re.escape(k) + r'\b', low) for k in _WX_TRIGGERS):
        return None
    return weather_reply(_extract_place(user_text))


if __name__ == "__main__":
    import sys
    arg = " ".join(sys.argv[1:]).strip() or None
    print(weather_reply(arg))
