"""
Zengge / Magic Home WiFi bulb control for Chloe.

These are the bulbs sold under "Magic Home", "MagicHue", "Surplife",
"LEDnet", and various other Zengge white-label brands. Identified by
JMZengge OUI (08:65:F0:...). Listen on TCP port 5577, discoverable via
UDP broadcast on port 48899. Local LAN — no cloud, no hub.

Built on flux_led (the same library Home Assistant uses).

Config persisted to C:\\Chloe\\secrets\\lights.json. Discovery saves the
list of bulbs; the user maps each MAC/IP to a friendly name once.
After that, Chloe references bulbs by name ("bedroom", "office", "all").

Surfaces:
    - discover()                  -> list of {mac, ip, model}
    - name_bulb(mac, name)        -> save mapping
    - get_bulb(name)              -> WifiLedBulb instance (cached) or None
    - list_bulbs()                -> [{name, mac, ip, online, on, ...}]
    - set_state(target, on=None, brightness=None, color=None, ct=None)
    - parse_intent(text)          -> kwargs for set_state, or None

CLI:
    python lights.py --discover           # find bulbs on LAN
    python lights.py --name-interactive   # blink each, prompt for name
    python lights.py --list               # show named bulbs + state
    python lights.py --test               # cycle first bulb on/blue/dim/off
    python lights.py "turn off the bedroom"
"""

from __future__ import annotations

import json
import sys
import time
import re
from pathlib import Path
from typing import Optional

import socket

try:
    from flux_led import WifiLedBulb
    HAVE_FLUX = True
except ImportError:
    HAVE_FLUX = False

SECRETS_DIR = Path(r"C:\Chloe\secrets")
CONFIG_PATH = SECRETS_DIR / "lights.json"

# Common color names → RGB tuples. Magic Home bulbs accept setRgb directly.
COLORS_RGB = {
    "red":     (255,   0,   0),
    "orange":  (255, 110,   0),
    "amber":   (255,  90,   0),
    "yellow":  (255, 230,   0),
    "lime":    (180, 255,   0),
    "green":   (  0, 255,   0),
    "mint":    ( 60, 255, 160),
    "cyan":    (  0, 230, 230),
    "teal":    (  0, 180, 180),
    "blue":    (  0,   0, 255),
    "indigo":  ( 75,   0, 180),
    "violet":  (140,   0, 255),
    "purple":  (180,   0, 220),
    "magenta": (255,   0, 200),
    "pink":    (255,  80, 180),
    "white":   (255, 255, 255),
}

# Color temperature in Kelvin (used by setWhiteTemperature on RGBW/CCT bulbs).
# For RGB-only bulbs we fall back to an RGB approximation.
COLOR_TEMPS_K = {
    "candle":      2200,
    "warm white":  2700,
    "warm":        2700,
    "soft white":  3000,
    "soft":        3000,
    "neutral":     4000,
    "daylight":    5000,
    "cool white":  5700,
    "cool":        5700,
    "icy":         6500,
}

# RGB approximations of the same color temps (for bulbs without a white channel)
CT_RGB_FALLBACK = {
    2200: (255, 147,  41),
    2700: (255, 169,  87),
    3000: (255, 180, 107),
    4000: (255, 209, 163),
    5000: (255, 228, 206),
    5700: (255, 236, 224),
    6500: (255, 249, 253),
}


# --------------------------------------------------------------------------- #
# Config persistence                                                          #
# --------------------------------------------------------------------------- #

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"bulbs": []}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        print(f"[lights] config load failed: {e}", file=sys.stderr)
        return {"bulbs": []}


def _save_config(cfg: dict) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #

def discover(timeout: int = 5) -> list[dict]:
    """UDP broadcast discovery on port 48899. Returns list of
    {mac, ip, model} dicts. Merges into config without overwriting names.

    Native socket implementation — flux_led's BulbScanner has a buffer-size
    bug on Windows (WinError 10040). The Zengge discovery protocol is just:
    broadcast 'HF-A11ASSISTHREAD' to 255.255.255.255:48899 and parse the
    'IP,MAC,MODEL' replies."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", 0))
        sock.sendto(b"HF-A11ASSISTHREAD", ("255.255.255.255", 48899))

        found: list[dict] = []
        seen_macs: set[str] = set()
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                break
            except OSError:
                continue
            text = data.decode("ascii", errors="ignore").strip()
            # Bulbs reply with "IP,MAC,MODEL" — own broadcast can echo back, skip
            if text == "HF-A11ASSISTHREAD":
                continue
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            ip, mac = parts[0], parts[1].upper().replace(":", "")
            model = parts[2] if len(parts) >= 3 else None
            if mac in seen_macs:
                continue
            seen_macs.add(mac)
            found.append({"mac": mac, "ip": ip, "model": model})
    finally:
        sock.close()

    cfg = _load_config()
    by_mac = {x["mac"]: x for x in cfg.get("bulbs", [])}
    for b in found:
        if b["mac"] in by_mac:
            # update IP (may have changed via DHCP) and model, keep name
            by_mac[b["mac"]]["ip"]    = b["ip"]
            by_mac[b["mac"]]["model"] = b["model"]
        else:
            by_mac[b["mac"]] = {"name": "", **b}
    cfg["bulbs"] = list(by_mac.values())
    cfg["last_discovered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_config(cfg)
    return found


def name_bulb(mac: str, name: str) -> bool:
    cfg = _load_config()
    mac = mac.upper().replace(":", "")
    for b in cfg.get("bulbs", []):
        if b["mac"] == mac:
            b["name"] = name.strip().lower()
            _save_config(cfg)
            return True
    return False


# --------------------------------------------------------------------------- #
# Bulb access (with simple connection cache)                                  #
# --------------------------------------------------------------------------- #

_bulb_cache: dict[str, WifiLedBulb] = {}


def _connect(ip: str) -> Optional[WifiLedBulb]:
    """Return a connected WifiLedBulb for `ip`, cached. None on failure."""
    if not HAVE_FLUX:
        return None
    bulb = _bulb_cache.get(ip)
    if bulb is not None:
        try:
            bulb.update_state()
            return bulb
        except Exception:
            _bulb_cache.pop(ip, None)
    try:
        bulb = WifiLedBulb(ip)
        bulb.update_state()
        _bulb_cache[ip] = bulb
        return bulb
    except Exception as e:
        print(f"[lights] connect {ip} failed: {e}", file=sys.stderr)
        return None


def list_bulbs() -> list[dict]:
    """Return current state of every named bulb in config."""
    cfg = _load_config()
    out = []
    for b in cfg.get("bulbs", []):
        info = dict(b)
        bulb = _connect(b["ip"])
        if bulb is None:
            info["online"] = False
        else:
            info["online"] = True
            try:
                info["on"]         = bool(bulb.is_on)
                info["brightness"] = int(getattr(bulb, "brightness", 0) or 0)
                info["mode"]       = str(getattr(bulb, "mode", "") or "")
            except Exception:
                pass
        out.append(info)
    return out


def _resolve_targets(name: str) -> list[dict]:
    """Resolve a free-text target into a list of bulb config entries."""
    cfg = _load_config()
    bulbs = [b for b in cfg.get("bulbs", []) if b.get("name")]
    if not bulbs:
        return []
    n = (name or "").strip().lower()
    if not n or n in ("all", "everything", "everywhere", "house", "lights"):
        return bulbs
    # exact match
    for b in bulbs:
        if b["name"] == n:
            return [b]
    # substring
    matches = [b for b in bulbs if n in b["name"]]
    return matches


# --------------------------------------------------------------------------- #
# State setter                                                                #
# --------------------------------------------------------------------------- #

def _has_white_channel(bulb: WifiLedBulb) -> bool:
    """Best-effort check for an RGBW or CCT bulb."""
    for attr in ("rgbwprotocol", "rgbwcapable", "_rgbwcapable"):
        v = getattr(bulb, attr, None)
        if isinstance(v, bool) and v:
            return True
    # newer flux_led exposes a "color_modes" set
    modes = getattr(bulb, "color_modes", None)
    if modes and ({"WHITE", "CCT", "RGBW", "RGBWW"} & set(modes)):
        return True
    return False


def _scale_rgb(rgb: tuple[int, int, int], pct: int) -> tuple[int, int, int]:
    f = max(0, min(100, pct)) / 100.0
    return tuple(int(round(c * f)) for c in rgb)


def set_state(
    target: str,
    on: Optional[bool] = None,
    brightness: Optional[int] = None,   # 0-100 percent
    color: Optional[str] = None,        # color name from COLORS_RGB
    ct: Optional[str | int] = None,     # name from COLOR_TEMPS_K or raw kelvin
    transition: Optional[float] = None, # seconds (best-effort; not all bulbs honor)
) -> dict:
    """Apply a state change to one or more bulbs matching `target`.
    Returns {ok, target, results: [{bulb_name, ok, error?}]}."""
    targets = _resolve_targets(target)
    if not targets:
        return {"ok": False, "error": f"no bulb matching {target!r} (have you named them?)"}

    # Resolve color/ct intent into the actual values we'll send
    rgb: Optional[tuple[int, int, int]] = None
    kelvin: Optional[int] = None
    if color is not None:
        c = color.strip().lower()
        if c not in COLORS_RGB:
            return {"ok": False, "error": f"unknown color {color!r}"}
        rgb = COLORS_RGB[c]
    if ct is not None:
        if isinstance(ct, str):
            t = ct.strip().lower()
            if t not in COLOR_TEMPS_K:
                return {"ok": False, "error": f"unknown color temp {ct!r}"}
            kelvin = COLOR_TEMPS_K[t]
        else:
            kelvin = max(1700, min(6500, int(ct)))

    # If brightness given alone (no color/ct), scale current state.
    # Easiest universal approach: re-send last RGB at new brightness.
    # But we don't track last RGB. So if bulb is RGBW, use setBrightness;
    # if RGB-only and no color specified, fall back to white at given brightness.
    pct = brightness  # may be None

    results = []
    for b in targets:
        r = _apply_one(b, on=on, pct=pct, rgb=rgb, kelvin=kelvin, transition=transition)
        results.append(r)

    overall_ok = all(r.get("ok") for r in results) if results else False
    return {"ok": overall_ok, "target": target, "results": results,
            "intent": {"on": on, "brightness": pct, "color": color, "ct": ct}}


def _apply_one(b: dict, on, pct, rgb, kelvin, transition) -> dict:
    name = b.get("name") or b.get("mac")
    bulb = _connect(b["ip"])
    if bulb is None:
        return {"bulb_name": name, "ok": False, "error": "offline"}

    try:
        # ON / OFF
        if on is False:
            bulb.turnOff()
            return {"bulb_name": name, "ok": True, "action": "off"}

        if on is True or rgb is not None or kelvin is not None or pct is not None:
            # any active intent implies bulb should be on
            if not bulb.is_on:
                bulb.turnOn()

        # Color (RGB)
        if rgb is not None:
            r, g, bl = rgb
            if pct is not None:
                r, g, bl = _scale_rgb((r, g, bl), pct)
            try:
                bulb.setRgb(r, g, bl, persist=True)
            except TypeError:
                bulb.setRgb(r, g, bl)
            return {"bulb_name": name, "ok": True, "action": f"rgb({r},{g},{bl})"}

        # Color temperature
        if kelvin is not None:
            bri = pct if pct is not None else 100
            level_255 = max(1, min(255, int(round(bri * 255 / 100))))
            if _has_white_channel(bulb):
                try:
                    bulb.setWhiteTemperature(kelvin, level_255)
                    return {"bulb_name": name, "ok": True, "action": f"ct({kelvin}K, {bri}%)"}
                except Exception:
                    pass  # fall through to RGB approximation
            # closest fallback CT entry
            closest = min(CT_RGB_FALLBACK.keys(), key=lambda k: abs(k - kelvin))
            r, g, bl = _scale_rgb(CT_RGB_FALLBACK[closest], bri)
            try:
                bulb.setRgb(r, g, bl, persist=True)
            except TypeError:
                bulb.setRgb(r, g, bl)
            return {"bulb_name": name, "ok": True, "action": f"ct~rgb({r},{g},{bl})"}

        # Brightness only
        if pct is not None:
            level_255 = max(1, min(255, int(round(pct * 255 / 100))))
            try:
                bulb.setBrightness(level_255)
                return {"bulb_name": name, "ok": True, "action": f"brightness({pct}%)"}
            except AttributeError:
                # older flux_led: re-send white at this level
                r = g = bl = level_255
                try:
                    bulb.setRgb(r, g, bl, persist=True)
                except TypeError:
                    bulb.setRgb(r, g, bl)
                return {"bulb_name": name, "ok": True, "action": f"white({pct}%)"}

        # Plain on
        if on is True:
            return {"bulb_name": name, "ok": True, "action": "on"}

        return {"bulb_name": name, "ok": False, "error": "empty intent"}
    except Exception as e:
        # connection might be dead — drop from cache so next call retries
        _bulb_cache.pop(b["ip"], None)
        return {"bulb_name": name, "ok": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# Intent parser (unchanged from Hue version — pure text)                      #
# --------------------------------------------------------------------------- #

LIGHT_WORDS  = {"light", "lights", "lamp", "lamps", "bulb", "bulbs", "lighting"}
OFF_WORDS    = {"off", "out", "kill", "stop"}
ON_WORDS     = {"on"}
DIM_WORDS    = {"dim", "dimmer", "lower"}
BRIGHT_WORDS = {"brighter", "brighten", "bright", "boost"}

FILLER = {"the", "a", "an", "to", "in", "of", "my", "please", "chloe",
          "hey", "turn", "set", "make", "switch", "color", "colour",
          "lights", "light", "lamp", "lamps", "bulb", "bulbs",
          "on", "off", "down", "up", "dim", "brighter", "brighten",
          "bright", "lower", "raise", "boost", "out", "stop", "kill",
          "percent", "%"}


def parse_intent(text: str) -> Optional[dict]:
    """Return kwargs for set_state(), or None if this isn't a lights command."""
    if not text:
        return None
    raw = text.strip().lower()
    has_light_word = any(w in raw for w in LIGHT_WORDS)
    starts_with_action = re.match(r"^\s*(turn|set|make|switch|dim|brighten)\b", raw)
    if not has_light_word and not starts_with_action:
        return None

    args: dict = {"target": None}

    for cname in COLORS_RGB:
        if re.search(rf"\b{cname}\b", raw):
            args["color"] = cname
            break

    for tname in sorted(COLOR_TEMPS_K, key=len, reverse=True):
        if re.search(rf"\b{re.escape(tname)}\b", raw):
            args["ct"] = tname
            break

    m = re.search(r"(\d{1,3})\s*%", raw) or re.search(r"\bto\s+(\d{1,3})\b", raw)
    if m:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            args["brightness"] = pct

    tokens = set(re.findall(r"[a-z]+", raw))
    if tokens & OFF_WORDS:
        args["on"] = False
    elif tokens & ON_WORDS and not args.get("color") and not args.get("ct") and "brightness" not in args:
        if (re.search(r"\b(turn|switch)\s+on\b", raw)
                or re.search(r"\blights?\s+on\b", raw)
                or re.search(r"\bon\s+the\b", raw)):
            args["on"] = True
    if tokens & DIM_WORDS and "brightness" not in args:
        args["brightness"] = 30
    if tokens & BRIGHT_WORDS and "brightness" not in args:
        args["brightness"] = 100

    cleaned = re.sub(r"[^\w\s]", " ", raw)
    parts = [p for p in cleaned.split() if p not in FILLER and not p.isdigit()]
    if args.get("color"):
        parts = [p for p in parts if p != args["color"]]
    if args.get("ct"):
        ct_words = set(args["ct"].split())
        parts = [p for p in parts if p not in ct_words]
    target = " ".join(parts).strip()
    if not target:
        target = "all"
    args["target"] = target

    if not any(args.get(k) is not None for k in ("on", "brightness", "color", "ct")):
        return None
    return args


# --------------------------------------------------------------------------- #
# Presets + raw actions (for HUD CH02 panel)                                  #
# --------------------------------------------------------------------------- #

# Default presets seeded into config on first run. "all" sentinel = apply to
# every named bulb. User-saved presets store per-bulb state instead.
DEFAULT_PRESETS = [
    {"name": "movie",    "all": {"on": True,  "ct": 2200, "brightness": 10}},
    {"name": "reading",  "all": {"on": True,  "ct": 4000, "brightness": 80}},
    {"name": "focus",    "all": {"on": True,  "ct": 5500, "brightness": 100}},
    {"name": "sunset",   "all": {"on": True,  "rgb": [255, 100, 30], "brightness": 60}},
    {"name": "romantic", "all": {"on": True,  "rgb": [180,   0, 30], "brightness": 30}},
    {"name": "off",      "all": {"on": False}},
]


def ensure_presets() -> None:
    """Seed DEFAULT_PRESETS into config if no presets exist yet. Idempotent."""
    cfg = _load_config()
    if cfg.get("presets"):
        return
    cfg["presets"] = [dict(p) for p in DEFAULT_PRESETS]
    _save_config(cfg)


def get_state_snapshot() -> dict:
    """Full state dump for the HUD: bulbs (with live state) + presets."""
    ensure_presets()
    cfg = _load_config()
    return {
        "bulbs":   list_bulbs(),
        "presets": cfg.get("presets", []),
    }


def apply_action(target: str, **kwargs) -> dict:
    """Like set_state, but also accepts raw rgb=(r, g, b) tuples (used by the
    HUD color wheel). Returns same shape as set_state."""
    rgb = kwargs.pop("rgb", None)
    if rgb is None:
        return set_state(target, **kwargs)

    # Raw RGB path — bypass color-name lookup, send setRgb directly.
    targets = _resolve_targets(target)
    if not targets:
        return {"ok": False, "error": f"no bulb matching {target!r}"}
    try:
        r, g, b = (int(x) for x in rgb)
        r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
    except (TypeError, ValueError):
        return {"ok": False, "error": f"invalid rgb {rgb!r}"}

    on        = kwargs.get("on")
    pct       = kwargs.get("brightness")
    if pct is not None:
        r, g, b = _scale_rgb((r, g, b), pct)

    results = []
    for bulb_cfg in targets:
        name = bulb_cfg.get("name") or bulb_cfg.get("mac")
        bulb = _connect(bulb_cfg["ip"])
        if bulb is None:
            results.append({"bulb_name": name, "ok": False, "error": "offline"})
            continue
        try:
            if on is False:
                bulb.turnOff()
                results.append({"bulb_name": name, "ok": True, "action": "off"})
                continue
            if not bulb.is_on:
                bulb.turnOn()
            try:
                bulb.setRgb(r, g, b, persist=True)
            except TypeError:
                bulb.setRgb(r, g, b)
            results.append({"bulb_name": name, "ok": True, "action": f"rgb({r},{g},{b})"})
        except Exception as e:
            _bulb_cache.pop(bulb_cfg["ip"], None)
            results.append({"bulb_name": name, "ok": False, "error": str(e)})
    overall = all(r.get("ok") for r in results) if results else False
    return {"ok": overall, "target": target, "results": results,
            "intent": {"on": on, "brightness": pct, "rgb": [r, g, b]}}


def rename_bulb(mac: str, new_name: str) -> bool:
    """Public wrapper around name_bulb for the HUD rename flow."""
    return name_bulb(mac, new_name)


# ─── presets ────────────────────────────────────────────────────────────────

def _find_preset(name: str) -> Optional[dict]:
    cfg = _load_config()
    n = name.strip().lower()
    for p in cfg.get("presets", []):
        if p.get("name", "").lower() == n:
            return p
    return None


def list_presets() -> list[dict]:
    ensure_presets()
    return _load_config().get("presets", [])


def apply_preset(name: str) -> dict:
    """Apply a preset to all matching bulbs. Returns {ok, applied: [...]}."""
    p = _find_preset(name)
    if p is None:
        return {"ok": False, "error": f"no preset named {name!r}"}

    if "all" in p:
        # uniform: apply same state to every named bulb
        result = apply_action("all", **p["all"])
        return {"ok": result.get("ok", False), "preset": name,
                "applied_to": "all", "result": result}

    # per-bulb state
    bulb_states = p.get("bulb_states", {})
    results = []
    for bulb_name, state in bulb_states.items():
        r = apply_action(bulb_name, **state)
        results.append({"bulb": bulb_name, "ok": r.get("ok", False), "result": r})
    overall = all(r["ok"] for r in results) if results else False
    return {"ok": overall, "preset": name, "results": results}


def save_preset(name: str, capture_current: bool = True,
                bulb_states: Optional[dict] = None) -> dict:
    """Save (or update) a named preset. Defaults to capturing the current
    on/brightness/rgb state of every named bulb."""
    name = name.strip().lower()
    if not name:
        return {"ok": False, "error": "empty name"}

    if bulb_states is None and capture_current:
        bulb_states = {}
        for b in list_bulbs():
            bname = b.get("name")
            if not bname or not b.get("online"):
                continue
            bulb_states[bname] = {
                "on":         bool(b.get("on")),
                "brightness": int(round((b.get("brightness", 0) or 0) * 100 / 255)),
            }

    cfg = _load_config()
    presets = cfg.get("presets", [])
    # replace if name exists, else append
    presets = [p for p in presets if p.get("name", "").lower() != name]
    presets.append({"name": name, "bulb_states": bulb_states or {}})
    cfg["presets"] = presets
    _save_config(cfg)
    return {"ok": True, "preset": name, "bulb_states": bulb_states}


def delete_preset(name: str) -> dict:
    name = name.strip().lower()
    cfg = _load_config()
    before = len(cfg.get("presets", []))
    cfg["presets"] = [p for p in cfg.get("presets", []) if p.get("name", "").lower() != name]
    after = len(cfg["presets"])
    if before == after:
        return {"ok": False, "error": f"no preset named {name!r}"}
    _save_config(cfg)
    return {"ok": True, "preset": name, "deleted": True}


# --------------------------------------------------------------------------- #
# Chloe dispatch — slash command + natural language                           #
# --------------------------------------------------------------------------- #

def _format_result(result: dict, intent: dict) -> str:
    """Voice-friendly one-line summary of a set_state result."""
    if not result.get("ok"):
        err = result.get("error")
        if err:
            return f"Lights: {err}."
        offline = [r["bulb_name"] for r in result.get("results", []) if not r.get("ok")]
        if offline:
            return f"Couldn't reach: {', '.join(offline)}."
        return "Lights command failed."

    # Use the resolved bulb names rather than the raw target text — feels more
    # natural ("Bedroom off" instead of "Bed off" when user said "turn off bed").
    names = [r["bulb_name"] for r in result.get("results", []) if r.get("ok")]
    if len(names) == 1:
        label = names[0].title()
    else:
        label = "Lights"

    if intent.get("on") is False:
        return f"{label} off."

    parts: list[str] = []
    if intent.get("color"):
        parts.append(intent["color"])
    elif intent.get("ct"):
        parts.append(str(intent["ct"]))
    if intent.get("brightness") is not None:
        parts.append(f"at {intent['brightness']} percent")

    if not parts:
        return f"{label} on." if intent.get("on") is True else f"{label} set."
    return f"{label} {' '.join(parts)}."


def _format_status() -> str:
    bulbs = list_bulbs()
    if not bulbs:
        return "No bulbs configured. Run python lights.py --discover."
    lines = []
    for b in bulbs:
        if not b.get("online"):
            lines.append(f"{b.get('name') or b['mac']}: offline")
        else:
            on = "on" if b.get("on") else "off"
            lines.append(f"{b.get('name') or b['mac']}: {on}")
    return " | ".join(lines)


def try_handle_lights_command(text: str):
    """Returns a string reply if `text` is a lights command, else None.
    Mirrors try_handle_brain_command's contract for jarvis.py dispatch."""
    if not text:
        return None
    cleaned = text.strip()
    forced = False
    lower = cleaned.lower()

    # /lights or /lights status -> show state
    if lower in ("/lights", "/lights status", "/lights list"):
        return _format_status()
    if lower.startswith("/lights "):
        cleaned = cleaned[8:].strip()
        forced = True

    intent = parse_intent(cleaned)
    if not intent:
        return None

    # Safety gate: without /lights prefix or an explicit light/lamp/bulb word,
    # only claim the message if the target matches a configured bulb name.
    # Prevents us from grabbing "turn off the alarm" or "turn the conversation around".
    has_light_word = any(re.search(rf"\b{w}\b", cleaned.lower()) for w in LIGHT_WORDS)
    if not forced and not has_light_word:
        if not _resolve_targets(intent["target"]):
            return None

    result = set_state(**intent)
    return _format_result(result, intent)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _cli_discover():
    print("scanning LAN for Zengge bulbs (UDP broadcast, ~5s)...")
    found = discover()
    if not found:
        print("no bulbs found.\n"
              "checks:\n"
              "  - bulbs powered on?\n"
              "  - PC and bulbs on the same WiFi (same SSID, same subnet)?\n"
              "  - tried again? UDP discovery is flaky.")
        return 1
    print(f"found {len(found)} bulb(s):")
    for b in found:
        print(f"  mac={b['mac']}  ip={b['ip']}  model={b['model']}")
    print(f"\nsaved to {CONFIG_PATH}")
    print("next: python lights.py --name-interactive")
    return 0


def _cli_name_interactive():
    cfg = _load_config()
    bulbs = cfg.get("bulbs", [])
    if not bulbs:
        print("no bulbs in config. run: python lights.py --discover")
        return 1
    print(f"naming {len(bulbs)} bulb(s). I'll flash each one red so you can identify it.")
    print("type a name (e.g. bedroom, office, lamp) — or blank to skip, q to quit.\n")
    for b in bulbs:
        print(f"--- mac={b['mac']}  ip={b['ip']}  current name={b.get('name') or '(none)'}")
        bulb = _connect(b["ip"])
        if bulb is None:
            print("  OFFLINE — skipping")
            continue
        # save current state
        was_on = bool(bulb.is_on)
        # blink red 3x
        try:
            for _ in range(3):
                bulb.turnOn()
                bulb.setRgb(255, 0, 0)
                time.sleep(0.4)
                bulb.turnOff()
                time.sleep(0.3)
            if was_on:
                bulb.turnOn()
                bulb.setRgb(255, 230, 200)  # warmish white
        except Exception as e:
            print(f"  blink failed: {e}")
        try:
            ans = input("  name > ").strip()
        except EOFError:
            ans = ""
        if ans.lower() == "q":
            break
        if ans:
            b["name"] = ans.lower()
            print(f"  saved as {ans!r}")
    _save_config(cfg)
    print("\ndone.")
    return 0


def _cli_list():
    cfg = _load_config()
    bulbs = cfg.get("bulbs", [])
    if not bulbs:
        print("no bulbs. run: python lights.py --discover")
        return 1
    print(f"config: {CONFIG_PATH}")
    print(f"{'NAME':<16} {'MAC':<14} {'IP':<16} {'STATE'}")
    for b in list_bulbs():
        name = b.get("name") or "(unnamed)"
        if not b["online"]:
            state = "offline"
        else:
            state = ("ON " if b.get("on") else "off") + f"  bri={b.get('brightness', 0)}"
        print(f"{name:<16} {b['mac']:<14} {b['ip']:<16} {state}")
    return 0


def _cli_test():
    cfg = _load_config()
    bulbs = [b for b in cfg.get("bulbs", []) if b.get("name")]
    if not bulbs:
        print("no named bulbs. run: --discover then --name-interactive")
        return 1
    target_name = bulbs[0]["name"]
    print(f"cycling {target_name!r}: on white -> blue -> dim -> warm -> off")
    for label, kw in [
        ("on white", {"on": True, "ct": "neutral", "brightness": 80}),
        ("blue",     {"color": "blue"}),
        ("dim red",  {"color": "red", "brightness": 20}),
        ("warm",     {"ct": "warm white", "brightness": 60}),
        ("off",      {"on": False}),
    ]:
        print(f"  -> {label}")
        r = set_state(target_name, **kw)
        if not r.get("ok"):
            print(f"    FAIL: {r}")
            return 1
        time.sleep(1.5)
    print("test complete.")
    return 0


def _cli_command(text: str):
    intent = parse_intent(text)
    if not intent:
        print(f"not a lights command: {text!r}")
        return 1
    print(f"intent: {intent}")
    r = set_state(**intent)
    print(json.dumps(r, indent=2, default=str))
    return 0 if r.get("ok") else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "--discover":          return _cli_discover()
    if cmd == "--name-interactive":  return _cli_name_interactive()
    if cmd == "--list":              return _cli_list()
    if cmd == "--test":              return _cli_test()
    return _cli_command(" ".join(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
