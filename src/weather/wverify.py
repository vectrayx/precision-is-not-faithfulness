"""Verify weather claims against the structured forecast oracle.

Same supported/contradicted/unverifiable semantics as the F1 verifier
(src/eval/verify.py): supported if the record confirms the claim (within tolerance),
contradicted if the record has the field and it differs, unverifiable if absent.
"""
from __future__ import annotations

from src.eval.claims import Claim
from src.eval.verify import SUPPORTED, CONTRADICTED, UNVERIFIABLE

# weather claim types
W_TEMP, W_WIND, W_WIND_DIR, W_PRECIP, W_SKY = "temp", "wind", "wind_dir", "precip_prob", "sky"

TEMP_TOL, WIND_TOL, PRECIP_TOL = 3.0, 5.0, 15.0

_DIRS = {  # normalize spelled-out / localized directions to compass codes
    "north": "N", "south": "S", "east": "E", "west": "W",
    "northeast": "NE", "northwest": "NW", "southeast": "SE", "southwest": "SW",
    "norte": "N", "sur": "S", "este": "E", "oeste": "W", "leste": "E",
    "nordeste": "NE", "noroeste": "NW", "sudeste": "SE", "sudoeste": "SW",
    "noreste": "NE",
}
# sky categories that count as matching each other (coarse equivalence classes)
_SKY_ALIASES = {
    "sunny": "clear", "mostly_clear": "clear", "fair": "clear",
    "overcast": "cloudy", "mostly_cloudy": "cloudy",
    "showers": "rain", "drizzle": "rain", "tstorms": "thunderstorms",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_sky(s):
    s = (s or "").strip().lower().replace(" ", "_")
    # map common phrasings
    for key in ("thunder", "storm"):
        if key in s:
            return "thunderstorms"
    if "snow" in s:
        return "snow"
    if "rain" in s or "shower" in s or "drizzle" in s:
        return "rain"
    if "fog" in s:
        return "fog"
    if "partly" in s:
        return "partly_cloudy"
    if "cloud" in s or "overcast" in s:
        return "cloudy"
    if "sun" in s or "clear" in s or "fair" in s:
        return "clear"
    return _SKY_ALIASES.get(s, s)


def verify_weather(c: Claim, gt: dict) -> tuple[str, str]:
    f = c.fields
    if c.type == W_TEMP:
        if gt.get("temp") is None:
            return UNVERIFIABLE, "no temp in record"
        v = _num(f.get("value"))
        if v is None:
            return UNVERIFIABLE, "no temp value"
        if abs(v - gt["temp"]) <= TEMP_TOL:
            return SUPPORTED, f"temp {gt['temp']} (~{v})"
        return CONTRADICTED, f"temp {gt['temp']}, not {v}"

    if c.type == W_WIND:
        lo, hi = gt.get("wind_min"), gt.get("wind_max")
        if lo is None:
            return UNVERIFIABLE, "no wind in record"
        v = _num(f.get("value"))
        if v is None:
            return UNVERIFIABLE, "no wind value"
        if lo - WIND_TOL <= v <= hi + WIND_TOL:
            return SUPPORTED, f"wind {lo}-{hi} (~{v})"
        return CONTRADICTED, f"wind {lo}-{hi}, not {v}"

    if c.type == W_WIND_DIR:
        if not gt.get("wind_dir"):
            return UNVERIFIABLE, "no wind dir in record"
        d = str(f.get("dir", "")).strip()
        dn = _DIRS.get(d.lower(), d.upper())
        if dn == gt["wind_dir"]:
            return SUPPORTED, f"wind dir {gt['wind_dir']}"
        return CONTRADICTED, f"wind dir {gt['wind_dir']}, not {dn}"

    if c.type == W_PRECIP:
        if gt.get("precip_prob") is None:
            return UNVERIFIABLE, "no precip in record"
        v = _num(f.get("value"))
        if v is None:
            return UNVERIFIABLE, "no precip value"
        if abs(v - gt["precip_prob"]) <= PRECIP_TOL:
            return SUPPORTED, f"precip {gt['precip_prob']}% (~{v}%)"
        return CONTRADICTED, f"precip {gt['precip_prob']}%, not {v}%"

    if c.type == W_SKY:
        if not gt.get("sky"):
            return UNVERIFIABLE, "no sky in record"
        if _norm_sky(f.get("condition")) == gt["sky"] or _norm_sky(f.get("condition")) == _norm_sky(gt.get("sky_raw")):
            return SUPPORTED, f"sky {gt['sky']}"
        return CONTRADICTED, f"sky {gt['sky']}, not {_norm_sky(f.get('condition'))}"

    return UNVERIFIABLE, f"unknown weather claim {c.type}"
