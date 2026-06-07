"""Collect real forecasts from the NOAA/NWS public API into structured instances.

Second domain (weather) for the precision/recall study. Each forecast PERIOD is an
instance whose structured record is a COMPLETE oracle: the enumerable set of facts a
good forecast should state (temperature, wind, precipitation chance, sky condition).
Public-domain data (no licensing restriction, unlike F1/FOM).

    python -m src.weather.collect            # writes data/weather/instances.jsonl
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "weather" / "instances.jsonl"
UA = {"User-Agent": "(f1-telemetry-paper weather study, research)"}

# A spread of US locations (NWS only covers the US); the task is to describe each
# record in language X, so US-only data is fine.
CITIES = {
    "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298), "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740), "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936), "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970), "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431), "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903), "Boston": (42.3601, -71.0589),
    "Miami": (25.7617, -80.1918), "Atlanta": (33.7490, -84.3880),
    "Minneapolis": (44.9778, -93.2650), "New Orleans": (29.9511, -90.0715),
    "Portland OR": (45.5152, -122.6784), "Las Vegas": (36.1699, -115.1398),
    "Detroit": (42.3314, -83.0458), "Salt Lake City": (40.7608, -111.8910),
    "Kansas City": (39.0997, -94.5786), "Nashville": (36.1627, -86.7816),
    "Buffalo": (42.8864, -78.8784), "Anchorage": (61.2181, -149.9003),
    "Honolulu": (21.3069, -157.8583), "Albuquerque": (35.0844, -106.6504),
    "Oklahoma City": (35.4676, -97.5164), "Fargo": (46.8772, -96.7898),
    "Billings": (45.7833, -108.5007), "El Paso": (31.7619, -106.4850),
    "Burlington VT": (44.4759, -73.2121), "Tampa": (27.9506, -82.4572),
    "Sacramento": (38.5816, -121.4944),
}


def _get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25))


def _wind(s: str) -> tuple[int | None, int | None]:
    nums = [int(x) for x in re.findall(r"\d+", s or "")]
    if not nums:
        return None, None
    return min(nums), max(nums)


def _sky(short: str) -> str:
    """Normalize NWS shortForecast into a coarse sky/precip category."""
    s = (short or "").lower()
    if any(w in s for w in ("thunder", "t-storm")):
        return "thunderstorms"
    if "snow" in s or "flurr" in s:
        return "snow"
    if "rain" in s or "showers" in s or "drizzle" in s:
        return "rain"
    if "fog" in s:
        return "fog"
    if "mostly sunny" in s or "mostly clear" in s:
        return "mostly_clear"
    if "partly" in s:
        return "partly_cloudy"
    if "mostly cloudy" in s or "overcast" in s:
        return "cloudy"
    if "sunny" in s or "clear" in s:
        return "clear"
    if "cloud" in s:
        return "cloudy"
    return "other"


def build_instance(city: str, p: dict) -> dict:
    wlo, whi = _wind(p.get("windSpeed", ""))
    precip = (p.get("probabilityOfPrecipitation") or {}).get("value")
    temp_kind = "high" if p.get("isDaytime") else "low"
    oracle = {
        "temp": p.get("temperature"), "temp_unit": p.get("temperatureUnit", "F"),
        "temp_kind": temp_kind,
        "wind_min": wlo, "wind_max": whi, "wind_dir": p.get("windDirection") or None,
        "precip_prob": precip, "sky": _sky(p.get("shortForecast")),
        "sky_raw": p.get("shortForecast"),
    }
    # The recall denominator: the facts a good forecast for this period should cover.
    facts = ["temp", "sky"]
    if wlo is not None:
        facts.append("wind")
    if precip is not None:
        facts.append("precip_prob")
    ctx = (f"Location: {city}. Period: {p.get('name')}.\n"
           f"Temperature {temp_kind}: {oracle['temp']} {oracle['temp_unit']}\n"
           f"Sky: {oracle['sky_raw']}\n"
           f"Wind: {p.get('windSpeed')} {oracle['wind_dir'] or ''}\n"
           f"Chance of precipitation: {precip if precip is not None else 0}%")
    return {"id": f"{city.replace(' ', '_')}::{p.get('number')}", "domain": "weather",
            "location": city, "period": p.get("name"),
            "ground_truth": oracle, "key_facts": facts, "context_text": ctx,
            "reference": p.get("detailedForecast")}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for city, (lat, lon) in CITIES.items():
        try:
            pt = _get(f"https://api.weather.gov/points/{lat},{lon}")
            fc = _get(pt["properties"]["forecast"])
            periods = fc["properties"]["periods"]
            for p in periods:
                if p.get("temperature") is None:
                    continue
                rows.append(build_instance(city, p))
            print(f"{city}: {len(periods)} periods", flush=True)
        except Exception as e:
            print(f"{city}: FAIL {repr(e)[:120]}", flush=True)
        time.sleep(0.6)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"\nWrote {len(rows)} instances to {OUT}")


if __name__ == "__main__":
    main()
