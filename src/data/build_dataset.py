"""CLI: build the structured strategic-event ground truth for one or more races.

Usage:
    python -m src.data.build_dataset --year 2024 --gp Monza
    python -m src.data.build_dataset --year 2024 --gp "Italian Grand Prix" --session R

Writes data/structured/<year>_<gp>_<session>.json with the derived events.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .load_session import load_race
from .events import build_race_events

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = _REPO_ROOT / "data" / "structured"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def build(year: int, gp: str, session: str = "R") -> Path:
    ses = load_race(year, gp, session)
    events = build_race_events(ses)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{year}_{_slug(gp)}_{session}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(events), f, indent=2, ensure_ascii=False)
    logging.info(
        "Wrote %s: %d stints, %d pit stops, %d pit battles",
        out_path, len(events.stints), len(events.pit_stops), len(events.pit_battles),
    )
    return out_path


def build_season(year: int, session: str = "R") -> list[Path]:
    """Build every completed round of a season. Skips rounds that fail to load."""
    import fastf1
    from .load_session import enable_cache
    enable_cache()
    sched = fastf1.get_event_schedule(year, include_testing=False)
    paths = []
    for _, ev in sched.iterrows():
        gp = str(ev["EventName"])
        try:
            paths.append(build(year, gp, session))
        except Exception as e:  # round not run yet, or data gap
            logging.warning("Skipping %s %s: %s", year, gp, e)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build structured F1 strategic events.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", type=str, help="single GP; omit with --season")
    ap.add_argument("--season", action="store_true", help="build the whole season")
    ap.add_argument("--session", type=str, default="R")
    args = ap.parse_args()
    if args.season:
        paths = build_season(args.year, args.session)
        print(f"Wrote {len(paths)} races for {args.year}")
    else:
        if not args.gp:
            ap.error("--gp is required unless --season is set")
        print(f"Wrote {build(args.year, args.gp, args.session)}")


if __name__ == "__main__":
    main()
