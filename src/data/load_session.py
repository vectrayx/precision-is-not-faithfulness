"""Load F1 sessions via FastF1, with a local cache under data/raw/.

FastF1 uses the official live-timing API + Ergast. No login or F1TV required.
Cached raw data is not versioned (data/raw is in .gitignore).
"""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1

# Cache under data/raw/cache (gitignored). Resolved relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "raw" / "cache"


def enable_cache() -> None:
    """Enable the FastF1 cache. Idempotent."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_race(year: int, gp: str, session: str = "R"):
    """Load a session and return the FastF1 Session object with data loaded.

    Args:
        year: season, e.g. 2024.
        gp: GP name or round, e.g. "Monza" or "Italian Grand Prix".
        session: 'R' (race), 'Q', 'S' (sprint), 'FP1'...

    Telemetry loads lazily; here we fetch laps + weather + race control messages.
    """
    enable_cache()
    ses = fastf1.get_session(year, gp, session)
    ses.load(laps=True, telemetry=False, weather=True, messages=True)
    logging.info("Loaded %s %s %s: %d laps", year, gp, session, len(ses.laps))
    return ses
