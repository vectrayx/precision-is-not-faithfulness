"""Derive strategic events from a session's laps.

These deterministically derived events are the structured GROUND TRUTH later used by
the faithfulness metric to verify generated explanations.

Any number appearing in a generated explanation must be verifiable against what this
module produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# FastF1 TrackStatus codes (concatenated as a string in the column).
TRACK_STATUS = {
    "1": "clear",
    "2": "yellow",
    "4": "safety_car",
    "5": "red_flag",
    "6": "vsc",
    "7": "vsc_ending",
}
_NEUTRALIZED = {"safety_car", "vsc", "vsc_ending", "red_flag"}


def _to_seconds(td) -> Optional[float]:
    """Timedelta -> float seconds, or None if NaT."""
    if pd.isna(td):
        return None
    return float(td.total_seconds())


@dataclass
class Stint:
    driver: str
    stint: int
    compound: Optional[str]
    start_lap: int
    end_lap: int
    n_laps: int
    tyre_life_start: Optional[float]
    avg_laptime_s: Optional[float]
    degradation_s_per_lap: Optional[float]  # pace slope within the stint


@dataclass
class PitStop:
    driver: str
    lap: int                      # pit-in lap
    from_compound: Optional[str]
    to_compound: Optional[str]
    stop_duration_s: Optional[float]  # pit-lane duration if available
    under_neutralization: bool        # stop made under SC/VSC/red flag


@dataclass
class PitBattle:
    """Undercut or overcut between two drivers via differing pit timing."""
    kind: str                 # "undercut" | "overcut"
    attacker: str
    defender: str
    attacker_pit_lap: int
    defender_pit_lap: int
    gap_before_s: Optional[float]  # gap (defender - attacker) before the first stop
    gap_after_s: Optional[float]   # gap after both have pitted
    gained_s: Optional[float]      # positive => attacker gained
    position_swap: bool            # attacker went from behind to ahead


@dataclass
class Defense:
    """A sustained on-track defensive hold: a faster pursuer kept behind a slower car
    for several laps (the 'rear gunner' / wingman move, e.g. Perez/Alonso 2021)."""
    defender: str
    pursuer: str
    start_lap: int
    end_lap: int
    n_laps: int
    pursuer_pace_s: Optional[float]
    defender_pace_s: Optional[float]
    pace_delta_s: Optional[float]      # how much faster the pursuer's pace was
    teammate_protected: bool           # defender's teammate finished ahead of the pursuer
    kind: str = "sustained"            # "sustained" (>= min_laps) or "brief" (short hold ending in an overtake)


@dataclass
class RaceEvents:
    year: int
    gp: str
    total_laps: Optional[int] = None
    fastest_lap: Optional[dict] = None
    stints: list[dict] = field(default_factory=list)
    pit_stops: list[dict] = field(default_factory=list)
    pit_battles: list[dict] = field(default_factory=list)
    defenses: list[dict] = field(default_factory=list)
    track_status_periods: list[dict] = field(default_factory=list)
    classification: list[dict] = field(default_factory=list)


def _green_laps(driver_laps: pd.DataFrame) -> pd.DataFrame:
    """Pace-representative laps: exclude in/out laps, SC/VSC/yellow, and inaccurate
    laps. Used to measure degradation and pace."""
    df = driver_laps.copy()
    mask = (
        df["PitInTime"].isna()
        & df["PitOutTime"].isna()
        & df["LapTime"].notna()
    )
    if "IsAccurate" in df.columns:
        mask &= df["IsAccurate"].fillna(False)
    if "TrackStatus" in df.columns:
        mask &= df["TrackStatus"].astype(str).fillna("") == "1"  # green only
    return df[mask]


def _degradation(green: pd.DataFrame) -> Optional[float]:
    """Slope (s/lap) of laptime vs tyre age on green laps."""
    if len(green) < 3 or "TyreLife" not in green.columns:
        return None
    x = green["TyreLife"].astype(float).to_numpy()
    y = green["LapTime"].dt.total_seconds().to_numpy()
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3:
        return None
    return round(float(np.polyfit(x[ok], y[ok], 1)[0]), 4)


def extract_stints(laps: pd.DataFrame) -> list[Stint]:
    out: list[Stint] = []
    for (drv, stint_id), grp in laps.groupby(["Driver", "Stint"], dropna=True):
        grp = grp.sort_values("LapNumber")
        green = _green_laps(grp)
        avg = green["LapTime"].dt.total_seconds().mean() if len(green) else np.nan
        out.append(
            Stint(
                driver=str(drv),
                stint=int(stint_id),
                compound=(grp["Compound"].dropna().iloc[0] if grp["Compound"].notna().any() else None),
                start_lap=int(grp["LapNumber"].min()),
                end_lap=int(grp["LapNumber"].max()),
                n_laps=int(len(grp)),
                tyre_life_start=(float(grp["TyreLife"].dropna().iloc[0]) if "TyreLife" in grp and grp["TyreLife"].notna().any() else None),
                avg_laptime_s=(round(float(avg), 3) if not np.isnan(avg) else None),
                degradation_s_per_lap=_degradation(green),
            )
        )
    return out


def _status_at(periods: list[dict], t: Optional[float]) -> str:
    """Active track status at session time t (step function from periods)."""
    if t is None or not periods:
        return "clear"
    cur = "clear"
    for p in periods:
        if p["time_s"] is not None and p["time_s"] <= t:
            cur = p["status"]
        else:
            break
    return cur


def extract_pit_stops(laps: pd.DataFrame, stints: list[Stint],
                      periods: list[dict]) -> list[PitStop]:
    """Detect pit stops as transitions between a driver's consecutive stints."""
    out: list[PitStop] = []
    by_driver: dict[str, list[Stint]] = {}
    for s in stints:
        by_driver.setdefault(s.driver, []).append(s)
    for drv, sl in by_driver.items():
        sl = sorted(sl, key=lambda s: s.stint)
        drv_laps = laps[laps["Driver"] == drv]
        for prev, nxt in zip(sl, sl[1:]):
            pit_lap = prev.end_lap
            dur = None
            in_row = drv_laps[(drv_laps["LapNumber"] == pit_lap) & drv_laps["PitInTime"].notna()]
            out_row = drv_laps[(drv_laps["LapNumber"] == pit_lap + 1) & drv_laps["PitOutTime"].notna()]
            t_in = _to_seconds(in_row["PitInTime"].iloc[0]) if len(in_row) else None
            if len(in_row) and len(out_row):
                t_out = _to_seconds(out_row["PitOutTime"].iloc[0])
                if t_in is not None and t_out is not None:
                    dur = round(t_out - t_in, 3)
            out.append(
                PitStop(
                    driver=drv,
                    lap=int(pit_lap),
                    from_compound=prev.compound,
                    to_compound=nxt.compound,
                    stop_duration_s=dur,
                    under_neutralization=_status_at(periods, t_in) in _NEUTRALIZED,
                )
            )
    return out


def _gap_at_lap(laps: pd.DataFrame, drv_a: str, drv_b: str, lap: int) -> Optional[float]:
    """Gap (b - a) in seconds at the end of `lap`, using per-lap session Time."""
    a = laps[(laps["Driver"] == drv_a) & (laps["LapNumber"] == lap)]
    b = laps[(laps["Driver"] == drv_b) & (laps["LapNumber"] == lap)]
    if not len(a) or not len(b):
        return None
    ta, tb = _to_seconds(a["Time"].iloc[0]), _to_seconds(b["Time"].iloc[0])
    if ta is None or tb is None:
        return None
    return round(tb - ta, 3)


def detect_pit_battles(laps: pd.DataFrame, pit_stops: list[PitStop],
                       window: int = 4, max_gap_s: float = 5.0,
                       min_gain_s: float = 0.5) -> list[PitBattle]:
    """Detect undercuts/overcuts between drivers racing each other.

    For each ordered pair (A pits first, B pits later within `window` laps), measure
    the gap before A's stop and after both stops. We keep only pairs that were
    genuinely wheel-to-wheel (|gap_before| <= max_gap_s, ~within pit-relevant range)
    and where the timing-driven swing exceeds `min_gain_s`. Tight `max_gap_s` keeps
    the oracle high-precision: at that proximity the pit sequence is the dominant
    cause of the gap swing. A gain for the earlier-stopping car is an undercut; for
    the later-stopping car, an overcut.
    """
    out: list[PitBattle] = []
    for p in pit_stops:
        for q in pit_stops:
            if p.driver == q.driver:
                continue
            if not (0 < q.lap - p.lap <= window):
                continue  # p pits strictly before q, within window
            gap_before = _gap_at_lap(laps, p.driver, q.driver, p.lap - 1)
            gap_after = _gap_at_lap(laps, p.driver, q.driver, q.lap + 1)
            if gap_before is None or gap_after is None:
                continue
            if abs(gap_before) > max_gap_s:
                continue  # not actually racing each other
            gained = round(gap_after - gap_before, 3)  # gap (b-a) growing => A gained
            if abs(gained) < min_gain_s:
                continue
            swap = (gap_before < 0) != (gap_after < 0)
            if gained > 0:   # earlier-stopping car (A=p) gained -> undercut by A
                out.append(PitBattle("undercut", p.driver, q.driver, p.lap, q.lap,
                                     gap_before, gap_after, gained, swap))
            else:            # later-stopping car (B=q) gained -> overcut by B
                out.append(PitBattle("overcut", q.driver, p.driver, q.lap, p.lap,
                                     -gap_before if gap_before is not None else None,
                                     -gap_after if gap_after is not None else None,
                                     -gained, swap))
    return out


def extract_track_status(session) -> list[dict]:
    """SC/VSC/flag periods from the session's track_status."""
    periods: list[dict] = []
    ts = getattr(session, "track_status", None)
    if ts is None or not len(ts):
        return periods
    ts = ts.sort_values("Time")
    for _, row in ts.iterrows():
        code = str(row["Status"])
        periods.append({
            "time_s": _to_seconds(row["Time"]),
            "status": TRACK_STATUS.get(code, code),
            "code": code,
        })
    return periods


def extract_fastest_lap(laps: pd.DataFrame) -> Optional[dict]:
    valid = laps[laps["LapTime"].notna()]
    if not len(valid):
        return None
    row = valid.loc[valid["LapTime"].idxmin()]
    return {
        "driver": str(row["Driver"]),
        "lap": int(row["LapNumber"]),
        "laptime_s": round(_to_seconds(row["LapTime"]), 3),
        "compound": (str(row["Compound"]) if pd.notna(row.get("Compound")) else None),
    }


def extract_classification(session) -> list[dict]:
    res = getattr(session, "results", None)
    if res is None or not len(res):
        return []
    out = []
    for _, r in res.iterrows():
        out.append({
            "driver": str(r.get("Abbreviation", "")),
            "team": (str(r["TeamName"]) if pd.notna(r.get("TeamName")) else None),
            "position": (int(r["Position"]) if pd.notna(r.get("Position")) else None),
            "grid": (int(r["GridPosition"]) if pd.notna(r.get("GridPosition")) else None),
            "status": str(r.get("Status", "")),
            "points": (float(r["Points"]) if pd.notna(r.get("Points")) else None),
        })
    return out


def detect_defenses(laps: pd.DataFrame, classification: list[dict],
                    min_laps: int = 5, max_gap_s: float = 1.5,
                    min_brief: int = 2) -> list[Defense]:
    """Detect on-track defensive holds: a faster pursuer kept within `max_gap_s` behind a
    slower car. Two kinds:
      - "sustained": held for >= `min_laps` consecutive laps (e.g. Alonso/Hamilton,
        Hungary 2021).
      - "brief": a short hold (`min_brief`..`min_laps`-1 laps) that ENDS IN AN OVERTAKE
        (the pursuer passes the defender) -- the fierce corner-by-corner battles the
        lap-count threshold misses (e.g. Perez holding Hamilton ~2 laps before being
        passed, Abu Dhabi 2021)."""
    drivers = [d for d in laps["Driver"].unique()]
    max_lap = int(laps["LapNumber"].max()) if len(laps) else 0
    team = {c["driver"]: c.get("team") for c in classification}
    pos = {c["driver"]: c.get("position") for c in classification}

    # median green-lap pace per driver
    pace = {}
    for d in drivers:
        g = _green_laps(laps[laps["Driver"] == d])
        pace[d] = float(g["LapTime"].dt.total_seconds().median()) if len(g) else None

    def time_at(d, lap):
        r = laps[(laps["Driver"] == d) & (laps["LapNumber"] == lap)]
        return _to_seconds(r["Time"].iloc[0]) if len(r) else None

    runs: dict[tuple, list[int]] = {}   # (pursuer, defender) -> [start_lap, len]
    sustained: list[Defense] = []
    brief: list[Defense] = []

    def _make(P, D, start, n):
        tm = any(team.get(x) == team.get(D) and x != D and pos.get(x) is not None
                 and pos.get(P) is not None and pos[x] < pos[P] for x in drivers)
        return Defense(
            defender=D, pursuer=P, start_lap=start, end_lap=start + n - 1, n_laps=n,
            pursuer_pace_s=round(pace[P], 3), defender_pace_s=round(pace[D], 3),
            pace_delta_s=round(pace[D] - pace[P], 3), teammate_protected=bool(tm))

    def close_run_end(key, swapped):
        start, n = runs.pop(key)
        P, D = key
        if not (pace.get(P) and pace.get(D) and pace[P] < pace[D]):
            return  # pursuer must be genuinely faster
        if n >= min_laps:
            sustained.append(_make(P, D, start, n))
        elif n >= min_brief and swapped:   # short hold ended by an overtake
            d = _make(P, D, start, n); d.kind = "brief"; brief.append(d)

    for lap in range(2, max_lap + 1):
        order = sorted(((d, time_at(d, lap)) for d in drivers), key=lambda z: (z[1] is None, z[1]))
        order = [(d, x) for d, x in order if x is not None]
        rank = {d: i for i, (d, _) in enumerate(order)}
        active = set()
        for i in range(1, len(order)):
            D, tD = order[i - 1]; P, tP = order[i]
            if 0 < tP - tD < max_gap_s:
                key = (P, D); active.add(key)
                runs[key] = [runs[key][0], runs[key][1] + 1] if key in runs else [lap, 1]
        for key in list(runs):
            if key not in active:
                P, D = key            # swap = pursuer now ahead of the defender this lap
                close_run_end(key, swapped=rank.get(P, 99) < rank.get(D, 99))
    for key in list(runs):
        close_run_end(key, swapped=False)

    # sustained: keep the longest hold per pursuer (the decisive one)
    sustained.sort(key=lambda d: -d.n_laps)
    seen, out = set(), []
    for d in sustained:
        if d.pursuer not in seen:
            seen.add(d.pursuer); out.append(d)
    # brief: add per (pursuer, defender) pair not already covered by a sustained hold
    have = {(d.pursuer, d.defender) for d in out}
    for d in sorted(brief, key=lambda d: -d.n_laps):
        if (d.pursuer, d.defender) not in have:
            have.add((d.pursuer, d.defender)); out.append(d)
    return out


def build_race_events(session) -> RaceEvents:
    """Orchestrate the full derivation for a loaded session."""
    laps = session.laps
    year = int(session.event.year) if hasattr(session, "event") else None
    gp = str(session.event["EventName"]) if hasattr(session, "event") else ""
    periods = extract_track_status(session)
    stints = extract_stints(laps)
    pit_stops = extract_pit_stops(laps, stints, periods)
    battles = detect_pit_battles(laps, pit_stops)
    classification = extract_classification(session)
    defenses = detect_defenses(laps, classification)
    return RaceEvents(
        year=year,
        gp=gp,
        total_laps=int(laps["LapNumber"].max()) if len(laps) else None,
        fastest_lap=extract_fastest_lap(laps),
        stints=[asdict(s) for s in stints],
        pit_stops=[asdict(p) for p in pit_stops],
        pit_battles=[asdict(b) for b in battles],
        defenses=[asdict(d) for d in defenses],
        track_status_periods=periods,
        classification=classification,
    )
