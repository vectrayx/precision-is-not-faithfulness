"""Championship standings and summaries aggregated from the per-race classifications.

Drivers' and constructors' points are summed across a season's structured race files.
Used by the demo's championship tab and as grounded context for season summaries.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURED = _REPO_ROOT / "data" / "structured"


def _race_classifications(year: int):
    """Yield each race's classification for `year`.

    Prefer the per-race JSON files; fall back to the race_summary instances in
    instances.jsonl (which carry the points-scoring classification) so the demo works
    when only instances.jsonl is shipped (e.g. the Hugging Face Space)."""
    races = sorted(STRUCTURED.glob(f"{year}_*_R.json"))
    if races:
        for jf in races:
            yield json.loads(jf.read_text(encoding="utf-8")).get("classification", [])
        return
    insts = STRUCTURED / "instances.jsonl"
    if insts.exists():
        for line in insts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            i = json.loads(line)
            if i.get("decision_type") == "race_summary" and i.get("year") == year:
                yield i["ground_truth"].get("classification", [])


SEASON_POINTS = STRUCTURED / "season_points.json"


def _sprint_points(year: int):
    """Sprint-session points per driver/team for `year` (needs FastF1; build-time).
    Returns (drv_pts, team_pts) or ({}, {}) if no sprints / load fails."""
    import fastf1
    from .load_session import enable_cache
    enable_cache()
    drv, team = collections.defaultdict(float), collections.defaultdict(float)
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return drv, team
    for _, ev in sched.iterrows():
        fmt = str(ev.get("EventFormat", "")).lower()
        if "sprint" not in fmt:
            continue
        for code in ("S", "Sprint"):
            try:
                s = fastf1.get_session(year, str(ev["EventName"]), code)
                s.load(laps=False, telemetry=False, weather=False, messages=False)
                for _, r in s.results.iterrows():
                    d = str(r.get("Abbreviation", "")); pts = float(r["Points"]) if pd.notna(r.get("Points")) else 0.0
                    drv[d] += pts
                    if pd.notna(r.get("TeamName")):
                        team[str(r["TeamName"])] += pts
                break
            except Exception:
                continue
    return drv, team


def build_season_points(years) -> Path:
    """Combine race points (incl. fastest-lap, from race JSONs) + sprint points into
    season_points.json so the demo shows official championship totals."""
    import pandas as _pd  # noqa
    out = {}
    for year in years:
        drv = collections.defaultdict(float); team = collections.defaultdict(float)
        drv_team = {}; wins = collections.defaultdict(int); n = 0
        for cl in _race_classifications(year):
            n += 1
            for c in cl:
                d = c["driver"]; drv[d] += c.get("points") or 0.0
                if c.get("team"):
                    drv_team[d] = c["team"]; team[c["team"]] += c.get("points") or 0.0
                if c.get("position") == 1:
                    wins[d] += 1
        sp_d, sp_t = _sprint_points(year)
        for d, p in sp_d.items():
            drv[d] += p
        for t, p in sp_t.items():
            team[t] += p
        if not n:
            continue
        out[str(year)] = {
            "year": year, "races": n,
            "drivers": sorted(((d, round(p, 1), drv_team.get(d), wins.get(d, 0))
                               for d, p in drv.items()), key=lambda x: -x[1]),
            "constructors": sorted(((t, round(p, 1)) for t, p in team.items()), key=lambda x: -x[1]),
        }
    SEASON_POINTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return SEASON_POINTS


def season_standings(year: int) -> dict:
    """Return {drivers: [(driver, pts, team, wins)], constructors: [(team, pts)], races: n}."""
    if SEASON_POINTS.exists():
        cache = json.loads(SEASON_POINTS.read_text())
        if str(year) in cache:
            c = cache[str(year)]
            return {"year": year, "races": c["races"],
                    "drivers": [tuple(x) for x in c["drivers"]],
                    "constructors": [tuple(x) for x in c["constructors"]]}
    drv_pts: dict[str, float] = collections.defaultdict(float)
    drv_team: dict[str, str] = {}
    team_pts: dict[str, float] = collections.defaultdict(float)
    wins: dict[str, int] = collections.defaultdict(int)
    n_races = 0
    for classification in _race_classifications(year):
        n_races += 1
        for c in classification:
            drv, pts, team, pos = c["driver"], c.get("points") or 0.0, c.get("team"), c.get("position")
            drv_pts[drv] += pts
            if team:
                drv_team[drv] = team
                team_pts[team] += pts
            if pos == 1:
                wins[drv] += 1
    drivers = sorted(((d, round(p, 1), drv_team.get(d), wins.get(d, 0))
                      for d, p in drv_pts.items()), key=lambda x: -x[1])
    constructors = sorted(((t, round(p, 1)) for t, p in team_pts.items()), key=lambda x: -x[1])
    return {"year": year, "races": n_races, "drivers": drivers, "constructors": constructors}


def championship_summary_text(year: int, lang: str = "en") -> str:
    s = season_standings(year)
    if not s["drivers"]:
        return f"No data for {year}."
    d = s["drivers"]
    leader, lpts, lteam, lwins = d[0]
    gap = round(lpts - d[1][1], 1) if len(d) > 1 else lpts
    second = d[1][0] if len(d) > 1 else "-"
    ctop = s["constructors"][0][0] if s["constructors"] else "-"
    if lang == "es":
        return (f"Tras {s['races']} carreras de {year}, {leader} lidera el campeonato con "
                f"{lpts} puntos ({lwins} victorias), {gap} por delante de {second}. "
                f"En constructores lidera {ctop}.")
    if lang == "pt":
        return (f"Após {s['races']} corridas de {year}, {leader} lidera o campeonato com "
                f"{lpts} pontos ({lwins} vitórias), {gap} à frente de {second}. "
                f"Nos construtores, {ctop} lidera.")
    return (f"After {s['races']} races of {year}, {leader} leads the championship with "
            f"{lpts} points ({lwins} wins), {gap} ahead of {second}. "
            f"{ctop} leads the constructors'.")


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
    s = season_standings(yr)
    print(championship_summary_text(yr))
    print("Top 5 drivers:")
    for drv, pts, team, w in s["drivers"][:5]:
        print(f"  {drv:4s} {pts:6.1f}  {w}W  {team}")
