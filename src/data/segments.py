"""Turn structured race events into benchmark instances ("decision segments").

Each instance pairs:
  - context:        the structured facts the model is allowed to use (input),
  - context_text:   a deterministic serialization of those facts (model-facing),
  - prompt:         the decision question, in EN/ES/PT,
  - ground_truth:   structured facts the faithfulness metric verifies against.

The model must explain a strategic decision grounded ONLY in the provided context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_DIR = _REPO_ROOT / "data" / "structured"

LANGS = ("en", "es", "pt")


# Season-based split to avoid leakage: train on past seasons, test on the latest.
TEST_YEAR = 2025


def split_for_year(year: int) -> str:
    return "test" if year >= TEST_YEAR else "train"


@dataclass
class Instance:
    id: str
    year: int
    gp: str
    split: str                    # "train" | "test" (by season)
    decision_type: str            # "stint_strategy" | "undercut" | "overcut"
    focus_drivers: list[str]
    context: dict                 # structured facts shown to the model
    context_text: str             # serialized table (language-neutral numbers)
    prompts: dict                 # {lang: prompt}
    ground_truth: dict            # structured facts for verification


# ---- serialization -------------------------------------------------------

def _fmt_stint(s: dict) -> str:
    deg = "" if s["degradation_s_per_lap"] is None else f", deg {s['degradation_s_per_lap']:+.3f}s/lap"
    avg = "" if s["avg_laptime_s"] is None else f", avg {s['avg_laptime_s']:.2f}s"
    return (f"  Stint {s['stint']}: {s['compound']} laps {s['start_lap']}-{s['end_lap']} "
            f"({s['n_laps']} laps{avg}{deg})")


def _driver_stints(events: dict, drv: str) -> list[dict]:
    return [s for s in events["stints"] if s["driver"] == drv]


def _driver_pits(events: dict, drv: str) -> list[dict]:
    return [p for p in events["pit_stops"] if p["driver"] == drv]


def _final_pos(events: dict, drv: str) -> Optional[int]:
    for c in events["classification"]:
        if c["driver"] == drv:
            return c["position"]
    return None


def _stint_context_text(events: dict, drivers: list[str]) -> str:
    lines = [f"Race: {events['gp']} {events['year']} ({events.get('total_laps')} laps)"]
    for drv in drivers:
        pos = _final_pos(events, drv)
        lines.append(f"Driver {drv} (finished P{pos}):")
        for s in _driver_stints(events, drv):
            lines.append(_fmt_stint(s))
        for p in _driver_pits(events, drv):
            sc = " under SC/VSC" if p["under_neutralization"] else ""
            lines.append(f"  Pit lap {p['lap']}: {p['from_compound']}->{p['to_compound']}{sc}")
    return "\n".join(lines)


# ---- instance builders ---------------------------------------------------

def _stint_strategy_instances(events: dict) -> list[Instance]:
    out = []
    # one instance per driver who finished in the points (interesting strategies)
    pointers = [c["driver"] for c in events["classification"]
                if c["position"] is not None and c["position"] <= 10]
    for drv in pointers:
        stints = _driver_stints(events, drv)
        pits = _driver_pits(events, drv)
        if not stints:
            continue
        n_stops = len(pits)
        gt = {
            "driver": drv,
            "final_position": _final_pos(events, drv),
            "n_stops": n_stops,
            "stints": stints,
            "pit_stops": pits,
        }
        ctx_text = _stint_context_text(events, [drv])
        out.append(Instance(
            id=f"{events['year']}_{_slug(events['gp'])}::stint::{drv}",
            year=events["year"], gp=events["gp"], split=split_for_year(events["year"]),
            decision_type="stint_strategy", focus_drivers=[drv],
            context=gt, context_text=ctx_text,
            prompts={
                "en": f"Explain {drv}'s tyre strategy in this race and how it shaped the result. Ground every claim in the data provided.",
                "es": f"Explica la estrategia de neumáticos de {drv} en esta carrera y cómo influyó en el resultado. Fundamenta cada afirmación en los datos provistos.",
                "pt": f"Explique a estratégia de pneus de {drv} nesta corrida e como ela moldou o resultado. Fundamente cada afirmação nos dados fornecidos.",
            },
            ground_truth=gt,
        ))
    return out


def _battle_instances(events: dict) -> list[Instance]:
    out = []
    for b in events["pit_battles"]:
        a, d = b["attacker"], b["defender"]
        gt = {
            "kind": b["kind"],
            "attacker": a, "defender": d,
            "attacker_pit_lap": b["attacker_pit_lap"],
            "defender_pit_lap": b["defender_pit_lap"],
            "gap_before_s": b["gap_before_s"],
            "gap_after_s": b["gap_after_s"],
            "gained_s": b["gained_s"],
            "position_swap": b["position_swap"],
            "stints": _driver_stints(events, a) + _driver_stints(events, d),
            "pit_stops": _driver_pits(events, a) + _driver_pits(events, d),
        }
        ctx_text = _stint_context_text(events, [a, d])
        ctx_text += (f"\nPit-timing battle: {a} pitted lap {b['attacker_pit_lap']}, "
                     f"{d} pitted lap {b['defender_pit_lap']}; gap {a}->{d} went "
                     f"{b['gap_before_s']}s to {b['gap_after_s']}s.")
        verb = "undercut" if b["kind"] == "undercut" else "overcut"
        out.append(Instance(
            id=f"{events['year']}_{_slug(events['gp'])}::{b['kind']}::{a}_{d}",
            year=events["year"], gp=events["gp"], split=split_for_year(events["year"]),
            decision_type=b["kind"], focus_drivers=[a, d],
            context=gt, context_text=ctx_text,
            prompts={
                "en": f"Explain how {a} {verb} {d} via pit-stop timing, and whether it worked. Ground every claim in the data provided.",
                "es": f"Explica cómo {a} le hizo el {verb} a {d} mediante el timing de la parada, y si funcionó. Fundamenta cada afirmación en los datos provistos.",
                "pt": f"Explique como {a} fez o {verb} em {d} pelo timing do pit stop, e se funcionou. Fundamente cada afirmação nos dados fornecidos.",
            },
            ground_truth=gt,
        ))
    return out


def _defense_instances(events: dict) -> list[Instance]:
    out = []
    for d in events.get("defenses", []):
        def_, pur = d["defender"], d["pursuer"]
        gt = {"defender": def_, "pursuer": pur, "n_laps": d["n_laps"],
              "pace_delta_s": d["pace_delta_s"], "teammate_protected": d["teammate_protected"],
              "start_lap": d["start_lap"], "end_lap": d["end_lap"],
              "stints": _driver_stints(events, def_) + _driver_stints(events, pur),
              "pit_stops": _driver_pits(events, def_) + _driver_pits(events, pur)}
        ctx = (f"Race: {events['gp']} {events['year']}\n"
               f"Defensive hold: {def_} kept {pur} behind for {d['n_laps']} laps "
               f"(laps {d['start_lap']}-{d['end_lap']}); {pur} was faster by "
               f"{d['pace_delta_s']}s/lap on pace. teammate_protected={d['teammate_protected']}.")
        out.append(Instance(
            id=f"{events['year']}_{_slug(events['gp'])}::defense::{def_}_{pur}",
            year=events["year"], gp=events["gp"], split=split_for_year(events["year"]),
            decision_type="defense", focus_drivers=[def_, pur],
            context=gt, context_text=ctx,
            prompts={
                "en": f"Explain how {def_} defended against (held up) {pur} and its strategic effect. Ground every claim in the data.",
                "es": f"Explica cómo {def_} contuvo (hizo de tapón a) {pur} y su efecto estratégico. Fundamenta cada afirmación en los datos.",
                "pt": f"Explique como {def_} segurou (conteve) {pur} e seu efeito estratégico. Fundamente cada afirmação nos dados.",
            },
            ground_truth=gt))
    return out


def _race_summary_instances(events: dict) -> list[Instance]:
    cls = [c for c in events["classification"] if c.get("position")]
    if not cls:
        return []
    cls = sorted(cls, key=lambda c: c["position"])
    top = cls[:10]
    winner = cls[0]["driver"]
    fl = events.get("fastest_lap")
    battles = sorted(events.get("pit_battles", []), key=lambda b: -(b.get("gained_s") or 0))[:3]
    defs = sorted(events.get("defenses", []), key=lambda d: -d["n_laps"])[:2]
    gt = {"classification": top, "fastest_lap": fl, "winner": winner,
          "pit_battles": battles, "defenses": defs}
    lines = [f"Race: {events['gp']} {events['year']} ({events.get('total_laps')} laps)",
             "Result (top 5): " + ", ".join(f"P{c['position']} {c['driver']}" for c in top[:5])]
    if fl:
        lines.append(f"Fastest lap: {fl['driver']} ({fl['laptime_s']}s)")
    for b in battles:
        lines.append(f"Pit battle: {b['attacker']} {b['kind']} {b['defender']} (gained {b['gained_s']}s)")
    for d in defs:
        lines.append(f"Defense: {d['defender']} held {d['pursuer']} {d['n_laps']} laps")
    out = [Instance(
        id=f"{events['year']}_{_slug(events['gp'])}::race_summary::{winner}",
        year=events["year"], gp=events["gp"], split=split_for_year(events["year"]),
        decision_type="race_summary", focus_drivers=[winner],
        context=gt, context_text="\n".join(lines),
        prompts={
            "en": "Summarize this race: the result and the key strategic moments. Ground every claim in the data.",
            "es": "Resume esta carrera: el resultado y los momentos estratégicos clave. Fundamenta cada afirmación en los datos.",
            "pt": "Resuma esta corrida: o resultado e os momentos estratégicos-chave. Fundamente cada afirmação nos dados.",
        },
        ground_truth=gt)]
    return out


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def instances_from_events(events: dict) -> list[Instance]:
    return (_stint_strategy_instances(events) + _battle_instances(events)
            + _defense_instances(events) + _race_summary_instances(events))


def build_instances(out_path: Optional[Path] = None) -> Path:
    """Read all structured race JSONs and write a JSONL of benchmark instances."""
    out_path = out_path or (STRUCTURED_DIR / "instances.jsonl")
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for jf in sorted(STRUCTURED_DIR.glob("*_R.json")):
            events = json.loads(jf.read_text(encoding="utf-8"))
            for inst in instances_from_events(events):
                f.write(json.dumps(asdict(inst), ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} instances to {out_path}")
    return out_path


if __name__ == "__main__":
    build_instances()
