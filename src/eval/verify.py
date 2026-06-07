"""Verify typed atomic claims against an instance's structured ground truth.

Semantics: faithfulness is grounding in the PROVIDED context. A claim is
  - supported    if the context's structured facts confirm it,
  - contradicted if the context contains the relevant fact and it differs,
  - unverifiable if the context does not contain the relevant fact at all.

Unverifiable counts against faithfulness (the model asserted something the data it
was given does not support), but is reported separately from hard contradictions.
"""
from __future__ import annotations

from typing import Optional

from .claims import (
    Claim, PIT_LAP, COMPOUND_CHANGE, N_STOPS, STINT_COMPOUND,
    FINAL_POSITION, BATTLE, BATTLE_OUTCOME, GAIN, DEFENSE, WINNER,
)

SUPPORTED, CONTRADICTED, UNVERIFIABLE = "supported", "contradicted", "unverifiable"

GAIN_TOL_S = 1.5


def _num(v):
    """Coerce an extracted field to a number, or None if missing/malformed.
    Cross-family extractors occasionally emit a claim with a null/blank numeric
    field (e.g. a `gain` with no value); such claims are unverifiable, not errors."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _drivers_in(gt: dict) -> set[str]:
    ds = set()
    for s in gt.get("stints", []):
        ds.add(s["driver"])
    for p in gt.get("pit_stops", []):
        ds.add(p["driver"])
    for k in ("driver", "attacker", "defender"):
        if gt.get(k):
            ds.add(gt[k])
    return ds


def _driver_pits(gt: dict, drv: str) -> list[dict]:
    return [p for p in gt.get("pit_stops", []) if p["driver"] == drv]


def _driver_stints(gt: dict, drv: str) -> list[dict]:
    return [s for s in gt.get("stints", []) if s["driver"] == drv]


def _res(label: str, reason: str) -> tuple[str, str]:
    return label, reason


def verify_claim(c: Claim, gt: dict) -> tuple[str, str]:
    drivers = _drivers_in(gt)
    f = c.fields

    if c.type == PIT_LAP:
        drv = f["driver"]
        if drv not in drivers:
            return _res(UNVERIFIABLE, f"{drv} not in context")
        lap = _num(f.get("lap"))
        if lap is None:
            return _res(UNVERIFIABLE, "no pit lap value")
        laps = {p["lap"] for p in _driver_pits(gt, drv)}
        if lap in laps:
            return _res(SUPPORTED, f"{drv} pitted lap {int(lap)}")
        return _res(CONTRADICTED, f"{drv} pitted on {sorted(laps)}, not lap {int(lap)}")

    if c.type == COMPOUND_CHANGE:
        drv = f["driver"]
        if drv not in drivers:
            return _res(UNVERIFIABLE, f"{drv} not in context")
        for p in _driver_pits(gt, drv):
            if (f["from"] in (None, p["from_compound"])) and (f["to"] in (None, p["to_compound"])):
                return _res(SUPPORTED, f"{drv} {p['from_compound']}->{p['to_compound']}")
        return _res(CONTRADICTED, f"{drv} has no {f['from']}->{f['to']} change")

    if c.type == N_STOPS:
        drv = f["driver"]
        if drv not in drivers:
            return _res(UNVERIFIABLE, f"{drv} not in context")
        fn = _num(f.get("n"))
        if fn is None:
            return _res(UNVERIFIABLE, "no stop count value")
        n = gt["n_stops"] if (gt.get("driver") == drv and "n_stops" in gt) else len(_driver_pits(gt, drv))
        if n == fn:
            return _res(SUPPORTED, f"{drv} made {n} stops")
        return _res(CONTRADICTED, f"{drv} made {n} stops, not {int(fn)}")

    if c.type == STINT_COMPOUND:
        drv = f["driver"]
        if drv not in drivers:
            return _res(UNVERIFIABLE, f"{drv} not in context")
        comps = {s["compound"] for s in _driver_stints(gt, drv)}
        if f["compound"] in comps:
            return _res(SUPPORTED, f"{drv} used {f['compound']}")
        return _res(CONTRADICTED, f"{drv} used {comps}, not {f['compound']}")

    if c.type == FINAL_POSITION:
        drv = f["driver"]
        actual = None
        if gt.get("driver") == drv and gt.get("final_position") is not None:
            actual = gt["final_position"]
        else:  # race-summary style: look up the driver in the classification
            for cl in gt.get("classification", []):
                if cl["driver"] == drv:
                    actual = cl.get("position"); break
        if actual is None:
            return _res(UNVERIFIABLE, f"no final position for {drv} in context")
        pos = _num(f.get("position"))
        if pos is None:
            return _res(UNVERIFIABLE, "no position value")
        if actual == pos:
            return _res(SUPPORTED, f"{drv} finished P{int(pos)}")
        return _res(CONTRADICTED, f"{drv} finished P{actual}, not P{int(pos)}")

    if c.type == WINNER:
        drv = f["driver"]
        winner = None
        if gt.get("final_position") == 1:
            winner = gt.get("driver")
        for cl in gt.get("classification", []):
            if cl.get("position") == 1:
                winner = cl["driver"]; break
        if winner is None:
            return _res(UNVERIFIABLE, "no winner in context")
        return _res(SUPPORTED if winner == drv else CONTRADICTED,
                    f"winner was {winner}")

    if c.type == DEFENSE:
        defs = gt.get("defenses")
        if defs is None and gt.get("defender"):
            defs = [{"defender": gt["defender"], "pursuer": gt["pursuer"]}]
        if not defs:
            return _res(UNVERIFIABLE, "no defense in context")
        for dfn in defs:
            if dfn["defender"] == f["defender"] and dfn["pursuer"] == f["pursuer"]:
                return _res(SUPPORTED, f"{f['defender']} held {f['pursuer']}")
        # reversed?
        for dfn in defs:
            if dfn["defender"] == f["pursuer"] and dfn["pursuer"] == f["defender"]:
                return _res(CONTRADICTED, f"it was {dfn['defender']} holding {dfn['pursuer']}")
        return _res(UNVERIFIABLE, "different drivers than the context defense")

    if c.type == BATTLE:
        if "kind" not in gt or not gt.get("attacker"):
            return _res(UNVERIFIABLE, "no pit battle in context")
        same = {f["attacker"], f["defender"]} == {gt["attacker"], gt["defender"]}
        if same and f["kind"] == gt["kind"] and f["attacker"] == gt["attacker"]:
            return _res(SUPPORTED, f"{gt['attacker']} {gt['kind']} {gt['defender']}")
        if same:
            return _res(CONTRADICTED,
                        f"context: {gt['attacker']} {gt['kind']} {gt['defender']}")
        return _res(UNVERIFIABLE, "different drivers than the context battle")

    if c.type == BATTLE_OUTCOME:
        if "kind" not in gt:
            return _res(UNVERIFIABLE, "no battle outcome in context")
        actual = bool(gt.get("position_swap") or (gt.get("gained_s") or 0) > 0)
        if f["success"] == actual:
            return _res(SUPPORTED, f"move {'worked' if actual else 'did not work'}")
        return _res(CONTRADICTED, f"move actually {'worked' if actual else 'did not work'}")

    if c.type == GAIN:
        g = gt.get("gained_s")
        if g is None:
            return _res(UNVERIFIABLE, "no gain in context")
        val = _num(f.get("value"))
        if val is None:
            return _res(UNVERIFIABLE, "no gain value")
        if abs(g - val) <= GAIN_TOL_S:
            return _res(SUPPORTED, f"gained {g}s (~{val}s)")
        return _res(CONTRADICTED, f"gained {g}s, not {val}s")

    return _res(UNVERIFIABLE, f"unknown claim type {c.type}")
