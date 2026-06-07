"""Coverage (recall) against the complete F1 oracle, for the demo and analysis.

Faithfulness = precision (are the stated claims supported?). It is gameable by
abstention. Because the oracle is complete, we can also enumerate the facts that
mattered for each decision and measure recall (how many were correctly stated).
Shared by app.py (live demo) and the offline experiments.
"""
from __future__ import annotations

from .claims import (
    PIT_LAP, COMPOUND_CHANGE, N_STOPS, STINT_COMPOUND, FINAL_POSITION,
    BATTLE, BATTLE_OUTCOME, GAIN, DEFENSE, WINNER,
)

# fact-tag -> claim type used to satisfy it
_FACT_CLAIM = {
    "n_stops": N_STOPS, "final_position": FINAL_POSITION,
    "compound_change": COMPOUND_CHANGE, "pit_lap": PIT_LAP,
    "battle": BATTLE, "battle_outcome": BATTLE_OUTCOME, "gain": GAIN,
    "defense": DEFENSE, "winner": WINNER,
}


def key_facts(inst: dict) -> list[tuple]:
    """The salient, checkable facts a good explanation of THIS decision should cover."""
    gt = inst["ground_truth"]
    t = inst["decision_type"]
    facts: list[tuple] = []
    if t == "stint_strategy":
        d = gt["driver"]
        facts.append(("n_stops", d))
        if gt.get("final_position") is not None:
            facts.append(("final_position", d))
        for p in gt.get("pit_stops", []):
            facts.append(("pit_lap", d, p["lap"]))
            facts.append(("compound_change", d))
    elif t in ("undercut", "overcut"):
        a, b = gt["attacker"], gt["defender"]
        facts += [("battle",), ("battle_outcome",), ("gain",)]
        if gt.get("attacker_pit_lap") is not None:
            facts.append(("pit_lap", a, gt["attacker_pit_lap"]))
        if gt.get("defender_pit_lap") is not None:
            facts.append(("pit_lap", b, gt["defender_pit_lap"]))
    elif t == "defense":
        facts.append(("defense", gt.get("defender")))
    elif t == "race_summary":
        facts.append(("winner",))
        for cl in (gt.get("classification") or [])[:3]:
            facts.append(("final_position", cl["driver"]))
    return facts


def _covered(fact: tuple, supported: list[dict]) -> bool:
    """Is this key fact satisfied by some supported claim? `supported` are claim rows
    ({type, fields, label, ...}) with label == 'supported'."""
    want = _FACT_CLAIM.get(fact[0])
    for c in supported:
        if c["type"] != want:
            continue
        f = c.get("fields", {})
        if fact[0] in ("battle", "battle_outcome", "gain", "winner"):
            return True
        if fact[0] == "pit_lap" and f.get("driver") == fact[1] and f.get("lap") == fact[2]:
            return True
        if fact[0] in ("n_stops", "final_position", "compound_change", "defense") \
                and (fact[1] is None or f.get("driver") == fact[1] or f.get("defender") == fact[1]):
            return True
    return False


def coverage(claim_rows: list[dict], inst: dict) -> dict:
    """Recall of the key facts. `claim_rows` = FaithfulnessResult.claims (verified)."""
    facts = key_facts(inst)
    supported = [c for c in claim_rows if c.get("label") == "supported"]
    n_cov = sum(_covered(f, supported) for f in facts)
    n = len(facts)
    return {"covered": n_cov, "total": n, "recall": (n_cov / n if n else 0.0),
            "missed": [f[0] for f in facts if not _covered(f, supported)]}
