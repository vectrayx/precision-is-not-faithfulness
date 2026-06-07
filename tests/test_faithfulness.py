"""Unit tests for claim extraction + verification + faithfulness scoring.

Run: python -m pytest tests/  (or: python tests/test_faithfulness.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.claims import regex_extract, PIT_LAP, COMPOUND_CHANGE, N_STOPS, BATTLE
from src.eval.verify import verify_claim, SUPPORTED, CONTRADICTED, UNVERIFIABLE
from src.eval.faithfulness import score_text

# Ground truth: a stint-strategy instance for LEC at Monza 2024 (one-stop).
GT_STINT = {
    "driver": "LEC", "final_position": 1, "n_stops": 1,
    "stints": [
        {"driver": "LEC", "stint": 1, "compound": "MEDIUM", "start_lap": 1, "end_lap": 15},
        {"driver": "LEC", "stint": 2, "compound": "HARD", "start_lap": 16, "end_lap": 53},
    ],
    "pit_stops": [
        {"driver": "LEC", "lap": 15, "from_compound": "MEDIUM", "to_compound": "HARD"},
    ],
}

# Ground truth: a pit battle (NOR undercut LEC, gained 3.3s, swap True).
GT_BATTLE = {
    "kind": "undercut", "attacker": "NOR", "defender": "LEC",
    "attacker_pit_lap": 13, "defender_pit_lap": 15,
    "gap_before_s": -0.95, "gap_after_s": 2.32, "gained_s": 3.27, "position_swap": True,
    "stints": [], "pit_stops": [
        {"driver": "NOR", "lap": 13, "from_compound": "MEDIUM", "to_compound": "HARD"},
        {"driver": "LEC", "lap": 15, "from_compound": "MEDIUM", "to_compound": "HARD"},
    ],
}


def _labels(text, gt):
    return [verify_claim(c, gt) for c in regex_extract(text)]


def test_extract_basic_claims():
    cs = regex_extract("LEC pitted on lap 15 and switched from medium to hard.")
    types = {c.type for c in cs}
    assert PIT_LAP in types and COMPOUND_CHANGE in types


def test_supported_true_statements():
    text = ("LEC ran a one-stop strategy. He pitted on lap 15, switching from "
            "medium to hard, and finished P1.")
    r = score_text(text, GT_STINT)
    assert r.contradicted == 0, r.claims
    assert r.supported >= 3
    assert r.faithfulness == 1.0


def test_contradicted_false_statements():
    # wrong pit lap, wrong stop count, wrong compound change
    text = ("LEC pitted on lap 30. LEC made two pit stops, switching from soft to medium.")
    r = score_text(text, GT_STINT)
    assert r.contradicted >= 2, r.claims
    assert r.faithfulness < 0.5


def test_unverifiable_out_of_context_driver():
    # VER is not in this instance's context
    text = "VER pitted on lap 20."
    r = score_text(text, GT_STINT)
    assert r.unverifiable == 1 and r.supported == 0 and r.contradicted == 0


def test_battle_supported_and_contradicted():
    good = score_text("NOR undercut LEC and the undercut worked, gaining 3 seconds.", GT_BATTLE)
    assert good.contradicted == 0 and good.supported >= 2, good.claims
    # wrong direction: claim LEC undercut NOR
    bad = score_text("LEC undercut NOR.", GT_BATTLE)
    assert any(lbl == CONTRADICTED for lbl, _ in _labels("LEC undercut NOR.", GT_BATTLE)), bad.claims


def test_discrimination_gap():
    faithful = ("LEC ran a one-stop, pitted on lap 15, went from medium to hard, finished P1.")
    hallucinated = ("LEC ran a two-stop, pitted on lap 8 and lap 30, going from soft to medium, finished P4.")
    rf = score_text(faithful, GT_STINT)
    rh = score_text(hallucinated, GT_STINT)
    assert rf.faithfulness > rh.faithfulness + 0.5, (rf, rh)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
