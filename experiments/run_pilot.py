"""Run the end-to-end pilot: generate explanations, score faithfulness, aggregate.

By default it runs the two offline template baselines (no API key / GPU needed),
which doubles as the metric-validation experiment (faithful vs. perturbed text).
Add real backends with --generators, e.g.:

    python experiments/run_pilot.py --generators azure_openai --lang en

Outputs:
    experiments/results/pilot_<lang>.json   (summary + per-instance)
    experiments/results/summary.md          (human-readable table)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.faithfulness import score_text
from src.models.generate import REGISTRY

DEFAULT_INSTANCES = ROOT / "data" / "structured" / "instances.jsonl"
RESULTS_DIR = ROOT / "experiments" / "results"


def load_instances(path=DEFAULT_INSTANCES) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def run_generator(name: str, instances: list[dict], lang: str) -> dict:
    gen = REGISTRY[name]
    per = []
    for inst in instances:
        text = gen(inst, lang)
        r = score_text(text, inst["ground_truth"])
        per.append({
            "id": inst["id"], "decision_type": inst["decision_type"],
            "text": text,
            "faithfulness": r.faithfulness, "hallucination_rate": r.hallucination_rate,
            "n_claims": r.n_claims, "supported": r.supported,
            "contradicted": r.contradicted, "unverifiable": r.unverifiable,
        })
    # macro = mean over instances; micro = pooled over claims
    scored = [p for p in per if p["n_claims"] > 0]
    tot = sum(p["n_claims"] for p in per)
    sup = sum(p["supported"] for p in per)
    con = sum(p["contradicted"] for p in per)
    summary = {
        "generator": name,
        "n_instances": len(per),
        "macro_faithfulness": round(statistics.mean(p["faithfulness"] for p in scored), 4) if scored else 0.0,
        "macro_hallucination": round(statistics.mean(p["hallucination_rate"] for p in scored), 4) if scored else 0.0,
        "micro_faithfulness": round(sup / tot, 4) if tot else 0.0,
        "micro_hallucination": round(con / tot, 4) if tot else 0.0,
        "total_claims": tot,
    }
    return {"summary": summary, "per_instance": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+",
                    default=["template_faithful", "template_noisy"])
    ap.add_argument("--lang", default="en", choices=["en", "es", "pt"])
    ap.add_argument("--instances", default=str(DEFAULT_INSTANCES))
    args = ap.parse_args()

    instances = load_instances(args.instances)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"lang": args.lang, "n_instances": len(instances), "runs": {}}
    for name in args.generators:
        res = run_generator(name, instances, args.lang)
        out["runs"][name] = res
        s = res["summary"]
        print(f"{name:18s} faith(macro)={s['macro_faithfulness']:.3f} "
              f"halluc={s['macro_hallucination']:.3f} claims={s['total_claims']}")

    (RESULTS_DIR / f"pilot_{args.lang}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # human-readable table
    lines = [f"# Pilot results (lang={args.lang}, N={len(instances)} instances)\n",
             "| Generator | Faithfulness (macro) | Hallucination | Total claims |",
             "|---|---|---|---|"]
    for name in args.generators:
        s = out["runs"][name]["summary"]
        lines.append(f"| {name} | {s['macro_faithfulness']:.3f} | "
                     f"{s['macro_hallucination']:.3f} | {s['total_claims']} |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR/('pilot_'+args.lang+'.json')} and summary.md")


if __name__ == "__main__":
    main()
