"""Re-score saved frontier generations with the language-agnostic LLM extractor.

The frontier sweep saved generated texts per (model, lang). For a fair cross-lingual
comparison (RQ2) we re-extract claims with the LLM extractor (works in EN/ES/PT) and
re-verify against the structured ground truth. Writes frontier_llm.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.llm_extract import llm_extract
from src.eval.faithfulness import score_text

RESULTS = ROOT / "experiments" / "results"
IN = RESULTS / "frontier.json"
OUT = RESULTS / "frontier_llm.json"
INSTANCES = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / "data/structured/instances.jsonl").read_text().splitlines() if l.strip()}


def rescore_one(p: dict) -> dict:
    if p.get("error") or not p.get("text"):
        return {**p, "faithfulness": 0.0, "n_claims": 0, "supported": 0,
                "contradicted": 0, "unverifiable": 0}
    gt = INSTANCES[p["id"]]["ground_truth"]
    r = score_text(p["text"], gt, extractor=llm_extract)
    return {"id": p["id"], "decision_type": p.get("decision_type"),
            "faithfulness": r.faithfulness, "hallucination_rate": r.hallucination_rate,
            "n_claims": r.n_claims, "supported": r.supported,
            "contradicted": r.contradicted, "unverifiable": r.unverifiable}


def summarize(per, model, lang):
    scored = [p for p in per if p["n_claims"] > 0]
    tot = sum(p["n_claims"] for p in per)
    return {"model": model, "lang": lang, "n_instances": len(per),
            "macro_faithfulness": round(statistics.mean(p["faithfulness"] for p in scored), 4) if scored else 0.0,
            "macro_hallucination": round(statistics.mean(p["hallucination_rate"] for p in scored), 4) if scored else 0.0,
            "micro_faithfulness": round(sum(p["supported"] for p in per) / tot, 4) if tot else 0.0,
            "total_claims": tot}


def main():
    data = json.loads(IN.read_text())
    out = {"models": data["models"], "langs": data["langs"],
           "n_instances": data["n_instances"], "summaries": [], "runs": {}}
    for key, per in data["runs"].items():
        model, lang = key.split("|")
        with ThreadPoolExecutor(max_workers=8) as ex:
            rescored = list(ex.map(rescore_one, per))
        s = summarize(rescored, model, lang)
        out["summaries"].append(s)
        out["runs"][key] = rescored
        print(f"{model:14s} {lang}  faith={s['macro_faithfulness']:.3f} "
              f"halluc={s['macro_hallucination']:.3f} claims={s['total_claims']}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["# Frontier results (LLM extractor, RQ1/RQ2)\n",
             "| Model | Lang | Faithfulness | Hallucination | Claims |",
             "|---|---|---|---|---|"]
    for s in out["summaries"]:
        lines.append(f"| {s['model']} | {s['lang']} | {s['macro_faithfulness']:.3f} | "
                     f"{s['macro_hallucination']:.3f} | {s['total_claims']} |")
    (RESULTS / "frontier_llm_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
