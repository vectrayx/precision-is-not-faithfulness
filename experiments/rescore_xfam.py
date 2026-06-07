"""Re-score saved frontier generations with a CROSS-FAMILY LLM extractor.

The headline scores use the gpt-5.x LLM extractor (frontier_llm.json). To show the
metric is not an artifact of that family, we re-extract the SAME generations with a
non-gpt-5 extractor (DeepSeek-V3.2 by default, set EXTRACTOR_DEPLOYMENT + AIServices
creds in env) and re-verify against the structured ground truth. Writes
experiments/results/frontier_xfam.json (same shape as frontier_llm.json).

Run (creds fetched by the caller, see the cross-family block in extractor_agreement
notes):
    AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... EXTRACTOR_DEPLOYMENT=deepseek-v32 \
        python experiments/rescore_xfam.py
"""
from __future__ import annotations

import json
import os
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
OUT = RESULTS / os.environ.get("XFAM_OUT", "frontier_xfam.json")
INSTANCES = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / "data/structured/instances.jsonl").read_text().splitlines()
             if l.strip()}


def rescore_one(p: dict) -> dict:
    if p.get("error") or not p.get("text"):
        return {"id": p["id"], "decision_type": p.get("decision_type"),
                "faithfulness": 0.0, "hallucination_rate": 0.0, "n_claims": 0,
                "supported": 0, "contradicted": 0, "unverifiable": 0}
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
    extractor = os.environ.get("EXTRACTOR_DEPLOYMENT", "deepseek-v32")
    # Resume: keep already-completed runs from a prior (possibly crashed) checkpoint.
    if OUT.exists():
        out = json.loads(OUT.read_text())
        out["extractor"] = extractor
    else:
        out = {"models": data["models"], "langs": data["langs"], "extractor": extractor,
               "n_instances": data["n_instances"], "summaries": [], "runs": {}}
    done = set(out["runs"])
    for key, per in data["runs"].items():
        if key in done:
            print(f"skip {key} (already scored)", flush=True)
            continue
        model, lang = key.split("|")
        with ThreadPoolExecutor(max_workers=6) as ex:
            rescored = list(ex.map(rescore_one, per))
        s = summarize(rescored, model, lang)
        out["summaries"].append(s)
        out["runs"][key] = rescored
        print(f"{model:14s} {lang}  faith={s['macro_faithfulness']:.3f} "
              f"halluc={s['macro_hallucination']:.3f} claims={s['total_claims']}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))  # checkpoint
    print(f"\nWrote {OUT} (extractor={extractor})")


if __name__ == "__main__":
    main()
