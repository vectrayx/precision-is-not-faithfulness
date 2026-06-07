"""Build the small-model comparison table (RQ3), all on the same subset and the same
LLM extractor for fairness.

Rows: gpt-5.5 and gpt-5.4-mini (frontier, from frontier_llm.json, restricted to the
subset), gpt-5.4-mini + self-correct (method.json final texts re-scored), and
Qwen2.5-3B base / fine-tuned (out_base.jsonl / out_ft.jsonl generated on the VM).

Writes experiments/results/small_models.json consumed by make_paper_assets.py.
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

RES = ROOT / "experiments" / "results"
INST = {json.loads(l)["id"]: json.loads(l)
        for l in (ROOT / "data/structured/instances.jsonl").read_text().splitlines() if l.strip()}
SUBSET = [json.loads(l)["id"] for l in (ROOT / "data/structured/test_sample.jsonl").read_text().splitlines() if l.strip()]
SUBSET_SET = set(SUBSET)


def _mean_faith(scored):
    """Return (macro_faithfulness, macro_hallucination, total_claims, claims_per_instance).

    Hallucination is the macro mean of per-instance hallucination_rate -- the SAME
    definition as frontier_llm.json's macro_hallucination -- so frontier rows are
    identical across the frontier and model-comparison tables. Coverage (claims per
    scored instance) is reported so faithfulness is read alongside informativeness.
    """
    s = [x for x in scored if x["n"] > 0]
    tot = sum(x["n"] for x in scored)
    return (round(statistics.mean(x["f"] for x in s), 4) if s else 0.0,
            round(statistics.mean(x["c"] for x in s), 4) if s else 0.0,
            tot,
            round(tot / len(s), 2) if s else 0.0)


def _score_texts(items):
    """items: list of (id, text). Returns scored dicts via LLM extractor."""
    def go(it):
        _id, text = it
        if not text:
            return {"f": 0.0, "c": 0.0, "n": 0}
        r = score_text(text, INST[_id]["ground_truth"], extractor=llm_extract)
        return {"f": r.faithfulness, "c": r.hallucination_rate, "n": r.n_claims}
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(go, items))


def frontier_subset(model_key):
    data = json.loads((RES / "frontier_llm.json").read_text())
    per = data["runs"][f"{model_key}|en"]
    sub = [p for p in per if p["id"] in SUBSET_SET]
    scored = [{"f": p["faithfulness"], "c": p["hallucination_rate"], "n": p["n_claims"]} for p in sub]
    return _mean_faith(scored)


def main():
    rows = []
    # every frontier model present in frontier_llm.json (LLM-extractor scored),
    # restricted to the test sample, in a stable order.
    fr = json.loads((RES / "frontier_llm.json").read_text())
    fmodels = sorted({s["model"] for s in fr["summaries"]})
    for key in fmodels:
        f, c, n, cov = frontier_subset(key)
        rows.append({"system": f"{key} (zero-shot)", "faithfulness": f,
                     "hallucination": c, "claims": n, "coverage": cov})

    # small open model: base and fine-tuned (subset)
    for label, fname in [("Qwen2.5-3B (zero-shot)", "out_base.jsonl"),
                         ("Qwen2.5-3B (fine-tuned)", "out_ft.jsonl")]:
        p = RES / fname
        if not p.exists():
            print(f"missing {fname}; skip")
            continue
        items = [(json.loads(l)["id"], json.loads(l)["text"])
                 for l in p.read_text().splitlines() if l.strip()]
        f, c, n, cov = _mean_faith(_score_texts(items))
        rows.append({"system": label, "faithfulness": f, "hallucination": c,
                     "claims": n, "coverage": cov})

    out = {"n_subset": len(SUBSET), "extractor": "llm", "rows": rows}
    (RES / "small_models.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for r in rows:
        print(f"{r['system']:32s} faith={r['faithfulness']:.3f} halluc={r['hallucination']:.3f} "
              f"claims/inst={r['coverage']}")


if __name__ == "__main__":
    main()
