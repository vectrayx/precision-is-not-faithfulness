"""Coverage (recall) against the complete structured oracle.

Reference-free faithfulness = precision (supported / extracted). It is gameable by
abstention: a model that says almost nothing scores high. Because our oracle is
DETERMINISTIC AND COMPLETE, we can also measure recall -- of the key facts that
mattered for each decision, how many did the model correctly state? -- which
open-domain faithfulness benchmarks structurally cannot. We report precision, recall,
and their harmonic mean (F1), and check whether requiring coverage changes the ranking.

Re-scores the saved generations in frontier.json. Extractor is configurable:
  default = regex (free, EN-first); set EXTRACTOR=llm (+ creds) for the LLM extractor.
Writes experiments/results/coverage[_suffix].json + summary.md.
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

from src.eval.claims import regex_extract
from src.eval.verify import verify_claim, SUPPORTED
from src.eval.coverage import key_facts, coverage as _coverage  # handles all 5 task types

RESULTS = ROOT / "experiments" / "results"
FRONTIER = RESULTS / os.environ.get("GEN_FILE", "frontier.json")
INSTANCES = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / "data/structured/instances.jsonl").read_text().splitlines()
             if l.strip()}


def get_extractor():
    if os.environ.get("EXTRACTOR") == "llm":
        from src.eval.llm_extract import llm_extract
        return llm_extract
    return regex_extract


def score_one(text: str, inst: dict, extractor) -> dict:
    facts = key_facts(inst)
    if not text:
        return {"precision": 0.0, "recall": 0.0, "n_extracted": 0, "n_facts": len(facts)}
    claims = extractor(text)
    rows = []
    for c in claims:
        label, _ = verify_claim(c, inst["ground_truth"])
        rows.append({"type": c.type, "fields": c.fields, "label": label})
    n_ext = len(claims)
    n_sup = sum(r["label"] == SUPPORTED for r in rows)
    precision = n_sup / n_ext if n_ext else 0.0
    cov = _coverage(rows, inst)
    return {"precision": precision, "recall": cov["recall"],
            "n_extracted": n_ext, "n_supported": n_sup,
            "n_covered": cov["covered"], "n_facts": cov["total"]}


def macro(vals):
    return round(statistics.mean(vals), 4) if vals else 0.0


def f1(p, r):
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def main():
    extractor = get_extractor()
    suffix = os.environ.get("SUFFIX") or ("_llm" if os.environ.get("EXTRACTOR") == "llm" else "")
    only_langs = os.environ.get("LANGS", "en,es,pt").split(",")
    data = json.loads(FRONTIER.read_text())
    models = sorted(data["models"])
    out_path = RESULTS / f"coverage{suffix}.json"
    rows = []
    done = set()
    if out_path.exists():  # resume after a crash
        prev = json.loads(out_path.read_text())
        rows = prev.get("rows", [])
        done = {(r["model"], r["lang"]) for r in rows}
    for m in models:
        for lang in only_langs:
            key = f"{m}|{lang}"
            per = data["runs"].get(key)
            if not per or (m, lang) in done:
                continue
            with ThreadPoolExecutor(max_workers=8) as ex:
                scored = list(ex.map(
                    lambda p: score_one(p.get("text", ""), INSTANCES[p["id"]], extractor), per))
            # precision over instances that produced >=1 claim (matches headline);
            # recall over all instances (abstention => 0 recall, the whole point).
            prec = macro([s["precision"] for s in scored if s["n_extracted"] > 0])
            rec = macro([s["recall"] for s in scored])
            rows.append({"model": m, "lang": lang, "precision": prec, "recall": rec,
                         "f1": f1(prec, rec),
                         "claims_per_inst": round(sum(s["n_extracted"] for s in scored) / len(scored), 2)})
            print(f"{m:16s} {lang}  P={prec:.3f} R={rec:.3f} F1={f1(prec,rec):.3f} "
                  f"claims/inst={rows[-1]['claims_per_inst']}", flush=True)
            out = {"extractor": "llm" if suffix else "regex", "rows": rows}
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))  # checkpoint

    out = {"extractor": "llm" if suffix else "regex", "rows": rows}
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # EN ranking by precision vs by F1 -- does requiring coverage change it?
    en = [r for r in rows if r["lang"] == "en"]
    by_p = [r["model"] for r in sorted(en, key=lambda r: -r["precision"])]
    by_f1 = [r["model"] for r in sorted(en, key=lambda r: -r["f1"])]
    md = [f"# Coverage vs faithfulness ({out['extractor']} extractor)\n",
          "| Model | Lang | Precision (faith) | Recall (coverage) | F1 | claims/inst |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['lang']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                  f"{r['f1']:.3f} | {r['claims_per_inst']} |")
    md += [f"\nEN ranking by precision: {by_p}", f"EN ranking by F1:        {by_f1}",
           f"Ranking changes when coverage is required: {by_p != by_f1}"]
    (RESULTS / f"coverage{suffix}_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md[-3:]))


if __name__ == "__main__":
    main()
