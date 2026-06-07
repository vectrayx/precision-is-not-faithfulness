"""Precision/recall/F1 for weather forecasts against the complete record oracle.

Second-domain replication of the F1 finding: precision-only faithfulness rewards
abstention; with a complete oracle, requiring coverage (recall) changes the ranking.
Reads weather_gen.json (saved forecasts), extracts weather claims, verifies, scores.

    EXTRACTOR=llm EXTRACTOR_DEPLOYMENT=gpt-54-mini python experiments/weather_coverage.py
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

from src.eval.verify import SUPPORTED
from src.weather.wverify import verify_weather, W_TEMP, W_WIND, W_PRECIP, W_SKY

INSTANCES = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / "data/weather/instances.jsonl").read_text().splitlines()
             if l.strip()}
GEN = ROOT / "experiments" / "results" / "weather_gen.json"
OUT = ROOT / "experiments" / "results" / "weather_coverage.json"

_FACT_CLAIM = {"temp": W_TEMP, "sky": W_SKY, "wind": W_WIND, "precip_prob": W_PRECIP}


def get_extractor():
    if os.environ.get("EXTRACTOR", "llm") == "llm":
        from src.weather.wextract import weather_extract
        return weather_extract
    raise SystemExit("only the llm extractor is defined for weather")


def score_one(text, inst, extractor):
    facts = inst["key_facts"]
    if not text:
        return {"precision": 0.0, "recall": 0.0, "n_extracted": 0, "n_facts": len(facts)}
    gt = inst["ground_truth"]
    claims = extractor(text)
    supported_types = set()
    n_sup = 0
    for c in claims:
        label, _ = verify_weather(c, gt)
        if label == SUPPORTED:
            n_sup += 1
            supported_types.add(c.type)
    n_ext = len(claims)
    precision = n_sup / n_ext if n_ext else 0.0
    n_cov = sum(1 for f in facts if _FACT_CLAIM.get(f) in supported_types)
    recall = n_cov / len(facts) if facts else 0.0
    return {"precision": precision, "recall": recall, "n_extracted": n_ext}


def macro(vals):
    return round(statistics.mean(vals), 4) if vals else 0.0


def f1(p, r):
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def main():
    extractor = get_extractor()
    data = json.loads(GEN.read_text())
    models = sorted(data["models"])
    langs = data["langs"]
    rows = []
    done = set()
    if OUT.exists():
        rows = json.loads(OUT.read_text()).get("rows", [])
        done = {(r["model"], r["lang"]) for r in rows}
    for m in models:
        for lang in langs:
            key = f"{m}|{lang}"
            per = data["runs"].get(key)
            if not per or (m, lang) in done:
                continue
            with ThreadPoolExecutor(max_workers=8) as ex:
                scored = list(ex.map(
                    lambda p: score_one(p.get("text", ""), INSTANCES[p["id"]], extractor), per))
            prec = macro([s["precision"] for s in scored if s["n_extracted"] > 0])
            rec = macro([s["recall"] for s in scored])
            rows.append({"model": m, "lang": lang, "precision": prec, "recall": rec,
                         "f1": f1(prec, rec),
                         "claims_per_inst": round(sum(s["n_extracted"] for s in scored) / len(scored), 2)})
            print(f"{m:14s} {lang}  P={prec:.3f} R={rec:.3f} F1={f1(prec,rec):.3f} "
                  f"cl/inst={rows[-1]['claims_per_inst']}", flush=True)
            OUT.write_text(json.dumps({"extractor": os.environ.get('EXTRACTOR_DEPLOYMENT', 'gpt-54-mini'),
                                       "rows": rows}, indent=2, ensure_ascii=False))

    en = [r for r in rows if r["lang"] == "en"]
    by_p = [r["model"] for r in sorted(en, key=lambda r: -r["precision"])]
    by_f1 = [r["model"] for r in sorted(en, key=lambda r: -r["f1"])]
    md = ["# Weather: coverage vs faithfulness (second domain)\n",
          "| Model | Lang | Precision | Recall | F1 | cl/inst |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['lang']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                  f"{r['f1']:.3f} | {r['claims_per_inst']} |")
    md += [f"\nEN ranking by precision: {by_p}", f"EN ranking by F1:        {by_f1}",
           f"Ranking changes when coverage required: {by_p != by_f1}"]
    (ROOT / "experiments" / "results" / "weather_coverage_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md[-3:]))


if __name__ == "__main__":
    main()
