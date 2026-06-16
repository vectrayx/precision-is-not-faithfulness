"""Score Claude Bedrock generations (EN/ES/PT) with the coverage metric.

Builds a frontier-format JSON from the per-language .jsonl files, then scores
with the LLM extractor. Writes to coverage_claude_multilang.json.

Usage:
    EXTRACTOR=llm python experiments/score_claude_coverage.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.claims import regex_extract
from src.eval.verify import verify_claim, SUPPORTED
from src.eval.coverage import key_facts, coverage as _coverage

RESULTS = ROOT / "experiments" / "results"
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


def load_generations(lang: str) -> list[dict]:
    if lang == "en":
        p = RESULTS / "out_claude_bedrock.jsonl"
    else:
        p = RESULTS / f"out_claude_bedrock_{lang}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    extractor = get_extractor()
    out_path = RESULTS / "coverage_claude_multilang.json"

    langs = ["en", "es", "pt"]
    rows = []
    done = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        rows = prev.get("rows", [])
        done = {r["lang"] for r in rows}

    for lang in langs:
        if lang in done:
            print(f"[{lang}] already scored, skipping", flush=True)
            continue
        gens = load_generations(lang)
        if not gens:
            print(f"[{lang}] no generations found, skipping", flush=True)
            continue
        valid = [g for g in gens if not g.get("error") and g["id"] in INSTANCES]
        n_err = len(gens) - len(valid)
        print(f"[{lang}] scoring {len(valid)} instances ({n_err} errors)...", flush=True)

        scored = []
        for i, g in enumerate(valid):
            s = score_one(g.get("text", ""), INSTANCES[g["id"]], extractor)
            scored.append(s)
            if (i + 1) % 20 == 0:
                print(f"  [{lang}] {i+1}/{len(valid)}", flush=True)

        prec = macro([s["precision"] for s in scored if s["n_extracted"] > 0])
        rec = macro([s["recall"] for s in scored])
        row = {
            "model": "claude-sonnet-4-6", "lang": lang,
            "precision": prec, "recall": rec, "n_blocked": n_err,
            "f1": f1(prec, rec),
            "claims_per_inst": round(sum(s["n_extracted"] for s in scored) / len(scored), 2)
        }
        rows.append(row)
        print(f"[{lang}] P={prec:.3f} R={rec:.3f} F1={f1(prec,rec):.3f} "
              f"claims/inst={row['claims_per_inst']}", flush=True)

        out = {"extractor": os.environ.get("EXTRACTOR", "regex"), "rows": rows}
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\nDone. Results:", json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
