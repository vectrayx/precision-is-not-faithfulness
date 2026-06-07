"""Run Gemini (GCP Vertex AI) as a frontier baseline and merge into frontier.json.

Uses the Vertex REST generateContent API with a gcloud access token (no ADC login
needed). Mirrors run_frontier's output structure so rescore_llm.py and score_small.py
pick Gemini up automatically.

    python experiments/run_gemini.py --model gemini-2.5-pro --langs en es pt \
        --instances data/structured/test_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.generate import SYSTEM_PROMPT, _user_prompt
from src.eval.faithfulness import score_text
from experiments.run_frontier import summarize, OUT, RESULTS_DIR


def _project():
    return subprocess.check_output(["gcloud", "config", "get-value", "project"],
                                   text=True).strip()


def _token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def gemini_generate(model, project, location, token, inst, lang):
    url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/{location}/publishers/google/models/{model}:generateContent")
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": _user_prompt(inst, lang)}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800},
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=120)
    r.raise_for_status()
    d = r.json()
    cands = d.get("candidates", [])
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [{}])
    return "".join(p.get("text", "") for p in parts).strip()


def run_one(model, project, location, token, inst, lang):
    try:
        text = gemini_generate(model, project, location, token, inst, lang)
    except Exception as e:
        return {"id": inst["id"], "error": str(e)[:200], "n_claims": 0,
                "supported": 0, "contradicted": 0, "unverifiable": 0,
                "faithfulness": 0.0, "hallucination_rate": 0.0}
    r = score_text(text, inst["ground_truth"])
    return {"id": inst["id"], "decision_type": inst["decision_type"], "text": text,
            "faithfulness": r.faithfulness, "hallucination_rate": r.hallucination_rate,
            "n_claims": r.n_claims, "supported": r.supported,
            "contradicted": r.contradicted, "unverifiable": r.unverifiable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "pt"])
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--instances", default=str(ROOT / "data/structured/test_sample.jsonl"))
    args = ap.parse_args()

    project, token = _project(), _token()
    instances = [json.loads(l) for l in Path(args.instances).read_text().splitlines() if l.strip()]

    out = json.loads(OUT.read_text()) if OUT.exists() else \
        {"models": [], "langs": args.langs, "n_instances": len(instances),
         "summaries": [], "runs": {}}
    out["models"] = sorted(set(out.get("models", [])) | {args.model})
    out["summaries"] = [s for s in out["summaries"] if s["model"] != args.model]

    for lang in args.langs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            per = list(ex.map(lambda i: run_one(args.model, project, args.location, token, i, lang),
                              instances))
        s = summarize(per, args.model, lang)
        out["summaries"].append(s)
        out["runs"][f"{args.model}|{lang}"] = per
        print(f"{args.model} {lang}  faith={s['macro_faithfulness']:.3f} "
              f"halluc={s['macro_hallucination']:.3f} claims={s['total_claims']} "
              f"errors={s['errors']}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Merged Gemini into {OUT}")


if __name__ == "__main__":
    main()
