"""Generate weather forecasts with Gemini (Vertex AI), merged into weather_gen.json."""
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
from experiments.weather_gen import SYSTEM, LANG_NAME, INSTANCES, OUT


def _project():
    return subprocess.check_output(["gcloud", "config", "get-value", "project"], text=True).strip()


def _token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def gen(model, project, location, token, inst, lang):
    url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/{location}/publishers/google/models/{model}:generateContent")
    payload = {"systemInstruction": {"parts": [{"text": SYSTEM.format(lang=LANG_NAME[lang])}]},
               "contents": [{"role": "user", "parts": [{"text": inst["context_text"]}]}],
               "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}}
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=120)
        r.raise_for_status()
        cands = r.json().get("candidates", [])
        if not cands:
            return {"id": inst["id"], "error": "no candidates"}
        parts = cands[0].get("content", {}).get("parts", [{}])
        return {"id": inst["id"], "text": "".join(p.get("text", "") for p in parts).strip()}
    except Exception as e:
        return {"id": inst["id"], "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "pt"])
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    insts = [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]
    if args.sample:
        insts = insts[:args.sample]
    project, token = _project(), _token()
    out = json.loads(OUT.read_text()) if OUT.exists() else \
        {"models": [], "langs": args.langs, "n_instances": len(insts), "runs": {}}
    out["models"] = sorted(set(out.get("models", [])) | {args.model})
    for lang in args.langs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            per = list(ex.map(lambda i: gen(args.model, project, args.location, token, i, lang), insts))
        out["runs"][f"{args.model}|{lang}"] = per
        print(f"{args.model} {lang}  n={len(per)} errors={sum(1 for p in per if p.get('error'))}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Merged Gemini into {OUT}")


if __name__ == "__main__":
    main()
