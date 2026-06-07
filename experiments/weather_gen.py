"""Generate weather forecasts grounded in the structured record (second domain).

Generation only; scoring is separate (weather_coverage.py), mirroring the F1 split so
generation creds (Azure vs AIServices) never collide with the extractor creds. Saves
texts to experiments/results/weather_gen.json (per model|lang). Supports --merge.

    python experiments/weather_gen.py --models gpt-5.5=gpt-55 gpt-5.4-mini=gpt-54-mini \
        --langs en es pt --sample 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.generate import chat

INSTANCES = ROOT / "data" / "weather" / "instances.jsonl"
OUT = ROOT / "experiments" / "results" / "weather_gen.json"

SYSTEM = ("You are a meteorologist. Write a short, natural weather forecast using ONLY "
          "the provided record. Do not invent temperatures, wind, precipitation chances, "
          "or sky conditions not in the record. Write in {lang}.")
LANG_NAME = {"en": "English", "es": "Spanish", "pt": "Portuguese"}


def make_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        max_retries=6)


def _modern_chat(client, deployment, msgs, tries=4):
    """gpt-5.x-safe call: only the modern signature, with retries (no max_tokens
    fallback, which gpt-5.x rejects with a 400 and which corrupted the first pass)."""
    last = None
    for _ in range(tries):
        try:
            r = client.chat.completions.create(
                model=deployment, messages=msgs, max_completion_tokens=2000)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
    raise last


def run_one(client, deployment, inst, lang):
    msgs = [{"role": "system", "content": SYSTEM.format(lang=LANG_NAME[lang])},
            {"role": "user", "content": inst["context_text"]}]
    try:
        return {"id": inst["id"], "text": _modern_chat(client, deployment, msgs)}
    except Exception as e:
        return {"id": inst["id"], "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=deployment ...")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "pt"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=0, help="first N instances (0=all)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--fixerrors", action="store_true",
                    help="retry only errored instances in existing runs")
    args = ap.parse_args()

    insts = [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]
    if args.sample:
        insts = insts[:args.sample]
    models = dict(m.split("=", 1) for m in args.models)
    client = make_client()

    if args.merge and OUT.exists():
        out = json.loads(OUT.read_text())
        out["models"] = sorted(set(out.get("models", [])) | set(models))
    else:
        out = {"models": list(models), "langs": args.langs,
               "n_instances": len(insts), "runs": {}}
    by_id = {i["id"]: i for i in insts}
    for label, deployment in models.items():
        for lang in args.langs:
            key = f"{label}|{lang}"
            if key in out["runs"] and not args.fixerrors:
                print(f"skip {key} (already generated)", flush=True)
                continue
            if key in out["runs"] and args.fixerrors:  # retry only errored instances
                per = out["runs"][key]
                bad = [p for p in per if p.get("error")]
                if not bad:
                    print(f"{key}: no errors", flush=True)
                    continue
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    fixed = list(ex.map(lambda p: run_one(client, deployment, by_id[p["id"]], lang), bad))
                fixed_by_id = {f["id"]: f for f in fixed}
                out["runs"][key] = [fixed_by_id.get(p["id"], p) if p.get("error") else p for p in per]
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    out["runs"][key] = list(ex.map(lambda i: run_one(client, deployment, i, lang), insts))
            errs = sum(1 for p in out["runs"][key] if p.get("error"))
            print(f"{label:14s} {lang}  n={len(out['runs'][key])} errors={errs}", flush=True)
            OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
