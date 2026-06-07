"""Run frontier baselines on Azure OpenAI across models x languages, concurrently.

Reads creds from the environment (see .env). Scores each generation with the
faithfulness metric and writes experiments/results/frontier.json + updates the
summary table. Designed to be run in the background.

    python experiments/run_frontier.py \
        --models gpt-5.5=gpt-55 gpt-5.4-mini=gpt-54-mini \
        --langs en es pt --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.generate import chat, SYSTEM_PROMPT, _user_prompt
from src.eval.faithfulness import score_text

DEFAULT_INSTANCES = ROOT / "data" / "structured" / "instances.jsonl"
RESULTS_DIR = ROOT / "experiments" / "results"
OUT = RESULTS_DIR / "frontier.json"


def load_instances(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        max_retries=6,
    )


SYS = os.environ.get("SYSTEM_PROMPT_OVERRIDE") or SYSTEM_PROMPT


def run_one(client, deployment, inst, lang):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": _user_prompt(inst, lang)}]
    try:
        text = chat(client, deployment, msgs)
    except Exception as e:
        return {"id": inst["id"], "error": str(e), "n_claims": 0,
                "supported": 0, "contradicted": 0, "unverifiable": 0,
                "faithfulness": 0.0, "hallucination_rate": 0.0}
    r = score_text(text, inst["ground_truth"])
    return {"id": inst["id"], "decision_type": inst["decision_type"], "text": text,
            "faithfulness": r.faithfulness, "hallucination_rate": r.hallucination_rate,
            "n_claims": r.n_claims, "supported": r.supported,
            "contradicted": r.contradicted, "unverifiable": r.unverifiable}


def summarize(per: list[dict], model: str, lang: str) -> dict:
    scored = [p for p in per if p["n_claims"] > 0]
    tot = sum(p["n_claims"] for p in per)
    return {
        "model": model, "lang": lang, "n_instances": len(per),
        "errors": sum(1 for p in per if p.get("error")),
        "macro_faithfulness": round(statistics.mean(p["faithfulness"] for p in scored), 4) if scored else 0.0,
        "macro_hallucination": round(statistics.mean(p["hallucination_rate"] for p in scored), 4) if scored else 0.0,
        "micro_faithfulness": round(sum(p["supported"] for p in per) / tot, 4) if tot else 0.0,
        "micro_hallucination": round(sum(p["contradicted"] for p in per) / tot, 4) if tot else 0.0,
        "total_claims": tot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=deployment ...")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "pt"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--instances", default=str(DEFAULT_INSTANCES))
    ap.add_argument("--merge", action="store_true",
                    help="merge into existing output instead of overwriting "
                         "(for adding a model served from a different resource)")
    ap.add_argument("--out", default=str(OUT), help="output json path")
    args = ap.parse_args()

    out_path = Path(args.out)
    models = dict(m.split("=", 1) for m in args.models)
    instances = load_instances(Path(args.instances))
    client = make_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge and out_path.exists():
        out = json.loads(out_path.read_text())
        out["models"] = sorted(set(out.get("models", [])) | set(models))
        # drop any prior summaries for the models we are about to (re)run
        out["summaries"] = [s for s in out["summaries"] if s["model"] not in models]
    else:
        out = {"models": list(models), "langs": args.langs,
               "n_instances": len(instances), "summaries": [], "runs": {}}
    for label, deployment in models.items():
        for lang in args.langs:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                per = list(ex.map(lambda i: run_one(client, deployment, i, lang), instances))
            s = summarize(per, label, lang)
            out["summaries"].append(s)
            out["runs"][f"{label}|{lang}"] = per
            print(f"{label:14s} {lang}  faith={s['macro_faithfulness']:.3f} "
                  f"halluc={s['macro_hallucination']:.3f} claims={s['total_claims']} "
                  f"errors={s['errors']}", flush=True)
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))  # checkpoint

    # markdown table
    lines = [f"# Frontier results (N={len(instances)} instances)\n",
             "| Model | Lang | Faithfulness | Hallucination | Claims | Errors |",
             "|---|---|---|---|---|---|"]
    for s in out["summaries"]:
        lines.append(f"| {s['model']} | {s['lang']} | {s['macro_faithfulness']:.3f} | "
                     f"{s['macro_hallucination']:.3f} | {s['total_claims']} | {s['errors']} |")
    (RESULTS_DIR / "frontier_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT} and frontier_summary.md")


if __name__ == "__main__":
    main()
