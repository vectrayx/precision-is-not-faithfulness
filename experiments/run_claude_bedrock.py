"""Generate Claude Sonnet responses via AWS Bedrock Converse API for ES/PT (and optionally EN).

Reads test_sample.jsonl, calls Bedrock with the same system+user prompt used for all
frontier models, checkpoints to out_claude_bedrock_{lang}.jsonl as it goes.

Usage:
    python experiments/run_claude_bedrock.py --langs es pt --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.generate import SYSTEM_PROMPT, _user_prompt

RESULTS_DIR = ROOT / "experiments" / "results"
TEST_SAMPLE = ROOT / "data" / "structured" / "test_sample.jsonl"

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-2"


def make_client():
    import boto3
    csv = Path.home() / "rootkey.csv"
    lines = csv.read_text().strip().splitlines()
    parts = lines[1].split(",")
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=parts[0],
        aws_secret_access_key=parts[1],
    )


def generate_one(client, inst: dict, lang: str, tries: int = 4) -> dict:
    user_content = _user_prompt(inst, lang)
    messages = [{"role": "user", "content": [{"text": user_content}]}]
    system = [{"text": SYSTEM_PROMPT}]
    last_err = None
    for k in range(tries):
        try:
            resp = client.converse(
                modelId=MODEL_ID,
                system=system,
                messages=messages,
                inferenceConfig={"maxTokens": 2000},
            )
            text = resp["output"]["message"]["content"][0]["text"]
            return {"id": inst["id"], "text": text}
        except Exception as e:
            last_err = e
            time.sleep(2 * (k + 1))
    return {"id": inst["id"], "error": str(last_err)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["es", "pt"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    instances = [json.loads(l) for l in TEST_SAMPLE.read_text().splitlines() if l.strip()]
    client = make_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        out_path = RESULTS_DIR / f"out_claude_bedrock_{lang}.jsonl"
        done_ids = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        remaining = [i for i in instances if i["id"] not in done_ids]
        print(f"[{lang}] {len(done_ids)} done, {len(remaining)} remaining", flush=True)
        if not remaining:
            continue

        with open(out_path, "a") as f:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(generate_one, client, inst, lang): inst
                           for inst in remaining}
                for i, fut in enumerate(as_completed(futures), 1):
                    result = fut.result()
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 20 == 0 or i == len(remaining):
                        errs = "err" if result.get("error") else "ok"
                        print(f"  [{lang}] {i}/{len(remaining)} ({errs})", flush=True)

        total = sum(1 for l in out_path.read_text().splitlines() if l.strip())
        errors = sum(1 for l in out_path.read_text().splitlines()
                     if l.strip() and json.loads(l).get("error"))
        print(f"[{lang}] DONE: {total} instances, {errors} errors", flush=True)


if __name__ == "__main__":
    main()
