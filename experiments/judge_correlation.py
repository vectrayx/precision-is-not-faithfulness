"""Metric validation by correlation with an independent LLM judge.

For a mixed sample of explanations (frontier generations + perturbed templates,
spanning a range of true faithfulness), we compare our automatic faithfulness score
against an LLM judge (gpt-5.5, distinct from the gpt-5.4-mini claim extractor) that
estimates the fraction of supported claims directly. High correlation indicates the
automatic metric tracks holistic judgment. The judge is an LLM proxy; human-judge
correlation is future work.

Writes experiments/results/judge_correlation.json.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.faithfulness import score_text
from src.models.generate import template_noisy, template_faithful, chat

RES = ROOT / "experiments" / "results"
INST = {json.loads(l)["id"]: json.loads(l)
        for l in (ROOT / "data/structured/instances.jsonl").read_text().splitlines() if l.strip()}

JUDGE_SYS = "You are a strict fact-checker for F1 race-strategy text."
JUDGE_TMPL = """Given the DATA (ground truth) and an EXPLANATION, estimate the
fraction of the explanation's factual claims (pit laps, compounds, stop counts,
positions, undercut/overcut and outcomes, gaps) that are SUPPORTED by the DATA.
Reply with a single number between 0 and 1 (1 = all claims supported).

DATA:
{ctx}

EXPLANATION:
{exp}

Fraction supported (0-1):"""


def _client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        max_retries=6)


JUDGE_DEPLOYMENT = os.environ.get("JUDGE_DEPLOYMENT", "gpt-55")


def build_sample(n_instances: int = 40):
    """Mixed items with varied faithfulness: frontier (real) + faithful/noisy templates."""
    items = []
    ids = list(INST)[:n_instances]
    # frontier gpt-5.5 EN texts, if available
    fr = json.loads((RES / "frontier.json").read_text()) if (RES / "frontier.json").exists() else None
    fr_map = {}
    if fr:
        for p in fr["runs"].get("gpt-5.5|en", []):
            fr_map[p["id"]] = p.get("text", "")
    for _id in ids:
        inst = INST[_id]
        if fr_map.get(_id):
            items.append((_id, fr_map[_id]))
        items.append((_id, template_faithful(inst, "en")))
        items.append((_id, template_noisy(inst, "en", seed=7)))
    return items


def judge_one(client, ctx, exp):
    try:
        out = chat(client, JUDGE_DEPLOYMENT,
                   [{"role": "system", "content": JUDGE_SYS},
                    {"role": "user", "content": JUDGE_TMPL.format(ctx=ctx, exp=exp)}])
        return float(out.strip().split()[0])
    except Exception:
        return None


def pearson(x, y):
    x, y = np.array(x), np.array(y)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def main():
    client = _client()
    sample = build_sample()
    ours, judge = [], []

    def work(it):
        _id, exp = it
        gt = INST[_id]["ground_truth"]
        our = score_text(exp, gt).faithfulness  # regex (EN)
        js = judge_one(client, INST[_id]["context_text"], exp)
        return our, js

    with ThreadPoolExecutor(max_workers=6) as ex:
        for our, js in ex.map(work, sample):
            if js is not None:
                ours.append(our); judge.append(js)

    out = {"n": len(ours), "judge": JUDGE_DEPLOYMENT,
           "pearson": round(pearson(ours, judge), 4),
           "spearman": round(spearman(ours, judge), 4),
           "mean_metric": round(float(np.mean(ours)), 4),
           "mean_judge": round(float(np.mean(judge)), 4)}
    (RES / "judge_correlation.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
