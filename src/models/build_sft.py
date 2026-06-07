"""Build supervised fine-tuning data: grounded explanations as targets.

Silver targets are the deterministic faithful templates, optionally filtered to
faithfulness == 1.0 by the metric (a self-check). This teaches a small model to
produce explanations grounded in the provided context. Frontier-generated,
verifier-filtered targets can be swapped in later for higher-quality supervision.

Output: data/structured/sft.jsonl with chat-format messages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.generate import template_faithful, SYSTEM_PROMPT, _user_prompt
from src.eval.faithfulness import score_text

INSTANCES = ROOT / "data" / "structured" / "instances.jsonl"
OUT = ROOT / "data" / "structured" / "sft.jsonl"


def build(lang: str = "en", require_faithful: bool = True) -> Path:
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for line in INSTANCES.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            inst = json.loads(line)
            if inst.get("split", "train") != "train":
                continue  # never train on the held-out test season
            target = template_faithful(inst, lang)
            if require_faithful and score_text(target, inst["ground_truth"]).faithfulness < 1.0:
                continue
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(inst, lang)},
                {"role": "assistant", "content": target},
            ]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} SFT examples to {OUT}")
    return OUT


if __name__ == "__main__":
    build()
