"""Run verifier-guided self-correction (the method) vs single-shot baseline.

For each instance: generate with the base backend, then iteratively feed back
contradicted/unverifiable claims for revision. Saves both the first-round and final
texts so they can be re-scored with the LLM extractor for a fair comparison against
the frontier baselines. Azure backend, English.
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

from src.models.self_correct import generate_self_correct

INSTANCES = ROOT / "data" / "structured" / "instances.jsonl"
OUT = ROOT / "experiments" / "results" / "method.json"


def one(inst):
    try:
        r = generate_self_correct(inst, lang="en", backend="azure_openai", max_rounds=2)
        return {"id": inst["id"], "text": r["text"], "rounds": r["rounds"],
                "trace": r["faithfulness_trace"]}
    except Exception as e:
        return {"id": inst["id"], "error": str(e), "trace": []}


def main():
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = os.environ.get("METHOD_DEPLOYMENT", "gpt-54-mini")
    insts = [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one, insts))
    traces = [r["trace"] for r in rows if r.get("trace")]
    first = statistics.mean(t[0] for t in traces if t)
    final = statistics.mean(t[-1] for t in traces if t)
    out = {"deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
           "n": len(rows), "errors": sum(1 for r in rows if r.get("error")),
           "regex_first_round_faithfulness": round(first, 4),
           "regex_final_round_faithfulness": round(final, 4),
           "rows": rows}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"first={first:.3f} final={final:.3f} (regex, EN) n={len(rows)} "
          f"errors={out['errors']}")


if __name__ == "__main__":
    main()
