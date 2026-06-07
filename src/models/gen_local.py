"""Generate explanations with a local HF model (base or LoRA-fine-tuned).

Runs on the GPU VM. Writes a JSONL of {id, decision_type, text} that is scored
off-box with the faithfulness metric. Used to compare the base small model vs. its
fine-tuned variant against the frontier baselines.

    python3 src/models/gen_local.py --model Qwen/Qwen2.5-3B-Instruct \
        --instances data/structured/instances.jsonl --lang en --out out_base.jsonl
    python3 src/models/gen_local.py --model Qwen/Qwen2.5-3B-Instruct \
        --adapter checkpoints/qwen3b-f1-lora --lang en --out out_ft.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are an F1 strategy analyst. Explain the strategic decision using ONLY the "
    "data provided. Do not invent laps, compounds, gaps, positions, or outcomes that "
    "are not in the data. Be concise."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--instances", default="data/structured/instances.jsonl")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=400)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    insts = [json.loads(l) for l in open(args.instances) if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for inst in insts:
            user = f"{inst['prompts'][args.lang]}\n\nData:\n{inst['context_text']}"
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(model.device)
            n_in = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
            f.write(json.dumps({"id": inst["id"], "decision_type": inst["decision_type"],
                                "text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
