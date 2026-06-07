"""LoRA fine-tuning of a small open model on grounded F1 explanations.

Runs on the GCP GPU VM (see scripts/gcp_gpu_vm.sh). Requires:
    pip install -r requirements-cloud.txt   # transformers, peft, trl, accelerate, datasets

Example:
    python src/models/finetune.py --model Qwen/Qwen2.5-3B-Instruct \
        --data data/structured/sft.jsonl --out checkpoints/qwen3b-f1-lora
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/structured/sft.jsonl")
    ap.add_argument("--out", default="checkpoints/qwen3b-f1-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    # Lazy imports so the repo stays importable without heavy deps.
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    ds = load_dataset("json", data_files=args.data, split="train")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=8,
        learning_rate=args.lr, logging_steps=10, save_strategy="epoch",
        bf16=True, packing=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
