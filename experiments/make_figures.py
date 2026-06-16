"""Generate the paper's results figure from the real result JSONs.

Produces paper/fig_results.pdf with two panels:
  (a) F1 (precision+recall) by system: frontier zero-shot vs small fine-tuned (RQ3),
  (b) frontier precision by language (RQ2).
Re-run after results refresh; the figure reads the JSONs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
PAPER = ROOT / "paper"

GREEN, BLUE, ORANGE, GREY = "#10b981", "#2563eb", "#f59e0b", "#9ca3af"


def main():
    fr = json.loads((RES / "frontier_llm.json").read_text())
    rq3 = json.loads((RES / "rq3_small_models_clean.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8))

    # --- (a) F1 comparison: frontier (EN) + Claude + FT models ---
    # Frontier rows from coverage_llm.json (EN only)
    cov_src = RES / "coverage_llm.json"
    cov_rows = json.loads(cov_src.read_text())["rows"] if cov_src.exists() else []
    en_frontier = [r for r in cov_rows if r["lang"] == "en"]

    rows_a = []
    for r in en_frontier:
        rows_a.append({"name": r["model"], "f1": r["f1"], "kind": "frontier"})

    # FT models
    ft_map = {
        "strat_llama": "Llama-3.2-1B (FT)",
        "strat_phi": "Phi-4-mini (FT)",
        "strat_mistral": "Mistral-7B (FT)",
        "multi_phi": "Phi-4-mini (FT, multi)",
    }
    for key, label in ft_map.items():
        m = rq3["models"].get(key)
        if m:
            rows_a.append({"name": label, "f1": m["F1"], "kind": "ft"})

    rows_a.sort(key=lambda r: r["f1"])

    names = [r["name"] for r in rows_a]
    vals = [r["f1"] for r in rows_a]
    colors = [GREEN if r["kind"] == "ft" else GREY for r in rows_a]
    ax1.barh(range(len(names)), vals, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("$F_1$ (precision + recall)")
    ax1.set_title("(a) Coverage $F_1$: frontier vs fine-tuned", fontsize=9)
    for i, v in enumerate(vals):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=6.5)

    # --- (b) frontier precision by language ---
    langs = ["en", "es", "pt"]
    lang_colors = {"en": "#1d4ed8", "es": "#10b981", "pt": "#f59e0b"}
    models = sorted({s["model"] for s in fr["summaries"]})
    n = len(models)
    width = 0.26
    for j, l in enumerate(langs):
        ys = [next((s["macro_faithfulness"] for s in fr["summaries"]
                    if s["model"] == m and s["lang"] == l), 0) for m in models]
        xs = [k + (j - 1) * width for k in range(n)]
        ax2.bar(xs, ys, width, label=l.upper(), color=lang_colors[l])
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([m.replace("gpt-", "gpt").replace("gemini-", "gem")
                         .replace("DeepSeek-", "DS").replace("grok-", "grok")
                         for m in models], fontsize=6, rotation=30, ha="right")
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("Precision (faithfulness)")
    ax2.set_title("(b) Frontier precision by language", fontsize=9)
    ax2.legend(fontsize=7, loc="lower right", ncol=3)

    fig.tight_layout()
    fig.savefig(PAPER / "fig_results.pdf", bbox_inches="tight")
    print("Wrote paper/fig_results.pdf")


if __name__ == "__main__":
    main()
