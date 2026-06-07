"""Generate the paper's results figure from the real result JSONs.

Produces paper/fig_results.pdf with two panels:
  (a) faithfulness by system on the held-out test (RQ3: small fine-tuned vs frontier),
  (b) frontier faithfulness by language (RQ2).
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

GREEN, BLUE, GREY = "#10b981", "#2563eb", "#9ca3af"


def main():
    sm = json.loads((RES / "small_models.json").read_text())
    fr = json.loads((RES / "frontier_llm.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2))

    # (a) systems
    rows = sm["rows"]
    names = [r["system"].replace(" (zero-shot)", "").replace(" (fine-tuned)", "*") for r in rows]
    vals = [r["faithfulness"] for r in rows]
    colors = [GREEN if "*" in n else (BLUE if "gpt" in n.lower() else GREY) for n in names]
    ax1.barh(range(len(names)), vals, color=colors)
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis(); ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Faithfulness"); ax1.set_title("(a) Held-out 2025 test", fontsize=9)
    for i, v in enumerate(vals):
        ax1.text(v - 0.02, i, f"{v:.2f}", va="center", ha="right", color="white", fontsize=7)

    # (b) frontier by language: x = models, grouped bars per language
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
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("Faithfulness")
    ax2.set_title("(b) Frontier by language", fontsize=9)
    ax2.legend(fontsize=7, loc="lower right", ncol=3)

    fig.tight_layout()
    fig.savefig(PAPER / "fig_results.pdf", bbox_inches="tight")
    print("Wrote paper/fig_results.pdf")


if __name__ == "__main__":
    main()
