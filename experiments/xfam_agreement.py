"""Cross-family extractor agreement: gpt-5.x vs DeepSeek-V3.2 extractor.

Compares the headline LLM-extractor scores (frontier_llm.json, gpt-5.x) against the
cross-family re-extraction (frontier_xfam.json, DeepSeek) over the SAME generations,
across all three languages. This is the strong version of the extractor-robustness
defense (the regex check in extractor_agreement.py is English-first).

Writes experiments/results/xfam_agreement.json + summary.md. LaTeX macros are emitted
by make_paper_assets.py from the json (single source of truth).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"
A_PATH = RESULTS / "frontier_llm.json"   # gpt-5.x extractor (headline)
B_PATH = RESULTS / "frontier_xfam.json"  # cross-family (DeepSeek)


def macro(vals):
    return round(statistics.mean(vals), 4) if vals else 0.0


def runs_by_id(data):
    return {key: {p["id"]: p for p in per} for key, per in data["runs"].items()}


def main():
    A = json.loads(A_PATH.read_text())
    B = json.loads(B_PATH.read_text())
    langs = A["langs"]
    models = sorted(A["models"])
    a, b = runs_by_id(A), runs_by_id(B)

    per_model = {}
    for m in models:
        a_lang, b_lang = {}, {}
        for lang in langs:
            key = f"{m}|{lang}"
            af = [p["faithfulness"] for p in a.get(key, {}).values() if p["n_claims"] > 0]
            bf = [p["faithfulness"] for p in b.get(key, {}).values() if p["n_claims"] > 0]
            a_lang[lang], b_lang[lang] = macro(af), macro(bf)
        per_model[m] = {"gpt5x": macro(list(a_lang.values())),
                        "xfam": macro(list(b_lang.values())),
                        "gpt5x_by_lang": a_lang, "xfam_by_lang": b_lang}

    def sys_corr(getter):
        av = [getter(m, "gpt5x") for m in models]
        bv = [getter(m, "xfam") for m in models]
        return {"spearman": round(spearmanr(av, bv).statistic, 4),
                "pearson": round(pearsonr(av, bv).statistic, 4)}

    sys_overall = sys_corr(lambda m, e: per_model[m][e])
    sys_by_lang = {lang: sys_corr(lambda m, e: per_model[m][f"{e}_by_lang"][lang]) for lang in langs}

    def inst_corr(keys):
        av, bv = [], []
        for key in keys:
            ad, bd = a.get(key, {}), b.get(key, {})
            for iid, ap in ad.items():
                bp = bd.get(iid)
                if bp and ap["n_claims"] > 0 and bp["n_claims"] > 0:
                    av.append(ap["faithfulness"]); bv.append(bp["faithfulness"])
        if len(av) < 3:
            return {"n": len(av), "pearson": None, "spearman": None}
        return {"n": len(av), "pearson": round(pearsonr(av, bv).statistic, 4),
                "spearman": round(spearmanr(av, bv).statistic, 4),
                "mean_gpt5x": macro(av), "mean_xfam": macro(bv)}

    all_keys = list(A["runs"].keys())
    inst_overall = inst_corr(all_keys)
    inst_by_lang = {lang: inst_corr([f"{m}|{lang}" for m in models]) for lang in langs}

    out = {"extractor_a": "gpt-5.x", "extractor_b": B.get("extractor", "DeepSeek-V3.2"),
           "models": models, "langs": langs, "per_model": per_model,
           "system_level": {"overall": sys_overall, "by_lang": sys_by_lang},
           "instance_level": {"overall": inst_overall, "by_lang": inst_by_lang}}
    (RESULTS / "xfam_agreement.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    md = [f"# Cross-family extractor agreement: gpt-5.x vs {out['extractor_b']}\n",
          f"System-level (avg over langs, N={len(models)} models): "
          f"Spearman={sys_overall['spearman']}, Pearson={sys_overall['pearson']}.\n",
          f"Instance-level (overall, N={inst_overall['n']}): "
          f"Pearson={inst_overall['pearson']}, Spearman={inst_overall['spearman']}.\n",
          "| Model | gpt-5.x | xfam |", "|---|---|---|"]
    for m in models:
        md.append(f"| {m} | {per_model[m]['gpt5x']:.3f} | {per_model[m]['xfam']:.3f} |")
    md += ["\n| Lang | sys Spearman | inst N | inst Pearson | inst Spearman |",
           "|---|---|---|---|---|"]
    for lang in langs:
        s, i = sys_by_lang[lang], inst_by_lang[lang]
        md.append(f"| {lang} | {s['spearman']} | {i['n']} | {i['pearson']} | {i['spearman']} |")
    (RESULTS / "xfam_agreement_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
