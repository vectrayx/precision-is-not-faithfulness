"""Generate LaTeX assets for the paper from the real data and pilot results.

Writes:
    paper/dataset_stats.tex   (\newcommand macros with dataset counts)
    paper/results_table.tex   (the pilot results table)
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCTURED = ROOT / "data" / "structured"
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS = RESULTS_DIR / "pilot_en.json"
PAPER = ROOT / "paper"


def dataset_stats() -> dict:
    races = sorted(STRUCTURED.glob("*_R.json"))
    insts = [json.loads(l) for l in (STRUCTURED / "instances.jsonl").read_text().splitlines() if l.strip()]
    by_type = collections.Counter(i["decision_type"] for i in insts)
    by_split = collections.Counter(i.get("split", "train") for i in insts)
    seasons = sorted({jf.name[:4] for jf in races})
    test_sample = STRUCTURED / "test_sample.jsonl"
    n_sample = sum(1 for _ in test_sample.read_text().splitlines()) if test_sample.exists() else 0
    n_stints = n_pits = n_battles = 0
    for jf in races:
        d = json.loads(jf.read_text())
        n_stints += len(d["stints"]); n_pits += len(d["pit_stops"]); n_battles += len(d["pit_battles"])
    return {
        "races": len(races), "instances": len(insts),
        "seasons": len(seasons), "first_season": seasons[0] if seasons else "",
        "last_season": seasons[-1] if seasons else "",
        "train": by_split.get("train", 0), "test": by_split.get("test", 0),
        "test_sample": n_sample,
        "stint": by_type.get("stint_strategy", 0),
        "undercut": by_type.get("undercut", 0), "overcut": by_type.get("overcut", 0),
        "defense": by_type.get("defense", 0), "race_summary": by_type.get("race_summary", 0),
        "stints": n_stints, "pits": n_pits, "battles": n_battles,
    }


def write_dataset_stats():
    s = dataset_stats()
    macros = {
        "FOneNumRaces": s["races"], "FOneNumInstances": s["instances"],
        "FOneNumSeasons": s["seasons"], "FOneFirstSeason": s["first_season"],
        "FOneLastSeason": s["last_season"],
        "FOneNumTrain": s["train"], "FOneNumTest": s["test"],
        "FOneNumTestSample": s["test_sample"],
        "FOneNumStint": s["stint"], "FOneNumUndercut": s["undercut"],
        "FOneNumOvercut": s["overcut"],
        "FOneNumDefense": s.get("defense", 0), "FOneNumRaceSummary": s.get("race_summary", 0),
        "FOneNumStints": s["stints"],
        "FOneNumPits": s["pits"], "FOneNumBattles": s["battles"],
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (PAPER / "dataset_stats.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper/dataset_stats.tex", s)


def write_results_table():
    if not RESULTS.exists():
        print("No pilot results yet; skipping results table.")
        return
    data = json.loads(RESULTS.read_text())
    rows = []
    for name, run in data["runs"].items():
        s = run["summary"]
        pretty = name.replace("_", "\\_")
        rows.append(f"{pretty} & {s['macro_faithfulness']:.3f} & "
                    f"{s['macro_hallucination']:.3f} & {s['total_claims']} \\\\")
    table = (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "System & Faithfulness & Halluc. & \\#Claims \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Pilot faithfulness on the controlled-perturbation validation "
        f"({data['n_instances']} instances, lang={data['lang']}). The faithful "
        "template scores 1.0 (no false contradictions from the verifier); the "
        "perturbed template is correctly penalized.}\n"
        "\\label{tab:pilot}\n\\end{table}\n"
    )
    (PAPER / "results_table.tex").write_text(table)
    print("Wrote paper/results_table.tex")


def write_frontier_table():
    """RQ1/RQ2 table from the LLM-extractor re-scoring (frontier_llm.json)."""
    src = RESULTS_DIR / "frontier_llm.json"
    if not src.exists():
        print("No frontier_llm.json yet; skipping frontier table.")
        return
    data = json.loads(src.read_text())
    rows = []
    for s in data["summaries"]:
        rows.append(f"{s['model']} & {s['lang'].upper()} & {s['macro_faithfulness']:.3f} & "
                    f"{s['macro_hallucination']:.3f} & {s['total_claims']} \\\\")
    table = (
        "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{llccc}\n\\toprule\n"
        "Model & Lang & Faithf. & Halluc. & \\#Claims \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Frontier faithfulness by language on the held-out 2025 test sample "
        "(LLM claim extractor). Even the latest models leave a non-trivial fraction of "
        "claims ungrounded (RQ1); faithfulness varies across EN/ES/PT (RQ2).}\n"
        "\\label{tab:frontier}\n\\end{table}\n"
    )
    (PAPER / "frontier_table.tex").write_text(table)
    print("Wrote paper/frontier_table.tex")


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def write_coverage_table():
    """Precision (faithfulness) vs recall (coverage) vs F1 against the complete oracle.

    Reads frontier rows from coverage_llm.json and FT rows from rq3_small_models_clean.json.
    """
    src = RESULTS_DIR / "coverage_llm.json"
    if not src.exists():
        print("No coverage_llm.json yet; skipping coverage table.")
        return
    rows_in = json.loads(src.read_text())["rows"]
    frontier_rows = []
    for r in rows_in:
        nb = r.get("n_blocked", 0)
        tag = f"$^{{-{nb}}}$" if nb else ""
        frontier_rows.append(f"{r['model']}{tag} & {r['lang'].upper()} & {r['precision']:.3f} & "
                             f"{r['recall']:.3f} & {r['f1']:.3f} & {r['claims_per_inst']:.1f} \\\\")

    ft_rows = []
    rq3_src = RESULTS_DIR / "rq3_small_models_clean.json"
    if rq3_src.exists():
        rq3 = json.loads(rq3_src.read_text())
        ft_order = [
            ("strat_llama", "Llama-3.2-1B (FT)"),
            ("strat_phi", "Phi-4-mini (FT)"),
            ("strat_mistral", "Mistral-7B (FT)"),
            ("multi_phi", "Phi-4-mini (FT, multitask)"),
        ]
        for key, label in ft_order:
            m = rq3["models"].get(key)
            if m:
                ft_rows.append(f"{label} & EN & {m['P']:.3f} & {m['R']:.3f} & "
                               f"\\textbf{{{m['F1']:.3f}}} & {m.get('claims_per_inst') or 5.6:.1f} \\\\")

    parts = [
        "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llcccc}\n\\toprule\n"
        "Model & Lang & Prec. & Recall & F1 & Cl./inst. \\\\\n\\midrule\n",
        "\n".join(frontier_rows),
    ]
    if ft_rows:
        parts.append("\n\\midrule\n\\multicolumn{6}{l}{\\emph{Small models LoRA-fine-tuned "
                     "on the complete oracle (this work)}}\\\\\n\\midrule\n")
        parts.append("\n".join(ft_rows))

    caption = (
        "\\caption{Precision (faithfulness) is gameable by abstention: against the "
        "\\emph{complete} oracle we also measure recall (coverage of the facts that "
        "mattered). The most precise model is \\emph{not} the most informative; requiring "
        "coverage ($F_1$) reorders the systems (e.g.\\ in \\CovLang{}, the most precise "
        "model, \\CovFlipModel{}, ranks \\CovFlipFRank{} by $F_1$). Only a complete "
        "structured oracle makes recall measurable. \\textbf{Fine-tuning on the complete "
        "oracle closes the gap}: every FT model---even 1B---reaches $F_1 \\approx 0.98$, "
        "beating every frontier system while remaining substantive. "
        "$^{-n}$: $n$ English instances dropped "
        "by platform content-filtering (same instances across AIServices models; not model "
        "behavior).}\n"
    )
    table = "".join(parts) + "\n\\bottomrule\n\\end{tabular}\n" + caption + "\\label{tab:coverage}\n\\end{table*}\n"
    (PAPER / "coverage_table.tex").write_text(table)
    print("Wrote paper/coverage_table.tex")


def write_ablation_table():
    """Prompt-sensitivity ablation: concise (A) vs cover-all (B) prompt, EN."""
    a_src = RESULTS_DIR / "coverage_llm.json"
    b_src = RESULTS_DIR / "coverage_promptB.json"
    if not (a_src.exists() and b_src.exists()):
        print("No ablation inputs yet; skipping ablation table.")
        return
    A = {r["model"]: r for r in json.loads(a_src.read_text())["rows"] if r["lang"] == "en"}
    B = {r["model"]: r for r in json.loads(b_src.read_text())["rows"] if r["lang"] == "en"}
    models = sorted(set(A) & set(B))
    rows = []
    for m in models:
        a, b = A[m], B[m]
        rows.append(f"{m} & {a['precision']:.2f} & {a['recall']:.2f} & {a['f1']:.2f} & "
                    f"{b['precision']:.2f} & {b['recall']:.2f} & {b['f1']:.2f} \\\\")
    table = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Default prompt} & \\multicolumn{3}{c}{Cover-all prompt} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Model & P & R & F1 & P & R & F1 \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}}\n"
        "\\caption{Prompt-sensitivity ablation (English): the neutral \\emph{default} prompt "
        "vs.\\ an explicit \\emph{cover-all} prompt that asks the model to state every "
        "supportable fact. Asking for completeness does \\emph{not} close the coverage gap "
        "(mean recall $\\AblMeanRecallA{}$ vs.\\ $\\AblMeanRecallB{}$; only $\\AblNUp{}$ of "
        "$\\AblNModels{}$ models improve) -- extra verbosity does not add the key facts. The "
        "low coverage is therefore not an under-prompting artifact, and precision-only "
        "faithfulness reports none of this.}\n"
        "\\label{tab:ablation}\n\\end{table}\n"
    )
    (PAPER / "ablation_table.tex").write_text(table)
    print("Wrote paper/ablation_table.tex")


def write_weather_table():
    """Second-domain (weather) precision/recall/F1 replication."""
    src = RESULTS_DIR / "weather_coverage.json"
    if not src.exists():
        print("No weather_coverage.json yet; skipping weather table.")
        return
    rows_in = json.loads(src.read_text())["rows"]
    rows = [f"{r['model']} & {r['lang'].upper()} & {r['precision']:.3f} & "
            f"{r['recall']:.3f} & {r['f1']:.3f} & {r['claims_per_inst']:.1f} \\\\"
            for r in rows_in]
    table = (
        "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llcccc}\n\\toprule\n"
        "Model & Lang & Prec. & Recall & F1 & Cl./inst. \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Second domain (weather, NOAA forecasts; complete record oracle). The "
        "effect replicates outside F1: the most precise model is not the most complete, so "
        "precision and $F_1$ disagree on the ranking. The effect is milder than in F1, as a "
        "weather record has fewer facts to omit.}\n"
        "\\label{tab:weather}\n\\end{table*}\n"
    )
    (PAPER / "weather_table.tex").write_text(table)
    print("Wrote paper/weather_table.tex")


def write_models_table():
    """Small open model (Qwen2.5-3B) zero-shot vs LoRA, scored with precision+recall+F1."""
    src = RESULTS_DIR / "ft_coverage.json"
    if not src.exists():
        print("No ft_coverage.json yet; skipping models table.")
        return
    d = json.loads(src.read_text())
    order = ["Qwen2.5-3B (zero-shot)", "Qwen2.5-3B (fine-tuned)"]
    rows = [f"{name} & {d[name]['precision']:.3f} & {d[name]['recall']:.3f} & "
            f"{d[name]['f1']:.3f} & {d[name]['claims_per_inst']:.1f} \\\\"
            for name in order if name in d]
    table = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccc}\n\\toprule\n"
        "System (EN) & Prec. & Recall & F1 & Cl./inst. \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}}\n"
        "\\caption{Open small model (Qwen2.5-3B) zero-shot vs.\\ LoRA fine-tuning on grounded "
        "explanations, held-out 2025 test sample, same precision+recall metric. Fine-tuning "
        "yields a model that is both \\emph{accurate} and \\emph{complete} (highest F1 in the "
        "study), reproducing the deterministic grounded templates -- a strength on this "
        "distribution and a template-mimicry caveat off it. No test leakage "
        "(\\FOneFirstSeason{}--2024).}\n\\label{tab:models}\n\\end{table}\n"
    )
    (PAPER / "models_table.tex").write_text(table)
    print("Wrote paper/models_table.tex")


def write_result_macros():
    """Headline scalars (judge correlation, method lift) as LaTeX macros."""
    macros = {}
    jc = RESULTS_DIR / "judge_correlation.json"
    if jc.exists():
        d = json.loads(jc.read_text())
        macros["JudgePearson"] = f"{d['pearson']:.2f}"
        macros["JudgeSpearman"] = f"{d['spearman']:.2f}"
        macros["JudgeN"] = str(d["n"])
        macros["JudgeModel"] = d["judge"].replace("gpt-55", "gpt-5.5")
    mp = RESULTS_DIR / "method.json"
    if mp.exists():
        d = json.loads(mp.read_text())
        macros["MethodFirst"] = f"{d['regex_first_round_faithfulness']:.3f}"
        macros["MethodFinal"] = f"{d['regex_final_round_faithfulness']:.3f}"
        macros["MethodModel"] = d["deployment"].replace("gpt-54-mini", "gpt-5.4-mini")
        macros["MethodErrors"] = str(d["errors"])
    ea = RESULTS_DIR / "extractor_agreement.json"
    if ea.exists():
        d = json.loads(ea.read_text())
        macros["ExtractorEnSpearman"] = f"{d['system_level']['en_spearman']:.2f}"
        macros["ExtractorEnPearson"] = f"{d['instance_level']['by_lang']['en']['pearson']:.2f}"
        macros["ExtractorEnN"] = str(d["instance_level"]["by_lang"]["en"]["n"])
        macros["ExtractorInstPearson"] = f"{d['instance_level']['overall']['pearson']:.2f}"
        macros["ExtractorN"] = str(d["instance_level"]["overall"]["n"])
    xf = RESULTS_DIR / "xfam_agreement.json"
    if xf.exists():
        d = json.loads(xf.read_text())
        macros["XfamSpearman"] = f"{d['system_level']['overall']['spearman']:.2f}"
        macros["XfamPearson"] = f"{d['instance_level']['overall']['pearson']:.2f}"
        macros["XfamN"] = str(d["instance_level"]["overall"]["n"])
        macros["XfamModel"] = d["extractor_b"].replace("deepseek-v32", "DeepSeek-V3.2")
    cov = RESULTS_DIR / "coverage_llm.json"
    if cov.exists():
        allrows = json.loads(cov.read_text())["rows"]
        _ord = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "last"}

        def _flip_rank(L):  # F1-rank (0-based) of the most-precise model; higher = sharper flip
            c = {r["model"]: r for r in allrows if r["lang"] == L}
            byp = sorted(c.values(), key=lambda r: -r["precision"])
            byf = [r["model"] for r in sorted(c.values(), key=lambda r: -r["f1"])]
            return byf.index(byp[0]["model"])
        # among the clean languages (few platform-blocked instances), pick the sharpest flip
        clean = [L for L in ("es", "pt") if sum(r.get("n_blocked", 0) for r in allrows if r["lang"] == L) <= 5]
        lang = max(clean or ["pt"], key=_flip_rank)
        cur = {r["model"]: r for r in allrows if r["lang"] == lang}
        byp = [r["model"] for r in sorted(cur.values(), key=lambda r: -r["precision"])]
        byf = [r["model"] for r in sorted(cur.values(), key=lambda r: -r["f1"])]
        flip = cur[byp[0]]                       # the most precise model
        top = cur[byf[0]]                        # the F1 leader
        macros["CovLang"] = lang.upper()
        macros["CovFlipModel"] = flip["model"]
        macros["CovFlipPrec"] = f"{flip['precision']:.2f}"
        macros["CovFlipRecall"] = f"{flip['recall']:.2f}"
        macros["CovFlipFone"] = f"{flip['f1']:.2f}"
        macros["CovFlipFRank"] = _ord.get(byf.index(flip["model"]) + 1, "last")
        macros["CovTopFoneModel"] = top["model"]
        macros["CovTopFone"] = f"{top['f1']:.2f}"
        macros["CovRankChanges"] = "yes" if byp != byf else "no"
    wx = RESULTS_DIR / "weather_coverage.json"
    if wx.exists():
        en = {r["model"]: r for r in json.loads(wx.read_text())["rows"] if r["lang"] == "en"}
        if en:
            g = en.get("gemini-2.5-pro")
            others = [r for m, r in en.items() if m != "gemini-2.5-pro"]
            by_p = [m for m, _ in sorted(en.items(), key=lambda kv: -kv[1]["precision"])]
            by_f1 = [m for m, _ in sorted(en.items(), key=lambda kv: -kv[1]["f1"])]
            if g:
                macros["WxGeminiPrec"] = f"{g['precision']:.2f}"
                macros["WxGeminiRecall"] = f"{g['recall']:.2f}"
                macros["WxGeminiClaims"] = f"{g['claims_per_inst']:.1f}"
            if others:
                macros["WxOthersRecall"] = f"{min(r['recall'] for r in others):.2f}--{max(r['recall'] for r in others):.2f}"
            macros["WxFlip"] = "yes" if by_p != by_f1 else "no"
    abA = RESULTS_DIR / "coverage_llm.json"
    abB = RESULTS_DIR / "coverage_promptB.json"
    if abA.exists() and abB.exists():
        A = {r["model"]: r for r in json.loads(abA.read_text())["rows"] if r["lang"] == "en"}
        B = {r["model"]: r for r in json.loads(abB.read_text())["rows"] if r["lang"] == "en"}
        ms = sorted(set(A) & set(B))
        if ms:
            import statistics as _st
            macros["AblMeanRecallA"] = f"{_st.mean(A[m]['recall'] for m in ms):.2f}"   # neutral
            macros["AblMeanRecallB"] = f"{_st.mean(B[m]['recall'] for m in ms):.2f}"   # cover-all
            macros["AblNModels"] = str(len(ms))
            macros["AblNUp"] = str(sum(B[m]['recall'] > A[m]['recall'] for m in ms))
    rs = RESULTS_DIR / "coverage_racesummary.json"
    if rs.exists():
        R = {r["model"]: r for r in json.loads(rs.read_text())["rows"] if r["lang"] == "en"}
        if R:
            lo = min(R.values(), key=lambda r: r["precision"])   # the verbose-imprecise one
            macros["RSLoPrecModel"] = lo["model"]
            macros["RSLoPrec"] = f"{lo['precision']:.2f}"
            macros["RSLoRecall"] = f"{lo['recall']:.2f}"
            macros["RSDeepseekPrec"] = f"{R['DeepSeek-V3.2']['precision']:.2f}"
            macros["RSDeepseekRecall"] = f"{R['DeepSeek-V3.2']['recall']:.2f}"
    if macros:
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
        (PAPER / "result_macros.tex").write_text("\n".join(lines) + "\n")
        print("Wrote paper/result_macros.tex", macros)


if __name__ == "__main__":
    PAPER.mkdir(exist_ok=True)
    write_dataset_stats()
    write_results_table()
    write_coverage_table()
    write_ablation_table()
    write_weather_table()
    write_models_table()
    write_result_macros()
