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
    """Precision (faithfulness) vs recall (coverage) vs F1 against the complete oracle."""
    src = RESULTS_DIR / "coverage_llm.json"
    if not src.exists():
        print("No coverage_llm.json yet; skipping coverage table.")
        return
    rows_in = json.loads(src.read_text())["rows"]
    # hallucination (contradicted rate) from the frontier re-scoring, matched by cell
    halluc = {}
    fl = RESULTS_DIR / "frontier_llm.json"
    if fl.exists():
        for s in json.loads(fl.read_text())["summaries"]:
            halluc[(s["model"], s["lang"])] = s["macro_hallucination"]
    rows = [f"{r['model']} & {r['lang'].upper()} & {r['precision']:.3f} & "
            f"{r['recall']:.3f} & {r['f1']:.3f} & "
            f"{halluc.get((r['model'], r['lang']), 0.0):.3f} & {r['claims_per_inst']:.1f} \\\\"
            for r in rows_in]
    table = (
        "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llccccc}\n\\toprule\n"
        "Model & Lang & Prec. & Recall & F1 & Hall. & Cl./inst. \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Precision (faithfulness) is gameable by abstention: against the "
        "\\emph{complete} oracle we also measure recall (coverage of the facts that "
        "mattered). Even precise models leave claims ungrounded and produce hard "
        "contradictions (Hall.); gemini-2.5-pro is the most precise yet the least "
        "informative (lowest recall), so requiring coverage (F1) inverts the ranking, "
        "moving it from first to last. Only a complete structured oracle makes recall "
        "measurable.}\n"
        "\\label{tab:coverage}\n\\end{table*}\n"
    )
    (PAPER / "coverage_table.tex").write_text(table)
    print("Wrote paper/coverage_table.tex")


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
        "abstention effect replicates outside F1: the same model that abstains there "
        "(gemini-2.5-pro) again states the fewest facts and has by far the lowest recall, "
        "so precision and $F_1$ disagree on the ranking. The effect is milder than in F1, "
        "as weather records have fewer facts to omit.}\n"
        "\\label{tab:weather}\n\\end{table*}\n"
    )
    (PAPER / "weather_table.tex").write_text(table)
    print("Wrote paper/weather_table.tex")


def write_models_table():
    """Small-model vs frontier comparison from small_models.json (if present)."""
    src = (ROOT / "experiments" / "results" / "small_models.json")
    if not src.exists():
        print("No small_models.json yet; skipping models table.")
        return
    data = json.loads(src.read_text())
    rows = [f"{r['system']} & {r['faithfulness']:.3f} & {r['hallucination']:.3f} & "
            f"{r.get('coverage', 0):.1f} \\\\"
            for r in data["rows"]]
    table = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lccc}\n\\toprule\n"
        "System (EN) & Faithf. & Halluc. & Claims/inst. \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}}\n"
        "\\caption{Small open model (Qwen2.5-3B) before/after fine-tuning vs.\\ frontier "
        "on the held-out 2025 test sample, English, same LLM-extractor metric. "
        "Faithfulness is read alongside coverage (claims per instance): the fine-tuned "
        "model is both more faithful and more concise. Fine-tuned only on "
        "\\FOneFirstSeason{}--2024 (no test leakage).}\n\\label{tab:models}\n\\end{table}\n"
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
        rows = {(r["model"], r["lang"]): r for r in json.loads(cov.read_text())["rows"]}
        en = {m: r for (m, l), r in rows.items() if l == "en"}
        if en:
            g = en.get("gemini-2.5-pro")
            top_f1 = max(en.values(), key=lambda r: r["f1"])
            if g:
                macros["CovGeminiPrec"] = f"{g['precision']:.2f}"
                macros["CovGeminiRecall"] = f"{g['recall']:.2f}"
                macros["CovGeminiFone"] = f"{g['f1']:.2f}"
            macros["CovTopFoneModel"] = top_f1["model"]
            macros["CovTopFone"] = f"{top_f1['f1']:.2f}"
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
    if macros:
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
        (PAPER / "result_macros.tex").write_text("\n".join(lines) + "\n")
        print("Wrote paper/result_macros.tex", macros)


if __name__ == "__main__":
    PAPER.mkdir(exist_ok=True)
    write_dataset_stats()
    write_results_table()
    write_coverage_table()
    write_weather_table()
    write_models_table()
    write_result_macros()
