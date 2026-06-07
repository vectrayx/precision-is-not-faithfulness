"""Faithful Strategy Engineer -- interactive demo (Hugging Face Space).

Pick a race and a strategic decision; the app generates a natural-language strategy
briefing and AUDITS every atomic claim against the telemetry-derived ground truth
(supported / contradicted / unverifiable). The point for race-strategy use: an
assistant whose every statement is checkable -- it does not invent pit calls.

Backends:
  - "Grounded (offline)": deterministic faithful briefing from the structured data.
  - "Inject errors (demo)": a perturbed briefing, to show the verifier catching
    hallucinations in red.
  - "Frontier LLM (Azure)": live LLM generation if Azure OpenAI env vars are set.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

import re

from src.eval.faithfulness import score_text
from src.eval.coverage import coverage
from src.models.generate import template_faithful, template_noisy, commentate
from src.data.championship import season_standings, championship_summary_text

ROOT = Path(__file__).resolve().parent
INSTANCES = ROOT / "data" / "structured" / "instances.jsonl"

_INSTS = [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]
_BY_ID = {i["id"]: i for i in _INSTS}

LABEL_STYLE = {
    "supported": ("#10b981", "✓"),     # green check
    "contradicted": ("#ef4444", "✗"),  # red cross
    "unverifiable": ("#9ca3af", "?"),       # grey question
}


def years():
    return sorted({str(i["year"]) for i in _INSTS})


def gps_for(year):
    return sorted({i["gp"] for i in _INSTS if str(i["year"]) == str(year)})


def instances_for(year, gp):
    opts = [(f"{i['decision_type']}: {', '.join(i['focus_drivers'])}", i["id"])
            for i in _INSTS if str(i["year"]) == str(year) and i["gp"] == gp]
    return sorted(opts)


def _audit_html(result) -> str:
    rows = []
    for c in result.claims:
        color, mark = LABEL_STYLE.get(c["label"], ("#9ca3af", "?"))
        rows.append(
            f"<div style='padding:6px 10px;margin:4px 0;border-left:4px solid {color};"
            f"background:#1118;border-radius:4px'>"
            f"<b style='color:{color}'>{mark} {c['label']}</b> "
            f"<span style='opacity:.85'>&mdash; {c['type']}</span><br>"
            f"<span style='opacity:.7;font-size:.9em'>{c['reason']}</span></div>")
    if not rows:
        rows.append("<i>No checkable claims extracted.</i>")
    return "".join(rows)


def _terse(inst, lang):
    """One true sentence: faithful (high precision) but low coverage -- shows abstention."""
    full = template_faithful(inst, lang).strip()
    parts = re.split(r"(?<=[.!?])\s+", full)
    return parts[0] if parts else full


def run(instance_id, backend, lang, commentator=False):
    inst = _BY_ID[instance_id]
    note = ""
    if backend == "Inject errors (demo)":
        text = template_noisy(inst, lang, seed=3)
    elif backend == "Terse / abstain (demo)":
        text = _terse(inst, lang)
    elif backend == "Frontier LLM (Azure)":
        try:
            from src.models.generate import azure_openai
            text = azure_openai(inst, lang)
        except Exception as e:
            text = template_faithful(inst, lang)
            note = f"⚠ Azure not configured ({type(e).__name__}); showing grounded offline briefing."
    else:
        text = template_faithful(inst, lang)
    if commentator:
        text = commentate(inst, text, lang)  # flair added; facts inside still audited

    r = score_text(text, inst["ground_truth"])
    cov = coverage(r.claims, inst)

    def _c(v):  # green/amber/red by value
        return "#10b981" if v >= 0.9 else "#f59e0b" if v >= 0.6 else "#ef4444"

    prec = r.faithfulness
    rec = cov["recall"]
    gauge = (
        "<div style='display:flex;gap:18px'>"
        f"<div><div style='font-size:1.9em;font-weight:700;color:{_c(prec)}'>{prec*100:.0f}%</div>"
        f"<div style='opacity:.7;font-size:.85em'>precision (faithful)<br>"
        f"{r.supported} ok &middot; {r.contradicted} wrong &middot; {r.unverifiable} unverif.</div></div>"
        f"<div><div style='font-size:1.9em;font-weight:700;color:{_c(rec)}'>{rec*100:.0f}%</div>"
        f"<div style='opacity:.7;font-size:.85em'>recall (coverage)<br>"
        f"covered {cov['covered']}/{cov['total']} key facts</div></div>"
        "</div>")
    if rec < 0.6 and prec >= 0.8:
        gauge += ("<div style='margin-top:8px;padding:6px 10px;background:#f59e0b22;"
                  "border-left:4px solid #f59e0b;border-radius:4px;font-size:.9em'>"
                  "⚠ High precision, low coverage: faithful but uninformative &mdash; "
                  "precision alone rewards saying little.</div>")
    ctx = inst["context_text"]
    return (note or ""), text, gauge, _audit_html(r), ctx


def championship_view(year, lang):
    s = season_standings(int(year))
    summary = championship_summary_text(int(year), lang)
    rows = "".join(
        f"<tr><td>{i+1}</td><td><b>{d}</b></td><td>{pts}</td><td>{w}</td><td>{team or ''}</td></tr>"
        for i, (d, pts, team, w) in enumerate(s["drivers"][:10]))
    table = ("<table style='width:100%;border-collapse:collapse'>"
             "<tr style='text-align:left;opacity:.7'><th>#</th><th>Driver</th>"
             "<th>Points</th><th>Wins</th><th>Team</th></tr>" + rows + "</table>"
             "<div style='opacity:.6;font-size:.85em;margin-top:6px'>"
             "Official points (includes sprint races and fastest-lap bonus).</div>")
    return summary, table


with gr.Blocks(title="Precision Is Not Faithfulness (F1)") as demo:
    gr.Markdown(
        "# 🏎️ Precision Is Not Faithfulness\n"
        "Telemetry-grounded F1 strategy briefings, audited two ways against the data: "
        "**precision** (are the stated claims supported?) *and* **recall / coverage** "
        "(of the facts that mattered, how many did it state?). Precision alone rewards "
        "abstention &mdash; try **Terse / abstain** to see a briefing that is ~100% "
        "faithful yet covers almost nothing. *Research demo; companion to the paper; "
        "not affiliated with any F1 team or FOM.*")
    with gr.Tabs():
        with gr.Tab("Strategy briefing"):
            with gr.Row():
                y = gr.Dropdown(years(), label="Season", value=years()[-1])
                g = gr.Dropdown(gps_for(years()[-1]), label="Grand Prix")
                inst = gr.Dropdown([], label="Strategic decision (strategy / undercut / overcut / defense / race summary)")
            with gr.Row():
                backend = gr.Radio(
                    ["Grounded (offline)", "Terse / abstain (demo)",
                     "Inject errors (demo)", "Frontier LLM (Azure)"],
                    value="Grounded (offline)", label="Briefing source")
                lang = gr.Radio(["en", "es", "pt"], value="en", label="Language / Idioma")
                commentator = gr.Checkbox(value=False, label="🎙️ Commentator mode")
            btn = gr.Button("Generate briefing", variant="primary")
            note = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Strategy briefing")
                    out_text = gr.Textbox(label="", lines=8)
                    out_gauge = gr.HTML()
                with gr.Column():
                    gr.Markdown("### Audit vs telemetry: precision + coverage")
                    out_audit = gr.HTML()
            with gr.Accordion("Telemetry context provided to the model", open=False):
                out_ctx = gr.Textbox(label="", lines=10)
            y.change(lambda yy: gr.update(choices=gps_for(yy), value=None), y, g)
            g.change(lambda yy, gg: gr.update(choices=instances_for(yy, gg), value=None), [y, g], inst)
            btn.click(run, [inst, backend, lang, commentator], [note, out_text, out_gauge, out_audit, out_ctx])

        with gr.Tab("Championship"):
            with gr.Row():
                cy = gr.Dropdown(years(), label="Season", value=years()[-1])
                clang = gr.Radio(["en", "es", "pt"], value="en", label="Language")
            cbtn = gr.Button("Show standings", variant="primary")
            csum = gr.Markdown()
            ctable = gr.HTML()
            cbtn.click(championship_view, [cy, clang], [csum, ctable])


if __name__ == "__main__":
    demo.launch()
