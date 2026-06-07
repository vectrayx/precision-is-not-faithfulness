"""Find a real frontier generation where the verifier catches a hallucination and
emit paper/example.tex (a boxed qualitative example).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.faithfulness import score_text

RES = ROOT / "experiments" / "results"
INST = {json.loads(l)["id"]: json.loads(l)
        for l in (ROOT / "data/structured/instances.jsonl").read_text().splitlines() if l.strip()}


def find_example():
    fr = json.loads((RES / "frontier.json").read_text())
    best = None
    for key, per in fr["runs"].items():
        if not key.endswith("|en"):
            continue
        model = key.split("|")[0]
        for p in per:
            txt = p.get("text", "")
            if not txt:
                continue
            r = score_text(txt, INST[p["id"]]["ground_truth"])
            con = [c for c in r.claims if c["label"] == "contradicted"]
            sup = [c for c in r.claims if c["label"] == "supported"]
            # want a clear case: at least one contradiction and some supported claims
            if con and len(sup) >= 2:
                score = len(sup) - len(con)
                if best is None or score > best[0]:
                    best = (score, model, p["id"], txt, r, con[0], sup[:2])
    return best


def latex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def main():
    b = find_example()
    if not b:
        print("No suitable example found"); return
    _, model, _id, txt, r, con, sups = b
    excerpt = textwrap.shorten(" ".join(txt.split()), width=320, placeholder=" [...]")
    body = (
        "\\begin{figure}[t]\n\\centering\n\\small\n"
        "\\fbox{\\parbox{0.93\\columnwidth}{\n"
        f"\\textbf{{{latex_escape(model)} briefing (excerpt).}} "
        f"\\textit{{{latex_escape(excerpt)}}}\\\\[4pt]\n"
        "\\textbf{Faithfulness audit.}\\\\\n"
        f"\\textcolor{{red}}{{$\\times$ contradicted}}: {latex_escape(con['reason'])}\\\\\n"
        f"\\textcolor{{teal}}{{$\\checkmark$ supported}}: {latex_escape(sups[0]['reason'])}; "
        f"{latex_escape(sups[1]['reason'])}\\\\\n"
        f"Score: {r.supported}/{r.n_claims} supported, {r.contradicted} contradicted.\n"
        "}}\n"
        "\\caption{A real frontier briefing where the verifier flags an ungrounded "
        "claim against the telemetry while confirming the rest.}\n"
        "\\label{fig:example}\n\\end{figure}\n"
    )
    (ROOT / "paper" / "example.tex").write_text(body)
    print(f"Wrote paper/example.tex from {model} / {_id}")
    print("contradiction:", con["reason"])


if __name__ == "__main__":
    main()
