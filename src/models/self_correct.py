"""Verifier-guided generation (the paper's method contribution).

Wrap any real-LLM generator: generate -> score with the faithfulness verifier ->
feed back the contradicted/unverifiable claims as targeted feedback -> ask the model
to revise. Iterate up to `max_rounds`. This raises faithfulness without a reference,
using only the structured verifier as the signal.
"""
from __future__ import annotations

import os

from src.eval.faithfulness import score_text
from src.eval.verify import CONTRADICTED, UNVERIFIABLE
from src.models.generate import SYSTEM_PROMPT, _user_prompt, chat


def _feedback(result) -> str:
    bad = [c for c in result.claims if c["label"] in (CONTRADICTED, UNVERIFIABLE)]
    lines = ["Some statements are not grounded in the data. Fix or remove ONLY these, "
             "keeping everything else:"]
    for c in bad:
        lines.append(f'- "{c["span"].strip()}" -> {c["label"]}: {c["reason"]}')
    return "\n".join(lines)


def generate_self_correct(inst: dict, lang: str = "en", backend: str = "azure_openai",
                          max_rounds: int = 2) -> dict:
    """Returns {text, rounds, faithfulness_trace}."""
    if backend == "azure_openai":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    else:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        model = os.environ["OPENAI_MODEL"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(inst, lang)}]
    trace = []
    text = ""
    for _ in range(max_rounds + 1):
        text = chat(client, model, messages)
        r = score_text(text, inst["ground_truth"])
        trace.append(r.faithfulness)
        if r.contradicted == 0 and r.unverifiable == 0:
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": _feedback(r)}]
    return {"text": text, "rounds": len(trace) - 1, "faithfulness_trace": trace}
