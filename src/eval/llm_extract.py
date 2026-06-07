"""Language-agnostic claim extraction via an LLM (Azure OpenAI).

Returns the SAME typed Claim schema as the regex extractor, but works for free-form
output in any language (EN/ES/PT), which the regex extractor cannot. Used to score
non-English generations fairly for the cross-lingual study (RQ2).

Reads Azure creds from env; uses a cheap deployment (default gpt-54-mini).
"""
from __future__ import annotations

import json
import os

from .claims import Claim

_SCHEMA = """Extract every atomic, checkable factual claim about race strategy.
Output JSON: {"claims": [ {"type": ..., "fields": {...}}, ... ]}.
Allowed types and fields (use 3-letter UPPERCASE driver codes; compounds in English
UPPERCASE: SOFT/MEDIUM/HARD/INTERMEDIATE/WET):
- pit_lap: {"driver": str, "lap": int}              (a driver pitted on a lap)
- compound_change: {"driver": str, "from": str, "to": str}
- n_stops: {"driver": str, "n": int}                (number of pit stops)
- stint_compound: {"driver": str, "compound": str}  (driver ran this compound)
- final_position: {"driver": str, "position": int}
- battle: {"kind": "undercut"|"overcut", "attacker": str, "defender": str}
- battle_outcome: {"success": true|false}           (the move worked or not)
- gain: {"value": float}                            (seconds gained)
Only include claims actually asserted in the text. No commentary."""


def _client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        max_retries=4, timeout=60.0,   # avoid hung connections blocking a worker forever
    )


_CLIENT = None
_DEPLOYMENT = os.environ.get("EXTRACTOR_DEPLOYMENT", "gpt-54-mini")


def llm_extract(text: str) -> list[Claim]:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _client()
    msgs = [
        {"role": "system", "content": "You are a precise information extractor."},
        {"role": "user", "content": f"{_SCHEMA}\n\nTEXT:\n{text}"},
    ]
    try:
        resp = _CLIENT.chat.completions.create(
            model=_DEPLOYMENT, messages=msgs,
            response_format={"type": "json_object"}, max_completion_tokens=2000)
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return []
    out: list[Claim] = []
    for c in data.get("claims", []):
        if isinstance(c, dict) and "type" in c and isinstance(c.get("fields"), dict):
            out.append(Claim(c["type"], c["fields"], span=""))
    return out
