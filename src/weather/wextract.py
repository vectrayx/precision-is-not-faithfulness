"""Language-agnostic weather-claim extraction via an LLM (Azure OpenAI / AIServices).

Returns the same Claim schema as the F1 extractor but with weather claim types, so the
same precision/recall machinery applies. Reads creds + EXTRACTOR_DEPLOYMENT from env.
"""
from __future__ import annotations

import json
import os

from src.eval.claims import Claim

_SCHEMA = """Extract every atomic, checkable weather claim from the forecast text.
Output JSON: {"claims": [ {"type": ..., "fields": {...}}, ... ]}.
Allowed types and fields (numbers as plain integers; no units in values):
- temp:       {"kind": "high"|"low", "value": int}   (a stated temperature, in the text's unit)
- wind:       {"value": int}                          (wind speed, mph)
- wind_dir:   {"dir": str}                            (compass direction, e.g. NW or northwest)
- precip_prob:{"value": int}                          (chance of precipitation, percent)
- sky:        {"condition": str}                      (sunny/clear/partly cloudy/cloudy/rain/snow/thunderstorms/fog)
Only include claims actually asserted in the text. Do not invent values. No commentary."""


def _client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        max_retries=4, timeout=60.0,
    )


_CLIENT = None
_DEPLOYMENT = os.environ.get("EXTRACTOR_DEPLOYMENT", "gpt-54-mini")


def weather_extract(text: str) -> list[Claim]:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _client()
    msgs = [{"role": "system", "content": "You are a precise information extractor."},
            {"role": "user", "content": f"{_SCHEMA}\n\nTEXT:\n{text}"}]
    try:
        resp = _CLIENT.chat.completions.create(
            model=_DEPLOYMENT, messages=msgs,
            response_format={"type": "json_object"}, max_completion_tokens=1500)
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return []
    out = []
    for c in data.get("claims", []):
        if isinstance(c, dict) and "type" in c and isinstance(c.get("fields"), dict):
            out.append(Claim(c["type"], c["fields"], span=""))
    return out
