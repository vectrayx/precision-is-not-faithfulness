"""Atomic claim schema and extraction for the faithfulness metric.

A generated explanation is decomposed into typed atomic claims, each of which the
verifier can check against the structured ground truth. Two extraction backends:

  - `regex_extract`: deterministic, dependency-free. Handles the constrained
    phrasings used by template baselines plus common natural phrasings. Used for the
    offline pilot and unit tests.
  - LLM-based extraction lives in `llm_extract.py` (optional, needs an API key) and
    returns the same Claim schema for free-form model outputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Claim types the verifier understands.
PIT_LAP = "pit_lap"
COMPOUND_CHANGE = "compound_change"
N_STOPS = "n_stops"
STINT_COMPOUND = "stint_compound"
FINAL_POSITION = "final_position"
BATTLE = "battle"               # X undercut/overcut Y
BATTLE_OUTCOME = "battle_outcome"  # the move worked / failed
GAIN = "gain"                   # gained N seconds
DEFENSE = "defense"             # X held up / defended against Y
WINNER = "winner"               # X won the race


@dataclass
class Claim:
    type: str
    fields: dict
    span: str                       # source text the claim came from
    label: Optional[str] = None     # filled by verifier: supported|contradicted|unverifiable
    reason: Optional[str] = None


_COMPOUNDS = {
    "soft": "SOFT", "softs": "SOFT",
    "medium": "MEDIUM", "mediums": "MEDIUM",
    "hard": "HARD", "hards": "HARD",
    "intermediate": "INTERMEDIATE", "intermediates": "INTERMEDIATE", "inters": "INTERMEDIATE",
    "wet": "WET", "wets": "WET",
}
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "zero": 0,
              "1": 1, "2": 2, "3": 3, "4": 4}
# Driver codes are always uppercase. Scope case-sensitivity to this group so that
# re.IGNORECASE on the surrounding pattern (for compounds) does not match lowercase
# words like "two" as a driver code.
_DRIVER = r"(?-i:\b([A-Z]{3})\b)"
_COMP = r"(soft|medium|hard|intermediate|wet)s?"


def normalize_compound(s: str) -> Optional[str]:
    return _COMPOUNDS.get(s.strip().lower())


_PRONOUN = re.compile(r"\b(?:He|She|he|she)\b")
_CODE = re.compile(r"(?-i:\b[A-Z]{3}\b)")


def _resolve_coref(text: str) -> str:
    """Light coref: a sentence with no driver code inherits the most recently
    mentioned code for its pronoun subjects. Best-effort; the LLM extractor handles
    harder cases. Conservative: only fires when the sentence names no driver itself.
    """
    out, last = [], None
    for sent in re.split(r"(?<=[.;])\s+", text):
        codes = _CODE.findall(sent)
        if not codes and last:
            sent = _PRONOUN.sub(last, sent)
        if codes:
            last = codes[-1]
        out.append(sent)
    return " ".join(out)


def regex_extract(text: str) -> list[Claim]:
    """Extract typed atomic claims from free text using surface patterns."""
    claims: list[Claim] = []
    t = _resolve_coref(text)

    # "X pitted/stopped/boxed on lap N"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(?:pit(?:ted|s|ted in)?|stopped|boxed)\b[^.]*?\blap\s+(\d+)", t):
        claims.append(Claim(PIT_LAP, {"driver": m.group(1), "lap": int(m.group(2))}, m.group(0)))

    # "X switched/changed/went from MEDIUM to HARD" (and "from mediums to hards")
    for m in re.finditer(_DRIVER + r"[^.]*?\bfrom\s+" + _COMP + r"\s+to\s+" + _COMP, t, re.I):
        claims.append(Claim(COMPOUND_CHANGE, {
            "driver": m.group(1),
            "from": normalize_compound(m.group(2)),
            "to": normalize_compound(m.group(3)),
        }, m.group(0)))

    # "X made N pit stops" / "X made two stops"
    for m in re.finditer(_DRIVER + r"[^.]*?\bmade\s+(\w+)\s+(?:pit\s+)?stops?\b", t, re.I):
        n = _NUM_WORDS.get(m.group(2).lower())
        if n is not None:
            claims.append(Claim(N_STOPS, {"driver": m.group(1), "n": n}, m.group(0)))

    # "X ran a one-stop / two-stop(per) (strategy)"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(one|two|three|1|2|3)[-\s]stop(?:per)?\b", t, re.I):
        claims.append(Claim(N_STOPS, {"driver": m.group(1), "n": _NUM_WORDS[m.group(2).lower()]}, m.group(0)))

    # "X used/started on the HARD tyre"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(?:used|ran|started on|started)\s+(?:the\s+)?" + _COMP + r"\s+(?:tyres?|compound)?", t, re.I):
        comp = normalize_compound(m.group(2))
        if comp:
            claims.append(Claim(STINT_COMPOUND, {"driver": m.group(1), "compound": comp}, m.group(0)))

    # "X finished P5" / "finished in P5" / "finished fifth/5th"
    for m in re.finditer(_DRIVER + r"[^.]*?\bfinished\b[^.]*?\bP?(\d{1,2})(?:st|nd|rd|th)?\b", t, re.I):
        claims.append(Claim(FINAL_POSITION, {"driver": m.group(1), "position": int(m.group(2))}, m.group(0)))

    # "X undercut/overcut Y"
    for m in re.finditer(_DRIVER + r"\s+(?:successfully\s+|tried to\s+)?(undercut|overcut)\b[^.]*?" + _DRIVER, t, re.I):
        claims.append(Claim(BATTLE, {
            "kind": m.group(2).lower(), "attacker": m.group(1), "defender": m.group(3),
        }, m.group(0)))

    # outcome: "the undercut/overcut/move worked|paid off|failed|did not work"
    for m in re.finditer(r"\b(undercut|overcut|move|strategy)\b[^.]*?\b(worked|paid off|succeeded|failed|did not work|didn't work|backfired)\b", t, re.I):
        ok = m.group(2).lower() in {"worked", "paid off", "succeeded"}
        claims.append(Claim(BATTLE_OUTCOME, {"success": ok}, m.group(0)))

    # "gained N seconds" / "gained N.N s"
    for m in re.finditer(r"\bgain(?:ed|ing|s)?\s+(?:about\s+|~)?(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", t, re.I):
        claims.append(Claim(GAIN, {"value": float(m.group(1))}, m.group(0)))

    # "X held up / held off / defended (against/from) Y"  (EN)
    for m in re.finditer(_DRIVER + r"\s+(?:held(?:\s+up|\s+off)?|defended(?:\s+against|\s+from)?|kept behind)\s+(?:from\s+)?" + _DRIVER, t):
        claims.append(Claim(DEFENSE, {"defender": m.group(1), "pursuer": m.group(2)}, m.group(0)))

    # "X won (the race)" (EN)
    for m in re.finditer(_DRIVER + r"\s+won\b", t):
        claims.append(Claim(WINNER, {"driver": m.group(1)}, m.group(0)))

    claims += _extract_es_pt(t)
    return _dedupe(claims)


# Spanish/Portuguese number words for stop counts.
_NUM_ES_PT = {"una": 1, "uno": 1, "uma": 1, "dos": 2, "duas": 2, "tres": 3,
              "três": 3, "1": 1, "2": 2, "3": 3}


def _extract_es_pt(t: str) -> list[Claim]:
    """Surface patterns for Spanish/Portuguese briefings. Compound names stay in
    English (SOFT/MEDIUM/HARD) as in the structured data, even inside ES/PT text."""
    cs: list[Claim] = []

    # pit lap: "par(ó|ou)/box(eó)/entró ... vuelta/volta N"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(?:par[oó]|parou|box|entr[oó])\w*[^.]*?\b(?:vuelta|volta)\s+(\d+)", t, re.I):
        cs.append(Claim(PIT_LAP, {"driver": m.group(1), "lap": int(m.group(2))}, m.group(0)))

    # compound change: "de X a/para Y"
    for m in re.finditer(_DRIVER + r"[^.]*?\bde\s+" + _COMP + r"\s+(?:a|para)\s+" + _COMP, t, re.I):
        cs.append(Claim(COMPOUND_CHANGE, {"driver": m.group(1),
                  "from": normalize_compound(m.group(2)), "to": normalize_compound(m.group(3))}, m.group(0)))

    # n stops: "a/de una/dos/... parada(s)"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(una|uno|uma|dos|duas|tres|três|\d)\s+parad", t, re.I):
        n = _NUM_ES_PT.get(m.group(2).lower())
        if n is not None:
            cs.append(Claim(N_STOPS, {"driver": m.group(1), "n": n}, m.group(0)))

    # stint compound: "us[oó]/empez[oó]/comeou con (el) X" or "con el neumático X"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(?:us[oó]|empez[oó]|come[cç]ou|começou|rod[oó]|con|com)\b[^.]*?(?:neumático|pneu)?\s*" + _COMP, t, re.I):
        comp = normalize_compound(m.group(2))
        if comp:
            cs.append(Claim(STINT_COMPOUND, {"driver": m.group(1), "compound": comp}, m.group(0)))

    # final position: "termin(ó|ou)/finaliz(ó)/acab(ó) ... P?N"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(?:termin[oó]|terminou|finaliz[oó]|acab[oó]|acabou)\b[^.]*?P?(\d{1,2})\b", t, re.I):
        cs.append(Claim(FINAL_POSITION, {"driver": m.group(1), "position": int(m.group(2))}, m.group(0)))

    # battle: "X ... undercut/overcut ... Y" (broad, ES/PT word order)
    for m in re.finditer(_DRIVER + r"[^.]*?\b(undercut|overcut)\b[^.]{0,40}?" + _DRIVER, t, re.I):
        cs.append(Claim(BATTLE, {"kind": m.group(2).lower(), "attacker": m.group(1), "defender": m.group(3)}, m.group(0)))

    # outcome: "(no )?funcion(ó|ou)"
    for m in re.finditer(r"\b(undercut|overcut|movimiento|jogada|estrat[eé]gia)\b[^.]*?\b(no\s+funcion[oó]|n[aã]o\s+funcionou|funcion[oó]|funcionou)\b", t, re.I):
        ok = not m.group(2).lower().startswith(("no", "não", "nao"))
        cs.append(Claim(BATTLE_OUTCOME, {"success": ok}, m.group(0)))

    # gain: "gan(ó|ando)/ganh(ou|ando) N segundos"
    for m in re.finditer(r"\b(?:gan[oó]|ganando|ganhou|ganhando)\s+(?:cerca de\s+|~)?(\d+(?:\.\d+)?)\s*(?:segundos?|s)\b", t, re.I):
        cs.append(Claim(GAIN, {"value": float(m.group(1))}, m.group(0)))

    # defense: "X contuvo/aguantó/defendió ... Y" / "X conteve/segurou/defendeu Y"
    for m in re.finditer(_DRIVER + r"[^.]*?\b(?:contuvo|aguant[oó]|defendi[oó]|conteve|segurou|defendeu|retuvo)\b[^.]{0,30}?" + _DRIVER, t, re.I):
        cs.append(Claim(DEFENSE, {"defender": m.group(1), "pursuer": m.group(2)}, m.group(0)))

    # winner: "X gan[oó]/venci[oó]/venceu/ganhou"
    for m in re.finditer(_DRIVER + r"\s+(?:gan[oó]|venci[oó]|venceu|ganhou)\b", t, re.I):
        cs.append(Claim(WINNER, {"driver": m.group(1)}, m.group(0)))

    return cs


def _dedupe(claims: list[Claim]) -> list[Claim]:
    seen, out = set(), []
    for c in claims:
        key = (c.type, tuple(sorted((k, str(v)) for k, v in c.fields.items())))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
