"""Reference-free faithfulness scoring for generated F1 strategy explanations.

Pipeline: extract atomic claims -> verify each against the instance ground truth ->
aggregate. The headline metric is the supported fraction; the hard-hallucination rate
(contradicted fraction) is reported alongside.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .claims import Claim, regex_extract
from .verify import verify_claim, SUPPORTED, CONTRADICTED, UNVERIFIABLE

Extractor = Callable[[str], list[Claim]]


@dataclass
class FaithfulnessResult:
    n_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    faithfulness: float        # supported / n_claims  (primary)
    hallucination_rate: float  # contradicted / n_claims  (hard errors)
    precision: float           # supported / (supported + contradicted)
    claims: list[dict] = field(default_factory=list)


def score_text(text: str, ground_truth: dict,
               extractor: Extractor = regex_extract) -> FaithfulnessResult:
    claims = extractor(text)
    sup = con = unv = 0
    rows = []
    for c in claims:
        label, reason = verify_claim(c, ground_truth)
        c.label, c.reason = label, reason
        sup += label == SUPPORTED
        con += label == CONTRADICTED
        unv += label == UNVERIFIABLE
        rows.append({"type": c.type, "fields": c.fields, "label": label,
                     "reason": reason, "span": c.span})
    n = len(claims)
    return FaithfulnessResult(
        n_claims=n,
        supported=sup, contradicted=con, unverifiable=unv,
        faithfulness=(sup / n if n else 0.0),
        hallucination_rate=(con / n if n else 0.0),
        precision=(sup / (sup + con) if (sup + con) else 0.0),
        claims=rows,
    )
