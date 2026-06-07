"""Explanation generators (baselines) for the benchmark.

A generator maps (instance, lang) -> explanation text. Backends:

  - `template_faithful` / `template_noisy`: deterministic, offline. The faithful one
    emits only true statements from the ground truth; the noisy one injects controlled
    perturbations. Together they form the metric-validation experiment that runs with
    no API key or GPU.
  - `azure_openai`: frontier baseline via Azure OpenAI (needs env vars + credit).
  - `openai_compatible`: any OpenAI-compatible endpoint (e.g. a Qwen/Gemma model
    served with vLLM on the GCP GPU VM).

All real-LLM backends share the same grounded system prompt and respect the prompt
language (en/es/pt) for RQ2.
"""
from __future__ import annotations

import os
import random
from typing import Callable

Generator = Callable[[dict, str], str]

SYSTEM_PROMPT = (
    "You are an F1 strategy analyst. Explain the strategic decision using ONLY the "
    "data provided. Do not invent laps, compounds, gaps, positions, or outcomes that "
    "are not in the data. Be concise."
)


def _user_prompt(inst: dict, lang: str) -> str:
    return f"{inst['prompts'][lang]}\n\nData:\n{inst['context_text']}"


def chat(client, model: str, messages: list) -> str:
    """Call chat.completions robustly across model families.

    Newer (reasoning) models require `max_completion_tokens` and reject a custom
    `temperature`; older models use `max_tokens` + `temperature`. Try the modern
    signature first, fall back to the legacy one.
    """
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=2000)
    except Exception:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=400, temperature=0.3)
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------
# Offline template baselines (for metric validation)
# --------------------------------------------------------------------------

def _stint_sentences(gt: dict, lang: str = "en") -> list[str]:
    drv = gt["driver"]
    stints = sorted(gt["stints"], key=lambda x: x["stint"])
    c0 = stints[0]["compound"] if stints else None
    pos = gt.get("final_position")
    s = []
    if lang == "es":
        s.append(f"{drv} hizo una estrategia a {gt['n_stops']} parada(s).")
        if c0: s.append(f"{drv} empezó con el neumático {c0}.")
        for p in gt["pit_stops"]:
            s.append(f"{drv} paró en la vuelta {p['lap']}, cambiando de {p['from_compound']} a {p['to_compound']}.")
        if pos: s.append(f"{drv} terminó P{pos}.")
    elif lang == "pt":
        s.append(f"{drv} fez uma estratégia de {gt['n_stops']} parada(s).")
        if c0: s.append(f"{drv} começou com o pneu {c0}.")
        for p in gt["pit_stops"]:
            s.append(f"{drv} parou na volta {p['lap']}, mudando de {p['from_compound']} para {p['to_compound']}.")
        if pos: s.append(f"{drv} terminou P{pos}.")
    else:
        s.append(f"{drv} ran a {gt['n_stops']}-stop strategy.")
        if c0: s.append(f"{drv} started on the {c0} tyre.")
        for p in gt["pit_stops"]:
            s.append(f"{drv} pitted on lap {p['lap']}, switching from {p['from_compound']} to {p['to_compound']}.")
        if pos: s.append(f"{drv} finished P{pos}.")
    return s


def _battle_sentences(gt: dict, lang: str = "en") -> list[str]:
    a, d, k = gt["attacker"], gt["defender"], gt["kind"]
    worked = bool(gt.get("position_swap") or (gt.get("gained_s") or 0) > 0)
    al, dl, g = gt["attacker_pit_lap"], gt["defender_pit_lap"], gt.get("gained_s")
    if lang == "es":
        s = [f"{a} le hizo el {k} a {d}.",
             f"{a} paró en la vuelta {al} y {d} en la vuelta {dl}."]
        if g is not None:
            s.append(f"El {k} {'funcionó' if worked else 'no funcionó'}, ganando {abs(g):.1f} segundos.")
    elif lang == "pt":
        s = [f"{a} fez o {k} em {d}.",
             f"{a} parou na volta {al} e {d} na volta {dl}."]
        if g is not None:
            s.append(f"O {k} {'funcionou' if worked else 'não funcionou'}, ganhando {abs(g):.1f} segundos.")
    else:
        s = [f"{a} {k} {d}.",
             f"{a} pitted on lap {al} and {d} on lap {dl}."]
        if g is not None:
            s.append(f"The {k} {'worked' if worked else 'did not work'}, gaining {abs(g):.1f} seconds.")
    return s


def _defense_sentences(gt: dict, lang: str = "en") -> list[str]:
    D, P, n = gt["defender"], gt["pursuer"], gt["n_laps"]
    tm = gt.get("teammate_protected")
    if lang == "es":
        s = [f"{D} contuvo a {P} durante {n} vueltas.",
             f"{P} era más rápido en ritmo pero quedó atascado detrás de {D}."]
        if tm: s.append(f"{D} protegió a un compañero de equipo por delante.")
    elif lang == "pt":
        s = [f"{D} segurou {P} por {n} voltas.",
             f"{P} era mais rápido no ritmo mas ficou preso atrás de {D}."]
        if tm: s.append(f"{D} protegeu um companheiro de equipe à frente.")
    else:
        s = [f"{D} held up {P} for {n} laps.",
             f"{P} was faster on pace but stuck behind {D}."]
        if tm: s.append(f"{D} protected a teammate ahead.")
    return s


def _race_summary_sentences(gt: dict, lang: str = "en") -> list[str]:
    cls = sorted(gt["classification"], key=lambda c: c["position"])[:3]
    w = gt["winner"]
    fl = gt.get("fastest_lap")
    if lang == "es":
        s = [f"{w} ganó la carrera."]
        s += [f"{c['driver']} terminó P{c['position']}." for c in cls]
        if fl: s.append(f"La vuelta rápida fue de {fl['driver']}.")
    elif lang == "pt":
        s = [f"{w} venceu a corrida."]
        s += [f"{c['driver']} terminou P{c['position']}." for c in cls]
        if fl: s.append(f"A volta mais rápida foi de {fl['driver']}.")
    else:
        s = [f"{w} won the race."]
        s += [f"{c['driver']} finished P{c['position']}." for c in cls]
        if fl: s.append(f"The fastest lap was set by {fl['driver']}.")
    return s


def commentate(inst: dict, text: str, lang: str = "en") -> str:
    """Wrap a grounded briefing in F1-commentator flair. The factual sentences are
    kept verbatim inside, so the faithfulness audit still verifies every claim."""
    gp = inst.get("gp", "Gran Premio")
    if lang == "es":
        intro = (f"🎙️ ¡Amigos de América, bienvenidos al {gp}, una nueva fecha del "
                 f"Campeonato Mundial de Fórmula 1! Vamos con el análisis:")
        outro = "¡Qué espectáculo, señoras y señores! ¡Nos vemos en la próxima fecha! 🏁"
    elif lang == "pt":
        intro = (f"🎙️ Amigos, sejam bem-vindos ao {gp} do Campeonato Mundial de "
                 f"Fórmula 1! Vamos à análise:")
        outro = "Que espetáculo, senhoras e senhores! Até a próxima etapa! 🏁"
    else:
        intro = (f"🎙️ Welcome, racing fans, to the {gp} of the Formula 1 World "
                 f"Championship! Here's the strategy breakdown:")
        outro = "What a show, ladies and gentlemen! See you next time out! 🏁"
    return f"{intro} {text} {outro}"


COMMENTATOR_PROMPT = (
    SYSTEM_PROMPT + " Narrate in the energetic style of a Latin-American F1 TV "
    "commentator (e.g. open with a warm welcome to the audience), but keep every "
    "factual claim strictly grounded in the data.")


def template_commentator(inst: dict, lang: str = "en") -> str:
    return commentate(inst, template_faithful(inst, lang), lang)


def template_faithful(inst: dict, lang: str = "en") -> str:
    gt = inst["ground_truth"]
    dt = inst["decision_type"]
    if dt == "stint_strategy":
        sents = _stint_sentences(gt, lang)
    elif dt in ("undercut", "overcut"):
        sents = _battle_sentences(gt, lang)
    elif dt == "defense":
        sents = _defense_sentences(gt, lang)
    elif dt == "race_summary":
        sents = _race_summary_sentences(gt, lang)
    else:
        sents = _stint_sentences(gt, lang)
    return " ".join(sents)


def template_noisy(inst: dict, lang: str = "en", rate: float = 0.5, seed: int = 0) -> str:
    """Faithful text with controlled factual perturbations injected at `rate`."""
    rng = random.Random(seed + hash(inst["id"]) % 10_000)
    gt = inst["ground_truth"]
    if inst["decision_type"] == "stint_strategy":
        drv = gt["driver"]
        n = gt["n_stops"] + (1 if rng.random() < rate else 0)              # maybe wrong stop count
        pits = []
        for p in gt["pit_stops"]:
            lap = p["lap"] + (rng.choice([-6, 5, 8]) if rng.random() < rate else 0)
            to = "SOFT" if rng.random() < rate else p["to_compound"]       # maybe wrong compound
            pits.append((lap, p["from_compound"], to))
        pos = (gt.get("final_position") or 1) + (3 if rng.random() < rate else 0)
        if lang == "es":
            parts = [f"{drv} hizo una estrategia a {n} parada(s)."]
            parts += [f"{drv} paró en la vuelta {l}, cambiando de {fr} a {to}." for l, fr, to in pits]
            parts.append(f"{drv} terminó P{pos}.")
        elif lang == "pt":
            parts = [f"{drv} fez uma estratégia de {n} parada(s)."]
            parts += [f"{drv} parou na volta {l}, mudando de {fr} para {to}." for l, fr, to in pits]
            parts.append(f"{drv} terminou P{pos}.")
        else:
            parts = [f"{drv} ran a {n}-stop strategy."]
            parts += [f"{drv} pitted on lap {l}, switching from {fr} to {to}." for l, fr, to in pits]
            parts.append(f"{drv} finished P{pos}.")
        return " ".join(parts)
    if inst["decision_type"] == "defense":
        D, P, n = gt["defender"], gt["pursuer"], gt["n_laps"]
        if rng.random() < rate:          # maybe swap who held whom
            D, P = P, D
        if rng.random() < rate:          # maybe wrong lap count
            n = n + rng.choice([-3, 4, 7])
        if lang == "es":
            return f"{D} contuvo a {P} durante {n} vueltas. {P} era más rápido pero quedó atascado."
        if lang == "pt":
            return f"{D} segurou {P} por {n} voltas. {P} era mais rápido mas ficou preso."
        return f"{D} held up {P} for {n} laps. {P} was faster but stuck behind."
    if inst["decision_type"] == "race_summary":
        cls = sorted(gt["classification"], key=lambda c: c["position"])[:3]
        w = gt["winner"]
        if rng.random() < rate and len(cls) > 1:   # maybe wrong winner
            w = cls[1]["driver"]
        if lang == "es":
            return f"{w} ganó la carrera. " + " ".join(f"{c['driver']} terminó P{c['position']}." for c in cls)
        if lang == "pt":
            return f"{w} venceu a corrida. " + " ".join(f"{c['driver']} terminou P{c['position']}." for c in cls)
        return f"{w} won the race. " + " ".join(f"{c['driver']} finished P{c['position']}." for c in cls)
    # battle
    a, d, k = gt["attacker"], gt["defender"], gt["kind"]
    flip = rng.random() < rate
    at, df = (d, a) if flip else (a, d)                                    # maybe wrong direction
    al, dl = gt["attacker_pit_lap"], gt["defender_pit_lap"]
    worked = bool(gt.get("position_swap") or (gt.get("gained_s") or 0) > 0)
    if rng.random() < rate:
        worked = not worked                                               # maybe wrong outcome
    gain = abs(gt.get("gained_s") or 0) + (rng.choice([4, 6]) if rng.random() < rate else 0)
    if lang == "es":
        parts = [f"{at} le hizo el {k} a {df}.",
                 f"{a} paró en la vuelta {al} y {d} en la vuelta {dl}.",
                 f"El {k} {'funcionó' if worked else 'no funcionó'}, ganando {gain:.1f} segundos."]
    elif lang == "pt":
        parts = [f"{at} fez o {k} em {df}.",
                 f"{a} parou na volta {al} e {d} na volta {dl}.",
                 f"O {k} {'funcionou' if worked else 'não funcionou'}, ganhando {gain:.1f} segundos."]
    else:
        parts = [f"{at} {k} {df}.",
                 f"{a} pitted on lap {al} and {d} on lap {dl}.",
                 f"The {k} {'worked' if worked else 'did not work'}, gaining {gain:.1f} seconds."]
    return " ".join(parts)


# --------------------------------------------------------------------------
# Real LLM backends
# --------------------------------------------------------------------------

def azure_openai(inst: dict, lang: str = "en") -> str:
    """Frontier baseline via Azure OpenAI. Requires:
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
        (optional) AZURE_OPENAI_API_VERSION (default 2024-08-01-preview)
    """
    from openai import AzureOpenAI  # lazy import
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    return chat(client, os.environ["AZURE_OPENAI_DEPLOYMENT"],
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": _user_prompt(inst, lang)}])


def openai_compatible(inst: dict, lang: str = "en") -> str:
    """Open-weights model via any OpenAI-compatible server (e.g. vLLM on GCP GPU).
    Requires: OPENAI_BASE_URL, OPENAI_API_KEY (any string for vLLM), OPENAI_MODEL.
    """
    from openai import OpenAI  # lazy import
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    return chat(client, os.environ["OPENAI_MODEL"],
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": _user_prompt(inst, lang)}])


REGISTRY: dict[str, Generator] = {
    "template_faithful": template_faithful,
    "template_noisy": template_noisy,
    "azure_openai": azure_openai,
    "openai_compatible": openai_compatible,
}
