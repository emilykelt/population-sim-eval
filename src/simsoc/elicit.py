"""Elicitation methods. Each returns, per persona, a dict message_id -> score, plus an optional ranking.

Methods (all take the same message set so they are directly comparable):
  judge          : no persona, model predicts how the target segment would respond (LLM-judge baseline)
  label_single   : demographic-label persona, one sample
  grounded_single: interview-grounded persona, one sample
  grounded_ensemble: many grounded personas per cell, aggregated (the "silicon crowd")
  grounded_debate: grounded persona answers, a sceptical peer from the same community pushes back,
                   persona revises. Adversarial, not consensus, by design (see 01_SYNTHESIS.md §4.4).

Elicitation is RELATIVE and behavioural: the persona allocates 100 points across the variants according to
how much each would actually change what they do, then names the one they'd most ignore. This avoids the
"rate this health message 1-5" social-desirability trap (Salecha 2024; Sun 2024).
"""
from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .personas import Persona
from .providers import complete


@dataclass
class Message:
    id: str
    text: str


def _format_messages(msgs: list[Message], order: list[int]) -> str:
    return "\n\n".join(f"[{chr(65 + k)}] {msgs[i].text}" for k, i in enumerate(order))


def _parse_points(text: str, n: int) -> list[float] | None:
    d = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
    if d is None:
        m = re.search(r'"points"\s*:\s*(\{[^{}]*\})', text)
        if not m:
            return None
        try:
            d = {"points": json.loads(m.group(1))}
        except json.JSONDecodeError:
            return None
    pts = d.get("points") or d.get("allocation")
    if not isinstance(pts, dict):
        return None
    out = []
    for k in range(n):
        v = pts.get(chr(65 + k))
        if v is None:
            return None
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    s = sum(out)
    return [100 * x / s for x in out] if s > 0 else None


TASK = (
    "Below are {n} alternative public messages about food that you might see on a poster, a social media post, "
    "or a food label. Read them as yourself, not as an expert.\n\n{messages}\n\n"
    "Task: you have 100 points. Allocate them across the messages in proportion to how much each one would "
    "ACTUALLY change what you buy or eat over the next month, given your real life, budget and habits. A message "
    "you would scroll past gets few or zero points. Be honest; nobody is judging you. {outcome_hint}\n\n"
    "Think for two or three sentences about your gut reaction first, then output a JSON object on the last line "
    'in exactly this form: {{"points": {{"A": <int>, "B": <int>, ...}}, "most_ignored": "<letter>", "why": "<one sentence>"}}'
)


def _persona_system(p: Persona, grounded: bool) -> str:
    base = (
        "You are taking part in a consumer research interview. Stay fully in character as the person described. "
        "Answer as that person would in private, including indifference, scepticism or irritation. "
        "Do not mention being an AI."
    )
    if grounded and p.backstory:
        return base + "\n\nWho you are, in your own words from an earlier interview:\n" + p.backstory
    return base + "\n\n" + p.label_text()


def elicit_persona(
    p: Persona, msgs: list[Message], model: str, *, grounded: bool, order: list[int], outcome_hint: str = "", seed: int = 0
) -> dict:
    system = _persona_system(p, grounded)
    user = TASK.format(n=len(msgs), messages=_format_messages(msgs, order), outcome_hint=outcome_hint)
    text = complete(model, system, user, temperature=1.0, max_tokens=900, seed=seed)
    pts = _parse_points(text, len(msgs))
    return {"persona": p.id, "cell": p.cell, "order": order, "raw": text, "points": pts}


def elicit_debate(
    p: Persona, msgs: list[Message], model: str, *, order: list[int], outcome_hint: str = "", seed: int = 0
) -> dict:
    """Round 1: persona answers. Round 2: a sceptical peer critiques. Round 3: persona revises."""
    first = elicit_persona(p, msgs, model, grounded=True, order=order, outcome_hint=outcome_hint, seed=seed)
    critic_system = (
        "You are a friend of the interviewee, from the same community and similar circumstances. You are blunt. "
        "Your job is to challenge their answer: point out where they are being idealistic about their own habits, "
        "where a message would actually bounce off people like you both, and where they have underrated something "
        "that would genuinely land. Do not be polite. Four sentences maximum."
    )
    critic_user = (
        f"Here is who they are:\n{p.backstory}\n\nHere are the messages:\n{_format_messages(msgs, order)}\n\n"
        f"Here is their answer:\n{first['raw']}\n\nChallenge it."
    )
    critique = complete(model, critic_system, critic_user, temperature=1.0, max_tokens=250, seed=seed)
    revise_user = (
        TASK.format(n=len(msgs), messages=_format_messages(msgs, order), outcome_hint=outcome_hint)
        + f"\n\nYou answered earlier:\n{first['raw']}\n\nA friend who knows you well said:\n{critique}\n\n"
        "Reconsider. Keep what you still believe, change what you now think was idealistic. Output the same JSON form."
    )
    text = complete(model, _persona_system(p, True), revise_user, temperature=1.0, max_tokens=900, seed=seed)
    pts = _parse_points(text, len(msgs))
    return {
        "persona": p.id, "cell": p.cell, "order": order, "raw": text, "points": pts,
        "round1_points": first["points"], "critique": critique,
    }


def elicit_judge(cell: dict, msgs: list[Message], model: str, *, order: list[int], outcome_hint: str = "", seed: int = 0) -> dict:
    system = "You are an expert in public health communication and consumer behaviour."
    cell_desc = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in cell.items())
    user = (
        f"Target audience: {cell_desc}.\n\nMessages:\n{_format_messages(msgs, order)}\n\n"
        f"Predict how this audience would respond. Allocate 100 points across the messages in proportion to how much "
        f"each would change what the average member of this audience buys or eats over the next month. {outcome_hint}\n"
        'Reason briefly, then output JSON on the last line: {"points": {"A": <int>, ...}, "most_ignored": "<letter>", "why": "<one sentence>"}'
    )
    text = complete(model, system, user, temperature=1.0, max_tokens=900, seed=seed)
    return {"persona": None, "cell": cell, "order": order, "raw": text, "points": _parse_points(text, len(msgs))}


def run_many(fn, jobs: list[dict], workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda kw: fn(**kw), jobs))


SNACKS = [
    ("Doritos Tangy Cheese (crisps)", True), ("apple crisps", False), ("Galaxy chocolate bar", True),
    ("Nim's beetroot crisps", False), ("Müller Corner yoghurt", False), ("McCoy's salt and vinegar crisps", True),
    ("apple and grapes fruit pot", False), ("millionaire shortbread", True), ("Cadbury Dairy Milk bar", True),
    ("mango and pineapple fruit pot", False), ("Gü chocolate dessert pot", True), ("KIND nut bar", False),
]

CHOICE_TASK = (
    "You are at a vending machine and you are going to have a snack now. These are the twelve options:\n{options}\n\n"
    "Labelling in this machine: {arm}\n\n"
    "Pick the ONE snack you would actually take right now, as yourself, given your real tastes, habits and mood. "
    "Do not try to be healthy or make a point. One sentence of gut reaction, then on the last line output JSON "
    'exactly like {{"choice": <number 1-12>}}'
)


CHOICE_TASK_NEUTRAL = CHOICE_TASK.replace("Do not try to be healthy or make a point. ", "")


def elicit_choice(p: Persona, arm_id: str, arm_text: str, model: str, *, grounded: bool = True, seed: int = 0, variant: str = "default") -> dict:
    """Between-subjects replication of a labelling experiment: persona sees one arm, picks one snack.
    variant="neutral" drops the 'do not try to be healthy' instruction (prompt-sensitivity ablation)."""
    rng = random.Random(f"{p.id}-{arm_id}-{seed}")
    idx = list(range(len(SNACKS)))
    rng.shuffle(idx)
    options = "\n".join(f"{k + 1}. {SNACKS[i][0]}" for k, i in enumerate(idx))
    task = CHOICE_TASK_NEUTRAL if variant == "neutral" else CHOICE_TASK
    user = task.format(options=options, arm=arm_text)
    text = complete(model, _persona_system(p, grounded), user, temperature=1.0, max_tokens=300, seed=seed)
    m = re.search(r'"choice"\s*:\s*"?(\d+)', text)
    choice = int(m.group(1)) if m else None
    energy_dense = SNACKS[idx[choice - 1]][1] if choice and 1 <= choice <= 12 else None
    return {"persona": p.id, "cell": p.cell, "arm": arm_id, "raw": text, "choice": choice, "energy_dense": energy_dense}
