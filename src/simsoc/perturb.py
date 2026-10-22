"""Prompt-perturbation families for Idea B.

Four components, five semantically-equivalent wordings each:
  SYSTEM   : persona framing (how the model is told to be the person)
  TASK     : the instruction for each design (poll / choice)
  ARMS     : the six arm descriptions (paraphrased once by Sonnet, cached, human-reviewed)
  FORMAT   : the output-format instruction
A 5x5 Graeco-Latin square gives 25 wordings in which every level of every component appears 5 times
and each pair of components is balanced.
"""
from __future__ import annotations

import json
from pathlib import Path

from .providers import complete

ROOT = Path(__file__).resolve().parents[2]
ARM_CACHE = ROOT / "data" / "arm_paraphrases.json"

SYSTEM = [
    # 0 = original
    "You are taking part in a consumer research interview. Stay fully in character as the person described. "
    "Answer as that person would in private, including indifference, scepticism or irritation. Do not mention being an AI.",
    "Role-play the person below for a market research study. Respond exactly as they would when nobody is watching: "
    "bored, cynical or annoyed if that is how they would feel. Never break character.",
    "Below is a transcript from someone we interviewed. From now on you are that person. Give their honest, private "
    "reaction, not a polished one. Stay in character throughout.",
    "Adopt the identity of the individual described here and answer every question as them, in their voice, with their "
    "habits and attitudes, including apathy where it fits. No references to being a model.",
    "You will answer as the interviewee whose words appear below. Be that person: their priorities, their mood, their "
    "blind spots. Candid, not performative. Remain in character.",
]

CHOICE_TASK = [
    "You are at a vending machine and you are going to have a snack now. These are the twelve options:\n{options}\n\n"
    "Labelling in this machine: {arm}\n\nPick the ONE snack you would actually take right now, as yourself, given your "
    "real tastes, habits and mood. Do not try to be healthy or make a point. One sentence of gut reaction, then {fmt}",
    "It is mid-afternoon and you are standing at a vending machine, about to buy a snack. The machine holds:\n{options}\n\n"
    "How the snacks are labelled: {arm}\n\nWhich one do you actually buy, being completely honest about what you would "
    "really pick today? Give one sentence of reaction, then {fmt}",
    "Snack time. Here is the vending machine in front of you:\n{options}\n\nLabels on the products: {arm}\n\n"
    "Choose the single item you would take, as the person you are, with your usual habits. Say one sentence about why, then {fmt}",
    "You have a few coins and you are going to get something from this vending machine right now. Options:\n{options}\n\n"
    "Note on labelling: {arm}\n\nWhat do you pick? Answer as yourself, truthfully, not aspirationally. One short sentence, then {fmt}",
    "Imagine you are at work and you fancy a snack from the machine in the corridor. It contains:\n{options}\n\n"
    "The labelling scheme in use: {arm}\n\nName the one snack you would really choose, given who you are. One line of reasoning, then {fmt}",
]

POLL_TASK = [
    "Below are {n} alternative public messages about food that you might see on a poster, a social media post, or a food "
    "label. Read them as yourself, not as an expert.\n\n{messages}\n\nTask: you have 100 points. Allocate them across the "
    "messages in proportion to how much each one would ACTUALLY change what you buy or eat over the next month, given your "
    "real life, budget and habits. A message you would scroll past gets few or zero points. Be honest; nobody is judging you. "
    "{hint}\n\nThink for two or three sentences about your gut reaction first, then {fmt}",
    "Here are {n} different ways food could be labelled or promoted to you.\n\n{messages}\n\nSplit 100 points between them "
    "according to how much each would genuinely shift your actual buying and eating over the coming month. Give nothing to "
    "ones you would ignore. Honest answers only. {hint}\n\nA couple of sentences of reaction, then {fmt}",
    "You are shown {n} labelling or messaging approaches.\n\n{messages}\n\nDistribute 100 points across them to reflect how "
    "much each would truly alter what you eat or buy in the next few weeks, as the person you are, with your budget and "
    "routines. {hint}\n\nBriefly say what strikes you, then {fmt}",
    "Consider these {n} options for how snacks could be labelled or advertised.\n\n{messages}\n\nYou have 100 points. Give "
    "more to the options that would really change your behaviour over the next month and fewer or none to those that would "
    "wash over you. Answer for yourself, not for an ideal version of yourself. {hint}\n\nTwo or three sentences of thought, then {fmt}",
    "{n} food-messaging options are listed below.\n\n{messages}\n\nAllocate 100 points in proportion to the real effect each "
    "would have on what you actually buy and eat this month, considering your true habits and finances. Zero is fine for "
    "ones you would not notice. {hint}\n\nReact in a sentence or two, then {fmt}",
]

CHOICE_FMT = [
    'output JSON exactly like {"choice": <number 1-12>}',
    'finish with a JSON object of the form {"choice": N} where N is the option number.',
    'end your answer with {"choice": <option number>} on its own line.',
    'give your final pick as JSON: {"choice": <1-12>}.',
    'close with a single line of JSON, {"choice": <number>}, naming the option you take.',
]

POLL_FMT = [
    'output a JSON object on the last line in exactly this form: {"points": {"A": <int>, "B": <int>, ...}, "most_ignored": "<letter>", "why": "<one sentence>"}',
    'finish with one line of JSON: {"points": {"A": <int>, "B": <int>, ...}, "most_ignored": "<letter>", "why": "<one sentence>"}',
    'end with a JSON object giving your allocation, like {"points": {"A": .., "B": .., ...}, "most_ignored": "<letter>", "why": "<sentence>"}.',
    'give your allocation as JSON on the final line: {"points": {"A": <int>, ...}, "most_ignored": "<letter>", "why": "<one sentence>"}.',
    'your last line must be JSON of the form {"points": {"A": <int>, "B": <int>, ...}, "most_ignored": "<letter>", "why": "<one sentence>"}.',
]


def design_25() -> list[tuple[int, int, int, int]]:
    """Graeco-Latin square: (system, task, arms, format) levels for 25 runs."""
    runs = []
    for i in range(5):
        for j in range(5):
            runs.append((i, j, (i + j) % 5, (i + 2 * j) % 5))
    return runs


def arm_paraphrases(messages: dict[str, str], model: str = "claude-sonnet-5-5") -> list[dict[str, str]]:
    """Level 0 = original. Levels 1-4 = paraphrases generated once and cached for review."""
    if ARM_CACHE.exists():
        return json.loads(ARM_CACHE.read_text())
    out = [messages]
    styles = [
        "plainer and shorter, like a shop notice",
        "slightly more formal, as a researcher would describe the condition",
        "in the voice of a friend describing what they saw",
        "with the sentences reordered and different but equivalent vocabulary",
    ]
    for s in styles:
        system = "You rewrite experimental stimulus descriptions. Preserve every factual element exactly: which products carry labels, whether pictures are present, the exact warning wording in quotes, whether calorie badges appear and on which products. Change only the surrounding phrasing."
        user = (f"Rewrite each of these six descriptions {s}. Keep every fact. Return JSON with the same keys and the "
                f"rewritten strings as values, nothing else.\n\n{json.dumps(messages, indent=1)}")
        text = complete(model, system, user, temperature=1.0, max_tokens=1500, seed=0)
        start, end = text.find("{"), text.rfind("}")
        d = json.loads(text[start:end + 1])
        assert set(d) == set(messages), d.keys()
        out.append(d)
    ARM_CACHE.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out
