"""Generic task execution for the eval: a twin is placed in ONE arm and produces an outcome on the same scale as
the human outcome column.

Task types (set in the eval YAML):
  choice      : pick one option from a list; outcome = property of the chosen option (e.g. 1 if lower-energy)
  menu_choice : several menus, pick one item from each; outcome = sum/mean of a property over picks (Evans: salt g)
  binary      : choose between two described products; outcome = 1 if the 'target' product chosen
  likert      : answer one or more scale items 1..k; outcome = mean (reverse-coded items handled in YAML)

The arm text is injected where the YAML puts {arm}. Option order is randomised per call. Prompts avoid honesty
admonitions (see docs/07: they move the base rate more than the treatment does).
"""
from __future__ import annotations

import json
import random
import re

from .providers import complete

SYSTEM = (
    "You are the person described below, taking part in an online study. Answer as that person would, in their own "
    "voice, with their habits, budget and mood. Keep answers short.\n\nWho you are, in your own words from an earlier interview:\n{backstory}"
)


def _json_tail(text: str) -> dict | None:
    m = re.findall(r"\{[^{}]*\}", text)
    for cand in reversed(m):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def run_choice(backstory: str, task: dict, arm_text: str, model: str, seed: int) -> dict:
    rng = random.Random(seed)
    opts = list(task["options"])  # each: {label, value}
    rng.shuffle(opts)
    listing = "\n".join(f"{i + 1}. {o['label']}" for i, o in enumerate(opts))
    user = task["prompt"].format(options=listing, arm=arm_text) + '\nFinish with JSON on its own line: {"choice": <number>}'
    text = complete(model, SYSTEM.format(backstory=backstory), user, temperature=1.0, max_tokens=300, seed=seed)
    d = _json_tail(text) or {}
    try:
        k = int(d.get("choice"))
        y = float(opts[k - 1]["value"])
    except (TypeError, ValueError, IndexError):
        y = None
    return {"y": y, "raw": text}


def run_menu_choice(backstory: str, task: dict, arm_text: str, model: str, seed: int) -> dict:
    """One prompt, several menus. Returns the aggregate of the property over chosen items and the per-menu picks."""
    rng = random.Random(seed)
    blocks, keys = [], []
    menus = task["menus"]
    for mi, menu in enumerate(menus):
        items = list(menu["items"])
        rng.shuffle(items)
        lab = task.get("labelled_marker", "")
        lines = []
        for i, it in enumerate(items):
            mark = f" {arm_text}" if (it.get("labelled") and arm_text) else ""
            lines.append(f"  {i + 1}. {it['label']}{mark}")
        blocks.append(f"{menu['title']}\n" + "\n".join(lines))
        keys.append(items)
    user = task["prompt"].format(menus="\n\n".join(blocks), arm_intro=task["arm_intro"].get(arm_text, "") if isinstance(task.get("arm_intro"), dict) else "")
    user += '\nFinish with JSON on its own line: {"choices": [<number for menu 1>, <number for menu 2>, ...]}'
    text = complete(model, SYSTEM.format(backstory=backstory), user, temperature=1.0, max_tokens=500, seed=seed)
    d = _json_tail(text) or {}
    picks = d.get("choices")
    vals, labelled = [], []
    try:
        for mi, k in enumerate(picks):
            it = keys[mi][int(k) - 1]
            vals.append(float(it["value"]))
            labelled.append(1.0 if it.get("labelled") else 0.0)
        agg = task.get("aggregate", "sum")
        y = sum(vals) if agg == "sum" else sum(vals) / len(vals)
    except (TypeError, ValueError, IndexError):
        y, labelled = None, []
    return {"y": y, "raw": text, "share_labelled": (sum(labelled) / len(labelled)) if labelled else None}


def run_binary(backstory: str, task: dict, arm_text: str, model: str, seed: int) -> dict:
    rng = random.Random(seed)
    a, b = task["options"]
    first_is_target = rng.random() < 0.5
    o1, o2 = (a, b) if first_is_target else (b, a)
    user = task["prompt"].format(opt1=o1["label"].format(arm=arm_text), opt2=o2["label"].format(arm=arm_text), arm=arm_text)
    user += '\nFinish with JSON on its own line: {"choice": 1 or 2}'
    text = complete(model, SYSTEM.format(backstory=backstory), user, temperature=1.0, max_tokens=250, seed=seed)
    d = _json_tail(text) or {}
    try:
        k = int(d.get("choice"))
        chosen = o1 if k == 1 else o2
        y = float(chosen["value"])
    except (TypeError, ValueError):
        y = None
    return {"y": y, "raw": text}


def run_likert(backstory: str, task: dict, arm_text: str, model: str, seed: int) -> dict:
    items = task["items"]  # each: {text, reverse(bool), k}
    listing = "\n".join(f"Q{i + 1}. {it['text']}" for i, it in enumerate(items))
    user = task["prompt"].format(arm=arm_text, items=listing)
    user += '\nFinish with JSON on its own line: {"answers": [<number per question>]}'
    text = complete(model, SYSTEM.format(backstory=backstory), user, temperature=1.0, max_tokens=400, seed=seed)
    d = _json_tail(text) or {}
    ans = d.get("answers")
    try:
        vals = []
        for it, v in zip(items, ans):
            v = float(v)
            k = it.get("k", 5)
            if not (1 <= v <= k):
                raise ValueError
            vals.append((k + 1 - v) if it.get("reverse") else v)
        y = sum(vals) / len(vals)
    except (TypeError, ValueError):
        y = None
    return {"y": y, "raw": text}


RUNNERS = {"choice": run_choice, "menu_choice": run_menu_choice, "binary": run_binary, "likert": run_likert}
