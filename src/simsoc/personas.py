"""Persona construction.

Two conditions, following Park et al. 2024 and 'The Prompt Makes the Persona' (2025):
  * label   -> a bare demographic description ("You are a 34-year-old woman ...").
  * grounded -> a first-person, interview-style backstory generated once per persona and cached,
               constrained by the same demographic cell plus a few randomly drawn 'life facts' so that
               personas within a cell differ (counter to flattening).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .providers import complete

PERSONA_DIR = Path(__file__).resolve().parents[2] / "results" / "raw" / "personas"
PERSONA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Persona:
    cell: dict  # demographic cell, e.g. {"age_band": "25-34", "gender": "woman", "income": "low", "parent": True}
    idx: int
    backstory: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha1(json.dumps([self.cell, self.idx], sort_keys=True).encode()).hexdigest()[:10]

    def label_text(self) -> str:
        parts = []
        for k, v in self.cell.items():
            parts.append(f"{k.replace('_', ' ')}: {v}")
        return "You are a real person with these characteristics. " + "; ".join(parts) + "."


# Generic variation seeds; keeps within-cell diversity without implying stereotypes.
LIFE_FACTS = [
    "works shifts", "works a regular 9-to-5", "is between jobs", "is studying part time", "is retired",
    "lives with housemates", "lives alone", "lives with a partner", "lives with extended family",
    "cooks most evenings", "rarely cooks", "mostly eats on the go", "shops once a week at a big supermarket",
    "shops at the corner shop most days", "uses a food bank occasionally", "grows some vegetables",
    "has a long commute", "cycles everywhere", "has a chronic health condition", "is training for a race",
    "is very price-sensitive", "spends freely on food", "is sceptical of government advice", "trusts the NHS",
    "gets news from TikTok", "reads a daily paper", "rarely follows the news", "cares about animal welfare",
    "cares about climate", "thinks healthy eating advice changes every week",
]


def generate_backstory(persona: Persona, model: str) -> str:
    path = PERSONA_DIR / f"{persona.id}_{model.replace('/', '__')}.txt"
    if path.exists():
        return path.read_text()
    rng = random.Random(persona.id)
    facts = rng.sample(LIFE_FACTS, 3)
    cell_desc = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in persona.cell.items())
    system = (
        "You write realistic first-person interview transcripts for social research. "
        "The speaker is an ordinary person. Avoid stereotypes, avoid caricature, avoid making the person "
        "unusually virtuous or unusually health-conscious. Give them specific, mundane detail. Use British English."
    )
    user = (
        f"Write the speaker's answers to a 6-question life interview, in the first person, as a single flowing "
        f"transcript of about 220 words. Do not include the questions, only the answers. Do not state the "
        f"demographic facts as a list; let them come through naturally.\n\n"
        f"Demographic facts (must be consistent with): {cell_desc}.\n"
        f"Extra facts to weave in: {', '.join(facts)}.\n\n"
        f"The six questions were: a typical weekday; how they decide what to eat; money and food; who they trust "
        f"for health information; something they have changed about their diet and why (or why not); "
        f"what annoys them about public health campaigns."
    )
    text = complete(model, system, user, temperature=1.0, max_tokens=700, seed=persona.idx)
    path.write_text(text)
    return text


def build_personas(cells: list[dict], n_per_cell: int, grounded: bool, model: str, workers: int = 8) -> list[Persona]:
    from concurrent.futures import ThreadPoolExecutor

    out = [Persona(cell=cell, idx=i) for cell in cells for i in range(n_per_cell)]
    if grounded:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            stories = list(ex.map(lambda p: generate_backstory(p, model), out))
        for p, s in zip(out, stories):
            p.backstory = s
    return out
