"""Twin personas: one simulated person per REAL respondent row.

A twin carries the respondent's actual covariates (age, gender, education, income, BMI, habits ...), the
arm the real person was randomised to, and the real person's outcome. The backstory is generated once from
those covariates and cached. This makes three things possible that synthetic personas cannot do:
  * individual fidelity   (twin in the same arm vs the real person's own outcome)
  * mechanism fidelity    (do the same covariates predict the simulated outcome as predict the human one)
  * matched distributions (human and simulated samples have identical covariate composition by construction)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .providers import complete

TWIN_DIR = Path(__file__).resolve().parents[2] / "results" / "raw" / "twins"
TWIN_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Twin:
    dataset: str
    row_id: str
    covariates: dict          # human-readable covariate: value, used to write the backstory and as segment keys
    numeric: dict             # numeric covariates for mechanism regressions
    human_arm: str
    human_outcome: float | None
    backstory: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.dataset}:{self.row_id}".encode()).hexdigest()[:10]

    def segment(self, keys: list[str]) -> dict:
        return {k: str(self.covariates.get(k)) for k in keys}


def backstory_for(twin: Twin, model: str, country: str = "UK", topic_hint: str = "") -> str:
    path = TWIN_DIR / f"{twin.id}_{model.replace('/', '__')}.txt"
    if path.exists():
        return path.read_text()
    facts = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in twin.covariates.items() if v not in (None, "", "nan"))
    system = (
        "You write realistic first-person interview transcripts for social research. The speaker is an ordinary person. "
        "Avoid stereotypes and caricature. Do not make the person unusually virtuous, health-conscious or articulate. "
        f"Use the everyday register of someone living in {country}. Every stated fact below must be true of the speaker, "
        "and nothing you add may contradict them; invent only mundane texture."
    )
    user = (
        "Write the speaker's answers to a short life interview, first person, as one flowing transcript of about 200 words. "
        "Do not include the questions. Do not list the facts; let them surface naturally.\n\n"
        f"Facts about the speaker (from their survey record): {facts}.\n\n"
        f"The interview asked about: a typical weekday; how they decide what to eat and buy; money; who they trust for "
        f"health information; what they think of public health messages{('; ' + topic_hint) if topic_hint else ''}."
    )
    text = complete(model, system, user, temperature=1.0, max_tokens=700, seed=0)
    path.write_text(text)
    return text


def sample_twins(df: pd.DataFrame, dataset: str, id_col: str, arm_col: str, outcome_col: str,
                 covariate_cols: dict[str, str], numeric_cols: list[str], n: int, seed: int = 0,
                 stratify: str | None = None) -> list[Twin]:
    """covariate_cols maps human-readable name -> dataframe column (already decoded to labels)."""
    d = df.dropna(subset=[arm_col, outcome_col]).copy()
    if stratify:
        per = max(1, n // d[stratify].nunique())
        d = d.groupby(stratify, group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
    else:
        d = d.sample(min(n, len(d)), random_state=seed)
    out = []
    for _, r in d.iterrows():
        cov = {k: (None if pd.isna(r[c]) else r[c]) for k, c in covariate_cols.items()}
        num = {c: (None if pd.isna(r[c]) else float(r[c])) for c in numeric_cols}
        out.append(Twin(dataset, str(r[id_col]), cov, num, str(r[arm_col]), float(r[outcome_col])))
    return out


def save_twins(twins: list[Twin], path: Path) -> None:
    path.write_text(json.dumps([t.__dict__ for t in twins], indent=1, default=str))


def load_twins(path: Path) -> list[Twin]:
    return [Twin(**d) for d in json.loads(path.read_text())]
