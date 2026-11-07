"""Rank draft posts with the Upworthy-trained headline classifier (no API calls).

  PYTHONPATH=src uv run python src/rank_drafts.py drafts.txt      # one draft per line

Scores are relative: only the order within one batch means anything (the model was trained on within-test
differences). Trained on 2013-15 Facebook headlines, so treat it as a wording tiebreaker, not a virality oracle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from upworthy_ml import embed, features

ROOT = Path(__file__).resolve().parents[1]


def rank(drafts: list[str]) -> pd.DataFrame:
    w = np.load(ROOT / "results/upworthy/classifier_weights.npy")
    d = pd.DataFrame({"headline": drafts})
    s = features(d, embed(drafts)) @ w
    d["score"] = (s - s.mean()) / (s.std() or 1)
    return d.sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    lines = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    r = rank(lines)
    for i, row in r.iterrows():
        print(f"{i + 1:>2}. {row.score:+.2f}  {row.headline}")
