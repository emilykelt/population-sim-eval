"""Upworthy Research Archive -> headline-only A/B eval set, noise ceiling, heuristic baselines.

  PYTHONPATH=src uv run python src/build_upworthy.py [exploratory|confirmatory]

Raw: OSF jd64p (Matias, Munger, Aubin Le Quere, Ebersole 2021, Scientific Data). Download into data/upworthy_raw/.
Keeps tests where every variant shares the same image (eyecatcher_id) and has a distinct headline, so only the
headline differs. Ceiling = how often the observed best headline stays best when clicks are redrawn from
Beta posteriors (the best any predictor can agree with a noisy observed winner).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "upworthy_raw"
OUT = ROOT / "data" / "evals"


def load(split: str) -> pd.DataFrame:
    d = pd.read_csv(RAW / f"{split}.csv", low_memory=False)
    d = d.dropna(subset=["headline", "impressions", "clicks"])
    d = d[d.impressions > 0]
    d["headline"] = d.headline.str.strip()
    g = d.groupby("clickability_test_id")
    keep = g.filter(lambda x: len(x) >= 2 and x.eyecatcher_id.nunique() == 1 and x.headline.nunique() == len(x))
    keep = keep.assign(ctr=keep.clicks / keep.impressions)
    return keep[["clickability_test_id", "headline", "impressions", "clicks", "ctr", "eyecatcher_id", "test_week"]]


def ceiling(d: pd.DataFrame, draws: int = 400, seed: int = 0) -> dict:
    """Agreement of a re-run of the same test with the observed one: top-1 match and within-test Spearman."""
    rng = np.random.default_rng(seed)
    top1, rhos, pair = [], [], []
    for _, x in d.groupby("clickability_test_id"):
        a, b = x.clicks.to_numpy() + 1, (x.impressions - x.clicks).to_numpy() + 1
        p = rng.beta(a[:, None], b[:, None], size=(len(x), draws))
        obs = x.ctr.to_numpy()
        top1.append(np.mean(p.argmax(0) == obs.argmax()))
        if len(x) >= 3:
            rhos.append(np.nanmean([spearmanr(p[:, k], obs)[0] for k in range(0, draws, 20)]))
        i, j = np.triu_indices(len(x), 1)
        pair.append(np.mean(np.sign(p[i] - p[j]) == np.sign(obs[i] - obs[j])[:, None]))
    return {"top1": float(np.mean(top1)), "spearman": float(np.nanmean(rhos)), "pairwise": float(np.mean(pair))}


FEATURES = {
    "length": lambda h: len(h),
    "neg_length": lambda h: -len(h),
    "question": lambda h: float("?" in h),
    "you": lambda h: float(bool(re.search(r"\byou(r|'re)?\b", h, re.I))),
    "number": lambda h: float(bool(re.search(r"\d", h))),
    "this": lambda h: float(bool(re.search(r"\b(this|these)\b", h, re.I))),
    "quote": lambda h: float('"' in h or "“" in h or "'" in h[:1]),
}


def pairwise_acc(d: pd.DataFrame, score: pd.Series) -> float:
    """Share of within-test pairs (with a score difference) whose order matches the observed CTR order."""
    hits = []
    for _, x in d.assign(s=score).groupby("clickability_test_id"):
        s, c = x.s.to_numpy(), x.ctr.to_numpy()
        i, j = np.triu_indices(len(x), 1)
        m = (s[i] != s[j]) & (c[i] != c[j])
        hits.extend(np.sign(s[i] - s[j])[m] == np.sign(c[i] - c[j])[m])
    return float(np.mean(hits)) if hits else float("nan")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "exploratory"
    d = load(split)
    n_tests = d.clickability_test_id.nunique()
    print(f"{split}: {n_tests} headline-only tests, {len(d)} variants, median {d.groupby('clickability_test_id').size().median():.0f} per test")
    ceil = ceiling(d)
    print(f"noise ceiling (re-run agreement): top1 {ceil['top1']:.2f} | pairwise {ceil['pairwise']:.2f} | spearman {ceil['spearman']:.2f}")
    chance_top1 = float(np.mean(1 / d.groupby("clickability_test_id").size()))
    print(f"chance: top1 {chance_top1:.2f} | pairwise 0.50")
    base = {name: pairwise_acc(d, d.headline.map(f)) for name, f in FEATURES.items()}
    for k, v in base.items():
        print(f"  heuristic {k:<11} pairwise {v:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"upworthy_{split}.csv", index=False)
    (OUT / f"upworthy_{split}_baselines.json").write_text(json.dumps(
        {"n_tests": n_tests, "n_variants": len(d), "ceiling": ceil, "chance_top1": chance_top1, "heuristics": base}, indent=1))


if __name__ == "__main__":
    main()
