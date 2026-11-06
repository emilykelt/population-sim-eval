"""Train the headline classifier on the large confirmatory split (+ exploratory pool), test on the holdout split.

  PYTHONPATH=src uv run python src/upworthy_scale.py [--emb bge_small]

Train: confirmatory + exploratory pool/calib tests. Test: holdout (run once) and exploratory dev (comparable with
the LLM judges). Training tests sharing any headline with a test-set headline are dropped (Upworthy reused headlines).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_upworthy import score_all, split_ids
from upworthy_ml import embed, features, fit

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "upworthy"


def fast_pairs(d: pd.DataFrame, X: np.ndarray, min_z: float = 1.0):
    """Vectorised within-test pairs, both orders, |z| >= min_z. d must have a 0..n-1 index aligned with X."""
    rows_a, rows_b = [], []
    for idx in d.groupby("clickability_test_id").indices.values():
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        rows_a.append(idx[i])
        rows_b.append(idx[j])
    a, b = np.concatenate(rows_a), np.concatenate(rows_b)
    ctr, imp, clk = d.ctr.to_numpy(), d.impressions.to_numpy(), d.clicks.to_numpy()
    p = (clk[a] + clk[b]) / (imp[a] + imp[b])
    se = np.sqrt(p * (1 - p) * (1 / imp[a] + 1 / imp[b]))
    z = np.where(se > 0, (ctr[a] - ctr[b]) / np.where(se > 0, se, 1), 0)
    keep = np.abs(z) >= min_z
    a, b, z = a[keep], b[keep], z[keep]
    diff = X[a] - X[b]
    y = (z > 0).astype(int)
    return np.vstack([diff, -diff]), np.concatenate([y, 1 - y])


def main():
    exp = pd.read_csv(ROOT / "data/evals/upworthy_exploratory.csv")
    con = pd.read_csv(ROOT / "data/evals/upworthy_confirmatory.csv")
    hol = pd.read_csv(ROOT / "data/evals/upworthy_holdout.csv")
    dev, _, _ = split_ids(exp)
    dev_df = exp[exp.clickability_test_id.isin(dev)]
    train = pd.concat([con, exp[~exp.clickability_test_id.isin(dev)]], ignore_index=True)
    test_heads = set(hol.headline) | set(dev_df.headline)
    leak = train.groupby("clickability_test_id").headline.transform(lambda h: h.isin(test_heads).any())
    print(f"train {train.clickability_test_id.nunique()} tests; dropping {train[leak].clickability_test_id.nunique()} that share a headline with test sets")
    train = train[~leak].reset_index(drop=True)

    emb = embed(pd.concat([train.headline, hol.headline, dev_df.headline]).tolist())
    Xtr = features(train, emb)
    Xp, yp = fast_pairs(train, Xtr)
    print(f"{train.clickability_test_id.nunique()} train tests, {len(yp) // 2} training pairs (|z|>=1)")
    out, ws = {}, {}
    dv = dev_df.reset_index(drop=True)
    for C in (0.03, 0.1, 0.3):  # C chosen on dev only; holdout is scored once, with the chosen C
        ws[C] = fit(Xp, yp, C).coef_[0]
        out[f"dev C={C}"] = r = score_all(dv, pd.Series(features(dv, emb) @ ws[C]))
        print(f"dev      C={C:<5} {fmt(r)}")
    C = max(ws, key=lambda c: out[f"dev C={c}"]["pairwise"])
    hv = hol.reset_index(drop=True)
    out[f"holdout C={C}"] = r = score_all(hv, pd.Series(features(hv, emb) @ ws[C]))
    print(f"HOLDOUT  C={C:<5} {fmt(r)}")
    (RES / "scale_summary.json").write_text(json.dumps(out, indent=1))
    np.save(RES / "classifier_weights.npy", ws[C])


def fmt(r: dict) -> str:
    return (f"pairwise {r['pairwise']:.3f} | top1 {r['top1']:.3f} | best-vs-worst {r['bvw']:.3f} | "
            f"z>=2 {r['z2']:.3f} | z>=4 {r['z4']:.3f} (n={r['n_z4']})")


if __name__ == "__main__":
    main()
