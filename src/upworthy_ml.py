"""Classical headline classifier and judge+classifier blend for the Upworthy eval.

  PYTHONPATH=src uv run python src/upworthy_ml.py

Classifier: local sentence embeddings (fastembed, BAAI/bge-small-en-v1.5) + heuristic features, pairwise logistic
regression on feature differences within a test (Bradley-Terry style), trained on pool + calib tests only.
Training pairs need |z| >= 1 so pure-noise pairs do not teach the model. Calib scores are out-of-fold.
Blend: within-test z-scored judge and classifier scores, combined by a 2-feature pairwise logistic regression fitted
on calib (out-of-fold classifier + judge run on calib), then applied once to dev.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from build_upworthy import FEATURES, pair_z, score_all, split_ids

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "upworthy"
EMB_MODEL = "BAAI/bge-small-en-v1.5"


def embed(headlines: list[str]) -> dict[str, np.ndarray]:
    path = RES / "emb_bge_small.npz"
    cache = dict(np.load(path, allow_pickle=True)["d"].item()) if path.exists() else {}
    todo = [h for h in dict.fromkeys(headlines) if h not in cache]
    if todo:
        from fastembed import TextEmbedding
        for h, v in zip(todo, TextEmbedding(EMB_MODEL).embed(todo, batch_size=64)):
            cache[h] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(path, d=np.array(cache, dtype=object))
    return cache


def features(d: pd.DataFrame, emb: dict) -> np.ndarray:
    e = np.stack([emb[h] for h in d.headline])
    hand = np.array([[f(h) for f in FEATURES.values()] for h in d.headline], dtype=np.float32)
    hand[:, 0] /= 100  # length in hundreds of characters
    return np.hstack([e, hand])


def pairs(d: pd.DataFrame, X: np.ndarray, min_z: float = 1.0):
    """Within-test pair differences with label = first headline won; both orders included so no intercept is needed."""
    rows, ys, groups = [], [], []
    for t, x in d.groupby("clickability_test_id"):
        idx = x.index.to_numpy()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                hi, lo = (i, j) if d.ctr[i] >= d.ctr[j] else (j, i)
                if abs(pair_z(d.loc[hi], d.loc[lo])) < min_z:
                    continue
                for u, v, y in ((hi, lo, 1), (lo, hi, 0)):
                    rows.append(X[u] - X[v])
                    ys.append(y)
                    groups.append(t)
    return np.array(rows), np.array(ys), np.array(groups)


def fit(X, y, C):
    return LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(X, y)


def zscore_within(d: pd.DataFrame, s: pd.Series) -> pd.Series:
    g = s.groupby(d.clickability_test_id)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0)


def main():
    d = pd.read_csv(ROOT / "data/evals/upworthy_exploratory.csv")
    dev, calib, pool = split_ids(d)
    emb = embed(d.headline.tolist())
    X = features(d, emb)
    train_mask = ~d.clickability_test_id.isin(dev)
    tr = d[train_mask]

    # choose C by grouped CV on train pairs
    Xp, yp, gp = pairs(tr, X)
    best = None
    for C in (0.01, 0.1, 1.0):
        accs = []
        for a, b in GroupKFold(5).split(Xp, yp, gp):
            accs.append((fit(Xp[a], yp[a], C).predict(Xp[b]) == yp[b]).mean())
        print(f"C={C}: CV pair accuracy (|z|>=1 pairs) {np.mean(accs):.3f}")
        if best is None or np.mean(accs) > best[1]:
            best = (C, np.mean(accs))
    C = best[0]

    clf = pd.Series(np.nan, index=d.index)
    model = fit(Xp, yp, C)
    clf[~train_mask] = X[~train_mask] @ model.coef_[0]
    # out-of-fold scores on train tests (used for calib blending)
    tests = tr.clickability_test_id.drop_duplicates().to_numpy()
    for k, (_, hold) in enumerate(GroupKFold(5).split(tests, groups=tests)):
        held = set(tests[hold])
        m = ~d.clickability_test_id.isin(held) & train_mask
        Xa, ya, _ = pairs(d[m], X)
        clf[d.clickability_test_id.isin(held)] = X[d.clickability_test_id.isin(held)] @ fit(Xa, ya, C).coef_[0]
    d["clf"] = clf
    d.to_csv(RES / "exploratory_classifier.csv", index=False)

    is_dev, is_cal = d.clickability_test_id.isin(dev), d.clickability_test_id.isin(calib)
    out = {"classifier": score_all(d[is_dev], d.clf[is_dev]), "C": C}
    print(f"classifier (dev):        {fmt(out['classifier'])}")

    for judge in sorted(p for p in RES.glob("exploratory_*.csv") if p.stem.endswith(("_judge", "_fewshot"))):
        j = pd.read_csv(judge)[["clickability_test_id", "headline", "score"]]
        dj = d.merge(j, on=["clickability_test_id", "headline"], how="left")
        name = judge.stem.replace("exploratory_", "")
        dv = dj[dj.clickability_test_id.isin(dev)]
        if dv.score.notna().sum() == 0:
            continue
        out[name] = score_all(dv, dv.score)
        print(f"{name:<24} (dev): {fmt(out[name])}")
        cal = dj[dj.clickability_test_id.isin(calib)].dropna(subset=["score"])
        if cal.empty:
            continue
        # blend weights fitted on calib only
        f = lambda x: np.column_stack([zscore_within(x, x.score.fillna(x.score.min() - 1)), zscore_within(x, x.clf)])
        Fc = f(cal)
        Pc, yc, _ = pairs(cal.reset_index(drop=True), Fc, min_z=1.0)
        w = fit(Pc, yc, 1.0).coef_[0]
        blend = pd.Series(f(dv) @ w, index=dv.index)
        out[name + "+clf"] = score_all(dv, blend) | {"weights_judge_clf": w.round(3).tolist()}
        print(f"{name + '+clf':<24} (dev): {fmt(out[name + '+clf'])}  weights judge/clf {w.round(2)}")
    (RES / "exploratory_ml_summary.json").write_text(json.dumps(out, indent=1))


def fmt(r: dict) -> str:
    return (f"pairwise {r['pairwise']:.3f} | top1 {r['top1']:.3f} | best-vs-worst {r['bvw']:.3f} | "
            f"z>=2 {r['z2']:.3f} | z>=4 {r['z4']:.3f} (n={r['n_z4']})")


if __name__ == "__main__":
    main()
