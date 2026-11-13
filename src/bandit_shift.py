"""When does online learning help a posting bot? Bandit replay under real audience drift, with A/B feedback ($0).

  PYTHONPATH=src uv run python src/bandit_shift.py                  # real Upworthy drift only
  PYTHONPATH=src uv run python src/bandit_shift.py --clickbait 0.5  # plus a stand-in Twitter shift (see below)

bandit_sim.py found that a bandit never beats the Upworthy classifier, because the simulated audience was the one the
classifier was trained on and each post's reward is dominated by how popular the topic is. This script fixes both:

1. Drift. The classifier is retrained on the earliest 60% of exploratory + holdout tests (2013 to mid-2014) only;
   the bandit then runs on later tests, standing in for "a new platform / a new audience".
2. A/B feedback. Instead of one post per topic, the bot shows two drafts of the same topic and learns from the
   difference in their click rates, which cancels the topic's baseline popularity. (On X: two posts of the same
   topic, or a small paid A/B of two promoted posts.)

--clickbait g: a semi-synthetic shift standing in for a platform whose audience punishes curiosity bait. On late
tests only, each headline's CTR is multiplied by exp(-g * (#question + #you + #this)), the hand features the
old classifier rewards. Early tests (the classifier's training data) are unchanged.

Late tests are split into an environment the bandit learns in and an eval set it never touches. At checkpoints the
bandit's current best guess (posterior mean) picks one headline per eval test; score = lift captured
(random 0, always the real winner 1), scored on real CTR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bandit_sim import RES, TS, K_PCA, lift
from build_upworthy import FEATURES
from upworthy_ml import embed, features, fit, pairs

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [0, 30, 100, 365, 1000, 3000]


def load(clickbait: float = 0.0):
    d = pd.concat([pd.read_csv(ROOT / f"data/evals/upworthy_{s}.csv") for s in ("exploratory", "holdout")], ignore_index=True)
    d = d[d.groupby("clickability_test_id").headline.transform("size") >= 2].reset_index(drop=True)
    first = d.groupby("clickability_test_id").test_week.min().sort_values(kind="stable")
    ids = first.index.to_numpy()
    n_early = int(0.6 * len(ids))
    rng = np.random.default_rng(0)
    late = rng.permutation(ids[n_early:])
    if clickbait:
        bait = sum(d.headline.map(FEATURES[k]) for k in ("question", "you", "this"))
        is_late = d.clickability_test_id.isin(set(ids[n_early:]))
        d.loc[is_late, "ctr"] = d.ctr[is_late] * np.exp(-clickbait * bait[is_late])
    return d, set(ids[:n_early]), set(late[: len(late) // 2]), set(late[len(late) // 2:]), first


def tests(d, ids):
    return [g.index.to_numpy() for t, g in d.groupby("clickability_test_id") if t in ids and g.ctr.max() > g.ctr.mean()]


def eval_lift(score, d, ts) -> float:
    ctr = d.ctr.to_numpy()
    return float(np.mean([lift(ctr[i], int(np.argmax(score[i]))) for i in ts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clickbait", type=float, default=0.0)
    args = ap.parse_args()
    d, early, env_ids, eval_ids, first = load(args.clickbait)
    X = features(d, embed(d.headline.tolist()))
    in_ = lambda ids: d.clickability_test_id.isin(ids).to_numpy()
    print(f"{len(early)} early tests (weeks {first[list(early)].min()}-{first[list(early)].max()}), "
          f"{len(env_ids)} late env tests, {len(eval_ids)} late eval tests")

    # classifier trained on the past only
    Xp, yp, _ = pairs(d[in_(early)], X)
    w_old = fit(Xp, yp, 0.1).coef_[0]
    clf = X @ w_old
    # reference: same classifier trained on the late env tests with every headline's CTR known (offline upper bound)
    Xq, yq, _ = pairs(d[in_(env_ids)], X)
    clf_new = X @ fit(Xq, yq, 0.1).coef_[0]
    t_env, t_eval = tests(d, env_ids), tests(d, eval_ids)
    ref = {"old classifier (2013-14)": eval_lift(clf, d, t_eval),
           f"classifier trained offline on all {len(env_ids)} late env tests": eval_lift(clf_new, d, t_eval),
           "random": 0.0}
    for k, v in ref.items():
        print(f"  {k:<58} {v:.3f}")

    # bandit features: PCA of embeddings + hand features + old classifier score, all standardised
    E = X[:, :384]
    mu = E.mean(0)
    vt = np.linalg.svd(E - mu, full_matrices=False)[2]
    keep = [i for i, k in enumerate(FEATURES) if k != "neg_length"]
    Z = np.hstack([(E - mu) @ vt[:K_PCA].T, X[:, 384:][:, keep], ((clf - clf.mean()) / clf.std())[:, None]])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    dim = Z.shape[1]
    ctr = d.ctr.to_numpy()
    gm = d.ctr.mean()

    def run(mode, views, prior, seed):
        rng = np.random.default_rng(seed)
        binom = float((d.ctr * (1 - d.ctr) / views).mean() / gm ** 2)
        # one post: topic-to-topic spread + sampling noise; A/B difference: the topic cancels, sampling noise twice
        s2 = 2 * binom if mode == "ab" else binom + float((d.ctr / gm).var())
        if prior:
            m0 = np.zeros(dim); m0[-1] = 0.1
            pv = np.full(dim, 1e-4); pv[-1] = 1e-3
        else:
            m0, pv = None, 1e-2
        agent = TS(dim, s2, m0, pv)
        curve = {}
        for t in range(CHECKS[-1] + 1):
            if t in CHECKS:
                curve[t] = eval_lift(Z @ agent.mean(), d, t_eval)
            idx = t_env[rng.integers(len(t_env))]
            r = lambda k: rng.binomial(views, ctr[idx[k]]) / views / gm - 1
            if mode == "single":
                k = agent.pick(Z[idx], rng)
                agent.update(Z[idx[k]], r(k))
            else:  # show the two drafts that win two independent Thompson draws (forced distinct)
                a = agent.pick(Z[idx], rng)
                rest = [j for j in range(len(idx)) if j != a]
                b = rest[agent.pick(Z[idx[rest]], rng)]
                agent.update(Z[idx[a]] - Z[idx[b]], r(a) - r(b))
        return curve

    out = {"reference": ref, "curves": {}}
    print(f"\nLift on held-out late tests after N posts (topics); mean of 10 seeds")
    print(f"{'setup':<44}" + "".join(f"{c:>8}" for c in CHECKS))
    for views in (200, 2000):
        for mode in ("single", "ab"):
            for prior in (False, True):
                name = f"{views} views, {'A/B pair' if mode == 'ab' else 'one post'}, {'old clf prior' if prior else 'scratch'}"
                curves = [run(mode, views, prior, s) for s in range(10)]
                row = {c: float(np.mean([cv[c] for cv in curves])) for c in CHECKS}
                out["curves"][name] = row
                print(f"{name:<44}" + "".join(f"{v:>8.3f}" for v in row.values()))
    (RES / f"bandit_shift{'_clickbait' + str(args.clickbait) if args.clickbait else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
