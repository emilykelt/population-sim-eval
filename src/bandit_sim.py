"""Offline simulator for the tweet bot: a contextual bandit replaying real Upworthy A/B tests (no API calls, $0).

  PYTHONPATH=src uv run python src/bandit_sim.py

Each round = one Upworthy test (a "topic"). Arms = its 2-5 headlines (the "drafts"). The policy sees each draft's
features (16 PCA components of the bge-small embedding + hand features, optionally the Upworthy classifier score),
posts one, and gets reward = clicks / views, with clicks ~ Binomial(views, the headline's real measured CTR).
Only the posted draft's reward is seen, as on Twitter. Policy: linear Thompson sampling.

Experiment 1 (reach): how much of the achievable lift each policy captures, at 20 / 200 / 2,000 / 20,000 views per post,
over the first 30, 365 and 2,000 posts. Environment = 2,237 holdout tests (the classifier never saw them).

Experiment 2 (sim-to-real): train the bandit on the 200 dev tests twice, once rewarded by the cached simulated persona
crowd (20 Haiku readers per test, results/upworthy/exploratory_claude-haiku-5-5_crowd.csv) and once by real clicks.
Then freeze both and pick headlines greedily on the holdout tests, scored by real CTR. The gap is the cost of
training in the persona simulator instead of on real people.

Lift captured per round = (CTR of pick - test mean CTR) / (best CTR - test mean CTR): random = 0, oracle = 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_upworthy import FEATURES, split_ids
from upworthy_ml import embed, features

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "upworthy"
K_PCA = 16


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = pd.read_csv(ROOT / "data/evals/upworthy_holdout.csv")
    expl = pd.read_csv(ROOT / "data/evals/upworthy_exploratory.csv")
    dev, _, _ = split_ids(expl)
    crowd = pd.read_csv(RES / "exploratory_claude-haiku-5-5_crowd.csv")[["clickability_test_id", "headline", "score"]]
    dev_d = expl[expl.clickability_test_id.isin(dev)].merge(crowd, on=["clickability_test_id", "headline"], how="left")
    dev_d["score"] = dev_d.score.fillna(0)
    return hold.reset_index(drop=True), dev_d.reset_index(drop=True)


def design(frames: list[pd.DataFrame]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per frame: (X without classifier, classifier z-score). PCA fitted on all headlines (no labels used)."""
    w = np.load(RES / "classifier_weights.npy")
    raw = [features(f, embed(f.headline.tolist())) for f in frames]
    allE = np.vstack([r[:, :384] for r in raw])
    mu = allE.mean(0)
    _, _, vt = np.linalg.svd(allE - mu, full_matrices=False)
    keep = [i for i, k in enumerate(FEATURES) if k != "neg_length"]
    Z_all = np.vstack([np.hstack([(r[:, :384] - mu) @ vt[:K_PCA].T, r[:, 384:][:, keep]]) for r in raw])
    zm, zs = Z_all.mean(0), Z_all.std(0) + 1e-9
    out = []
    for r in raw:
        Z = (np.hstack([(r[:, :384] - mu) @ vt[:K_PCA].T, r[:, 384:][:, keep]]) - zm) / zs
        c = r @ w
        out.append((Z, (c - c.mean()) / c.std()))
    return out


class TS:
    """Bayesian linear regression with a Gaussian prior, Thompson sampling over arms."""

    def __init__(self, dim: int, sigma2: float, prior_mean: np.ndarray | None = None, prior_var=1.0):
        self.m0 = np.zeros(dim) if prior_mean is None else prior_mean
        self.A = np.diag(np.broadcast_to(1.0 / np.asarray(prior_var, dtype=float), (dim,)).copy())
        self.b = self.A @ self.m0
        self.s2 = sigma2

    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def pick(self, X: np.ndarray, rng) -> int:
        cov = np.linalg.inv(self.A)
        w = rng.multivariate_normal(self.mean(), (cov + cov.T) / 2)
        return int(np.argmax(X @ w))

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x) / self.s2
        self.b += x * r / self.s2


def tests_of(d: pd.DataFrame) -> list[np.ndarray]:
    return [g.index.to_numpy() for _, g in d.groupby("clickability_test_id") if len(g) >= 2 and g.ctr.max() > g.ctr.mean()]


def lift(ctr: np.ndarray, k: int) -> float:
    return (ctr[k] - ctr.mean()) / (ctr.max() - ctr.mean())


def reward_scale(d: pd.DataFrame, views: int) -> tuple[float, float]:
    gm = d.ctr.mean()
    # noise the policy must see through: topic-to-topic CTR spread + binomial sampling at this many views
    return gm, float((d.ctr / gm).var() + (d.ctr * (1 - d.ctr) / views).mean() / gm ** 2)


def run(d, X, tests, policy, views, T, rng, clf=None):
    gm, s2 = reward_scale(d, views)
    ctr = d.ctr.to_numpy()
    agent = None
    if policy == "ts":
        agent = TS(X.shape[1], s2)
    elif policy == "ts+prior":
        X = np.hstack([X, clf[:, None]])
        # starts as the classifier ranking: within-test lifts are ~0.1 in reward units, so the classifier weight
        # gets mean 0.1, and the other features a prior tight enough (sd 0.01) not to drown it before data arrives
        m0 = np.zeros(X.shape[1]); m0[-1] = 0.1
        pv = np.full(X.shape[1], 1e-4); pv[-1] = 1e-3
        agent = TS(X.shape[1], s2, m0, prior_var=pv)
    out = np.empty(T)
    for t in range(T):
        idx = tests[rng.integers(len(tests))]
        if policy == "random":
            k = int(rng.integers(len(idx)))
        elif policy == "classifier":
            k = int(np.argmax(clf[idx]))
        else:
            k = agent.pick(X[idx], rng)
            r = rng.binomial(views, ctr[idx[k]]) / views / gm - 1
            agent.update(X[idx[k]], r)
        out[t] = lift(ctr[idx], k)
    return out, agent


def greedy_eval(w, X, d, tests) -> float:
    ctr = d.ctr.to_numpy()
    return float(np.mean([lift(ctr[i], int(np.argmax(X[i] @ w))) for i in tests]))


def main():
    hold, dev = load()
    (Xh, ch), (Xd, cd) = design([hold, dev])
    th, td = tests_of(hold), tests_of(dev)
    rng = np.random.default_rng(0)
    T, SEEDS, windows = 2000, 20, {"first 30": 30, "first 365": 365, "all 2000": 2000}
    res = {"reach": {}, "sim_to_real": {}}

    print(f"Experiment 1: {len(th)} holdout tests, {SEEDS} seeds, lift captured (random 0, oracle 1)")
    print(f"{'views/post':>10} {'policy':<11} " + " ".join(f"{w:>10}" for w in windows))
    for views in (20, 200, 2000, 20000):
        for pol in ("random", "classifier", "ts", "ts+prior"):
            runs = np.array([run(hold, Xh, th, pol, views, T, np.random.default_rng(s), ch)[0] for s in range(SEEDS)])
            row = {w: float(runs[:, :n].mean()) for w, n in windows.items()}
            res["reach"][f"{views}|{pol}"] = row
            print(f"{views:>10} {pol:<11} " + " ".join(f"{v:>10.3f}" for v in row.values()))

    print(f"\nExperiment 2: train on {len(td)} dev tests (persona-crowd reward vs real clicks at 2,000 views), "
          f"freeze, pick greedily on {len(th)} holdout tests")
    share = dev.score / dev.groupby("clickability_test_id").score.transform("sum").replace(0, 1)
    sims = {"persona crowd": dev.assign(ctr=share.to_numpy() * dev.ctr.mean() * 2), "real clicks": dev}
    for name, env in sims.items():
        scores = []
        for s in range(SEEDS):
            _, agent = run(env, Xd, tests_of(env), "ts", 2000, T, np.random.default_rng(s))
            scores.append(greedy_eval(agent.mean(), Xh, hold, th))
        res["sim_to_real"][name] = {"mean": float(np.mean(scores)), "sd": float(np.std(scores))}
        print(f"  trained on {name:<14} -> holdout lift {np.mean(scores):.3f} (sd {np.std(scores):.3f})")
    res["sim_to_real"]["classifier (12k real tests)"] = {"mean": greedy_eval(np.array([1.0]), ch[:, None], hold, th)}
    print(f"  classifier (trained on ~10k real tests) -> holdout lift {res['sim_to_real']['classifier (12k real tests)']['mean']:.3f}")
    ag = [np.corrcoef(g.score, g.ctr)[0, 1] for _, g in dev.groupby("clickability_test_id") if g.score.std() > 0 and len(g) > 2]
    res["persona_vs_real_within_test_r"] = float(np.nanmean(ag))
    print(f"  persona crowd vs real CTR, mean within-test correlation: {np.nanmean(ag):+.3f}")
    (RES / "bandit_sim.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
