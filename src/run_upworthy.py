"""Predict which Upworthy headline won its A/B test.

  PYTHONPATH=src uv run python src/run_upworthy.py --split exploratory --tests 200 --models claude-haiku-5-5 \
      --methods judge crowd --crowd 20

Methods:
  judge : one call per test; the model ranks all headlines by expected click-through.
  crowd : `--crowd` simulated readers per test, each scrolling a 2014 Facebook feed, picks the one headline they
          would click (or none). Score = share of picks. Replicates the decision rather than polling for a forecast.
Scores: pairwise accuracy against observed CTR order, top-1, against chance and the re-run noise ceiling
(from build_upworthy.py).
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from build_upworthy import pairwise_acc
from simsoc.providers import USAGE, complete
from simsoc.tasks import _json_tail

ROOT = Path(__file__).resolve().parents[1]

AGES = [("18-29", 0.30), ("30-44", 0.30), ("45-64", 0.30), ("65+", 0.10)]
GENDERS = [("woman", 0.6), ("man", 0.4)]
POLITICS = [("liberal", 0.45), ("moderate", 0.35), ("conservative", 0.20)]
EDU = [("high school", 0.35), ("some college", 0.30), ("college degree", 0.35)]
MOODS = ["bored on your lunch break", "half-watching TV", "on the bus", "procrastinating at work", "in bed before sleep"]


def _pick(rng, pairs):
    return rng.choices([p for p, _ in pairs], weights=[w for _, w in pairs])[0]


def reader(seed: int) -> str:
    rng = random.Random(seed)
    return (f"You are a {_pick(rng, AGES)} year old {_pick(rng, GENDERS)} in the US, politically {_pick(rng, POLITICS)}, "
            f"education: {_pick(rng, EDU)}. It is 2014 and you are scrolling your Facebook feed, {rng.choice(MOODS)}.")


def judge(headlines: list[str], model: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    order = list(range(len(headlines)))
    rng.shuffle(order)
    listing = "\n".join(f"{k + 1}. {headlines[i]}" for k, i in enumerate(order))
    user = ("These headlines were A/B tested for the same Upworthy article on Facebook in 2013-2015. Each was shown to "
            "a few thousand people. Rank them from most to least clicked.\n\n" + listing +
            '\n\nFinish with JSON on its own line: {"ranking": [numbers, best first]}')
    text = complete(model, "You are an experienced social media editor.", user, temperature=1.0, max_tokens=400, seed=seed)
    rank = (_json_tail(text) or {}).get("ranking") or []
    score = [np.nan] * len(headlines)
    for pos, k in enumerate(rank):
        try:
            score[order[int(k) - 1]] = -pos
        except (ValueError, IndexError, TypeError):
            pass
    return score


def crowd_pick(headlines: list[str], model: str, seed: int) -> int | None:
    rng = random.Random(seed)
    order = list(range(len(headlines)))
    rng.shuffle(order)
    listing = "\n".join(f"{k + 1}. {headlines[i]}" for k, i in enumerate(order))
    user = ("These posts are in your feed right now:\n\n" + listing +
            '\n\nWhich one, if any, do you click? Finish with JSON on its own line: {"click": <number or 0 for none>}')
    text = complete(model, reader(seed), user, temperature=1.0, max_tokens=200, seed=seed)
    try:
        k = int((_json_tail(text) or {}).get("click"))
        return order[k - 1] if k > 0 else None
    except (TypeError, ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="exploratory")
    ap.add_argument("--tests", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=["claude-haiku-5-5"])
    ap.add_argument("--methods", nargs="+", default=["judge", "crowd"])
    ap.add_argument("--crowd", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    d = pd.read_csv(ROOT / f"data/evals/upworthy_{args.split}.csv")
    base = json.loads((ROOT / f"data/evals/upworthy_{args.split}_baselines.json").read_text())
    ids = d.clickability_test_id.drop_duplicates().sample(frac=1, random_state=0).head(args.tests).tolist()
    d = d[d.clickability_test_id.isin(ids)].reset_index(drop=True)
    tests = {t: x.index.tolist() for t, x in d.groupby("clickability_test_id")}
    print(f"{len(tests)} tests, {len(d)} headlines | chance pairwise 0.50, top1 {base['chance_top1']:.2f} | "
          f"ceiling pairwise {base['ceiling']['pairwise']:.2f}, top1 {base['ceiling']['top1']:.2f}")
    out_dir = ROOT / "results" / "upworthy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for method in args.methods:
            t0, c0 = time.time(), USAGE.cost_usd
            score = pd.Series(np.nan, index=d.index)
            if method == "judge":
                def job(item):
                    t, rows = item
                    return rows, judge(d.headline[rows].tolist(), model, zlib.crc32(str(t).encode()) % 10**6)
                with ThreadPoolExecutor(args.workers) as ex:
                    for rows, s in ex.map(job, tests.items()):
                        score[rows] = s
            else:
                jobs = [(t, rows, r) for t, rows in tests.items() for r in range(args.crowd)]
                def job(x):
                    t, rows, r = x
                    return rows, crowd_pick(d.headline[rows].tolist(), model, (zlib.crc32(str(t).encode()) % 10**6) * 100 + r)
                score[:] = 0.0
                with ThreadPoolExecutor(args.workers) as ex:
                    for rows, k in ex.map(job, jobs):
                        if k is not None:
                            score[rows[k]] += 1
            top1 = np.mean([d.ctr[rows].to_numpy().argmax() == np.nanargmax(score[rows].to_numpy())
                            for rows in tests.values() if score[rows].notna().any()])
            pw = pairwise_acc(d, score)
            print(f"  {model:<22} {method:<6} pairwise {pw:.3f} | top1 {top1:.3f} | ${USAGE.cost_usd - c0:.2f} {time.time() - t0:.0f}s")
            d.assign(score=score).to_csv(out_dir / f"{args.split}_{model.replace('/', '__')}_{method}.csv", index=False)
    print("TOTAL $%.2f" % USAGE.cost_usd)


if __name__ == "__main__":
    main()
