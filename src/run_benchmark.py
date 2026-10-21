"""Run every elicitation method x model on a case-study YAML and score against human ground truth.

Usage:
  PYTHONPATH=src uv run python src/run_benchmark.py data/<case>.yaml --models claude-haiku-5-5 claude-haiku-4-5-20251001 --n 12

Case YAML schema:
  name: str
  outcome_hint: str            # one sentence telling the persona what outcome is being predicted
  messages: {id: text, ...}
  segments:                    # list of demographic cells
    - cell: {age_band: ..., gender: ...}
      truth: {message_id: human_effect, ...}   # higher = message worked better for this segment
  source: str                  # citation / URL of the human data
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import yaml

from simsoc import metrics
from simsoc.elicit import Message, elicit_choice, elicit_debate, elicit_judge, elicit_persona, run_many
from simsoc.personas import build_personas
from simsoc.providers import USAGE

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--models", nargs="+", default=["claude-haiku-5-5"])
    ap.add_argument("--n", type=int, default=12, help="personas per segment for ensemble/debate")
    ap.add_argument("--methods", nargs="+", default=["judge", "label_single", "grounded_single", "grounded_ensemble", "grounded_debate"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debate_n", type=int, default=None, help="personas per segment for debate (default = n)")
    args = ap.parse_args()

    case = yaml.safe_load(Path(args.case).read_text())
    msg_ids = list(case["messages"])
    msgs = [Message(i, case["messages"][i]) for i in msg_ids]
    cells = [s["cell"] for s in case["segments"]]
    truth = {metrics.cell_key(s["cell"]): s["truth"] for s in case["segments"]}
    hint = case.get("outcome_hint", "")
    debate_n = args.debate_n or args.n

    out_dir = ROOT / "results" / case["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for model in args.models:
        print(f"\n=== {model} ===")
        personas = build_personas(cells, args.n, grounded=True, model=model, workers=args.workers)
        for method in args.methods:
            t0 = time.time()
            cost0 = USAGE.cost_usd
            jobs = []
            rng = random.Random(42)

            def order():
                o = list(range(len(msgs)))
                rng.shuffle(o)
                return o

            if method == "judge":
                # one call per segment per sample; 5 samples to average out sampling noise
                for cell in cells:
                    for s in range(5):
                        jobs.append(dict(cell=cell, msgs=msgs, model=model, order=order(), outcome_hint=hint, seed=s))
                results = run_many(elicit_judge, jobs, args.workers)
            elif method == "label_single":
                for cell in cells:
                    for s in range(5):
                        p = next(p for p in personas if p.cell == cell)
                        jobs.append(dict(p=p, msgs=msgs, model=model, grounded=False, order=order(), outcome_hint=hint, seed=s))
                results = run_many(elicit_persona, jobs, args.workers)
            elif method == "grounded_single":
                for cell in cells:
                    p = next(p for p in personas if p.cell == cell)
                    for s in range(5):
                        jobs.append(dict(p=p, msgs=msgs, model=model, grounded=True, order=order(), outcome_hint=hint, seed=s))
                results = run_many(elicit_persona, jobs, args.workers)
            elif method == "grounded_ensemble":
                for p in personas:
                    jobs.append(dict(p=p, msgs=msgs, model=model, grounded=True, order=order(), outcome_hint=hint, seed=0))
                results = run_many(elicit_persona, jobs, args.workers)
            elif method == "grounded_debate":
                for p in personas:
                    if p.idx < debate_n:
                        jobs.append(dict(p=p, msgs=msgs, model=model, order=order(), outcome_hint=hint, seed=0))
                results = run_many(elicit_debate, jobs, args.workers)
            elif method in ("grounded_choice", "label_choice", "grounded_choice_neutral"):
                grounded = method.startswith("grounded")
                variant = "neutral" if method.endswith("neutral") else "default"
                reps = 2 if grounded else 1
                for p in personas:
                    for mid in msg_ids:
                        for s in range(reps):
                            jobs.append(dict(p=p, arm_id=mid, arm_text=case["messages"][mid], model=model, grounded=grounded, seed=s, variant=variant))
                results = run_many(elicit_choice, jobs, args.workers)
            else:
                raise SystemExit(f"unknown method {method}")

            pred = metrics.aggregate_choice(results, msg_ids) if "_choice" in method else metrics.aggregate(results, msg_ids)
            sc = metrics.score(pred, truth, msg_ids)
            sc.update(model=model, method=method, cost_usd=round(USAGE.cost_usd - cost0, 4), seconds=round(time.time() - t0, 1),
                      within_sd=metrics.within_sd(pred), n_jobs=len(jobs))
            if method == "grounded_debate":
                # also score round-1 answers so debate's marginal effect is isolated
                r1 = [dict(r, points=r["round1_points"]) for r in results]
                sc["round1"] = {k: v for k, v in metrics.score(metrics.aggregate(r1, msg_ids), truth, msg_ids).items() if k != "per_segment"}
            summary.append(sc)
            (out_dir / f"{model}__{method}.json").write_text(json.dumps({"score": sc, "results": results}, indent=1, default=str))
            print(f"{method:18s} rho={sc['mean_spearman']:.3f} top1={sc['top1_accuracy']:.2f} gap={sc['gap_preservation']:.3f} "
                  f"valid={sc['n_valid_answers']} sd={sc['within_sd']:.1f} ${sc['cost_usd']:.3f} {sc['seconds']}s")

    # merge with earlier runs so partial reruns don't clobber other methods/models
    prev_path = out_dir / "summary.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else []
    keep = [r for r in prev if (r["model"], r["method"]) not in {(r["model"], r["method"]) for r in summary}]
    prev_path.write_text(json.dumps(keep + summary, indent=1, default=str))
    print("\nTotal cost this run: $%.3f" % USAGE.cost_usd, {k: round(v["cost"], 3) for k, v in USAGE.by_model.items()})


if __name__ == "__main__":
    main()
