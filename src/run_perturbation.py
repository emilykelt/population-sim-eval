"""Idea B: run the benchmark under 25 prompt wordings, both designs, fixed personas.

  PYTHONPATH=src uv run python src/run_perturbation.py data/clarke2020_study2_labels.yaml --models claude-haiku-5-5 claude-sonnet-5-5 --n 6
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import yaml

from simsoc import metrics
from simsoc.elicit import SNACKS, _parse_points, run_many
from simsoc.personas import build_personas
from simsoc.perturb import CHOICE_FMT, CHOICE_TASK, POLL_FMT, POLL_TASK, SYSTEM, arm_paraphrases, design_25
from simsoc.providers import USAGE, complete

ROOT = Path(__file__).resolve().parents[1]


def persona_system(level: int, backstory: str) -> str:
    return SYSTEM[level] + "\n\nWho you are, in your own words from an earlier interview:\n" + backstory


def choice_job(p, arm_id, arm_text, model, levels, seed=0):
    rng = random.Random(f"{p.id}-{arm_id}-{seed}")
    idx = list(range(len(SNACKS)))
    rng.shuffle(idx)
    options = "\n".join(f"{k + 1}. {SNACKS[i][0]}" for k, i in enumerate(idx))
    user = CHOICE_TASK[levels[1]].format(options=options, arm=arm_text, fmt=CHOICE_FMT[levels[3]])
    text = complete(model, persona_system(levels[0], p.backstory), user, temperature=1.0, max_tokens=300, seed=seed)
    m = re.search(r'"choice"\s*:\s*"?(\d+)', text)
    choice = int(m.group(1)) if m else None
    ed = SNACKS[idx[choice - 1]][1] if choice and 1 <= choice <= 12 else None
    return {"persona": p.id, "cell": p.cell, "arm": arm_id, "energy_dense": ed, "raw": text}


def poll_job(p, msgs, model, levels, order, hint, seed=0):
    body = "\n\n".join(f"[{chr(65 + k)}] {msgs[i][1]}" for k, i in enumerate(order))
    user = POLL_TASK[levels[1]].format(n=len(msgs), messages=body, hint=hint, fmt=POLL_FMT[levels[3]])
    text = complete(model, persona_system(levels[0], p.backstory), user, temperature=1.0, max_tokens=900, seed=seed)
    return {"persona": p.id, "cell": p.cell, "order": order, "points": _parse_points(text, len(msgs)), "raw": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--models", nargs="+", default=["claude-haiku-5-5"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--designs", nargs="+", default=["choice", "poll"])
    args = ap.parse_args()

    case = yaml.safe_load(Path(args.case).read_text())
    msg_ids = list(case["messages"])
    cells = [s["cell"] for s in case["segments"]]
    truth = {metrics.cell_key(s["cell"]): s["truth"] for s in case["segments"]}
    pooled = np.array([case["pooled_truth"][m] for m in msg_ids])
    arms = arm_paraphrases(case["messages"])
    out_dir = ROOT / "results" / f"{case['name']}_perturb"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in args.models:
        personas = build_personas(cells, args.n, grounded=True, model=model, workers=args.workers)
        for design in args.designs:
            for w, levels in enumerate(design_25()):
                t0, c0 = time.time(), USAGE.cost_usd
                msgs = [(m, arms[levels[2]][m]) for m in msg_ids]
                rng = random.Random(1000 + w)
                if design == "choice":
                    jobs = [dict(p=p, arm_id=m, arm_text=t, model=model, levels=levels) for p in personas for m, t in msgs]
                    res = run_many(choice_job, jobs, args.workers)
                    pred = metrics.aggregate_choice(res, msg_ids)
                    base = float(np.mean([0 if r["energy_dense"] else 1 for r in res if r["energy_dense"] is not None]))
                else:
                    jobs = []
                    for p in personas:
                        o = list(range(len(msgs)))
                        rng.shuffle(o)
                        jobs.append(dict(p=p, msgs=msgs, model=model, levels=levels, order=o, hint=case["outcome_hint"]))
                    res = run_many(poll_job, jobs, args.workers)
                    pred = metrics.aggregate(res, msg_ids)
                    base = float("nan")
                sc = metrics.score(pred, truth, msg_ids)
                sim_pooled = np.array([np.mean([x for seg in pred.values() for x in seg[m]]) for m in msg_ids])
                from scipy.stats import spearmanr
                row = dict(model=model, design=design, wording=w, lv_system=levels[0], lv_task=levels[1], lv_arms=levels[2], lv_format=levels[3],
                           rho=sc["mean_spearman"], top1=sc["top1_accuracy"], gap=sc["gap_preservation"],
                           pooled_rho=float(spearmanr(sim_pooled, pooled).correlation), base_rate=base,
                           top1_by_seg={seg: msg_ids[int(np.argmax(d["pred"]))] for seg, d in sc["per_segment"].items()},
                           n_valid=sc["n_valid_answers"], cost=round(USAGE.cost_usd - c0, 3), secs=round(time.time() - t0, 1))
                rows.append(row)
                (out_dir / f"{model}__{design}__w{w:02d}.json").write_text(json.dumps({"row": row, "results": res}, default=str))
                print(f"{model} {design} w{w:02d} lv={levels} rho={row['rho']:.2f} top1={row['top1']:.2f} pooled={row['pooled_rho']:.2f} base={base:.2f} ${row['cost']:.2f}", flush=True)
                (out_dir / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    print("TOTAL $%.2f" % USAGE.cost_usd, {k: round(v["cost"], 2) for k, v in USAGE.by_model.items()})


if __name__ == "__main__":
    main()
