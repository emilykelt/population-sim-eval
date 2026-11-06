"""Run the three-fidelity eval for one dataset spec.

  PYTHONPATH=src uv run python src/run_eval.py data/evals/<name>.yaml --models claude-sonnet-5-5 --twins 120

Spec YAML (written by build_evals.py):
  name, task_type, country, arms: {id: text}, control_arm, outcome: {name, binary, support, higher_is}
  segment_keys: [..], mechanism_covariates: [..], task: {...task-type specific...}
  human_table: path to parquet/csv with columns id, arm, y, segment keys, covariates
  twins: path to twins json
Each twin is run in EVERY arm (matched-composition crowd); its own-arm run also feeds individual fidelity.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simsoc import fidelity
from simsoc.providers import USAGE
from simsoc.tasks import RUNNERS
from simsoc.twins import backstory_for, load_twins

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--models", nargs="+", default=["claude-sonnet-5-5"])
    ap.add_argument("--twins", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--draws", type=int, default=1)
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text())
    arms = list(spec["arms"])
    human = pd.read_csv(ROOT / spec["human_table"])
    human["arm"] = human["arm"].astype(str)
    human = human.dropna(subset=spec["segment_keys"] + ["y"])
    twins = load_twins(ROOT / spec["twins"])[: args.twins]
    keys, covs = spec["segment_keys"], spec["mechanism_covariates"]
    runner = RUNNERS[spec["task_type"]]
    out_dir = ROOT / "results" / "evals" / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ceiling = fidelity.split_half_ceiling(human, arms, keys)
    print(f"{spec['name']}: {len(human)} humans, {len(twins)} twins, {len(arms)} arms; split-half ceiling {ceiling:.2f}")

    for model in args.models:
        t0, c0 = time.time(), USAGE.cost_usd
        with ThreadPoolExecutor(args.workers) as ex:
            stories = list(ex.map(lambda t: backstory_for(t, model, spec.get("country", "UK"), spec.get("topic_hint", "")), twins))
        for t, s in zip(twins, stories):
            t.backstory = s
        jobs = [(t, a, d) for t in twins for a in arms for d in range(args.draws)]

        def job(x):
            t, a, d = x
            seed = int(t.id[:6], 16) * 31 + arms.index(a) * 7 + d
            r = runner(t.backstory, spec["task"], spec["arms"][a], model, seed)
            row = {"id": t.id, "arm": a, "y": r["y"], "draw": d, "own_arm": a == t.human_arm, "y_human": t.human_outcome}
            row.update({k: str(t.extra.get("segments", {}).get(k, t.covariates.get(k))) for k in keys})
            row.update({c: t.numeric.get(c) for c in covs})
            if "share_labelled" in r:
                row["share_labelled"] = r["share_labelled"]
            return row, r["raw"]

        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(job, jobs))
        sim = pd.DataFrame([r for r, _ in res])
        raws = [{"id": r["id"], "arm": r["arm"], "raw": raw} for r, raw in res]
        valid = sim.dropna(subset=["y"])
        print(f"  {model}: {len(valid)}/{len(sim)} valid answers, ${USAGE.cost_usd - c0:.2f}, {time.time() - t0:.0f}s")

        binary = bool(spec["outcome"].get("binary"))
        support = spec["outcome"].get("support")
        rep = {
            "model": model, "n_twins": len(twins), "n_valid": int(len(valid)), "n_jobs": len(sim),
            "ceiling_split_half": ceiling,
            "effect": fidelity.effect_fidelity(human, valid, arms, keys),
            "distribution": fidelity.distributional_fidelity(human, valid, arms, keys, support),
            "mechanism": fidelity.mechanism_fidelity(human.dropna(subset=covs), valid.dropna(subset=covs), covs, arms),
            "individual": fidelity.individual_fidelity(valid[valid["own_arm"]].rename(columns={"y": "y_sim"}), keys, binary, human),
            "cost_usd": round(USAGE.cost_usd - c0, 3), "seconds": round(time.time() - t0),
        }
        if "share_labelled" in valid:
            rep["share_labelled_by_arm"] = valid.groupby("arm")["share_labelled"].mean().round(3).to_dict()
        stem = model.replace("/", "__")
        (out_dir / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str))
        (out_dir / f"{stem}_raw.json").write_text(json.dumps(raws, default=str))
        valid.to_csv(out_dir / f"{stem}_sim_table.csv", index=False)
        e, d_, m, i = rep["effect"], rep["distribution"], rep["mechanism"], rep["individual"]
        print(f"  EFFECT  pooled rho {e['pooled_rho']:.2f} | seg rho {e['segment_rho_mean']:.2f} (baseline {e['pooled_baseline']:.2f}, ceiling {ceiling:.2f}) | top1 {e['top1_accuracy']:.2f} | gap r {e['gap_preservation_r']:.2f} slope {e['gap_inflation_slope']:.2f} | effect-size slope {e['effect_size_slope']:.2f}")
        print(f"  DIST    W1 {d_['wasserstein_mean']:.3f} | TV {d_['tv_mean']:.3f} | dispersion ratio {d_['dispersion_ratio_mean']:.2f} | seg-arm MAE {d_['segment_arm_mae']:.3f} bias {d_['segment_arm_bias']:+.3f}")
        if "error" not in m:
            print(f"  MECH    main r {m['main_effect_r']:.2f} sign {m['main_effect_sign_agreement']:.2f} | interaction r {m['interaction_r']:.2f} | covariate R2 human {m['covariate_r2_human']:.3f} sim {m['covariate_r2_sim']:.3f} | arm R2 human {m['arm_r2_human']:.3f} sim {m['arm_r2_sim']:.3f}")
        print(f"  INDIV   {i}")
    print("TOTAL $%.2f" % USAGE.cost_usd)


if __name__ == "__main__":
    main()
