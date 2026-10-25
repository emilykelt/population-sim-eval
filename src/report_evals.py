"""Summarise results/evals/*/<model>.json into one markdown report with the three fidelity families."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "results" / "evals"


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}" if x is not None and x == x else "n.a."
    except (TypeError, ValueError):
        return "n.a."


def main() -> str:
    rows = []
    for d in sorted(EV.iterdir()):
        for p in sorted(d.glob("*.json")):
            if p.name.endswith("_raw.json"):
                continue
            r = json.loads(p.read_text())
            r["dataset"] = d.name
            rows.append(r)
    L = ["# Three-fidelity eval: results", "", "One row per dataset x model. 150 real-respondent twins per dataset, each run in every arm. Human reference lines are computed from the human data alone.", ""]
    L += ["## Effect fidelity (do arms differ the way they did for humans)", "",
          "| dataset | model | pooled rho | effect-size slope | segment rho | pooled baseline | ceiling | top-1 | gap r | gap slope | valid |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["effect"]
        L.append(f"| {r['dataset']} | {r['model']} | {f(e['pooled_rho'])} | {f(e['effect_size_slope'])} | {f(e['segment_rho_mean'])} | {f(e['pooled_baseline'])} | {f(r['ceiling_split_half'])} | {f(e['top1_accuracy'])} | {f(e['gap_preservation_r'])} | {f(e['gap_inflation_slope'])} | {r['n_valid']}/{r['n_jobs']} |")
    L += ["", "Effect-size slope: simulated arm effect (vs control) regressed on human arm effect; 1 = calibrated, >1 exaggerated, <0 wrong direction. Gap slope: same for between-segment differences.", ""]
    L += ["## Distributional fidelity (does the response distribution per arm match)", "",
          "| dataset | model | outcome | W1 (outcome units) | TV | dispersion ratio | segment-arm MAE | bias | human SD | sim SD |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d_ = r["distribution"]
        L.append(f"| {r['dataset']} | {r['model']} | {OUTCOMES.get(r['dataset'], '')} | {f(d_['wasserstein_mean'], 3)} | {f(d_['tv_mean'], 3)} | {f(d_['dispersion_ratio_mean'])} | {f(d_['segment_arm_mae'], 3)} | {f(d_['segment_arm_bias'], 3)} | {f(d_['human_pooled_sd'], 3)} | {f(d_['sim_pooled_sd'], 3)} |")
    L += ["", "Dispersion ratio = simulated SD / human SD within arm; <1 = collapsed, >1 = over-dispersed. Bias = mean(sim) - mean(human) over segment x arm cells.", ""]
    L += ["## Mechanism (structural) fidelity (do the same covariates drive the outcome)", "",
          "| dataset | model | main-effect r | sign agreement | interaction r | covariate R2 human | covariate R2 sim | arm R2 human | arm R2 sim |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["mechanism"]
        if "error" in m:
            L.append(f"| {r['dataset']} | {r['model']} | error: {m['error'][:60]} | | | | | | |")
            continue
        L.append(f"| {r['dataset']} | {r['model']} | {f(m['main_effect_r'])} | {f(m['main_effect_sign_agreement'])} | {f(m['interaction_r'])} | {f(m['covariate_r2_human'], 3)} | {f(m['covariate_r2_sim'], 3)} | {f(m['arm_r2_human'], 3)} | {f(m['arm_r2_sim'], 3)} |")
    L += ["", "Covariate R2: share of outcome variance explained by who the person is (age, sex, education, income, BMI, habits). If sim >> human, the model over-determines behaviour from demographics. Arm R2: variance explained by the treatment.", ""]
    L += ["## Individual fidelity (twin in the real person's own arm vs that person's outcome)", "",
          "| dataset | model | n | sim | lookup-table baseline | majority / r |", "|---|---|---|---|---|---|"]
    for r in rows:
        i = r["individual"]
        if "accuracy_sim" in i:
            L.append(f"| {r['dataset']} | {r['model']} | {i['n']} | acc {f(i['accuracy_sim'])} | acc {f(i['accuracy_lookup'])} | majority {f(i['accuracy_majority'])} |")
        elif "mae_sim" in i:
            L.append(f"| {r['dataset']} | {r['model']} | {i['n']} | MAE {f(i['mae_sim'])} | MAE {f(i['mae_lookup'])} | r {f(i['r_sim_human'])} |")
        else:
            L.append(f"| {r['dataset']} | {r['model']} | {i.get('n')} | | | |")
    L += ["", "## Per-arm means (human vs simulated)", ""]
    for r in rows:
        e = r["effect"]
        L.append(f"- **{r['dataset']} / {r['model']}**: human {e['human_arm_means']} vs sim {e['sim_arm_means']}; cost ${r['cost_usd']}, {r['seconds']}s")
        for seg, v in e["per_segment"].items():
            L.append(f"  - {seg}: rho {f(v['rho'])}, top1 {'hit' if v['top1_hit'] else 'miss'}, n_h {v['n_human']}")
    text = "\n".join(L)
    (EV / "REPORT.md").write_text(text)
    return text


OUTCOMES = {"clarke2020_s2": "P(lower-energy snack)", "evans2025_salt": "total salt g", "kerr2021_s1": "willingness 1-5", "moraplazas2024_drink": "P(no-sugar drink)"}

if __name__ == "__main__":
    print(main())
