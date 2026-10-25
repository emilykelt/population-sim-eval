"""Three fidelity families, computed from a human table and a simulated table with identical columns:

    id, arm, segment keys..., covariates..., y

  EFFECT fidelity        does the simulation reproduce how arms differ, overall and within segments
  DISTRIBUTIONAL fidelity does the simulated response distribution per arm (and per segment) match the human one
  MECHANISM fidelity      do the same covariates and the same arm x covariate interactions predict y in both
                          (structural proxy; true mechanism needs interpretability access)
  INDIVIDUAL fidelity     when twins are matched to real respondents: twin outcome in the real person's arm vs
                          that person's actual outcome, benchmarked against a demographic lookup table

Every function returns plain dicts so results can be dumped to JSON.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance


# ----------------------------------------------------------------------------- helpers
def _arm_means(df: pd.DataFrame, arms: list[str]) -> np.ndarray:
    m = df.groupby("arm")["y"].mean()
    return np.array([m.get(a, np.nan) for a in arms])


def _seg_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    return df[keys].apply(lambda r: "|".join(str(v) for v in r.values), axis=1)


def split_half_ceiling(human: pd.DataFrame, arms: list[str], keys: list[str], reps: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    h = human.assign(seg=_seg_key(human, keys))
    rhos = []
    for _ in range(reps):
        mask = rng.random(len(h)) < 0.5
        for seg, ga in h[mask].groupby("seg"):
            gb = h[~mask & (h["seg"] == seg)]
            if len(gb) == 0:
                continue
            a, b = _arm_means(ga, arms), _arm_means(gb, arms)
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() >= 3:
                rhos.append(spearmanr(a[ok], b[ok]).correlation)
    return float(np.nanmean(rhos)) if rhos else np.nan


# ----------------------------------------------------------------------------- effect fidelity
def effect_fidelity(human: pd.DataFrame, sim: pd.DataFrame, arms: list[str], keys: list[str]) -> dict:
    hp, sp = _arm_means(human, arms), _arm_means(sim, arms)
    pooled_rho = float(spearmanr(hp, sp).correlation)
    # effect sizes vs the first arm listed (treated as control if the task says so)
    h_eff, s_eff = hp - hp[0], sp - sp[0]
    slope = float(np.polyfit(h_eff[1:], s_eff[1:], 1)[0]) if len(arms) > 2 and np.std(h_eff[1:]) > 0 else np.nan
    h = human.assign(seg=_seg_key(human, keys))
    s = sim.assign(seg=_seg_key(sim, keys))
    per_seg, rhos, hits = {}, [], []
    hseg, sseg = {}, {}
    for seg, gh in h.groupby("seg"):
        gs = s[s["seg"] == seg]
        if len(gs) == 0:
            continue
        a, b = _arm_means(gh, arms), _arm_means(gs, arms)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 3:
            continue
        rho = spearmanr(a[ok], b[ok]).correlation
        hit = float(np.nanargmax(a) == np.nanargmax(b))
        rhos.append(rho); hits.append(hit); hseg[seg] = a; sseg[seg] = b
        per_seg[seg] = {"rho": float(rho), "top1_hit": hit, "human": np.round(a, 3).tolist(), "sim": np.round(b, 3).tolist(), "n_human": int(len(gh)), "n_sim": int(len(gs))}
    pooled_baseline = float(np.nanmean([spearmanr(v, hp).correlation for v in hseg.values()])) if hseg else np.nan
    # gap preservation: between-segment differences arm by arm
    gap = np.nan
    segs = list(hseg)
    if len(segs) >= 2:
        hd, sd = [], []
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                hd += (hseg[segs[i]] - hseg[segs[j]]).tolist(); sd += (sseg[segs[i]] - sseg[segs[j]]).tolist()
        hd, sd = np.array(hd), np.array(sd); ok = ~(np.isnan(hd) | np.isnan(sd))
        if ok.sum() > 3 and np.std(hd[ok]) > 0 and np.std(sd[ok]) > 0:
            gap = float(pearsonr(hd[ok], sd[ok])[0])
            gap_slope = float(np.polyfit(hd[ok], sd[ok], 1)[0])
        else:
            gap_slope = np.nan
    else:
        gap_slope = np.nan
    return {
        "pooled_rho": pooled_rho, "effect_size_slope": slope,
        "segment_rho_mean": float(np.nanmean(rhos)) if rhos else np.nan,
        "top1_accuracy": float(np.mean(hits)) if hits else np.nan,
        "pooled_baseline": pooled_baseline, "gap_preservation_r": gap, "gap_inflation_slope": gap_slope,
        "human_arm_means": np.round(hp, 3).tolist(), "sim_arm_means": np.round(sp, 3).tolist(),
        "per_segment": per_seg,
    }


# ----------------------------------------------------------------------------- distributional fidelity
def _dist(values: pd.Series, support: list | None) -> np.ndarray:
    if support is None:
        return values.to_numpy(dtype=float)
    return np.array([(values == v).mean() for v in support])


def distributional_fidelity(human: pd.DataFrame, sim: pd.DataFrame, arms: list[str], keys: list[str], support: list | None = None) -> dict:
    """support = list of discrete outcome values (Likert points, or [0,1]); None = treat y as continuous.
    Reports per arm: Wasserstein-1 (in outcome units), total variation (discrete only), dispersion ratio."""
    out, per_arm = {}, {}
    w, tv, disp = [], [], []
    for a in arms:
        hy, sy = human.loc[human["arm"] == a, "y"], sim.loc[sim["arm"] == a, "y"]
        if len(hy) < 5 or len(sy) < 5:
            continue
        wd = float(wasserstein_distance(hy, sy))
        t = float(0.5 * np.abs(_dist(hy, support) - _dist(sy, support)).sum()) if support is not None else np.nan
        dr = float(sy.std() / hy.std()) if hy.std() > 0 else np.nan
        w.append(wd); tv.append(t); disp.append(dr)
        per_arm[a] = {"wasserstein": wd, "tv": t, "dispersion_ratio": dr, "human_mean": float(hy.mean()), "sim_mean": float(sy.mean()), "human_sd": float(hy.std()), "sim_sd": float(sy.std())}
    # segment-level mean absolute error of rates / means
    h = human.assign(seg=_seg_key(human, keys)); s = sim.assign(seg=_seg_key(sim, keys))
    hm = h.groupby(["seg", "arm"])["y"].mean(); sm = s.groupby(["seg", "arm"])["y"].mean()
    common = hm.index.intersection(sm.index)
    mae = float((hm[common] - sm[common]).abs().mean()) if len(common) else np.nan
    bias = float((sm[common] - hm[common]).mean()) if len(common) else np.nan
    out.update({
        "wasserstein_mean": float(np.mean(w)) if w else np.nan,
        "tv_mean": float(np.nanmean(tv)) if tv else np.nan,
        "dispersion_ratio_mean": float(np.nanmean(disp)) if disp else np.nan,
        "segment_arm_mae": mae, "segment_arm_bias": bias,
        "human_pooled_sd": float(human["y"].std()), "sim_pooled_sd": float(sim["y"].std()),
        "per_arm": per_arm,
    })
    return out


# ----------------------------------------------------------------------------- mechanism (structural) fidelity
def mechanism_fidelity(human: pd.DataFrame, sim: pd.DataFrame, covariates: list[str], arms: list[str]) -> dict:
    """Fit y ~ covariates + arm (+ arm x covariates) by OLS on each table with standardised covariates; compare.
    Reports: correlation and sign agreement of main-effect coefficients, same for interaction coefficients,
    and variance in y explained by covariates alone in each table (over-determination check)."""
    import statsmodels.api as sm

    def design(df: pd.DataFrame, with_inter: bool):
        X = pd.DataFrame(index=df.index)
        for c in covariates:
            v = pd.to_numeric(df[c], errors="coerce")
            X[c] = (v - v.mean()) / (v.std() if v.std() > 0 else 1)
        for a in arms[1:]:
            X[f"arm_{a}"] = (df["arm"] == a).astype(float)
            if with_inter:
                for c in covariates:
                    X[f"arm_{a}:{c}"] = X[f"arm_{a}"] * X[c]
        X = sm.add_constant(X.fillna(0))
        return X

    def fit(df, with_inter):
        X = design(df, with_inter)
        m = sm.OLS(df["y"].astype(float), X).fit()
        return m

    try:
        hm, smod = fit(human, False), fit(sim, False)
        hi, si = fit(human, True), fit(sim, True)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    main_names = covariates
    h_main = np.array([hm.params.get(c, np.nan) for c in main_names]); s_main = np.array([smod.params.get(c, np.nan) for c in main_names])
    inter_names = [c for c in hi.params.index if ":" in c]
    h_int = np.array([hi.params[c] for c in inter_names]); s_int = np.array([si.params.get(c, np.nan) for c in inter_names])

    def agree(a, b):
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 3:
            return np.nan, np.nan
        return float(pearsonr(a[ok], b[ok])[0]), float((np.sign(a[ok]) == np.sign(b[ok])).mean())

    r_main, sign_main = agree(h_main, s_main)
    r_int, sign_int = agree(h_int, s_int)
    # covariate-only R2: how much of y is "explained" by who you are, human vs sim
    def cov_r2(df):
        X = design(df, False)[["const"] + covariates]
        return float(sm.OLS(df["y"].astype(float), X).fit().rsquared)
    return {
        "main_effect_r": r_main, "main_effect_sign_agreement": sign_main,
        "interaction_r": r_int, "interaction_sign_agreement": sign_int,
        "covariate_r2_human": cov_r2(human), "covariate_r2_sim": cov_r2(sim),
        "human_main": dict(zip(main_names, np.round(h_main, 3).tolist())),
        "sim_main": dict(zip(main_names, np.round(s_main, 3).tolist())),
        "arm_r2_human": float(hm.rsquared - cov_r2(human)), "arm_r2_sim": float(smod.rsquared - cov_r2(sim)),
    }


# ----------------------------------------------------------------------------- individual fidelity
def individual_fidelity(matched: pd.DataFrame, keys: list[str], binary: bool, human: pd.DataFrame | None = None) -> dict:
    """matched: one row per twin that was run in its real person's arm, columns y_human, y_sim, segment keys, arm.
    Baseline: predict each person with the mean of their (segment, arm) cell in the FULL human table, excluding
    the person themselves (leave-one-out lookup table)."""
    m = matched.dropna(subset=["y_human", "y_sim"]).copy()
    if len(m) < 10:
        return {"n": int(len(m))}
    m["seg"] = _seg_key(m, keys)
    if human is not None:
        h = human.assign(seg=_seg_key(human, keys))
        g = h.groupby(["seg", "arm"])["y"].agg(["sum", "count"])
        def loo(r):
            s_, c = g.loc[(r["seg"], r["arm"])] if (r["seg"], r["arm"]) in g.index else (np.nan, 0)
            return (s_ - r["y_human"]) / (c - 1) if c > 1 else np.nan
        cell_mean = m.apply(loo, axis=1)
    else:
        cell_mean = m.groupby(["seg", "arm"])["y_human"].transform("mean")
    if binary:
        acc_sim = float((m["y_sim"].round() == m["y_human"]).mean())
        acc_lookup = float(((cell_mean > 0.5).astype(float) == m["y_human"]).mean())
        acc_majority = float(max(m["y_human"].mean(), 1 - m["y_human"].mean()))
        return {"n": int(len(m)), "accuracy_sim": acc_sim, "accuracy_lookup": acc_lookup, "accuracy_majority": acc_majority}
    mae_sim = float((m["y_sim"] - m["y_human"]).abs().mean()); mae_lookup = float((cell_mean - m["y_human"]).abs().mean())
    r = float(pearsonr(m["y_sim"], m["y_human"])[0]) if m["y_sim"].std() > 0 else np.nan
    return {"n": int(len(m)), "mae_sim": mae_sim, "mae_lookup": mae_lookup, "r_sim_human": r}
