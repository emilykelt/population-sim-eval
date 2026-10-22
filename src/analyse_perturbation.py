"""Analyse results/<case>_perturb/rows.json: variance decomposition over prompt components, sign-flip rates,
and a cost-fidelity frontier with wording spread.

  PYTHONPATH=src uv run python src/analyse_perturbation.py clarke2020_study2_labels
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def anova_share(df: pd.DataFrame, y: str) -> pd.Series:
    """Share of sum of squares explained by each prompt component (type II ANOVA), per model x design."""
    d = df.dropna(subset=[y]).copy()
    for c in ["lv_system", "lv_task", "lv_arms", "lv_format"]:
        d[c] = d[c].astype(str)
    m = smf.ols(f"{y} ~ C(lv_system) + C(lv_task) + C(lv_arms) + C(lv_format)", data=d).fit()
    tab = sm.stats.anova_lm(m, typ=2)
    tot = tab["sum_sq"].sum()
    return (tab["sum_sq"] / tot).rename(lambda s: s.replace("C(lv_", "").replace(")", ""))


def main(case: str) -> None:
    d = ROOT / "results" / f"{case}_perturb"
    rows = json.loads((d / "rows.json").read_text())
    df = pd.DataFrame(rows)
    lines = [f"# Prompt-perturbation analysis: {case}", ""]
    lines.append("25 wordings per model x design (5x5 Graeco-Latin square over system / task / arm-description / format).")
    lines.append("")
    lines.append("## Score distribution over wordings")
    lines.append("")
    lines.append("| model | design | n | rho median [IQR] | rho min..max | top-1 median | pooled rho median [IQR] | base rate median [IQR] | $ total |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for (mdl, des), g in df.groupby(["model", "design"]):
        q = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]"
        br = q(g["base_rate"]) if g["base_rate"].notna().any() else "n.a."
        lines.append(f"| {mdl} | {des} | {len(g)} | {q(g['rho'])} | {g['rho'].min():.2f}..{g['rho'].max():.2f} | {g['top1'].median():.2f} | {q(g['pooled_rho'])} | {br} | {g['cost'].sum():.2f} |")
    lines.append("")

    lines.append("## Variance decomposition (share of sum of squares, type II ANOVA on the 25 wordings)")
    lines.append("")
    lines.append("| model | design | outcome | system | task | arms | format | residual |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (mdl, des), g in df.groupby(["model", "design"]):
        for y in ["rho", "pooled_rho", "base_rate"]:
            if g[y].notna().sum() < 10 or g[y].std() == 0:
                continue
            s = anova_share(g, y)
            lines.append(f"| {mdl} | {des} | {y} | " + " | ".join(f"{s.get(k, 0):.2f}" for k in ["system", "task", "arms", "format", "Residual"]) + " |")
    lines.append("")

    # between-model vs within-model (wording) variance for rho
    lines.append("## Wording variance vs model variance")
    lines.append("")
    for des, g in df.groupby("design"):
        if g["model"].nunique() < 2:
            lines.append(f"- {des}: only one model run so far; wording SD of rho = {g['rho'].std():.3f}")
            continue
        within = g.groupby("model")["rho"].var().mean()
        between = g.groupby("model")["rho"].mean().var()
        lines.append(f"- {des}: mean within-model (wording) variance of rho = {within:.4f}; between-model variance of mean rho = {between:.4f}; ratio wording/model = {within / max(between, 1e-9):.2f}")
    lines.append("")

    lines.append("## Sign-flip rate of the top-1 pick per segment")
    lines.append("")
    lines.append("Fraction of wordings whose top-1 pick for a segment differs from the modal pick across wordings (0 = perfectly stable).")
    lines.append("")
    lines.append("| model | design | mean flip rate | worst segment (rate) | modal picks |")
    lines.append("|---|---|---|---|---|")
    for (mdl, des), g in df.groupby(["model", "design"]):
        segs = {}
        for t in g["top1_by_seg"]:
            for seg, pick in t.items():
                segs.setdefault(seg, []).append(pick)
        rates, modal = {}, {}
        for seg, picks in segs.items():
            c = Counter(picks).most_common(1)[0]
            modal[seg] = c[0]
            rates[seg] = 1 - c[1] / len(picks)
        worst = max(rates, key=rates.get)
        short = {k.split("|")[0].split("=")[-1] + "/" + k.split("|")[-1].split("=")[-1]: v for k, v in modal.items()}
        lines.append(f"| {mdl} | {des} | {np.mean(list(rates.values())):.2f} | {worst} ({rates[worst]:.2f}) | {short} |")
    lines.append("")

    # frontier figure
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for (mdl, des), g in df.groupby(["model", "design"]):
        x = g["cost"].mean()
        ax.errorbar(x, g["rho"].median(), yerr=[[g["rho"].median() - g["rho"].quantile(.25)], [g["rho"].quantile(.75) - g["rho"].median()]],
                    fmt="s" if des == "choice" else "o", capsize=3, label=f"{mdl.replace('claude-', '')} / {des}")
    ax.set_xscale("log")
    ax.set_xlabel("cost per wording (USD, log)")
    ax.set_ylabel("segment-level Spearman rho (median, IQR over 25 wordings)")
    ax.axhline(0.82, ls=":", c="grey", lw=1)
    ax.text(ax.get_xlim()[0], 0.83, "pooled-population baseline", fontsize=7, color="grey")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(d / "frontier.png", dpi=160)
    plt.savefig(d / "frontier.pdf")

    # per-wording strip plot
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    keys = sorted(df.groupby(["model", "design"]).groups)
    for i, k in enumerate(keys):
        g = df[(df["model"] == k[0]) & (df["design"] == k[1])]
        ax.scatter(np.full(len(g), i) + np.random.uniform(-.12, .12, len(g)), g["rho"], s=12, alpha=.7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k[0].replace('claude-', '')}\n{k[1]}" for k in keys], fontsize=7)
    ax.set_ylabel("rho per wording")
    ax.axhline(0, c="grey", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(d / "strip.png", dpi=160)

    text = "\n".join(lines)
    (d / "ANALYSIS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
