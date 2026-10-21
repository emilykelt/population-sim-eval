"""Turn results/<case>/summary.json into a markdown table with the human baselines alongside."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(case_yaml: str) -> str:
    case = yaml.safe_load(Path(case_yaml).read_text())
    out_dir = ROOT / "results" / case["name"]
    rows = json.loads((out_dir / "summary.json").read_text())
    diag = case["diagnostics"]
    lines = [f"## {case['name']}", "", f"Source: {case['source']}", ""]
    lines.append("Human reference points (Spearman, segment ranking of the 6 variants):")
    for k, v in diag.items():
        lines.append(f"- {k}: predict every segment with the pooled-population ranking = **{v['pooled_baseline_spearman']}**; split-half human noise ceiling = **{v['split_half_ceiling_spearman']}**")
    lines += ["", "| model | method | mean rho | top-1 acc | gap preservation | within-seg SD | valid answers | cost $ | secs |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["model"], r["method"])):
        lines.append(f"| {r['model']} | {r['method']} | {r['mean_spearman']:.3f} | {r['top1_accuracy']:.2f} | {r['gap_preservation']:.3f} | {r['within_sd']:.1f} | {r['n_valid_answers']}/{r['n_jobs']} | {r['cost_usd']:.3f} | {r['seconds']} |")
        if "round1" in r:
            r1 = r["round1"]
            lines.append(f"| {r['model']} | ... debate round 1 only | {r1['mean_spearman']:.3f} | {r1['top1_accuracy']:.2f} | {r1['gap_preservation']:.3f} | | {r1['n_valid_answers']} | | |")
    lines.append("")
    # per-segment detail for the best method per model
    for model in sorted({r["model"] for r in rows}):
        best = max((r for r in rows if r["model"] == model), key=lambda r: r["mean_spearman"])
        lines.append(f"Best for {model}: {best['method']} (rho {best['mean_spearman']:.3f}). Per segment:")
        for seg, d in best["per_segment"].items():
            lines.append(f"- {seg}: rho {d['spearman']:.2f}, top1 {'hit' if d['top1_hit'] else 'miss'}; pred {d['pred']} vs truth {d['truth']}")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "REPORT.md").write_text(text)
    return text


if __name__ == "__main__":
    print(main(sys.argv[1]))
