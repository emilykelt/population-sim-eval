"""Scoring a method's predictions against human ground truth.

Ground truth per segment: a vector of human effect sizes / mean outcomes per message variant.
Prediction per segment: mean allocated points per message variant across personas.

Metrics
  spearman_by_segment : rank agreement within each segment (the product question: which variant for whom)
  top1_hit            : did the method pick the human-best variant in the segment
  gap_preservation    : correlation between human and simulated between-segment DIFFERENCES per message
                        (does the model know that message X lands with segment 1 but not segment 2?)
  flattening_index    : simulated within-segment SD of points / human within-segment SD of outcome (if available).
                        << 1 means the silicon crowd is more homogeneous than real people.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, pearsonr


def aggregate(results: list[dict], msgs: list[str]) -> dict[str, dict]:
    """Group results by cell key -> {message_id: [points...]} after undoing the per-run shuffle."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if not r.get("points"):
            continue
        key = cell_key(r["cell"])
        bucket = out.setdefault(key, {m: [] for m in msgs})
        for k, i in enumerate(r["order"]):
            bucket[msgs[i]].append(r["points"][k])
    return out


def cell_key(cell: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(cell.items()))


def score(pred: dict[str, dict[str, list[float]]], truth: dict[str, dict[str, float]], msgs: list[str]) -> dict:
    per_seg = {}
    rhos, hits = [], []
    pred_means, truth_means = {}, {}
    for seg, tvals in truth.items():
        if seg not in pred:
            continue
        pm = np.array([np.mean(pred[seg][m]) if pred[seg][m] else np.nan for m in msgs])
        tv = np.array([tvals[m] for m in msgs])
        if np.isnan(pm).any():
            continue
        rho = spearmanr(pm, tv).correlation if len(msgs) > 2 else np.sign((pm[0] - pm[1]) * (tv[0] - tv[1]))
        hit = float(msgs[int(np.argmax(pm))] == msgs[int(np.argmax(tv))])
        rhos.append(rho)
        hits.append(hit)
        pred_means[seg] = pm
        truth_means[seg] = tv
        per_seg[seg] = {"spearman": float(rho), "top1_hit": hit, "pred": pm.round(1).tolist(), "truth": tv.round(3).tolist()}

    gap = np.nan
    segs = list(pred_means)
    if len(segs) >= 2:
        # between-segment differences, message by message, for every segment pair
        pd, td = [], []
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                pd.extend((pred_means[segs[a]] - pred_means[segs[b]]).tolist())
                td.extend((truth_means[segs[a]] - truth_means[segs[b]]).tolist())
        if np.std(pd) > 0 and np.std(td) > 0:
            gap = float(pearsonr(pd, td)[0])

    n_valid = sum(len(v) for seg in pred.values() for v in seg.values())
    return {
        "mean_spearman": float(np.nanmean(rhos)) if rhos else np.nan,
        "top1_accuracy": float(np.mean(hits)) if hits else np.nan,
        "gap_preservation": gap,
        "n_segments_scored": len(rhos),
        "n_valid_answers": n_valid,
        "per_segment": per_seg,
    }


def within_sd(pred: dict[str, dict[str, list[float]]]) -> float:
    sds = [np.std(v) for seg in pred.values() for v in seg.values() if len(v) > 1]
    return float(np.mean(sds)) if sds else np.nan


def aggregate_choice(results: list[dict], msgs: list[str]) -> dict[str, dict[str, list[float]]]:
    """Choice replication: per cell per arm, list of 'effective' indicators (1 = chose a lower-energy snack)."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r.get("energy_dense") is None:
            continue
        bucket = out.setdefault(cell_key(r["cell"]), {m: [] for m in msgs})
        bucket[r["arm"]].append(0.0 if r["energy_dense"] else 1.0)
    return out
