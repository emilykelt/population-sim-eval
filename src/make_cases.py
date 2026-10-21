"""Build benchmark case YAMLs from Clarke et al. (2020) OSF data (osf.io/k7tw5), and report
(a) the pooled-population baseline (predict every segment with the overall ranking) and
(b) a split-half noise ceiling for the human segment rankings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "clarke2020_raw"

AGE_BINS, AGE_LABELS = [17, 34, 54, 120], ["18-34", "35-54", "55+"]


def seg_table(df: pd.DataFrame, cond: str, y: str, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys + [cond], observed=True)[y].mean().unstack()


def ceiling(df: pd.DataFrame, cond: str, y: str, keys: list[str], reps: int = 200, seed: int = 0) -> float:
    """Mean Spearman between segment rankings computed on two random halves of the human sample."""
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(reps):
        mask = rng.random(len(df)) < 0.5
        a = seg_table(df[mask], cond, y, keys)
        b = seg_table(df[~mask], cond, y, keys)
        for seg in a.index:
            if seg in b.index:
                rhos.append(spearmanr(a.loc[seg], b.loc[seg]).correlation)
    return float(np.nanmean(rhos))


def pooled_baseline(tab: pd.DataFrame, pooled: pd.Series) -> float:
    return float(np.mean([spearmanr(tab.loc[s], pooled).correlation for s in tab.index]))


def cells_from_index(tab: pd.DataFrame, keys: list[str], extra: dict) -> list[dict]:
    out = []
    for idx, row in tab.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        cell = dict(extra)
        cell.update({k: str(v) for k, v in zip(keys, idx)})
        out.append({"cell": cell, "truth": {str(c): round(float(v), 4) for c, v in row.items()}, "n": None})
    return out


def study2() -> dict:
    s2 = pd.read_excel(RAW / "Study2" / "FoodLabellingOnline2Dataset.xlsx")
    s2 = s2.rename(columns={"Primary outcome": "chose_energy_dense", "Condition": "cond"})
    s2 = s2[s2["Gender"].isin([1, 2])].copy()
    s2["effective"] = 1 - s2["chose_energy_dense"]  # higher = label pushed people away from energy-dense snack
    s2["age_band"] = pd.cut(s2["Age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    s2["gender"] = s2["Gender"].map({1: "man", 2: "woman"})
    s2["education"] = np.where(s2["Education"].isin([5, 6]), "university degree", "no university degree")
    s2["cond"] = s2["cond"].map({1: "pictorial_plus_calories", 2: "text_plus_calories", 3: "calories_only",
                                 4: "pictorial", 5: "text_only", 6: "no_label"})
    messages = {
        "pictorial_plus_calories": "Every high-calorie snack (crisps, chocolate, shortbread, dessert) carries a large warning label with a graphic medical photograph (a bowel tumour, open-heart surgery, or a bleeding diabetic eye) and the words 'Excess calories cause obesity which causes bowel cancer / heart disease / Type 2 diabetes'. Every snack, healthy or not, also shows its calorie count in a bold badge, e.g. '240 CALORIES'.",
        "text_plus_calories": "Every high-calorie snack carries a plain black-and-white text warning: 'Excess calories cause obesity which causes bowel cancer / heart disease / Type 2 diabetes'. No pictures. Every snack, healthy or not, also shows its calorie count in a bold badge, e.g. '240 CALORIES'.",
        "calories_only": "No warnings. Every snack, healthy or not, shows its calorie count in a bold badge, e.g. '240 CALORIES' on a Dairy Milk bar, '95 CALORIES' on a fruit pot.",
        "pictorial": "Every high-calorie snack carries a large warning label with a graphic medical photograph (a bowel tumour, open-heart surgery, or a bleeding diabetic eye) and the words 'Excess calories cause obesity which causes bowel cancer / heart disease / Type 2 diabetes'. No calorie counts anywhere.",
        "text_only": "Every high-calorie snack carries a plain black-and-white text warning: 'Excess calories cause obesity which causes bowel cancer / heart disease / Type 2 diabetes'. No pictures, no calorie counts.",
        "no_label": "The snacks are shown exactly as they appear in the shops, with no extra labels of any kind.",
    }
    segs = []
    for keys in (["gender", "age_band"], ["education"]):
        tab = seg_table(s2, "cond", "effective", keys)
        segs += cells_from_index(tab, keys, {"country": "UK"})
        for s, k in zip(segs[-len(tab):], tab.index):
            s["n"] = int(s2.groupby(keys, observed=True).size().loc[k])
    pooled = s2.groupby("cond")["effective"].mean()
    diag = {}
    for keys in (["gender", "age_band"], ["education"]):
        tab = seg_table(s2, "cond", "effective", keys)
        diag["+".join(keys)] = {"pooled_baseline_spearman": round(pooled_baseline(tab, pooled), 3),
                               "split_half_ceiling_spearman": round(ceiling(s2, "cond", "effective", keys), 3)}
    return {
        "name": "clarke2020_study2_labels",
        "source": "Clarke et al. 2020, Appetite; data osf.io/k7tw5 Study 2 (N=4,134 UK adults, 6 arms, primary outcome = chose energy-dense snack in a 12-item vending machine task). Truth = 1 - P(chose energy-dense) per segment per arm.",
        "outcome_hint": "Concretely: imagine a vending machine with 12 snacks, six high-calorie (Doritos, McCoy's, Dairy Milk, Galaxy, millionaire shortbread, a Gü chocolate dessert) and six lower-calorie (apple crisps, beetroot crisps, a Müller yoghurt, fruit pots, a KIND bar). Each option below is a different way the snacks could be labelled. Give more points to the labelling scheme that would most push YOU, today, toward picking one of the lower-calorie snacks instead of a high-calorie one.",
        "messages": messages,
        "pooled_truth": {str(k): round(float(v), 4) for k, v in pooled.items()},
        "diagnostics": diag,
        "segments": segs,
    }


def study1() -> dict:
    s1 = pd.read_excel(RAW / "Study1" / "Food labelling study 1 final dataset.xlsx")
    s1 = s1.rename(columns={"Health condition": "cond", "Average N.E.A ": "nea", "Desirability (scored 1-7)": "desire",
                            "Education ": "edu_raw"})
    s1 = s1[s1["Gender"].isin(["Male", "Female"]) & (s1["Age"] >= 18)].copy()
    s1["age_band"] = pd.cut(s1["Age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    s1["gender"] = s1["Gender"].map({"Male": "man", "Female": "woman"})
    s1["education"] = np.where(s1["edu_raw"].astype(str).str.contains("Bachelor|Post-Graduate"), "university degree", "no university degree")
    s1["cond"] = s1["cond"].map({"Bowel cancer": "bowel_cancer", "Non-specified cancer": "cancer", "13 types of cancer": "thirteen_cancers",
                                 "Type 2 diabetes": "type2_diabetes", "Heart disease": "heart_disease", "Obesity": "obesity"})
    s1["effective"] = 8 - s1["desire"]  # lower desire to eat the chocolate bar = label worked
    messages = {
        "bowel_cancer": "A warning label on a chocolate bar: 'Excess calories cause obesity which causes bowel cancer', above a graphic photograph of a cancerous bowel.",
        "cancer": "A warning label on a chocolate bar: 'Excess calories cause obesity which causes cancer', above a photograph of cancer surgery.",
        "thirteen_cancers": "A warning label on a chocolate bar: 'Excess calories cause obesity which causes 13 types of cancer', above a photograph of cancer surgery.",
        "type2_diabetes": "A warning label on a chocolate bar: 'Excess calories cause obesity which causes Type 2 diabetes', above a photograph of a bleeding diabetic eye.",
        "heart_disease": "A warning label on a chocolate bar: 'Excess calories cause obesity which causes heart disease', above a photograph of open-heart surgery.",
        "obesity": "A warning label on a chocolate bar: 'Excess calories cause obesity', above a photograph of three overweight people seen from behind.",
    }
    segs = []
    for keys in (["gender", "age_band"], ["education"]):
        tab = seg_table(s1, "cond", "effective", keys)
        new = cells_from_index(tab, keys, {"country": "UK"})
        for s, k in zip(new, tab.index):
            s["n"] = int(s1.groupby(keys, observed=True).size().loc[k])
        segs += new
    pooled = s1.groupby("cond")["effective"].mean()
    diag = {}
    for keys in (["gender", "age_band"], ["education"]):
        tab = seg_table(s1, "cond", "effective", keys)
        diag["+".join(keys)] = {"pooled_baseline_spearman": round(pooled_baseline(tab, pooled), 3),
                               "split_half_ceiling_spearman": round(ceiling(s1, "cond", "effective", keys), 3)}
    # secondary truth: negative emotional arousal (the study's own primary outcome)
    nea = {}
    for keys in (["gender", "age_band"], ["education"]):
        tab = seg_table(s1, "cond", "nea", keys)
        for idx, row in tab.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            nea["|".join(f"{k}={v}" for k, v in sorted(dict(zip(keys, idx), country="UK").items()))] = {str(c): round(float(v), 3) for c, v in row.items()}
    return {
        "name": "clarke2020_study1_framing",
        "source": "Clarke et al. 2020 protocol Study 1; data osf.io/k7tw5 (N=4,607 UK adults, 18 pictorial labels collapsed to 6 disease framings). Truth = 8 - desire to eat the chocolate bar (1-7) per segment per framing; secondary truth = negative emotional arousal.",
        "outcome_hint": "Concretely: you are shown a Dairy Milk bar carrying one of these warning labels and asked how much you want to eat it right now. Give more points to the label that would most reduce YOUR desire to eat the bar.",
        "messages": messages,
        "pooled_truth": {str(k): round(float(v), 4) for k, v in pooled.items()},
        "secondary_truth_nea": nea,
        "diagnostics": diag,
        "segments": segs,
    }


if __name__ == "__main__":
    for case in (study1(), study2()):
        path = ROOT / "data" / f"{case['name']}.yaml"
        path.write_text(yaml.safe_dump(case, sort_keys=False, allow_unicode=True, width=1000))
        print(case["name"], json.dumps(case["diagnostics"]), "pooled:", case["pooled_truth"])
