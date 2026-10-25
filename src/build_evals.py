"""Convert raw datasets into eval specs: data/evals/<name>.yaml + <name>_human.csv + <name>_twins.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import yaml

from simsoc.twins import Twin, save_twins

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evals"
OUT.mkdir(exist_ok=True)
N_TWINS = 150


def band(age):
    return pd.cut(age, [17, 34, 54, 120], labels=["18-34", "35-54", "55+"]).astype(str)


def write(name, spec, human, twins):
    human.to_csv(OUT / f"{name}_human.csv", index=False)
    save_twins(twins, OUT / f"{name}_twins.json")
    spec.update(name=name, human_table=f"data/evals/{name}_human.csv", twins=f"data/evals/{name}_twins.json")
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000))
    print(name, "humans", len(human), "twins", len(twins), "arms", human.groupby("arm")["y"].mean().round(3).to_dict())


def make_twins(df, name, covariate_cols, numeric_cols, seed=0):
    rng = np.random.default_rng(seed)
    per_arm = N_TWINS // df["arm"].nunique() + 1
    parts = [g.sample(min(per_arm, len(g)), random_state=seed) for _, g in df.groupby("arm")]
    rows = pd.concat(parts).sample(frac=1, random_state=seed).head(N_TWINS)
    twins = []
    for _, r in rows.iterrows():
        cov = {k: (None if pd.isna(r[c]) else (str(r[c]) if not isinstance(r[c], str) else r[c])) for k, c in covariate_cols.items()}
        num = {c: (None if pd.isna(r[c]) else float(r[c])) for c in numeric_cols}
        seg = {k: str(r[k]) for k in ("gender", "age_band", "education") if k in r.index}
        twins.append(Twin(name, str(r["id"]), cov, num, str(r["arm"]), float(r["y"]), extra={"segments": seg}))
    return twins


# ----------------------------------------------------------------------------- Clarke 2020 Study 2
def clarke():
    s = pd.read_excel(ROOT / "data/clarke2020_raw/Study2/FoodLabellingOnline2Dataset.xlsx")
    s = s[s["Gender"].isin([1, 2])].copy()
    s["id"] = s["Participant number"].astype(str)
    s["arm"] = s["Condition"].map({1: "pictorial_plus_calories", 2: "text_plus_calories", 3: "calories_only", 4: "pictorial", 5: "text_only", 6: "no_label"})
    s["y"] = 1 - s["Primary outcome"]
    s["gender"] = s["Gender"].map({1: "man", 2: "woman"})
    s["age_band"] = band(s["Age"])
    s["education"] = np.where(s["Education"].isin([5, 6]), "university degree", "no university degree")
    s["edu_level"] = s["Education"].where(s["Education"] > 0)
    s["ethnicity"] = s["Ethnicity"].map({1: "White", 2: "Mixed", 3: "Asian or Asian British", 4: "Black or Black British", 5: "Other", 7: "Other"})
    s["bmi"] = s["BMI"].where(s["BMI"] > 10)
    s["income_rank"] = s["Income"].map({c: i for i, c in enumerate("ABCDEFGHIJKLMNO")})
    inc_text = {0: "under £10k", 1: "£10-15k", 2: "£15-20k", 3: "£20-25k", 4: "£25-30k", 5: "£30-35k", 6: "£35-40k", 7: "£40-45k", 8: "£45-50k", 9: "£50-60k", 10: "£60-70k", 11: "£70-80k", 12: "£80-100k", 13: "£100-150k", 14: "over £150k"}
    s["household_income"] = s["income_rank"].map(inc_text)
    s["snack_freq"] = s["Snack consumption frequency"].map({1: "eats energy-dense snacks daily", 2: "weekly", 3: "monthly", 4: "never"})
    s["snack_daily"] = (s["Snack consumption frequency"] == 1).astype(float)
    s["age"] = s["Age"]
    s["female"] = (s["Gender"] == 2).astype(float)
    cols = ["id", "arm", "y", "gender", "age_band", "education", "age", "female", "edu_level", "bmi", "income_rank", "snack_daily"]
    human = s[cols].copy()
    cov = {"age": "Age", "gender": "gender", "ethnicity": "ethnicity", "highest_qualification_level_1_to_7": "Education", "household_income": "household_income", "bmi_rounded": "bmi", "snacking": "snack_freq"}
    s["bmi"] = s["bmi"].round(0)
    twins = make_twins(s, "clarke2020_s2", cov, ["age", "female", "edu_level", "bmi", "income_rank", "snack_daily"])
    arms = yaml.safe_load((ROOT / "data/clarke2020_study2_labels.yaml").read_text())["messages"]
    snacks = [("Doritos Tangy Cheese (crisps)", 0), ("apple crisps", 1), ("Galaxy chocolate bar", 0), ("Nim's beetroot crisps", 1), ("Müller Corner yoghurt", 1), ("McCoy's salt and vinegar crisps", 0),
              ("apple and grapes fruit pot", 1), ("millionaire shortbread", 0), ("Cadbury Dairy Milk bar", 0), ("mango and pineapple fruit pot", 1), ("Gü chocolate dessert pot", 0), ("KIND nut bar", 1)]
    spec = dict(
        task_type="choice", country="UK", control_arm="no_label",
        arms={k: arms[k] for k in ["no_label", "calories_only", "text_only", "text_plus_calories", "pictorial", "pictorial_plus_calories"]},
        outcome=dict(name="chose lower-energy snack", binary=True, support=[0, 1], higher_is="label worked"),
        segment_keys=["gender", "age_band"], mechanism_covariates=["age", "female", "edu_level", "bmi", "income_rank", "snack_daily"],
        task=dict(options=[{"label": l, "value": v} for l, v in snacks],
                  prompt="You are at a vending machine and you are going to have a snack now. The twelve options:\n{options}\n\nLabelling in this machine: {arm}\n\nWhich one do you take? One sentence of reaction, then answer."),
        topic_hint="snacks and what they grab when hungry",
    )
    write("clarke2020_s2", spec, human, twins)


# ----------------------------------------------------------------------------- Evans 2025 salt labels
def evans():
    d = pd.read_excel(ROOT / "data/evans2025_salt_raw/rctdataF.xlsx", sheet_name="casesremoved")
    d = d[d["Sex"].isin([1, 2])].copy()
    d["id"] = d["PPT ID"].astype(str)
    d["arm"] = d["Condition"].map({1: "red_triangle", 2: "red_octagon", 3: "black_triangle", 4: "black_octagon", 5: "qr_control"})
    gcols = ["Sandwichg", "Snackg", "Crispsg", "KFCg", "Bellag", "Wagag"]
    d["y"] = d[gcols].sum(axis=1)
    d["gender"] = d["Sex"].map({1: "man", 2: "woman"})
    d["age"] = d["Agey"]
    d["age_band"] = band(d["Agey"])
    d["education"] = d["Education"].map({1: "university degree", 2: "no university degree"})
    d["edu_level"] = d["EducationLevel"]
    d["female"] = (d["Sex"] == 2).astype(float)
    d["bmi"] = pd.to_numeric(d["BMI"], errors="coerce").where(lambda x: (x > 12) & (x < 70))
    d["income_rank"] = pd.to_numeric(d["IncomeCat"], errors="coerce").where(lambda x: x <= 9)
    d["restaurant_freq"] = pd.to_numeric(d["RestaurantFreq"], errors="coerce")
    d["ethnicity"] = d["Ethnicity"].map({1: "Asian or Asian British", 2: "Black or Black British", 3: "Mixed", 4: "White", 5: "Other", 6: "Other"})
    inc = {1: "under £10k", 2: "£10-20k", 3: "£20-30k", 4: "£30-40k", 5: "£40-50k", 6: "£50-60k", 7: "£60-70k", 8: "£70-80k", 9: "over £80k"}
    d["household_income"] = d["income_rank"].map(inc)
    d["eats_out"] = d["restaurant_freq"].map({1: "eats out or gets takeaway less than monthly", 2: "about monthly", 3: "two or three times a month", 4: "weekly", 5: "several times a week", 6: "most days"})
    human = d[["id", "arm", "y", "gender", "age_band", "education", "age", "female", "edu_level", "bmi", "income_rank", "restaurant_freq"]].copy()
    cov = {"age": "age", "gender": "gender", "ethnicity": "ethnicity", "education_level_1_to_6": "EducationLevel", "household_income": "household_income", "bmi_rounded": "bmi", "eating_out": "eats_out"}
    d["bmi"] = d["bmi"].round(0)
    twins = make_twins(d, "evans2025_salt", cov, ["age", "female", "edu_level", "bmi", "income_rank", "restaurant_freq"])

    def menu(title, rows, unit="portion"):
        return {"title": title, "items": [{"label": f"{n} ({p}){', ' + extra if extra else ''}", "value": g, "labelled": hs == "Y"} for n, p, hs, g, extra in rows]}

    menus = [
        menu("1. Tesco meal-deal sandwiches", [("Tesco Chicken & Sweetcorn Sandwich", "£2.10", "N", 1.03, ""), ("Tesco Chicken Salad Sandwich", "£2.60", "N", 1.12, ""), ("Tesco Chicken & Stuffing Sandwich", "£2.60", "N", 1.18, ""), ("Tesco Chicken, Bacon & Lettuce Sandwich", "£2.60", "Y", 2.09, ""), ("Tesco Roast Chicken, Mozzarella & Pesto Sandwich", "£2.60", "Y", 1.85, ""), ("Tesco BBQ Chicken, Bacon & Cheese Sandwich", "£2.85", "Y", 1.93, "")]),
        menu("2. Crisps (grab bags)", [("Walkers Cheese & Onion 45g", "£1.00", "N", 0.52, ""), ("Walkers Baked Cheese & Onion 37.5g", "£1.00", "N", 0.33, ""), ("McCoy's Cheddar & Onion 45g", "£1.00", "N", 0.44, ""), ("Walkers Quavers Cheese 34g", "£1.00", "Y", 0.64, ""), ("Walkers Monster Munch Pickled Onion", "£1.00", "Y", 0.62, ""), ("Jacob's Mini Cheddars 45g", "£1.00", "Y", 0.60, "")]),
        menu("3. Savoury snacks", [("Fridge Raiders Southern Style Chicken Bites 45g", "£1.30", "N", 0.70, ""), ("Philadelphia Light soft cheese snack", "£1.30", "N", 0.40, ""), ("Popchips BBQ 23g", "£1.00", "N", 0.20, ""), ("Peperami Original 22.5g", "£1.30", "Y", 0.88, ""), ("Cathedral City Cheddar sticks with pickle dip 60g", "£1.25", "Y", 1.62, ""), ("Love Corn BBQ roasted corn 45g", "£1.00", "Y", 1.00, "")]),
        menu("4. KFC meals", [("Twister wrap meal", "£7.99", "N", 2.77, "766 kcal"), ("Regular popcorn chicken meal", "£6.99", "N", 2.14, "536 kcal"), ("Colonel's meal, 2 piece", "£6.99", "N", 2.86, "726 kcal"), ("Twister wrap box meal", "£9.99", "Y", 3.57, "956 kcal"), ("Large popcorn chicken meal", "£8.49", "Y", 3.11, "716 kcal"), ("Mighty bucket for one", "£10.49", "Y", 5.14, "1166 kcal")]),
        menu("5. Wagamama mains", [("Chicken katsu curry", "£14.00", "N", 2.26, "998 kcal"), ("Ginger chicken udon", "£14.30", "N", 2.05, "678 kcal"), ("Shu's shiok chicken", "£14.00", "N", 2.35, "552 kcal"), ("Grilled chicken ramen", "£14.00", "Y", 5.17, "504 kcal"), ("Chicken gyoza ramen", "£15.50", "Y", 10.12, "698 kcal"), ("Chicken teriyaki donburi", "£14.50", "Y", 3.61, "738 kcal")]),
        menu("6. Bella Italia mains", [("Pomodoro Mozzarella", "£11.49", "N", 2.10, "684 kcal"), ("Pollo Funghi", "£14.59", "N", 2.30, "788 kcal"), ("Bolognese", "£13.59", "N", 2.80, "669 kcal"), ("Spicy Sausage rigatoni", "£14.99", "Y", 3.70, "936 kcal"), ("Pollo Cacciatore", "£15.49", "Y", 3.80, "670 kcal"), ("Spaghetti and Meatballs", "£14.99", "Y", 4.40, "1168 kcal")]),
    ]
    arms = {
        "qr_control": "[small QR code]",
        "red_triangle": "[red-outlined white triangle with a black exclamation mark and the words HIGH IN SALT]",
        "red_octagon": "[solid red stop-sign octagon with a white exclamation mark and the words HIGH IN SALT]",
        "black_triangle": "[solid black triangle with a white exclamation mark and the words HIGH IN SALT]",
        "black_octagon": "[solid black stop-sign octagon with a white exclamation mark and the words HIGH IN SALT]",
    }
    spec = dict(
        task_type="menu_choice", country="UK", control_arm="qr_control", arms=arms,
        outcome=dict(name="total salt (g) across six choices", binary=False, support=None, higher_is="label failed"),
        segment_keys=["gender", "age_band"], mechanism_covariates=["age", "female", "edu_level", "bmi", "income_rank", "restaurant_freq"],
        task=dict(menus=menus, aggregate="sum",
                  prompt="Imagine six separate occasions over the next couple of weeks. On each you are choosing one item from the list, as yourself, with your usual tastes and budget. Some items carry a small symbol next to their name, shown in square brackets; it appears exactly as printed on the pack or menu.\n\n{menus}\n\nFor each of the six lists, say which item you would pick. One short sentence overall, then answer."),
        topic_hint="takeaways, meal deals and what they order when eating out",
    )
    write("evans2025_salt", spec, human, twins)


# ----------------------------------------------------------------------------- Kerr 2021 Study 1 (null-effect, distributional task)
def kerr():
    df, meta = pyreadstat.read_sav(ROOT / "data/kerr2021_raw/study1_data.sav")
    d = df[df["DemGen"].isin([1, 2]) & df["exp_con"].notna()].copy()
    d = d[d["vax_status"] != 1]
    d["id"] = d.index.astype(str)
    d["arm"] = d["exp_con"].astype(str)
    items = [f"ox_vax_{i}" for i in range(1, 8)]
    d["y"] = 6 - d[items].mean(axis=1)  # higher = more willing
    d["gender"] = d["DemGen"].map({1: "man", 2: "woman"})
    d["female"] = (d["DemGen"] == 2).astype(float)
    d["age_band"] = d["quota_age"].map({1: "18-34", 2: "18-34", 3: "35-54", 4: "35-54", 5: "55+", 6: "55+"})
    d["age_group"] = d["quota_age"].map(meta.variable_value_labels["quota_age"])
    d["age_mid"] = d["quota_age"].map({1: 21, 2: 30, 3: 40, 4: 50, 5: 60, 6: 70})
    d["education"] = np.where(d["DemEdu"] >= 4, "university degree", "no university degree")
    d["edu_level"] = d["DemEdu"]
    d["edu_text"] = d["DemEdu"].map(meta.variable_value_labels["DemEdu"])
    d["politics"] = d["Politics"]
    d["politics_text"] = d["Politics"].map(meta.variable_value_labels["Politics"])
    d["social_status"] = d["Social_status"]
    d["ethnicity"] = d["short_eth"]
    d["minority"] = (d["Ethnic_min"] == 1).astype(float)
    d["healthcare_worker"] = d["DemHealthcare"].map({1: "works in healthcare", 2: "does not work in healthcare"})
    human = d[["id", "arm", "y", "gender", "age_band", "education", "age_mid", "female", "edu_level", "politics", "social_status", "minority"]].copy()
    cov = {"age_group": "age_group", "gender": "gender", "ethnicity": "ethnicity", "highest_qualification": "edu_text", "political_views": "politics_text", "self_rated_social_standing_1_to_9": "Social_status", "job": "healthcare_worker"}
    twins = make_twins(d, "kerr2021_s1", cov, ["age_mid", "female", "edu_level", "politics", "social_status", "minority"])
    preface = "It is mid-January 2021. Several new vaccines have been developed to protect people against COVID-19. "
    arms = {
        "control": preface + "(You were shown no further information.)",
        "factbox": preface + "You read a one-page fact box comparing a group who had the vaccine with a group who had a placebo in a large trial. It reported that the vaccine was 95.5% effective at preventing COVID-19 in 18 to 64 year olds and 86.2% in people aged 65 and over, and listed side effects side by side: for example, pain at the injection site in 90.1% of vaccinated people versus 18.8% of the placebo group, with tiredness, headache and muscle aches also common after vaccination, and serious side effects rare in both groups.",
        "text": preface + "You read a short question-and-answer text, adapted from the European Medicines Agency, explaining that in a large trial the vaccine reduced symptomatic COVID-19 by about 94% overall, that the most common side effects were mild or moderate and affected more than 1 in 10 people (pain at the injection site, tiredness, headache, muscle and joint pain, chills, fever), usually resolving within a few days, and that serious allergic reactions were rare.",
        "approval": preface + "You read a short text from the European Medicines Agency explaining how the vaccine was approved so quickly: regulators reviewed data on a rolling basis as it came in rather than waiting for the end of trials, the trials themselves were the normal size and length for a vaccine, no safety steps were skipped, and the vaccine is monitored continuously now it is in use.",
        "mechanism": preface + "You read a short text explaining how vaccines, including the new mRNA vaccines, work: the vaccine gives your cells the instructions to make a harmless piece of the virus's spike protein, your immune system learns to recognise it and makes antibodies, the instructions break down within days, the vaccine cannot give you COVID-19 and does not alter your DNA, and if you later meet the real virus your body already knows how to fight it.",
    }
    ox_items = [{"text": meta.column_names_to_labels[c].replace("\n", " ") + " (" + "; ".join(f"{int(k)} = {v}" for k, v in sorted(meta.variable_value_labels[c].items())) + ")", "reverse": True, "k": 5} for c in items]
    spec = dict(
        task_type="likert", country="UK", control_arm="control", arms=arms,
        outcome=dict(name="vaccine willingness (Oxford scale, 1-5, higher = more willing)", binary=False, support=None, higher_is="more willing"),
        segment_keys=["gender", "age_band"], mechanism_covariates=["age_mid", "female", "edu_level", "politics", "social_status", "minority"],
        task=dict(items=ox_items, prompt="{arm}\n\nNow answer the following as yourself, in January 2021, before you have been offered a vaccine. For each question give the number of the option that fits you best.\n\n{items}"),
        topic_hint="their attitude to the NHS, to COVID restrictions, and to vaccines generally, in January 2021",
    )
    write("kerr2021_s1", spec, human, twins)


# ----------------------------------------------------------------------------- Mora-Plazas 2024 Colombia drink choice
def mora():
    df, meta = pyreadstat.read_dta(ROOT / "data/moraplazas2024_raw/Cleaned.dta")
    d = df[df["arm"].notna() & df["buywhich_sug"].notna() & df["gender"].isin([1, 2])].copy()
    d["id"] = d["id"].astype(str)
    d["arm"] = d["arm"].map({1: "no_label", 2: "nutri_score", 3: "nutrient_warning", 4: "gda"})
    d["y"] = (d["buywhich_sug"] == 1).astype(float)  # chose the no-added-sugar drink
    d["gender"] = d["gender"].map({1: "man", 2: "woman"})
    d["female"] = (d["gender"] == "woman").astype(float)
    d["age_band"] = band(d["age"])
    d["education"] = np.where(d["education"] >= 5, "university degree", "no university degree")
    d["edu_level"] = df.loc[d.index, "education"]
    d["edu_text"] = d["edu_level"].map(meta.variable_value_labels["education"])
    d["finance"] = d["finance_sit"]
    d["finance_text"] = d["finance_sit"].map(meta.variable_value_labels["finance_sit"])
    d["region"] = d["dept"].map(meta.variable_value_labels["dept"])
    d["parent"] = d["children"].map({1: "parent or caregiver of a child under 18", 0: "no children under 18 at home"})
    d["parent_num"] = d["children"]
    d["sugar_drinks"] = d["sug_consume"]
    human = d[["id", "arm", "y", "gender", "age_band", "education", "age", "female", "edu_level", "finance", "parent_num", "sugar_drinks"]].copy()
    cov = {"age": "age", "gender": "gender", "education": "edu_text", "financial_situation": "finance_text", "department_of_residence": "region", "children": "parent"}
    twins = make_twins(d, "moraplazas2024_drink", cov, ["age", "female", "edu_level", "finance", "parent_num", "sugar_drinks"])
    arms = {
        "no_label": "Neither drink carries any front-of-pack nutrition label.",
        "nutrient_warning": "Drink B carries a black octagonal warning seal on the front reading 'EXCESO DE AZÚCARES' (excess sugar), with the Ministry of Health logo and the line 'evitar su alto consumo' (avoid high consumption). Drink A has no seal because it does not exceed the threshold.",
        "nutri_score": "Both drinks carry a Nutri-Score colour-graded letter (A dark green to E dark red) on the front. Drink A shows a green B; Drink B shows an orange D.",
        "gda": "Both drinks carry a Guideline Daily Amount panel on the front: serving size and small blocks for calories, total fat, saturated fat, sugar and sodium as a percentage of a 2,000-calorie diet. Drink A's sugar block reads 3 g (3%); Drink B's reads 18 g (20%).",
    }
    spec = dict(
        task_type="binary", country="Colombia", control_arm="no_label", arms=arms,
        outcome=dict(name="chose the no-added-sugar drink", binary=True, support=[0, 1], higher_is="label worked"),
        segment_keys=["gender", "age_band"], mechanism_covariates=["age", "female", "edu_level", "finance", "parent_num", "sugar_drinks"],
        task=dict(options=[{"label": "Fruit drink A: a boxed fruit juice drink, same size and price as B. Front of pack: {arm_a}", "value": 1}, {"label": "Fruit drink B: a boxed fruit juice drink, same size and price as A. Front of pack: {arm_b}", "value": 0}],
                  prompt="You are shopping and comparing two boxed fruit drinks on the shelf, same size and price. If you turn them over, both have the usual nutrition table on the back: Drink A lists 3 g of sugars per serving (all from fruit, no added sugar); Drink B lists 18 g of sugars per serving including added sugar.\n\nFront of pack: {arm}\n\nOption 1: {opt1}\nOption 2: {opt2}\n\nWhich of these would you rather buy? One short sentence, then answer."),
        topic_hint="what they drink day to day and shop for the household",
    )
    # the binary runner formats {arm} inside option labels; here the arm text already describes both drinks, so blank the per-option slot
    for o in spec["task"]["options"]:
        o["label"] = o["label"].replace(" Front of pack: {arm_a}", "").replace(" Front of pack: {arm_b}", "")
    write("moraplazas2024_drink", spec, human, twins)


if __name__ == "__main__":
    clarke(); evans(); kerr(); mora()
