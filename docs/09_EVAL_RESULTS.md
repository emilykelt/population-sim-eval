# Three-fidelity eval: first run across four datasets (25 Oct 2026)

Setup: for each dataset, 150 real respondents were sampled across arms and turned into twins (a first-person backstory generated from their actual survey covariates: age, gender, ethnicity, education, income, BMI, habits). Every twin was placed in every arm and made the study's decision. Models: Claude Sonnet 5.5 and Claude Haiku 5.5. Cost: $17.90 total. Code: `src/build_evals.py`, `src/run_eval.py`, `src/simsoc/fidelity.py`. Full tables: `results/evals/REPORT.md`.

| dataset | country | arms | human outcome | human effect | ceiling (sex x age) |
|---|---|---|---|---|---|
| Clarke 2020 S2 | UK | 6 snack labelling schemes | chose lower-energy snack | 0.41 to 0.63 | 0.61 |
| Evans 2025 | UK | 4 salt-warning designs + QR control | total salt over 6 choices | 13.4 g control, 12.0 to 12.5 labelled | 0.42 |
| Kerr 2021 S1 | UK | 5 vaccine-information texts | willingness 1-5 | null (4.16 to 4.28) | 0.08 |
| Mora-Plazas 2024 | Colombia | 4 front-of-pack label systems | chose no-sugar drink | 0.69 to 0.84 | 0.87 |

## Effect fidelity: right ordering, wrong magnitude, and one outright miss

- **Clarke**: both models reproduce the arm ordering (pooled rho 0.94) and Sonnet's segment rankings (0.77) exceed the ceiling. But Sonnet's effect sizes are **5.3x** the human ones: labels explain 55% of simulated choice variance against 2.4% in humans. Haiku's slope is 0.62, closer to calibrated, but only because Haiku twins barely respond to anything.
- **Evans**: on total salt, no signal from either model (pooled rho 0.10, slope negative). The twins did react to the labels: the share of labelled items chosen fell from 27% (QR) to 5% (any warning), versus 48% to 33% for humans. They then picked low-salt unlabelled items, so total salt barely moved. The models exaggerate avoidance of the labelled item and ignore the actual nutrient. Top-1 of 0.67 to 0.83 here is a tie artefact: the four designs are within 0.5 g for humans.
- **Kerr (null task)**: Sonnet correctly produced no effect (slope -0.03). Haiku invented a negative effect of information (slope -0.39, pooled rho -0.60). A null task catches a model that manufactures effects; Haiku fails it.
- **Mora-Plazas**: with back-of-pack sugar figures visible, every twin on both models chose the healthier drink in every arm (dispersion 0). Removing the numbers (v2) restores ordering (pooled rho 0.77, segment 0.69) but all arms are pushed toward the healthy choice (bias +0.15). The twins read everything and act on it; the human 31% who bought the sugary drink under no label do not exist in the simulation.

## Distributional fidelity: collapsed everywhere

Dispersion ratio (simulated SD over human SD within arm): Clarke 0.55 to 0.64, Evans 0.35 to 0.47, Kerr 0.40 to 0.45, Mora v2 0.26. No dataset, no model reaches 0.7. Wasserstein distances are large in outcome units: 2.4 to 2.9 g of salt on a human SD of 3.1; 0.45 to 0.52 scale points on Kerr. Bias direction is consistent: toward the healthier or more compliant answer (Clarke Haiku -0.45 is the exception, from its crisps collapse). This is the flattening and social-desirability result from the literature reproduced on four behavioural datasets with matched real covariates, so it cannot be blamed on synthetic personas.

## Mechanism (structural) fidelity: the right levers, pulled too hard

- Where humans have structure, the models find it: Kerr main-effect correlation 0.84 with 100% sign agreement on Sonnet (politics, age, social status predict willingness the same way); Clarke 0.80 with 83% sign agreement.
- But covariates explain far more of simulated behaviour than of human behaviour: Kerr 0.59 vs 0.10, Clarke Haiku 0.19 vs 0.03, Evans Sonnet 0.32 vs 0.08. The twins are their demographics; the humans mostly are not. This is the over-determination finding quantified on four datasets.
- Arm x covariate interactions (who responds more to which label) show no agreement anywhere (r from -0.57 to 0.60, centred on zero). The one thing a segment-targeting tool is for is the one thing not recovered.
- Evans has essentially no human covariate structure to find (r -0.02), so a good model should return nothing; Sonnet returned nothing, which is correct.

## Individual fidelity: never beats a lookup table

Twin in its real person's arm versus that person's own outcome, against a leave-one-out (segment, arm) cell mean from the full human sample:
- Clarke: Sonnet 58% vs lookup 61% vs majority 60%. Haiku 43%.
- Evans: Sonnet MAE 3.1 g vs lookup 2.2 g; correlation with the real person 0.01.
- Kerr: Sonnet MAE 0.77 vs lookup 0.78, r 0.25. The only dataset where the twin adds anything, and only on an attitude.
- Mora: all at the majority rate.
Consistent with Twin-2K-500 and "When Synthetic Users Fail": real covariates do not make the twin predict the person.

## What this says about the three fidelities as evaluation targets

1. **Effect fidelity is the only one the models partly pass**, and only for ordering. Magnitude is off by 5x in one direction (Clarke Sonnet) or absent (Evans). Any product output must be a ranking plus a human-calibrated scale factor.
2. **Distributional fidelity fails uniformly** at dispersion ratios of 0.3 to 0.6. Independent sampling of grounded twins at temperature 1 does not restore human variance. Candidate fixes, untested here: verbalised distribution prompting, base or lightly post-trained open models, mixing model families, or post-hoc rescaling against a human pilot.
3. **Mechanism fidelity splits**: main effects of covariates are recovered with the right signs but 3 to 6x the strength; interactions are not recovered. Report both separately. A tool that claims "this message works better for group X" is reporting interaction structure, which is exactly the part with zero fidelity.
4. **A null task is mandatory.** Kerr separated the models when the positive tasks did not.
5. **Prompt omniscience is a confound.** Any fact placed in the prompt is used with certainty (Mora v1). Stimulus descriptions must include only what a human would plausibly attend to, and that choice should be perturbed like any other wording.

## Caveats
150 twins per dataset and one draw per arm; one prompt wording per task (the perturbation study on Clarke showed the wording family matters); Anthropic models only; the Evans and Mora stimuli are text reconstructions of visual labels; backstories were generated by the model under test, so a weaker model also writes weaker twins.
