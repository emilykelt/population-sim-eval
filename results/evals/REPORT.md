# Three-fidelity eval: results

One row per dataset x model. 150 real-respondent twins per dataset, each run in every arm. Human reference lines are computed from the human data alone.

## Effect fidelity (do arms differ the way they did for humans)

| dataset | model | pooled rho | effect-size slope | segment rho | pooled baseline | ceiling | top-1 | gap r | gap slope | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| clarke2020_s2 | claude-haiku-5-5 | 0.94 | 0.62 | 0.66 | 0.82 | 0.61 | 0.17 | 0.42 | 0.23 | 900/900 |
| clarke2020_s2 | claude-sonnet-5-5 | 0.94 | 5.32 | 0.77 | 0.82 | 0.61 | 0.33 | 0.49 | 0.67 | 900/900 |
| evans2025_salt | claude-haiku-5-5 | 0.10 | -0.19 | 0.33 | 0.68 | 0.42 | 0.83 | -0.00 | -0.00 | 750/750 |
| evans2025_salt | claude-sonnet-5-5 | 0.10 | -0.17 | 0.12 | 0.68 | 0.42 | 0.67 | -0.25 | -0.08 | 750/750 |
| kerr2021_s1 | claude-haiku-5-5 | -0.60 | -0.39 | -0.22 | 0.28 | 0.08 | 0.17 | 0.38 | 0.10 | 750/750 |
| kerr2021_s1 | claude-sonnet-5-5 | -0.10 | -0.03 | -0.26 | 0.28 | 0.08 | 0.00 | 0.78 | 0.30 | 750/750 |
| moraplazas2024_drink | claude-haiku-5-5 | n.a. | 0.00 | n.a. | 0.90 | 0.87 | 0.00 | n.a. | n.a. | 600/600 |
| moraplazas2024_drink | claude-sonnet-5-5 | n.a. | 0.00 | n.a. | 0.90 | 0.87 | 0.00 | n.a. | n.a. | 600/600 |
| moraplazas2024_drink_v2 | claude-haiku-5-5 | 0.77 | 0.00 | 0.69 | 0.90 | 0.87 | 0.33 | 0.15 | 0.17 | 600/600 |
| moraplazas2024_drink_v2 | claude-sonnet-5-5 | 0.77 | -0.00 | 0.69 | 0.90 | 0.87 | 0.33 | 0.05 | 0.09 | 600/600 |

Effect-size slope: simulated arm effect (vs control) regressed on human arm effect; 1 = calibrated, >1 exaggerated, <0 wrong direction. Gap slope: same for between-segment differences.

## Distributional fidelity (does the response distribution per arm match)

| dataset | model | outcome | W1 (outcome units) | TV | dispersion ratio | segment-arm MAE | bias | human SD | sim SD |
|---|---|---|---|---|---|---|---|---|---|
| clarke2020_s2 | claude-haiku-5-5 | P(lower-energy snack) | 0.450 | 0.450 | 0.55 | 0.448 | -0.448 | 0.499 | 0.280 |
| clarke2020_s2 | claude-sonnet-5-5 | P(lower-energy snack) | 0.309 | 0.309 | 0.64 | 0.302 | -0.101 | 0.499 | 0.495 |
| evans2025_salt | claude-haiku-5-5 | total salt g | 2.856 | n.a. | 0.47 | 2.869 | -2.869 | 3.101 | 1.544 |
| evans2025_salt | claude-sonnet-5-5 | total salt g | 2.431 | n.a. | 0.35 | 2.479 | -2.479 | 3.101 | 1.147 |
| kerr2021_s1 | claude-haiku-5-5 | willingness 1-5 | 0.523 | n.a. | 0.45 | 0.229 | -0.134 | 0.977 | 0.446 |
| kerr2021_s1 | claude-sonnet-5-5 | willingness 1-5 | 0.451 | n.a. | 0.40 | 0.224 | 0.216 | 0.977 | 0.388 |
| moraplazas2024_drink | claude-haiku-5-5 | P(no-sugar drink) | 0.228 | 0.228 | 0.00 | 0.233 | 0.233 | 0.419 | 0.000 |
| moraplazas2024_drink | claude-sonnet-5-5 | P(no-sugar drink) | 0.228 | 0.228 | 0.00 | 0.233 | 0.233 | 0.419 | 0.000 |
| moraplazas2024_drink_v2 | claude-haiku-5-5 |  | 0.194 | 0.194 | 0.27 | 0.194 | 0.111 | 0.419 | 0.327 |
| moraplazas2024_drink_v2 | claude-sonnet-5-5 |  | 0.162 | 0.162 | 0.26 | 0.178 | 0.152 | 0.419 | 0.286 |

Dispersion ratio = simulated SD / human SD within arm; <1 = collapsed, >1 = over-dispersed. Bias = mean(sim) - mean(human) over segment x arm cells.

## Mechanism (structural) fidelity (do the same covariates drive the outcome)

| dataset | model | main-effect r | sign agreement | interaction r | covariate R2 human | covariate R2 sim | arm R2 human | arm R2 sim |
|---|---|---|---|---|---|---|---|---|
| clarke2020_s2 | claude-haiku-5-5 | 0.65 | 0.67 | -0.02 | 0.027 | 0.188 | 0.024 | 0.028 |
| clarke2020_s2 | claude-sonnet-5-5 | 0.80 | 0.83 | 0.03 | 0.027 | 0.067 | 0.024 | 0.552 |
| evans2025_salt | claude-haiku-5-5 | -0.12 | 0.50 | 0.60 | 0.078 | 0.121 | 0.021 | 0.131 |
| evans2025_salt | claude-sonnet-5-5 | -0.02 | 0.67 | -0.57 | 0.078 | 0.321 | 0.021 | 0.110 |
| kerr2021_s1 | claude-haiku-5-5 | 0.74 | 0.83 | 0.24 | 0.099 | 0.334 | 0.002 | 0.011 |
| kerr2021_s1 | claude-sonnet-5-5 | 0.84 | 1.00 | -0.11 | 0.099 | 0.591 | 0.002 | 0.000 |
| moraplazas2024_drink | claude-haiku-5-5 | -0.90 | 0.00 | 0.16 | 0.004 | -inf | 0.028 | n.a. |
| moraplazas2024_drink | claude-sonnet-5-5 | -0.90 | 0.00 | 0.16 | 0.004 | -inf | 0.028 | n.a. |
| moraplazas2024_drink_v2 | claude-haiku-5-5 | 0.26 | 0.67 | 0.26 | 0.004 | 0.004 | 0.028 | 0.416 |
| moraplazas2024_drink_v2 | claude-sonnet-5-5 | -0.36 | 0.33 | 0.13 | 0.004 | 0.014 | 0.028 | 0.283 |

Covariate R2: share of outcome variance explained by who the person is (age, sex, education, income, BMI, habits). If sim >> human, the model over-determines behaviour from demographics. Arm R2: variance explained by the treatment.

## Individual fidelity (twin in the real person's own arm vs that person's outcome)

| dataset | model | n | sim | lookup-table baseline | majority / r |
|---|---|---|---|---|---|
| clarke2020_s2 | claude-haiku-5-5 | 150 | acc 0.43 | acc 0.61 | majority 0.60 |
| clarke2020_s2 | claude-sonnet-5-5 | 150 | acc 0.58 | acc 0.61 | majority 0.60 |
| evans2025_salt | claude-haiku-5-5 | 150 | MAE 3.71 | MAE 2.20 | r -0.06 |
| evans2025_salt | claude-sonnet-5-5 | 150 | MAE 3.12 | MAE 2.20 | r 0.01 |
| kerr2021_s1 | claude-haiku-5-5 | 150 | MAE 0.84 | MAE 0.78 | r 0.17 |
| kerr2021_s1 | claude-sonnet-5-5 | 150 | MAE 0.77 | MAE 0.78 | r 0.25 |
| moraplazas2024_drink | claude-haiku-5-5 | 150 | acc 0.79 | acc 0.79 | majority 0.79 |
| moraplazas2024_drink | claude-sonnet-5-5 | 150 | acc 0.79 | acc 0.79 | majority 0.79 |
| moraplazas2024_drink_v2 | claude-haiku-5-5 | 150 | acc 0.74 | acc 0.79 | majority 0.79 |
| moraplazas2024_drink_v2 | claude-sonnet-5-5 | 150 | acc 0.77 | acc 0.79 | majority 0.79 |

## Per-arm means (human vs simulated)

- **clarke2020_s2 / claude-haiku-5-5**: human [0.41, 0.462, 0.524, 0.564, 0.631, 0.624] vs sim [0.04, 0.047, 0.073, 0.06, 0.167, 0.127]; cost $0.0, 0s
  - man|18-34: rho 0.91, top1 miss, n_h 425
  - man|35-54: rho 0.49, top1 miss, n_h 732
  - man|55+: rho 0.74, top1 miss, n_h 844
  - woman|18-34: rho 0.78, top1 miss, n_h 785
  - woman|35-54: rho 0.97, top1 hit, n_h 766
  - woman|55+: rho 0.09, top1 miss, n_h 574
- **clarke2020_s2 / claude-sonnet-5-5**: human [0.41, 0.462, 0.524, 0.564, 0.631, 0.624] vs sim [0.073, 0.087, 0.24, 0.313, 0.873, 0.987]; cost $0.0, 1s
  - man|18-34: rho 0.83, top1 miss, n_h 425
  - man|35-54: rho 0.70, top1 miss, n_h 732
  - man|55+: rho 0.83, top1 hit, n_h 844
  - woman|18-34: rho 0.94, top1 hit, n_h 785
  - woman|35-54: rho 0.93, top1 miss, n_h 766
  - woman|55+: rho 0.38, top1 miss, n_h 574
- **evans2025_salt / claude-haiku-5-5**: human [13.376, 12.273, 12.557, 12.01, 12.439] vs sim [10.781, 9.493, 9.333, 9.418, 9.349]; cost $0.0, 0s
  - man|18-34: rho -0.50, top1 miss, n_h 324
  - man|35-54: rho 0.20, top1 hit, n_h 564
  - man|55+: rho 0.30, top1 hit, n_h 306
  - woman|18-34: rho 0.70, top1 hit, n_h 289
  - woman|35-54: rho 0.90, top1 hit, n_h 566
  - woman|55+: rho 0.40, top1 hit, n_h 318
- **evans2025_salt / claude-sonnet-5-5**: human [13.376, 12.273, 12.557, 12.01, 12.439] vs sim [10.867, 9.922, 9.838, 9.951, 9.923]; cost $0.0, 0s
  - man|18-34: rho -0.70, top1 miss, n_h 324
  - man|35-54: rho 0.70, top1 hit, n_h 564
  - man|55+: rho -0.30, top1 miss, n_h 306
  - woman|18-34: rho 0.60, top1 hit, n_h 289
  - woman|35-54: rho 0.10, top1 hit, n_h 566
  - woman|55+: rho 0.30, top1 hit, n_h 318
- **kerr2021_s1 / claude-haiku-5-5**: human [4.211, 4.19, 4.163, 4.277, 4.167] vs sim [4.042, 4.178, 4.117, 4.072, 4.094]; cost $0.0, 0s
  - man|18-34: rho 0.30, top1 hit, n_h 422
  - man|35-54: rho -0.70, top1 miss, n_h 468
  - man|55+: rho -0.10, top1 miss, n_h 335
  - woman|18-34: rho 0.70, top1 miss, n_h 425
  - woman|35-54: rho -0.90, top1 miss, n_h 447
  - woman|55+: rho -0.60, top1 miss, n_h 298
- **kerr2021_s1 / claude-sonnet-5-5**: human [4.211, 4.19, 4.163, 4.277, 4.167] vs sim [4.45, 4.446, 4.461, 4.448, 4.441]; cost $0.0, 0s
  - man|18-34: rho -0.82, top1 miss, n_h 422
  - man|35-54: rho 0.20, top1 miss, n_h 468
  - man|55+: rho 0.72, top1 miss, n_h 335
  - woman|18-34: rho -0.60, top1 miss, n_h 425
  - woman|35-54: rho -0.67, top1 miss, n_h 447
  - woman|55+: rho -0.40, top1 miss, n_h 298
- **moraplazas2024_drink / claude-haiku-5-5**: human [0.689, 0.831, 0.725, 0.844] vs sim [1.0, 1.0, 1.0, 1.0]; cost $0.0, 0s
  - man|18-34: rho n.a., top1 miss, n_h 2868
  - man|35-54: rho n.a., top1 miss, n_h 1838
  - man|55+: rho n.a., top1 miss, n_h 316
  - woman|18-34: rho n.a., top1 miss, n_h 2759
  - woman|35-54: rho n.a., top1 miss, n_h 1805
  - woman|55+: rho n.a., top1 miss, n_h 246
- **moraplazas2024_drink / claude-sonnet-5-5**: human [0.689, 0.831, 0.725, 0.844] vs sim [1.0, 1.0, 1.0, 1.0]; cost $0.0, 1s
  - man|18-34: rho n.a., top1 miss, n_h 2868
  - man|35-54: rho n.a., top1 miss, n_h 1838
  - man|55+: rho n.a., top1 miss, n_h 316
  - woman|18-34: rho n.a., top1 miss, n_h 2759
  - woman|35-54: rho n.a., top1 miss, n_h 1805
  - woman|55+: rho n.a., top1 miss, n_h 246
- **moraplazas2024_drink_v2 / claude-haiku-5-5**: human [0.689, 0.831, 0.725, 0.844] vs sim [0.513, 1.0, 1.0, 1.0]; cost $0.075, 79s
  - man|18-34: rho 0.77, top1 miss, n_h 2868
  - man|35-54: rho 0.77, top1 miss, n_h 1838
  - man|55+: rho 0.26, top1 miss, n_h 316
  - woman|18-34: rho 0.77, top1 miss, n_h 2759
  - woman|35-54: rho 0.77, top1 hit, n_h 1805
  - woman|55+: rho 0.77, top1 hit, n_h 246
- **moraplazas2024_drink_v2 / claude-sonnet-5-5**: human [0.689, 0.831, 0.725, 0.844] vs sim [0.64, 1.0, 1.0, 1.0]; cost $1.414, 116s
  - man|18-34: rho 0.77, top1 miss, n_h 2868
  - man|35-54: rho 0.77, top1 miss, n_h 1838
  - man|55+: rho 0.26, top1 miss, n_h 316
  - woman|18-34: rho 0.77, top1 miss, n_h 2759
  - woman|35-54: rho 0.77, top1 hit, n_h 1805
  - woman|55+: rho 0.77, top1 hit, n_h 246