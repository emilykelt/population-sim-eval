# Data

Raw third-party trial data is not redistributed here. Each dataset has its own licence; download it from the original archive into the path shown before running `src/make_cases.py` or `src/build_evals.py`.

The derived eval specs (`data/evals/*.yaml`), per-participant extracts of the four trials below (`data/evals/*_human.csv`: arm, outcome, a few demographics; credit to the original authors), twin personas (`data/evals/*_twins.json`) and case YAMLs are committed, so `src/run_eval.py` and `src/run_benchmark.py` run without the raw files.

## Datasets used by the code

| Study | Source | Put files in | Used by |
|---|---|---|---|
| Clarke, Pechey, Marteau et al. 2020, health warning labels (Appetite) | OSF https://osf.io/k7tw5/ (Study1, Study2 folders) | `data/clarke2020_raw/Study1/`, `data/clarke2020_raw/Study2/` | `make_cases.py`, `build_evals.py` |
| Evans, Robinson et al. 2025, salt warning labels (Lancet Public Health) | OSF https://osf.io/vyh2x/ (`rctdataF.xlsx`, protocol) | `data/evans2025_salt_raw/` | `build_evals.py` |
| Kerr, Freeman, Marteau, van der Linden 2021, vaccine information (Vaccines) | OSF https://osf.io/78hr9/ (`study1_data.sav`) | `data/kerr2021_raw/` | `build_evals.py` |
| Mora-Plazas et al. 2024, front-of-pack labels in Colombia (PLOS ONE) | OSF https://osf.io/5vbzm/ (`Datasets.zip` -> `Cleaned.dta`) | `data/moraplazas2024_raw/` | `build_evals.py` |

## Other candidate datasets (not yet used)

Listed with sources in `docs/04_CASE_STUDIES_BRIEF.md` and `docs/08_MORE_CASE_STUDIES.md`: Turnwald 2019 (OSF 39vsg), Vlasceanu 2024 (Zenodo 10345806), Taillie 2023 (Harvard Dataverse HQMFUS, OB8ZFJ), Biggs 2025 (OSF 69xct), Potter 2022 (OSF byztm), Hughes 2024 (OSF gm3pr), and others. The local download script put these under `data/agent_downloads/<osf-id>/`.
