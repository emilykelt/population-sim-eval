# More case studies with open human data (23 Oct 2026)

Part 1: neighbouring domains (vaccines, misinformation, climate, alcohol, political messaging). Verified by a research agent against the OSF, Zenodo and Dataverse APIs; OPEN means an actual data file was listed. Part 2 (food and nutrition, 2022-2026) follows when that search completes.

## Verified OPEN, usable now

| dataset | design | outcome | N | demographics | files |
|---|---|---|---|---|---|
| **Vlasceanu et al. 2024, climate intervention tournament** (Sci Adv; Zenodo 10345806) | 11 interventions + control, 63 countries incl. UK | belief, policy support, sharing intention, and a behavioural tree-planting effort task | 59,440 | gender, age, politics, education, income, SES, country | `data63.xlsx` (columns verified) |
| **Strengthening Democracy Challenge, Voelkel et al. 2024** (Science; OSF 2sv7p) | 25 treatments + control | partisan animosity, antidemocratic attitudes (attitudinal) | 32,059 | party, ideology, age, gender, race, education, income | `SDC - Data - Anonymized.csv`, codebook |
| **Porter & Wood 2021, global fact-checking** (PNAS; Dataverse DVN/Y8WPFR) | 3 arms x 22 fact-checks | belief accuracy | ~11,000 incl. ~3,000 UK | age, gender, region, education, ideology | `ff_ipsos.rds` and others |
| **Roozenbeek et al. 2022, inoculation videos** (Sci Adv; OSF 3769y) | 5 technique videos vs control | manipulation recognition, sharing discernment | ~6,400 | age, gender, education, politics | `Studies 1-5 - datasets.xlsx`, codebook |
| **Kerr, Freeman, Marteau, van der Linden 2021, vaccine information** (Vaccines; OSF 78hr9) | multiple information framings, 2 UK quota-sampled experiments | vaccination intention | several thousand (confirm) | age, gender, education, region, politics | `study1_data.sav`, `study2_data.sav` |
| **TESS newman1436, 2022, values-consistent vaccine messaging** (OSF 239bc) | 4 arms: same message from three different senders + control | vaccine intention 0-100 | 3,058 Republicans | full AmeriSpeak demographics | `newman1436.zip` |
| **Clarke, Pechey, Marteau 2023, alcohol warning + calorie labels, online purchasing** (Addiction; OSF w5xf8) | factorial HWL x calorie labels | selection then actual online purchase (behavioural) | 651 | age, gender, SES, education | `ONLINEPURCHASING_HWL_CALORIE_ALCOHOL_FINAL DATASET APRIL.xlsx` |
| **StepUp megastudy, Milkman et al. 2021** (Nature; OSF 9av87) | 54 conditions | gym visits (behavioural) | 61,293 | gender, age, race | folder `Codes&Data/` verified, file names not listed |

## Partial or blocked

- **Dai et al. 2021 Nature vaccine nudges** (OSF qn8hr): online intention experiments OPEN as Stata 10 `.dta` (read with pyreadstat, not pandas); the behavioural RCT microdata is on request (UCLA Health).
- **Milkman 2024 CVS booster megastudy**: microdata has condition and outcome only, no demographics. Unusable for segments.
- **Milkman 2022 Walmart flu**: aggregates only.
- **OCEANS-III** (Freeman 2021, UK, 10 vaccine-information arms, N = 18,855, full demographics): on request from Daniel Freeman with a data-sharing agreement. Highest-value UK dataset if obtained.
- **James, Bokemper et al. 2021** (12+ vaccine message arms, YouGov): OSF has only the pre-analysis plan; check the Yale ISPS data archive.
- **Pennycook & Rand 2022 accuracy prompts**: study-level only at the meta node; per-study OSF pages needed.
- Voter registration megastudy (OSF cexv9): simulated data only so far. Green investments megastudy (OSF 368ba): pre-registration only.
- Nothing verifiable found for charitable giving, UK energy saving, tax letters, screening, organ donation or road safety meeting all criteria.

## Contamination flags
All TESS studies and the Strengthening Democracy Challenge were used by Hewitt et al. and follow-ups; treat them as positive controls, not novel tests. Vlasceanu and StepUp are moderate risk. Porter & Wood, Kerr, Roozenbeek and the Clarke alcohol study show no evidence of LLM-simulation use.

## Ranking for our purpose
1. Vlasceanu 2024: 12 arms, behavioural task, full demographics, UK subset possible. Best structural fit.
2. Kerr 2021: the only open UK multi-arm message experiment with microdata. Intention outcome.
3. Clarke 2023 alcohol: UK, behavioural, Cambridge group, same shape as our Clarke 2020 case; small N.
4. Porter & Wood 2021: UK included, 22 items x 3 arms, ideology.
5. Strengthening Democracy Challenge: 25 arms, as a calibration set given contamination.

## Verified locally (23 Oct)
- **Vlasceanu 2024**: `data/vlasceanu2024_raw/data63.xlsx`, 59,440 rows, 12 conditions, UK subset 2,403 rows (about 200 per arm). Behavioural WEPT task (0-8 pages of effort) and share intention. Gender, age, education, income, politics (0-100). UK arm means on WEPT range 4.10 to 5.03, so the between-arm signal is small; pooled across countries it will be clearer.
- **Kerr 2021 Study 1**: `data/kerr2021_raw/study1_data.sav`, 2,488 UK adults (Prolific + Respondi quota), 5 arms (control, factbox, text, approval, mechanism), outcome Oxford hesitancy scale and a binary would-vaccinate item. Gender, age band, education, ethnicity (20 codes), minority self-ID, subjective social status, left-right politics. Study 2 (2,509 unvaccinated UK adults) is a content x length factorial; decode from `Study2_analysis.R`.
- **Hall 2022**: `data/hall2022_raw/S1_data.dta`, 325 parents, 2 arms, real purchase. Sign test only.
- **Turnwald 2019**: supplementary studies A (147 participants x 30 dishes x 3 label conditions, within-subject) and B are individual-level with age, sex, education, race; the multisite file is day-level.
