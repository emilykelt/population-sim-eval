# population-sim-eval

**Three Fidelities: evaluating LLM twins against behavioural message-testing trials.**

Commercial "synthetic respondent" tools predict how an audience will react to a public message by prompting language-model personas. Almost all the evidence for these tools comes from survey items, and almost all of it is one number per model from one prompt. This repo is an evaluation harness that replicates open randomised trials with LLM *twins* and scores the simulation against what the real participants did.

Paper: [`paper/paper.pdf`](paper/paper.pdf) (draft, 25 October 2026).

## Why three fidelities

"Similar to a human" is not one property. A simulator can be faithful to:

| Fidelity | Question | Why it matters |
|---|---|---|
| **Effect** | Do arms differ as they did for humans? | What a message-testing product needs |
| **Distributional** | Does the spread of responses within each arm match? | Without it, the product is unsafe to use |
| **Mechanism** | Do the same covariates, and covariate x arm interactions, predict the outcome? | What segment-targeting products sell |

A fourth score, **individual fidelity**, checks whether a twin predicts the specific person it was built from better than a leave-one-out lookup table does.

## Method

**Datasets.** Four open randomised experiments. Each randomised people to 3 or more message or label variants, recorded a choice or behavioural outcome, and released individual-level data with demographics.

| Study | Country | Arms | N | Outcome |
|---|---|---|---|---|
| Clarke et al. 2020, Study 2 | UK | 6 snack-labelling schemes | 4,134 | chose lower-energy snack (12 options) |
| Evans & Robinson 2025 | UK | 4 salt-warning designs + QR control | 2,391 | total salt (g) over 6 menu choices |
| Kerr et al. 2021, Study 1 | UK | 5 vaccine-information texts | 2,488 | Oxford willingness scale, 1-5 |
| Mora-Plazas et al. 2024 | Colombia | 4 front-of-pack label systems | 8,004 | chose no-added-sugar drink |

Kerr is a **null task**: the framings had no detectable human effect, so a simulator that finds one is manufacturing it.

**Twins.** 150 real respondents per dataset. The model under test writes a ~200-word first-person interview for each one. The interview is constrained to that person's recorded covariates (age, gender, ethnicity, education, income, BMI, relevant habits) and must not contradict them. Twins therefore match the covariate composition of the human sample, and each one maps back to a real person.

**Replicate, don't poll.** Each twin is placed in one arm at a time and makes the study's actual decision with the study's options, such as picking one of twelve snacks or choosing between two labelled drinks. Every twin runs in every arm, so arm comparisons hold composition fixed.

**Scores.**
- *Effect:* Spearman correlation of human vs simulated arm means (pooled and within segment); slope of simulated on human effects (1 = calibrated); top-1 arm. Scored against a split-half human ceiling and a pooled-ranking baseline.
- *Distribution:* Wasserstein-1 per arm; total variation for discrete outcomes; dispersion ratio (simulated SD / human SD).
- *Mechanism:* OLS of outcome on standardised covariates and arm dummies in both tables; correlation and sign agreement of covariate coefficients and of arm x covariate interactions.
- *Individual:* twin outcome in its person's own arm vs that person's outcome, against a leave-one-out (segment, arm) cell mean.

Two further audits run on Clarke: **design** (asking personas which label would move them vs having them choose) and **perturbation** (25 semantically equivalent rewordings of the same task).

## Findings (Claude Sonnet 5.5 and Haiku 5.5)

1. **Effect: ordering yes, magnitude no.** Sonnet recovers arm orderings (Clarke pooled rho = 0.94) and correctly finds nothing on the null task. Its effects are 5x human size on Clarke, it misses the Evans effect entirely, and every label system saturates on Mora-Plazas.
2. **Distribution: collapsed everywhere.** Simulated variance is 26-64% of human variance on every dataset and model.
3. **Mechanism: right levers, pulled too hard, no interactions.** Covariate main effects have the right signs (r = 0.84 on Kerr) but 3-6x human strength. Covariate x arm interactions, the quantity a segment-targeting tool reports, show no agreement anywhere.
4. **Individual: never beats a lookup table.**
5. **Audits.** Asking personas *which* label would move them predicts the *reverse* of what people did. Rewording the prompt moves a cheap model's score across a range of 0.77 and a frontier model's across 0.39.

Full four-dataset run: $17.90.

**Recommendations for audience simulators:** replicate decisions rather than poll opinions; report rankings with a scale fitted on a small human pilot, not levels; publish a null task and a wording family with every score; make no segment-specific claims until interaction fidelity is shown on held-out human data.

## Repo layout

| Path | Contents |
|---|---|
| `src/simsoc/` | twins, tasks, fidelity metrics, perturbation, model providers |
| `src/build_evals.py` | raw trial data -> eval specs, human tables, twin samples in `data/evals/` |
| `src/run_eval.py` | runs one eval spec across models |
| `src/report_evals.py` | -> `results/evals/REPORT.md` |
| `data/evals/` | committed eval specs, per-participant human extracts, twin files |
| `results/` | simulated tables and scores per model, perturbation runs, logs |
| `paper/` | LaTeX source, figures, PDF |
| `docs/` | plan, literature synthesis, case-study briefs, results notes |

## Run

Raw trial data is not in the repo; see [`data/README.md`](data/README.md) for download links. The eval specs, human tables and twins in `data/evals/` are committed, so `run_eval.py` works without them.

```bash
uv sync
PYTHONPATH=src uv run python src/run_eval.py data/evals/clarke2020_s2.yaml \
    --models claude-sonnet-5-5 claude-haiku-5-5 --twins 150
PYTHONPATH=src uv run python src/report_evals.py                     # -> results/evals/REPORT.md
```

Evals: `clarke2020_s2`, `evans2025_salt`, `kerr2021_s1`, `moraplazas2024_drink`, `moraplazas2024_drink_v2`. Rebuild them from raw data with `src/build_evals.py`.

`.env` needs `ANTHROPIC_API_KEY`. All model calls are cached under `results/raw/cache/` keyed on model + prompt + seed, so reruns are free.

### Other providers

Put the key in `.env` and use the prefixed id: `openai/gpt-5-nano` (OPENAI_API_KEY), `gemini/gemini-2.5-flash-lite` (GOOGLE_API_KEY), `deepseek/deepseek-chat` (DEEPSEEK_API_KEY), `together/Qwen/Qwen3-32B` (TOGETHER_API_KEY), `groq/llama-3.3-70b-versatile` (GROQ_API_KEY), `mistral/mistral-small-latest` (MISTRAL_API_KEY), `ollama/<local model>` (no key; needs Ollama running). Vercel AI Gateway ids such as `google/gemini-2.5-flash-lite` also work with `AI_GATEWAY_API_KEY`.

### Earlier benchmark (synthetic personas, Clarke 2020)

```bash
PYTHONPATH=src uv run python src/make_cases.py                       # builds data/*.yaml from the Clarke 2020 OSF files
PYTHONPATH=src uv run python src/run_benchmark.py data/clarke2020_study2_labels.yaml \
    --models claude-sonnet-5-5 --n 12 --methods judge grounded_choice
PYTHONPATH=src uv run python src/report.py data/clarke2020_study2_labels.yaml   # -> results/<case>/REPORT.md
```
Methods: `judge`, `label_single`, `grounded_single`, `grounded_ensemble`, `grounded_debate`, `grounded_choice`, `label_choice`, `grounded_choice_neutral`. Perturbation audit: `src/run_perturbation.py`, `src/analyse_perturbation.py`.

## Extension: predicting which headline wins

Same question, different outcome: given the headline variants of a real A/B test, which one got the most clicks? Ground truth is the Upworthy Research Archive (32,487 tests, CC-BY 4.0; see `data/README.md`). Only tests where every variant shares the same image are used.

Results on the 2,237-test holdout split, scored once:

| Predictor | Pairwise | Top-1 (chance 0.23) | Best vs worst | Decisive pairs (z >= 4) |
|---|---|---|---|---|
| Embedding classifier (bge-small + pairwise logistic, local, $0) | 0.635 | 0.387 | 0.740 | 0.858 |
| Noise ceiling (re-running the same test) | 0.84 | 0.72 | | |

On 400 exploratory tests, the free classifier matched a Claude Sonnet 5.5 judge (pairwise 0.613 vs 0.606) and beat it on decisive pairs (0.84 vs 0.74). Few-shot examples and a judge + classifier blend added nothing. A crowd of 20 simulated readers per test scored at chance, even though 69% of their clicks agreed. As in the food trials, polling personas fails where a direct forecast or a trained model does not.

```bash
PYTHONPATH=src uv run python src/build_upworthy.py exploratory            # eval set, noise ceiling, heuristics
PYTHONPATH=src uv run python src/run_upworthy.py --methods judge crowd    # LLM judge and simulated crowd on dev
PYTHONPATH=src uv run python src/upworthy_scale.py                        # train on confirmatory, score holdout
PYTHONPATH=src uv run python src/rank_drafts.py drafts/mathematical_bridge.txt   # rank your own drafts
PYTHONPATH=src uv run python src/bandit_sim.py                            # offline bandit simulator ($0)
```

### Can a posting bot learn online? An offline bandit simulator

`src/bandit_sim.py` replays the holdout tests as a contextual bandit: each test is a topic, its headlines are the drafts, the policy posts one and sees only that one's clicks (Binomial at the real CTR). Lift captured: random 0, always picking the real winner 1. Mean of 20 seeds, 200 views per post:

| Policy | First 30 posts | First 365 | First 2,000 |
|---|---|---|---|
| Thompson sampling from scratch | 0.02 | 0.09 | 0.16 |
| Thompson sampling, classifier as prior | 0.29 | 0.30 | 0.31 |
| Classifier only, no learning | 0.33 | 0.32 | 0.33 |

Learning from your own posts does not catch up with a classifier trained on 10,000 tests within 2,000 posts, and more views per post barely help (20 to 20,000 views moves the from-scratch bandit from 0.11 to 0.16): topic-to-topic differences in click rate swamp the wording signal. Training the bandit on the simulated persona crowd instead of real clicks, then testing on real holdout clicks, gives 0.06 against 0.15 (the crowd's within-test correlation with real CTR is -0.02). The persona crowd is not usable as a training environment.

## Limitations

One draw per twin per arm; one wording per task in the twin study; Anthropic models only so far; stimuli are text reconstructions of visual labels; twins are written by the model under test; Kerr and Mora-Plazas outcomes are intentions or hypothetical choices, not purchases.

## Citation

Kelt, E. (2026). *Three Fidelities: Evaluating LLM Twins Against Behavioural Message-Testing Trials.* Draft. https://github.com/emilykelt/population-sim-eval

Trial data belongs to the original authors; see `data/README.md`.
