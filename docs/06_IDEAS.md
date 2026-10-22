# Two research directions, expanded (22 Oct 2026)

## Idea A: BehaveBench, a behavioural message-testing benchmark

### The gap
Every public benchmark for "LLMs as simulated humans" scores survey items: OpinionQA (Pew), SubPOP (ATP + GSS), Twin-2K-500 (500 questionnaire items), SimBench (20 survey datasets), SocSci210 (TESS survey experiments). All measure whether a model reproduces what people *say*. None measures whether it reproduces what people *did* after seeing a message. The pilot shows these can disagree in sign, so the survey benchmarks may be ranking models on the wrong quantity for the message-testing use case.

### What the benchmark is
A fixed set of open, pre-registered randomised experiments where (a) participants were exposed to one of K message or label variants, (b) a behavioural or quasi-behavioural outcome was recorded (choice, purchase, selection, consumption), (c) individual demographics are available, and (d) the data are downloadable. Each dataset becomes a task: given the K variant descriptions and a segment definition, predict the segment's ranking of the variants and, optionally, the effect size relative to control.

Candidate tasks already identified (docs/04):

| dataset | variants | outcome | N | segments | status |
|---|---|---|---|---|---|
| Clarke 2020 Study 2 | 6 label schemes | snack choice | 4,134 | sex, age, education, BMI | in hand |
| Clarke 2020 Study 1 | 6 disease framings | desire (1-7) | 4,607 | same | in hand, near-zero heterogeneity, keep as null task |
| Turnwald 2019 | 3 vegetable label framings | cafeteria selection, 138k decisions | 5 sites | site-level only | OSF 39vsg, download |
| Hall 2022 PLOS Med | pictorial warning vs control | real purchase for child | 325 | race, income, education | S1 Data, download |
| Potter 2024 | 4 ecolabel designs | basket impact, online supermarket | 1,051 | age, gender | OSF link to verify |
| Wolfson 2022 | 3 climate labels | fast-food item choice | 5,049 | sex, age, race, income, ideology | request |
| Hall 2021 Prev Med | 4 topics x 3 designs | intentions + PME | 1,078 | Latino, English use | request |
| Milkman 2024 | 22 booster nudges | booster uptake | 690k | none released | analogue, no demographics |

Six to eight tasks is enough. SimBench launched with 20 datasets but most benchmarks that get cited started smaller and grew.

### What makes it a contribution rather than a data dump
1. **Two human reference lines per task, computed from the human data only.** The pooled-population baseline (predict every segment with the overall ranking) and the split-half noise ceiling (how well half the humans predict the other half). No existing benchmark reports either. Without them, a score of 0.79 is uninterpretable; with them it reads as "matches the population ranking, exceeds the ceiling, adds nothing from segmentation". The null task (Clarke Study 1, ceiling 0.17) is included deliberately to catch methods that manufacture segment differences.
2. **A fixed elicitation protocol with two registered designs**, poll and replicate, so that the benchmark tests the design choice as well as the model. Submissions must report both.
3. **Three scores, not one.** Rank fidelity (Spearman per segment), decision accuracy (top-1 per segment, the thing a comms team acts on), and gap calibration (slope of simulated vs human between-segment differences, 1.0 = calibrated, >1 = stereotyped). The pilot's 3x inflation would show up here as a slope of about 3.
4. **Cost on the leaderboard.** Dollars per task next to the score. SimBench and Twin-2K ignore cost; for a public-sector buyer it is half the decision.
5. **Prompt-perturbation variance as a required column** (see Idea B). A model's score is a distribution, and the leaderboard shows its spread.

### Positioning against prior work
- SimBench (ICLR 2026): surveys, population-level distributions, no behaviour, no cost, no ceiling. Cite as the template and the gap.
- Hewitt et al. (Nature): LLMs predict experiment effects at r = 0.85, but on survey experiments, population level, one model family, and no public benchmark harness. Our contribution is the harness, the behavioural outcomes, and the segment level.
- SocSci210 / Socrates: 210 TESS experiments, open, individual-level. This is the closest competitor. Difference: TESS outcomes are attitudinal survey responses; ours are choices. Worth including one TESS task as a bridge so results are comparable.
- "When Synthetic Users Fail" (2026): shows gap inflation on GSS/WVS. We operationalise their finding as a benchmark metric.

### Effort and risk
- Data: two to three weeks to clean five to six tasks into one schema (the `make_cases.py` pattern already does this for Clarke). Author requests for Wolfson and Hall 2021 run in parallel and may fail; the benchmark works without them.
- Compute: at pilot scale, about $3 per model per task on Sonnet, $0.30 on Haiku. A full leaderboard of ten models at 200 draws per cell is a few hundred dollars.
- Main risk: most open behavioural RCTs have small true heterogeneity (Clarke's pooled baseline is 0.82). A benchmark where "ignore the segment" is near-optimal is still a useful result, but it needs to be framed as such: the benchmark's first finding would be that segment-level claims are mostly unsupported by the human data.
- Second risk: contamination. Clarke 2020 and Turnwald 2019 are in training data. Mitigation: report a contamination probe (ask the model for the paper's results directly), and prioritise 2024 to 2026 datasets and any unpublished data obtained by request.
- Venue: ICLR or NeurIPS datasets-and-benchmarks track, or EMNLP Findings. A Part II dissertation can be the benchmark plus the first leaderboard.

## Idea B: prompt-perturbation as the unit of measurement

### The observation
One semantically neutral sentence ("do not try to be healthy or make a point") moved Haiku 5.5 from rho 0.33 to 0.47 and its base rate from 0.07 to 0.19. The treatment effect in the human data is 0.41 to 0.63. So the prompt moved the estimate by more than the thing being estimated. Bisbee et al. (2024) showed this for survey items and an "analytic flexibility" paper (arXiv 2509.13397) showed it for simulation pipelines generally, but every leaderboard still reports a single number per model from a single prompt.

### The proposal
Treat each (model, design) score as a random variable over prompt wordings, and report its distribution.

1. **Generate a paraphrase family.** For each of the four prompt components (persona system prompt, task instruction, arm descriptions, output format) write or generate 5 paraphrases that hold the semantics fixed. That is 5^4 = 625 combinations; sample 25 to 30 by Latin hypercube so each component is varied independently.
2. **Run the benchmark under each sampled prompt.** Cheap because the choice task is short: on Clarke Study 2 at pilot scale that is about $8 for Sonnet, under $1 for Haiku, per 30 wordings.
3. **Decompose the variance.** Fit a mixed model: score ~ model + design + (1 | wording) + (1 | persona) + residual. Report the intraclass correlation for wording. If wording explains more variance than model, the leaderboard ordering is not a property of the models. Also report the sign-flip rate: the fraction of wordings under which the model's top-1 pick changes.
4. **Report each model as median and interquartile range**, and draw the cost-fidelity frontier with vertical bars. A model only dominates another if its whole interquartile range is above.
5. **Identify robust prompts.** Within the family, some wordings will sit near the centre for every model. Publish those as the benchmark's canonical prompts, with the evidence that they are central rather than lucky.

### Why it is a methodological contribution
- It converts a known nuisance into a measured quantity with a standard error, which is what every downstream user (a comms team, a regulator) actually needs.
- It gives a principled answer to "which prompt should I use": the one whose score is closest to the family median, not the one that scored highest.
- It generalises beyond this project: the same decomposition applies to any LLM eval, and almost none report it. A short paper with the decomposition on two or three existing benchmarks (OpinionQA, Twin-2K, plus ours) would be citable on its own.
- It is testable cheaply. The entire study on one dataset and three models is under $50.

### Specific hypotheses worth pre-registering
- H1: wording variance exceeds between-model variance among cheap models (Haiku 5.5, Gemini Flash-Lite, GPT-5-nano) but not among frontier models.
- H2: the poll design has higher wording variance than the replicate design, because it invites rationalisation.
- H3: wording sensitivity is largest for the base rate (how often the persona picks the unhealthy option) and smaller for the ranking of arms, which would mean relative claims are safer than absolute ones. The pilot hints at this: Haiku's population-level ordering was 0.75 to 0.89 across the two wordings while its base rate nearly tripled.

### Effort
Paraphrase families: a day. Runs: a day of wall-clock. Analysis: a week. This is the smaller, faster paper and can be a section of Idea A or stand alone.

## How the two fit together
Idea B is the measurement discipline that makes Idea A credible. A benchmark whose scores move 0.15 with a rewording is not a benchmark; one that publishes the spread and canonical central prompts is. If choosing one to start, start with B on the Clarke data already in hand: it needs no new data, costs under $50, and its result decides how Idea A should report scores.
