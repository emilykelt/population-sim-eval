# Results: first validation run, 2026-10-21

Ground truth: Clarke et al. 2020 (OSF k7tw5), Study 2. 4,134 UK adults, six labelling arms, outcome = whether the person chose an energy-dense snack from a 12-item vending machine. Segments: gender x age band (6) and education (2). Full tables in `results/clarke2020_study2_labels/REPORT.md`. Total API spend for everything in this document: about $9.

## Human reference points
| segmentation | pooled-population baseline (Spearman) | split-half human ceiling |
|---|---|---|
| gender x age | 0.82 | 0.60 |
| education | 0.94 | 0.88 |

The pooled baseline is high. In this dataset the real segment-level heterogeneity is small: the labels that work for the whole population work for every segment. Beating 0.82 is not realistic; matching it while preserving between-segment gaps is the bar.

## Headline table (mean Spearman between predicted and human ranking of the six arms, per segment)

| model | ask-the-persona methods (judge / label / grounded single / crowd of 12 / crowd + critic) | behavioural replication (grounded choice) | cost of the choice run |
|---|---|---|---|
| Claude Haiku 5.5 | 0.29 / 0.12 / -0.39 / -0.37 / -0.36 | 0.33 (0.47 with neutral prompt) | $0.26 |
| Claude Haiku 4.5 | 0.40 / -0.24 / -0.45 / -0.38 / 0.01 | -0.17 | $1.17 |
| Claude Sonnet 5.5 | 0.78 / 0.41 / n.a. / 0.20 / 0.26 | **0.79** (top-1 0.62, gap preservation 0.36) | $2.10 |

## Five findings

**1. Asking personas which message would work on them predicts the opposite of what people did.** Every persona method, on every model, allocated most points to calorie badges and least to graphic warnings, with rationales like "gory pictures would make me roll my eyes". Real participants chose fewer energy-dense snacks under graphic warnings (0.63 lower-energy) than under calorie badges (0.46) or no label (0.41). This is a stated-versus-revealed gap: self-reported immunity to fear appeals is a known human bias and the models reproduce it faithfully. For message testing, do not ask simulated people what would persuade them.

**2. Replicating the experiment works, on a capable model.** Putting each grounded persona in one arm and recording a snack choice (the Turing-experiment design from Aher et al. and Hewitt et al.) gave Sonnet 5.5 a mean rho of 0.79 against the human segment rankings, top-1 accuracy 0.62, and positive gap preservation. It reproduced the human ordering (pictorial > text > calories > nothing) in every segment. This matches the pooled-population baseline (0.82) and exceeds the human noise ceiling for the gender x age split (0.60).

**3. The cheap models have no signal yet, and the reason is collapse, not reasoning.** Haiku 5.5 and 4.5 personas chose an energy-dense snack 94 to 97 percent of the time regardless of arm (humans: 59 percent in the control arm). With no variance there is nothing to rank. Removing one sentence from the prompt ("do not try to be healthy or make a point") raised Haiku 5.5's lower-energy rate from 0.07 to 0.19 and its rho from 0.33 to 0.47. That is the Bisbee et al. sensitivity result reproduced: cosmetic prompt changes move the answer more than the treatment does. Haiku is not usable as a persona engine in this form. It may be usable as a cheap judge (0.29 to 0.40), which is still worse than Sonnet's judge (0.78).

**4. Sonnet over-determines demographics.** Its choice simulation put the no-degree segment at 0.1 to 0.3 lower-energy and the degree segment at 0.3 to 0.9. Humans: 0.36 to 0.62 and 0.44 to 0.66. The direction is right (degree holders respond slightly more) but the gap is inflated roughly three-fold, exactly the 2 to 4x inflation reported by "When Synthetic Users Fail" (2026). Any product output needs a shrinkage step calibrated on a human pilot before segment differences are shown to a comms team.

**5. The adversarial critic does not help.** The critic moved allocations by 17 to 29 points per persona and increased within-segment variance slightly (SD 5.1 to 5.9 on Sonnet), so it is not a consensus-collapse mechanism, but rank accuracy stayed flat (Haiku 5.5: -0.37 to -0.36; Sonnet: 0.20 to 0.26) at two to three times the cost. Because the underlying elicitation (finding 1) is wrong-signed, a critic cannot rescue it. Debate is dropped from the architecture.

**Null check (Study 1, six disease framings).** Humans barely distinguish the framings (split-half ceiling 0.17). Haiku 5.5's methods scored 0.17 to 0.38, consistent with noise, and the best per-segment predictions still "found" a preferred framing in every cell. A tool must report the ceiling alongside its prediction or it will present noise as insight.

## Decisions
- **Architecture:** grounded personas, independent crowd, between-subjects behavioural replication of the actual decision (choose, buy, click), not a persuasiveness poll. No debate rounds. Report rank plus a shrunk segment-gap estimate plus the human ceiling where one exists.
- **Model:** Sonnet 5.5 for persona simulation at roughly $2 per 6-arm, 8-segment study. Haiku 5.5 only for backstory generation and bulk parsing. Re-test Haiku 5.5 with the neutral prompt and verbalized-distribution elicitation before ruling it out; its cost is 20x lower.
- **Non-Anthropic cheap models** (Gemini 2.5 Flash-Lite, GPT-5-nano, Qwen3-32B, DeepSeek V4-Flash) remain untested because the Vercel AI Gateway needs a card on the team. `providers.py` already routes them; one command reruns the benchmark.

## What this run cannot tell us
- Only one dataset with real behavioural outcomes and little true heterogeneity. The two datasets with published heterogeneity (Hall 2021: label design x Latino parents; Wolfson 2022: climate labels x sex) need author requests.
- Persona backstories were generated from synthetic life facts. Grounding in Food and You 2 microdata is the next step and should narrow the over-determination in finding 4.
- n = 8 to 12 personas per cell with 2 draws. Sun et al. suggest 200 draws per cell for stable distributions; the Sonnet result should be re-run at that scale (about $20) before being quoted.
- Single prompt wording per method. Finding 3 shows wording matters; a perturbation sweep is needed before any number here is treated as a property of the model rather than of the prompt.
