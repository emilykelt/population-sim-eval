# Plan: validating a simulated-audience tool for public food messaging

Written and executed 2026-10-21. Owner: Emily Kelt, with Dev Sharma.

## Goal
A tool that, given K candidate public food/nutrition messages and a target audience, predicts which message lands best for which segment and suggests tweaks. Before any deployment it must demonstrably beat cheaper methods on held-out human data.

## What the literature fixed before we started (docs/01-03)
1. Elicit **relative** preference (points across variants), never absolute ratings of a single health message.
2. Ground personas in **first-person interview text**, not demographic labels.
3. Use **independent** persona crowds. Consensus debate homogenises; the only debate worth testing is an adversarial same-community critic, which no paper has evaluated.
4. Treat small, cheap models as contenders, not fallbacks: RLHF strength, not size, drives persona collapse.
5. Report against two human reference points: the pooled-population ranking (does segmentation add anything?) and the split-half noise ceiling (what is even predictable?).

## Ground truth chosen (docs/04)
Clarke et al. 2020, OSF k7tw5. Fully open, UK, individual demographics.
- **Study 2** (N = 4,134): six labelling schemes on vending-machine snacks; outcome = chose an energy-dense snack. Segments: gender x age band (6 cells), education (2 cells). Split-half ceiling 0.60 / 0.88. Pooled baseline 0.82 / 0.94.
- **Study 1** (N = 4,607): six disease framings on a chocolate bar; outcome = desire to eat it. Split-half ceiling 0.17 / 0.21, so humans barely distinguish the framings. Used as a **null check**: a method that confidently separates segments here is over-determining.

## Methods compared (src/simsoc/elicit.py)
| method | what it is | role |
|---|---|---|
| judge | no persona; expert model predicts the segment | LLM-judge baseline |
| label_single | demographic-label persona | the naive competitor approach |
| grounded_single | one interview-grounded persona, 5 samples | tests grounding |
| grounded_ensemble | 12 grounded personas per cell, independent | the silicon crowd |
| grounded_debate | crowd + adversarial critic + revision | the one debate design worth testing |

Models: Claude Haiku 5.5 ($0.10/$0.50), Claude Haiku 4.5 ($1/$5), Claude Sonnet 5.5 ($2/$10). Non-Anthropic cheap models (Gemini Flash-Lite, GPT-5-nano, Qwen3, DeepSeek) are wired through Vercel AI Gateway in `providers.py` but the gateway refuses calls until a card is on the Vercel team.

## Metrics (src/simsoc/metrics.py)
- mean Spearman between predicted and human ranking of variants, per segment
- top-1 accuracy (did we pick the human-best variant for the segment)
- gap preservation: correlation of between-segment differences, message by message
- within-segment SD of allocated points (collapse detector)

## Steps
1. Synthesise vault papers. Done: docs/01.
2. Three research briefs (debate vs crowd; cheap-model fidelity; case studies). Done: docs/02-04.
3. Engine + benchmark harness. Done: src/.
4. Download Clarke data, build cases, compute baselines and ceilings. Done: data/*.yaml.
5. Run benchmark. Results: results/*/REPORT.md and docs/05_RESULTS.md.
6. Decide on architecture and model from the evidence. docs/05_RESULTS.md.

## Next steps after this session
- Add a card to the Vercel team and rerun `run_benchmark.py` with `--models google/gemini-2.5-flash-lite openai/gpt-5-nano alibaba/qwen-3-32b deepseek/deepseek-v4-flash` (the cheap-model shortlist from docs/03).
- Request microdata for Hall 2021 (topic x design x Latino parents) and Wolfson 2022, the two datasets with real published heterogeneity.
- Ask FSA social science team for the withdrawn 2023 rapid communications-testing data (43 real FSA messages).
- Build personas from Food and You 2 microdata (UK Data Service SN 8814) instead of synthetic life facts.
- Add PPI/DSL calibration from a small human pilot before any client use.
