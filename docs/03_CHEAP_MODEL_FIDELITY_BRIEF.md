# Research brief: which cheap LLMs most faithfully simulate persona / sub-group responses?

Compiled 2026-10-21 by a web-research agent. Prices checked against official pages or the Vercel AI Gateway catalogue on that date. UNVERIFIED items marked.

## 1. Headline conclusions
1. **Alignment/post-training is the dominant variable, not size.** Three independent 2026 studies find base models beat their own instruct/RLHF versions at *emulating* response distributions, with the gap widening at each stage (SFT -> DPO -> RLVR). Instruct models collapse to a modal answer per persona.
2. **Bigger is not better for minority portrayal.** OLMo-2-7B-Instruct beat Llama-3.3-70B-Instruct and Gemma-3-27B on stereotyping and Wasserstein distance to OpinionQA (the 70B and 27B were worse than random because they put >0.999 mass on one option in 88% of cases). At the aggregate "describe the population" task size does help (SimBench roughly log-linear in parameters).
3. **Fidelity-diversity trade-off**: highest per-persona fidelity models produce the most polarised populations (Chameleon's Limit; "When Synthetic Users Fail": segment gaps inflated 2-4x).
4. **No published benchmark scores Claude Haiku 5.5, GPT-5-nano/mini, or Gemini Flash-Lite on sub-group fidelity.** Only GPT-4.1-mini and Gemini 2.5 Flash have direct evidence (Twin-2K-500) and did as well as GPT-4.1. An empirical benchmark is genuinely needed.
5. **Chain-of-thought and thinking budgets do not help and sometimes hurt**; "describe the distribution" prompting helps for distributional targets but overshoots into over-dispersion; temperature 1.0 alone does not recover variance for instruct models.

## 2. Key papers
- **The Prompt Makes the Persona** (Lutz et al., EMNLP Findings 2025) https://arxiv.org/abs/2507.16076. OLMo-2-7B best on all open-ended measures and lowest opinion distance; Llama-3.3-70B and Gemma-3-27B below random baseline (WD 0.25). Interview-style and name-based priming reduce stereotyping.
- **Chameleon's Limit** (Xiao et al. 2026) https://arxiv.org/abs/2604.24698. Eight open models 4B-32B. Qwen3-32B highest per-persona fidelity (rho .94) and largest trait polarisation; Qwen3-4B best coverage (0.80). Persona fine-tuning narrowed population (coverage 0.47 -> 0.16). Disabling thinking changed little.
- **SimBench** (Hu et al., ICLR 2026) https://arxiv.org/abs/2510.17516. 45 LLMs, 20 datasets. Claude-3.7-Sonnet 40.8; GPT-4.1 34.6; DeepSeek-V3 32.9; Qwen2.5-72B 27.6; Qwen2.5-32B 23.8; Gemma-3-12B 18.6; Qwen2.5-7B 6.4; Llama-3.1-8B -0.15; OLMo-2-7B-Instruct -21.4. Instruction tuning helps on consensus items, hurts on contested ones (r = -0.94). Conditioning on a demographic group lowers every model's score. Extra inference compute does nothing (GPT-4.1 + CoT 34.6 -> 33.1).
- **Twin-2K-500** (Toubia et al. 2025) https://arxiv.org/abs/2505.17479. GPT-4.1-mini 71.7% vs GPT-4.1 71.1% vs Gemini-2.5-Flash 69.4%; random 59.2%; human retest 81.7%. CoT 70.4% (worse); temperature 0 vs 0.7 indistinguishable.
- **SubPOP** (ACL 2025) https://arxiv.org/abs/2502.16761. Fine-tuning cuts Wasserstein gap up to 46%; cheap open models + light fine-tuning on survey marginals is realistic.
- **Dominguez-Olmedo et al.** (NeurIPS 2024) https://arxiv.org/abs/2306.07951. Strong "A"-label bias; randomise option order; calibrate against uniform baseline.
- **Random Silicon Sampling** (Sun 2024) https://arxiv.org/abs/2402.18144. About 200+ samples per cell needed.
- **PersonaGym** https://arxiv.org/abs/2407.18416: model size does not imply better persona agents; GPT-4.1 and Llama-3-8B identical PersonaScore.
- **Reliability of persona-conditioned LLMs** (WWW 2026) https://arxiv.org/abs/2602.18462. Persona prompting hurt small subgroups.
- **When Synthetic Users Fail** (2026) https://arxiv.org/abs/2607.26348. No LLM beats a logistic baseline on individual prediction; between-segment gaps inflated 2-4x; scale does not fix either.

## 3. RLHF compresses diversity; mitigations
- **Instruction-tuned LMs cannot sample from distributions they can describe** (EMNLP Findings 2026) https://arxiv.org/abs/2607.25292. GPT-4o gave the identical answer on all 50 repeats for 57% of (persona, item) pairs at T=1.0. Base vs instruct TVD (7-8B): Llama-3.1 0.19 vs 0.34; Mistral 0.20 vs 0.45; Qwen2.5 0.20 vs 0.54. OLMo-2 stages: base 0.236 -> SFT 0.321 -> DPO 0.411 -> RLVR 0.442. Remedies: one "describe the distribution" call cuts TVD 0.46 -> 0.22; prompt-perturbed Argyle cuts it 21%; temperature 1.5 did not fix it.
- **Emulate or Estimate?** (2026) https://arxiv.org/abs/2608.03044. Base models track human inter-group differences (Spearman 0.61-0.75, compressed to ~70% of human magnitude); instruct models correlate worse (0.35-0.59) and exaggerate gaps ~2x. For direct distribution *estimation* instruct wins; Claude Opus 4.6 best (WD 0.267).
- **Distribution-First Population Simulation** (2026) https://arxiv.org/abs/2607.18310. Independent agents pile onto the mode; Verbalized Sampling removes under-dispersion but overshoots.
- **Verbalized Sampling** (Zhang 2025) https://arxiv.org/abs/2510.01171. Mode collapse = typicality bias in preference data; asking for k responses with probabilities gives 1.6-2.1x diversity.

Net: for emulation prefer base or lightly post-trained weights, or instruct + perturbation/verbalized-distribution prompting; for estimation a capable instruct model is fine and one call replaces hundreds.

## 4. Pricing (Oct 2026) and verdicts
| Model | $/MTok in/out | Fidelity evidence | Verdict |
|---|---|---|---|
| Claude Haiku 5.5 | 0.10 / 0.50 | none published; Claude family tops SimBench as estimator | benchmark |
| Claude Haiku 4.5 | 1.00 / 5.00 | none | older-gen comparison only |
| GPT-5-nano / mini | 0.05/0.40; 0.25/2.00 | GPT-4.1-mini = GPT-4.1 on Twin-2K | benchmark nano |
| Gemini 2.5 Flash-Lite | 0.10 / 0.40 | Gemini 2.5 Flash 69.4% Twin-2K | benchmark |
| Qwen3-32B base + instruct | ~0.08 / 0.28 | highest per-persona fidelity, most polarised; base beats instruct on emulation | benchmark both |
| OLMo-3-7B/32B base + instruct | self-host | best cheap evidence on minority portrayal | benchmark (ablation) |
| DeepSeek V4-Flash | 0.06 / 0.18 (gateway) | V3 best open non-reasoning on SimBench | benchmark |
| Llama 3.3 70B Instruct | 0.72 / 0.72 | worse than random on OpinionQA WD | negative control |
| Gemma-3-27B / Gemma-4 | ~0.14 / 0.40 | worst tier on stereotyping | negative control |

## 5. Heuristics
- Temperature 1.0 + many samples does not recover variance for instruct models; use perturbation / distribution prompting.
- First-person interview backstory helps (Lutz; Park 2024), but measure group-gap preservation, not just accuracy (blue-shift risk).
- CoT neutral to harmful; thinking budgets give nothing. Turn thinking down.
- Option-order randomisation mandatory.

## 6. Metric design
1-Wasserstein per item plus TVD; dispersion ratio (target 1.0, flag <0.7 or >1.3); sub-group-gap preservation (Spearman across items and slope, 1.0 preserved, >1 stereotyped); sign-flip rate; Kendall tau of message rankings per subgroup plus Kendall's W to detect artificial unanimity; >=200 draws per cell; non-LLM baseline that any model must beat; randomised option order; fixed seeds and dated snapshots.
