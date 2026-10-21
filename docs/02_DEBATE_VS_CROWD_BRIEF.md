# Research brief: do multi-agent methods beat single-persona prompting or classifiers for predicting sub-population response to messages?

Compiled 2026-10-21 by a web-research agent; numbers taken from the cited papers unless marked UNVERIFIED.

**Short answer:** No study found shows multi-agent debate, agent societies, or argue-then-vote architectures outperforming (a) a well-aggregated set of *independent* persona samples or (b) a supervised text classifier at predicting how a demographic sub-population responds to a message. The debate literature itself now says majority voting over independent samples captures most of the gain, and the social-simulation literature finds agent interaction homogenises stances without improving population fidelity. What does have evidence: independent silicon crowds for **relative** message ranking and aggregate treatment-effect prediction, grounded (interview/survey-based) personas, distribution elicitation, and small-human-sample calibration.

## 1. Multi-agent debate (MAD)
- Du et al. 2023 (ICML 2024): critique-and-revise debate improves factuality and arithmetic. https://arxiv.org/abs/2305.14325
- Liang et al. MAD (EMNLP 2024): "degeneration of thought" motivation; tit-for-tat debate with judge. https://arxiv.org/abs/2305.19118
- Smit et al. "Should we be going MAD?" (ICML 2024): debate did not consistently beat self-consistency/ensembling; hyperparameter-sensitive. https://proceedings.mlr.press/v235/smit24a.html
- "Stop Overvaluing MAD" (2025): 5 methods x 4 models x 9 benchmarks; MAD often fails to beat CoT and self-consistency; **model heterogeneity** is the one reliable improver. https://arxiv.org/abs/2502.08788
- "Debate or Vote" (2025): majority vote over independent samples beat every MAD variant (Qwen2.5-7B single 0.72, MV 0.77, best MAD 0.74). Theory: homogeneous agents' beliefs form a martingale, so debate cannot raise expected correctness. https://arxiv.org/html/2508.17536v1
- "Demystifying MAD" (Findings ACL 2026): vanilla MAD underperforms MV; fixes are diversity of initial viewpoints and calibrated confidence sharing. https://arxiv.org/abs/2601.19921
- "The Deliberative Illusion" (2026): deliberation lost up to 72.4% of issue-critical facts by round 3; stance entropy collapsed to unanimity in every topology; persona prompting changed speed of convergence, not fact loss. https://www.alphaxiv.org/abs/2606.03032
- 2026 MAD survey: shared personas produce premature convergence and shared errors. https://arxiv.org/html/2607.26212v1

**Bottom line:** debate is a consensus mechanism. Its documented behaviour (homogenisation, entropy collapse, information loss) is the opposite of what a population simulator needs.

## 2. LLM social simulation of audiences
- Park et al. 2024, "1,000 People": interview agents 85% normalised GSS accuracy vs 77% demographic-only; effect-size correlation 0.99 normalised; smaller accuracy gaps across race/ideology. https://arxiv.org/abs/2411.10109
- Twin-2K-500 (Toubia et al. 2025): 2,058 respondents; twins 68-72% item accuracy vs 59% random, 81.7% human retest. https://arxiv.org/abs/2505.17479
- "Digital Twins as Funhouse Mirrors" (2025): 19 preregistered studies, 164 outcomes; twin-human r = 0.20; five distortions: insufficient individuation, stereotyping, representation bias, ideological blue-shift, hyper-rationality. Richer data improved relative heterogeneity, not point predictions. https://arxiv.org/abs/2509.19088
- OpinionQA (Santurkar 2023): misalignment persists under steering; 65+ and widowed worst served. https://proceedings.mlr.press/v202/santurkar23a
- SubPOP (ACL 2025): fine-tuning on 70K subpopulation-response pairs cut Wasserstein distance by up to 46%. https://arxiv.org/abs/2502.16761
- "The Chameleon's Limit" (2026): persona collapse in all 8 models; sampling more personas does not fix it; fidelity-diversity trade-off. https://arxiv.org/abs/2604.24698
- "Persona manifold collapse" (2026): simple age-gender personas beat rich ICP personas at predicting tweet engagement and email CTR. https://arxiv.org/pdf/2606.18263
- 120K-persona Serbian study (Bojic et al., Sci Reports 2026): 1,511 participants, 27 LLMs; binary like/dislike MCC 0.29 vs TF-IDF classifier MCC 0.36; richer personas gave small inconsistent gains (69.6 / 71.2 / 71.5%, UNVERIFIED precision). https://www.nature.com/articles/s41598-026-66277-8
- "When Synthetic Users Fail" (2026): Claude Haiku 4.5 / Sonnet 4.6, Llama 8B/70B on GSS and WVS. Individual level: best LLM tied a demographic lookup table, lost to logistic regression. Aggregate distributions decent with a distribution prompt (JS 0.011-0.023 vs 0.056-0.090 single-answer). **Demographic over-determination**: segment-targeting decisions point to the wrong segment in 50% (GSS) / 72% (WVS) of cases, gaps inflated 2.3-2.5x. https://arxiv.org/html/2607.26348v1
- Persona-conditioned WVS reliability (2026): persona prompting often degraded alignment, hurting under-represented subgroups most. https://arxiv.org/abs/2602.18462
- Pew "silicon samples" (Sept 2026): best config (Claude Opus 4.6) 11-15 pp absolute error; person-level match ~55%. https://www.pewresearch.org/data-labs/2026/09/30/methodology-silicon-samples/
- Korean census-persona deliberation (2026): 640 personas, 6-agent rooms, 3 rounds; **deliberation did not improve fidelity**; sealed-monologue controls ended within one agent of full debates. https://arxiv.org/html/2609.07573v1
- Opinion-dynamics agents drift to factual consensus regardless of persona (Chuang 2023); simulations highly sensitive to trivial prompt edits (Ju 2024). https://arxiv.org/abs/2412.05093
- "LLM Voting" (AIES 2024): LLM voters more uniform than humans; CoT did not improve accuracy. https://arxiv.org/abs/2402.01766
- PyMC Labs / Colgate (57 surveys): free-text-then-embed-to-Likert reached ~90% of human test-retest reliability vs ~80% for direct Likert; direct numeric prompts collapse to midpoint. https://arxiv.org/abs/2510.08338
- Counter-evidence: Verasight GPT-5 coffee replication 19.8 pp error, fell apart at subgroup level. https://www.verasight.io/reports/coffee-llm

## 3. LLMs predicting persuasion / message effectiveness
- Hewitt / Ashokkumar et al. (Nature 2026): GPT-4 simulating representative samples predicted treatment effects of 70 preregistered survey experiments at r = 0.85 (0.90 on unpublished); systematically overestimated effect sizes; megastudies showed lower correlations. https://ai4pb.stanford.edu/projects/predicting-results-of-social-science-experiments-using-large-language-models
- Lippert et al. (RSOS 2024): GPT-4 aggregated forecasts of a megastudy r = 0.89 with realised effects (experts 0.87). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11421891/
- Anthropic (Durmus 2024): model-based persuasiveness scores correlated poorly with human judgments. https://www.anthropic.com/research/measuring-model-persuasiveness
- Rescala et al. (EMNLP 2024): predicting a demographically described voter's stance 42.8% vs supervised XGBoost 55-58%. https://arxiv.org/html/2404.00750v2
- "LLM Persuasion Is in the Eye of the Evaluation" (2026): nine persuasion measures agree at mean Spearman 0.25. https://arxiv.org/html/2610.10232v1
- Hackenburg & Margetts (PNAS 2024): microtargeted messages not more persuasive than untargeted. Hackenburg et al. (Science 2025): post-training and prompting matter far more than personalisation or scale. https://arxiv.org/abs/2507.13919
- Matz et al. (Sci Reports 2024): trait-tailored messages beat generic across 7 sub-studies. https://pmc.ncbi.nlm.nih.gov/articles/PMC10897294
- Wisdom of the Silicon Crowd (Sci Adv 2024): 12-model ensemble matched a 925-human forecasting crowd; acquiescence bias. https://arxiv.org/abs/2402.19379

## 4. Prompting / sampling design
- Lutz et al. (EMNLP 2025): interview-style and name-based prompts reduce stereotyping; Llama-3.3-70B worse than OLMo-2-7B. https://arxiv.org/abs/2507.16076
- Anthology (EMNLP 2024): first-person generated backstories improved distribution match up to 18%. https://arxiv.org/abs/2407.06576
- Temperature: higher temperature improves distribution alignment; persona-prompted small models stay under-dispersed even then. https://arxiv.org/html/2411.10080v1
- Statistically selected persona attributes beat LLM-chosen ones. https://arxiv.org/html/2609.02526v1
- Calibration: PPI/DSL bias-correct aggregates from a small human subsample. https://arxiv.org/html/2510.11408v2

## 5. What works / what fails / recommended architecture
**Works:** aggregate treatment effects and relative rankings (r 0.85-0.90); independent sampling + aggregation; grounding in real individual data; eliciting distributions; model-family heterogeneity; fine-tuning on subpopulation distributions; small-human-sample calibration; interview-style first-person prompts at moderate-high temperature.

**Fails:** debate/deliberation as a population model; single-persona or label prompting for individual prediction; richer hand-written personas; demographic over-determination (segment conclusions wrong 50-72% of the time naively); direct numeric Likert; LLM-as-judge of persuasiveness alone; adding more personas to fix diversity; microtargeted generation. Critic/adversary agents: **no study either way** (UNVERIFIED); nearest evidence suggests adversarial rounds amplify majority drift.

**Recommended architecture:** relative ranking of K messages within a segment; grounded persona bank per segment with statistically selected attributes; independent silicon crowd across at least two model families, distribution elicitation; hybrid with a classifier (stacking); small human pilot for PPI/DSL calibration; population-level diagnostics as release gates (stereotyping index, gap-inflation ratio, acquiescence); if a second stage is wanted use diversity-aware initialisation + confidence-weighted aggregation, never convergence rounds; report rank correlation and decision-level accuracy on held-out human results.

## 6. Open questions
No study directly pits debate architectures against independent crowds on message response for defined sub-populations. Whether a critic agent helps calibration is untested. Whether grounded personas beat classifiers on novel messages for novel segments (where classifiers have no signal) is the key unanswered question for this tool.
