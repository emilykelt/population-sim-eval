# Synthesis: the Simulated Societies papers in the vault

Source: `Startup Stuff/Simulated Societies Papers/` (16 notes) plus `Simulated Societies.md`, `Simulated Societies Research Papers.md`, `Competitor Analysis - simulated societies.md`. Written 2026-10-21 for the food-messaging tool being built with Dev Sharma.

## 1. One-paragraph verdict

The literature splits cleanly. The optimistic papers (Argyle, Horton, Park 2023, Park 2024, Aher) show that an LLM conditioned on a backstory reproduces *group-level* response distributions surprisingly well, especially on neutral topics and when grounded in interview-style first-person material. The critical papers (Wang, Bisbee, Park-Schoenegger-Zhu, Salecha, Sun, Cheng, Aher's own wisdom-of-crowds result) show the same model fails in a specific, predictable way: it **flattens within-group variance, drifts toward the socially desirable answer, caricatures marginalised groups, and is brittle to cosmetic prompt and model-version changes.** The failures are worst exactly where a GovTech / public-comms product cares most: minority groups and contested topics. Nothing in the vault says the method is useless. Everything in the vault says the method must be validated against real human data per use case, and that the product's defensible edge is calibrated uncertainty plus bias monitoring, not raw accuracy claims.

## 2. Paper-by-paper, grouped by what they tell the tool

### 2a. Why it can work at all

| Paper | Claim we can lean on | Caveat for us |
|---|---|---|
| Argyle et al. 2023, *Out of One, Many* | "Algorithmic fidelity": GPT-3 conditioned on real respondents' backstories reproduces subgroup response distributions on US surveys. Published in *Political Analysis*. | One model, US only, survey items not messages. Fidelity is group-level, not individual. |
| Horton et al. 2023/2026, *Homo Silicus* | Frame agents as (endowments, preferences, information). Replicates Charness-Rabin, Kahneman fairness, status-quo bias. | Validated against lab economics, not field behaviour. Use the 2026 revision. |
| Park et al. 2023, *Generative Agents* | Memory, reflection, planning loop gives persistent believable agents. | HCI believability study. Not evidence of statistical accuracy. Do not cite for accuracy. |
| Park et al. 2024, *1,000 People* | Grounding agents in 2-hour qualitative interview transcripts gives 85% replication of own answers on GSS and beats demographic-only prompting. | This is Simile AI's moat. We will not have interviews for UK food audiences. The transferable lesson is **rich first-person grounding beats demographic labels.** |
| Aher et al. 2023, *Turing Experiments* | Three of four classic studies replicate. | The fourth (Wisdom of Crowds) shows **hyper-accuracy distortion**: agents know too much, so crowd aggregation loses the diversity-of-error that makes it work. Direct warning for any "ensemble of agents" or debate design. |
| *The Prompt Makes the Persona* 2025 | Interview-style and name-based priming reduce stereotyping. **OLMo-2-7B beats Llama-3.3-70B on minority-group fidelity.** | Strongest single piece of evidence that cheap/small models are not automatically worse at persona work, and that RLHF at scale can hurt. |

### 2b. Why it fails, and how

| Paper | Failure mode | Implication for the food tool |
|---|---|---|
| Wang, Morgenstern, Dickerson 2024 | **Misportrayal + flattening** of identity groups. Conditioning helps but never removes it. | Report within-group spread, not only the mean. Never claim to represent a minority group without a human check. |
| Bisbee et al. 2024 | Estimates swing with wording, ordering, and model checkpoint. | Pin model versions. Run prompt-perturbation sweeps and report the spread as uncertainty. Reproducibility is an engineering requirement. |
| Park, Schoenegger, Zhu 2023 | **Conservative bias**: GPT-3.5 across 14 Many Labs 2 studies gives majority-normative answers; variance well below human. | Expect the simulator to over-predict "I'd eat healthier" responses. Calibrate against a human anchor. |
| Salecha et al. 2024 | **Social desirability bias** scales with capability (GPT-4 shifts 1.2 SD). Model infers it is being evaluated. | Health messaging is maximally socially desirable. Avoid "rate this message 1-5" framings that invite virtue signalling. Prefer forced-choice, behavioural, or relative-ranking elicitation. |
| Sun et al. 2024, *Random Silicon Sampling* | Good alignment on neutral topics, sharp divergence on sensitive topics and marginalised groups. Group-level conditioning is a useful baseline. | Food is mostly "neutral" so prospects are better than for politics. Sugar tax, obesity, religious diet, and poverty framings are where it will break. |
| Cheng, Piccardi, Yang 2023, *CoMPosT* | **Caricature**: steering toward a demographic amplifies the model's existing stereotype. | Do not use steering vectors for demographic personas. Use grounding text instead. |
| 2025 survey, *LLM Social Simulations Are a Promising Research Method* | Five failure modes; **alienness** is the key one: right output, non-human mechanism, so generalisation to new messages is unsafe. | Every claim must be backed by held-out human data. Validation per campaign, not once. |

### 2c. Interpretability and control

| Paper | What it adds |
|---|---|
| Chen, Arditi, Sleight, Evans, Lindsey 2025, *Persona Vectors* | Traits are single directions in the residual stream. Monitoring a projection gives a live readout of drift. **Assistant Axis** (2026) adds that persona space is low-rank (4-19 PCs explain 70%). |
| Neplenbroek et al. 2025/2026 | Demographic probes on real conversations are weak (0.60-0.70 balanced accuracy) and work mostly through **topic as proxy**. On templated data they hit 100%, which is leakage. |

Both are relevant to the long-term "calibrated confidence" differentiator but need open-weight models and are out of scope for the first prototype, which will run on API models.

### 2d. The baseline we must beat

| Paper | Baseline |
|---|---|
| Moraes et al. 2023 (Booking.com), *Uplift Modeling* | Once exposure data exists, causal uplift models beat LLM personas for "who responds to which treatment". |
| Competitor note: 120K-persona Serbian engagement study | LLM personas did not beat plain text classifiers on engagement prediction. |
| Taubenfeld et al. 2026, *Fuse* | Shows how to manufacture verifiable ground truth by construction. We cannot assign hidden food preferences to real people, so our ground truth must come from existing message-testing RCTs. |

So the defensible LLM value proposition is **cold start**: predicting before any exposure data exists, and **explaining why** a message lands or misses for a segment in a way a comms team can act on.

## 3. What the market context in the vault says

From the competitor analysis: eight funded players, nobody is API-first, nobody outputs calibrated uncertainty, healthcare and public-health comms is unclaimed, and the two structural risks are the accuracy problem and Goodhart (messaging converges to what the simulator likes). The vault's own GovTech wedge: statistics with error bars, explainability, equality and bias monitoring, deliberate representation of minority populations.

## 4. Design constraints the synthesis imposes on the tool

1. **Elicit relative, not absolute.** Ask personas to rank or choose between message variants and to predict behaviour, never to grade a single health message. This sidesteps social desirability (Salecha, Sun) and matches what comms teams need (which variant for which audience).
2. **Ground personas in first-person text, not labels.** Interview-style backstories built from real survey microdata (Food and You 2, Health Survey for England) rather than "You are a 45-year-old woman in Leeds". (Park 2024, Prompt Makes the Persona.)
3. **Sample many agents at temperature 1 and keep the distribution.** Report spread, a flattening index (simulated variance over human variance), and subgroup gaps. (Wang, Park-Schoenegger-Zhu.)
4. **Debate only with an audience-critic, never for consensus.** Consensus debate would compound the hyper-accuracy and homogenisation problems (Aher). Use adversarial roles: a persona answers, a "sceptical member of the same community" pushes back, the persona revises. Measure whether this restores variance or destroys it.
5. **Perturbation sweeps as uncertainty.** Re-run with paraphrased prompts and a second model; report the envelope. (Bisbee.)
6. **Validate on held-out human RCT data before any client use.** The success criterion is beating (a) single-persona prompting, (b) a demographic-only prompt, (c) a text-classifier or no-persona LLM judge, on rank correlation of message effectiveness per subgroup.
7. **Cheap model choice is an empirical question.** The vault's evidence (OLMo-2-7B result, RLHF compression) means small models should be tested head to head, not assumed worse.
8. **Minority-group guard rails.** Flag any segment where simulated spread collapses or where the model's output for a marked group diverges sharply from its output for the unmarked group with the same backstory (Cheng, Wang).

## 5. Gaps in the vault worth closing

- No paper on **message or persuasion prediction** specifically. Everything is survey and lab-experiment replication.
- No **UK** or **food** ground-truth dataset identified.
- No **head-to-head cost/fidelity comparison** of cheap models.
- Park 2024 is linked but has no note of its own.

The three research briefs in `docs/02_*`, `docs/03_*`, `docs/04_*` address these gaps.
