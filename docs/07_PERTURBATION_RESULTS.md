# Idea B results, part 1: Haiku 5.5 under 25 prompt wordings (22 Oct 2026)

Design: 5 wordings each of the persona system prompt, the task instruction, the six arm descriptions and the output-format line, combined in a 5x5 Graeco-Latin square (25 runs, every level of every component appears 5 times, pairs balanced). Same 48 fixed personas (6 per segment) in every run. Both designs: replicate (choice) and poll (points). Clarke 2020 Study 2 ground truth. Cost $2.45. Full tables: `results/clarke2020_study2_labels_perturb/ANALYSIS.md`. Sonnet half not yet run (needs go-ahead, about $25-30).

## Headline numbers

| design | rho median [IQR] | rho range | pooled-ordering rho range | base rate range |
|---|---|---|---|---|
| replicate (choice) | 0.16 [0.05, 0.36] | -0.11 to 0.66 | -0.23 to 0.86 | 0.03 to 0.38 |
| poll (points) | 0.02 [-0.10, 0.11] | -0.49 to 0.21 | -0.60 to 0.26 | n.a. |

The original wording from the pilot (w00) scored 0.37, in the top third. The single number reported in the pilot paper for Haiku 5.5 was therefore a mildly lucky draw.

## Four findings

1. **The wording range exceeds the model's whole score.** Haiku 5.5's replicate rho spans 0.77 units across wordings that any reviewer would call equivalent. Standard deviation 0.22. Any leaderboard entry for this model from one prompt is a sample of size one from a distribution wider than the gaps between models.

2. **Base rate is 90 percent task wording.** ANOVA on the 25 runs: the task-instruction component explains 90 percent of the variance in how often personas pick a lower-energy snack. Levels 0, 1 and 3 (all containing an honesty admonition: "do not try to be healthy", "truthfully, not aspirationally", "completely honest") give base rates of 0.04 to 0.06. Levels 2 and 4 (no admonition) give 0.24 to 0.29. Telling a small model to be honest about junk food makes it choose junk food 95 percent of the time. Humans in the control arm: 41 percent chose lower-energy.

3. **Rank accuracy is driven by a different component.** For the replicate design, rho variance is 39 percent persona system prompt, 24 percent residual, with task, arms and format each 10 to 14 percent. For the poll design it is 57 percent task wording. So the two things a user cares about, the level and the ordering, are controlled by different sentences, and fixing one does not fix the other. Hypothesis H3 from docs/06 is partly supported: ordering is more stable than level (pooled-ordering rho above 0.5 in 48 percent of replicate wordings) but it is far from stable.

4. **The poll design is wrong-signed under every wording.** Across all 25 poll wordings the modal top-1 pick in every segment was calories-only, the arm humans responded to second-least. 36 percent of poll wordings produced a negative population-level ordering correlation; the best poll wording reached 0.26. The replicate design's modal top-1 in every segment was pictorial-plus-calories, which is the human top-1 in 5 of 8 segments. The sign flip reported in the pilot is a property of the design, not of a prompt.

Sign-flip rate of the top-1 pick across wordings: 0.35 for replicate, 0.07 for poll. The poll is stable and wrong; the replicate is right on average and unstable.

## What this means for reporting

- Report each model as median and IQR over a published wording family, never a point.
- Keep honesty admonitions out of behavioural prompts; they are not neutral.
- Pre-register the wording family. The 25 wordings and the square are in `src/simsoc/perturb.py`.
- The pending Sonnet run tests H1: whether a frontier model's wording variance is smaller than its gap to Haiku. If Sonnet's IQR sits entirely above Haiku's range, the model choice survives the perturbation; if the ranges overlap, the leaderboard in the pilot paper is not supported.

# Part 2: Sonnet 5.5, replicate design, 10 wordings (22 Oct 2026, $8.27)

Ten of the 25 wordings, chosen so each system-prompt level and each task level appears twice (w00, 02, 06, 08, 12, 14, 15, 18, 21, 24). Same 48 personas. Replicate design only.

| model | n wordings | rho median [IQR] | range | pooled-ordering rho range | base rate range | top-1 median |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | 10 | **0.70 [0.58, 0.75]** | 0.41 to 0.80 | 0.77 to 1.00 | 0.17 to 0.63 | 0.38 |
| Haiku 5.5 | 25 | 0.16 [0.05, 0.36] | -0.11 to 0.66 | -0.23 to 0.86 | 0.03 to 0.38 | 0.38 |

## What it settles

1. **H1 holds: the model gap survives rewording.** Sonnet's worst wording (0.41) beats Haiku's third quartile (0.36). Mann-Whitney one-sided p = 1e-5. The ranges still overlap at the extremes (Haiku's single best wording, 0.66, exceeds Sonnet's two worst), so a one-prompt comparison could still get it wrong roughly one time in ten. The pilot paper's model claim stands if stated as a median with a bar: Sonnet 0.70 [0.58, 0.75], not 0.79.
2. **The pilot's 0.79 was a top-decile draw.** The original wording scored 0.67 at n=6 personas. Report 0.70.
3. **Ordering is robust, level is not.** Sonnet's population-level ordering correlation never fell below 0.77 across wordings, and reached 1.00 once. Its base rate swung from 0.17 to 0.63 around the human 0.41. Same pattern as Haiku, at a higher level. Relative claims ("which label wins") are defensible; absolute claims ("x percent will switch") are not without calibration. H3 supported on both models.
4. **Honesty admonitions depress the base rate on Sonnet too**, though less violently: task levels 0 and 3 (with admonition) 0.36 and 0.22; levels 1, 2, 4 at 0.51 to 0.63. Humans 0.41. The neutral wordings over-shoot healthy choice on Sonnet and under-shoot on Haiku, so there is no single "right" wording across models; only the median over a family is comparable.
5. Top-1 accuracy is the weak metric: median 0.38 on both models. Sonnet gets the ordering right but the human top-1 differs between pictorial and pictorial-plus-calories by one to five points per segment, below the ceiling's resolution. Top-1 on near-tied arms is noise and should be reported with a tie band.

The Sonnet ANOVA is not reported: 10 runs cannot support a 4-factor decomposition (16 parameters). The balanced 25-run version would cost another $12.

## Updated recommendation for the paper and benchmark
Report every (model, design) cell as median and IQR over the published wording family; state the pooled-ordering range separately from the segment-level rho; put a tie band on top-1; never report a base rate without a human anchor. With these rules the pilot's conclusions all survive: replicate beats poll, Sonnet beats Haiku, segment gaps are inflated, and absolute levels are prompt artefacts.
