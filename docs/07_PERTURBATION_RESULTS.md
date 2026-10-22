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
