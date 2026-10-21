## clarke2020_study2_labels

Source: Clarke et al. 2020, Appetite; data osf.io/k7tw5 Study 2 (N=4,134 UK adults, 6 arms, primary outcome = chose energy-dense snack in a 12-item vending machine task). Truth = 1 - P(chose energy-dense) per segment per arm.

Human reference points (Spearman, segment ranking of the 6 variants):
- gender+age_band: predict every segment with the pooled-population ranking = **0.819**; split-half human noise ceiling = **0.605**
- education: predict every segment with the pooled-population ranking = **0.943**; split-half human noise ceiling = **0.875**

| model | method | mean rho | top-1 acc | gap preservation | within-seg SD | valid answers | cost $ | secs |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5-20251001 | grounded_choice | -0.170 | 0.12 | -0.011 | 0.1 | 1151/1152 | 1.168 | 125.9 |
| claude-haiku-4-5-20251001 | grounded_debate | 0.007 | 0.25 | 0.129 | 12.1 | 486/96 | 0.528 | 68.1 |
| claude-haiku-4-5-20251001 | ... debate round 1 only | -0.379 | 0.00 | -0.034 | | 474 | | |
| claude-haiku-4-5-20251001 | grounded_ensemble | -0.379 | 0.00 | -0.034 | 10.5 | 474/96 | 0.247 | 34.9 |
| claude-haiku-4-5-20251001 | grounded_single | -0.450 | 0.00 | -0.125 | 8.4 | 186/40 | 0.102 | 15.5 |
| claude-haiku-4-5-20251001 | judge | 0.397 | 0.25 | -0.104 | 4.8 | 240/40 | 0.138 | 28.2 |
| claude-haiku-4-5-20251001 | label_choice | 0.016 | 0.38 | 0.178 | 0.0 | 576/576 | 0.383 | 60.1 |
| claude-haiku-4-5-20251001 | label_single | -0.236 | 0.00 | -0.095 | 8.0 | 180/40 | 0.082 | 13.2 |
| claude-haiku-5-5 | grounded_choice | 0.330 | 0.12 | 0.033 | 0.2 | 1152/1152 | 0.264 | 235.6 |
| claude-haiku-5-5 | grounded_choice_neutral | 0.465 | 0.50 | 0.374 | 0.3 | 1152/1152 | 0.278 | 248.5 |
| claude-haiku-5-5 | grounded_debate | -0.364 | 0.00 | -0.072 | 8.5 | 558/96 | 0.103 | 83.7 |
| claude-haiku-5-5 | ... debate round 1 only | -0.371 | 0.00 | 0.096 | | 546 | | |
| claude-haiku-5-5 | grounded_ensemble | -0.371 | 0.00 | 0.096 | 7.0 | 546/96 | 0.049 | 43.5 |
| claude-haiku-5-5 | grounded_single | -0.386 | 0.00 | 0.024 | 4.9 | 222/40 | 0.022 | 25.2 |
| claude-haiku-5-5 | judge | 0.293 | 0.12 | -0.129 | 3.8 | 240/40 | 0.022 | 23.5 |
| claude-haiku-5-5 | label_choice | 0.262 | 0.38 | 0.083 | 0.0 | 576/576 | 0.085 | 91.6 |
| claude-haiku-5-5 | label_single | 0.121 | 0.00 | 0.078 | 5.1 | 234/40 | 0.018 | 18.2 |
| claude-sonnet-5-5 | grounded_choice | 0.791 | 0.62 | 0.363 | 0.4 | 768/768 | 2.104 | 202.6 |
| claude-sonnet-5-5 | grounded_debate | 0.264 | 0.00 | 0.279 | 5.9 | 384/64 | 0.888 | 75.3 |
| claude-sonnet-5-5 | ... debate round 1 only | 0.200 | 0.00 | 0.080 | | 384 | | |
| claude-sonnet-5-5 | grounded_ensemble | 0.200 | 0.00 | 0.080 | 5.1 | 384/64 | 0.420 | 38.4 |
| claude-sonnet-5-5 | judge | 0.779 | 0.38 | 0.139 | 2.4 | 240/40 | 0.322 | 39.6 |
| claude-sonnet-5-5 | label_single | 0.414 | 0.38 | -0.166 | 3.5 | 240/40 | 0.220 | 28.7 |

Best for claude-haiku-4-5-20251001: judge (rho 0.397). Per segment:
- age_band=18-34|country=UK|gender=man: rho 0.37, top1 miss; pred [28.5, 28.0, 22.1, 8.9, 8.8, 3.7] vs truth [0.56, 0.416, 0.286, 0.615, 0.412, 0.314]
- age_band=35-54|country=UK|gender=man: rho 0.66, top1 hit; pred [23.8, 26.3, 20.6, 12.6, 12.5, 4.2] vs truth [0.517, 0.527, 0.352, 0.508, 0.474, 0.421]
- age_band=55+|country=UK|gender=man: rho 0.35, top1 miss; pred [25.8, 28.2, 20.4, 11.8, 11.8, 1.9] vs truth [0.613, 0.574, 0.507, 0.575, 0.579, 0.451]
- age_band=18-34|country=UK|gender=woman: rho 0.26, top1 miss; pred [18.1, 30.7, 24.0, 11.2, 11.3, 4.7] vs truth [0.667, 0.466, 0.414, 0.664, 0.439, 0.341]
- age_band=35-54|country=UK|gender=woman: rho 0.14, top1 miss; pred [20.4, 32.0, 24.4, 8.6, 10.6, 4.0] vs truth [0.707, 0.615, 0.507, 0.713, 0.587, 0.41]
- age_band=55+|country=UK|gender=woman: rho 0.83, top1 hit; pred [21.6, 25.8, 25.6, 10.8, 9.6, 6.6] vs truth [0.657, 0.778, 0.663, 0.75, 0.637, 0.495]
- country=UK|education=no university degree: rho 0.54, top1 miss; pred [31.1, 26.2, 17.1, 12.4, 9.1, 4.1] vs truth [0.597, 0.539, 0.478, 0.617, 0.511, 0.355]
- country=UK|education=university degree: rho 0.03, top1 miss; pred [15.8, 40.2, 20.8, 9.4, 11.2, 2.6] vs truth [0.664, 0.602, 0.442, 0.649, 0.54, 0.491]

Best for claude-haiku-5-5: grounded_choice_neutral (rho 0.465). Per segment:
- age_band=18-34|country=UK|gender=man: rho 0.44, top1 miss; pred [0.1, 0.0, 0.0, 0.0, 0.0, 0.1] vs truth [0.56, 0.416, 0.286, 0.615, 0.412, 0.314]
- age_band=35-54|country=UK|gender=man: rho -0.34, top1 miss; pred [0.1, 0.0, 0.1, 0.1, 0.1, 0.0] vs truth [0.517, 0.527, 0.352, 0.508, 0.474, 0.421]
- age_band=55+|country=UK|gender=man: rho 0.12, top1 hit; pred [0.1, 0.1, 0.0, 0.0, 0.0, 0.0] vs truth [0.613, 0.574, 0.507, 0.575, 0.579, 0.451]
- age_band=18-34|country=UK|gender=woman: rho 0.34, top1 hit; pred [0.2, 0.1, 0.1, 0.1, 0.1, 0.1] vs truth [0.667, 0.466, 0.414, 0.664, 0.439, 0.341]
- age_band=35-54|country=UK|gender=woman: rho 0.88, top1 miss; pred [0.2, 0.0, 0.0, 0.2, 0.0, 0.0] vs truth [0.707, 0.615, 0.507, 0.713, 0.587, 0.41]
- age_band=55+|country=UK|gender=woman: rho 0.60, top1 miss; pred [0.4, 0.3, 0.2, 0.2, 0.1, 0.1] vs truth [0.657, 0.778, 0.663, 0.75, 0.637, 0.495]
- country=UK|education=no university degree: rho 0.88, top1 hit; pred [0.1, 0.0, 0.0, 0.1, 0.1, 0.0] vs truth [0.597, 0.539, 0.478, 0.617, 0.511, 0.355]
- country=UK|education=university degree: rho 0.79, top1 hit; pred [0.3, 0.2, 0.2, 0.2, 0.2, 0.2] vs truth [0.664, 0.602, 0.442, 0.649, 0.54, 0.491]

Best for claude-sonnet-5-5: grounded_choice (rho 0.791). Per segment:
- age_band=18-34|country=UK|gender=man: rho 0.88, top1 miss; pred [0.7, 0.2, 0.0, 0.3, 0.0, 0.0] vs truth [0.56, 0.416, 0.286, 0.615, 0.412, 0.314]
- age_band=35-54|country=UK|gender=man: rho 0.74, top1 miss; pred [0.7, 0.2, 0.1, 0.4, 0.2, 0.1] vs truth [0.517, 0.527, 0.352, 0.508, 0.474, 0.421]
- age_band=55+|country=UK|gender=man: rho 0.83, top1 hit; pred [0.8, 0.5, 0.2, 0.6, 0.2, 0.1] vs truth [0.613, 0.574, 0.507, 0.575, 0.579, 0.451]
- age_band=18-34|country=UK|gender=woman: rho 0.52, top1 hit; pred [0.8, 0.1, 0.2, 0.6, 0.1, 0.2] vs truth [0.667, 0.466, 0.414, 0.664, 0.439, 0.341]
- age_band=35-54|country=UK|gender=woman: rho 0.97, top1 hit; pred [0.8, 0.2, 0.2, 0.8, 0.2, 0.2] vs truth [0.707, 0.615, 0.507, 0.713, 0.587, 0.41]
- age_band=55+|country=UK|gender=woman: rho 0.60, top1 miss; pred [0.9, 0.4, 0.1, 0.9, 0.4, 0.0] vs truth [0.657, 0.778, 0.663, 0.75, 0.637, 0.495]
- country=UK|education=no university degree: rho 0.85, top1 hit; pred [0.2, 0.1, 0.1, 0.3, 0.1, 0.1] vs truth [0.597, 0.539, 0.478, 0.617, 0.511, 0.355]
- country=UK|education=university degree: rho 0.94, top1 hit; pred [0.9, 0.7, 0.4, 0.9, 0.6, 0.3] vs truth [0.664, 0.602, 0.442, 0.649, 0.54, 0.491]
