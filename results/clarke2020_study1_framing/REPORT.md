## clarke2020_study1_framing

Source: Clarke et al. 2020 protocol Study 1; data osf.io/k7tw5 (N=4,607 UK adults, 18 pictorial labels collapsed to 6 disease framings). Truth = 8 - desire to eat the chocolate bar (1-7) per segment per framing; secondary truth = negative emotional arousal.

Human reference points (Spearman, segment ranking of the 6 variants):
- gender+age_band: predict every segment with the pooled-population ranking = **0.443**; split-half human noise ceiling = **0.171**
- education: predict every segment with the pooled-population ranking = **0.743**; split-half human noise ceiling = **0.208**

| model | method | mean rho | top-1 acc | gap preservation | within-seg SD | valid answers | cost $ | secs |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | grounded_debate | 0.272 | 0.50 | 0.002 | 6.6 | 570/96 | 0.095 | 125.6 |
| claude-haiku-5-5 | ... debate round 1 only | 0.171 | 0.62 | -0.131 | | 552 | | |
| claude-haiku-5-5 | grounded_ensemble | 0.171 | 0.62 | -0.131 | 6.7 | 552/96 | 0.036 | 52.3 |
| claude-haiku-5-5 | grounded_single | 0.376 | 0.50 | 0.121 | 5.6 | 234/40 | 0.015 | 23.2 |
| claude-haiku-5-5 | judge | 0.176 | 0.75 | -0.108 | 2.2 | 234/40 | 0.017 | 28.4 |
| claude-haiku-5-5 | label_single | 0.258 | 0.75 | -0.005 | 3.8 | 234/40 | 0.015 | 25.1 |

Best for claude-haiku-5-5: grounded_single (rho 0.376). Per segment:
- age_band=18-34|country=UK|gender=man: rho 0.26, top1 hit; pred [28.1, 14.0, 18.0, 17.0, 16.6, 6.3] vs truth [4.52, 3.863, 4.029, 3.71, 4.25, 4.197]
- age_band=35-54|country=UK|gender=man: rho 0.43, top1 hit; pred [24.6, 15.4, 14.5, 16.2, 17.2, 12.0] vs truth [4.745, 4.442, 4.353, 4.527, 4.224, 4.242]
- age_band=55+|country=UK|gender=man: rho -0.03, top1 miss; pred [24.5, 11.6, 11.1, 30.0, 14.1, 8.7] vs truth [4.76, 4.737, 4.675, 4.895, 4.648, 5.0]
- age_band=18-34|country=UK|gender=woman: rho 0.31, top1 miss; pred [20.2, 13.0, 14.1, 12.9, 17.5, 22.3] vs truth [4.972, 4.583, 4.538, 4.779, 4.781, 4.596]
- age_band=35-54|country=UK|gender=woman: rho 0.43, top1 miss; pred [18.5, 15.4, 18.3, 15.7, 14.1, 18.0] vs truth [4.874, 4.779, 5.034, 5.088, 4.824, 5.0]
- age_band=55+|country=UK|gender=woman: rho 0.64, top1 hit; pred [31.7, 14.5, 17.2, 15.7, 14.0, 6.9] vs truth [5.562, 5.167, 5.258, 4.992, 5.167, 5.008]
- country=UK|education=no university degree: rho 0.77, top1 miss; pred [23.7, 12.8, 10.8, 26.8, 9.5, 16.5] vs truth [4.896, 4.594, 4.692, 4.822, 4.602, 4.715]
- country=UK|education=university degree: rho 0.20, top1 hit; pred [26.1, 15.1, 14.9, 15.0, 11.9, 17.0] vs truth [4.964, 4.64, 4.66, 4.643, 4.719, 4.694]
