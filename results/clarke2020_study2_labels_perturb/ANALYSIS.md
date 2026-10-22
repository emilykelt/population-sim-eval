# Prompt-perturbation analysis: clarke2020_study2_labels

25 wordings per model x design (5x5 Graeco-Latin square over system / task / arm-description / format).

## Score distribution over wordings

| model | design | n | rho median [IQR] | rho min..max | top-1 median | pooled rho median [IQR] | base rate median [IQR] | $ total |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | choice | 25 | 0.16 [0.05, 0.36] | -0.11..0.66 | 0.38 | 0.46 [0.09, 0.77] | 0.07 [0.04, 0.23] | 1.79 |
| claude-haiku-5-5 | poll | 25 | 0.02 [-0.10, 0.11] | -0.49..0.21 | 0.00 | 0.14 [-0.14, 0.20] | n.a. | 0.66 |
| claude-sonnet-5-5 | choice | 10 | 0.70 [0.58, 0.75] | 0.41..0.80 | 0.38 | 0.94 [0.87, 0.94] | 0.48 [0.35, 0.56] | 8.27 |

## Variance decomposition (share of sum of squares, type II ANOVA on the 25 wordings)

| model | design | outcome | system | task | arms | format | residual |
|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | choice | rho | 0.39 | 0.10 | 0.14 | 0.13 | 0.24 |
| claude-haiku-5-5 | choice | pooled_rho | 0.47 | 0.19 | 0.16 | 0.04 | 0.14 |
| claude-haiku-5-5 | choice | base_rate | 0.03 | 0.90 | 0.03 | 0.01 | 0.03 |
| claude-haiku-5-5 | poll | rho | 0.11 | 0.57 | 0.07 | 0.04 | 0.20 |
| claude-haiku-5-5 | poll | pooled_rho | 0.09 | 0.49 | 0.12 | 0.11 | 0.18 |
| claude-sonnet-5-5 | choice | rho | 0.09 | 0.39 | 0.13 | 0.22 | 0.17 |
| claude-sonnet-5-5 | choice | pooled_rho | 0.04 | 0.32 | 0.07 | 0.36 | 0.20 |
| claude-sonnet-5-5 | choice | base_rate | 0.44 | 0.44 | 0.05 | 0.07 | 0.00 |

## Wording variance vs model variance

- choice: mean within-model (wording) variance of rho = 0.0306; between-model variance of mean rho = 0.1010; ratio wording/model = 0.30
- poll: only one model run so far; wording SD of rho = 0.179

## Sign-flip rate of the top-1 pick per segment

Fraction of wordings whose top-1 pick for a segment differs from the modal pick across wordings (0 = perfectly stable).

| model | design | mean flip rate | worst segment (rate) | modal picks |
|---|---|---|---|---|
| claude-haiku-5-5 | choice | 0.35 | age_band=35-54|country=UK|gender=man (0.56) | {'18-34/man': 'pictorial_plus_calories', '35-54/man': 'pictorial_plus_calories', '55+/man': 'pictorial_plus_calories', '18-34/woman': 'pictorial_plus_calories', '35-54/woman': 'pictorial_plus_calories', '55+/woman': 'pictorial_plus_calories', 'UK/no university degree': 'pictorial_plus_calories', 'UK/university degree': 'pictorial_plus_calories'} |
| claude-haiku-5-5 | poll | 0.07 | age_band=35-54|country=UK|gender=man (0.20) | {'18-34/man': 'calories_only', '35-54/man': 'calories_only', '55+/man': 'calories_only', '18-34/woman': 'calories_only', '35-54/woman': 'calories_only', '55+/woman': 'calories_only', 'UK/no university degree': 'calories_only', 'UK/university degree': 'calories_only'} |
| claude-sonnet-5-5 | choice | 0.16 | age_band=18-34|country=UK|gender=man (0.30) | {'18-34/man': 'pictorial_plus_calories', '35-54/man': 'pictorial_plus_calories', '55+/man': 'pictorial_plus_calories', '18-34/woman': 'pictorial_plus_calories', '35-54/woman': 'pictorial_plus_calories', '55+/woman': 'pictorial_plus_calories', 'UK/no university degree': 'pictorial_plus_calories', 'UK/university degree': 'pictorial_plus_calories'} |
