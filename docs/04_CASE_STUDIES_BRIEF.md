# Research brief: case studies with public human data for testing LLM-persona prediction of food-messaging response

Compiled 2026-10-21 by a web-research agent. OPEN = individual-level data downloadable now; ON REQUEST = author/controlled access; TABLES = published subgroup tables only. UNVERIFIED marked.

## Candidates
1. **Turnwald et al. 2019** taste-focused vs health-focused vs basic vegetable labels (US). Psych Sci 30(11). OSF https://osf.io/39vsg/ (verified files). 137,842 diner decisions, 5 campuses. Outcome: selection, consumption. Demographics site-level only. OPEN. Great for variant ranking, weak for segmentation.
2. **Clarke, Pechey, Marteau et al. 2020** health warning labels on snacks (UK). Appetite; PMC7450271. OSF https://osf.io/k7tw5/ (verified: dataset xlsx, protocol, analysis plan, stimuli). 3 x 2 = 6 arms, N = 4,134 UK adults. Outcomes: energy-dense snack selection; arousal, reactance, avoidance, risk, acceptability. Individual-level age, gender, ethnicity, education, BMI, income. No demographic moderation published. OPEN. **Top UK candidate.** (Study 1 in the same OSF project: 4,607 adults, 18 pictorial labels across 6 disease framings, outcomes arousal / acceptability / desire.)
3. **Hall, Grummon, Taillie et al. 2022** pictorial warnings on sugary drinks, naturalistic store (US). PLOS Med 19(2). S1 Data + Stata. N = 325 parents, diverse; real purchase outcome (45% vs 28%). OPEN. Clean test of predicting a null heterogeneity.
4. **Grummon, Reimold, Hall 2022** San Francisco SSB warning equity analysis. J Acad Nutr Diet; PMC9127741. N = 2,160 parents; Table 4 subgroup effects by age, gender, race, education, income, language. TABLES / ON REQUEST (UNVERIFIED).
5. **Hall, Lazard, Grummon et al. 2021** warning topic x design, Latino vs non-Latino parents. Prev Med 148; PMC8958622. 4 topics x 3 designs, N = 1,078. Icon-vs-text effect larger for Latino and low-English-use parents. TABLES / ON REQUEST. Ideal structure; email authors.
6. **Wolfson et al. 2022** climate-impact menu labels (US). JAMA Netw Open. N = 5,049 representative. Only sex moderated (women 68.4% vs men 53.7% sustainable choice under red label). Data sharing in Supplement 3, UNVERIFIED.
7. **Potter et al. 2024** ecolabels in online supermarket (UK). PLOS ONE; N = 1,051; OSF rn5za returned 404 (UNVERIFIED). Thin demographics.
8. **Mantzari et al. 2018** SSB warning labels, UK parents. Prev Med Reports 12; N = 2,002; 6 variants. ON REQUEST (UNVERIFIED).
9. **Hughes, Weick, Vasiljevic 2023** pictorial warnings on meat meals (UK). Appetite 190; N = 1,001. Data status UNVERIFIED.
10. **FSA Food and You 2** Waves 1-7, UK Data Service SN 8814. No message testing; use as UK population prior for persona construction.
11. **FSA Consumer Insights Tracker** monthly tables on data.food.gov.uk. Calibration only.
12. **FSA / Ipsos rapid communications testing (2023)**: 43 FSA communications tested; report URLs now HTTP 410. Most on-target UK food-safety message test but withdrawn; request from FSA social science team.
13. **Veg Power "Eat Them to Defeat Them" 2022** evaluation: 3,000 parents/children; subgroup tables by child gender, age, income; no control group. TABLES. Secondary benchmark.
14. **Change4Life cluster RCT** (Croker 2012): single campaign; low suitability.
15. **Food Standards Scotland** evaluations: tiny N, no variants. Radu, Erdem, Campbell 2026 responsibility prompts N = 2,343, data UNVERIFIED.
16. **OCEANS-III** (Freeman 2021, UK): 10 vaccine-information conditions, ~15,000 adults; personal-benefit beat collective-benefit among hesitant. Data UNVERIFIED.
17. **Milkman 2024 booster megastudy**: OSF participant-level condition and outcome, no demographics.

## Top 5
1. Clarke 2020 Study 2 (UK, 6 arms, individual demographics, downloadable). 2. Hall 2022 PLOS Med. 3. Wolfson 2022 (promote if data confirmed). 4. Hall 2021 Prev Med (author request). 5. Turnwald 2019.

## Published baselines to beat
- Hewitt, Ashokkumar, Ghezae, Willer: GPT-4 simulated samples predict treatment effects across 70 survey experiments at r = 0.85; demo at treatmeneffect.app.
- SocSci210 / Socrates (Kolluri, Wu, Park, Bernstein 2025; HF socratesft/SocSci210): fine-tuned Qwen-14B 26% better aligned with human response distributions than base, 13% better than GPT-4o. Natural distributional baseline.
- Twin-2K-500 and Mega-Study (HF LLM-Digital-Twin): GPT-4.1-mini ~71.7% individual accuracy.
- Park 2024 generative agents (restricted API).
- In-paper econometric baselines: logistic regression with arm x demographic interactions. A causal-forest uplift baseline on the open microdata is unpublished and would be a fair comparator.

## Practical notes
OSF pages are JS-rendered; use api.osf.io/v2/nodes/<id>/files/osfstorage/. PMC intermittently captchas. FSA comms-testing pages are gone (410).
