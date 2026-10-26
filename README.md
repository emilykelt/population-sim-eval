# Simulated societies: food-messaging audience simulator (validation prototype)

Predicts how UK population segments respond to candidate public food / nutrition messages, and benchmarks that prediction against open human RCT data before any deployment.

Read in order: `docs/00_PLAN.md` -> `docs/05_RESULTS.md`. Literature: `docs/01_SYNTHESIS.md` (vault papers), `docs/02` (debate vs crowd), `docs/03` (cheap-model fidelity), `docs/04` (case studies with open data).

## Run
```bash
uv sync
PYTHONPATH=src uv run python src/make_cases.py                       # builds data/*.yaml from the Clarke 2020 OSF files, prints baselines + ceilings
PYTHONPATH=src uv run python src/run_benchmark.py data/clarke2020_study2_labels.yaml \
    --models claude-sonnet-5-5 --n 12 --methods judge grounded_choice
PYTHONPATH=src uv run python src/report.py data/clarke2020_study2_labels.yaml   # -> results/<case>/REPORT.md
```
Methods: `judge`, `label_single`, `grounded_single`, `grounded_ensemble`, `grounded_debate`, `grounded_choice`, `label_choice`, `grounded_choice_neutral`.
Models: any `claude-*` id (direct), or a Vercel AI Gateway id such as `google/gemini-2.5-flash-lite` once a card is on the Vercel team.

`.env` needs `ANTHROPIC_API_KEY`; optional `AI_GATEWAY_API_KEY`. All model calls are cached under `results/raw/cache/` keyed on model + prompt + seed, so reruns are free.

### Other providers, direct (no Vercel needed)
Put the key in `.env` and use the prefixed id: `gemini/gemini-2.5-flash-lite` (GOOGLE_API_KEY), `deepseek/deepseek-chat` (DEEPSEEK_API_KEY), `together/Qwen/Qwen3-32B` (TOGETHER_API_KEY), `groq/llama-3.3-70b-versatile` (GROQ_API_KEY), `mistral/mistral-small-latest` (MISTRAL_API_KEY), `ollama/<local model>` (no key; needs Ollama running).
