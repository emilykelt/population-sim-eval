"""Model access with cost tracking.

Two routes:
  * Anthropic models (ids starting with "claude-") go direct through the Anthropic SDK.
  * Anything else goes through Vercel AI Gateway's OpenAI-compatible endpoint
    (needs AI_GATEWAY_API_KEY and a card on the Vercel account).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# USD per million tokens (input, output). Anthropic first-party prices Oct 2026; gateway prices from its /v1/models.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-opus-5-5": (4.00, 20.00),
    # OpenAI direct (ids starting with "gpt-" or "o"); prices per million tokens, Oct 2026 list
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini/gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "together/Qwen/Qwen3-32B": (0.80, 0.80),
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "openai/gpt-5-nano": (0.05, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "alibaba/qwen-3-14b": (0.12, 0.24),
    "alibaba/qwen-3-32b": (0.15, 0.60),
    "meta/llama-3.1-8b": (0.22, 0.22),
    "meta/llama-4-scout": (0.17, 0.66),
    "mistral/mistral-small": (0.15, 0.60),
    "openai/gpt-oss-120b": (0.10, 0.50),
    "google/gemma-4-31b-it": (0.14, 0.40),
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "raw" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, inp: int, out: int) -> None:
        pi, po = PRICES.get(model, (0.0, 0.0))
        cost = (inp * pi + out * po) / 1e6
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cost_usd += cost
            m = self.by_model.setdefault(model, {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
            m["calls"] += 1
            m["input"] += inp
            m["output"] += out
            m["cost"] += cost


USAGE = Usage()

_anthropic_client = None


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(max_retries=4, timeout=120)
    return _anthropic_client


def _cache_key(model: str, system: str, user: str, temperature: float, seed: int, max_tokens: int) -> Path:
    h = hashlib.sha256(json.dumps([model, system, user, temperature, seed, max_tokens]).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


def complete(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 1.0,
    max_tokens: int = 600,
    seed: int = 0,
    use_cache: bool = True,
) -> str:
    """Single-turn completion. `seed` only differentiates cache entries for repeated samples."""
    path = _cache_key(model, system, user, temperature, seed, max_tokens)
    if use_cache and path.exists():
        return json.loads(path.read_text())["text"]

    for attempt in range(5):
        try:
            if model.startswith("claude-"):
                text, inp, out = _call_anthropic(model, system, user, temperature, max_tokens)
            elif model.startswith(("gpt-", "o1", "o3", "o4")):
                text, inp, out = _call_openai(model, system, user, temperature, max_tokens)
            elif model.split("/")[0] in DIRECT_PROVIDERS:
                text, inp, out = _call_compatible(model, system, user, temperature, max_tokens)
            else:
                text, inp, out = _call_gateway(model, system, user, temperature, max_tokens)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    USAGE.add(model, inp, out)
    path.write_text(json.dumps({"model": model, "text": text, "input_tokens": inp, "output_tokens": out}))
    return text


def _call_anthropic(model, system, user, temperature, max_tokens):
    client = _anthropic()
    kwargs: dict = dict(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
    # SDK 1.x removed `temperature` from the typed signature; 5.x models only accept the default (1.0) anyway.
    if model.startswith(("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5")):
        kwargs["extra_body"] = {"temperature": temperature}
    else:
        # 5.x models always think; thinking tokens count against max_tokens, so leave headroom.
        kwargs["output_config"] = {"effort": "low"}
        kwargs["max_tokens"] = max_tokens + 2500
    resp = client.messages.create(**kwargs)
    if resp.stop_reason == "max_tokens" and not any(getattr(b, "type", "") == "text" for b in resp.content):
        kwargs["max_tokens"] = kwargs["max_tokens"] + 4000
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        return "", resp.usage.input_tokens, resp.usage.output_tokens
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_gateway(model, system, user, temperature, max_tokens):
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        raise RuntimeError("AI_GATEWAY_API_KEY not set")
    r = requests.post(
        "https://ai-gateway.vercel.sh/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _call_openai(model, system, user, temperature, max_tokens):
    """Direct OpenAI chat completions. GPT-5 family ignores temperature and needs max_completion_tokens;
    reasoning models spend tokens on reasoning first, so give headroom and set reasoning effort low."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4")):
        body["max_completion_tokens"] = max_tokens + 2000
        body["reasoning_effort"] = "low"
    else:
        body["max_tokens"] = max_tokens
        body["temperature"] = temperature
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=120)
    if r.status_code == 400 and "reasoning_effort" in r.text:
        body.pop("reasoning_effort", None)
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=120)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


# Direct, OpenAI-compatible endpoints. Model id = "<prefix>/<provider model name>", e.g.
#   gemini/gemini-2.5-flash-lite   deepseek/deepseek-chat   together/Qwen/Qwen3-32B   groq/llama-3.3-70b-versatile
#   fireworks/accounts/fireworks/models/qwen3-32b   ollama/qwen3:32b (local, no key)
DIRECT_PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GOOGLE_API_KEY"),
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY"),
    "together": ("https://api.together.xyz/v1/chat/completions", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "dashscope": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", "DASHSCOPE_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
    "ollama": ("http://localhost:11434/v1/chat/completions", None),
    "local": ("http://localhost:8080/v1/chat/completions", None),  # mlx_lm.server / llama-server / LM Studio on :8080
}


def _call_compatible(model, system, user, temperature, max_tokens):
    prefix, name = model.split("/", 1)
    url, env = DIRECT_PROVIDERS[prefix]
    key = os.environ.get(env) if env else "ollama"
    if not key:
        raise RuntimeError(f"{env} not set in .env (needed for {model})")
    body = {"model": name, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if prefix == "gemini":
        body["reasoning_effort"] = "low"  # keeps Gemini 2.5 thinking from eating the budget; ignored by non-thinking models
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=180)
    if r.status_code == 400 and "reasoning_effort" in r.text:
        body.pop("reasoning_effort", None)
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=180)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {}) or {}
    return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
