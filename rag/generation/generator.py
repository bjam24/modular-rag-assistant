"""
Generation module.

Handles interaction with the configured LLM provider to generate
answers and summaries from retrieved context.
"""

from __future__ import annotations

import os
import requests

from openai import OpenAI

from rag.config import DEFAULT_PROVIDER, GEN_MODEL, OLLAMA_BASE_URL
from rag.generation.prompts import answer_prompt, summary_prompt


PRICING = {
    "gpt-4.1-mini": {
        "input": 0.40 / 1_000_000,
        "output": 1.60 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0

    pricing = PRICING[model]

    return (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
    )


def _generate_with_ollama(prompt: str, model: str) -> dict:
    """
    Generate text using Ollama HTTP API.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to connect to Ollama at {OLLAMA_BASE_URL}: {e}"
        ) from e

    data = response.json()

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")

    return {
        "answer": data["message"]["content"].strip(),
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": 0.0,
    }


def _generate_with_openai(prompt: str, model: str) -> dict:
    """
    Generate text using OpenAI API.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    usage = response.usage

    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    cost_usd = _calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


def _generate(
    prompt: str,
    model: str = GEN_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> dict:
    """
    Route generation to the selected provider.
    """
    if provider == "ollama":
        return _generate_with_ollama(prompt=prompt, model=model)

    if provider == "openai":
        return _generate_with_openai(prompt=prompt, model=model)

    raise ValueError(
        f"Unsupported provider='{provider}'. "
        "Supported providers: ['ollama', 'openai']"
    )


def generate_answer(
    query: str,
    context: str,
    history: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = GEN_MODEL,
) -> dict:
    """
    Generate a grounded answer from retrieved context.
    """
    prompt = answer_prompt(query, context, history)

    return _generate(
        prompt=prompt,
        model=model,
        provider=provider,
    )


def generate_summary(
    topic: str,
    context: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = GEN_MODEL,
) -> dict:
    """
    Generate a summary based on retrieved context.
    """
    prompt = summary_prompt(topic, context)

    return _generate(
        prompt=prompt,
        model=model,
        provider=provider,
    )