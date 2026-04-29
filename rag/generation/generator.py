"""
Generation module.

Handles interaction with the configured LLM provider to generate
answers and summaries from retrieved context.
"""

from __future__ import annotations

import requests

from rag.config import DEFAULT_PROVIDER, GEN_MODEL, OLLAMA_BASE_URL
from rag.generation.prompts import answer_prompt, summary_prompt

print("LOADED NEW GENERATOR.PY")


def _generate_with_ollama(prompt: str, model: str = GEN_MODEL) -> str:
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

    return data["message"]["content"].strip()


def _generate(
    prompt: str,
    model: str = GEN_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    """
    Route generation to the configured provider.
    """
    if provider == "ollama":
        return _generate_with_ollama(prompt=prompt, model=model)

    raise ValueError(
        f"Unsupported provider='{provider}'. "
        "Currently supported providers: ['ollama']"
    )


def generate_answer(query: str, context: str, history: str) -> str:
    """
    Generate a grounded answer from retrieved context.
    """
    prompt = answer_prompt(query, context, history)
    return _generate(prompt)


def generate_summary(topic: str, context: str) -> str:
    """
    Generate a summary based on retrieved context.
    """
    prompt = summary_prompt(topic, context)
    return _generate(prompt)