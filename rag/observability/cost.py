from __future__ import annotations

from rag.config import MODEL_PRICING_USD_PER_1M_TOKENS


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate LLM API cost in USD.

    Local models should have pricing set to 0.0 in config.
    """
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model)

    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return round(input_cost + output_cost, 6)


def estimate_tokens(text: str) -> int:
    """
    Lightweight token estimate.

    This avoids provider-specific tokenizers and works for both
    local Ollama models and future OpenAI integration.

    Approximation:
    1 token ≈ 4 characters
    """
    if not text:
        return 0

    return max(1, len(text) // 4)