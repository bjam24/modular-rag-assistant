"""
Embedding module.

Generates normalized embeddings using a local model (Ollama)
for both indexing and query-time retrieval.
"""

from __future__ import annotations

import faiss
import numpy as np
import requests

from rag.config import EMBED_MODEL, OLLAMA_BASE_URL


def _embed_with_ollama(text: str) -> list[float]:
    """
    Generate embedding using Ollama HTTP API.
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"

    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to connect to Ollama embeddings at {OLLAMA_BASE_URL}: {e}"
        ) from e

    data = response.json()

    if "embedding" not in data:
        raise RuntimeError(f"Unexpected Ollama embedding response format: {data}")

    return data["embedding"]


def embed_text(text: str) -> np.ndarray:
    """
    Generate a normalized embedding for a single text.
    """
    if not text.strip():
        return np.zeros((1, 1), dtype="float32")

    vector = _embed_with_ollama(text)

    vec = np.array([vector], dtype="float32")
    faiss.normalize_L2(vec)

    return vec


def build_embeddings(chunks: list[dict]) -> np.ndarray:
    """
    Generate normalized embeddings for document chunks.
    """
    vectors = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        vector = _embed_with_ollama(text)
        vectors.append(vector)

    if not vectors:
        return np.empty((0, 0), dtype="float32")

    embeddings = np.array(vectors, dtype="float32")
    faiss.normalize_L2(embeddings)

    return embeddings