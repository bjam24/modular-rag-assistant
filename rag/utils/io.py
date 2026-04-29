"""
I/O utilities for loading RAG artifacts.

Loads the FAISS index and chunk metadata used during retrieval.
"""

import pickle
from pathlib import Path

import faiss

from rag.config import INDEX_PATH, CHUNKS_PATH


def load_index() -> faiss.Index:
    """
    Load the FAISS index from disk.
    """
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

    return faiss.read_index(str(INDEX_PATH))


def load_chunks() -> list[dict]:
    """
    Load chunk metadata from disk.
    """
    if not Path(CHUNKS_PATH).exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "rb") as file:
        return pickle.load(file)