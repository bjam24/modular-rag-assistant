"""
Pre-retrieval query processing module.

Normalizes and optionally corrects user queries before retrieval.
"""

import re
from difflib import get_close_matches


def build_vocabulary(chunks: list[dict]) -> set[str]:
    """
    Build vocabulary from indexed document chunks.
    """
    vocab = set()

    for chunk in chunks:
        words = re.findall(r"\w+", chunk["text"].lower())
        vocab.update(words)

    return vocab


def correct_word(word: str, vocab: set[str], cutoff: float = 0.8) -> str:
    """
    Correct a single word using fuzzy matching against the document vocabulary.
    """
    if word in vocab:
        return word

    matches = get_close_matches(word, vocab, n=1, cutoff=cutoff)
    return matches[0] if matches else word


def correct_query(query: str, vocab: set[str]) -> str:
    """
    Correct query typos using the document vocabulary.
    """
    words = re.findall(r"\w+", query.lower())
    corrected_words = [correct_word(word, vocab) for word in words]

    return " ".join(corrected_words)


def rewrite_query(query: str, vocab: set[str] | None = None) -> tuple[str, str]:
    """
    Returns (original_query, corrected_query)
    """
    original = query.strip()
    corrected = original

    if not original:
        return original, corrected

    if vocab:
        corrected = correct_query(original, vocab)

    return original, corrected


def expand_query(query: str) -> list[str]:
    """
    Return query variants for retrieval.
    """
    query = query.strip()
    return [query] if query else []