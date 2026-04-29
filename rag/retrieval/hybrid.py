"""
Hybrid retrieval module.

Combines dense (FAISS) and sparse (TF-IDF) search by normalizing
their scores and fusing them with a weighted parameter.

Also provides a unified retrieval function that allows selecting:
- dense-only retrieval
- sparse-only retrieval
- hybrid retrieval
"""

import numpy as np

from rag.retrieval.dense import dense_search
from rag.retrieval.sparse import sparse_search


def normalize_scores(score_list: list[tuple[int, float]]) -> dict[int, float]:
    """
    Normalize scores to the range [0, 1] for fair comparison.
    """
    if not score_list:
        return {}

    values = np.array([score for _, score in score_list], dtype=float)
    min_v = values.min()
    max_v = values.max()

    # All scores equal → assign full score
    if max_v - min_v < 1e-12:
        return {idx: 1.0 for idx, _ in score_list}

    return {
        idx: (score - min_v) / (max_v - min_v)
        for idx, score in score_list
    }


def hybrid_search(query: str, index, chunks: list[dict],vectorizer, tfidf_matrix, top_k: int = 5, faiss_k: int = 20,
    tfidf_k: int = 20, alpha: float = 0.6) -> list[dict]:
    """
    Perform hybrid retrieval using dense and sparse search.

    Returns top-k results ranked by a weighted combination of both scores.
    """
    if not query.strip():
        return []

    # Ensure alpha is within [0, 1]
    alpha = min(max(alpha, 0.0), 1.0)

    # Dense (semantic) retrieval
    dense_results = dense_search(query, index, top_n=faiss_k)

    # Sparse (keyword) retrieval
    sparse_results = sparse_search(query, vectorizer, tfidf_matrix, top_n=tfidf_k)

    # Normalize scores
    dense_norm = normalize_scores(dense_results)
    sparse_norm = normalize_scores(sparse_results)

    # Merge candidate IDs
    candidate_ids = set(dense_norm.keys()) | set(sparse_norm.keys())

    candidates = []
    for idx in candidate_ids:
        vec_score = dense_norm.get(idx, 0.0)
        lex_score = sparse_norm.get(idx, 0.0)

        hybrid_score = alpha * vec_score + (1 - alpha) * lex_score

        candidates.append(
            {
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "vector_score": vec_score,
                "tfidf_score": lex_score,
                "hybrid_score": hybrid_score,
            }
        )

    # Sort by hybrid score
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return candidates[:top_k]


def retrieve_chunks(query: str, index, chunks: list[dict], vectorizer, tfidf_matrix, top_k: int = 5, faiss_k: int = 20, tfidf_k: int = 20,
    alpha: float = 0.6, mode: str = "hybrid") -> list[dict]:
    """
    Retrieve chunks using the selected retrieval mode.

    Supported modes:
    - dense: FAISS semantic search only
    - sparse: TF-IDF keyword search only
    - hybrid: combined FAISS + TF-IDF search
    """
    if not query.strip():
        return []

    if mode == "hybrid":
        return hybrid_search(
            query=query,
            index=index,
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_k=top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
        )

    if mode == "dense":
        dense_results = dense_search(query, index, top_n=top_k)

        return [
            {
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "vector_score": float(score),
                "tfidf_score": 0.0,
                "hybrid_score": float(score),
            }
            for idx, score in dense_results
        ]

    if mode == "sparse":
        sparse_results = sparse_search(
            query=query,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_n=top_k,
        )

        return [
            {
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "vector_score": 0.0,
                "tfidf_score": float(score),
                "hybrid_score": float(score),
            }
            for idx, score in sparse_results
        ]

    raise ValueError(
        f"Unknown retrieval mode: {mode}. "
        "Use one of: 'dense', 'sparse', 'hybrid'."
    )