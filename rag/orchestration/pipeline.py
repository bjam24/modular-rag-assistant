"""
Core orchestration module for the Modular RAG system.

Defines the end-to-end pipeline that connects:
- query transformation
- hybrid retrieval (FAISS + TF-IDF)
- reranking and filtering
- grounded generation
- cost and latency tracking
"""

from __future__ import annotations

import time

from rag.config import (
    DEFAULT_ALPHA,
    DEFAULT_FAISS_K,
    DEFAULT_TFIDF_K,
    DEFAULT_TOP_K,
    GENERATION_MODES,
)
from rag.pre_retrieval.query_transform import rewrite_query, build_vocabulary
from rag.retrieval.hybrid import retrieve_chunks
from rag.post_retrieval.reranker import rerank_results
from rag.post_retrieval.filters import is_context_sufficient, build_grounded_context
from rag.generation.generator import generate_answer, generate_summary
from rag.observability.cost import estimate_cost_usd, estimate_tokens
from rag.observability.logger import log_query


FALLBACK_ANSWER = "I could not find the answer in the documents."
FALLBACK_SUMMARY = "I could not find enough information in the documents."


class ModularRAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(self, index, chunks: list[dict], vectorizer, tfidf_matrix) -> None:
        """
        Initialize the pipeline with retrieval resources.
        """
        self.index = index
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.vocab = build_vocabulary(chunks)

    def _rewrite_query(self, query: str) -> tuple[str, str | None]:
        """
        Rewrite query and return the corrected version if it changed.
        """
        original_query, rewritten_query = rewrite_query(query, self.vocab)
        corrected_query = rewritten_query if rewritten_query != original_query else None

        return rewritten_query, corrected_query

    def _get_generation_settings(self, generation_mode: str) -> dict:
        """
        Return generation settings for selected cost-aware mode.
        """
        if generation_mode not in GENERATION_MODES:
            raise ValueError(
                f"Unknown generation_mode='{generation_mode}'. "
                f"Available modes: {list(GENERATION_MODES.keys())}"
            )

        return GENERATION_MODES[generation_mode]

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        faiss_k: int = DEFAULT_FAISS_K,
        tfidf_k: int = DEFAULT_TFIDF_K,
        alpha: float = DEFAULT_ALPHA,
        retrieval_mode: str = "hybrid",
    ) -> list[dict]:
        """
        Retrieve and rerank relevant document chunks using selected retrieval mode.
        """
        rewritten_query, _ = self._rewrite_query(query)

        retrieved = retrieve_chunks(
            query=rewritten_query,
            index=self.index,
            chunks=self.chunks,
            vectorizer=self.vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            top_k=max(top_k, faiss_k, tfidf_k),
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            mode=retrieval_mode,
        )

        reranked = rerank_results(retrieved, rewritten_query)

        return reranked[:top_k]

    def run_chat(
        self,
        query: str,
        history: str,
        top_k: int | None = None,
        faiss_k: int = DEFAULT_FAISS_K,
        tfidf_k: int = DEFAULT_TFIDF_K,
        alpha: float = DEFAULT_ALPHA,
        retrieval_mode: str = "hybrid",
        generation_mode: str = "balanced",
    ) -> dict:
        """
        Run the RAG pipeline for question answering.
        """
        start_time = time.time()

        generation_settings = self._get_generation_settings(generation_mode)

        selected_top_k = top_k or generation_settings["top_k"]
        max_context_chars = generation_settings["max_context_chars"]
        model = generation_settings["model"]

        rewritten_query, corrected_query = self._rewrite_query(query)

        results = self.retrieve(
            query=rewritten_query,
            top_k=selected_top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            retrieval_mode=retrieval_mode,
        )

        if not results:
            latency_sec = round(time.time() - start_time, 3)

            log_query(
                {
                    "query": query,
                    "retrieval_mode": retrieval_mode,
                    "generation_mode": generation_mode,
                    "model": model,
                    "top_k": selected_top_k,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": latency_sec,
                    "num_sources": 0,
                    "fallback": True,
                }
            )

            return {
                "answer": FALLBACK_ANSWER,
                "results": [],
                "sources": [],
                "corrected_query": corrected_query,
                "rewritten_query": rewritten_query,
                "retrieval_mode": retrieval_mode,
                "generation_mode": generation_mode,
                "latency": latency_sec,
                "tokens": {
                    "input": 0,
                    "output": 0,
                },
                "cost_usd": 0.0,
            }

        # Optional strict filtering can be re-enabled later.
        # if not is_context_sufficient(results):
        #     return {"answer": FALLBACK_ANSWER, "results": results}

        context = build_grounded_context(results)
        context = context[:max_context_chars]

        answer = generate_answer(rewritten_query, context, history)

        input_tokens = estimate_tokens(
            rewritten_query + "\n" + context + "\n" + (history or "")
        )
        output_tokens = estimate_tokens(answer)

        cost_usd = estimate_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        latency_sec = round(time.time() - start_time, 3)

        sources = [
            {
                "source": result.get("source"),
                "score": result.get("score"),
                "text": result.get("text"),
            }
            for result in results
        ]

        log_query(
            {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "generation_mode": generation_mode,
                "model": model,
                "top_k": selected_top_k,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "latency_sec": latency_sec,
                "num_sources": len(results),
                "fallback": False,
            }
        )

        return {
            "answer": answer,
            "results": results,
            "sources": sources,
            "corrected_query": corrected_query,
            "rewritten_query": rewritten_query,
            "retrieval_mode": retrieval_mode,
            "generation_mode": generation_mode,
            "latency": latency_sec,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
            },
            "cost_usd": cost_usd,
        }

    def run_summary(
        self,
        topic: str,
        top_k: int = DEFAULT_TOP_K,
        faiss_k: int = DEFAULT_FAISS_K,
        tfidf_k: int = DEFAULT_TFIDF_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> dict:
        """
        Run the RAG pipeline for topic summarization.
        """
        results = self.retrieve(
            query=topic,
            top_k=top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
        )

        if not results:
            return {"summary": FALLBACK_SUMMARY, "results": []}

        if not is_context_sufficient(results):
            return {"summary": FALLBACK_SUMMARY, "results": results}

        context = build_grounded_context(results)
        summary = generate_summary(topic, context)

        return {"summary": summary, "results": results}