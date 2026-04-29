"""
Retrieval evaluation script.

Measures whether the RAG retrieval pipeline returns expected source documents
for predefined test queries.

Supports comparison of retrieval modes:
- dense
- sparse
- hybrid
"""

from rag.orchestration.pipeline import ModularRAGPipeline
from rag.retrieval.sparse import build_tfidf_index
from rag.utils.io import load_chunks, load_index
from evaluation.test_cases import TEST_QUERIES


RETRIEVAL_MODES = ["dense", "sparse", "hybrid"]


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    expected_set = set(expected_sources)

    if not expected_set:
        return 0.0

    retrieved_top_k = set(retrieved_sources[:k])
    return len(retrieved_top_k & expected_set) / len(expected_set)


def evaluate(
    test_queries: list[dict],
    pipeline: ModularRAGPipeline,
    top_k: int = 5,
    faiss_k: int = 20,
    tfidf_k: int = 20,
    alpha: float = 0.6,
    retrieval_mode: str = "hybrid",
) -> dict:
    if not test_queries:
        raise ValueError("No test queries provided.")

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    mrr_total = 0.0
    recall_at_1_total = 0.0
    recall_at_3_total = 0.0
    recall_at_5_total = 0.0

    detailed_results = []

    for item in test_queries:
        query = item["query"]
        expected_sources = item["expected_sources"]

        results = pipeline.retrieve(
            query=query,
            top_k=top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            retrieval_mode=retrieval_mode,
        )

        retrieved_sources = [result["source"] for result in results]

        top1_hit = any(src in expected_sources for src in retrieved_sources[:1])
        top3_hit = any(src in expected_sources for src in retrieved_sources[:3])
        top5_hit = any(src in expected_sources for src in retrieved_sources[:5])

        rr = reciprocal_rank(retrieved_sources, expected_sources)
        r_at_1 = recall_at_k(retrieved_sources, expected_sources, 1)
        r_at_3 = recall_at_k(retrieved_sources, expected_sources, 3)
        r_at_5 = recall_at_k(retrieved_sources, expected_sources, 5)

        top1_correct += int(top1_hit)
        top3_correct += int(top3_hit)
        top5_correct += int(top5_hit)

        mrr_total += rr
        recall_at_1_total += r_at_1
        recall_at_3_total += r_at_3
        recall_at_5_total += r_at_5

        detailed_results.append(
            {
                "query": query,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top5_hit": top5_hit,
                "reciprocal_rank": rr,
                "recall_at_1": r_at_1,
                "recall_at_3": r_at_3,
                "recall_at_5": r_at_5,
            }
        )

    num_queries = len(test_queries)

    return {
        "retrieval_mode": retrieval_mode,
        "top1_accuracy": top1_correct / num_queries,
        "top3_hit_rate": top3_correct / num_queries,
        "top5_hit_rate": top5_correct / num_queries,
        "mrr": mrr_total / num_queries,
        "recall_at_1": recall_at_1_total / num_queries,
        "recall_at_3": recall_at_3_total / num_queries,
        "recall_at_5": recall_at_5_total / num_queries,
        "num_queries": num_queries,
        "details": detailed_results,
    }


def build_pipeline() -> ModularRAGPipeline:
    index = load_index()
    chunks = load_chunks()
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)

    return ModularRAGPipeline(
        index=index,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


def summarize_metrics(metrics: dict) -> dict:
    return {
        "mode": metrics["retrieval_mode"],
        "top1_accuracy": round(metrics["top1_accuracy"], 4),
        "top3_hit_rate": round(metrics["top3_hit_rate"], 4),
        "top5_hit_rate": round(metrics["top5_hit_rate"], 4),
        "mrr": round(metrics["mrr"], 4),
        "recall_at_5": round(metrics["recall_at_5"], 4),
        "num_queries": metrics["num_queries"],
    }


def compare_retrieval_modes(
    test_queries: list[dict] = TEST_QUERIES,
    top_k: int = 5,
    faiss_k: int = 20,
    tfidf_k: int = 20,
    alpha: float = 0.6,
) -> list[dict]:
    pipeline = build_pipeline()
    comparison = []

    for mode in RETRIEVAL_MODES:
        metrics = evaluate(
            test_queries=test_queries,
            pipeline=pipeline,
            top_k=top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            retrieval_mode=mode,
        )
        comparison.append(summarize_metrics(metrics))

    return comparison


def build_sample_qa_table(
    test_queries: list[dict],
    pipeline: ModularRAGPipeline,
    top_k: int = 3,
    retrieval_mode: str = "hybrid",
    max_samples: int = 10,
) -> list[dict]:
    """
    Build qualitative sample table with query, expected source, retrieved sources and generated answer.
    """
    rows = []

    for item in test_queries[:max_samples]:
        query = item["query"]
        expected_sources = item["expected_sources"]

        output = pipeline.run_chat(
            query=query,
            history="",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )

        results = output["results"]
        retrieved_sources = [result["source"] for result in results]

        rows.append(
            {
                "query": query,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "answer": output["answer"],
            }
        )

    return rows


def print_metrics(metrics: dict) -> None:
    print("=" * 80)
    print("METRICS")
    print("=" * 80)
    print(f"Mode: {metrics['retrieval_mode']}")
    print(f"Queries: {metrics['num_queries']}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-3 Hit Rate: {metrics['top3_hit_rate']:.2%}")
    print(f"Top-5 Hit Rate: {metrics['top5_hit_rate']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Recall@1: {metrics['recall_at_1']:.2%}")
    print(f"Recall@3: {metrics['recall_at_3']:.2%}")
    print(f"Recall@5: {metrics['recall_at_5']:.2%}")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for item in metrics["details"]:
        print(f"Query: {item['query']}")
        print(f"Expected: {item['expected_sources']}")
        print(f"Retrieved: {item['retrieved_sources']}")
        print(
            f"Top1: {item['top1_hit']} | "
            f"Top3: {item['top3_hit']} | "
            f"Top5: {item['top5_hit']}"
        )
        print(
            f"RR: {item['reciprocal_rank']:.4f} | "
            f"R@1: {item['recall_at_1']:.2%} | "
            f"R@3: {item['recall_at_3']:.2%} | "
            f"R@5: {item['recall_at_5']:.2%}"
        )
        print("-" * 80)


def print_comparison_table(comparison: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("RETRIEVAL MODE COMPARISON")
    print("=" * 80)
    print(
        f"{'Mode':<10} "
        f"{'Top-1':>10} "
        f"{'Top-3':>10} "
        f"{'Top-5':>10} "
        f"{'MRR':>10} "
        f"{'Recall@5':>10}"
    )
    print("-" * 80)

    for row in comparison:
        print(
            f"{row['mode']:<10} "
            f"{row['top1_accuracy']:>10.4f} "
            f"{row['top3_hit_rate']:>10.4f} "
            f"{row['top5_hit_rate']:>10.4f} "
            f"{row['mrr']:>10.4f} "
            f"{row['recall_at_5']:>10.4f}"
        )


def print_sample_qa_table(rows: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("SAMPLE QA TABLE")
    print("=" * 80)

    for i, row in enumerate(rows, start=1):
        print(f"\n[{i}] Query:")
        print(row["query"])

        print("Expected sources:")
        print(row["expected_sources"])

        print("Retrieved sources:")
        print(row["retrieved_sources"])

        print("Answer:")
        print(row["answer"])

        print("-" * 80)


if __name__ == "__main__":
    pipeline = build_pipeline()
    all_results = []

    for mode in RETRIEVAL_MODES:
        print("\n" + "=" * 80)
        print(f"EVALUATION MODE: {mode.upper()}")
        print("=" * 80)

        metrics = evaluate(
            test_queries=TEST_QUERIES,
            pipeline=pipeline,
            retrieval_mode=mode,
        )

        all_results.append(summarize_metrics(metrics))
        print_metrics(metrics)

    print_comparison_table(all_results)

    sample_rows = build_sample_qa_table(
        test_queries=TEST_QUERIES,
        pipeline=pipeline,
        top_k=3,
        retrieval_mode="hybrid",
        max_samples=10,
    )
    print_sample_qa_table(sample_rows)