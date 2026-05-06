from fastapi import FastAPI, HTTPException
from api.schemas import AskRequest, AskResponse
from rag.orchestration.pipeline import ModularRAGPipeline
from rag.retrieval.sparse import build_tfidf_index
from rag.utils.io import load_chunks, load_index


app = FastAPI(
    title="Modular RAG Assistant API",
    version="1.0.0",
)


pipeline: ModularRAGPipeline | None = None


def load_pipeline() -> ModularRAGPipeline:
    """Build RAG pipeline."""
    index = load_index()
    chunks = load_chunks()
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)

    return ModularRAGPipeline(
        index=index,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


@app.on_event("startup")
def startup() -> None:
    """Load app resources."""
    global pipeline
    pipeline = load_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    """Check API status."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    """Answer a RAG question."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    try:
        result = pipeline.run_chat(
            query=request.query,
            history=request.history,
            retrieval_mode=request.retrieval_mode,
            generation_mode=request.generation_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": result["answer"],
        "sources": result.get("sources", result.get("results", [])),
        "latency": result.get("latency", 0.0),
        "tokens": result.get("tokens", {"input": 0, "output": 0}),
        "cost_usd": result.get("cost_usd", 0.0),
    }