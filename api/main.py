from fastapi import FastAPI, HTTPException

from api.schemas import AskRequest, AskResponse
from rag.orchestration.pipeline import ModularRAGPipeline
from rag.utils.io import load_chunks, load_index
from rag.retrieval.sparse import build_tfidf_index


app = FastAPI(
    title="Modular RAG Assistant API",
    version="1.0.0",
)


index = load_index()
chunks = load_chunks()
vectorizer, tfidf_matrix = build_tfidf_index(chunks)

pipeline = ModularRAGPipeline(
    index=index,
    chunks=chunks,
    vectorizer=vectorizer,
    tfidf_matrix=tfidf_matrix,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = pipeline.run_chat(
            query=request.query,
            history=request.history,
            retrieval_mode=request.retrieval_mode,
            generation_mode=request.generation_mode,
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "latency": result["latency"],
            "tokens": result["tokens"],
            "cost_usd": result["cost_usd"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))