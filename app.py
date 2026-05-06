"""Streamlit UI for the Modular RAG Assistant."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from rag.config import DATA_DIR
from rag.indexing.builder import rebuild_knowledge_base
from rag.orchestration.pipeline import ModularRAGPipeline
from rag.retrieval.sparse import build_tfidf_index
from rag.utils.history import build_history
from rag.utils.io import load_chunks, load_index

import faiss
import streamlit as st


QUALITY_MODE_MAP = {
    "Fast": "cheap",
    "Balanced": "balanced",
    "Accurate": "accurate",
}

OLLAMA_MODELS = ["llama3.1:8b"]
OPENAI_MODELS = ["gpt-4.1-mini", "gpt-4o-mini"]

DEFAULT_FAISS_K = 20
DEFAULT_TFIDF_K = 20


@st.cache_resource(show_spinner=False)
def cached_load_index() -> faiss.Index:
    """Load cached FAISS index."""
    return load_index()


@st.cache_data(show_spinner=False)
def cached_load_chunks() -> list[dict[str, Any]]:
    """Load cached chunks."""
    return load_chunks()


@st.cache_resource(show_spinner=False)
def cached_build_tfidf_index(chunks: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Build cached sparse index."""
    return build_tfidf_index(chunks)


def setup_page() -> None:
    """Set page config and sidebar styles."""
    st.set_page_config(page_title="Modular RAG Assistant", layout="wide")

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 1.35rem !important;
            font-weight: 800 !important;
        }

        .sidebar-section-title {
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 1.25rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize session defaults."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "total_cost_usd" not in st.session_state:
        st.session_state.total_cost_usd = 0.0

    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0

    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0


def render_section_title(title: str) -> None:
    """Render sidebar title."""
    st.markdown(
        f'<div class="sidebar-section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render app header."""
    st.title("Modular RAG Assistant")
    st.markdown(
        "Ask questions about your documents using a modular RAG pipeline with "
        "hybrid retrieval, reranking, and transparent usage metrics."
    )
    st.caption("⚡ Powered by Hybrid RAG: semantic search + keyword search + reranking")


def render_session_usage() -> None:
    """Render session totals."""
    st.metric("Total session cost (USD)", f"${st.session_state.total_cost_usd:.6f}")
    st.metric("Total input tokens", st.session_state.total_input_tokens)
    st.metric("Total output tokens", st.session_state.total_output_tokens)


def update_session_usage(tokens: dict[str, int] | None, cost_usd: float | None) -> None:
    """Update session totals."""
    tokens = tokens or {}
    st.session_state.total_cost_usd += cost_usd or 0.0
    st.session_state.total_input_tokens += tokens.get("input", 0)
    st.session_state.total_output_tokens += tokens.get("output", 0)


def reset_conversation() -> None:
    """Reset conversation state."""
    st.session_state.messages = []
    st.session_state.total_cost_usd = 0.0
    st.session_state.total_input_tokens = 0
    st.session_state.total_output_tokens = 0
    st.rerun()


def load_pipeline() -> ModularRAGPipeline | None:
    """Load RAG pipeline."""
    try:
        index = cached_load_index()
        chunks = cached_load_chunks()
        vectorizer, tfidf_matrix = cached_build_tfidf_index(chunks)
    except FileNotFoundError:
        return None

    return ModularRAGPipeline(
        index=index,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


def save_uploaded_file() -> None:
    """Save uploaded file."""
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    if uploaded_file is None:
        return

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    st.success(f"Saved file: {uploaded_file.name}")


def handle_rebuild() -> None:
    """Handle index rebuild."""
    if not st.button("Rebuild knowledge base"):
        return

    with st.spinner("Rebuilding knowledge base..."):
        try:
            rebuild_knowledge_base()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Knowledge base updated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error while updating knowledge base: {exc}")


def render_sidebar() -> dict[str, Any]:
    """Render sidebar settings."""
    with st.sidebar:
        render_section_title("Assistant settings")

        mode = st.radio(
            "Mode",
            ["Chat", "Summary"],
            help="Chat answers questions. Summary creates structured summaries.",
        )

        quality_label = st.selectbox(
            "Quality mode",
            ["Fast", "Balanced", "Accurate"],
            index=1,
            help=(
                "Fast uses less context and is quicker. "
                "Balanced is the default. "
                "Accurate uses more context and may be slower."
            ),
        )
        st.caption("Choose how much context the assistant should use.")

        with st.expander("LLM provider settings"):
            provider_label = st.selectbox(
                "Provider",
                ["Ollama local", "OpenAI API"],
                help="Choose whether answers should be generated locally or through OpenAI API.",
            )

            if provider_label == "Ollama local":
                llm_provider = "ollama"
                llm_model = st.selectbox("Model", OLLAMA_MODELS, index=0)
                st.caption("Runs locally through Ollama. API cost is always $0.")
            else:
                llm_provider = "openai"
                llm_model = st.selectbox("Model", OPENAI_MODELS, index=0)
                st.caption("Uses OpenAI API. Cost is estimated from input and output tokens.")

        with st.expander("Advanced retrieval settings"):
            retrieval_mode = st.selectbox(
                "Retrieval mode",
                options=["hybrid", "dense", "sparse"],
                index=0,
            )

            top_k = st.slider(
                "Number of retrieved chunks",
                min_value=1,
                max_value=10,
                value=5,
            )

            alpha = st.slider(
                "Hybrid alpha",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Higher value gives more weight to dense vector search.",
            )

        with st.expander("Session usage"):
            render_session_usage()
            if llm_provider == "openai":
                st.caption("Using OpenAI API — cost accumulates per query.")
            else:
                st.caption("Using local model — no API cost.")

        render_section_title("Your documents")
        save_uploaded_file()
        handle_rebuild()

        if st.button("Reset conversation"):
            reset_conversation()

    return {
        "mode": mode,
        "quality_label": quality_label,
        "generation_mode": QUALITY_MODE_MAP[quality_label],
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "alpha": alpha,
    }


def render_request_usage(
    tokens: dict[str, int] | None,
    cost_usd: float,
    config: dict[str, Any],
    latency: float | None = None,
) -> None:
    """Render usage for one request."""
    tokens = tokens or {}

    with st.expander("Usage details"):
        if latency is None:
            col1, col2 = st.columns(2)
        else:
            col0, col1, col2 = st.columns(3)
            with col0:
                st.metric("Latency", f"{latency:.2f}s")

        with col1:
            st.metric("Estimated cost", f"${cost_usd:.6f}")

        with col2:
            st.metric(
                "Tokens",
                f"{tokens.get('input', 0)} in / {tokens.get('output', 0)} out",
            )

        details = [
            f"Provider: {config['llm_provider']}",
            f"Model: {config['llm_model']}",
        ]

        if config.get("quality_label"):
            details.append(f"Quality mode: {config['quality_label']}")

        details.append(f"Retrieval mode: {config['retrieval_mode']}")
        st.caption(" | ".join(details))


def render_sources(results: list[dict[str, Any]]) -> None:
    """Render source chunks."""
    if not results:
        return

    st.subheader("Sources")

    for result in results:
        label = (
            f"{result.get('rank', '?')} | {result.get('source', 'unknown source')} "
            f"| hybrid: {result.get('hybrid_score', 0.0):.3f} "
            f"| vec: {result.get('vector_score', 0.0):.3f} "
            f"| tfidf: {result.get('tfidf_score', 0.0):.3f}"
        )

        with st.expander(label):
            st.write(result.get("text", ""))


def render_chat_history() -> None:
    """Render previous messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def render_chat_mode(pipeline: ModularRAGPipeline, config: dict[str, Any]) -> None:
    """Render chat mode."""
    render_chat_history()

    query = st.chat_input("e.g. What are the key ideas in this document?")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    history = build_history(st.session_state.messages[:-1])

    with st.spinner("Searching documents and generating answer..."):
        try:
            output = pipeline.run_chat(
                query=query,
                history=history,
                top_k=config["top_k"],
                faiss_k=DEFAULT_FAISS_K,
                tfidf_k=DEFAULT_TFIDF_K,
                alpha=config["alpha"],
                retrieval_mode=config["retrieval_mode"],
                generation_mode=config["generation_mode"],
                llm_provider=config["llm_provider"],
                llm_model=config["llm_model"],
            )
        except Exception as exc:
            output = {
                "answer": f"An error occurred: {exc}",
                "results": [],
                "corrected_query": None,
                "latency": 0.0,
                "tokens": {"input": 0, "output": 0},
                "cost_usd": 0.0,
            }

    answer = output.get("answer", "")
    corrected_query = output.get("corrected_query")
    results = output.get("results", [])
    tokens = output.get("tokens", {})
    cost_usd = output.get("cost_usd", 0.0)
    latency = output.get("latency", 0.0)

    update_session_usage(tokens, cost_usd)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        if corrected_query and corrected_query.casefold() != query.casefold():
            st.caption(f"Did you mean: {corrected_query}?")

        st.write(answer)
        render_request_usage(tokens, cost_usd, config, latency=latency)

    render_sources(results)


def render_summary_mode(pipeline: ModularRAGPipeline, config: dict[str, Any]) -> None:
    """Render summary mode."""
    st.subheader("Summary Generator")

    topic = st.text_input(
        "What topic should be summarized?",
        placeholder="e.g. retrieval augmented generation, tokenization, BERT",
    )

    if not st.button("Generate summary") or not topic.strip():
        return

    with st.spinner("Searching documents and generating summary..."):
        try:
            output = pipeline.run_summary(
                topic=topic.strip(),
                top_k=config["top_k"],
                faiss_k=DEFAULT_FAISS_K,
                tfidf_k=DEFAULT_TFIDF_K,
                alpha=config["alpha"],
                retrieval_mode=config["retrieval_mode"],
                llm_provider=config["llm_provider"],
                llm_model=config["llm_model"],
            )
        except Exception as exc:
            output = {
                "summary": f"An error occurred: {exc}",
                "results": [],
                "tokens": {"input": 0, "output": 0},
                "cost_usd": 0.0,
            }

    summary = output.get("summary", "")
    results = output.get("results", [])
    tokens = output.get("tokens", {})
    cost_usd = output.get("cost_usd", 0.0)

    update_session_usage(tokens, cost_usd)

    st.subheader("Summary")
    st.write(summary)
    render_request_usage(tokens, cost_usd, config)
    render_sources(results)


def main() -> None:
    """Run app."""
    setup_page()
    init_session_state()
    render_header()

    config = render_sidebar()
    pipeline = load_pipeline()

    if pipeline is None:
        st.info("Upload documents in the sidebar, then click 'Rebuild knowledge base'.")
        st.stop()

    if config["mode"] == "Chat":
        render_chat_mode(pipeline, config)
    else:
        render_summary_mode(pipeline, config)


if __name__ == "__main__":
    main()