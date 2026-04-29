"""
Streamlit UI for the Modular RAG Assistant.

User-friendly interface for:
- document question answering
- summarization
- document upload
- cost/latency transparency

Advanced retrieval settings are hidden by default.
"""

from pathlib import Path

import faiss
import streamlit as st

from rag.config import DATA_DIR
from rag.indexing.builder import rebuild_knowledge_base
from rag.orchestration.pipeline import ModularRAGPipeline
from rag.retrieval.sparse import build_tfidf_index
from rag.utils.history import build_history
from rag.utils.io import load_chunks, load_index


QUALITY_MODE_MAP = {
    "Fast": "cheap",
    "Balanced": "balanced",
    "Accurate": "accurate",
}


@st.cache_resource
def cached_load_index() -> faiss.Index:
    return load_index()


@st.cache_data
def cached_load_chunks() -> list[dict]:
    return load_chunks()


@st.cache_resource
def cached_build_tfidf_index(chunks: list[dict]) -> tuple:
    return build_tfidf_index(chunks)


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

st.title("Modular RAG Assistant")
st.markdown(
    "Ask questions about your documents using a modular RAG pipeline with "
    "hybrid retrieval, reranking, and transparent usage metrics."
)
st.caption("⚡ Powered by Hybrid RAG: semantic search + keyword search + reranking")
st.info("Upload documents in the sidebar, rebuild the knowledge base, then ask a question below.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0

if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0

if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0


index = cached_load_index()
chunks = cached_load_chunks()
vectorizer, tfidf_matrix = cached_build_tfidf_index(chunks)

pipeline = ModularRAGPipeline(
    index=index,
    chunks=chunks,
    vectorizer=vectorizer,
    tfidf_matrix=tfidf_matrix,
)


with st.sidebar:
    st.markdown(
        '<div class="sidebar-section-title">Assistant settings</div>',
        unsafe_allow_html=True,
    )

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

    generation_mode = QUALITY_MODE_MAP[quality_label]
    st.caption("Choose how much context the assistant should use.")

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
        st.metric("Estimated API cost", f"${st.session_state.total_cost_usd:.6f}")
        st.metric("Input tokens", st.session_state.total_input_tokens)
        st.metric("Output tokens", st.session_state.total_output_tokens)
        st.caption(
            "Local Ollama models show $0 API cost. "
            "OpenAI models use configured pricing."
        )

    st.markdown(
        '<div class="sidebar-section-title">Your documents</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
    )

    if uploaded_file:
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        file_path = Path(DATA_DIR) / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Saved file: {uploaded_file.name}")

    if st.button("Rebuild knowledge base"):
        with st.spinner("Rebuilding knowledge base..."):
            try:
                rebuild_knowledge_base()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Knowledge base updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Error while updating knowledge base: {e}")

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.total_cost_usd = 0.0
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()


if mode == "Chat":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("e.g. What are the key ideas in this document?")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        history = build_history(st.session_state.messages[:-1])

        with st.spinner("Searching documents and generating answer..."):
            try:
                output = pipeline.run_chat(
                    query=query,
                    history=history,
                    top_k=top_k,
                    faiss_k=20,
                    tfidf_k=20,
                    alpha=alpha,
                    retrieval_mode=retrieval_mode,
                    generation_mode=generation_mode,
                )

                answer = output["answer"]
                results = output["results"]
                corrected_query = output.get("corrected_query")

                latency = output.get("latency", 0.0)
                tokens = output.get("tokens", {})
                cost = output.get("cost_usd", 0.0)

                st.session_state.total_cost_usd += cost
                st.session_state.total_input_tokens += tokens.get("input", 0)
                st.session_state.total_output_tokens += tokens.get("output", 0)

            except Exception as e:
                answer = f"An error occurred: {e}"
                results = []
                corrected_query = None
                latency = 0.0
                tokens = {"input": 0, "output": 0}
                cost = 0.0

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            if corrected_query and corrected_query.lower() != query.lower():
                st.caption(f"Did you mean: {corrected_query}?")

            st.write(answer)

            with st.expander("Usage details"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Latency", f"{latency:.2f}s")

                with col2:
                    st.metric("Estimated cost", f"${cost:.6f}")

                with col3:
                    st.metric(
                        "Tokens",
                        f"{tokens.get('input', 0)} in / {tokens.get('output', 0)} out",
                    )

                st.caption(
                    f"Quality mode: {quality_label} | "
                    f"Retrieval mode: {retrieval_mode}"
                )

        if results:
            st.subheader("Sources")

            for r in results:
                with st.expander(
                    f"{r['rank']} | {r['source']} | hybrid: {r['hybrid_score']:.3f} "
                    f"| vec: {r['vector_score']:.3f} | tfidf: {r['tfidf_score']:.3f}"
                ):
                    st.write(r["text"])


elif mode == "Summary":
    st.subheader("Summary Generator")

    topic = st.text_input(
        "What topic should be summarized?",
        placeholder="e.g. retrieval augmented generation, tokenization, BERT",
    )

    if st.button("Generate summary") and topic.strip():
        with st.spinner("Searching documents and generating summary..."):
            try:
                output = pipeline.run_summary(
                    topic=topic,
                    top_k=top_k,
                    faiss_k=20,
                    tfidf_k=20,
                    alpha=alpha,
                    retrieval_mode=retrieval_mode,
                )

                summary = output["summary"]
                results = output["results"]

            except Exception as e:
                summary = f"An error occurred: {e}"
                results = []

        st.subheader("Summary")
        st.write(summary)

        if results:
            st.subheader("Sources")

            for r in results:
                with st.expander(
                    f"{r['rank']} | {r['source']} | hybrid: {r['hybrid_score']:.3f} "
                    f"| vec: {r['vector_score']:.3f} | tfidf: {r['tfidf_score']:.3f}"
                ):
                    st.write(r["text"])