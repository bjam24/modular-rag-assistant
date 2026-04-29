"""
Knowledge base builder.

Loads source documents, chunks them, embeds the chunks, builds a FAISS index,
and saves retrieval artifacts to disk.
"""

import pickle

import faiss
import numpy as np

from rag.config import CHUNKS_PATH, DATA_DIR, INDEX_PATH, STORAGE_DIR
from rag.indexing.chunker import chunk_documents
from rag.indexing.embedder import build_embeddings
from rag.indexing.loader import load_documents


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index from dense embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def rebuild_knowledge_base() -> None:
    """
    Rebuild and save the full knowledge base from source documents.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents)
    embeddings = build_embeddings(chunks)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)


if __name__ == "__main__":
    print("Building knowledge base...")
    rebuild_knowledge_base()
    print("Done. Index saved to storage/")