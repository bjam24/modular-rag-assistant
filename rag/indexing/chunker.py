"""
Text chunking module.

Splits documents into clean paragraphs and fixed-size chunks
for embedding and retrieval.
"""

import re


def split_into_paragraphs(text: str) -> list[str]:
    """
    Clean and split raw text into paragraphs.
    """
    text = text.replace("\r\n", "\n")

    # Fix hyphenation from PDFs (e.g. "exam-\nple" -> "example")
    text = re.sub(r"-\n", "", text)

    # Join broken lines inside paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spacing
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = [
        p.strip()
        for p in text.strip().split("\n\n")
        if p.strip()
    ]

    return paragraphs


def chunk_paragraphs(
    paragraphs: list[str],
    max_words: int = 100,
    overlap_words: int = 20,
    max_chars: int = 1500,
) -> list[str]:
    """
    Convert paragraphs into overlapping chunks.

    Uses both word and character limits to avoid exceeding embedding model context.
    """
    chunks = []
    current_chunk = []
    current_len = 0

    def flush_current():
        if current_chunk:
            text = " ".join(current_chunk).strip()
            if text:
                chunks.append(text[:max_chars])

    for paragraph in paragraphs:
        words = paragraph.split()
        length = len(words)

        if length > max_words:
            start = 0
            step = max_words - overlap_words

            while start < length:
                end = start + max_words
                chunk = " ".join(words[start:end]).strip()
                if chunk:
                    chunks.append(chunk[:max_chars])
                start += step

            continue

        candidate = " ".join(current_chunk + [paragraph]).strip()

        if current_len + length <= max_words and len(candidate) <= max_chars:
            current_chunk.append(paragraph)
            current_len += length
        else:
            flush_current()
            current_chunk = [paragraph]
            current_len = length

    flush_current()

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into chunked units with metadata.
    """
    all_chunks = []

    for doc in documents:
        paragraphs = split_into_paragraphs(doc["text"])
        chunks = chunk_paragraphs(paragraphs)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })

    return all_chunks