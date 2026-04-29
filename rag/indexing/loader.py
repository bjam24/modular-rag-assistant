"""
Document loading module.

Loads supported files from disk and converts them into
a unified format for indexing.
"""

from pathlib import Path
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path


def load_txt(file_path: Path) -> str:
    """
    Load text content from a .txt file.
    """
    return file_path.read_text(encoding="utf-8").strip()


def load_pdf(file_path: Path) -> str:
    """
    Extract text from PDF.

    First tries standard extraction.
    Falls back to OCR if no text is found (scanned PDFs).
    """
    reader = PdfReader(str(file_path))
    pages = []

    # --- normal extraction ---
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    text_content = "\n\n".join(pages).strip()

    # jeśli mamy tekst → zwracamy
    if text_content:
        return text_content

    # --- OCR fallback ---
    print(f"[OCR] Falling back to OCR for: {file_path.name}")

    images = convert_from_path(str(file_path), dpi=300)

    ocr_pages = []
    for i, img in enumerate(images, start=1):
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text.strip():
            ocr_pages.append(f"\n\n--- Page {i} ---\n{ocr_text.strip()}")

    return "\n\n".join(ocr_pages).strip()


def load_documents(data_dir: Path) -> list[dict]:
    """
    Load all supported documents from a directory.
    """
    documents = []

    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            text = load_txt(file_path)
        elif suffix == ".pdf":
            text = load_pdf(file_path)
        else:
            continue

        if not text:
            continue

        documents.append({
            "source": file_path.name,
            "text": text,
        })

    return documents