# loader.py
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import re


def load_document(pdf_path: str) -> list:
    """
    Load a PDF and return a list of LangChain Document objects.
    One Document per page.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"Loading: {path.name}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    print(f"  Pages loaded     : {len(pages)}")
    print(f"  Total characters : {sum(len(p.page_content) for p in pages):,}")
    print(f"  Avg chars/page   : {sum(len(p.page_content) for p in pages) // len(pages):,}")

    return pages


def clean_documents(pages: list) -> list:
    """
    Basic cleaning — remove excessive whitespace, fix hyphenated line breaks.
    This matters a lot for chunking quality.
    """
    cleaned = []
    for doc in pages:
        text = doc.page_content

        # Fix hyphenated line-breaks common in PDFs (e.g. "infor-\nmation" → "information")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Collapse multiple newlines → double newline (paragraph boundary)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        text = text.strip()

        # Copy the doc with cleaned content
        doc.page_content = text
        cleaned.append(doc)

    # Filter out near-empty pages (table of contents, blank pages, etc.)
    cleaned = [d for d in cleaned if len(d.page_content) > 100]
    print(f"  Pages after cleaning: {len(cleaned)}")
    return cleaned


def get_full_text(pages: list) -> str:
    """
    Join all pages into a single string.
    Needed for semantic chunker which works on the full document.
    """
    return "\n\n".join([p.page_content for p in pages])


if __name__ == "__main__":
    # Test your loader — run: python loader.py
    pages = load_document("data/document.pdf")
    pages = clean_documents(pages)

    print("\n--- Sample from page 1 ---")
    print(pages[0].page_content[:500])
    print("\n--- Metadata ---")
    print(pages[0].metadata)