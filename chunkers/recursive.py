# chunkers/recursive.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker:
    """
    Splits documents using hierarchical separators.
    Tries to maintain semantic coherence.
    
    Parameters:
    -----------
    chunk_size : int
        Maximum characters per chunk (default: 1000)
    chunk_overlap : int
        Overlap between chunks (default: 200)
    separators : list
        Ordered list of separators to try (default: paragraph > line > word)
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        # Initialize LangChain's splitter
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split raw text using recursive separator strategy."""
        return self.splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive strategy."""
        all_chunks = self.splitter.split_documents(documents)
        
        # Enrich metadata
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata.update({
                "chunk_index": idx,
                "chunking_strategy": "recursive",
                "chunk_size": self.chunk_size,
                "separators_used": str(self.separators)
            })
        
        # Add total count to each
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata["total_chunks"] = total
        
        return all_chunks
    
    def get_config(self) -> dict:
        return {
            "strategy": "recursive",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loader import load_document, clean_documents
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "document.pdf")
    pages = load_document(pdf_path)
    pages = clean_documents(pages)
    
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(pages)
    
    print(f"\n📊 Recursive Results:")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg length  : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print(f"  Min length  : {min(len(c.page_content) for c in chunks)} chars")
    print(f"  Max length  : {max(len(c.page_content) for c in chunks)} chars")
    
    print("\n--- Sample Chunk (notice clean boundaries) ---")
    print(chunks[5].page_content[:400])