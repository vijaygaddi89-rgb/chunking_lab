# chunkers/fixed_size.py
from typing import List, Optional
from langchain_core.documents import Document


class FixedSizeChunker:
    """
    Splits documents into fixed-size chunks.
    
    Parameters:
    -----------
    chunk_size : int
        Maximum characters per chunk (default: 1000)
    chunk_overlap : int  
        Characters shared between adjacent chunks (default: 200)
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def split_text(self, text: str) -> List[str]:
        """Split raw text into chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not at the end, try to break at word boundary
            if end < len(text) and text[end] != ' ':
                # Look backwards for last space
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move forward by (chunk_size - overlap)
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split list of Documents into smaller Document chunks."""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text_chunks = self.split_text(doc.page_content)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                # Inherit metadata + add chunk info
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "chunking_strategy": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "source_document_index": doc_idx
                })
                
                all_chunks.append(
                    Document(page_content=chunk, metadata=metadata)
                )
        
        return all_chunks
    
    def get_config(self) -> dict:
        """Return configuration for logging/comparison."""
        return {
            "strategy": "fixed_size",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from loader import load_document, clean_documents
    
    pages = load_document(r"C:\Users\gaddi\Desktop\chunking_lab\data\document.pdf")
    pages = clean_documents(pages)
    
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents(pages)
    
    print(f"\n📊 Fixed-Size Results:")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg length  : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print(f"  Min length  : {min(len(c.page_content) for c in chunks)} chars")
    print(f"  Max length  : {max(len(c.page_content) for c in chunks)} chars")
    
    print("\n--- Sample Chunk ---")
    print(chunks[0].page_content[:300])
    print("\n--- Metadata ---")
    print(chunks[0].metadata)