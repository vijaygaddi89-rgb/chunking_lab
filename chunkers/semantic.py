# chunkers/semantic.py
import numpy as np
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import re


class SemanticChunker:
    """
    Splits documents based on embedding similarity.
    Creates coherent topical chunks regardless of size.
    
    Parameters:
    -----------
    model_name : str
        Sentence transformer model (default: 'all-MiniLM-L6-v2')
    similarity_threshold : float
        Minimum cosine similarity to stay in same chunk (default: 0.5)
    min_sentences : int
        Minimum sentences per chunk (default: 2)
    max_chunk_size : int
        Safety limit to prevent huge chunks (default: 3000)
    device : str
        Device for SentenceTransformer inference: 'cuda' or 'cpu' (default: 'cuda')
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_sentences: int = 2,
        max_chunk_size: int = 3000,
        device: str = "cuda"
    ):
        import torch
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.max_chunk_size = max_chunk_size
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Loading SemanticChunker model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("[OK] SemanticChunker model loaded!")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple regex approach)."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty/whitespace-only sentences
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for all sentences (runs on GPU if available)."""
        return self.model.encode(
            sentences,
            show_progress_bar=False,
            batch_size=64,          # larger batch = faster on GPU
            convert_to_numpy=True,
        )
    
    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Compute cosine similarity between consecutive sentences."""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(float(sim))
        return similarities
    
    def _find_split_points(self, similarities: List[float]) -> List[int]:
        """Find indices where similarity drops below threshold."""
        split_points = [0]  # Always start new chunk at beginning
        
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)  # Split AFTER this sentence
        
        return split_points
    
    def _create_chunks_from_splits(
        self, 
        sentences: List[str], 
        split_points: List[int]
    ) -> List[str]:
        """Group sentences into chunks based on split points."""
        chunks = []
        
        for i in range(len(split_points)):
            start = split_points[i]
            end = split_points[i + 1] if i + 1 < len(split_points) else len(sentences)
            
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            # Apply safety limits
            if len(chunk_text) > self.max_chunk_size:
                # Force split if too long
                chunk_text = chunk_text[:self.max_chunk_size]
            
            if len(chunk_sentences) >= self.min_sentences:
                chunks.append(chunk_text)
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """Main method: split text semantically."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < self.min_sentences:
            return [text]
        
        embeddings = self._compute_embeddings(sentences)
        similarities = self._compute_similarities(embeddings)
        split_points = self._find_split_points(similarities)
        chunks = self._create_chunks_from_splits(sentences, split_points)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents semantically."""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text_chunks = self.split_text(doc.page_content)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "chunking_strategy": "semantic",
                    "similarity_threshold": self.similarity_threshold,
                    "model": self.model_name,
                    "source_document_index": doc_idx
                })
                
                all_chunks.append(Document(page_content=chunk, metadata=metadata))
        
        # Add global totals
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata["global_total_chunks"] = total
        
        return all_chunks
    
    def get_config(self) -> dict:
        return {
            "strategy": "semantic",
            "model": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "min_sentences": self.min_sentences,
            "max_chunk_size": self.max_chunk_size
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
    
    chunker = SemanticChunker(
        similarity_threshold=0.45,  # Tune this!
        min_sentences=3
    )
    chunks = chunker.split_documents(pages)
    
    print(f"\nSemantic Results:")
    print(f"  Total chunks : {len(chunks)}")
    lengths = [len(c.page_content) for c in chunks]
    print(f"  Avg length  : {sum(lengths) // len(lengths)} chars")
    print(f"  Min length  : {min(lengths)} chars")
    print(f"  Max length  : {max(lengths)} chars")
    print(f"  Std Dev     : {np.std(lengths):.1f} chars")  # High variance expected!
    
    print("\n--- Sample Chunks (notice variable sizes) ---")
    for i in [0, 5, 10]:
        if i < len(chunks):
            print(f"\n[Chunk {i} - {len(chunks[i].page_content)} chars]")
            print(chunks[i].page_content[:250] + "...")