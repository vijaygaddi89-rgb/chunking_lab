# chunkers/hierarchical.py
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from .recursive import RecursiveChunker


class HierarchicalChunker:
    """
    Creates parent-child chunk hierarchy.
    Children are indexed for retrieval; parents provide context.
    
    Parameters:
    -----------
    parent_chunk_size : int
        Size of parent/context chunks (default: 2000)
    child_chunk_size : int
        Size of child/retrieval chunks (default: 256)
    parent_overlap : int
        Overlap for parent chunks (default: 200)
    child_overlap : int
        Overlap for child chunks (default: 50)
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 256,
        parent_overlap: int = 200,
        child_overlap: int = 50
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        
        # Use recursive splitter for both levels
        self.parent_splitter = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap
        )
        self.child_splitter = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap
        )
    
    def _assign_child_to_parent(
        self, 
        child: Document, 
        parents: List[Document]
    ) -> int:
        """Find which parent this child belongs to (by content overlap)."""
        best_parent_idx = 0
        best_overlap = 0
        
        for p_idx, parent in enumerate(parents):
            # Simple check: does child text appear in parent?
            if child.page_content in parent.page_content:
                return p_idx
            
            # Fallback: count overlapping words
            child_words = set(child.page_content.lower().split())
            parent_words = set(parent.page_content.lower().split())
            overlap = len(child_words & parent_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_parent_idx = p_idx
        
        return best_parent_idx
    
    def split_documents(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Returns:
        --------
        tuple: (child_chunks, parent_chunks)
               Index children for retrieval; use parents for context
        """
        # Level 1: Create parent chunks
        parents = self.parent_splitter.split_documents(documents)
        
        # Level 2: Create children FROM PARENTS (not original docs)
        children = self.child_splitter.split_documents(parents)
        
        # Link children to parents
        for child in children:
            parent_idx = self._assign_child_to_parent(child, parents)
            
            child.metadata.update({
                "chunking_strategy": "hierarchical_child",
                "parent_id": f"parent_{parent_idx}",
                "parent_chunk_size": self.parent_chunk_size,
                "child_chunk_size": self.child_chunk_size,
                "level": "child"
            })
        
        # Mark parents
        for p_idx, parent in enumerate(parents):
            parent.metadata.update({
                "chunking_strategy": "hierarchical_parent",
                "parent_id": f"parent_{p_idx}",
                "level": "parent"
            })
        
        return children, parents
    
    def get_config(self) -> dict:
        return {
            "strategy": "hierarchical",
            "parent_chunk_size": self.parent_chunk_size,
            "child_chunk_size": self.child_chunk_size,
            "parent_overlap": self.parent_overlap,
            "child_overlap": self.child_overlap
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
    
    chunker = HierarchicalChunker(
        parent_chunk_size=1500,
        child_chunk_size=300
    )
    
    children, parents = chunker.split_documents(pages)
    
    print(f"\n📊 Hierarchical Results:")
    print(f"  Parent chunks : {len(parents)} (avg {sum(len(p.page_content) for p in parents)//len(parents)} chars)")
    print(f"  Child chunks  : {len(children)} (avg {sum(len(c.page_content) for c in children)//len(children)} chars)")
    print(f"  Ratio         : {len(children)/len(parents):.1f}x children per parent")
    
    print("\n--- Sample Parent (context) ---")
    print(f"[{parents[0].metadata['parent_id']}] {len(parents[0].page_content)} chars")
    print(parents[0].page_content[:300] + "...\n")
    
    print("--- Sample Child (retrieval unit) ---")
    print(f"[{children[0].metadata['parent_id']}] {len(children[0].page_content)} chars")
    print(children[0].page_content + "\n")
    
    print("--- Child Metadata (shows linkage) ---")
    print(children[0].metadata)