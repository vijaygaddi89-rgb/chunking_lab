# indexer.py
import time
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstores"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

# These names must match exactly what your chunkers return
STRATEGY_NAMES = [
    "fixed_size",
    "recursive_split",
    "semantic_chunks",
    "hierarchical_chunks",
    "parent_doc_retriever",
]


# ── Embedding model (loaded once, reused for all 5) ──────────────────────────
def get_embedding_model():
    """
    Load HuggingFace embedding model.
    all-MiniLM-L6-v2: small (80MB), fast, 384-dim output.
    Perfect for local benchmarking.
    """
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},  # needed for cosine similarity
    )
    print("  Embedding model ready.")
    return embeddings


# ── Index one strategy ────────────────────────────────────────────────────────
def index_strategy(
    strategy_name: str,
    chunks: list,
    embeddings,
    overwrite: bool = True,
) -> Chroma:
    """
    Takes a list of LangChain Documents (chunks) and indexes them
    into a ChromaDB collection named after the strategy.

    Returns the Chroma vectorstore object so you can query it later.
    """
    persist_dir = str(Path(VECTORSTORE_DIR) / strategy_name)

    # Wipe existing collection so reruns are clean
    if overwrite and Path(persist_dir).exists():
        import shutil
        shutil.rmtree(persist_dir)

    print(f"\nIndexing: {strategy_name}")
    print(f"  Chunks to embed : {len(chunks)}")

    # Add strategy name to each chunk's metadata — important for RAGAS later
    for i, chunk in enumerate(chunks):
        chunk.metadata["strategy"] = strategy_name
        chunk.metadata["chunk_id"] = i

    start = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=strategy_name,
        persist_directory=persist_dir,
    )

    elapsed = time.time() - start
    print(f"  Indexed in      : {elapsed:.1f}s")
    print(f"  Avg per chunk   : {elapsed/len(chunks)*1000:.1f}ms")
    print(f"  Saved to        : {persist_dir}")

    return vectorstore


# ── Load an already-indexed store (for query time) ────────────────────────────
def load_vectorstore(strategy_name: str, embeddings) -> Chroma:
    """
    Load a previously built ChromaDB collection from disk.
    Use this in query_runner.py and dashboard.py instead of re-indexing.
    """
    persist_dir = str(Path(VECTORSTORE_DIR) / strategy_name)

    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No vectorstore found for '{strategy_name}'. Run indexer.py first."
        )

    vectorstore = Chroma(
        collection_name=strategy_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    count = vectorstore._collection.count()
    print(f"Loaded '{strategy_name}': {count} chunks")
    return vectorstore


# ── Run all 5 ─────────────────────────────────────────────────────────────────
def index_all(chunks_dict: dict) -> dict:
    """
    chunks_dict = {
        "fixed_size":           [Document, ...],
        "recursive_split":      [Document, ...],
        "semantic_chunks":      [Document, ...],
        "hierarchical_chunks":  [Document, ...],
        "parent_doc_retriever": [Document, ...],
    }

    Returns:
        vectorstores_dict = { strategy_name: Chroma, ... }
    """
    Path(VECTORSTORE_DIR).mkdir(exist_ok=True)

    embeddings = get_embedding_model()
    vectorstores = {}

    total_start = time.time()

    for name in STRATEGY_NAMES:
        if name not in chunks_dict:
            print(f"WARNING: No chunks found for strategy '{name}', skipping.")
            continue
        vectorstores[name] = index_strategy(name, chunks_dict[name], embeddings)

    total = time.time() - total_start
    print(f"\nAll 5 strategies indexed in {total:.1f}s total")

    # Print summary table
    print("\n── Summary ──────────────────────────────────")
    print(f"{'Strategy':<25} {'Chunks':>8}")
    print("─" * 35)
    for name, vs in vectorstores.items():
        count = vs._collection.count()
        print(f"{name:<25} {count:>8}")

    return vectorstores


# ── Smoke test — verify retrieval works ───────────────────────────────────────
def smoke_test(vectorstores: dict, query: str = "What is the main contribution?"):
    """
    Run a test query against all 5 stores and print top-1 result.
    Verifies everything is working before you run 50 questions.
    """
    print(f"\n── Smoke test: '{query}' ──────────────────")
    embeddings = get_embedding_model()

    for name, vs in vectorstores.items():
        results = vs.similarity_search(query, k=1)
        snippet = results[0].page_content[:120].replace("\n", " ") if results else "NO RESULTS"
        print(f"\n{name}:")
        print(f"  → {snippet}...")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Import your Phase 1 loader and Phase 2 chunkers
    from loader import load_document, clean_documents
    from chunkers.fixed_size        import FixedSizeChunker
    from chunkers.recursive         import RecursiveChunker
    from chunkers.semantic          import SemanticChunker
    from chunkers.hierarchical      import HierarchicalChunker

    # 1. Load document
    pages = load_document("data/document.pdf")
    pages = clean_documents(pages)

    # 2. Run all 5 chunkers
    print("\nRunning all 5 chunking strategies...")
    
    hierarchical_children, _ = HierarchicalChunker().split_documents(pages)
    
    chunks_dict = {
        "fixed_size":           FixedSizeChunker().split_documents(pages),
        "recursive_split":      RecursiveChunker().split_documents(pages),
        "semantic_chunks":      SemanticChunker().split_documents(pages),
        "hierarchical_chunks":  hierarchical_children,
        "parent_doc_retriever": RecursiveChunker().split_documents(pages),  # parent-doc uses same child chunks
    }

    # 3. Index all 5 into ChromaDB
    vectorstores = index_all(chunks_dict)

    # 4. Smoke test
    smoke_test(vectorstores)