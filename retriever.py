# retriever.py
from langchain_chroma import Chroma
from indexer import load_vectorstore, get_embedding_model, STRATEGY_NAMES


def build_retrievers(k: int = 5) -> dict:
    """
    Load all 5 vectorstores from disk and wrap them as retrievers.
    k = number of chunks to retrieve per query (used for precision@k, recall@k)

    Returns:
        { strategy_name: VectorStoreRetriever }
    """
    embeddings = get_embedding_model()
    retrievers = {}

    for name in STRATEGY_NAMES:
        try:
            vs = load_vectorstore(name, embeddings)
            retrievers[name] = vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )
            print(f"Retriever ready: {name} (k={k})")
        except FileNotFoundError as e:
            print(f"SKIP: {e}")

    return retrievers