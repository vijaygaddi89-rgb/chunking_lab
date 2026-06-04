# query_runner.py
import time
import json
import pandas as pd
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from retriever import build_retrievers
from data.questions import QUESTIONS, GROUND_TRUTHS

RESULTS_DIR = "results"
Path(RESULTS_DIR).mkdir(exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm():
    return Ollama(
        model="llama3.2",
        temperature=0,        # deterministic — important for fair comparison
        num_predict=256,      # cap output tokens to control latency
    )

# ── Prompt ────────────────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    template="""Use ONLY the context below to answer the question.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

# ── Run one question against one strategy ────────────────────────────────────
def run_single_query(retriever, llm, question: str) -> dict:
    """
    Returns:
        contexts  : list of retrieved chunk texts
        answer    : generated answer string
        latency   : seconds taken
        num_tokens: rough token count (len/4)
    """
    start = time.time()

    # Retrieve top-5 chunks
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    # Build context string for LLM
    context_str = "\n\n---\n\n".join(contexts)

    # Generate answer
    prompt = RAG_PROMPT.format(context=context_str, question=question)
    answer = llm.invoke(prompt)

    latency = time.time() - start

    # Rough token estimate (1 token ≈ 4 chars)
    num_tokens = len(prompt) // 4 + len(answer) // 4

    return {
        "contexts":   contexts,
        "answer":     answer.strip(),
        "latency":    round(latency, 3),
        "num_tokens": num_tokens,
    }


# ── Run all 50 questions for one strategy ────────────────────────────────────
def run_strategy(strategy_name: str, retriever, llm) -> list:
    """
    Returns a list of result dicts — one per question.
    """
    print(f"\n── Running: {strategy_name} ──────────────────")
    results = []

    for i, (question, ground_truth) in enumerate(zip(QUESTIONS[:10], GROUND_TRUTHS[:10])):
        print(f"  Q{i+1:02d}/10: {question[:60]}...")

        result = run_single_query(retriever, llm, question)
        result.update({
            "strategy":     strategy_name,
            "question":     question,
            "ground_truth": ground_truth,
        })
        results.append(result)

    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"  Done. Avg latency: {avg_latency:.2f}s")
    return results


# ── Run all 5 strategies ──────────────────────────────────────────────────────
def run_all_strategies() -> pd.DataFrame:
    print("Building retrievers...")
    retrievers = build_retrievers(k=5)
    llm = get_llm()

    all_results = []

    for strategy_name, retriever in retrievers.items():
        results = run_strategy(strategy_name, retriever, llm)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Save raw results for RAGAS evaluation
    raw_path = f"{RESULTS_DIR}/raw_results.json"
    df.to_json(raw_path, orient="records", indent=2)
    print(f"\nRaw results saved to: {raw_path}")

    return df


if __name__ == "__main__":
    df = run_all_strategies()
    print(f"\nTotal rows: {len(df)}")
    print(df[["strategy", "question", "latency", "num_tokens"]].head(10))