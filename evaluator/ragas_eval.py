# evaluator/ragas_eval.py
import pandas as pd
import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

import os
os.environ["MAX_WORKERS"] = "1"        # Force RAGAS to evaluate sequentially
os.environ["RAGAS_MAX_WORKERS"] = "1"

from ragas.run_config import RunConfig
ragas_run_config = RunConfig(timeout=3000, max_workers=1)

# Override RAGAS to use local Ollama + HuggingFace
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RAGAS evaluation using device: {device}")

ragas_llm = LangchainLLMWrapper(Ollama(model="llama3.2", temperature=0, timeout=3000.0))
ragas_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
)

RESULTS_DIR = str(Path(__file__).parent.parent / "results")


def build_ragas_dataset(df_strategy: pd.DataFrame) -> Dataset:
    """
    RAGAS expects a HuggingFace Dataset with these exact column names:
      - question
      - answer
      - contexts       (list of strings)
      - ground_truth   (string)
    """
    data = {
        "question":     df_strategy["question"].tolist(),
        "answer":       df_strategy["answer"].tolist(),
        "contexts":     df_strategy["contexts"].tolist(),
        "ground_truth": df_strategy["ground_truth"].tolist(),
    }
    return Dataset.from_dict(data)


def evaluate_strategy(strategy_name: str, df: pd.DataFrame) -> dict:
    """
    Run RAGAS evaluation for one strategy.
    Returns a dict of metric scores.
    """
    print(f"\nEvaluating: {strategy_name}")
    df_s = df[df["strategy"] == strategy_name].copy()

    dataset = build_ragas_dataset(df_s)

    # Run RAGAS — this calls an LLM internally to judge faithfulness
    # By default uses OpenAI — we'll override to use Ollama below
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=ragas_run_config,
        raise_exceptions=False,
    )

    result = {
        "strategy":          strategy_name,
        "faithfulness":      round(scores["faithfulness"], 4),
        "answer_relevancy":  round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),   # ≈ precision@k
        "context_recall":    round(scores["context_recall"], 4),      # ≈ recall@k
        "avg_latency":       round(df_s["latency"].mean(), 3),
        "avg_tokens":        round(df_s["num_tokens"].mean(), 1),
        "num_questions":     len(df_s),
    }

    print(f"  faithfulness     : {result['faithfulness']}")
    print(f"  answer_relevancy : {result['answer_relevancy']}")
    print(f"  context_precision: {result['context_precision']}")
    print(f"  context_recall   : {result['context_recall']}")
    print(f"  avg_latency      : {result['avg_latency']}s")

    return result


def evaluate_all(raw_results_path: str = None) -> pd.DataFrame:
    """
    Load raw results, run RAGAS on all 5 strategies, save final scores CSV.
    """
    if raw_results_path is None:
        raw_results_path = str(Path(__file__).parent.parent / "results" / "raw_results.json")
        
    df = pd.read_json(raw_results_path)
    strategies = df["strategy"].unique()

    all_scores = []
    for strategy in strategies:
        scores = evaluate_strategy(strategy, df)
        all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    # Save
    out_path = f"{RESULTS_DIR}/ragas_scores.csv"
    scores_df.to_csv(out_path, index=False)
    print(f"\nFinal scores saved to: {out_path}")

    # Pretty print
    print("\n── Final Leaderboard ────────────────────────────────────────────")
    print(scores_df.to_string(index=False))

    return scores_df


if __name__ == "__main__":
    evaluate_all()