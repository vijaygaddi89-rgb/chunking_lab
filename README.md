# RAG Chunking Benchmark Laboratory

An elegant benchmarking framework to implement, analyze, and evaluate document chunking strategies for Retrieval-Augmented Generation (RAG) pipelines.

This laboratory compares **five different chunking strategies** on retrieval precision, context recall, response faithfulness, latency, and token consumption using a local LLM judge and the **RAGAS** evaluation suite.

---

## Architecture & Modules

The pipeline is split into separate modules covering ingestion, chunking, indexing, retrieval, evaluation, and visual analytics:

### 1. Chunking Strategies (`chunkers/`)
* **Fixed Size**: splits text into uniform character lengths with specified overlap.
* **Recursive Split**: splits text based on a hierarchical set of separators (paragraphs, sentences, words).
* **Semantic Chunks**: determines chunk boundaries dynamically using embedding similarity between consecutive sentences.
* **Hierarchical / Parent-Child**: indexes small child chunks for precise retrieval, while returning larger parent chunks to the LLM for context.

### 2. Core Pipeline
* **`loader.py`**: Handles PDF extraction and text normalization (whitespace cleaning, hyphenation correction).
* **`indexer.py`**: Indexes document chunks into separate Chroma DB collections using the `all-MiniLM-L6-v2` embedding model.
* **`retriever.py`**: Sets up similarity retrievers for each collection.
* **`query_runner.py`**: Runs benchmark queries against the retrievers and collects generations from local Ollama (`llama3.2`).
* **`evaluator/ragas_eval.py`**: Computes RAGAS metrics (Faithfulness, Relevancy, Precision, and Recall) sequentially.

### 3. Analytics Dashboard
* **`dashboard.py`**: A Streamlit application visualizing leaderboard scores, chunk stats, size distributions, and side-by-side strategy generation comparisons.

---

## Quick Start

### Prerequisites
Make sure you have Python 3.10+ and a local instance of **Ollama** running with `llama3.2` loaded:
```bash
ollama pull llama3.2
```

### Installation
Activate your virtual environment and install the required dependencies:
```powershell
# Windows PowerShell
.\chunking\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Execution Pipeline

Run the pipeline steps in order to process the document and start the dashboard:

```bash
# 1. Chunk and index the PDF into Chroma DB
python indexer.py

# 2. Run benchmark queries and collect generated answers
python query_runner.py

# 3. (Optional) Run RAGAS metrics evaluation
python evaluator/ragas_eval.py

# 4. Start the visualization dashboard
streamlit run dashboard.py
```

---

## Dashboard Overview

* **Evaluation Leaderboard**: Compare latency, token counts, and RAGAS scores across all strategies.
* **RAG Sandbox Playground**: Test custom queries and compare retrieved chunks and generated answers side-by-side.
* **Chunk Explorer**: Search, inspect, and analyze size distributions of the chunks stored in your vector database.
