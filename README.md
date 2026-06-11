# 🧠 RAG Chunking Benchmark Laboratory

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Ragas Evaluated](https://img.shields.io/badge/Ragas-Evaluated-brightgreen.svg)](https://github.com/explodinggradients/ragas)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-orange.svg)](https://www.trychroma.com/)

An advanced, production-grade benchmarking framework to implement, visualize, and systematically evaluate document chunking strategies for Retrieval-Augmented Generation (RAG) pipelines.

This repository compares **five core chunking methodologies** across retrieval precision, semantic coverage, faithfulness (hallucination checks), latency, and token efficiency using local LLMs and the **RAGAS** assessment framework.

---

## 🚀 Key Features

* **Interactive Chunker Visualizer**: A live boundary simulator. Paste text, adjust thresholds or chunk sizes, and instantly see color-coded output chunks and nested parent-child trees.
* **Side-by-Side RAG Sandbox**: Test user queries against all five indexing strategies simultaneously, displaying retrieved passages and generated answers side-by-side.
* **Performance Leaderboard**: Compare token overhead, response times, and RAGAS scores (Faithfulness, Relevancy, Precision, Recall) using clear, interactive Plotly charts.
* **Automated Evaluation Suite**: Run a structured benchmark of 40 evaluation questions with ground truth answers, using local Ollama (`llama3.2`) and HuggingFace embeddings.

---

## 🛠️ Chunking Strategies Compared

| Strategy | Separation Logic | Pros | Cons | Ideal Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Fixed-Size** | Strict character limits + overlap | Fast, computationally trivial | Splits sentences and words in half | Homogeneous simple documents |
| **Recursive Split** | Paragraphs $\rightarrow$ Sentences $\rightarrow$ Words | Preserves semantic structural flow | Static character constraints | General-purpose recommendation |
| **Semantic Chunks** | Cosine similarity drops | Conceptually coherent boundaries | Slower (requires embedding every sentence) | Complex long-form documents |
| **Hierarchical Chunks** | Nested Child-Parent layers | High retrieval precision, rich context | Increased structural complexity | Multi-topic knowledge bases |
| **Parent-Doc Retriever** | Linked child units $\rightarrow$ Parent document | Avoids context loss for LLM | Dual-database synchronization | Enterprise search apps |

---

## 📁 Repository Structure

```filepath
├── chunkers/                # Core splitters for all 5 strategies
│   ├── fixed_size.py        # Character-boundary chunks
│   ├── recursive.py         # Hierarchical separator chunks
│   ├── semantic.py          # Embedding-similarity chunks
│   └── hierarchical.py      # Child-parent nested chunks
├── evaluator/
│   └── ragas_eval.py        # Local Ollama RAGAS metrics runner
├── data/
│   ├── document.pdf         # Source masterclass PDF
│   └── questions.py         # 40 golden Q&A evaluation dataset
├── results/
│   ├── raw_results.json     # Pre-computed query runs
│   └── ragas_scores.csv     # Evaluated RAGAS metrics leaderboard
├── loader.py                # PDF extractor & text cleaner
├── indexer.py               # ChromaDB embedding & indexing pipeline
├── retriever.py             # Vectorstore retriever builders
├── query_runner.py          # Benchmark query batch processor
├── dashboard.py             # Streamlit visual analytics app
└── requirements.txt         # Cleaned direct dependencies
```

---

## 🏁 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed and a local instance of **Ollama** running with `llama3.2` pulled:
```bash
ollama pull llama3.2
```

### 2. Setup Environment
Activate the virtual environment and install the dependencies:
```powershell
# Windows PowerShell
.\chunking\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Pipeline Execution
To rebuild the databases, process evaluation queries, and launch the UI:

```powershell
# Step 1: Parse and index the PDF into Chroma DB
python indexer.py

# Step 2: Batch run the evaluation queries against all retrievers
python query_runner.py

# Step 3: Run RAGAS metrics evaluation (optional)
python evaluator/ragas_eval.py

# Step 4: Start the visual dashboard
python -m streamlit run dashboard.py
```

---

## 📊 Evaluation Metrics Glossary

* **Faithfulness (Groundedness)**: Assesses if the generated response is strictly supported by the retrieved contexts, safeguarding against hallucinations (Target: `>0.85`).
* **Answer Relevancy**: Evaluates if the response directly addresses the user's intent without redundant padding (Target: `>0.80`).
* **Context Recall & Precision**: Measures the fraction of the ground-truth information captured in retrieved passages, and whether the retrieved context is clean and noise-free.
