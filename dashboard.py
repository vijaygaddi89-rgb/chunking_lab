# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import os
import re

# Import elements from the workspace project
from indexer import load_vectorstore, get_embedding_model, STRATEGY_NAMES
from data.questions import QUESTIONS, GROUND_TRUTHS
from loader import load_document, clean_documents, get_full_text
from query_runner import RAG_PROMPT

# Set page config
st.set_page_config(
    page_title="RAG Chunking Strategies Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Gradient Banner */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid #312e81;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
    }
    
    .header-title {
        background: linear-gradient(to right, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        max-width: 800px;
        line-height: 1.6;
    }

    /* Styled Metric Cards */
    .metric-card {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(to right, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }

    /* Strategy badge styling */
    .strategy-badge {
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-fixed { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-recursive { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-semantic { background-color: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-hierarchical { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-parent { background-color: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }

    /* Playground Box */
    .playground-answer-box {
        background-color: #1e293b;
        border-left: 5px solid #6366f1;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: #f1f5f9;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .playground-chunk-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    
    /* Document view block */
    .doc-page-content {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Courier New', Courier, monospace;
        color: #94a3b8;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ── CACHED DATA LOADING ────────────────────────────────────────────────────────

@st.cache_resource
def get_cached_embedding_model():
    """Cache the embedding model loading."""
    try:
        return get_embedding_model()
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_data
def get_cached_pdf_data(pdf_path="data/document.pdf"):
    """Cache PDF loading & cleaning to avoid page lag."""
    try:
        pages = load_document(pdf_path)
        cleaned_pages = clean_documents(pages)
        full_text = get_full_text(cleaned_pages)
        return pages, cleaned_pages, full_text
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def get_hierarchical_parents():
    """Extract parents for parent-child retrieval resolution."""
    _, cleaned_pages, _ = get_cached_pdf_data()
    if cleaned_pages:
        from chunkers.hierarchical import HierarchicalChunker
        try:
            _, parents = HierarchicalChunker().split_documents(cleaned_pages)
            return parents
        except Exception as e:
            return []
    return []

@st.cache_data
def load_raw_results(json_path="results/raw_results.json"):
    """Load benchmark raw run results."""
    path = Path(json_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_ragas_scores(csv_path="results/ragas_scores.csv"):
    """Load RAGAS evaluation metrics leaderboard."""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    return None

# Load resources
pages, cleaned_pages, full_text = get_cached_pdf_data()
raw_results = load_raw_results()
ragas_scores = load_ragas_scores()

# Strategy display helpers
STRATEGY_LABELS = {
    "fixed_size": "Fixed Size",
    "recursive_split": "Recursive Split",
    "semantic_chunks": "Semantic Chunks",
    "hierarchical_chunks": "Hierarchical Chunks",
    "parent_doc_retriever": "Parent Document Retriever"
}

def get_strategy_badge_html(name):
    label = STRATEGY_LABELS.get(name, name)
    if name == "fixed_size":
        return f'<span class="strategy-badge badge-fixed">{label}</span>'
    elif name == "recursive_split":
        return f'<span class="strategy-badge badge-recursive">{label}</span>'
    elif name == "semantic_chunks":
        return f'<span class="strategy-badge badge-semantic">{label}</span>'
    elif name == "hierarchical_chunks":
        return f'<span class="strategy-badge badge-hierarchical">{label}</span>'
    elif name == "parent_doc_retriever":
        return f'<span class="strategy-badge badge-parent">{label}</span>'
    return f'<span class="strategy-badge">{label}</span>'

# ── SIDEBAR CONFIGURATION ──────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/brain.png", width=70)
    st.title("Settings & Status")
    
    st.subheader("System Status")
    if pages:
        st.success("✅ Source PDF Loaded")
        st.caption(f"Pages: {len(pages)} | Cleaned: {len(cleaned_pages)}")
    else:
        st.error("❌ PDF Load Failed")
        st.caption("Check `data/document.pdf` path.")
        
    if raw_results:
        st.success("✅ Raw Results Loaded")
        st.caption(f"Runs recorded: {len(raw_results)}")
    else:
        st.warning("⚠️ No Raw Results")
        
    if ragas_scores is not None:
        st.success("✅ RAGAS Scores Loaded")
    else:
        st.info("ℹ️ RAGAS Leaderboard Offline")
        
    st.divider()
    
    # Live LLM Toggle
    st.subheader("LLM Sandbox Options")
    run_live_llm = st.toggle("Enable Live LLM Generation", value=False, 
                             help="Attempts to query local Ollama (Llama 3.2) for custom queries. Requires Ollama server running.")
    
    # Retrieve top K slider
    top_k = st.slider("Retrieve Chunks (k)", min_value=1, max_value=10, value=5, 
                      help="Number of retrieved chunks for interactive queries.")
    
    st.divider()
    st.caption("Designed for Advanced Agentic Chunking Benchmarking Lab (2026)")

# ── HEADER BANNER ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-container">
    <div class="header-title">RAG Chunking Benchmark Laboratory</div>
    <div class="header-subtitle">
        Analyze, explore, and evaluate the performance of 5 core document chunking methodologies. 
        Compare precision, semantic alignment, latency, and LLM output quality side-by-side on the 
        <i>Production-Grade RAG Masterclass</i> dataset.
    </div>
</div>
""", unsafe_allow_html=True)

# ── METRIC STATS ──
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Chunking Strategies</div></div>', unsafe_allow_html=True)
with m2:
    page_count = len(cleaned_pages) if cleaned_pages else 0
    st.markdown(f'<div class="metric-card"><div class="metric-value">{page_count}</div><div class="metric-label">Cleaned Pages</div></div>', unsafe_allow_html=True)
with m3:
    char_count = sum(len(p.page_content) for p in cleaned_pages) if cleaned_pages else 0
    st.markdown(f'<div class="metric-card"><div class="metric-value">{char_count:,}</div><div class="metric-label">Total Characters</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-value">all-MiniLM-L6-v2</div><div class="metric-label">Embedding Model</div></div>', unsafe_allow_html=True)
with m5:
    eval_q = len(QUESTIONS)
    st.markdown(f'<div class="metric-card"><div class="metric-value">{eval_q}</div><div class="metric-label">Eval Questions</div></div>', unsafe_allow_html=True)

st.write("")

# ── TABS CREATION ─────────────────────────────────────────────────────────────

tab_scores, tab_playground, tab_chunks, tab_doc = st.tabs([
    "🏆 Evaluation Leaderboard", 
    "🔮 RAG Sandbox Playground", 
    "📂 Chunk Explorer & Analytics", 
    "📄 Source Document & Questions"
])

# ── TAB 1: EVALUATION LEADERBOARD ─────────────────────────────────────────────

with tab_scores:
    st.subheader("Performance & RAGAS Leaderboard Comparison")
    st.write(
        "RAGAS (Retrieval Augmented Generation Assessment) scores evaluate how faithful the generated answers are "
        "to the retrieved contexts, the relevancy of the answer to the prompt, and the quality of the retrieval phase."
    )
    
    # 1. Main Leaderboard Table
    if ragas_scores is not None:
        # Standardize strategy column and sort
        df_display = ragas_scores.copy()
        df_display["Strategy Name"] = df_display["strategy"].map(STRATEGY_LABELS)
        
        # Reorder columns for optimal view
        cols = ["Strategy Name", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg_latency", "avg_tokens"]
        df_display = df_display[cols].rename(columns={
            "faithfulness": "Faithfulness Score (Hallucination ⬇️)",
            "answer_relevancy": "Answer Relevancy Score",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "avg_latency": "Avg Latency (s)",
            "avg_tokens": "Avg Token Count"
        })
        
        # Render a beautifully styled pandas dataframe
        st.dataframe(
            df_display.style.highlight_max(subset=df_display.columns[1:5], color="#1e3a5f")
                            .highlight_min(subset=["Avg Latency (s)", "Avg Token Count"], color="#1e3d30"),
            use_container_width=True,
            hide_index=True
        )
        
        # Plotly Charts
        st.write("")
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("RAGAS Accuracy Metrics (Higher is Better)")
            # Melt the metrics for grouped plotting
            melted_df = ragas_scores.melt(
                id_vars=["strategy"], 
                value_vars=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                var_name="Metric", 
                value_name="Score"
            )
            melted_df["Strategy Name"] = melted_df["strategy"].map(STRATEGY_LABELS)
            melted_df["Metric"] = melted_df["Metric"].str.replace("_", " ").str.title()
            
            fig = px.bar(
                melted_df, 
                x="Strategy Name", 
                y="Score", 
                color="Metric", 
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_dark"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Average RAG Latency vs. Average Tokens")
            fig = px.scatter(
                ragas_scores, 
                x="avg_tokens", 
                y="avg_latency", 
                text=ragas_scores["strategy"].map(STRATEGY_LABELS),
                size="avg_tokens",
                color="strategy",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={"avg_tokens": "Average Tokens (Prompt + Response)", "avg_latency": "Average Latency (seconds)"},
                template="plotly_dark"
            )
            fig.update_traces(textposition='top center', marker=dict(sizeref=5, line=dict(width=1, color='white')))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("💡 **RAGAS Evaluation Leaderboard is currently offline.**")
        st.markdown(
            """
            To generate full RAGAS evaluation scores, run the evaluator on your machine:
            ```bash
            python evaluator/ragas_eval.py
            ```
            Once finished, this tab will display detailed faithfulness, relevance, precision, and recall scores.
            """
        )
        
        # Display latency and tokens computed from raw_results.json
        if raw_results:
            st.subheader("Benchmark Performance (Derived from Raw Results)")
            df_raw = pd.DataFrame(raw_results)
            df_agg = df_raw.groupby('strategy').agg(
                avg_latency=('latency', 'mean'),
                avg_tokens=('num_tokens', 'mean'),
                num_queries=('question', 'count')
            ).reset_index()
            df_agg["Strategy Name"] = df_agg["strategy"].map(STRATEGY_LABELS)
            
            st.dataframe(
                df_agg[["Strategy Name", "avg_latency", "avg_tokens", "num_queries"]].rename(columns={
                    "avg_latency": "Avg Latency (s)",
                    "avg_tokens": "Avg Token Count",
                    "num_queries": "Queries Run"
                }).style.highlight_min(subset=["Avg Latency (s)"], color="#1e3d30"),
                use_container_width=True,
                hide_index=True
            )
            
            # Show simple latency plot
            fig = px.bar(
                df_agg,
                x="Strategy Name",
                y="avg_latency",
                color="Strategy Name",
                title="Average RAG Latency by Chunking Strategy (seconds)",
                labels={"avg_latency": "Latency (s)"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_dark"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No performance metrics or run results found in `results/raw_results.json`. Run `query_runner.py` first.")

    # Details about metrics
    st.divider()
    st.subheader("📚 Metric Glossaries")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        **Hallucination Protection (Faithfulness)**
        Measures if the generated answer is strictly grounded in the retrieved documents. A score of **1.0** indicates that every claim in the answer is fully supported by the text chunk context. 
        """)
    with g2:
        st.markdown("""
        **Context Recall / Precision**
        *Recall* evaluates if all critical details from the ground truth answer are present in the retrieved chunks. *Precision* checks if the retrieved chunks are noise-free and relevant.
        """)
    with g3:
        st.markdown("""
        **Efficiency (Latency & Tokens)**
        Highlights the trade-off of chunking density. Larger chunks (or semantic boundaries) provide better context for generation but increase token counts and local LLM latency.
        """)

# ── TAB 2: INTERACTIVE RAG SANDBOX PLAYGROUND ─────────────────────────────────

with tab_playground:
    st.subheader("🔮 RAG Pipeline Sandbox")
    st.write(
        "Compare retrieval strategies side-by-side. Choose one of the evaluation questions "
        "to pull pre-computed answers immediately, or write a custom question to test live."
    )
    
    # Select question mode
    query_mode = st.radio("Query Source", ["Select from Evaluation Golden Set", "Write Custom Query"], horizontal=True)
    
    selected_question = ""
    custom_query = ""
    ground_truth_ans = ""
    
    if query_mode == "Select from Evaluation Golden Set":
        # Load questions 1-10 which are evaluated in raw_results.json
        q_options = QUESTIONS[:10]
        q_idx = st.selectbox("Choose a Question", range(len(q_options)), format_func=lambda x: f"Q{x+1}: {q_options[x]}")
        selected_question = q_options[q_idx]
        ground_truth_ans = GROUND_TRUTHS[q_idx]
        
        st.markdown(f"**💡 Golden Ground Truth:**\n> {ground_truth_ans}")
    else:
        custom_query = st.text_input("Enter your custom query:", placeholder="e.g., What parameters define the HNSW algorithm?")
        if custom_query:
            selected_question = custom_query
            
    # Checkbox to choose strategies to display
    st.write("")
    selected_strategies = st.multiselect(
        "Select strategies to compare:",
        options=STRATEGY_NAMES,
        default=STRATEGY_NAMES,
        format_func=lambda x: STRATEGY_LABELS.get(x, x)
    )
    
    if selected_question:
        st.write("")
        st.subheader("Results Comparison")
        
        # Setup columns for the selected strategies
        cols = st.columns(len(selected_strategies)) if selected_strategies else []
        
        # Load embedding model for live retrieve if needed
        model = None
        if query_mode == "Write Custom Query" or run_live_llm:
            model = get_cached_embedding_model()
            
        for i, strategy_name in enumerate(selected_strategies):
            with cols[i]:
                # Print strategy header
                st.markdown(get_strategy_badge_html(strategy_name), unsafe_allow_html=True)
                
                # Check if we should load pre-computed results
                precomputed_result = None
                if query_mode == "Select from Evaluation Golden Set" and raw_results:
                    for row in raw_results:
                        if row["strategy"] == strategy_name and row["question"] == selected_question:
                            precomputed_result = row
                            break
                            
                answer_text = ""
                chunks_retrieved = []
                latency = None
                tokens_count = None
                
                if precomputed_result and not run_live_llm:
                    # Show pre-computed
                    answer_text = precomputed_result["answer"]
                    chunks_retrieved = precomputed_result["contexts"]
                    latency = precomputed_result.get("latency")
                    tokens_count = precomputed_result.get("num_tokens")
                    st.caption(f"⏱️ Pre-computed | Latency: {latency}s | Tokens: {tokens_count}")
                else:
                    # Run live query retrieve
                    if model is not None:
                        try:
                            # Load Chroma db
                            vs = load_vectorstore(strategy_name, model)
                            # Retrieve top K
                            retrieved_docs = vs.similarity_search(selected_question, k=top_k)
                            chunks_retrieved = [doc.page_content for doc in retrieved_docs]
                            metadatas = [doc.metadata for doc in retrieved_docs]
                            
                            # Live LLM generate if toggle enabled
                            if run_live_llm:
                                with st.spinner("Generating answer with Ollama Llama 3.2..."):
                                    try:
                                        from query_runner import get_llm
                                        import time
                                        llm = get_llm()
                                        context_str = "\n\n---\n\n".join(chunks_retrieved)
                                        prompt = RAG_PROMPT.format(context=context_str, question=selected_question)
                                        
                                        start_t = time.time()
                                        answer_text = llm.invoke(prompt).strip()
                                        latency = round(time.time() - start_t, 2)
                                        tokens_count = (len(prompt) + len(answer_text)) // 4
                                        st.caption(f"⚡ Live Generation | Latency: {latency}s | Est. Tokens: {tokens_count}")
                                    except Exception as ex:
                                        answer_text = f"⚠️ LLM Error: {ex}. Ensure Ollama is running."
                                        st.caption("⚡ Live Retrieve (LLM Offline)")
                            else:
                                answer_text = "💡 *Live generation disabled. Enable in the sidebar to generate answers.*"
                                st.caption("⚡ Live Retrieval Mode")
                        except Exception as e:
                            answer_text = f"❌ Error loading vector store: {e}"
                    else:
                        answer_text = "❌ Embedding model failed to load. Cannot run retrieval."
                
                # Output box for answer
                st.markdown(f'<div class="playground-answer-box">{answer_text}</div>', unsafe_allow_html=True)
                
                # Expandable chunks list
                with st.expander(f"📚 Retrieved Chunks ({len(chunks_retrieved)})"):
                    for c_idx, chunk_content in enumerate(chunks_retrieved):
                        # Extract page info if precomputed or live
                        page_info = "N/A"
                        parent_text = None
                        
                        # Try to resolve page number and parent details
                        if precomputed_result and not run_live_llm:
                            # Note: in pre-computed results, contexts are just strings.
                            pass
                        elif model is not None:
                            try:
                                doc_meta = retrieved_docs[c_idx].metadata
                                page_info = doc_meta.get("page_number", doc_meta.get("page", "N/A"))
                                
                                # Reconstruct parent for Hierarchical
                                if strategy_name == "hierarchical_chunks":
                                    parent_id = doc_meta.get("parent_id")
                                    if parent_id and parent_id.startswith("parent_"):
                                        p_num = int(parent_id.split("_")[1])
                                        parents = get_hierarchical_parents()
                                        if parents and p_num < len(parents):
                                            parent_text = parents[p_num].page_content
                            except Exception:
                                pass
                                
                        st.markdown(f"**Chunk #{c_idx+1} (Page: {page_info})**")
                        st.markdown(f'<div class="playground-chunk-box">{chunk_content}</div>', unsafe_allow_html=True)
                        
                        # If hierarchical parent details exist
                        if parent_text:
                            with st.expander(f"🔍 Show Parent Context (Size: {len(parent_text)} chars)"):
                                st.markdown(f'<div class="playground-chunk-box" style="border-color:#fbbf24;">{parent_text}</div>', unsafe_allow_html=True)

# ── TAB 3: CHUNK EXPLORER & ANALYTICS ─────────────────────────────────────────

with tab_chunks:
    st.subheader("📂 Vector Store Explorer & Analytics")
    st.write(
        "Inspect the contents of the Chroma DB collection for each strategy. "
        "Observe the differences in chunk sizes and count distributions."
    )
    
    # Strategy selector
    selected_explore_strategy = st.selectbox(
        "Choose strategy to explore:",
        options=STRATEGY_NAMES,
        format_func=lambda x: STRATEGY_LABELS.get(x, x)
    )
    
    # Load chunks for selected strategy
    @st.cache_data
    def get_strategy_chunks_list(strategy_name):
        try:
            # We load the embedding model to initialize Chroma
            # (Chroma requires the embedding function to fetch and index documents correctly)
            embeddings = get_cached_embedding_model()
            if not embeddings:
                return []
            vs = load_vectorstore(strategy_name, embeddings)
            data = vs.get()
            
            chunks = []
            for i in range(len(data['documents'])):
                doc_text = data['documents'][i]
                meta = data['metadatas'][i] if data['metadatas'] else {}
                chunk_id = data['ids'][i]
                
                # Clean clean metadata page fields
                page = meta.get("page_number", meta.get("page", meta.get("source_document_index", "N/A")))
                
                chunks.append({
                    "ID": chunk_id,
                    "Text Preview": doc_text[:150] + "..." if len(doc_text) > 150 else doc_text,
                    "Full Text": doc_text,
                    "Length (Chars)": len(doc_text),
                    "Word Count": len(doc_text.split()),
                    "Page": page,
                    "Metadata": str(meta)
                })
            return chunks
        except Exception as e:
            return str(e)
            
    with st.spinner(f"Loading chunks for '{selected_explore_strategy}' from vectorstore..."):
        chunks_data = get_strategy_chunks_list(selected_explore_strategy)
        
    if isinstance(chunks_data, str):
        st.error(f"Error reading from vector store: {chunks_data}")
        st.info("Please run `indexer.py` to create the vectorstores first.")
    elif len(chunks_data) == 0:
        st.warning("No chunks found in this vectorstore collection.")
    else:
        df_chunks = pd.DataFrame(chunks_data)
        
        # 1. Summary Statistics for active strategy
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Chunks", len(df_chunks))
        with s2:
            st.metric("Average Chunk Length", f"{int(df_chunks['Length (Chars)'].mean())} chars")
        with s3:
            st.metric("Min Chunk Length", f"{df_chunks['Length (Chars)'].min()} chars")
        with s4:
            st.metric("Max Chunk Length", f"{df_chunks['Length (Chars)'].max()} chars")
            
        # 2. Search & Browse Table
        st.write("")
        search_query = st.text_input("🔍 Search chunks by text content:", "")
        filtered_df = df_chunks
        if search_query:
            filtered_df = df_chunks[df_chunks["Full Text"].str.contains(search_query, case=False)]
            st.caption(f"Showing {len(filtered_df)} matches of {len(df_chunks)} total chunks.")
            
        # Columns to display in browse table
        display_cols = ["ID", "Page", "Length (Chars)", "Word Count", "Text Preview"]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
        
        # Inspect full text of a chunk
        st.write("")
        chunk_to_view = st.selectbox(
            "Select Chunk ID to read full text:",
            options=filtered_df["ID"].tolist()
        )
        if chunk_to_view:
            selected_row = filtered_df[filtered_df["ID"] == chunk_to_view].iloc[0]
            st.markdown("**Full Chunk Text:**")
            st.info(selected_row["Full Text"])
            st.markdown(f"**Chunk Metadata:** `{selected_row['Metadata']}`")
            
        st.divider()
        
        # 3. Size Distribution Plot
        st.subheader("Size Distribution Histogram")
        fig_hist = px.histogram(
            df_chunks, 
            x="Length (Chars)", 
            nbins=30, 
            title=f"Chunk Character Length Distribution - {STRATEGY_LABELS[selected_explore_strategy]}",
            labels={"Length (Chars)": "Character Count", "count": "Number of Chunks"},
            color_discrete_sequence=["#a855f7"],
            template="plotly_dark"
        )
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 4. Strategy comparison plots
        st.subheader("Global Chunk Comparisons")
        
        # We can aggregate info for all strategies if possible
        agg_data = []
        for name in STRATEGY_NAMES:
            c_list = get_strategy_chunks_list(name)
            if isinstance(c_list, list) and len(c_list) > 0:
                lens = [c["Length (Chars)"] for c in c_list]
                agg_data.append({
                    "strategy": name,
                    "Strategy": STRATEGY_LABELS[name],
                    "Total Chunks": len(lens),
                    "Avg Size (Chars)": int(np.mean(lens))
                })
                
        if agg_data:
            df_agg = pd.DataFrame(agg_data)
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_comp1 = px.bar(
                    df_agg, 
                    x="Strategy", 
                    y="Total Chunks", 
                    title="Total Chunks Generated",
                    color="Strategy",
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    template="plotly_dark"
                )
                fig_comp1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_comp1, use_container_width=True)
            with col_chart2:
                fig_comp2 = px.bar(
                    df_agg, 
                    x="Strategy", 
                    y="Avg Size (Chars)", 
                    title="Average Chunk Length (Characters)",
                    color="Strategy",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    template="plotly_dark"
                )
                fig_comp2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_comp2, use_container_width=True)

# ── TAB 4: SOURCE DOCUMENT & QUESTIONS ────────────────────────────────────────

with tab_doc:
    st.subheader("📄 Document & Evaluation Suite Inspector")
    
    d1, d2 = st.columns([2, 1])
    
    with d1:
        st.subheader("Document Viewer")
        if cleaned_pages:
            total_pdf_pages = len(cleaned_pages)
            page_num = st.slider("Select Page to View:", min_value=1, max_value=total_pdf_pages, value=1)
            
            st.markdown(f"**Page {page_num} of {total_pdf_pages}**")
            st.markdown(f'<div class="doc-page-content">{cleaned_pages[page_num-1].page_content}</div>', unsafe_allow_html=True)
            
            # Show metadata
            st.write("")
            st.markdown(f"**Page Metadata:** `{cleaned_pages[page_num-1].metadata}`")
        else:
            st.info("Source PDF text could not be loaded. Please ensure `data/document.pdf` is present and loadable.")
            
    with d2:
        st.subheader("Evaluation Set (40 Questions)")
        st.write("Browse the golden evaluation questions and ground truth responses used to test chunking RAG output.")
        
        q_search = st.text_input("🔍 Filter Questions:", "")
        
        eval_list = []
        for idx, (q, gt) in enumerate(zip(QUESTIONS, GROUND_TRUTHS)):
            if q_search.lower() in q.lower() or q_search.lower() in gt.lower():
                eval_list.append({"Index": idx+1, "Question": q, "Ground Truth": gt})
                
        for item in eval_list[:15]: # Cap at 15 for readability
            with st.expander(f"Q{item['Index']}: {item['Question']}"):
                st.write(f"**Ground Truth Answer:**\n{item['Ground Truth']}")
                
        if len(eval_list) > 15:
            st.caption(f"*Showing 15 of {len(eval_list)} filtered questions. Use search to find specific topics.*")
