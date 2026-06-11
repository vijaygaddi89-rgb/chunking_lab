# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path
import os
import re

# Import core elements from the workspace project
from indexer import load_vectorstore, get_embedding_model, STRATEGY_NAMES
from data.questions import QUESTIONS, GROUND_TRUTHS
from loader import load_document, clean_documents, get_full_text
from query_runner import RAG_PROMPT

# Set page config
st.set_page_config(
    page_title="RAG Chunking Strategies Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Reset and global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid #312e81;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(to right, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 900px;
        line-height: 1.6;
    }

    /* Cards */
    .premium-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    
    /* Strategy Explanatory Badges */
    .strategy-badge {
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-fixed { background-color: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-recursive { background-color: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-semantic { background-color: rgba(139, 92, 246, 0.1); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-hierarchical { background-color: rgba(245, 158, 11, 0.1); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-parent { background-color: rgba(236, 72,  pink, 0.1); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }

    /* Visualizer Output styling */
    .visualizer-container {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
    }
    
    .visual-chunk {
        display: inline;
        padding: 0.2rem 0.4rem;
        margin: 0.1rem;
        border-radius: 4px;
        cursor: help;
        transition: filter 0.2s;
    }
    .visual-chunk:hover {
        filter: brightness(1.2);
    }
    
    /* Interactive Sandbox Answers */
    .sandbox-answer-box {
        background-color: #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #f1f5f9;
    }
    
    .sandbox-chunk-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }

    /* Metric numbers */
    .metric-num {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ── CACHED DATA LOADING ────────────────────────────────────────────────────────

@st.cache_resource
def get_cached_embedding_model():
    """Cache embedding model loading."""
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

@st.cache_resource
def get_cached_semantic_chunker():
    """Cache the Semantic Chunker to avoid repeated model reloads."""
    from chunkers.semantic import SemanticChunker
    # Instantiates the models once on CPU/GPU
    return SemanticChunker()

# Load primary resources
pages, cleaned_pages, full_text = get_cached_pdf_data()
raw_results = load_raw_results()
ragas_scores = load_ragas_scores()

# Strategy labels & UI badges mapping
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

# ── HEADER BANNER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-title">RAG Chunking Explorer & Laboratory</div>
    <div class="header-subtitle">
        Chunking is the single most critical configuration in RAG pipelines. 
        This laboratory lets you compare 5 indexing strategies side-by-side. 
        Visualize chunk boundaries, test query retrieval performance, and review evaluation scores.
    </div>
</div>
""", unsafe_allow_html=True)

# Brief Info Panel
col_info_1, col_info_2, col_info_3, col_info_4 = st.columns(4)
with col_info_1:
    st.markdown(f'<div class="premium-card"><div class="metric-num">5</div><div class="metric-lbl">Strategies Indexed</div></div>', unsafe_allow_html=True)
with col_info_2:
    p_len = len(cleaned_pages) if cleaned_pages else 0
    st.markdown(f'<div class="premium-card"><div class="metric-num">{p_len}</div><div class="metric-lbl">Cleaned Document Pages</div></div>', unsafe_allow_html=True)
with col_info_3:
    st.markdown('<div class="premium-card"><div class="metric-num">all-MiniLM</div><div class="metric-lbl">Embedding Model</div></div>', unsafe_allow_html=True)
with col_info_4:
    q_len = len(QUESTIONS) if QUESTIONS else 0
    st.markdown(f'<div class="premium-card"><div class="metric-num">{q_len}</div><div class="metric-lbl">Benchmarked Questions</div></div>', unsafe_allow_html=True)

# ── THREE COHERENT TABS ───────────────────────────────────────────────────────
tab_visualizer, tab_sandbox, tab_leaderboard = st.tabs([
    "🧠 Interactive Chunker Visualizer",
    "🔮 RAG Sandbox Playground",
    "🏆 Performance Leaderboard"
])

# ── TAB 1: INTERACTIVE CHUNKER VISUALIZER ─────────────────────────────────────
with tab_visualizer:
    st.subheader("Chunker Simulator & Visualizer")
    st.write(
        "See exactly how each chunking strategy splits raw text. Select a strategy, adjust its parameters, "
        "and click **Run Simulation** to see color-coded output chunks."
    )
    
    # Visualizer Layout: Left = Config + Inputs, Right = Rendered visual chunks
    c_left, c_right = st.columns([1, 1.3])
    
    with c_left:
        st.markdown("##### 1. Select Chunker & Parameters")
        viz_strategy = st.selectbox(
            "Chunking Strategy", 
            options=STRATEGY_NAMES, 
            format_func=lambda x: STRATEGY_LABELS.get(x, x),
            key="viz_strat_select"
        )
        
        # Strategy descriptions for user understanding
        strategy_explanations = {
            "fixed_size": "Splits text at strict character count intervals. Fast, simple, but often breaks sentences or paragraphs in half.",
            "recursive_split": "Splits text hierarchically using a list of separators (like paragraphs, lines, spaces). Keeps sentences and paragraphs whole whenever possible. (Recommended Default)",
            "semantic_chunks": "Analyzes cosine similarity between adjacent sentences. Splits when topic similarity drops below a threshold. Captures coherent semantic themes.",
            "hierarchical_chunks": "Splits text into large parent chunks for LLM context, and nests smaller child chunks inside them for precise retrieval.",
            "parent_doc_retriever": "Indexes small child chunks to enable fast semantic matching, but retrieves and sends the larger parent chunk context to the LLM."
        }
        st.caption(f"💡 **About {STRATEGY_LABELS[viz_strategy]}:** {strategy_explanations[viz_strategy]}")
        
        st.write("")
        
        # Dynamic configuration inputs based on Strategy
        if viz_strategy in ["fixed_size", "recursive_split"]:
            viz_size = st.slider("Chunk Size (characters)", min_value=100, max_value=2000, value=500, step=50)
            viz_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=400, value=80, step=10)
        elif viz_strategy == "semantic_chunks":
            viz_sim = st.slider("Similarity Split Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05, 
                                help="Split chunk when similarity drop falls below this threshold. Lower values result in larger chunks.")
        else: # Hierarchical or Parent Doc
            viz_parent = st.slider("Parent Chunk Size", min_value=1000, max_value=3000, value=1500, step=100)
            viz_child = st.slider("Child Chunk Size", min_value=100, max_value=500, value=300, step=20)
            
        st.write("")
        st.markdown("##### 2. Text Snippet to Chunk")
        
        # Standard default text snippet
        default_snippet = (
            "Retrieval-Augmented Generation (RAG) is a highly effective architecture that connects LLMs to "
            "private databases. By fetching relevant passages, RAG anchors generated answers to factual reference sources.\n\n"
            "The performance of any RAG system relies heavily on chunking. If chunks are too small, the system loses context. "
            "For example, a sentence describing a statistic might lose the context of which year it belongs to.\n\n"
            "If chunks are too large, they dilute the specific information needed, filling the LLM context window with noise. "
            "Choosing the right boundary splitter (fixed, recursive, semantic, or hierarchical) directly balances "
            "retrieval precision against context richness."
        )
        
        viz_text = st.text_area(
            "Paste custom text or use this default RAG overview:",
            value=default_snippet,
            height=200
        )
        
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)
        
    with c_right:
        st.markdown("##### 3. Simulation Results")
        
        if run_btn or "visualizer_run" not in st.session_state:
            st.session_state["visualizer_run"] = True
            
            # Perform splitting based on selection
            chunks_list = []
            
            try:
                if viz_strategy == "fixed_size":
                    from chunkers.fixed_size import FixedSizeChunker
                    chunker = FixedSizeChunker(chunk_size=viz_size, chunk_overlap=viz_overlap)
                    chunks_list = chunker.split_text(viz_text)
                    
                elif viz_strategy == "recursive_split":
                    from chunkers.recursive import RecursiveChunker
                    chunker = RecursiveChunker(chunk_size=viz_size, chunk_overlap=viz_overlap)
                    chunks_list = chunker.split_text(viz_text)
                    
                elif viz_strategy == "semantic_chunks":
                    # Use cached model to avoid reloading transformer
                    chunker = get_cached_semantic_chunker()
                    chunker.similarity_threshold = viz_sim
                    chunks_list = chunker.split_text(viz_text)
                    
                else: # hierarchical_chunks / parent_doc_retriever
                    from chunkers.hierarchical import HierarchicalChunker
                    from langchain_core.documents import Document
                    chunker = HierarchicalChunker(parent_chunk_size=viz_parent, child_chunk_size=viz_child)
                    children, parents = chunker.split_documents([Document(page_content=viz_text)])
                    
                    # Store nested structure
                    chunks_list = {
                        "parents": [p.page_content for p in parents],
                        "children": [{"text": c.page_content, "parent_id": c.metadata["parent_id"]} for c in children]
                    }
            except Exception as ex:
                st.error(f"Failed to split text: {ex}")
                chunks_list = []
                
            st.session_state["viz_results"] = chunks_list
            st.session_state["viz_type"] = "nested" if viz_strategy in ["hierarchical_chunks", "parent_doc_retriever"] else "flat"
            
        # Display output
        if "viz_results" in st.session_state:
            results = st.session_state["viz_results"]
            viz_type = st.session_state["viz_type"]
            
            # Palette of pastel background colors for easy differentiation
            colors = [
                ("rgba(59, 130, 246, 0.12)", "#60a5fa", "Blue"),
                ("rgba(16, 185, 129, 0.12)", "#34d399", "Green"),
                ("rgba(139, 92, 246, 0.12)", "#a78bfa", "Purple"),
                ("rgba(245, 158, 11, 0.12)", "#fbbf24", "Amber"),
                ("rgba(236, 72, 153, 0.12)", "#f472b6", "Pink")
            ]
            
            if viz_type == "flat":
                if not results:
                    st.info("No chunks generated. Make sure input text is not empty.")
                else:
                    # Chunker stats
                    s_col1, s_col2, s_col3 = st.columns(3)
                    lengths = [len(c) for c in results]
                    s_col1.metric("Total Chunks", len(results))
                    s_col2.metric("Avg Chunk Size", f"{int(np.mean(lengths))} chars")
                    s_col3.metric("Size Std Dev", f"{int(np.std(lengths))} chars")
                    
                    # Highlighted inline visualizer
                    st.write("")
                    st.markdown("**Visual Boundary Map:**")
                    
                    html_str = '<div class="visualizer-container">'
                    for i, chunk in enumerate(results):
                        bg, text_col, c_name = colors[i % len(colors)]
                        # Clean special characters for safe html
                        safe_text = chunk.replace("\n", " ↵ ")
                        html_str += f'<span class="visual-chunk" style="background-color: {bg}; border-bottom: 2px solid {text_col};" title="Chunk {i+1} ({c_name} Chunk)">{safe_text}</span> '
                    html_str += '</div>'
                    
                    st.markdown(html_str, unsafe_allow_html=True)
                    st.caption("ℹ️ Hover over the colored text areas to see corresponding chunk boundaries.")
                    
                    # Details breakdown list
                    st.write("")
                    with st.expander("Show Chunks Breakdown List"):
                        for i, chunk in enumerate(results):
                            bg, text_col, _ = colors[i % len(colors)]
                            st.markdown(f"**Chunk #{i+1}** ({len(chunk)} characters)")
                            st.markdown(f'<div style="background-color: {bg}; border-left: 4px solid {text_col}; padding: 10px; border-radius: 4px; margin-bottom: 10px; font-size: 0.9rem;">{chunk}</div>', unsafe_allow_html=True)
            
            else: # Nested parents/children
                parents = results.get("parents", [])
                children = results.get("children", [])
                
                s_col1, s_col2 = st.columns(2)
                s_col1.metric("Parent Chunks", len(parents))
                s_col2.metric("Child Chunks", len(children))
                
                st.write("")
                st.markdown("**Hierarchical Tree Map:**")
                
                for p_idx, p_text in enumerate(parents):
                    p_id = f"parent_{p_idx}"
                    bg, text_col, _ = colors[p_idx % len(colors)]
                    
                    # Render parent container
                    st.markdown(f"📦 **Parent Chunk #{p_idx+1}** (Size: {len(p_text)} chars)")
                    
                    # Filter children belonging to this parent
                    p_children = [c for c in children if c["parent_id"] == p_id]
                    
                    child_html = ""
                    for c_idx, c in enumerate(p_children):
                        c_bg = "rgba(255, 255, 255, 0.05)"
                        child_html += f'<div style="background-color: {c_bg}; border-left: 3px dashed {text_col}; padding: 8px; margin: 6px 0; border-radius: 4px; font-size: 0.85rem;">👶 <b>Child #{c_idx+1}:</b> {c["text"]}</div>'
                    
                    parent_container_html = f"""
                    <div style="background-color: {bg}; border: 1px solid {text_col}; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
                        <div style="font-size: 0.9rem; font-style: italic; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 6px; margin-bottom: 8px;">{p_text}</div>
                        {child_html if child_html else '<div style="font-size: 0.8rem; color: #64748b;">No children linked.</div>'}
                    </div>
                    """
                    st.markdown(parent_container_html, unsafe_allow_html=True)

# ── TAB 2: INTERACTIVE RAG SANDBOX PLAYGROUND ─────────────────────────────────
with tab_sandbox:
    st.subheader("🔮 RAG Pipeline Sandbox")
    st.write(
        "Compare retrieval strategies side-by-side. Choose one of the pre-computed "
        "golden questions, or input custom questions to run live retrieval."
    )
    
    # Toggle choices
    query_mode = st.radio("Query Source", ["Select from Evaluation Golden Set", "Write Custom Query"], horizontal=True, key="sandbox_mode")
    
    selected_question = ""
    ground_truth_ans = ""
    
    if query_mode == "Select from Evaluation Golden Set":
        # Load questions dropdown
        q_idx = st.selectbox("Choose a Question", range(len(QUESTIONS)), format_func=lambda x: f"Q{x+1}: {QUESTIONS[x]}", key="golden_q_select")
        selected_question = QUESTIONS[q_idx]
        ground_truth_ans = GROUND_TRUTHS[q_idx]
        
        st.markdown(f"🎯 **Expected Answer (Ground Truth):**")
        st.markdown(f"> *{ground_truth_ans}*")
    else:
        selected_question = st.text_input("Enter your custom query:", placeholder="e.g., What are the parameters for the HNSW algorithm?", key="custom_q_input")
        
    st.divider()
    
    # Live LLM Toggle (only active if Ollama is accessible)
    c_live1, c_live2 = st.columns([1, 2])
    with c_live1:
        run_live_llm = st.toggle("Enable Live LLM Generation", value=False, help="Requires local Ollama running with Llama 3.2.")
    with c_live2:
        top_k = st.slider("Retrieve Chunks (k)", min_value=1, max_value=8, value=5)
        
    st.write("")
    selected_strategies = st.multiselect(
        "Select strategies to compare:",
        options=STRATEGY_NAMES,
        default=STRATEGY_NAMES,
        format_func=lambda x: STRATEGY_LABELS.get(x, x),
        key="sandbox_strat_compare"
    )
    
    if selected_question:
        st.write("")
        st.markdown(f"#### Retrieval Results for: *\"{selected_question}\"*")
        
        # Setup columns for the selected strategies
        cols = st.columns(len(selected_strategies)) if selected_strategies else []
        
        # Load embedding model if doing custom search or live LLM
        model = None
        if query_mode == "Write Custom Query" or run_live_llm:
            model = get_cached_embedding_model()
            
        for idx, strategy_name in enumerate(selected_strategies):
            with cols[idx]:
                st.markdown(get_strategy_badge_html(strategy_name), unsafe_allow_html=True)
                
                # Check for precomputed results first
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
                    answer_text = precomputed_result["answer"]
                    chunks_retrieved = precomputed_result["contexts"]
                    latency = precomputed_result.get("latency")
                    tokens_count = precomputed_result.get("num_tokens")
                    st.caption(f"⏱️ Pre-computed | Latency: {latency}s | Est. Tokens: {tokens_count}")
                else:
                    # Live query retrieve
                    if model is not None:
                        try:
                            # Load Chroma db
                            vs = load_vectorstore(strategy_name, model)
                            # Retrieve top K
                            retrieved_docs = vs.similarity_search(selected_question, k=top_k)
                            chunks_retrieved = [doc.page_content for doc in retrieved_docs]
                            
                            # Live LLM generate if toggle enabled
                            if run_live_llm:
                                with st.spinner("Generating answer..."):
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
                                answer_text = "💡 *Live generation disabled. Enable in sandbox options to generate responses.*"
                                st.caption("⚡ Live Retrieval Mode")
                        except Exception as e:
                            answer_text = f"❌ Vector store error: {e}"
                    else:
                        answer_text = "❌ Embedding model failed to load. Ensure indexer.py was run."
                
                # Render response box
                st.markdown(f'<div class="sandbox-answer-box">{answer_text}</div>', unsafe_allow_html=True)
                
                # Expandable retrieved chunks
                with st.expander(f"Show Retrieved Passages ({len(chunks_retrieved)})"):
                    for c_idx, chunk_content in enumerate(chunks_retrieved):
                        # Attempt parent context retrieval for Hierarchical strategy
                        parent_text = None
                        if strategy_name == "hierarchical_chunks" and model is not None:
                            try:
                                doc_meta = retrieved_docs[c_idx].metadata
                                parent_id = doc_meta.get("parent_id")
                                if parent_id and parent_id.startswith("parent_"):
                                    p_num = int(parent_id.split("_")[1])
                                    parents_list = get_hierarchical_parents()
                                    if parents_list and p_num < len(parents_list):
                                        parent_text = parents_list[p_num].page_content
                            except:
                                pass
                                
                        st.markdown(f"**Passage #{c_idx+1}**")
                        st.markdown(f'<div class="sandbox-chunk-box">{chunk_content}</div>', unsafe_allow_html=True)
                        
                        if parent_text:
                            with st.expander(f"🔍 Parent context detail"):
                                st.markdown(f'<div class="sandbox-chunk-box" style="border-color:#fbbf24;">{parent_text}</div>', unsafe_allow_html=True)

# ── TAB 3: PERFORMANCE LEADERBOARD ────────────────────────────────────────────
with tab_leaderboard:
    st.subheader("Performance & Quality Benchmarks")
    st.write(
        "Compare the evaluation scores of each chunking strategy. Accuracy is judged using RAGAS criteria "
        "and runtime efficiency is measured based on latency and tokens."
    )
    
    # 1. Main Leaderboard Table
    if ragas_scores is not None:
        df_display = ragas_scores.copy()
        df_display["Strategy"] = df_display["strategy"].map(STRATEGY_LABELS)
        
        # Standardize column headers
        cols = ["Strategy", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg_latency", "avg_tokens"]
        df_display = df_display[cols].rename(columns={
            "faithfulness": "Faithfulness (Groundedness 🎯)",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "avg_latency": "Avg Latency (s)",
            "avg_tokens": "Avg Tokens"
        })
        
        # Highlight best values
        st.dataframe(
            df_display.style.highlight_max(subset=df_display.columns[1:5], color="#1e3a5f")
                            .highlight_min(subset=["Avg Latency (s)", "Avg Tokens"], color="#1e3d30"),
            use_container_width=True,
            hide_index=True
        )
        
        # Plotly Charts
        st.write("")
        fig_cols = st.columns(2)
        
        with fig_cols[0]:
            st.markdown("##### Accuracy Comparison")
            melted_df = ragas_scores.melt(
                id_vars=["strategy"], 
                value_vars=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                var_name="Metric", 
                value_name="Score"
            )
            melted_df["Strategy"] = melted_df["strategy"].map(STRATEGY_LABELS)
            melted_df["Metric"] = melted_df["Metric"].str.replace("_", " ").str.title()
            
            fig = px.bar(
                melted_df, 
                x="Strategy", 
                y="Score", 
                color="Metric", 
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_dark"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
            
        with fig_cols[1]:
            st.markdown("##### Latency (s) vs. Prompt Size (Tokens)")
            fig = px.scatter(
                ragas_scores, 
                x="avg_tokens", 
                y="avg_latency", 
                text=ragas_scores["strategy"].map(STRATEGY_LABELS),
                size="avg_tokens",
                color="strategy",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={"avg_tokens": "Average Tokens", "avg_latency": "Average Latency (seconds)"},
                template="plotly_dark"
            )
            fig.update_traces(textposition='top center', marker=dict(sizeref=5, line=dict(width=1, color='white')))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        # Fallback to aggregated metrics from raw run results
        if raw_results:
            st.markdown("##### Aggregated Performance (Derived from Raw Run Results)")
            df_raw = pd.DataFrame(raw_results)
            df_agg = df_raw.groupby('strategy').agg(
                avg_latency=('latency', 'mean'),
                avg_tokens=('num_tokens', 'mean'),
                num_queries=('question', 'count')
            ).reset_index()
            df_agg["Strategy"] = df_agg["strategy"].map(STRATEGY_LABELS)
            
            st.dataframe(
                df_agg[["Strategy", "avg_latency", "avg_tokens", "num_queries"]].rename(columns={
                    "avg_latency": "Avg Latency (s)",
                    "avg_tokens": "Avg Token Count",
                    "num_queries": "Queries Run"
                }).style.highlight_min(subset=["Avg Latency (s)"], color="#1e3d30"),
                use_container_width=True,
                hide_index=True
            )
            
            st.write("")
            fig = px.bar(
                df_agg,
                x="Strategy",
                y="avg_latency",
                color="Strategy",
                title="Average Latency per Strategy (seconds)",
                labels={"avg_latency": "Latency (s)"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_dark"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No performance metrics or run results found in `results/raw_results.json`. Run `query_runner.py` first.")
            
    st.divider()
    
    # Explanations of critical metrics
    st.markdown("##### 📚 Evaluation Glossary")
    gl_col1, gl_col2, gl_col3 = st.columns(3)
    with gl_col1:
        st.markdown("""
        **🎯 Faithfulness**
        Measures if the generated answer is strictly grounded in the retrieved documents. A score of **1.0** indicates that every claim in the answer is fully supported by the text chunk context (no hallucinations). 
        """)
    with gl_col2:
        st.markdown("""
        **🔍 Context Recall & Precision**
        *Recall* evaluates if all critical details from the golden answer are present in the retrieved chunks. *Precision* checks if the retrieved chunks are noise-free and relevant.
        """)
    with gl_col3:
        st.markdown("""
        **⚡ Efficiency (Latency & Tokens)**
        Highlights the trade-off of chunking density. Larger chunks provide better context for generation but increase token counts and local LLM latency.
        """)
