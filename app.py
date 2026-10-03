import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import numpy as np
import faiss
import os
import requests
import pickle
import json
from sentence_transformers import SentenceTransformer

# ===========================================================================
# 1. CONFIGURATION AND GLOBAL STATE
# ===========================================================================
st.set_page_config(page_title="Enterprise AI Business Assistant", page_icon="💼", layout="wide")
st.title("Enterprise AI Business Assistant")
st.caption("RAG & Multi-Step Sales Analytics Agent")

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

# L2 Distance Threshold: Lower = more similar
MAX_DISTANCE = 1.6

BUSINESS_SYSTEM_PROMPT = """You are a Senior Business Analyst and AI Agent.
Your job is to support Product Managers with highly structured, fact-based insights.
Always format your output using this exact structure:

### 📊 Key Insights & KPIs
- [Point]

### 📈 Identified Trends
- [Point]

### ⚠️ Operational Risks & Concerns
- [Point]

### 💡 Strategic Recommended Actions
- [Point]

Be extremely concise and concrete. Only use information strictly grounded in the provided context or data.
If the information cannot be found in the context, explicitly state that."""

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Defensive Session State Initialization
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "last_df_summary" not in st.session_state:
    st.session_state.last_df_summary = None
if "raw_csv_string" not in st.session_state:
    st.session_state.raw_csv_string = None
if "has_valid_chart" not in st.session_state:
    st.session_state.has_valid_chart = False
# Results are stored in session_state (not just st.write()'d right after a
# button press) because Streamlit reruns the whole script on every
# interaction. Without this, a result shown after clicking "Run" disappears
# the moment you interact with anything else (a new question, another tab).
if "doc_qa_question" not in st.session_state:
    st.session_state.doc_qa_question = None
if "doc_qa_answer" not in st.session_state:
    st.session_state.doc_qa_answer = None
if "standard_analysis_result" not in st.session_state:
    st.session_state.standard_analysis_result = None
if "multistep_metrics" not in st.session_state:
    st.session_state.multistep_metrics = None
if "multistep_strategy" not in st.session_state:
    st.session_state.multistep_strategy = None

# ===========================================================================
# 2. CORE ENGINE & UTILITIES
# ===========================================================================
def ask_llm(prompt, temperature=0.2):
    """Robust API client with comprehensive error handling"""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Please configure Streamlit Secrets or environment variables.")
        return "Error: Missing API Key"

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "system", "content": BUSINESS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"
            
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        return f"Network error connecting to LLM API: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"Unexpected error parsing API response: {str(e)}"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def load_index_if_exists():
    """Robust loading of persistent index without crash risks"""
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
        except Exception as e:
            st.warning(f"Could not read previously saved database ({str(e)}). Initializing new index.")
            return None, None
    return None, None

# Automatic state restoration on app launch
if st.session_state.index is None:
    loaded_index, loaded_chunks = load_index_if_exists()
    if loaded_index is not None:
        st.session_state.index = loaded_index
        st.session_state.chunks = loaded_chunks
        st.info("Existing knowledge base loaded successfully from disk.")

# ===========================================================================
# 3. ADVANCED RAG LOGIC (Cross-page overlap & Distance threshold)
# ===========================================================================
def chunk_text_with_cross_page_overlap(reader, chunk_size=400, overlap=80, page_overlap_words=40):
    """
    Combined chunking algorithm: Resolves cross-page overlap using a 
    carry-over mechanism and injects exact page metadata for traceability.
    """
    all_chunks = []
    carry_over = [] 

    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        words = page_text.split()
        if not words:
            continue
            
        combined_words = carry_over + words
        
        # Internal chunking for the combined word list
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(combined_words), step):
            chunk_content = " ".join(combined_words[i : i + chunk_size])
            if chunk_content.strip():
                all_chunks.append(f"[Source: Page {idx + 1}] {chunk_content}")
            if i + chunk_size >= len(combined_words):
                break
                
        # Save the end of the page as carry-over for the next page
        carry_over = words[-page_overlap_words:] if len(words) > page_overlap_words else words
                
    return all_chunks

def retrieve(query, k=3, max_distance=MAX_DISTANCE):
    """Debug version: shows the chunks FAISS retrieves."""
    if st.session_state.index is None or not st.session_state.chunks:
        return []

    query_vec = np.array(embedder.encode([query])).astype("float32")
    distances, indices = st.session_state.index.search(query_vec, k)

    results = []

    st.markdown("### 🔍 RAG Debug — Retrieved Chunks")

    for rank, (dist, i) in enumerate(zip(distances[0], indices[0]), start=1):
        if i == -1 or i >= len(st.session_state.chunks):
            continue

        st.write(f"**Result {rank} — distance: {dist:.4f} — chunk index: {i}**")
        st.code(st.session_state.chunks[i])

        if dist <= max_distance:
            results.append(st.session_state.chunks[i])

    return results

# ===========================================================================
# 4. USER INTERFACE (Tabbed Structure)
# ===========================================================================
tab_docs, tab_sales, tab_reports = st.tabs([
    "📄 Document Knowledge Base", 
    "📊 Sales Data Analysis", 
    "🗓️ Agent Automation"
])

# ---------------------------------------------------------------------------
# TAB 1: KNOWLEDGE BASE (RAG + Q&A)
# ---------------------------------------------------------------------------
with tab_docs:
    st.header("📄 Manage Business Documents")
    pdf_file = st.file_uploader("Upload strategy document or quarterly report (PDF)", type=["pdf"])

    if pdf_file:
        with st.spinner("Analyzing text and building a traceable knowledge base..."):
            try:
                reader = PdfReader(pdf_file)
                chunks = chunk_text_with_cross_page_overlap(reader, chunk_size=80, overlap=20)
                
                if not chunks:
                    st.error("Unable to extract text. The PDF file may be empty or image-based.")
                else:
                    embeddings = np.array(embedder.encode(chunks)).astype("float32")
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(embeddings)
                    
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    
                    # Persistence
                    faiss.write_index(index, INDEX_PATH)
                    with open(CHUNKS_PATH, "wb") as f:
                        pickle.dump(chunks, f)
                        
                    st.success(f"Document successfully indexed! {len(chunks)} segments saved securely.")
            except Exception as e:
                st.error(f"An error occurred while processing the PDF file: {str(e)}")

    st.subheader("💬 Query the Document")
    question = st.text_input("Ask a strategic question to your knowledge base:")

    if st.button("🔎 Ask", key="doc_qa_ask_button"):
        if st.session_state.index is None:
            st.warning("Please upload and index a document first.")
        elif not question:
            st.warning("Please type a question first.")
        else:
            relevant_chunks = retrieve(question)
            if not relevant_chunks:
                st.session_state.doc_qa_question = question
                st.session_state.doc_qa_answer = None
                st.info("No sufficiently relevant context was found to answer the question accurately.")
            else:
                context = "\n\n".join(relevant_chunks)
                prompt = f"""Use ONLY the following context to answer the question. 
If the answer cannot be fully derived from the context, state that explicitly.

CONTEXT:
{context}

QUESTION:
{question}"""

                with st.spinner("Searching and analyzing..."):
                    # Save to session_state so the answer survives reruns
                    # triggered by other widgets (e.g. switching tabs or
                    # asking a new question later).
                    st.session_state.doc_qa_question = question
                    st.session_state.doc_qa_answer = ask_llm(prompt, temperature=0.2)

    # Always render the last-saved answer, not just right after the button press.
    if st.session_state.doc_qa_answer:
        st.markdown("### Agent Response")
        st.caption(f"Question: {st.session_state.doc_qa_question}")
        st.write(st.session_state.doc_qa_answer)

# ---------------------------------------------------------------------------
# TAB 2: SALES DATA (Raw Data, Plotly & Multi-Step Agent)
# ---------------------------------------------------------------------------
with tab_sales:
    st.header("📊 Market & Sales Data")
    csv_file = st.file_uploader("Upload sales data (CSV)", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.subheader("Data Preview (Top 10 Rows)")
            st.dataframe(df.head(10))
            
            # Flexible column detection: accept common synonyms instead of
            # requiring the exact literal names "Month" / "Revenue" / "Product".
            def find_col(candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                # case-insensitive fallback
                lower_map = {col.lower(): col for col in df.columns}
                for c in candidates:
                    if c.lower() in lower_map:
                        return lower_map[c.lower()]
                return None

            x_col = find_col(["Month", "quarter", "Quarter", "Date", "period"])
            y_col = find_col(["Revenue", "revenue_eur", "revenue", "Sales"])
            color_col = find_col(["Product", "product", "segment", "Segment"])

            if x_col and y_col:
                st.session_state.has_valid_chart = True
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title="Revenue Performance Dashboard")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.session_state.has_valid_chart = False
                missing = [name for name, col in [("a time/period column (e.g. Month or quarter)", x_col),
                                                   ("a revenue column (e.g. Revenue or revenue_eur)", y_col)] if col is None]
                st.warning(f"Unable to render chart. CSV is missing: {', '.join(missing)}")

            # ------------------------------------------------------------------
            # Build a COMPACT KPI summary instead of sending the raw CSV to the
            # LLM. Sending df.to_string() for a large file blows past Groq's
            # free-tier TPM limit (and wastes tokens/money on any provider).
            # ------------------------------------------------------------------
            def build_kpi_summary(df):
                lines = []
                lines.append(f"Total rows: {len(df)}")

                revenue_col = find_col(["revenue_eur", "Revenue", "revenue", "Sales"])
                margin_col = find_col(["gross_margin_eur", "Margin", "margin"])
                units_col = find_col(["units_sold", "Units", "units", "quantity"])
                quarter_col = find_col(["quarter", "Quarter", "Month", "period"])
                product_col = find_col(["product", "Product"])
                segment_col = find_col(["segment", "Segment"])
                country_col = find_col(["country", "Country"])

                if revenue_col:
                    lines.append(f"Total revenue: {df[revenue_col].sum():,.2f}")
                if margin_col:
                    lines.append(f"Total gross margin: {df[margin_col].sum():,.2f}")
                if units_col:
                    lines.append(f"Total units sold: {df[units_col].sum():,.0f}")

                def agg_block(group_col, label):
                    if not group_col:
                        return
                    agg_dict = {}
                    if revenue_col:
                        agg_dict[revenue_col] = "sum"
                    if margin_col:
                        agg_dict[margin_col] = "sum"
                    if units_col:
                        agg_dict[units_col] = "sum"
                    if not agg_dict:
                        return
                    grouped = df.groupby(group_col).agg(agg_dict).round(2)
                    lines.append(f"\n{label} breakdown:")
                    lines.append(grouped.to_string())

                agg_block(quarter_col, "Revenue/margin/units by quarter")
                agg_block(product_col, "Revenue/margin/units by product")
                agg_block(segment_col, "Revenue/margin/units by segment")
                agg_block(country_col, "Revenue/margin/units by country (top 10)" if country_col and df[country_col].nunique() > 10 else "Revenue/margin/units by country")

                # Simple anomaly flag: numeric outliers beyond 3 std devs
                numeric_cols = df.select_dtypes(include="number").columns
                anomalies = []
                for col in numeric_cols:
                    std = df[col].std()
                    mean = df[col].mean()
                    if std and std > 0:
                        outliers = df[(df[col] - mean).abs() > 3 * std]
                        if len(outliers) > 0:
                            anomalies.append(f"{col}: {len(outliers)} outlier row(s) beyond 3 std dev")
                if anomalies:
                    lines.append("\nPotential anomalies:")
                    lines.extend(anomalies)

                return "\n".join(lines)

            kpi_summary = build_kpi_summary(df)

            # Keep the full dataframe only for on-screen preview/debugging —
            # NEVER pass st.session_state.raw_csv_string to an LLM prompt.
            st.session_state.raw_csv_string = df.to_string()
            st.session_state.last_df_summary = kpi_summary

            # ------------------------------------------------------------------
            # Multi-Step Agentic Analysis is the primary / recommended mode:
            # separating deterministic fact-extraction (temp 0.0) from
            # strategic reasoning (temp 0.4) is a well-established pattern
            # for reducing hallucination, and it is also the core technical
            # showcase of this project. Standard Analysis (a single combined
            # call) is kept only as a quick, cheaper baseline, tucked into a
            # collapsed expander rather than presented as an equal choice.
            # ------------------------------------------------------------------
            st.subheader("🤖 Multi-Step Agentic Analysis")
            st.caption("Recommended: separates fact extraction (precise) from strategic reasoning (analytical) into two LLM calls.")

            if st.button("🤖 Run Multi-Step Agentic Analysis"):
                with st.spinner("Agent executing multi-step reasoning..."):
                    # Step 1: Data Collection & KPI Extraction (Deterministic temp 0.0)
                    # NOTE: we pass the compact pandas-computed KPI summary, not the raw CSV,
                    # to stay well under the LLM provider's tokens-per-minute limit.
                    step1_prompt = f"""Extract all critical numbers, total revenue, best performing products, 
and any visible data anomalies as a clean, raw fact list from this data:\n\n{st.session_state.last_df_summary}"""
                    extracted_metrics = ask_llm(step1_prompt, temperature=0.0)

                    # Step 2: Strategic Reasoning (Creative/Analytical temp 0.4)
                    step2_prompt = f"""Review these extracted business metrics and construct a final 
strategic execution plan with action points for the executive team:\n\n{extracted_metrics}"""
                    final_strategy = ask_llm(step2_prompt, temperature=0.4)

                    # Save to session_state so the result survives reruns
                    # (e.g. switching to another tab and back).
                    st.session_state.multistep_metrics = extracted_metrics
                    st.session_state.multistep_strategy = final_strategy

            # Always render the last-saved Multi-Step result, not just right after the button press.
            if st.session_state.multistep_metrics:
                st.subheader("📌 Step 1: Extracted Metrics (Precision Temp: 0.0)")
                st.info(st.session_state.multistep_metrics)
                st.subheader("🎯 Step 2: Strategic Recommendations (Reasoning Temp: 0.4)")
                st.write(st.session_state.multistep_strategy)

            with st.expander("📈 Standard Data Analysis (quick baseline, single LLM call)"):
                st.caption("Combines fact extraction and reasoning in one call — faster and cheaper, but generally less precise than Multi-Step.")
                if st.button("Run Standard Data Analysis"):
                    prompt = f"Perform a business analysis of the following sales data and trends.\n\n{st.session_state.last_df_summary}"
                    with st.spinner("Generating data analysis..."):
                        st.session_state.standard_analysis_result = ask_llm(prompt, temperature=0.15)

                if st.session_state.standard_analysis_result:
                    st.write(st.session_state.standard_analysis_result)

        except Exception as e:
            st.error(f"An error occurred while reading or parsing the CSV file: {str(e)}")

# ---------------------------------------------------------------------------
# TAB 3: AUTOMATION (Unified Weekly Report)
# ---------------------------------------------------------------------------
with tab_reports:
    st.header("🗓️ Executive Automation")
    st.caption("Automatically combines document insights and sales data into a unified strategic report.")

    if st.button("📅 Generate Final Weekly Report"):
        has_doc = st.session_state.chunks is not None
        has_csv = st.session_state.last_df_summary is not None
        
        if not has_doc and not has_csv:
            st.warning("Please upload either a document or a CSV file to generate a report.")
        else:
            with st.spinner("Synthesizing data sources into an executive summary..."):
                # Fetch top chunks as context if document exists
                doc_context = "\n\n".join(st.session_state.chunks[:6]) if has_doc else "No document available."
                # Compact KPI summary only — never the raw CSV — to stay under the LLM's token limit.
                # If a Multi-Step Agentic Analysis has already been run, its extracted
                # metrics + strategic reasoning are included too: Multi-Step is the
                # primary analysis mode in this app, so Agent Automation builds on
                # its output when available rather than needing a separate mode
                # selector or a combine/recommend mechanism.
                csv_context = st.session_state.last_df_summary if has_csv else "No sales data available."
                if st.session_state.multistep_strategy:
                    csv_context += f"""

PREVIOUSLY EXTRACTED METRICS (Multi-Step Analysis, Step 1):
{st.session_state.multistep_metrics}

PREVIOUSLY GENERATED STRATEGIC ANALYSIS (Multi-Step Analysis, Step 2):
{st.session_state.multistep_strategy}"""
                
                report_prompt = f"""Create a comprehensive Weekly Executive Business Report by weaving together 
the document insights and the market sales data provided below. 

Only reference data points that are explicitly present below—do not invent figures.

DOCUMENT RULES & STRATEGIES CONTEXT:
{doc_context}

ACTUAL SALES & TREND DATA CONTEXT:
{csv_context}

Ensure that conclusions drawn in the report directly bridge the gap between the document rules and the actual numbers."""

                report = ask_llm(report_prompt, temperature=0.3)
                st.subheader("📋 Final Combined Executive Report")
                st.write(report)