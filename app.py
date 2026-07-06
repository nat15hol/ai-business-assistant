import streamlit as st
import pandas as pd
# plotly.express är borttagen
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

# LÄGG TILL DIN POWER BI EMBED URL HÄR
POWER_BI_EMBED_URL = "https://app.powerbi.com/view?r=DIN_UNIKA_EMBED_STRÄNG_HÄR"

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
                "model": "llama-3.1-8b-instant",
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
    """Robust retrieval with index validation and adaptive threshold fallback"""
    if st.session_state.index is None or not st.session_state.chunks:
        return []
        
    query_vec = np.array(embedder.encode([query])).astype("float32")
    distances, indices = st.session_state.index.search(query_vec, k)
    
    results = []
    for dist, i in zip(distances[0], indices[0]):
        if i == -1 or i >= len(st.session_state.chunks):
            continue
        # Om avståndet är bra, lägg till i ordinarie resultat
        if dist <= max_distance:
            results.append(st.session_state.chunks[i])
            
    # FALLBACK: Om inget matchade under tröskelvärdet, men vi har en giltig förstaplats
    if not results and indices[0][0] != -1 and indices[0][0] < len(st.session_state.chunks):
        # Ta den absolut bästa träffen oavsett distans, så länge den inte är helt galen (t.ex. > 1.8)
        if distances[0][0] < 1.8:
            results.append(st.session_state.chunks[indices[0][0]])
        
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

    if question:
        if st.session_state.index is None:
            st.warning("Please upload and index a document first.")
        else:
            relevant_chunks = retrieve(question)
            if not relevant_chunks:
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
                    answer = ask_llm(prompt, temperature=0.2)
                    st.markdown("### Agent Response")
                    st.write(answer)

# ---------------------------------------------------------------------------
# TAB 2: SALES DATA (Raw Data, Power BI Embed & Multi-Step Agent)
# ---------------------------------------------------------------------------
with tab_sales:
    st.header("📊 Market & Sales Data")
    
    # Renderar Power BI instrumentpanelen högst upp i fliken
    st.subheader("🌐 Power BI Executive Dashboard")
    try:
        # Laddar in Power BI rapporten via en stabil iframe-komponent
        st.components.v1.iframe(src=POWER_BI_EMBED_URL, height=600, scrolling=False)
    except Exception as e:
        st.error(f"Kunde inte ladda Power BI-dashboarden: {str(e)}")
        
    st.markdown("---")
    
    # Behåller filuppladdaren för att agenten ska kunna läsa och analysera rådata
    st.subheader("📂 Upload Context Data for AI Agent")
    csv_file = st.file_uploader("Upload sales data for text analysis (CSV)", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.subheader("Data Preview (Top 10 Rows)")
            st.dataframe(df.head(10))
            
            # Beräknar statistisk sammanfattning för agenten
            summary_stats = df.describe(include='all').to_string()
            missing_data = df.isnull().sum().to_string()
            
            st.session_state.raw_csv_string = df.to_string()
            st.session_state.last_df_summary = f"""
STATISTICAL SUMMARY:
{summary_stats}

MISSING VALUES:
{missing_data}
"""
            # Action triggers
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📈 Run Standard Data Analysis"):
                    prompt = f"Perform a business analysis of the following sales data and trends.\n\n{st.session_state.last_df_summary}"
                    with st.spinner("Generating data analysis..."):
                        st.write(ask_llm(prompt, temperature=0.15))
                        
            with col2:
                if st.button("🤖 Run Multi-Step Agentic Analysis"):
                    with st.spinner("Agent executing multi-step reasoning..."):
                        
                        # Step 1: Data Collection & KPI Extraction (Deterministic temp 0.0)
                        step1_prompt = f"""Extract all critical numbers, total revenue, best performing products, 
and any visible data anomalies as a clean, raw fact list from this data:\n\n{st.session_state.raw_csv_string}"""
                        extracted_metrics = ask_llm(step1_prompt, temperature=0.0)
                        
                        # Step 2: Strategic Reasoning (Creative/Analytical temp 0.4)
                        step2_prompt = f"""Review these extracted business metrics and construct a final 
strategic execution plan with action points for the executive team:\n\n{extracted_metrics}"""
                        final_strategy = ask_llm(step2_prompt, temperature=0.4)
                        
                        # Visual presentation of agent steps
                        st.subheader("📌 Step 1: Extracted Metrics (Precision Temp: 0.0)")
                        st.info(extracted_metrics)
                        st.subheader("🎯 Step 2: Strategic Recommendations (Reasoning Temp: 0.4)")
                        st.write(final_strategy)
                        
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
                csv_context = f"{st.session_state.raw_csv_string}\n{st.session_state.last_df_summary}" if has_csv else "No sales data available."
                
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