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
# 1. KONFIGURATION OCH GLOBAL STATE
# ===========================================================================
st.set_page_config(page_title="Enterprise AI Business Assistant", page_icon="💼", layout="wide")
st.title("💼 Enterprise AI Business Assistant")
st.caption("Production-Grade RAG & Multi-Step Sales Analytics Agent")

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

# L2-avstånd tröskelvärde: Lägre = mer likt.
MAX_DISTANCE = 1.2 

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

# Defensiv initialisering av Session State
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
# 2. CORE ENGINE & UTILITIES (Robust felhantering)
# ===========================================================================
def ask_llm(prompt, temperature=0.2):
    """Robust API-klient med dubbel felhantering och explicit nyckelkoll"""
    if not GROQ_API_KEY:
        st.error("🔑 GROQ_API_KEY saknas. Vänligen konfigurera Streamlit Secrets eller miljövariabler.")
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
            return f"⚠️ API Felkod {response.status_code}: {response.text}"
            
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Nätverksfel mot LLM-API: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"⚠️ Oväntat fel vid parsning av API-svar: {str(e)}"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def load_index_if_exists():
    """Robust laddning av persistens utan kraschrisk"""
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
        except Exception as e:
            st.warning(f"Kunde inte läsa tidigare sparad databas ({str(e)}). Initierar ny.")
            return None, None
    return None, None

# Automatiskt återställande av state vid appstart
if st.session_state.index is None:
    loaded_index, loaded_chunks = load_index_if_exists()
    if loaded_index is not None:
        st.session_state.index = loaded_index
        st.session_state.chunks = loaded_chunks
        st.info("💾 Befintlig kunskapsbas lästes in från disk.")

# ===========================================================================
# 3. AVANCERAD RAG-LOGIK (Sidöverskridande overlap & Rättad tröskel)
# ===========================================================================
def chunk_text_with_cross_page_overlap(reader, chunk_size=400, overlap=80, page_overlap_words=40):
    """
    Kombinerad chunking-algoritm: Löser cross-page overlap med 
    carry_over-mekanism och injicerar exakt sidmetadata för spårbarhet.
    """
    all_chunks = []
    carry_over = [] 

    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        words = page_text.split()
        if not words:
            continue
            
        combined_words = carry_over + words
        
        # Intern chunking för den sammanslagna ordlistan
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(combined_words), step):
            chunk_content = " ".join(combined_words[i : i + chunk_size])
            if chunk_content.strip():
                all_chunks.append(f"[Source: Page {idx + 1}] {chunk_content}")
            if i + chunk_size >= len(combined_words):
                break
                
        # Spara slutet av sidan som carry-over till nästa
        carry_over = words[-page_overlap_words:] if len(words) > page_overlap_words else words
                
    return all_chunks

def retrieve(query, k=3, max_distance=MAX_DISTANCE):
    """Robust retrieval med indexskydd och RÄTTAD distans-threshold"""
    if st.session_state.index is None or not st.session_state.chunks:
        return []
        
    query_vec = np.array(embedder.encode([query])).astype("float32")
    distances, indices = st.session_state.index.search(query_vec, k)
    
    results = []
    for dist, i in zip(distances[0], indices[0]):
        # Validera mot index -1 och out of bounds
        if i == -1 or i >= len(st.session_state.chunks):
            continue
        # L2-avstånd: Ju mindre desto bättre. Filtrera bort om det överstiger MAX_DISTANCE
        if dist > max_distance:
            continue
        results.append(st.session_state.chunks[i])
        
    return results

# ===========================================================================
# 4. PRODUKTDESIGN / GRÄNSSNITT (Flikbaserad struktur)
# ===========================================================================
tab_docs, tab_sales, tab_reports = st.tabs([
    "📄 Document Knowledge Base", 
    "📊 Sales Data Analysis", 
    "🗓️ Agent Automation"
])

# ---------------------------------------------------------------------------
# FLIK 1: KUNSKAPSBAS (RAG + Q&A)
# ---------------------------------------------------------------------------
with tab_docs:
    st.header("📄 Hantera Affärsdokument")
    pdf_file = st.file_uploader("Ladda upp strategidokument eller kvartalsrapport (PDF)", type=["pdf"])

    if pdf_file:
        with st.spinner("Analyserar text och bygger spårbar kunskapsbas..."):
            try:
                reader = PdfReader(pdf_file)
                chunks = chunk_text_with_cross_page_overlap(reader)
                
                if not chunks:
                    st.error("Det gick inte att extrahera text. PDF-filen kan vara tom eller bildbaserad.")
                else:
                    embeddings = np.array(embedder.encode(chunks)).astype("float32")
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(embeddings)
                    
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    
                    # Persistens
                    faiss.write_index(index, INDEX_PATH)
                    with open(CHUNKS_PATH, "wb") as f:
                        pickle.dump(chunks, f)
                        
                    st.success(f"✅ Dokumentet har indexerats! {len(chunks)} segment sparas robust.")
            except Exception as e:
                st.error(f"Ett fel uppstod vid bearbetning av PDF-filen: {str(e)}")

    st.subheader("💬 Fråga dokumentet")
    question = st.text_input("Ställ en strategisk fråga till din databas:")

    if question:
        if st.session_state.index is None:
            st.warning("Vänligen ladda upp och indexera ett dokument först.")
        else:
            relevant_chunks = retrieve(question)
            if not relevant_chunks:
                st.info("Ingen tillräckligt relevant kontext hittades för att besvara frågan säkert.")
            else:
                context = "\n\n".join(relevant_chunks)
                prompt = f"""Use ONLY the following context to answer the question. 
If the answer cannot be fully derived from the context, state that explicitly.

CONTEXT:
{context}

QUESTION:
{question}"""
                
                with st.spinner("Söker och analyserar..."):
                    answer = ask_llm(prompt, temperature=0.2)
                    st.markdown("### Svar från Agenten")
                    st.write(answer)

# ---------------------------------------------------------------------------
# FLIK 2: SÄLJDATA (Rådata, Plotly & Flerstegsagent)
# ---------------------------------------------------------------------------
with tab_sales:
    st.header("📊 Marknads- & Försäljningsdata")
    csv_file = st.file_uploader("Ladda upp säljdata (CSV)", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.subheader("Förhandsvisning (Topp 10 rader)")
            st.dataframe(df.head(10)) # df.head(10) håller UI städat
            
            # Kolumnvalidering med visuell feedback
            required_cols = {"Month", "Revenue"}
            if required_cols.issubset(df.columns):
                st.session_state.has_valid_chart = True
                color_col = "Product" if "Product" in df.columns else None
                fig = px.bar(df, x="Month", y="Revenue", color=color_col, title="Revenue Performance Dashboard")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.session_state.has_valid_chart = False
                st.warning(f"⚠️ Diagram kan inte ritas. CSV saknar kolumner: {required_cols - set(df.columns)}")
                
            # Beräkna och spara statistisk sammanfattning EN gång
            summary_stats = df.describe(include='all').to_string()
            missing_data = df.isnull().sum().to_string()
            
            st.session_state.raw_csv_string = df.to_string()
            st.session_state.last_df_summary = f"""
STATISTISK SAMMANFATTNING:
{summary_stats}

SAKNADE VÄRDEN:
{missing_data}
"""
            # Snygg uppdelning av triggers
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📈 Kör Standard Dataanalys"):
                    prompt = f"Utför en affärsanalys av följande säljdata och trender.\n\n{st.session_state.last_df_summary}"
                    with st.spinner("Genererar dataanalys..."):
                        st.write(ask_llm(prompt, temperature=0.15))
                        
            with col2:
                if st.button("🤖 Kör Agentic Flerstegsanalys (Multi-Step)"):
                    with st.spinner("Agenten exekverar flerstegsresonemang..."):
                        
                        # Steg 1: Datainsamling och KPI-extraktion (Hård temperatur 0.0 för precision)
                        step1_prompt = f"""Extract all critical numbers, total revenue, best performing products, 
and any visible data anomalies as a clean, raw fact list from this data:\n\n{st.session_state.raw_csv_string}"""
                        extracted_metrics = ask_llm(step1_prompt, temperature=0.0)
                        
                        # Steg 2: Strategiskt resonemang (Högre temperatur 0.4 för affärskreativitet)
                        step2_prompt = f"""Review these extracted business metrics and construct a final 
strategic execution plan with action points for the executive team:\n\n{extracted_metrics}"""
                        final_strategy = ask_llm(step2_prompt, temperature=0.4)
                        
                        # Visuell presentation av stegen
                        st.subheader("📌 Steg 1: Extraherade Nyckeltal (Precision 0.0)")
                        st.info(extracted_metrics)
                        st.subheader("🎯 Steg 2: Strategiska Rekommendationer (Resonemang 0.4)")
                        st.write(final_strategy)
                        
        except Exception as e:
            st.error(f"Ett fel uppstod vid inläsning eller parsning av CSV-filen: {str(e)}")

# ---------------------------------------------------------------------------
# FLIK 3: AUTOMATION (Kombinerad Veckorapport utan hallucinationer)
# ---------------------------------------------------------------------------
with tab_reports:
    st.header("🗓️ Executive Automation")
    st.caption("Sammanfogar automatiskt dokumentinsikter och säljdata till en strategisk rapport.")

    if st.button("📅 Generera Slutgiltig Veckorapport"):
        has_doc = st.session_state.chunks is not None
        has_csv = st.session_state.last_df_summary is not None
        
        if not has_doc and not has_csv:
            st.warning("Det krävs att du laddar upp antingen ett dokument eller en CSV-fil för att bygga en rapport.")
        else:
            with st.spinner("Väver samman datakällor till en samlad executive summary..."):
                # Hämta de översta centrala chunkarna som kontext om dokument finns
                doc_context = "\n\n".join(st.session_state.chunks[:6]) if has_doc else "Inget dokument tillgängligt."
                csv_context = f"{st.session_state.raw_csv_string}\n{st.session_state.last_df_summary}" if has_csv else "Ingen säljdata tillgänglig."
                
                report_prompt = f"""Create a comprehensive Weekly Executive Business Report by weaving together 
the document insights and the market sales data provided below. 

Only reference data points that are explicitly present below—do not invent figures.

DOCUMENT RULES & STRATEGIES CONTEXT:
{doc_context}

ACTUAL SALES & TREND DATA CONTEXT:
{csv_context}

Ensure that conclusions drawn in the report directly bridge the gap between the document rules and the actual numbers."""

                report = ask_llm(report_prompt, temperature=0.3)
                st.subheader("📋 Slutgiltig Veckorapport (Kombinerad)")
                st.write(report)