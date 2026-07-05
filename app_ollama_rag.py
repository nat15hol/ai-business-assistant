import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title("AI Business Assistant (RAG + Local LLM)")


# -----------------------
# LLM (Ollama)
# -----------------------
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )
    return response.json()["response"]


# -----------------------
# RAG SETUP
# -----------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


# -----------------------
# STATE
# -----------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None

# -----------------------
# PDF UPLOAD
# -----------------------
st.header("📄 Upload PDF")
pdf_file = st.file_uploader("Upload a document", type=["pdf"])

pdf_text = ""

if pdf_file:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        pdf_text += page.extract_text() or ""

    st.success("File loaded!")

    # Build RAG index ONLY after upload
    chunks = chunk_text(pdf_text)
    embeddings = embedder.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    st.session_state.chunks = chunks
    st.session_state.index = index


# -----------------------
# RETRIEVAL FUNCTION
# -----------------------
def retrieve(query, k=3):
    query_vec = np.array(embedder.encode([query])).astype("float32")
    distances, indices = st.session_state.index.search(np.array(query_vec), k)
    return [st.session_state.chunks[i] for i in indices[0]]


# -----------------------
# Q&A
# -----------------------
st.subheader("💬 Ask the document")
question = st.text_input("Ask a question about the document")

if question:
    if st.session_state.index is None:
        st.warning("Please upload a PDF first.")
    else:
        relevant_chunks = retrieve(question)

        context = "\n\n".join(relevant_chunks)

        result = ask_llm(f"""
You are a business analyst.

Use ONLY this context:

{context}

Question: {question}
""")

        st.write(result)

# -----------------------
# CSV ANALYSIS (optional kvar)
# -----------------------
st.header("📊 Upload Sales Data")
csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.dataframe(df)

    fig = px.bar(df, x="Month", y="Revenue", color="Product")
    st.plotly_chart(fig)

    if st.button("Analyze sales"):
        result = ask_llm(f"""
Analyze this sales data:
{df.to_string()}
""")
        st.write(result)
