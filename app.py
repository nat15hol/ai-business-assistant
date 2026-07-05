import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import numpy as np
import faiss
import os
import requests
from sentence_transformers import SentenceTransformer

st.title("AI Business Assistant (RAG + Cloud LLM)")

# -----------------------
# LLM (CLOUD READY)
# -----------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def ask_llm(prompt):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30
    )

    if response.status_code != 200:
        return f"API error: {response.text}"

    return response.json()["choices"][0]["message"]["content"]


# -----------------------
# RAG SETUP
# -----------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
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

if pdf_file:
    pdf_text = ""

    reader = PdfReader(pdf_file)
    for page in reader.pages:
        pdf_text += page.extract_text() or ""

    st.success("File loaded!")

    chunks = chunk_text(pdf_text)
    embeddings = embedder.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    st.session_state.chunks = chunks
    st.session_state.index = index


# -----------------------
# RETRIEVAL
# -----------------------

def retrieve(query, k=3):
    if st.session_state.index is None:
        return []

    query_vec = np.array(embedder.encode([query])).astype("float32")
    distances, indices = st.session_state.index.search(query_vec, k)

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
Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION:
{question}
""")

        st.write(result)


# -----------------------
# CSV ANALYSIS
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