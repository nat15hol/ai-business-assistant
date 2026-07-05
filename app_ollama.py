import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import requests

st.title("AI Business Assistant (Local LLM)")

# -----------------------
# LLM (OLLAMA)
# -----------------------
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

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

    # -----------------------
    # SUMMARY
    # -----------------------
    if st.button("Summarize document"):
        with st.spinner("Generating summary..."):
            result = ask_llm(
                f"Summarize this business report in bullet points:\n\n{pdf_text[:6000]}"
            )
        st.write(result)

    # -----------------------
    # Q&A
    # -----------------------
    st.subheader("💬 Ask the document")
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Thinking..."):
            result = ask_llm(
                f"""
You are a business analyst.

Only use the document below to answer.

DOCUMENT:
{pdf_text[:6000]}

QUESTION:
{question}
"""
            )
        st.write(result)

    # -----------------------
    # EXTRA INSIGHTS
    # -----------------------
    if st.button("Show insights"):
        with st.spinner("Analyzing..."):
            result = ask_llm(
                f"""
Extract business insights from this report:

- risks
- opportunities
- key trends

Return structured bullet points.

TEXT:
{pdf_text[:6000]}
"""
            )
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
        with st.spinner("Analyzing sales data..."):
            result = ask_llm(
                f"""
You are a business analyst.

Analyze this sales data and provide:
- trends
- anomalies
- key insights

DATA:
{df.to_string()}
"""
            )
        st.write(result)