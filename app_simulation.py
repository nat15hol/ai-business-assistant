import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("AI Business Assistant")

# -----------------------
# PDF UPLOAD
# -----------------------
st.header("📄 Upload PDF")
pdf_file = st.file_uploader("Upload a document", type=["pdf"])

pdf_text = ""

if pdf_file:
    if pdf_file.name.endswith(".pdf"):
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text()
    else:
        pdf_text = pdf_file.read().decode("utf-8")

    st.success("File loaded!")

    # -----------------------
    # SUMMARY
    # -----------------------
    if st.button("Summarize document"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": f"Summarize this document:\n{pdf_text[:6000]}"}
            ]
        )
        st.write(response.choices[0].message.content)

    # -----------------------
    # Q&A
    # -----------------------
    st.subheader("💬 Ask the document")

    question = st.text_input("Ask a question about the document")

    if question:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analyst. Only use the document as context."},
                {"role": "user", "content": f"DOCUMENT:\n{pdf_text[:6000]}\n\nQUESTION:\n{question}"}
            ]
        )
        st.write(response.choices[0].message.content)

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": f"Analyze this sales data:\n{df.to_string()}"}
            ]
        )
        st.write(response.choices[0].message.content)