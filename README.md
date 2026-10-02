# Enterprise AI Business Assistant

**A Streamlit app that gives Product Managers fact-grounded answers from strategy documents (RAG) and turns raw sales CSVs into structured KPI analysis, using a multi-step LLM agent.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20search-FAISS-0467DF)
![Groq](https://img.shields.io/badge/LLM-Groq-F55036)
![License](https://img.shields.io/badge/License-MIT-green)

**Live demo:** https://ai-business-assistant-rag.streamlit.app/

<!-- Add a screenshot or GIF here, e.g.: -->
<!-- ![Demo](docs/demo.gif) -->

---

## What it does

| Tab | Purpose | How |
|---|---|---|
| 📄 **Document Knowledge Base** | Ask questions about a PDF (strategy doc, quarterly report) and get answers grounded only in its content | PDF → overlapping chunks → embeddings → FAISS → top-k retrieval → LLM |
| 📊 **Sales Data Analysis** | Upload a sales CSV, get an instant chart and an LLM-written business analysis | pandas computes KPIs, only a compact summary goes to the LLM |
| 🗓️ **Agent Automation** | One click produces an executive report combining document rules with actual sales numbers | Merges document context and KPI summary into one prompt |

Every LLM answer follows a fixed structure: **Key Insights & KPIs → Identified Trends → Operational Risks → Recommended Actions**. The model is told to say so explicitly when the context does not contain the answer.

## Highlights

- **Cross-page chunking.** Text is split into overlapping word windows, and the last 40 words of each page carry over into the next, so sentences that span a page break are not lost. Every chunk is tagged `[Source: Page N]`.
- **Distance-thresholded retrieval.** Results must fall under an L2 distance of `1.6`. If nothing qualifies, the single best hit is used as a fallback only if its distance is under `1.8`. Otherwise the app says it found nothing relevant instead of guessing.
- **Token-efficient analytics.** The raw CSV is never sent to the LLM. pandas builds a compact summary (totals, breakdowns by quarter, product, segment and country, and 3σ outlier flags), which keeps prompts small and stays under free-tier rate limits.
- **Two-step agent.** Step 1 extracts facts at `temperature=0.0` for precision. Step 2 reasons over those facts at `temperature=0.4` to draft the strategy. Both steps are shown in the UI.
- **Flexible CSV schema.** Column names are matched case-insensitively against common synonyms (see [CSV format](#csv-format)).
- **Persistent knowledge base.** The FAISS index and chunks are saved to disk and reloaded on the next start.

## Architecture

```mermaid
flowchart LR
    subgraph Docs["Document pipeline"]
        PDF[PDF upload] --> CH[Chunking with page carry-over]
        CH --> EMB[all-MiniLM-L6-v2 embeddings]
        EMB --> FAISS[(FAISS IndexFlatL2)]
        Q[Question] --> RET[Top-k retrieval + distance filter]
        FAISS --> RET
    end

    subgraph Sales["Sales pipeline"]
        CSV[CSV upload] --> PD[pandas KPI summary]
        CSV --> PLOT[Plotly bar chart]
    end

    RET --> LLM["Groq LLM<br/>(openai/gpt-oss-20b)"]
    PD --> LLM
    FAISS -. first 6 chunks .-> REP[Weekly report]
    PD --> REP
    REP --> LLM
    LLM --> OUT[Structured answer in Streamlit]
```

## Tech stack

- **UI:** [Streamlit](https://streamlit.io)
- **LLM:** [Groq](https://groq.com) chat completions API, model `openai/gpt-oss-20b`
- **Embeddings:** [`sentence-transformers`](https://www.sbert.net) with `all-MiniLM-L6-v2`
- **Vector search:** [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`, flat L2 index)
- **PDF parsing:** `pypdf`
- **Data and charts:** `pandas`, `plotly`

## Getting started

### Prerequisites

- Python 3.11 (matches the dev container and the deployed environment)
- A free [Groq API key](https://console.groq.com/keys)

### Installation

```bash
git clone https://github.com/nat15hol/ai-business-assistant.git
cd ai-business-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> The first run downloads the embedding model (roughly 90 MB), and `sentence-transformers` pulls in PyTorch, so the install can take a few minutes.

### Configuration

The app reads your key from the `GROQ_API_KEY` environment variable.

**Option A: environment variable**

```bash
export GROQ_API_KEY="your-key-here"        # Windows PowerShell: $env:GROQ_API_KEY="your-key-here"
```

**Option B: `.env` file or Streamlit secrets**

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-key-here"
```

Streamlit exposes top-level secrets as environment variables, so this works without code changes. Both `.env` and `.streamlit/secrets.toml` are already in `.gitignore`.

### Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

### GitHub Codespaces / Dev Container

The repo ships with a `.devcontainer` (Python 3.11). Open it in Codespaces or VS Code Dev Containers and dependencies install automatically, then the app starts on port `8501`. Add `GROQ_API_KEY` as a **Codespaces secret** beforehand so the app can reach the LLM.

## Usage

### 1. Ask questions about a document

1. Open **📄 Document Knowledge Base** and upload a PDF.
2. Wait for the *"Document successfully indexed"* message.
3. Ask a question, for example: *"What are our stated priorities for next quarter?"*

The PDF must contain selectable text. Scanned or image-only PDFs are rejected (no OCR).

### 2. Analyse sales data

1. Open **📊 Sales Data Analysis** and upload a CSV.
2. Review the preview and the auto-generated revenue chart.
3. Choose **Run Standard Data Analysis** for a single-pass analysis, or **Run Multi-Step Agentic Analysis** to see extracted metrics first and strategy second.

### 3. Generate the combined report

Open **🗓️ Agent Automation** and click **Generate Final Weekly Report**. It works with a PDF, a CSV, or both, and gives the best results when both are loaded.

## CSV format

Columns are detected automatically. The first match wins:

| Role | Recognised column names |
|---|---|
| Time axis (chart) | `Month`, `quarter`, `Quarter`, `Date`, `period` |
| Revenue (chart) | `Revenue`, `revenue_eur`, `revenue`, `Sales` |
| Colour grouping (chart) | `Product`, `product`, `segment`, `Segment` |
| Gross margin (KPIs) | `gross_margin_eur`, `Margin`, `margin` |
| Units (KPIs) | `units_sold`, `Units`, `units`, `quantity` |
| Breakdowns (KPIs) | `quarter`/`Month`/`period`, `product`, `segment`, `country` |

A time column and a revenue column are required for the chart. The KPI summary uses whatever it finds. Example:

```csv
quarter,product,segment,country,revenue_eur,gross_margin_eur,units_sold
2025-Q1,Alpha,Enterprise,SE,125000,52000,310
2025-Q1,Beta,SMB,DE,48000,17500,420
2025-Q2,Alpha,Enterprise,SE,139000,58000,342
```

## Project structure

```
ai-business-assistant/
├── app.py                  # Entire application (UI, RAG pipeline, agent logic)
├── requirements.txt        # Python dependencies
├── .devcontainer/
│   └── devcontainer.json   # Codespaces / VS Code Dev Container setup
└── .gitignore               # Excludes secrets and the generated index files
```

Generated at runtime (git-ignored): `faiss_index.bin` and `chunks.pkl`.

## Tuning

Key parameters live in `app.py`:

| Setting | Default | Effect |
|---|---|---|
| `MAX_DISTANCE` | `1.6` | Maximum L2 distance for a chunk to count as relevant. Lower is stricter. |
| `chunk_size` / `overlap` | `80` / `20` words | Chunk length and overlap, set at the call site in the PDF upload block. |
| `page_overlap_words` | `40` | Words carried from one page into the next. |
| `retrieve(..., k=3)` | `3` | Number of chunks passed to the LLM. |
| `model` | `openai/gpt-oss-20b` | Any chat model available on Groq. |

The system prompt (`BUSINESS_SYSTEM_PROMPT`) defines the output structure and can be edited to fit other roles or report formats.

## Privacy and security notes

- **Data leaves your machine.** Retrieved document chunks and CSV KPI summaries are sent to the Groq API. Do not upload confidential material unless that is acceptable under your data policy.
- **Never commit your API key.** Use environment variables or Streamlit secrets.
- **`chunks.pkl` is loaded with `pickle`.** Only run the app against index files it generated itself, since unpickling untrusted files can execute arbitrary code.

## Known limitations

- Only one document is indexed at a time. Uploading a new PDF overwrites the previous index on disk.
- The saved index is global to the running instance, so on a shared deployment all users see the same knowledge base.
- The weekly report uses the **first six chunks** of the document rather than query-relevant ones, so it reflects the start of the PDF.
- Source page numbers are included in the LLM context but are not yet displayed as citations in the UI.
- Groq's free tier enforces a tokens-per-minute (TPM) limit; the app's KPI-summary approach keeps requests well under it, but very frequent use could still occasionally hit a rate limit.
- `requirements.txt` is unpinned, so future library releases may change behaviour.

## Roadmap ideas

- [ ] Show cited source pages next to each answer
- [ ] Multi-document support with per-document filtering
- [ ] OCR for scanned PDFs
- [ ] Pin dependency versions
- [ ] Per-session indexes instead of a shared on-disk index
- [ ] Unit tests for chunking, retrieval and KPI summary logic

## Author

Henrik Oldehed

Data Engineer | Backend and Software Developer — building full-stack systems end-to-end

GitHub: https://github.com/nat15hol

LinkedIn: https://www.linkedin.com/in/henrikoldehed/

## License

[MIT](LICENSE)
