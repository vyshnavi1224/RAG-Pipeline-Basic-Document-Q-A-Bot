# RAG Document Q&A Bot — Science Edition

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask
natural-language questions against a collection of science documents and receive accurate,
grounded answers with source citations — powered by a local embedding model and Anthropic Claude.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Architecture Overview](#architecture-overview)
3. [Chunking Strategy](#chunking-strategy)
4. [Embedding Model & Vector Database](#embedding-model--vector-database)
5. [Setup Instructions](#setup-instructions)
6. [Environment Variables](#environment-variables)
7. [Running the Bot](#running-the-bot)
8. [Example Queries](#example-queries)
9. [Project Structure](#project-structure)
10. [Known Limitations](#known-limitations)

---

## Tech Stack

| Component          | Library / Tool                     | Version   |
|--------------------|------------------------------------|-----------|
| Embedding model    | sentence-transformers (all-MiniLM-L6-v2) | ≥ 2.7.0 |
| Vector database    | FAISS (CPU)                        | ≥ 1.8.0   |
| LLM                | Anthropic Claude (claude-3-haiku)  | ≥ 0.28.0  |
| PDF parsing        | pypdf                              | ≥ 4.0.0   |
| DOCX parsing       | python-docx                        | ≥ 1.1.0   |
| Web UI (bonus)     | Streamlit                          | ≥ 1.35.0  |
| Python             | Python                             | ≥ 3.11    |

---

## Architecture Overview

```
┌─────────────────────────────── INDEXING STEP (run once) ──────────────────────────────┐
│                                                                                         │
│  /data/*.txt / *.pdf / *.docx                                                          │
│          │                                                                              │
│          ▼                                                                              │
│   [ src/ingestion.py ]   → page-level records {text, source, page}                    │
│          │                                                                              │
│          ▼                                                                              │
│   [ src/chunker.py ]     → paragraph-aware chunks {text, source, page, chunk_id}     │
│          │                                                                              │
│          ▼                                                                              │
│   [ src/embedder.py ]    → float32 embeddings (batched, normalised)                   │
│          │                                                                              │
│          ▼                                                                              │
│   [ src/vectorstore.py ] → FAISS IndexFlatIP + JSON metadata sidecar → /vectorstore/  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── QUERY STEP (every question) ───────────────────────────┐
│                                                                                         │
│  User question                                                                          │
│       │                                                                                 │
│       ▼                                                                                 │
│  [ src/embedder.py ]     → query embedding                                             │
│       │                                                                                 │
│       ▼                                                                                 │
│  [ src/vectorstore.py ]  → top-k most similar chunks (cosine similarity via FAISS)    │
│       │                                                                                 │
│       ▼                                                                                 │
│  [ src/generator.py ]    → Claude API call (context + question → grounded answer)     │
│       │                                                                                 │
│       ▼                                                                                 │
│  Answer + citations displayed to user                                                  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Chunking Strategy

**Strategy chosen: Paragraph-aware chunking with character-based size limit**

**How it works:**
1. Each page of text is split on double-newlines (natural paragraph boundaries).
2. Paragraphs are accumulated until the running character count would exceed `CHUNK_SIZE` (default 600 chars ≈ 120 tokens).
3. When the limit is reached, the current chunk is flushed and the next chunk starts with the last `CHUNK_OVERLAP` (100) characters of the previous chunk for context continuity.

**Why this strategy?**
- Scientific documents are structured into self-contained paragraphs (definitions, equations, conclusions). Breaking inside a paragraph loses the conceptual unit.
- Fixed-size character chunking would arbitrarily split mid-sentence, degrading retrieval quality.
- Sentence-based splitting would create many tiny chunks that individually lack enough context for the LLM.
- Paragraph-aware chunking gives semantically coherent chunks while keeping a predictable size budget.

---

## Embedding Model & Vector Database

**Embedding model: `sentence-transformers/all-MiniLM-L6-v2`**
- Produces 384-dimensional dense vectors, optimised for semantic similarity search.
- Runs locally (no API call), keeping inference fast and free.
- Achieves strong performance on the MTEB benchmark for the model's size class.
- Embeddings are L2-normalised so cosine similarity reduces to an inner product.

**Vector database: FAISS (`IndexFlatIP`)**
- Brute-force inner-product search over normalised vectors = exact cosine similarity.
- Persists to disk as a single binary file; zero server infrastructure required.
- For a corpus of a few thousand chunks the exact index is faster than approximate methods.
- Clear separation: `index.py` (indexing) vs `query.py` / `app.py` (querying).

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-bot.git
cd rag-qa-bot
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate          # macOS / Linux
# or: venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` installs the CPU build. If you have a CUDA GPU, replace it with
> `faiss-gpu` in `requirements.txt` for faster indexing.

### 4. Set your API key

```bash
cp .env.example .env
# Open .env in your editor and paste your Anthropic API key
```

Then load it into your shell (or use a tool like `python-dotenv`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Windows PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 5. Build the vector index (run once)

```bash
python index.py
```

This reads all documents from `/data`, chunks them, embeds them, and saves the FAISS index
to `/vectorstore`. Expected output:

```
09:00:01  INFO      Step 1/4  Loading documents from: .../data
09:00:01  INFO      Loading quantum_mechanics_fundamentals.txt …
...
09:00:15  INFO      Indexing complete!  Vector store saved to: .../vectorstore
```

### 6. Ask questions

**Command-line interface:**
```bash
python query.py
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
# Opens http://localhost:8501 in your browser
```

---

## Environment Variables

| Variable            | Required | Description                                  |
|---------------------|----------|----------------------------------------------|
| `ANTHROPIC_API_KEY` | ✅ Yes    | Your Anthropic API key from console.anthropic.com |

Never commit your actual API key. The `.gitignore` excludes `.env`.

---

## Running the Bot

### CLI — basic

```bash
python query.py
```

### CLI — verbose (shows retrieved passages)

```bash
python query.py --verbose
```

### CLI — custom top-k

```bash
python query.py --top-k 3
```

### Web UI

```bash
streamlit run app.py
```

Use the sidebar to adjust top-k and toggle visibility of retrieved passages.

---

## Example Queries

| Query | Expected answer theme |
|---|---|
| What is the Heisenberg uncertainty principle? | Position-momentum tradeoff, Δx·Δp ≥ ℏ/2, quantum nature |
| How does CRISPR-Cas9 cut DNA? | Guide RNA, PAM sequence, Cas9 double-strand break mechanism |
| What is Hawking radiation and why is it important? | Virtual particle pairs, black hole evaporation, temperature formula |
| What are climate tipping points and give examples | WAIS, Greenland ice sheet, Amazon dieback, AMOC weakening |
| What SSP scenario matches current government pledges? | SSP2-4.5 / NDCs imply ~2.5-3°C, Paris Agreement gap |
| What was approved by the FDA in 2023 related to CRISPR? | Casgevy (exagamglogene autotemcel) for sickle cell / beta-thalassemia |
| What is Schrödinger's equation? | Time-dependent form, wave function, Born probability interpretation |
| How much has Arctic sea ice declined? | ~40% since 1979, Arctic amplification 4x global average |

**Out-of-scope query (the bot should say so):**
> "What is the speed of the fastest Formula 1 car?"
→ Expected: "I could not find an answer to this question in the provided documents."

---

## Project Structure

```
rag-qa-bot/
├── data/                              # Knowledge base documents
│   ├── quantum_mechanics_fundamentals.txt
│   ├── crispr_gene_editing.txt
│   ├── black_holes_general_relativity.txt
│   └── climate_science_global_warming.pdf   ← required PDF
│
├── vectorstore/                       # Auto-generated (gitignored)
│   ├── faiss.index
│   └── metadata.json
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py                   # Document loading (txt/pdf/docx)
│   ├── chunker.py                     # Paragraph-aware chunking
│   ├── embedder.py                    # Batched sentence-transformer embeddings
│   ├── vectorstore.py                 # FAISS build + search
│   └── generator.py                   # Claude API answer generation
│
├── config.py                          # All tuneable parameters
├── index.py                           # Indexing step (run once)
├── query.py                           # Interactive CLI
├── app.py                             # Streamlit web UI (bonus)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Known Limitations

- **Context window cap:** Each query sends at most `top_k × chunk_size` characters to the LLM. Very detailed multi-part questions may not receive complete answers if the relevant information spans many distant chunks.
- **No conversation memory:** The CLI and web UI treat each question independently. Follow-up questions like "tell me more about that" will not work as expected.
- **PDF quality dependency:** The PDF ingestion relies on text extraction (pypdf). Scanned PDFs (image-only) will produce empty text and will not be indexed. Use OCR (e.g., `pytesseract`) as a preprocessing step for scanned documents.
- **Embedding model language:** `all-MiniLM-L6-v2` is English-only. Multi-lingual documents require a model like `paraphrase-multilingual-MiniLM-L12-v2`.
- **FAISS exact search:** `IndexFlatIP` is exact but scales O(n) with corpus size. For corpora > 100,000 chunks, switch to an approximate index (`IndexIVFFlat` or `HNSW`).
- **No re-ranking:** A cross-encoder re-ranking pass after FAISS retrieval would improve precision at the cost of additional latency.
- **Single-machine only:** The current architecture stores the index on local disk. Deploying to multiple servers requires a shared network file system or migrating to a hosted vector DB (Pinecone, Qdrant Cloud, Weaviate Cloud).
