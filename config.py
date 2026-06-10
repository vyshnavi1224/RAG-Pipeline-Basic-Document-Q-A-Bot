"""
Configuration settings for the RAG Q&A Bot.
All tuneable parameters live here — change once, affects everywhere.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64          # chunks embedded per batch call

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE    = 600                # characters per chunk (≈ 120 tokens)
CHUNK_OVERLAP = 100                # overlap between adjacent chunks

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 5                          # number of chunks retrieved per query

# ── LLM (Anthropic Claude) ─────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = "llama-3.1-8b-instant"   # free, very fast on Groq LPUs
LLM_MAX_TOKENS    = 1024

# ── FAISS index file ───────────────────────────────────────────────────────
FAISS_INDEX_FILE    = VECTORSTORE_DIR / "faiss.index"
METADATA_FILE       = VECTORSTORE_DIR / "metadata.json"
