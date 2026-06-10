#!/usr/bin/env python3
"""
index.py
--------
One-time (or on-demand) indexing step:
  1. Loads all documents from /data
  2. Chunks them with paragraph-aware chunking
  3. Embeds all chunks in batches
  4. Saves the FAISS index + metadata to /vectorstore

Run with:
  python index.py
"""
import logging
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, VECTORSTORE_DIR,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    FAISS_INDEX_FILE, METADATA_FILE,
)
from src.ingestion   import load_documents
from src.chunker     import chunk_records
from src.embedder    import Embedder
from src.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("RAG Q&A Bot — Indexing Step")
    logger.info("=" * 60)

    # 1. Ingest
    logger.info("Step 1/4  Loading documents from: %s", DATA_DIR)
    records = load_documents(DATA_DIR)

    # 2. Chunk
    logger.info("Step 2/4  Chunking (size=%d, overlap=%d) …", CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunk_records(records, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info("  → %d chunks created", len(chunks))

    # 3. Embed
    logger.info("Step 3/4  Embedding with model: %s", EMBEDDING_MODEL)
    embedder   = Embedder(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    logger.info("  → Embeddings shape: %s", embeddings.shape)

    # 4. Store
    logger.info("Step 4/4  Building & saving FAISS index …")
    store = VectorStore(FAISS_INDEX_FILE, METADATA_FILE)
    store.build(chunks, embeddings)

    logger.info("=" * 60)
    logger.info("Indexing complete!  Vector store saved to: %s", VECTORSTORE_DIR)
    logger.info("Run  python query.py  to start the Q&A bot.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
