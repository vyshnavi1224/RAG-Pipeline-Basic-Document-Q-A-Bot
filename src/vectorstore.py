"""
src/vectorstore.py
------------------
Persists chunk embeddings in a FAISS index on disk and provides
similarity search.

Two public operations:
  build()   – index chunks and save to disk
  search()  – load from disk, embed query, return top-k chunks
"""
from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store with JSON metadata sidecar.

    Parameters
    ----------
    index_path    : path to the FAISS .index file
    metadata_path : path to the JSON metadata file
    """

    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path    = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._index        = None
        self._metadata: List[Dict[str, Any]] = []

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Create a new FAISS index from *embeddings* and persist it.

        Parameters
        ----------
        chunks     : list of chunk dicts (text, source, page, chunk_id)
        embeddings : float32 array of shape (n_chunks, dim)
        """
        import faiss

        n, dim = embeddings.shape
        logger.info("Building FAISS index: %d vectors, dim=%d", n, dim)

        # Inner-product index (cosine similarity because embeddings are normalised)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Persist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        metadata = [
            {
                "chunk_id": c["chunk_id"],
                "source":   c["source"],
                "page":     c["page"],
                "text":     c["text"],
            }
            for c in chunks
        ]
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Index saved → %s", self.index_path)
        logger.info("Metadata saved → %s", self.metadata_path)
        self._index    = index
        self._metadata = metadata

    # ── Load ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load a previously built index from disk."""
        import faiss

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index at {self.index_path}. "
                "Run the indexing step first:  python index.py"
            )
        logger.info("Loading FAISS index from %s …", self.index_path)
        self._index    = faiss.read_index(str(self.index_path))
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        logger.info("Index loaded: %d vectors", self._index.ntotal)

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return the *top_k* most similar chunks for *query_embedding*.

        Parameters
        ----------
        query_embedding : shape (dim,) float32
        top_k           : number of results

        Returns
        -------
        List of dicts: {text, source, page, chunk_id, score}
        """
        if self._index is None:
            self.load()

        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    @property
    def is_built(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()
