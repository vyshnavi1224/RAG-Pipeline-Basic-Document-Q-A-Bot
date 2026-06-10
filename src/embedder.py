"""
src/embedder.py
---------------
Generates dense vector embeddings for text chunks and queries
using a local sentence-transformers model.

All embedding calls are BATCHED (never one chunk at a time),
satisfying the technical requirement.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper around sentence-transformers.

    Parameters
    ----------
    model_name  : HuggingFace model identifier
    batch_size  : chunks processed per forward pass
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64) -> None:
        logger.info("Loading embedding model: %s …", model_name)
        from sentence_transformers import SentenceTransformer
        self.model      = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dim        = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dim)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Returns
        -------
        np.ndarray of shape (len(texts), dim), dtype float32
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        logger.info("Embedding %d texts in batches of %d …", len(texts), self.batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity = dot product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self.embed_texts([query])[0]
