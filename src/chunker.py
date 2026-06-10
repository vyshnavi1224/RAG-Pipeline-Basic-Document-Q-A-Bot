"""
src/chunker.py
--------------
Splits page-level text records into smaller, overlapping chunks.

Strategy: Paragraph-aware chunking
  - Split text on double newlines (paragraph boundaries) first.
  - Accumulate paragraphs until the running char count would exceed
    CHUNK_SIZE, then flush and start the next chunk with CHUNK_OVERLAP
    characters of look-back.
  - This preserves natural paragraph boundaries while keeping chunks
    within a predictable token budget.

Why paragraph-aware?
  Scientific documents are written in self-contained paragraphs.
  Breaking inside a paragraph loses the conceptual unit; breaking at
  paragraph boundaries keeps each chunk coherent and improves retrieval
  precision.

Each output chunk carries metadata:
  - source  : original filename
  - page    : page number within the source document
  - chunk_id: monotonically increasing int across the whole corpus
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _split_paragraphs(text: str) -> List[str]:
    """Return a list of non-empty paragraphs from *text*."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def chunk_records(
    records: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Convert page-level records into overlapping text chunks.

    Parameters
    ----------
    records      : output of ingestion.load_documents()
    chunk_size   : target character length for each chunk
    chunk_overlap: characters of previous chunk to prepend to the next

    Returns
    -------
    List of chunk dicts with keys: text, source, page, chunk_id
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for record in records:
        source = record["source"]
        page   = record["page"]
        text   = record["text"]

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            continue

        current_parts: List[str] = []
        current_len = 0
        overlap_text = ""

        for para in paragraphs:
            para_len = len(para)

            # If adding this paragraph keeps us under the limit, accumulate.
            if current_len + para_len <= chunk_size or not current_parts:
                current_parts.append(para)
                current_len += para_len + 1  # +1 for the joining newline
            else:
                # Flush current chunk
                chunk_text = overlap_text + "\n\n".join(current_parts)
                chunks.append({
                    "text":     chunk_text.strip(),
                    "source":   source,
                    "page":     page,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

                # Compute overlap: tail of the flushed chunk
                flushed = "\n\n".join(current_parts)
                overlap_text = flushed[-chunk_overlap:] if len(flushed) > chunk_overlap else flushed
                overlap_text = overlap_text.lstrip() + "\n\n" if overlap_text else ""

                # Start new chunk with current paragraph
                current_parts = [para]
                current_len   = para_len

        # Flush remaining parts
        if current_parts:
            chunk_text = overlap_text + "\n\n".join(current_parts)
            chunks.append({
                "text":     chunk_text.strip(),
                "source":   source,
                "page":     page,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    logger.info("Chunking complete: %d chunks from %d pages.", len(chunks), len(records))
    return chunks
