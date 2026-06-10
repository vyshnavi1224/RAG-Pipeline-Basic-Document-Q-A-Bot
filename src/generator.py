"""
src/generator.py
----------------
Generates answers from retrieved context chunks using Groq (free tier).

Groq provides a free API with generous rate limits powered by fast LPU hardware.
Default model: llama-3.1-8b-instant (free, very fast).

Get your free key at: https://console.groq.com

Key design decisions:
  - System prompt explicitly forbids answering outside retrieved context.
  - Each answer must include source citations.
  - If none of the chunks are relevant, the bot says so.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document Q&A assistant. Your ONLY knowledge source is
the context passages provided below. Follow these rules strictly:

1. Answer ONLY from the provided context. Do NOT use your own training knowledge.
2. If the answer is not in the context, respond exactly:
   "I could not find an answer to this question in the provided documents."
3. Always cite your sources using the format [source: FILENAME, page: PAGE_NUM] after
   each claim or sentence that comes from a specific passage.
4. Keep answers clear, factual, and concise.
5. If multiple sources support the answer, cite all of them."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Passage {i}] Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 1024,
) -> str:
    """
    Call the Groq API and return a grounded answer with citations.

    Parameters
    ----------
    query      : user's natural language question
    chunks     : top-k retrieved chunk dicts from VectorStore.search()
    api_key    : Groq API key (from environment variable GROQ_API_KEY)
    model      : Groq model identifier (default: llama-3.1-8b-instant)
    max_tokens : maximum tokens in the response

    Returns
    -------
    str  – the model's answer with citation markers
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package is required.  pip install groq")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "  1. Get a free key at https://console.groq.com\n"
            "  2. Export it:  export GROQ_API_KEY=gsk_..."
        )

    context_block = build_context_block(chunks)
    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    client = Groq(api_key=api_key)

    logger.debug("Calling Groq model: %s", model)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response.choices[0].message.content
