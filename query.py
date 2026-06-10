#!/usr/bin/env python3
"""
query.py
--------
Interactive command-line Q&A loop.

Run with:
  python query.py

Optional flags:
  --top-k INT     number of chunks to retrieve (default from config)
  --verbose       print retrieved chunks before the answer
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    FAISS_INDEX_FILE, METADATA_FILE,
    TOP_K,
)
from src.embedder    import Embedder
from src.vectorstore import VectorStore
from src.generator   import generate_answer

logging.basicConfig(
    level=logging.WARNING,          # quiet mode during interactive use
    format="%(levelname)s  %(message)s",
)


BANNER = """
╔══════════════════════════════════════════════════════╗
║          RAG Document Q&A Bot — Science Edition      ║
║  Knowledge base: Quantum Mechanics · CRISPR ·        ║
║                  Black Holes · Climate Science        ║
╚══════════════════════════════════════════════════════╝
Type your question and press Enter.
Type  'quit' or 'exit' to stop.
Type  'sources' to list indexed documents.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Q&A Bot — interactive CLI")
    parser.add_argument("--top-k",   type=int, default=TOP_K,
                        help="Number of chunks to retrieve (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved source chunks before the answer")
    return parser.parse_args()


def print_sources(metadata_file: Path) -> None:
    import json
    if not metadata_file.exists():
        print("  [Index not built yet. Run  python index.py  first.]\n")
        return
    meta = json.loads(metadata_file.read_text())
    sources = sorted({m["source"] for m in meta})
    print("\nIndexed documents:")
    for s in sources:
        print(f"  • {s}")
    print()


def format_retrieved(chunks: list) -> str:
    lines = ["\n── Retrieved Passages ──────────────────────────────"]
    for i, c in enumerate(chunks, 1):
        snippet = c["text"][:200].replace("\n", " ")
        lines.append(
            f"[{i}] {c['source']}  (page {c['page']}, score {c['score']:.3f})\n"
            f"    {snippet}…"
        )
    lines.append("────────────────────────────────────────────────────\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    print(BANNER)

    # Check index exists
    if not FAISS_INDEX_FILE.exists():
        print("⚠  Vector store not found. Building index now …\n")
        from index import main as build_index
        build_index()

    # Load models once
    print("Loading embedding model …")
    embedder = Embedder(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    store    = VectorStore(FAISS_INDEX_FILE, METADATA_FILE)
    store.load()
    print(f"Ready! ({store._index.ntotal} chunks indexed, top-k={args.top_k})\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if query.lower() == "sources":
            print_sources(METADATA_FILE)
            continue

        # Retrieve
        query_emb = embedder.embed_query(query)
        chunks    = store.search(query_emb, top_k=args.top_k)

        if args.verbose:
            print(format_retrieved(chunks))

        # Generate
        try:
            answer = generate_answer(
                query      = query,
                chunks     = chunks,
                api_key    = GROQ_API_KEY,
                model      = LLM_MODEL,
                max_tokens = LLM_MAX_TOKENS,
            )
        except ValueError as exc:
            print(f"\n⚠  Configuration error: {exc}\n")
            continue
        except Exception as exc:
            print(f"\n⚠  API error: {exc}\n")
            continue

        print(f"\nBot: {answer}\n")
        print("─" * 60)

        # Always show source filenames for citations
        seen = set()
        sources_used = []
        for c in chunks:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                sources_used.append(f"{c['source']} (p.{c['page']})")
        print(f"Sources consulted: {', '.join(sources_used)}\n")


if __name__ == "__main__":
    main()
