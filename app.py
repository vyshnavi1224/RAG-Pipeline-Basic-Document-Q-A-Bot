"""
app.py  —  Streamlit Web UI (bonus)
-------------------------------------
Run with:
  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    FAISS_INDEX_FILE, METADATA_FILE,
    TOP_K,
)
from src.embedder    import Embedder
from src.vectorstore import VectorStore
from src.generator   import generate_answer


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Science Q&A Bot",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 RAG Document Q&A Bot")
st.caption("Science Knowledge Base: Quantum Mechanics · CRISPR · Black Holes · Climate Science")
st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)
    show_chunks = st.toggle("Show retrieved passages", value=False)
    st.divider()
    st.markdown(
        "**Knowledge base documents:**\n"
        "- quantum_mechanics_fundamentals.txt\n"
        "- crispr_gene_editing.txt\n"
        "- black_holes_general_relativity.txt\n"
        "- climate_science_global_warming.pdf\n"
    )
    st.divider()
    st.markdown(
        "**Example questions:**\n"
        "- What is the Heisenberg uncertainty principle?\n"
        "- How does CRISPR-Cas9 cut DNA?\n"
        "- What is Hawking radiation?\n"
        "- What are climate tipping points?\n"
        "- What is SSP5-8.5?\n"
    )


# ── Load models (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model …")
def load_resources():
    embedder = Embedder(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    store    = VectorStore(FAISS_INDEX_FILE, METADATA_FILE)
    if not store.is_built:
        st.warning("Index not found — building now. This may take a moment …")
        from index import main as build_index
        build_index()
    store.load()
    return embedder, store


embedder, store = load_resources()
st.success(f"✅ Index loaded — {store._index.ntotal} chunks ready.", icon="✅")


# ── Chat history ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Input ──────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a science question …")

if query:
    # Add user message
    st.session_state.history.append({"role": "user", "content": query})

    with st.spinner("Retrieving and generating answer …"):
        q_emb  = embedder.embed_query(query)
        chunks = store.search(q_emb, top_k=top_k)
        try:
            answer = generate_answer(
                query      = query,
                chunks     = chunks,
                api_key    = GROQ_API_KEY,
                model      = LLM_MODEL,
                max_tokens = LLM_MAX_TOKENS,
            )
        except ValueError as exc:
            answer = f"⚠️ Configuration error: {exc}"
        except Exception as exc:
            answer = f"⚠️ API error: {exc}"

    # Build citation footer
    seen = set()
    cites = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            cites.append(f"`{c['source']}` (p.{c['page']})")

    full_answer = answer
    if cites:
        full_answer += f"\n\n---\n**Sources consulted:** {' · '.join(cites)}"

    st.session_state.history.append({
        "role": "assistant",
        "content": full_answer,
        "chunks": chunks,
    })


# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if show_chunks and msg["role"] == "assistant" and "chunks" in msg:
            with st.expander("📄 Retrieved passages"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(
                        f"**[{i}] {c['source']}** — page {c['page']}  "
                        f"*(score: {c['score']:.3f})*"
                    )
                    st.text(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                    st.divider()
