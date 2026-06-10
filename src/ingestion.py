"""
src/ingestion.py
----------------
Loads documents from the /data folder.
Supports: .txt, .pdf, .docx
Returns a list of dicts: { "text": str, "source": str, "page": int }
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_txt(filepath: Path) -> List[Dict[str, Any]]:
    """Load a plain-text file as a single page."""
    text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    return [{"text": text, "source": filepath.name, "page": 1}]


def load_pdf(filepath: Path) -> List[Dict[str, Any]]:
    """Load a PDF, one dict per page."""
    try:
        import pypdf  # pypdf >= 3.x
        reader = pypdf.PdfReader(str(filepath))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"text": text, "source": filepath.name, "page": i})
        return pages
    except ImportError:
        raise ImportError("pypdf is required for PDF support.  pip install pypdf")


def load_docx(filepath: Path) -> List[Dict[str, Any]]:
    """Load a .docx file, treating each paragraph as part of one page."""
    try:
        import docx as python_docx
        doc = python_docx.Document(str(filepath))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": full_text, "source": filepath.name, "page": 1}]
    except ImportError:
        raise ImportError("python-docx is required for DOCX support.  pip install python-docx")


def load_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Walk *data_dir* and load every supported file.
    Returns a flat list of page-level records.
    """
    LOADERS = {
        ".txt":  load_txt,
        ".pdf":  load_pdf,
        ".docx": load_docx,
    }

    records: List[Dict[str, Any]] = []
    files = sorted(data_dir.glob("*"))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    for fp in files:
        ext = fp.suffix.lower()
        if ext not in LOADERS:
            logger.debug("Skipping unsupported file: %s", fp.name)
            continue
        try:
            logger.info("Loading %s …", fp.name)
            pages = LOADERS[ext](fp)
            records.extend(pages)
            logger.info("  → %d page(s)", len(pages))
        except Exception as exc:
            logger.error("Failed to load %s: %s", fp.name, exc)

    logger.info("Total pages loaded: %d from %d file(s)", len(records), len(files))
    return records
