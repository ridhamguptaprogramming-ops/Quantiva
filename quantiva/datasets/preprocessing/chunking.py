"""
Document chunking utilities.

Splits long documents into smaller, roughly equal-length chunks while
respecting paragraph boundaries where possible. This is used both for
pretraining (concatenation into token streams) and for RAG retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    text: str
    doc_id: Optional[str] = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 0,
    respect_paragraphs: bool = True,
) -> List[str]:
    """
    Split ``text`` into chunks of at most ``chunk_size`` characters.

    Args:
        text: Input document.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
        respect_paragraphs: If True, try to break at paragraph boundaries to
            keep semantic units intact (fall back to hard splits when a single
            paragraph exceeds chunk_size).

    Returns:
        List of chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text]

    if not respect_paragraphs:
        return _hard_split(text, chunk_size, overlap)

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # If we're not at the end and not at a boundary, try to back up to a
        # paragraph break or the last space within the window.
        if end < n:
            window = text[start:end]
            # Find the last paragraph break.
            para_match = list(PARAGRAPH_SPLIT.finditer(window))
            if para_match:
                last = para_match[-1].end()
                if last > 0:
                    end = start + last
            else:
                # Fall back to the last space/newline.
                last_space = max(window.rfind(" "), window.rfind("\n"))
                if last_space > chunk_size // 2:
                    end = start + last_space + 1

        chunks.append(text[start:end].strip())
        # Advance; overlap allows the next chunk to re-read the tail.
        start = max(end - overlap, start + 1) if overlap else end

    # Drop any empty trailing chunk.
    return [c for c in chunks if c]


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into fixed-size chunks with optional overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1) if overlap else end
    return chunks


def chunk_documents(
    documents: Iterable[str],
    doc_ids: Optional[Iterable[str]] = None,
    chunk_size: int = 512,
    overlap: int = 0,
    respect_paragraphs: bool = True,
) -> List[Chunk]:
    """
    Chunk multiple documents, returning ``Chunk`` objects with metadata.

    Args:
        documents: Iterable of document strings.
        doc_ids: Optional parallel iterable of document ids.
        chunk_size / overlap / respect_paragraphs: see :func:`chunk_text`.
    """
    out: List[Chunk] = []
    ids_list: Optional[List[str]] = list(doc_ids) if doc_ids is not None else None
    for i, doc in enumerate(documents):
        doc_id = ids_list[i] if ids_list is not None else None
        parts = chunk_text(doc, chunk_size, overlap, respect_paragraphs)
        char_cursor = 0
        for j, part in enumerate(parts):
            start = doc.find(part, char_cursor)
            if start < 0:
                start = char_cursor
            out.append(
                Chunk(
                    text=part,
                    doc_id=doc_id,
                    chunk_index=j,
                    start_char=start,
                    end_char=start + len(part),
                )
            )
            char_cursor = start + len(part)
    return out


class Chunker:
    """Reusable chunking object with fixed configuration."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 0,
        respect_paragraphs: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_paragraphs = respect_paragraphs

    def __call__(self, text: str) -> List[str]:
        return chunk_text(
            text,
            self.chunk_size,
            self.overlap,
            self.respect_paragraphs,
        )

    def chunk_documents(
        self, documents: Iterable[str], doc_ids: Optional[Iterable[str]] = None
    ) -> List[Chunk]:
        return chunk_documents(
            documents,
            doc_ids,
            self.chunk_size,
            self.overlap,
            self.respect_paragraphs,
        )

