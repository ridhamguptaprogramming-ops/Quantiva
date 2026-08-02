"""
Datasets package.

Higher-level dataset construction helpers, including document chunking and
chat/SFT formatting utilities.
"""

from quantiva.datasets.preprocessing.chunking import chunk_text, chunk_documents, Chunker
from quantiva.datasets.preprocessing.formatting import (
    apply_chat_template,
    format_sft_example,
    ChatMessage,
)

__all__ = [
    "chunk_text",
    "chunk_documents",
    "Chunker",
    "apply_chat_template",
    "format_sft_example",
    "ChatMessage",
]

