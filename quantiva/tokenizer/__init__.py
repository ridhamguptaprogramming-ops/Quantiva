"""
Tokenizer package.

Provides a unified ``Tokenizer`` interface with pluggable backends:
  - Byte-Pair Encoding trained from scratch (``bpe.py``)
  - SentencePiece (``sentencepiece_wrapper.py``)
  - tiktoken / GPT-2 / cl100k (``tiktoken_wrapper.py``)

A factory (``factory.py``) instantiates the correct backend from config.
"""

from quantiva.tokenizer.base import Tokenizer
from quantiva.tokenizer.factory import get_tokenizer

__all__ = ["Tokenizer", "get_tokenizer"]

