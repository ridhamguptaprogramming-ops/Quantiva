"""
Quantiva — A production-grade, from-scratch LLM framework.

Inspired by nanoGPT/nanochat, rebuilt with clean architecture, scalability,
and modern engineering practices. Capable of training and serving real
GPT-style language models.
"""

__version__ = "0.1.0"

from quantiva.model.gpt import GPT, GPTConfig
from quantiva.tokenizer.base import Tokenizer
from quantiva.tokenizer.factory import get_tokenizer

__all__ = [
    "GPT",
    "GPTConfig",
    "Tokenizer",
    "get_tokenizer",
    "__version__",
]

