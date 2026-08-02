"""
Tokenizer factory.

Instantiates the correct tokenizer backend from a configuration dict. This
keeps the rest of the framework decoupled from specific tokenizer libraries
(SOLID: dependency inversion).
"""

from __future__ import annotations

import logging
from typing import Optional

from quantiva.tokenizer.base import Tokenizer
from quantiva.tokenizer.bpe import BPETokenizer
from quantiva.tokenizer.sentencepiece_wrapper import SentencePieceTokenizer
from quantiva.tokenizer.tiktoken_wrapper import TiktokenTokenizer

logger = logging.getLogger(__name__)

# Registry of available backends for easy introspection.
TOKENIZER_REGISTRY = {
    "bpe": BPETokenizer,
    "sentencepiece": SentencePieceTokenizer,
    "tiktoken": TiktokenTokenizer,
}


def get_tokenizer(
    backend: str = "tiktoken",
    **kwargs,
) -> Tokenizer:
    """
    Factory entry point.

    Args:
        backend: One of ``"bpe"``, ``"sentencepiece"``, ``"tiktoken"``.
        **kwargs: Backend-specific keyword arguments.

    Examples:
        >>> get_tokenizer("tiktoken", encoding_name="gpt2")
        >>> get_tokenizer("sentencepiece", model_path="sp.model")
        >>> get_tokenizer("bpe", vocab_size=8192, special_tokens=["<|endoftext|>"])
    """
    backend = backend.lower().replace("-", "_").replace(" ", "_")
    if backend not in TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer backend '{backend}'. "
            f"Available: {sorted(TOKENIZER_REGISTRY)}"
        )
    cls = TOKENIZER_REGISTRY[backend]
    logger.info("Instantiating tokenizer: %s(%s)", cls.__name__, kwargs)
    return cls(**kwargs)

