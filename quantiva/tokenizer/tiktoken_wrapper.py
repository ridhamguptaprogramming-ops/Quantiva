"""
tiktoken tokenizer wrapper.

Adapter that exposes OpenAI's ``tiktoken`` library through the framework's
unified ``Tokenizer`` interface. Supports the well-known encodings
(``gpt2``, ``cl100k_base``, ``o200k_base``) and custom encodings registered
from an exported tiktoken BPE vocabulary.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from quantiva.tokenizer.base import Tokenizer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import tiktoken # type: ignore
except ImportError:  # pragma: no cover
    tiktoken = None


class TiktokenTokenizer(Tokenizer):
    """
    Wrapper around a ``tiktoken`` encoding.

    Args:
        encoding_name: One of ``"gpt2"``, ``"cl100k_base"``, ``"o200k_base"``,
            or a custom encoding name registered via ``tiktoken``.
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        if tiktoken is None:
            raise ImportError(
                "tiktoken is not installed. Run `pip install tiktoken`."
            )
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        logger.info("Loaded tiktoken encoding: %s", encoding_name)

    @classmethod
    def from_special_tokens(
        cls,
        encoding_name: str,
        special_tokens: Optional[dict],
    ) -> "TiktokenTokenizer":
        """
        Create a tokenizer from an encoding plus extra special tokens.

        Args:
            encoding_name: Base encoding (e.g. ``"gpt2"``).
            special_tokens: Mapping of special token string -> token id.
        """
        if tiktoken is None:
            raise ImportError(
                "tiktoken is not installed. Run `pip install tiktoken`."
            )
        obj = cls.__new__(cls)
        obj.encoding_name = encoding_name
        obj.enc = tiktoken.Encoding(
            name=encoding_name,
            pat_str=tiktoken.get_encoding(encoding_name)._pat_str,
            mergeable_ranks=tiktoken.get_encoding(encoding_name)._mergeable_ranks,
            special_tokens=special_tokens or {},
        )
        return obj

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        return self.enc.encode(text, allowed_special=allowed_special)

    def encode_batch(self, texts: Iterable[str], allowed_special: str = "all") -> List[List[int]]:
        return self.enc.encode_batch(list(texts), allowed_special=allowed_special)

    def decode(self, ids: Iterable[int]) -> str:
        return self.enc.decode(list(ids))

    def decode_batch(self, batches: Iterable[Iterable[int]]) -> List[str]:
        return self.enc.decode_batch([list(b) for b in batches])

    def encode_single_piece(self, text: str) -> int:
        """Encode a single byte sequence to one token id (for BPE validation)."""
        return self.enc.encode_single_piece(text)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        """Decode a single token to its raw byte representation."""
        return self.enc.decode_single_token_bytes(token_id)

    def export_vocab(self, path: str) -> None:
        """Export mergeable ranks to a tiktoken-style vocab file."""
        ranks = sorted(self.enc._mergeable_ranks.items(), key=lambda kv: kv[1])
        with open(path, "wb") as f:
            for token_bytes, rank in ranks:
                f.write(token_bytes + b" " + str(rank).encode() + b"\n")

    @classmethod
    def load(cls, path: str) -> "TiktokenTokenizer":
        """Load a tiktoken encoding by name (path is interpreted as encoding name)."""
        return cls(path)

