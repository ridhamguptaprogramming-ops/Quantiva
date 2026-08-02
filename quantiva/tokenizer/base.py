"""
Abstract tokenizer interface.

All tokenizer backends (BPE, SentencePiece, tiktoken) implement this protocol
so the rest of the framework can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional


class Tokenizer(ABC):
    """Unified tokenizer interface."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token ids."""

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str:
        """Decode a list of token ids back into a string."""

    # ------------------------------------------------------------------
    # Optional helpers with sensible defaults
    # ------------------------------------------------------------------
    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        """Encode a batch of strings. Override for performance."""
        return [self.encode(t) for t in texts]

    def decode_batch(self, batches: Iterable[Iterable[int]]) -> List[str]:
        """Decode a batch of token-id sequences. Override for performance."""
        return [self.decode(ids) for ids in batches]

    def save(self, path: str) -> None:
        """Persist tokenizer state to ``path``. Override as needed."""

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load a tokenizer from ``path``. Override as needed."""
        raise NotImplementedError

    def evaluate(
        self,
        texts: Iterable[str],
        expected_tokens: Optional[List[int]] = None,
    ) -> dict:
        """
        Evaluate the tokenizer on a corpus.

        Returns round-trip accuracy (encode→decode identity), average tokens
        per sample, and (if provided) exact-match accuracy against expected ids.
        """
        n_roundtrip_ok = 0
        n_total = 0
        total_tokens = 0
        for text in texts:
            n_total += 1
            ids = self.encode(text)
            total_tokens += len(ids)
            if self.decode(ids) == text:
                n_roundtrip_ok += 1
        return {
            "roundtrip_accuracy": n_roundtrip_ok / max(n_total, 1),
            "avg_tokens_per_sample": total_tokens / max(n_total, 1),
            "vocab_size": self.vocab_size,
        }

