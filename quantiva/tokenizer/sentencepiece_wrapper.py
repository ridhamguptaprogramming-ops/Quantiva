"""
SentencePiece tokenizer wrapper.

Thin, production-oriented wrapper around Google's ``sentencepiece`` library so
it conforms to the framework's unified ``Tokenizer`` interface. SentencePiece
is a language-independent subword tokenizer supporting both BPE and Unigram
model types.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

from quantiva.tokenizer.base import Tokenizer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard for optional dependency
    import sentencepiece as spm # type: ignore
except ImportError:  # pragma: no cover
    spm = None


class SentencePieceTokenizer(Tokenizer):
    """
    Wrapper around ``sentencepiece.SentencePieceProcessor``.

    Args:
        model_path: Path to a pre-trained ``.model`` file.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if spm is None:
            raise ImportError(
                "sentencepiece is not installed. Run `pip install sentencepiece`."
            )
        self.sp = spm.SentencePieceProcessor()
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @staticmethod
    def train(
        corpus_path: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        max_sentence_length: int = 4192,
        num_threads: int = os.cpu_count() or 4,
        hard_vocab_limit: bool = False,
        **kwargs,
    ) -> "SentencePieceTokenizer":
        """
        Train a SentencePiece model on a raw-text corpus file.

        Args:
            corpus_path: Path to a plain-text corpus (one sentence per line).
            model_prefix: Output prefix; creates ``{prefix}.model`` and
                ``{prefix}.vocab``.
            vocab_size: Target vocabulary size.
            model_type: ``"bpe"`` or ``"unigram"``.
            character_coverage: Fraction of characters covered by the model.
            max_sentence_length: Longest sentence used in training.
            num_threads: Parallelism for training.
            hard_vocab_limit: If True, the vocab size is a hard limit.
        """
        if spm is None:
            raise ImportError(
                "sentencepiece is not installed. Run `pip install sentencepiece`."
            )
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            max_sentence_length=max_sentence_length,
            num_threads=num_threads,
            hard_vocab_limit=hard_vocab_limit,
            **kwargs,
        )
        logger.info(
            "Trained SentencePiece model: %s.model (vocab=%d)",
            model_prefix,
            vocab_size,
        )
        return SentencePieceTokenizer(f"{model_prefix}.model")

    def load(self, model_path: str) -> None:
        """Load a trained model from disk."""
        self.sp.Load(model_path)
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return self.sp.encode(list(texts), out_type=int)

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode(list(ids))

    def decode_batch(self, batches: Iterable[Iterable[int]]) -> List[str]:
        return self.sp.decode([list(b) for b in batches])

    def save(self, path: str) -> None:
        """Save the model to ``path`` (a ``.model`` file)."""
        self.sp.SaveModel(path)
        self.model_path = path

    @classmethod
    def load_from(cls, path: str) -> "SentencePieceTokenizer":
        """Alias for ``cls(path)`` — load a model file."""
        return cls(path)

    # ------------------------------------------------------------------
    # Extra SentencePiece conveniences
    # ------------------------------------------------------------------
    def id_to_piece(self, token_id: int) -> str:
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.PieceToId(piece)

    def is_unknown(self, token_id: int) -> bool:
        return self.sp.IsUnknown(token_id)

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    def export_vocab(self, path: str) -> None:
        """Export the vocabulary to a text file (one piece per line)."""
        with open(path, "w", encoding="utf-8") as f:
            for i in range(self.vocab_size):
                f.write(self.id_to_piece(i) + "\n")

