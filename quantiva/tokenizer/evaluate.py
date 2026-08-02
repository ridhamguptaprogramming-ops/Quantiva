"""
Tokenizer evaluation utilities.

Measures compression rate, round-trip fidelity, and vocabulary statistics for a
given tokenizer on a corpus. Useful when deciding between backends/vocab sizes.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, List

from quantiva.tokenizer.base import Tokenizer

logger = logging.getLogger(__name__)


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    texts: Iterable[str],
    reference_utf8: bool = True,
) -> dict:
    """
    Evaluate a tokenizer over a corpus.

    Metrics returned:
      - ``n_samples``: number of samples evaluated.
      - ``avg_tokens_per_sample``: mean token count per sample.
      - ``avg_chars_per_sample``: mean character count per sample.
      - ``compression_ratio``: bytes / tokens (higher = more compact).
      - ``roundtrip_accuracy``: fraction of samples that survive
        encode(decode) unchanged.
      - ``vocab_size``: size of the vocabulary.
      - ``special_token_usage``: counts of special tokens if any are present.

    Args:
        tokenizer: A ``Tokenizer`` instance.
        texts: Corpus of raw strings.
        reference_utf8: If True, base compression on UTF-8 byte length.
    """
    n = 0
    total_tokens = 0
    total_bytes = 0
    roundtrip_ok = 0
    special_counter: Counter = Counter()
    vocab = getattr(tokenizer, "vocab_size", 0)

    for text in texts:
        n += 1
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_bytes += len(text.encode("utf-8"))
        if tokenizer.decode(ids) == text:
            roundtrip_ok += 1
        # Count special tokens (if exposed).
        for tid in ids:
            piece = _token_to_repr(tokenizer, tid)
            if piece.startswith("<|") or piece.startswith("<"):
                special_counter[piece] += 1

    results = {
        "n_samples": n,
        "avg_tokens_per_sample": total_tokens / max(n, 1),
        "avg_chars_per_sample": (total_bytes / max(n, 1)),
        "compression_ratio": total_bytes / max(total_tokens, 1),
        "roundtrip_accuracy": roundtrip_ok / max(n, 1),
        "vocab_size": vocab,
    }
    if special_counter:
        results["special_token_usage"] = dict(special_counter)
    return results


def _token_to_repr(tokenizer: Tokenizer, token_id: int) -> str:
    """Best-effort human-readable representation of a token id."""
    try:
        return tokenizer.decode([token_id])
    except Exception:  # pragma: no cover
        return f"<{token_id}>"


def token_frequency(tokenizer: Tokenizer, texts: List[str]) -> Counter:
    """Count token frequency over a corpus (useful for pruning analysis)."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenizer.encode(text))
    return counter

