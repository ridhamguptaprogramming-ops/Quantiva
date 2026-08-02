"""
Byte-Pair Encoding (BPE) tokenizer trained from scratch.

Implements the classic GPT-2 style byte-level BPE:
  1. Convert text to UTF-8 bytes, then map each byte to a token id.
  2. Repeatedly merge the most frequent adjacent byte-pair in the corpus,
     adding a new token id for each merge.
  3. Optionally add ``special_tokens`` (e.g. ``<|endoftext|>``) and apply a
     "merges after specials" ranking scheme compatible with OpenAI's tokenizer.

The design mirrors the reference implementation in GPT-2's ``encoder.py`` and
``minbpe``, but is reimplemented with production concerns: explicit state,
deterministic behavior, and a clean save/load format.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from quantiva.tokenizer.base import Tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level byte-pair helpers (pure functions, easily unit tested)
# ---------------------------------------------------------------------------
def bytes_to_unicode() -> Dict[int, str]:
    """
    Map every byte (0..255) to a printable unicode character.

    This is the GPT-2 byte-to-unicode table: bytes that map to whitespace or
    control characters are re-mapped into the 256..259 range so every byte has
    a unique, printable unicode representation (which lets us treat text as a
    sequence of characters for BPE merging).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> set:
    """Return the set of adjacent character pairs in ``word`` (as a tuple)."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class BPETokenizer(Tokenizer):
    """
    Trainable byte-level BPE tokenizer.

    Attributes:
        vocab: mapping token_id -> token string.
        merges: list of (pair_a, pair_b) merges, in training order.
        special_tokens: dict mapping special token string -> id.
        byte_decoder: inverse of the byte->unicode table.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        special_tokens: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        self.vocab_size_target = vocab_size
        self.special_tokens = special_tokens or []
        self.verbose = verbose
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[int, str] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        """Seed vocabulary with 0..255 (one token per byte) plus specials."""
        self.vocab = {i: self.byte_encoder[i] for i in range(256)}
        next_id = 256
        for tok in self.special_tokens:
            if tok not in self.vocab.values():
                self.vocab[next_id] = tok
                next_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        texts: Iterable[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
    ) -> None:
        """
        Train the BPE merges on a corpus of raw text.

        Algorithm (per GPT-2 / minbpe):
          1. Tokenize each document into a sequence of bytes (as unicode chars).
          2. For each document, split into words by the GPT-2 regex so that
             spaces are preserved, then represent each word as a tuple of the
             unicode-encoded byte characters.
          3. Count pair frequencies across all words, then iteratively pick the
             most frequent pair, add it to ``merges``, and merge occurrences in
             every word. Repeat until the target vocab size is reached.
        """
        target = vocab_size or self.vocab_size_target
        if target < len(self.vocab):
            raise ValueError(
                f"vocab_size {target} is smaller than the base vocab "
                f"({len(self.vocab)})."
            )

        # GPT-2 regex: split on word boundaries but keep leading whitespace.
        pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Build the corpus as a list of word-tuples for fast pair counting.
        corpus: List[List[Tuple[str, ...]]] = []
        for text in texts:
            words = text.encode("utf-8")
            # Decode bytes to the unicode space using the byte encoder.
            encoded = "".join(self.byte_encoder[b] for b in words)
            corpus.append(
                [tuple(w) for w in pat.findall(encoded)]
            )

        # Count initial pair frequencies.
        pair_counts: Counter = Counter()
        for words in corpus:
            for word in words:
                pair_counts.update(get_pairs(word))

        while len(self.vocab) < target and pair_counts:
            # Most frequent pair.
            pair, count = pair_counts.most_common(1)[0]
            if count < min_frequency:
                logger.info(
                    "Stopping early: most frequent pair has count %d < %d",
                    count,
                    min_frequency,
                )
                break
            if self.verbose:
                logger.info("Merging %r (count=%d)", pair, count)

            # Register the merge.
            merged_token = pair[0] + pair[1]
            self.merges[pair] = len(self.vocab) - len(self.special_tokens) - 1
            self.vocab[len(self.vocab)] = merged_token

            # Apply the merge to the corpus.
            new_corpus: List[List[Tuple[str, ...]]] = []
            for words in corpus:
                new_words: List[Tuple[str, ...]] = []
                for word in words:
                    new_word = self._merge_word(word, pair)
                    new_words.append(new_word)
                    if len(new_word) > 1:
                        # Decrement old pair counts and increment new ones.
                        pair_counts.subtract(get_pairs(word))
                        pair_counts.update(get_pairs(new_word))
                new_corpus.append(new_words)
            corpus = new_corpus

        logger.info("BPE training complete. vocab_size=%d", len(self.vocab))

    @staticmethod
    def _merge_word(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Replace all occurrences of ``pair`` in ``word`` with a merged token."""
        first, second = pair
        out: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                out.append(first + second)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode raw text into token ids using the trained merges."""
        pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        # Handle special tokens first.
        if self.special_tokens:
            special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
            split_special = re.compile(f"({special_pattern})")
            segments = split_special.split(text)
        else:
            segments = [text]

        ids: List[int] = []
        lookup = self.token_to_id_lookup
        for segment in segments:
            if segment in self.special_tokens:
                ids.append(lookup[segment])
                continue
            encoded = "".join(self.byte_encoder[b] for b in segment.encode("utf-8"))
            for token in pat.findall(encoded):
                ids.extend(self._encode_word(token))
        return ids

    def _encode_word(self, word: str) -> List[int]:
        """Encode a single (regex-split) word into token ids."""
        if word in self.special_tokens:
            return [self.token_to_id_lookup[word]]
        # Convert the word into a tuple of characters and iteratively merge.
        tokens = list(word)
        while len(tokens) > 1:
            pairs = get_pairs(tuple(tokens))
            # Find the merge pair with the smallest merge rank (earliest trained).
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = list(self._merge_word(tuple(tokens), pair))
        # Map each remaining character to its token id via the reverse lookup.
        lookup = self.token_to_id_lookup
        return [lookup[t] for t in tokens]

    @property
    def token_to_id_lookup(self) -> Dict[str, int]:
        if getattr(self, "_reverse", None) is None:
            self._reverse = {v: k for k, v in self.vocab.items()}
        return self._reverse

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back into a string (byte level)."""
        parts: List[str] = []
        for token_id in ids:
            s = self.vocab[token_id]
            if s in self.special_tokens:
                parts.append(s)
            else:
                # Map unicode back to the original byte.
                byte = self.byte_decoder[s]
                parts.append(chr(byte))
        return "".join(parts).encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save tokenizer state to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "vocab_size_target": self.vocab_size_target,
            "special_tokens": self.special_tokens,
            "merges": [list(k) + [v] for k, v in self.merges.items()],
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved tokenizer to %s", path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(
            vocab_size=payload.get("vocab_size_target", 50257),
            special_tokens=payload.get("special_tokens", []),
        )
        tok.merges = {
            (str(k[0]), str(k[1])): int(k[2])
            for k in payload["merges"]
        }
        tok.vocab = {int(k): str(v) for k, v in payload["vocab"].items()}
        tok._reverse = {v: k for k, v in tok.vocab.items()}
        return tok

    def export_vocab(self, path: str) -> None:
        """Export the vocabulary as a simple text file (one token per line)."""
        with open(path, "w", encoding="utf-8") as f:
            for token_id in range(len(self.vocab)):
                f.write(self.vocab[token_id] + "\n")

