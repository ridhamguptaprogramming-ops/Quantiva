"""
Datasets.

Provides:
  - ``MemmapDataset``: memory-mapped access to a flat ``uint16`` token array
    (the nanoGPT/openwebtext format).
  - ``TokenDataset``: a PyTorch ``Dataset`` that yields (input, target)
    context blocks from a token stream.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MemmapDataset:
    """
    Memory-mapped access to a flat token array stored as uint16.

    The file layout matches the classic ``train.bin`` / ``val.bin`` format:
    raw uint16 little-endian token ids, one continuous stream.
    """

    def __init__(self, path: str, dtype: str = "uint16") -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.data[idx])

    def get_slice(self, start: int, end: int) -> np.ndarray:
        """Return a slice as a plain numpy array (useful for batching)."""
        return self.data[start:end]

    def close(self) -> None:
        """Release the memory map."""
        if hasattr(self.data, "_mmap") and self.data._mmap is not None:
            self.data._mmap.close()

    @staticmethod
    def write_tokens(path: str, tokens: np.ndarray, dtype: str = "uint16") -> str:
        """Write a token array to disk as a uint16 binary file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arr = np.asarray(tokens, dtype=np.dtype(dtype))
        arr.tofile(path)
        logger.info("Wrote %d tokens to %s", arr.size, path)
        return path


class TokenDataset(Dataset):
    """
    PyTorch dataset over a token stream that yields ``(input, target)`` pairs.

    Each sample is a context window of ``block_size`` tokens; the target is the
    same sequence shifted right by one. Supports both in-memory numpy arrays
    and memory-mapped files.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        block_size: int = 1024,
        stride: int = 1,
    ) -> None:
        """
        Args:
            tokens: flat token array (numpy).
            block_size: context length.
            stride: offset between consecutive samples (1 = fully overlapping).
        """
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride
        self._length = max(0, (len(tokens) - block_size) // stride)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return (
            torch.from_numpy(np.asarray(x, dtype=np.int64)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        )

    @classmethod
    def from_memmap(cls, path: str, block_size: int, stride: int = 1) -> "TokenDataset":
        """Build a TokenDataset directly from a ``.bin`` memmap file."""
        mm = MemmapDataset(path)
        tokens = np.asarray(mm.data[:], dtype=np.int64)
        mm.close()
        return cls(tokens, block_size=block_size, stride=stride)

