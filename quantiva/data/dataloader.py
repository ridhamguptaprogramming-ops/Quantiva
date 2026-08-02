"""
Dataloader.

Implements the classic nanoGPT-style streaming batch sampler:
  - Memory-maps a flat token array.
  - Maintains a cursor per process so consecutive batches are contiguous
    (the model sees an uninterrupted token stream across batch boundaries).
  - Supports distributed training (each rank sees a different slice of data).

This avoids the wasted tokens of random block sampling and is the standard
approach for training GPT-style models on concatenated documents.
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Streaming token dataloader with continuity across batches.

    Args:
        path: Path to a ``.bin`` file of raw uint16 tokens.
        batch_size: Micro-batch size (tokens per row).
        block_size: Context length (tokens per sample).
        device: Target device for batches.
        dtype: Numpy dtype of the token file.
        process_rank: Rank in distributed training (0 for single process).
        num_processes: Total number of processes (1 for single process).
        seed: Random seed for the initial offset.
        split: "train" or "val" — used only to pick a consistent offset seed.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        dtype: str = "uint16",
        process_rank: int = 0,
        num_processes: int = 1,
        seed: int = 1337,
        split: str = "train",
    ) -> None:
        self.path = path
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Open memory-mapped file.
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        n_tokens = len(self.data)
        if n_tokens < block_size:
            raise ValueError(
                f"Token file has {n_tokens} tokens, less than block_size {block_size}"
            )

        # Total number of full samples available.
        n_batches = n_tokens // (batch_size * block_size)
        self.n_batches = n_batches

        # Compute the number of tokens available to *this* process.
        tokens_per_proc = (n_tokens - block_size) // num_processes
        self.local_n_tokens = tokens_per_proc
        # Offset for this process (each rank gets a disjoint contiguous slice).
        self.start_offset = process_rank * tokens_per_proc

        # Random initial offset within this process's slice (per epoch).
        self.epoch = 0
        rng = random.Random(seed + split.__hash__() % 2**31)
        self.offset = rng.randint(0, max(0, tokens_per_proc - block_size))

        logger.info(
            "DataLoader: %s | batch=%d block=%d | proc %d/%d | batches/proc=%d",
            path, batch_size, block_size, process_rank, num_processes, n_batches,
        )

    def reset(self) -> None:
        """Reset the epoch counter and re-randomize the starting offset."""
        self.epoch = 0
        self.offset = random.Random(1337 + self.process_rank).randint(
            0, max(0, self.local_n_tokens - self.block_size)
        )

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return one ``(input, target)`` micro-batch of shape
        ``(batch_size, block_size)`` on ``self.device``.
        """
        if self.offset + self.batch_size * self.block_size + 1 > self.local_n_tokens:
            # We've exhausted this process's slice — restart from the beginning
            # of the slice (this functions as one "epoch").
            self.offset = 0
            self.epoch += 1

        buf = self.data[
            self.start_offset + self.offset :
            self.start_offset + self.offset + self.batch_size * self.block_size + 1
        ]
        buf = np.asarray(buf, dtype=np.int64)

        x = buf[:-1].view(self.batch_size, self.block_size)
        y = buf[1:].view(self.batch_size, self.block_size)

        self.offset += self.batch_size * self.block_size
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
        )

    def estimate_total_batches(self) -> int:
        """Total number of batches across all epochs for this process."""
        return self.n_batches


def get_batch(
    split: str,
    train_path: str,
    val_path: str,
    batch_size: int,
    block_size: int,
    device: str,
    process_rank: int = 0,
    num_processes: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional helper: return a batch from either the train or val stream.
    Uses module-level loaders that persist across calls to preserve continuity.
    """
    global _train_loader, _val_loader
    if split == "train":
        if _train_loader is None or _train_loader.path != train_path:
            _train_loader = DataLoader(
                train_path, batch_size, block_size, device,
                process_rank=process_rank, num_processes=num_processes,
            )
        return _train_loader.get_batch()
    else:
        if _val_loader is None or _val_loader.path != val_path:
            _val_loader = DataLoader(
                val_path, batch_size, block_size, device,
                process_rank=process_rank, num_processes=num_processes,
            )
        return _val_loader.get_batch()


# Module-level loaders for the functional API.
_train_loader: Optional[DataLoader] = None
_val_loader: Optional[DataLoader] = None

