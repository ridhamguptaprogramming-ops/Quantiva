"""
Data package.

Dataset + dataloader infrastructure for training language models:
  - Memory-mapped token datasets (``.bin`` files)
  - Streaming / block-based sampling with cross-batch continuity
  - Preprocessing utilities (chunking, chat formatting)
"""

from quantiva.data.dataset import TokenDataset, MemmapDataset
from quantiva.data.dataloader import DataLoader, get_batch

__all__ = ["TokenDataset", "MemmapDataset", "DataLoader", "get_batch"]

