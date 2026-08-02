"""
Model package.

Implements a modern GPT-style decoder-only transformer from scratch:
  - Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)
  - Rotary Positional Embeddings (RoPE)
  - LayerNorm and RMSNorm
  - GELU and SwiGLU MLPs
  - Residual connections, dropout, weight initialization
  - Flash Attention support (via PyTorch SDPA when available)
  - KV cache for efficient autoregressive inference
  - Gradient checkpointing and mixed-precision helpers
"""

from quantiva.model.config import ModelConfig
from quantiva.model.gpt import GPT, GPTConfig

__all__ = ["ModelConfig", "GPT", "GPTConfig"]

