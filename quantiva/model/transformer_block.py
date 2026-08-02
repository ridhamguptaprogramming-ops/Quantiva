"""
Transformer block.

A single decoder block composed of:
  - Normalization + Multi-Head Attention + residual connection
  - Normalization + MLP + residual connection

Supports both pre-norm (GPT-2 / LLaMA, the default) and post-norm ordering.
"""

from __future__ import annotations

from typing import Optional

import torch # type: ignore
import torch.nn as nn # type: ignore

from quantiva.model.attention import CausalSelfAttention
from quantiva.model.config import ModelConfig
from quantiva.model.mlp import build_mlp
from quantiva.model.normalization import build_norm


class TransformerBlock(nn.Module):
    """
    One transformer decoder block.

    Pre-norm (default):
        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    Post-norm:
        x = norm(x + attn(x))
        x = norm(x + mlp(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_norm = config.pre_norm

        self.norm1 = build_norm(config.n_embd, rmsnorm=config.rmsnorm, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.norm2 = build_norm(config.n_embd, rmsnorm=config.rmsnorm, bias=config.bias)
        self.mlp = build_mlp(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        layer_past: Optional[tuple] = None,
    ) -> tuple:
        """
        Args:
            x: (B, T, n_embd)
            positions: absolute positions (T,)
            use_cache: whether to return KV cache
            layer_past: (past_k, past_v) from previous decode step

        Returns:
            (x, present) where present is (k, v) or None.
        """
        if self.pre_norm:
            attn_out, present = self.attn(
                self.norm1(x), positions=positions, use_cache=use_cache, layer_past=layer_past
            )
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        else:
            attn_out, present = self.attn(
                x, positions=positions, use_cache=use_cache, layer_past=layer_past
            )
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.mlp(x))
        return x, present

