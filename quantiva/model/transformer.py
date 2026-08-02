"""
Transformer stack.

The body of the decoder-only transformer: embedding -> N blocks -> final norm.
Supports gradient checkpointing (to trade compute for memory) and can
return/accept KV caches for autoregressive decoding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from quantiva.model.config import ModelConfig
from quantiva.model.embedding import Embedding
from quantiva.model.normalization import build_norm
from quantiva.model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    """Full transformer body (no LM head)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = Embedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm_f = build_norm(config.n_embd, rmsnorm=config.rmsnorm, bias=config.bias)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        idx: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
    ) -> tuple:
        """
        Args:
            idx: token ids (B, T).
            positions: absolute positions (T,). Defaults to arange(T).
            use_cache: if True, returns a list of (k, v) per layer.
            past_key_values: list of (past_k, past_v) per layer from prior decode.

        Returns:
            (hidden, presents) where hidden is (B, T, n_embd) and presents is
            a list of (k, v) tuples (or None if not caching).
        """
        B, T = idx.size()
        if positions is None:
            positions = torch.arange(T, device=idx.device, dtype=torch.long)

        x = self.embed(idx, positions)

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        presents: Optional[list] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                # Recompute forward activations during backprop to save memory.
                x, present = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    positions,
                    use_cache,
                    past_key_values[i],
                    use_reentrant=False,
                )
            else:
                x, present = block(
                    x,
                    positions=positions,
                    use_cache=use_cache,
                    layer_past=past_key_values[i],
                )
            if use_cache:
                presents.append(present)

        x = self.norm_f(x)
        return x, presents

