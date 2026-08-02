"""
Embedding layers.

Provides:
  - ``TokenEmbedding``: learned token embeddings.
  - ``LearnedPositionalEmbedding``: learned absolute position embeddings
    (GPT-2 style).
  - ``Embedding``: a small composable module that applies token embedding
    plus the configured positional encoding (learned or RoPE).

RoPE is applied inside the attention module (not on the sum), so the embedding
class only supports learned positions directly and exposes a flag to indicate
whether RoPE is in use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quantiva.model.config import ModelConfig


class TokenEmbedding(nn.Module):
    """Standard learned token embedding table."""

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) -> (B, T, n_embd)"""
        return torch.nn.functional.embedding(idx, self.weight)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute position embedding (GPT-2 style)."""

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(block_size, n_embd))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (T,) -> (T, n_embd)"""
        return torch.nn.functional.embedding(positions, self.weight)


class Embedding(nn.Module):
    """
    Token embedding plus optional learned positional embedding.

    When ``pos_emb == "rope"``, no positional embedding is added here;
    positions are handled by the attention layers via RoPE.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        if config.pos_emb == "learned":
            self.position_embedding = LearnedPositionalEmbedding(config.block_size, config.n_embd)
        else:
            self.position_embedding = None

    def forward(self, idx: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: token ids of shape (B, T).
            positions: absolute positions of shape (T,).

        Returns:
            Token embeddings (plus positions if learned) of shape (B, T, n_embd).
        """
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            pos_emb = self.position_embedding(positions)
            x = x + pos_emb.unsqueeze(0)
        return self.dropout(x)

