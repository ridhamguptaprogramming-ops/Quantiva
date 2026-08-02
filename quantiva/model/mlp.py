"""
MLP (feed-forward) layers.

Implements the transformer feed-forward network with a choice of activation:
  - GELU (GPT-2 style): ``linear -> GELU -> linear``
  - SwiGLU (LLaMA-style): ``linear x 3 -> SwiGLU gating``

Both use residual-projection scaling during initialization for stability.
"""

from __future__ import annotations

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

from quantiva.model.config import ModelConfig


class GELUMLP(nn.Module):
    """GPT-2 style MLP: Linear -> GELU -> Linear."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n_embd = config.n_embd
        hidden = config.intermediate_size or (4 * n_embd)
        self.c_fc = nn.Linear(n_embd, hidden, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden, n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.mlp_dropout if config.mlp_dropout else config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network.

    ``out = (act(gate_proj(x)) * up_proj(x)) @ down_proj``

    Reference: https://arxiv.org/abs/2002.05202 and LLaMA.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n_embd = config.n_embd
        hidden = config.intermediate_size or int(8 / 3 * n_embd)
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(config.mlp_dropout if config.mlp_dropout else config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


def build_mlp(config: ModelConfig) -> nn.Module:
    """Factory helper to build the configured MLP variant."""
    if config.activation.lower() == "swiglu":
        return SwiGLUMLP(config)
    return GELUMLP(config)

