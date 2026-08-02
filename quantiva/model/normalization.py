"""
Normalization layers.

Implements LayerNorm (with optional bias) and RMSNorm (root-mean-square
normalization, used by LLaMA and many modern LLMs). Both are small, focused
modules that satisfy the framework's normalization needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    Layer Normalization with an optional bias term.

    Reference: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center the activations (no mean
    subtraction), which saves compute and has been shown to be equally
    effective.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, ndim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def build_norm(ndim: int, rmsnorm: bool = False, bias: bool = True) -> nn.Module:
    """Factory helper to build the configured normalization layer."""
    if rmsnorm:
        return RMSNorm(ndim)
    return LayerNorm(ndim, bias=bias)

