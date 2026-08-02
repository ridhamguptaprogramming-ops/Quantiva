"""
Attention module.

Implements Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) with
support for:
  - Flash Attention (via PyTorch's scaled_dot_product_attention when available)
  - Causal masking
  - Rotary Positional Embeddings (RoPE)
  - KV cache for efficient autoregressive decoding

GQA (https://arxiv.org/abs/2305.13245) reduces memory bandwidth by sharing
key/value heads across groups of query heads. When ``n_kv_head == n_head``,
it reduces to standard MHA.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantiva.model.config import ModelConfig
from quantiva.model.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with optional GQA, RoPE, flash attention, and KV cache.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.head_dim or (config.n_embd // config.n_head)
        self.n_embd = config.n_embd

        # Query projection always projects to n_head * head_dim.
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        # Key/value projections share the (smaller) n_kv_head groups.
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.mlp_dropout if config.mlp_dropout else config.dropout)

        self.flash = config.flash_attn and hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Pre-allocate a causal mask buffer for the non-flash path.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "mask",
                mask.view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

        if config.pos_emb == "rope":
            self.rope = RotaryEmbedding(
                self.head_dim,
                config.block_size,
                theta=config.rope_theta,
                scaling_factor=config.rope_scaling.get("factor", 1.0) if config.rope_scaling else 1.0,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        layer_past: Optional[tuple] = None,
    ) -> tuple:
        """
        Args:
            x: Input of shape (B, T, n_embd).
            positions: Absolute positions (T,). If None, ``arange(T)``.
            use_cache: If True, return (output, (new_k, new_v)) for KV caching.
            layer_past: (past_k, past_v) from a previous decode step, each of
                shape (B, n_kv_head, past_len, head_dim).

        Returns:
            ``(output, present)`` where ``output`` is (B, T, n_embd) and
            ``present`` is either None or ``(k, v)`` of shape
            ``(B, n_kv_head, T+past_len, head_dim)``.
        """
        B, T, C = x.size()

        if positions is None:
            positions = torch.arange(T, device=x.device, dtype=torch.long)

        # Project to q, k, v.
        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)

        # Reshape into (B, n_heads, T, head_dim).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k.
        if self.rope is not None:
            if positions is not None and (positions != torch.arange(positions.numel(), device=positions.device)).any():
                # Non-contiguous positions (incremental decoding): compute fresh.
                cos, sin = self.rope.forward_with_positions(q, positions)
            else:
                cos, sin = self.rope(q, T)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # KV cache concatenation.
        present = None
        if layer_past is not None:
            past_k, past_v = layer_past
            # past_k/past_v: (B, n_kv_head, past_len, head_dim)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
            present = (k, v)
        elif use_cache:
            present = (k, v)

        if self.flash:
            # Flash attention (causal) with optional grouped-query broadcast.
            if self.n_kv_head == self.n_head:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # GQA: expand k/v from n_kv_head to n_head groups.
                q_heads = self.n_head
                kv_heads = self.n_kv_head
                group = q_heads // kv_heads
                # (B, n_kv_head, T, head_dim) -> (B, n_kv_head, group, T, head_dim)
                # -> (B, n_kv_head * group, T, head_dim)
                k = k[:, :, None, :, :].expand(B, kv_heads, group, k.size(2), self.head_dim).reshape(B, q_heads, k.size(2), self.head_dim)
                v = v[:, :, None, :, :].expand(B, kv_heads, group, v.size(2), self.head_dim).reshape(B, q_heads, v.size(2), self.head_dim)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # Manual attention (with optional GQA broadcast).
            if self.n_kv_head != self.n_head:
                group = self.n_head // self.n_kv_head
                k = k[:, :, None, :, :].expand(B, self.n_kv_head, group, k.size(2), self.head_dim).reshape(B, self.n_head, k.size(2), self.head_dim)
                v = v[:, :, None, :, :].expand(B, self.n_kv_head, group, v.size(2), self.head_dim).reshape(B, self.n_head, v.size(2), self.head_dim)

            T_full = k.size(2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            # Apply causal mask for the full (possibly cached) sequence.
            mask = self.mask[:, :, :T, :T_full]
            if T_full > T:
                # We have cached keys; the mask should be padded on the left
                # with -inf so new tokens can't attend to the future.
                causal = torch.tril(torch.ones(T, T_full, device=x.device, dtype=torch.bool))
            else:
                causal = mask.bool()
            att = att.masked_fill(causal == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_head, T, head_dim)

        # Merge heads back.
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.resid_dropout(self.o_proj(y))
        return y, present

