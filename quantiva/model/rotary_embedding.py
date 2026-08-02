"""
Rotary Positional Embeddings (RoPE).

RoPE encodes token positions by rotating the query and key vectors with
position-dependent angles. It allows the model to capture relative position
information naturally, and generalizes better to longer sequences than
learned absolute position embeddings.

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch # type: ignore
import torch.nn as nn # type: ignore


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Precompute the inverse-frequency table used to build rotation matrices.

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base for the geometric progression (``10000`` in the paper).
        scaling_factor: RoPE scaling (linear scaling for longer contexts).
        device/dtype: Target device/dtype for the returned tensor.

    Returns:
        ``inv_freq`` of shape ``(head_dim // 2,)``.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Standard RoPE: freqs = 1 / theta^(2i / head_dim), i in [0, head_dim/2).
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (exponent / head_dim))
    if scaling_factor != 1.0:
        inv_freq = inv_freq / scaling_factor
    return inv_freq.to(dtype=dtype)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query/key tensors.

    Args:
        x: Tensor of shape ``(B, n_heads, T, head_dim)``.
        cos: Cosine table of shape ``(1, 1, T, head_dim)``.
        sin: Sine table of shape ``(1, 1, T, head_dim)``.

    Returns:
        Rotated tensor of the same shape as ``x``.

    Rotating (x0, x1) by angle theta is:
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos
    For a full head_dim we rotate each consecutive pair. The vectorized trick
    splits x into even and odd channels and combines them.
    """
    # x_even/x_odd: (B, n_heads, T, head_dim/2)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Rotate:
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    # Interleave back into (B, n_heads, T, head_dim).
    rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
    return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
    """
    Precomputed rotary position embedding module.

    Produces per-position cosine/sine tables for the full context window and
    applies them to query/key tensors.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor

        # (head_dim/2,)
        inv_freq = precompute_rope_frequencies(
            head_dim, max_seq_len, theta=theta, scaling_factor=scaling_factor
        )
        self.register_buffer("inv_freq", inv_freq)

        # Build cos/sin tables for the full context length.
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin tables for sequences up to ``seq_len``."""
        if hasattr(self, "cos_cached") and self.cos_cached.shape[0] >= seq_len:
            return
        # positions: (seq_len,)
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # angles: (seq_len, head_dim/2)
        angles = torch.outer(positions, self.inv_freq)
        # Expand each angle into (cos, sin) for the pair layout.
        # Build a (seq_len, head_dim) table where every pair (2i, 2i+1) shares the angle.
        emb = torch.cat((angles, angles), dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Args:
            x: Query/key tensor (used only for device/dtype).
            seq_len: Current sequence length.

        Returns:
            (cos, sin) each of shape ``(1, 1, seq_len, head_dim)``.
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.device).to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.device).to(x.dtype),
        )

    def forward_with_positions(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Like :meth:`forward` but for arbitrary (non-contiguous) position ids
        (e.g. during incremental decoding with a KV cache).
        """
        # angles: (T, head_dim/2)
        angles = torch.outer(positions, self.inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos.to(x.device).to(x.dtype), sin.to(x.device).to(x.dtype)

