"""
Model configuration.

A single dataclass captures all hyperparameters for a GPT-style decoder-only
transformer. This keeps configuration explicit, serializable, and easy to
reason about (SOLID: single responsibility — config owns all knobs).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Quantiva GPT transformer."""

    # --- Architecture ---
    vocab_size: int = 50304          # Number of tokens in the vocabulary.
    block_size: int = 1024           # Max context length (sequence length).
    n_layer: int = 12                # Number of transformer blocks.
    n_head: int = 12                 # Number of query heads.
    n_kv_head: Optional[int] = None  # Number of key/value heads (GQA). None => MHA.
    n_embd: int = 768                # Embedding dimension.
    head_dim: Optional[int] = None   # Head dimension override (default: n_embd // n_head).
    intermediate_size: Optional[int] = None  # MLP hidden size (default: 4 * n_embd).

    # --- Activation / MLP ---
    activation: str = "gelu"         # "gelu" (GPT-2) or "swiglu" (modern LLMs).
    rmsnorm: bool = False            # Use RMSNorm instead of LayerNorm.
    pre_norm: bool = True            # Pre-norm (GPT-2 style) vs post-norm.
    bias: bool = True                # Whether Linear/LayerNorm layers use bias.
    dropout: float = 0.0             # Dropout rate (0 = disabled).

    # --- Positional encoding ---
    pos_emb: str = "learned"         # "learned" (GPT-2) or "rope" (RoPE).
    rope_theta: float = 10000.0      # RoPE base frequency.
    rope_scaling: Optional[dict] = field(
        default_factory=lambda: {"factor": 1.0, "type": "linear"}
    )  # RoPE scaling config (linear / dynamic).

    # --- Attention ---
    flash_attn: bool = True          # Use Flash Attention via SDPA when available.
    attn_dropout: float = 0.0        # Attention dropout.
    mlp_dropout: float = 0.0         # MLP dropout.

    # --- Training ---
    gradient_checkpointing: bool = False  # Trade compute for memory.
    tied_embeddings: bool = True     # Tie input embedding with output head.

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head  # Default to MHA.
        if self.n_kv_head > self.n_head:
            raise ValueError("n_kv_head cannot exceed n_head.")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head.")
        if self.head_dim is None:
            self.head_dim = self.n_embd // self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.intermediate_size is None:
            if self.activation == "swiglu":
                # SwiGLU typically uses 8/3 * n_embd.
                self.intermediate_size = int(8 / 3 * self.n_embd)
            else:
                self.intermediate_size = 4 * self.n_embd

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for checkpoints/config files)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_num_heads(self) -> int:
        return self.n_head

    def get_num_kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

