"""
GPT language model.

The top-level model: Transformer body + output LM head, with:
  - Tied embeddings (optional)
  - GPT-2 style weight initialization
  - Forward pass for training (with targets) and inference
  - KV-cache aware forward for incremental decoding
  - Helper methods: param count, FLOPs / MFU, config persistence
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from typing import Optional

import torch # type: ignore
import torch.nn as nn # pyright: ignore[reportMissingImports]

from quantiva.model.config import ModelConfig
from quantiva.model.transformer import Transformer

# Compatibility alias: some callers (and the top-level package) refer to
# ``GPTConfig``. It is the same configuration dataclass as ``ModelConfig``.
GPTConfig = ModelConfig

logger = logging.getLogger(__name__)


class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)

        if config.tied_embeddings:
            # Reuse the token embedding weight for the LM head.
            self.lm_head = nn.Linear(
                config.n_embd, config.vocab_size, bias=False
            )
            self.lm_head.weight = self.transformer.embed.token_embedding.weight
        else:
            self.lm_head = nn.Linear(
                config.n_embd, config.vocab_size, bias=False
            )

        self.apply(self._init_weights)
        # Apply special scaled init to residual projections (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward / Loss
    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
    ) -> dict:
        """
        Args:
            idx: token ids (B, T).
            targets: optional target token ids (B, T) for loss computation.
            positions: absolute positions (T,).
            use_cache: return KV caches.
            past_key_values: list of (past_k, past_v) per layer.

        Returns:
            dict with keys ``logits`` (B, T, vocab_size), ``loss`` (optional),
            ``presents`` (optional KV caches).
        """
        hidden, presents = self.transformer(
            idx,
            positions=positions,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, V), targets.view(B * T)
            )

        out: dict = {"logits": logits, "loss": loss, "presents": presents}
        return out

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_key_values: Optional[list] = None,
    ) -> dict:
        """
        Single-step (or small-block) forward for autoregressive decoding.

        Args:
            idx: token ids (B, T) where T is typically 1 during generation.
            past_key_values: cached (k, v) tuples per layer.

        Returns:
            dict with ``logits`` (B, T, vocab), ``loss`` (None), ``presents``.
        """
        positions = torch.arange(past_key_values[0][0].size(2), past_key_values[0][0].size(2) + idx.size(1), device=idx.device, dtype=torch.long) if past_key_values else torch.arange(idx.size(1), device=idx.device, dtype=torch.long)
        return self.forward(
            idx,
            targets=None,
            positions=positions,
            use_cache=True,
            past_key_values=past_key_values,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        """
        Count model parameters.

        Args:
            non_embedding: if True, exclude the token embedding / LM head
                parameters (used for "non-embedding params" comparison).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.embed.token_embedding.weight.numel()
            if not self.config.tied_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate Model FLOPs Utilization (MFU).

        Reference: PaLM paper, https://arxiv.org/abs/2204.02311

        Args:
            fwdbwd_per_iter: number of forward+backward passes per iteration
                (e.g. ``1`` for a single micro-batch, ``grad_accum * micro_bsz``).
            dt: wall-clock time in seconds for one iteration.
        """
        N = self.num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPs.
        mfu = flops_achieved / flops_promised
        return mfu

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple = (0.9, 0.95),
        device_type: str = "cuda",
    ) -> torch.optim.Optimizer:
        """
        Build an AdamW optimizer with weight decay applied only to 2D
        parameters (matrices), following the GPT-2 / Quantiva recipe.
        """
        # Collect parameters that require grad.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            "num decayed parameter tensors: %d, with %d parameters",
            len(decay_params), num_decay_params,
        )
        logger.info(
            "num non-decayed parameter tensors: %d, with %d parameters",
            len(nodecay_params), num_nodecay_params,
        )
        # Create AdamW with fused kernel when available.
        fused_available = hasattr(torch.optim.AdamW, "fused")
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )
        return optimizer

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Save model weights (and optionally optimizer state) to ``path``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model_config": self.config.to_dict(),
            "model": self.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
        logger.info("Saved model checkpoint to %s", path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "GPT":
        """Load a model saved with :meth:`save`."""
        payload = torch.load(path, map_location=map_location)
        config = ModelConfig.from_dict(payload["model_config"])
        model = cls(config)
        model.load_state_dict(payload["model"])
        model.eval()
        return model

    def save_config(self, path: str) -> None:
        """Save just the model config as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

