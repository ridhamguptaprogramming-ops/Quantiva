"""
Training engine.

A flexible Trainer that supports:
  - Gradient accumulation
  - Mixed precision (bf16/float16) via GradScaler or torch.autocast
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Checkpointing and resume
  - Optional W&B logging
  - Distributed training (DDP/FSDP hooks)

The trainer is model-agnostic: it accepts any ``nn.Module`` that returns a
dict containing ``loss`` (and optionally ``logits``).
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""

    # --- Optimization ---
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    betas: tuple = (0.9, 0.95)
    max_iters: int = 600000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 1e-4
    warmdown_iters: int = 0  # linear cooldown after warmup (0 = cosine only)

    # --- Mixed precision ---
    dtype: str = "bfloat16"  # "float32", "bfloat16", "float16"
    compile: bool = False

    # --- Logging / checkpointing ---
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    always_save_checkpoint: bool = True
    out_dir: str = "out"
    wandb_log: bool = False
    wandb_project: str = "quantiva"
    wandb_run_name: str = "run"
    resume: bool = False

    # --- DDP ---
    backend: str = "nccl"
    device: str = ""  # auto-detected if empty

    # --- Callbacks ---
    # Optional: callback(model, iter, metrics) -> None
    on_log: Optional[Callable] = None
    on_eval: Optional[Callable] = None

    def __post_init__(self) -> None:
        if self.lr_decay_iters <= 0:
            self.lr_decay_iters = self.max_iters
        if self.warmdown_iters <= 0:
            self.warmdown_iters = self.lr_decay_iters


class Trainer:
    """
    Core training loop.

    Args:
        model: Any ``nn.Module`` with a forward returning ``{"loss": ...}``.
        config: ``TrainingConfig``.
        train_batch_fn: Callable returning ``(x, y)`` batch tensors.
        val_batch_fn: Optional callable returning a val batch.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_batch_fn: Callable[[], tuple],
        val_batch_fn: Optional[Callable[[], tuple]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_batch_fn = train_batch_fn
        self.val_batch_fn = val_batch_fn

        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        if config.dtype in ("bfloat16", "float16"):
            self.amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        else:
            self.amp_dtype = torch.float32
        self.scaler = GradScaler(enabled=(config.dtype == "float16"))

        self.optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=config.betas,
            device_type=self.device.type,
        ) if hasattr(model, "configure_optimizers") else torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, betas=config.betas
        )

        self.iter_num = 0
        self.best_val_loss = float("inf")
        self._setup_logging()
        self._compile()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _resolve_device(self, device: str) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _setup_logging(self) -> None:
        if self.config.wandb_log:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.__dict__,
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; disabling logging.")
                self.wandb = None
        else:
            self.wandb = None

    def _compile(self) -> None:
        if self.config.compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:  # pragma: no cover
                logger.warning("torch.compile failed: %s", e)

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------
    def get_lr(self, it: int) -> float:
        """Cosine schedule with linear warmup (GPT-2 style)."""
        cfg = self.config
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / cfg.warmup_iters
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run a few eval batches and return average val/train loss."""
        if self.val_batch_fn is None:
            return {"val_loss": float("nan")}
        self.model.eval()
        losses = []
        for _ in range(self.config.eval_iters):
            x, y = self.val_batch_fn()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype != torch.float32,
            ):
                out = self.model(x, targets=y)
            losses.append(out["loss"].item())
        self.model.train()
        return {"val_loss": sum(losses) / len(losses)}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str = "ckpt") -> None:
        os.makedirs(self.config.out_dir, exist_ok=True)
        path = os.path.join(self.config.out_dir, f"{tag}.pt")
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, tag: str = "ckpt") -> bool:
        path = os.path.join(self.config.out_dir, f"{tag}.pt")
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iter_num = payload["iter_num"]
        self.best_val_loss = payload["best_val_loss"]
        logger.info("Resumed from checkpoint: %s (iter %d)", path, self.iter_num)
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the training loop."""
        model, cfg = self.model, self.config
        model.train()
        raw_model = model.module if hasattr(model, "module") else model

        if cfg.resume:
            self.load_checkpoint()

        t0 = time.time()
        running_loss = 0.0
        local_iter = 0

        while self.iter_num < cfg.max_iters:
            self.optimizer.zero_grad(set_to_none=True)

            # Accumulated loss over grad_accum_steps.
            loss_accum = 0.0
            for micro_step in range(cfg.grad_accum_steps):
                x, y = self.train_batch_fn()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_dtype != torch.float32,
                ):
                    out = model(x, targets=y)
                    loss = out["loss"] / cfg.grad_accum_steps
                # Scale loss for DDP gradient sync on the last micro-step.
                self.scaler.scale(loss).backward()

            # Gradient clipping.
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # Step optimizer.
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update LR.
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.get_lr(self.iter_num)

            self.iter_num += 1
            local_iter += 1
            running_loss += loss_accum if loss_accum else loss.item() * cfg.grad_accum_steps

            # Logging.
            if self.iter_num % cfg.log_interval == 0:
                dt = time.time() - t0
                lossf = running_loss / max(local_iter, 1)
                mfu = raw_model.estimate_mfu(cfg.grad_accum_steps, dt) if hasattr(raw_model, "estimate_mfu") else 0.0
                logger.info(
                    "iter %d: loss %.4f, time %.3fms, mfu %.2f%%",
                    self.iter_num, lossf, dt * 1000, mfu * 100,
                )
                if self.wandb is not None:
                    self.wandb.log({"loss": lossf, "mfu": mfu, "lr": self.get_lr(self.iter_num)})
                if cfg.on_log is not None:
                    cfg.on_log(raw_model, self.iter_num, {"loss": lossf})
                t0 = time.time()
                running_loss = 0.0
                local_iter = 0

            # Evaluation + checkpoint.
            if self.iter_num % cfg.eval_interval == 0:
                metrics = self.evaluate()
                val_loss = metrics["val_loss"]
                logger.info("iter %d: val_loss %.4f", self.iter_num, val_loss)
                if cfg.on_eval is not None:
                    cfg.on_eval(raw_model, self.iter_num, metrics)
                if self.wandb is not None:
                    self.wandb.log(metrics)
                if val_loss < self.best_val_loss or cfg.always_save_checkpoint:
                    improved = val_loss < self.best_val_loss
                    if improved:
                        self.best_val_loss = val_loss
                    self.save_checkpoint("ckpt")
                    if improved:
                        self.save_checkpoint("best")

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info("Training complete after %d iterations.", self.iter_num)

