"""
Distributed training utilities.

Helpers for launching and wrapping models with:
  - DDP (Distributed Data Parallel) — standard multi-GPU training.
  - FSDP (Fully Sharded Data Parallel) — shards parameters/optimizer/grads.
  - DeepSpeed integration hooks.

Also provides a ``distributed_setup`` context manager to initialize the
process group consistently across launches (torchrun, deepspeed, etc.).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch # type: ignore
import torch.nn as nn # type: ignore

logger = logging.getLogger(__name__)


def init_process_group(backend: str = "nccl") -> None:
    """Initialize the distributed process group (idempotent)."""
    if torch.distributed.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.distributed.init_process_group(backend=backend)
    else:
        logger.info("Distributed: single process, no process group needed.")


@contextmanager
def distributed_setup(backend: str = "nccl"):
    """Context manager that initializes and tears down the process group."""
    init_process_group(backend)
    try:
        yield
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_master() -> bool:
    return get_rank() == 0


def wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Wrap a model in DDP when world_size > 1.

    Args:
        model: The model to wrap.
        device: The device the model lives on.
    """
    if torch.distributed.is_initialized() and get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
        logger.info("Wrapped model in DDP (world_size=%d)", get_world_size())
    return model


def wrap_fsdp(
    model: nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
) -> nn.Module:
    """
    Wrap a model with PyTorch FSDP.

    Args:
        model: The model to wrap.
        sharding_strategy: "full" (FULL_SHARD), "shard_grad" (SHARD_GRAD_OP),
            "no_shard".
        mixed_precision: Whether to enable bf16 mixed precision under FSDP.
    """
    try:
        from torch.distributed.fsdp import ( # type: ignore
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
    except ImportError:  # pragma: no cover
        logger.warning("FSDP unavailable; returning unwrapped model.")
        return model

    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    mp = None
    if mixed_precision and torch.cuda.is_bf16_supported():
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    model = FSDP(model, sharding_strategy=strategy, mixed_precision=mp)
    logger.info("Wrapped model in FSDP (strategy=%s)", sharding_strategy)
    return model


def wrap_deepspeed(model: nn.Module, ds_config: Optional[dict] = None) -> nn.Module:
    """
    Wrap a model with DeepSpeed (if installed and a config is available).

    Args:
        model: The model to wrap.
        ds_config: DeepSpeed config dict. If None, a minimal default is used.

    Returns:
        ``(model, optimizer, _, _)`` as a tuple, matching deepspeed.initialize.
        For API simplicity, this returns the ``engine`` object directly.
    """
    try:
        import deepspeed # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover
        logger.warning("deepspeed not installed; returning unwrapped model.")
        return model

    if ds_config is None:
        ds_config = {
            "train_batch_size": 1,
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        }
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    logger.info("Wrapped model in DeepSpeed engine")
    return engine


def broadcast_state_dict(state_dict: dict, src_rank: int = 0) -> dict:
    """Broadcast a state dict from rank 0 to all ranks (for resume)."""
    if not torch.distributed.is_initialized() or get_world_size() == 1:
        return state_dict
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor = value.contiguous()
            torch.distributed.broadcast(tensor, src=src_rank)
            state_dict[key] = tensor
        elif isinstance(value, dict):
            broadcast_state_dict(value, src_rank)
    return state_dict

