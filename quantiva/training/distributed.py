"""
Distributed training utilities.

Helpers for launching and wrapping models with:
  - DDP (Distributed Data Parallel) — standard multi-GPU training.
  - FSDP (Fully Sharded Data Parallel) — shards parameters/optimizer/grads.
  - DeepSpeed integration hooks.

Also provides a ``distributed_setup`` context manager to initialize the
process group consistently across launches (torchrun, deepspeed, etc.),
plus a battery of convenience helpers:

  - Process-group introspection: rank, world size, local rank, backend.
- Rank-0 helpers: ``master_print``, ``master_log``, ``barrier``.
  - Collectives: all-reduce (mean/sum/max/min/tensor), all-gather (list and
    single-tensor forms), gather/broadcast of arbitrary Python objects, and
    metric-dict reduction.
  - Tensor-parallel style sharding helpers: ``split_tensor`` / ``gather_tensor``.
  - Checkpoint helpers that save only on rank 0 and resume via broadcast, plus
    lightweight shared-filesystem (non-broadcast) variants.
  - Seeding, device resolution, TF32, parameter-count reports, and parameter /
    buffer sync for startup.

All functions are no-ops / single-process friendly when the process group is
not initialized, so they are safe to call in plain single-GPU / CPU runs.
"""

from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch # type: ignore
import torch.nn as nn # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    # Setup / teardown
    "init_process_group",
    "distributed_setup",
    "is_distributed",
    "barrier",
    # Rank / topology
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_local_world_size",
    "is_master",
    "master_print",
    "get_world_info",
    # Devices
    "get_device",
    "get_rank_device",
    "enable_tf32",
    # Wrappers
    "wrap_ddp",
    "wrap_fsdp",
    "wrap_deepspeed",
# Tensor collectives
    "all_reduce_tensor",
    "all_reduce_mean",
    "all_reduce_sum",
    "all_reduce_max",
    "all_reduce_min",
    "all_reduce_mean_tensor",
    "all_reduce_sum_tensor",
    "all_gather",
    "all_gather_into_tensor",
    # Tensor-parallel helpers
    "split_tensor",
    "gather_tensor",
    # Object collectives
    "gather_object",
    "broadcast_object",
    "reduce_dict",
    "broadcast_state_dict",
    # Model / checkpoint helpers
    "sync_params",
    "broadcast_buffers",
    "replace_parameter",
    "save_checkpoint_distributed",
    "load_checkpoint_distributed",
    "save_shared_checkpoint",
    "load_shared_checkpoint",
    # Model wrappers
    "wrap_ddp_auto",
    # Parameter accounting
    "get_parameter_total",
    "get_parameter_breakdown",
    # Logging / misc
    "master_log",
    "log_world_info",
    "get_device_count",
    "set_seed",
]

# ---------------------------------------------------------------------------
# Process-group setup / teardown
# ---------------------------------------------------------------------------


def init_process_group(
    backend: Optional[str] = None,
    timeout: Optional[datetime.timedelta] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    init_method: Optional[str] = None,
) -> None:
    """
    Initialize the distributed process group (idempotent).

    Args:
        backend: Backend to use. ``None`` auto-selects ``nccl`` on CUDA and
            ``gloo`` elsewhere. Falls back to ``gloo`` if ``nccl`` is
            requested but CUDA is unavailable (e.g. CPU-only containers).
        timeout: Optional ``datetime.timedelta`` for distributed ops.
        rank: Explicit rank. Defaults to ``RANK`` from the environment
            (set by ``torchrun`` / ``deepspeed``).
        world_size: Explicit world size. Defaults to ``WORLD_SIZE`` from the
            environment. If the world size is ``1`` no process group is
            created (single-process mode).
        init_method: Optional init method URL (``tcp://host:port``,
            ``env://``, ``file://...``). Defaults to ``env://``.
    """
    if torch.distributed.is_initialized():
        return

    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE", "1")
    effective_world = int(world_size if world_size is not None else env_world)

    if effective_world <= 1:
        logger.info("Distributed: single process, no process group needed.")
        return

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning("nccl requested but CUDA is unavailable; using gloo.")
        backend = "gloo"

    kwargs: Dict[str, Any] = {"backend": backend}
    if rank is not None:
        kwargs["rank"] = rank
    elif env_rank is not None:
        kwargs["rank"] = int(env_rank)
    if world_size is not None:
        kwargs["world_size"] = world_size
    else:
        kwargs["world_size"] = effective_world
    if init_method is not None:
        kwargs["init_method"] = init_method
    if timeout is not None:
        kwargs["timeout"] = timeout

    torch.distributed.init_process_group(**kwargs)
    logger.info(
        "Distributed: initialized process group (backend=%s, rank=%d, world=%d)",
        backend, get_rank(), get_world_size(),
    )


@contextmanager
def distributed_setup(backend: Optional[str] = None):
    """
    Context manager that initializes and tears down the process group.

    Example:
        ``with distributed_setup(): ...``
    """
    init_process_group(backend)
    try:
        yield
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def is_distributed() -> bool:
    """Return ``True`` if a multi-process process group is active."""
    return torch.distributed.is_initialized() and get_world_size() > 1


def barrier() -> None:
    """Synchronize all processes (no-op in single-process mode)."""
    if is_distributed():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Rank / topology
# ---------------------------------------------------------------------------


def get_rank() -> int:
    """Global rank of the current process (``0`` in single-process mode)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Total number of participating processes (``1`` single-process)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_local_rank() -> int:
    """Node-local rank (rank within the current machine / host)."""
    if torch.distributed.is_initialized():
        return int(os.environ.get("LOCAL_RANK", str(get_rank())))
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_local_world_size() -> int:
    """Number of processes on the current machine."""
    if torch.distributed.is_initialized():
        return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def is_master() -> bool:
    """Return ``True`` on global rank 0."""
    return get_rank() == 0


def master_print(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0 (accepts ``file=``, ``flush=``, etc.)."""
    if is_master():
        print(*args, **kwargs)


def get_world_info() -> dict:
    """Return a summary dict of the current distributed environment."""
    return {
        "rank": get_rank(),
        "world_size": get_world_size(),
        "local_rank": get_local_rank(),
        "local_world_size": get_local_world_size(),
        "device": str(get_rank_device()),
        "backend": (
            torch.distributed.get_backend()
            if torch.distributed.is_initialized()
            else None
        ),
        "master": is_master(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_rank_device() -> torch.device:
    """
    Return the device for the current rank.

    With ``torchrun`` each process sees all GPUs, so the correct device is
    ``cuda:<local_rank % num_gpus>``. Falls back to :func:`get_device`.
    """
    if torch.cuda.is_available():
        if torch.distributed.is_initialized():
            num = max(torch.cuda.device_count(), 1)
            return torch.device(f"cuda:{get_local_rank() % num}")
        return torch.device("cuda")
    return get_device()


def _get_backend_device() -> torch.device:
    """
    Return a device compatible with the active backend.

    NCCL requires CUDA tensors; Gloo works best with CPU tensors. This is
    used internally by the scalar/object collectives so they work across
    backends.
    """
    if torch.distributed.is_initialized():
        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            return get_rank_device()
        return torch.device("cpu")
    return get_device()


def enable_tf32() -> None:
    """Enable TF32 matmul on Ampere+ GPUs (faster, slightly less precise)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 matmul enabled.")
    else:
        logger.info("TF32 skipped (no CUDA device).")


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def wrap_ddp(
    model: nn.Module,
    device: Optional[torch.device] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Wrap a model in DDP when world_size > 1.

    Args:
        model: The model to wrap.
        device: The device the model lives on (auto-detected if ``None``).
        find_unused_parameters: Set ``True`` if some params are unused in
            the forward pass (e.g. with dropout / LoRA in some configs).
        gradient_as_bucket_view: Reduces memory by aliasing grads to the
            bucket view.
        static_graph: Set ``True`` if the graph never changes between
            iterations (enables additional optimizations).
        **kwargs: Extra args forwarded to ``DistributedDataParallel``.
    """
    if is_distributed():
        resolved_device = device if device is not None else get_rank_device()
        ddp_kwargs: Dict[str, Any] = dict(kwargs)
        ddp_kwargs["find_unused_parameters"] = find_unused_parameters
        ddp_kwargs["gradient_as_bucket_view"] = gradient_as_bucket_view
        ddp_kwargs["static_graph"] = static_graph
        if resolved_device.type == "cuda":
            ddp_kwargs["device_ids"] = [
                resolved_device.index if resolved_device.index is not None else get_local_rank()
            ]
        else:
            ddp_kwargs["device_ids"] = None
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        logger.info("Wrapped model in DDP (world_size=%d)", get_world_size())
    return model


def wrap_fsdp(
    model: nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
    device_id: Optional[int] = None,
    cpu_offload: bool = False,
    sync_module_states: bool = False,
    use_orig_params: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Wrap a model with PyTorch FSDP.

    Args:
        model: The model to wrap.
        sharding_strategy: ``"full"`` (FULL_SHARD), ``"shard_grad"``
            (SHARD_GRAD_OP), or ``"no_shard"``.
        mixed_precision: Whether to enable bf16 mixed precision under FSDP.
        device_id: CUDA device for this process (defaults to local rank).
        cpu_offload: Offload parameters to CPU (extra memory savings).
        sync_module_states: Broadcast module states from rank 0 at startup
            (needed when not every rank has the full state dict).
        use_orig_params: Keep original parameter objects (needed by many
            optimizers / parameter-based code).
        **kwargs: Extra args forwarded to ``FullyShardedDataParallel``.
    """
    try:
        from torch.distributed.fsdp import ( # type: ignore
            CPUOffload,
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

    fsdp_kwargs: Dict[str, Any] = dict(kwargs)
    fsdp_kwargs["sharding_strategy"] = strategy
    fsdp_kwargs["use_orig_params"] = use_orig_params
    fsdp_kwargs["sync_module_states"] = sync_module_states

    mp = None
    if mixed_precision and torch.cuda.is_bf16_supported():
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    if mp is not None:
        fsdp_kwargs["mixed_precision"] = mp

    if cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    if device_id is None and torch.cuda.is_available():
        device_id = get_local_rank()
    if device_id is not None:
        fsdp_kwargs["device_id"] = device_id

    model = FSDP(model, **fsdp_kwargs)
    logger.info(
        "Wrapped model in FSDP (strategy=%s, cpu_offload=%s, mp=%s)",
        sharding_strategy, cpu_offload, mp is not None,
    )
    return model


def wrap_deepspeed(
    model: nn.Module,
    ds_config: Optional[dict] = None,
    model_parameters: Optional[Sequence[nn.Parameter]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Wrap a model with DeepSpeed (if installed and a config is available).

    Args:
        model: The model to wrap.
        ds_config: DeepSpeed config dict. If ``None``, a minimal ZeRO-2
            config is used (with a micro-batch size derived from the
            ``MICRO_BATCH_SIZE`` / ``TRAIN_BATCH_SIZE`` env vars).
        model_parameters: Optional parameter list for the optimizer
            (defaults to ``model.parameters()``).
        **kwargs: Extra args forwarded to ``deepspeed.initialize``.

    Returns:
        The DeepSpeed ``engine`` object (tuple ``(engine, opt, _, _)`` is
        flattened to just ``engine`` for API simplicity).
    """
    try:
        import deepspeed # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover
        logger.warning("deepspeed not installed; returning unwrapped model.")
        return model

    if ds_config is None:
        micro_batch = int(os.environ.get("MICRO_BATCH_SIZE", "1"))
        train_batch = int(os.environ.get("TRAIN_BATCH_SIZE", str(micro_batch)))
        ds_config = {
            "train_batch_size": train_batch,
            "train_micro_batch_size_per_gpu": micro_batch,
            "gradient_accumulation_steps": max(train_batch // micro_batch, 1),
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters if model_parameters is not None else model.parameters(),
        config=ds_config,
        **kwargs,
    )
    logger.info("Wrapped model in DeepSpeed engine")
    return engine


# ---------------------------------------------------------------------------
# Tensor collectives
# ---------------------------------------------------------------------------


def all_reduce_tensor(
    tensor: torch.Tensor,
    op: str = "mean",
    group=None,
) -> torch.Tensor:
    """
    All-reduce a tensor across ranks.

    Args:
        tensor: Tensor to reduce.
        op: ``"mean"``, ``"sum"``, ``"max"``, ``"min"``, or ``"prod"``.
        group: Optional process group (defaults to the world group).

    Returns:
        The reduced tensor (a new tensor when distributed; the input
        otherwise).
    """
    if not is_distributed():
        return tensor

    op_map = {
        "mean": torch.distributed.ReduceOp.SUM,
        "sum": torch.distributed.ReduceOp.SUM,
        "max": torch.distributed.ReduceOp.MAX,
        "min": torch.distributed.ReduceOp.MIN,
        "prod": torch.distributed.ReduceOp.PRODUCT,
    }
    if op not in op_map:
        raise ValueError(
            f"Unsupported reduce op: {op!r} (expected one of {sorted(op_map)})"
        )

    t = tensor.detach().contiguous()
    if op == "mean" and not t.is_floating_point():
        t = t.float()
    torch.distributed.all_reduce(t, op=op_map[op], group=group)
    if op == "mean":
        t = t / get_world_size()
    return t


def all_reduce_mean(value: Union[float, int, torch.Tensor], group=None) -> float:
    """
    All-reduce a scalar (or 0-dim tensor) and return the mean across ranks.

    Useful for aggregating metrics like loss / accuracy every logging step.
    """
    if not is_distributed():
        return float(value)
    tensor = torch.tensor([float(value)], device=_get_backend_device())
    return float(all_reduce_tensor(tensor, op="mean", group=group).item())


def all_reduce_sum(value: Union[float, int, torch.Tensor], group=None) -> float:
    """All-reduce a scalar and return the sum across ranks."""
    if not is_distributed():
        return float(value)
    tensor = torch.tensor([float(value)], device=_get_backend_device())
    return float(all_reduce_tensor(tensor, op="sum", group=group).item())


def all_gather(tensor: torch.Tensor, group=None) -> List[torch.Tensor]:
    """
    Gather ``tensor`` from every rank into a list of ``world_size`` tensors.

    The input tensors must have matching shapes on all ranks.
    """
    if not is_distributed():
        return [tensor]
    world = get_world_size()
    t = tensor.detach().to(_get_backend_device()).contiguous()
    gathered = [torch.zeros_like(t) for _ in range(world)]
    torch.distributed.all_gather(gathered, t, group=group)
    return gathered


# ---------------------------------------------------------------------------
# Object collectives
# ---------------------------------------------------------------------------


def gather_object(obj: Any, dst: int = 0, group=None) -> Optional[List[Any]]:
    """
    Gather arbitrary Python objects.

    Args:
        obj: Object contributed by the calling rank.
        dst: Destination rank. If ``None``, gather from all ranks to all
            ranks (returns a list on every rank). Otherwise only ``dst``
            receives the list and other ranks get ``None``.
    """
    if not is_distributed():
        return [obj]

    world = get_world_size()
    if dst is None:
        gathered: List[Any] = [None] * world
        torch.distributed.all_gather_object(gathered, obj, group=group)
        return gathered

    if get_rank() == dst:
        gathered = [None] * world
        torch.distributed.gather_object(
            obj, object_gather_list=gathered, dst=dst, group=group
        )
        return gathered
    torch.distributed.gather_object(obj, object_gather_list=None, dst=dst, group=group)
    return None


def broadcast_object(obj: Any, src: int = 0, group=None) -> Any:
    """
    Broadcast an arbitrary Python object from ``src`` to all ranks.

    Example:
        ``config = broadcast_object(config)``  # rank 0 -> everyone
    """
    if not is_distributed():
        return obj
    obj_list: List[Any] = [obj] if get_rank() == src else [None]
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list[0]


def reduce_dict(metrics: Dict[str, Any], op: str = "mean", group=None) -> Dict[str, Any]:
    """
    Aggregate a dict of scalar metrics across ranks.

    Args:
        metrics: e.g. ``{"loss": 1.23, "acc": 0.87}``.
        op: ``"mean"`` or ``"sum"``.

    Returns:
        A new dict with the same keys and aggregated values.
    """
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            # Tensor metrics (e.g. per-sample losses) -> all-reduce sum.
            out[key] = all_reduce_tensor(value, op="sum", group=group)
        elif isinstance(value, dict):
            out[key] = reduce_dict(value, op=op, group=group)
        else:
            fn = all_reduce_mean if op == "mean" else all_reduce_sum
            out[key] = fn(float(value), group=group)
    return out


def broadcast_state_dict(state_dict: dict, src_rank: int = 0) -> dict:
    """
    Broadcast a state dict from ``src_rank`` to all ranks (for resume).

    Recursively broadcasts tensors nested in dicts, lists, and tuples.
    Non-tensor values (ints, floats, strings, ``None``) are left untouched
    (they are cheap to assume identical across ranks).
    """
    if not is_distributed():
        return state_dict

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().contiguous()
            torch.distributed.broadcast(tensor, src=src_rank)
            state_dict[key] = tensor
        elif isinstance(value, dict):
            broadcast_state_dict(value, src_rank)
        elif isinstance(value, (list, tuple)):
            items = list(value)
            changed = False
            for i, item in enumerate(items):
                if isinstance(item, torch.Tensor):
                    t = item.detach().clone().contiguous()
                    torch.distributed.broadcast(t, src=src_rank)
                    items[i] = t
                    changed = True
            if changed:
                state_dict[key] = items if isinstance(value, list) else tuple(items)
    return state_dict


# ---------------------------------------------------------------------------
# Model / checkpoint helpers
# ---------------------------------------------------------------------------


def sync_params(model: nn.Module, src_rank: int = 0) -> None:
    """
    Broadcast model parameters from ``src_rank`` to all ranks.

    Useful after rank-0-only initialization (e.g. random-init followed by a
    seed, or loading a checkpoint only on rank 0) so every rank starts from
    identical weights.
    """
    if not is_distributed():
        return
    for param in model.parameters():
        tensor = param.data if param.data.is_contiguous() else param.data.contiguous()
        if tensor.data_ptr() != param.data.data_ptr():
            param.data = tensor
        torch.distributed.broadcast(tensor, src=src_rank)
    logger.info("Synced model parameters from rank %d", src_rank)


def save_checkpoint_distributed(
    state: dict,
    path: str,
    main_process_only: bool = True,
    sync: bool = True,
) -> None:
    """
    Save a checkpoint, optionally only from rank 0.

    Args:
        state: Dict to persist (model / optimizer / config / iter_num ...).
        path: Destination file path.
        main_process_only: If ``True``, only rank 0 writes to disk.
        sync: If ``True``, all ranks wait until the save is complete
            (prevents races when resuming immediately afterward).
    """
    if main_process_only and not is_master():
        if sync:
            barrier()
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    logger.info("Saved distributed checkpoint to %s", path)
    if sync:
        barrier()


def load_checkpoint_distributed(
    path: str,
    map_location: Optional[str] = None,
    src_rank: int = 0,
    broadcast: bool = True,
    group=None,
) -> Optional[dict]:
    """
    Load a checkpoint on all ranks.

    Only ``src_rank`` reads the file from disk; the resulting dict is then
    broadcast to every rank, guaranteeing a consistent resume state without
    requiring a shared filesystem.

    Args:
        path: Checkpoint file path (checked on ``src_rank``).
        map_location: Passed to ``torch.load``.
        src_rank: Rank that reads the file.
        broadcast: If ``True``, the state dict is broadcast to all ranks.

    Returns:
        The state dict, or ``None`` if the file does not exist.
    """
    if not is_distributed():
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location=map_location)

    state: Optional[dict] = None
    if get_rank() == src_rank:
        if os.path.exists(path):
            state = torch.load(path, map_location=map_location)
            logger.info("Loaded checkpoint from %s", path)
        else:
            logger.warning("Checkpoint not found on rank %d: %s", src_rank, path)

    if broadcast:
        state = broadcast_object(state, src=src_rank, group=group)
    barrier()
    return state


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def set_seed(seed: int = 1337) -> None:
    """
    Seed Python, NumPy, and all CUDA devices in a distributed-safe way.

    Call this on *every* rank with the same seed to keep data loading and
    initialization reproducible (the dataloader adds its own per-rank
    offset, so data remains correctly partitioned).
    """
    import random

    import numpy as np # type: ignore

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

