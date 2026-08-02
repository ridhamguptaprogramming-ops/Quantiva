"""
Training package.

Pretraining, fine-tuning, preference optimization, RL, and LoRA support:
  - ``trainer``: core training loop (grad accum, AMP, checkpointing, LR).
  - ``pretrain``: causal-language-model pretraining entry point.
  - ``sft``: supervised fine-tuning.
  - ``dpo``: Direct Preference Optimization.
  - ``grpo``: Group Relative Policy Optimization.
  - ``lora``: LoRA / QLoRA adapters.
  - ``rl``: RLHF-style reinforcement learning helpers.
  - ``distributed``: DDP / FSDP / DeepSpeed utilities.
"""

from quantiva.training.trainer import Trainer, TrainingConfig
from quantiva.training.lora import apply_lora, merge_lora_weights
from quantiva.training.rl import kl_penalty, gae, ppo_clip_loss
from quantiva.training.dpo import dpo_loss
from quantiva.training.distributed import (
    all_gather,
    all_reduce_mean,
    all_reduce_sum,
    all_reduce_tensor,
    barrier,
    broadcast_object,
    broadcast_state_dict,
    distributed_setup,
    gather_object,
    get_device,
    get_local_rank,
    get_local_world_size,
    get_rank,
    get_rank_device,
    get_world_info,
    get_world_size,
    init_process_group,
    is_distributed,
    is_master,
    load_checkpoint_distributed,
    master_print,
    reduce_dict,
    save_checkpoint_distributed,
    set_seed,
    sync_params,
    wrap_ddp,
    wrap_deepspeed,
    wrap_fsdp,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "apply_lora",
    "merge_lora_weights",
    "kl_penalty",
    "gae",
    "ppo_clip_loss",
    "dpo_loss",
    # Distributed utilities.
    "init_process_group",
    "distributed_setup",
    "is_distributed",
    "barrier",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_local_world_size",
    "is_master",
    "master_print",
    "get_world_info",
    "get_device",
    "get_rank_device",
    "wrap_ddp",
    "wrap_fsdp",
    "wrap_deepspeed",
    "all_gather",
    "all_reduce_mean",
    "all_reduce_sum",
    "all_reduce_tensor",
    "gather_object",
    "broadcast_object",
    "reduce_dict",
    "broadcast_state_dict",
    "sync_params",
    "save_checkpoint_distributed",
    "load_checkpoint_distributed",
    "set_seed",
]

