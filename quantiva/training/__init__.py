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

__all__ = [
    "Trainer",
    "TrainingConfig",
    "apply_lora",
    "merge_lora_weights",
    "kl_penalty",
    "gae",
    "ppo_clip_loss",
    "dpo_loss",
]

