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

__all__ = ["Trainer", "TrainingConfig"]

