"""
Reinforcement Learning helpers (RLHF-style).

Provides building blocks for PPO-style RL training:
  - A small reward model wrapper (learned scalar reward).
  - KL penalty computation against a frozen reference model.
  - Advantage estimation (Generalized Advantage Estimation).
  - PPO clipped surrogate loss.

These utilities are used by higher-level RL fine-tuning scripts.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def kl_penalty(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float = 0.02,
) -> torch.Tensor:
    """
    Approximate KL divergence penalty between policy and reference model.

    Uses the estimator from the original RLHF paper:
        KL ~ (exp(log_p - log_ref) - 1) - (log_p - log_ref)

    Args:
        policy_logprobs: (B, T) policy log-probs.
        ref_logprobs: (B, T) reference log-probs.
        beta: Scaling coefficient.

    Returns:
        Scalar KL penalty (mean over batch/tokens).
    """
    log_ratio = policy_logprobs - ref_logprobs
    kl = torch.exp(log_ratio) - 1.0 - log_ratio
    return beta * kl.mean()


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Generalized Advantage Estimation.

    Reference: https://arxiv.org/abs/1506.02438

    Args:
        rewards: (T, B) rewards at each step.
        values: (T, B) value predictions (bootstrap for the last step assumed
            to be included at index T).
        dones: (T, B) 1 where the episode ended.
        gamma: Discount factor.
        lam: GAE smoothing parameter.

    Returns:
        (T, B) advantages.
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae_ = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae_ = delta + gamma * lam * (1 - dones[t]) * gae_
        advantages[t] = gae_
    return advantages


def ppo_clip_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PPO clipped surrogate objective.

    Args:
        logprobs: (B, T) current policy log-probs.
        old_logprobs: (B, T) rollout log-probs.
        advantages: (B,) or (B, 1) advantage per sample.
        clip_eps: Clipping range.
        mask: (B, T) token mask (optional).

    Returns:
        Scalar PPO loss.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages.unsqueeze(-1) if advantages.dim() == 1 else advantages
    pg_losses = -ratio * adv
    pg_clipped = -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = torch.max(pg_losses, pg_clipped)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


class RewardModel(nn.Module):
    """
    Simple scalar reward model: transformer body + a linear head.

    Used as a learned reward model in RLHF-style training. The head maps the
    final hidden state to a scalar reward per token/sequence.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone  # e.g. a ``GPT`` instance (transformer body)
        n_embd = backbone.config.n_embd
        self.reward_head = nn.Linear(n_embd, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) token ids.

        Returns:
            (B,) scalar reward per sequence (mean over valid positions).
        """
        hidden, _ = self.backbone.transformer(idx)
        # Take the last token's representation.
        last_hidden = hidden[:, -1, :]  # (B, n_embd)
        return self.reward_head(last_hidden).squeeze(-1)

    def reward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.forward(idx)


def reward_from_hf(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: str = "cpu",
) -> float:
    """
    Score a completion with a Hugging Face-style reward model (optional).

    This is a convenience helper; requires ``transformers``. Falls back to a
    heuristic reward if ``transformers`` is unavailable.
    """
    try:
        from transformers import pipeline  # type: ignore

        rp = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        result = rp(f"{prompt}{completion}")[0]
        score = result["score"]
        return float(score) if result.get("label", "POSITIVE") == "POSITIVE" else float(-score)
    except Exception as e:  # pragma: no cover
        logger.warning("HF reward model unavailable (%s); using heuristic", e)
        return float(len(completion.split()))


def build_rl_parser() -> None:
    """Placeholder for a future CLI parser; RL training is script-driven."""
    raise NotImplementedError("Use the GRPO/DPO entry points for CLI training.")

