"""
Group Relative Policy Optimization (GRPO).

A simplified, self-contained implementation of GRPO (DeepSeekMath style).
Unlike PPO, GRPO does not use a separate value network — instead it ranks a
group of sampled completions for each prompt and computes advantages relative
to the group mean. This is well-suited to small/educational implementations
and verifiable rewards.

Reference: https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token log-probs of ``labels`` under ``logits``.

    Args:
        logits: (B, T, V)
        labels: (B, T)

    Returns:
        (B, T) log-probs.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)


def compute_grpo_loss(
    policy_logps: torch.Tensor,          # (G, T) log-probs from the policy
    old_logps: torch.Tensor,             # (G, T) log-probs at rollout time
    advantages: torch.Tensor,            # (G,) per-completion advantage
    mask: torch.Tensor,                  # (G, T) valid-token mask
    beta: float = 0.04,                  # KL penalty coefficient
    ref_logps: Optional[torch.Tensor] = None,  # (G, T) ref-model log-probs
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    GRPO objective:
        max E[ clip(pi/old, 1±eps) * A - beta * KL(pi || ref) ]

    The loss (to minimize) is the negative of the objective.

    Args:
        policy_logps: Log-probs of the sampled completions under the current
            policy model.
        old_logps: Log-probs under the frozen rollout policy.
        advantages: Group-normalized advantages (per completion).
        mask: Token mask (1 = compute loss on this token).
        beta: KL penalty.
        ref_logps: Log-probs under a reference model (SFT init) for KL.
        clip_eps: PPO-style clipping range.

    Returns:
        Scalar loss.
    """
    # Importance ratio: pi / old.
    ratio = torch.exp(policy_logps - old_logps)  # (G, T)

    # Advantages broadcast over tokens.
    adv = advantages.unsqueeze(-1)  # (G, 1)

    # Clipped surrogate objective.
    pg_losses = -ratio * adv
    pg_clipped = -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_loss = torch.max(pg_losses, pg_clipped)

    # KL penalty (approximate KL from log-ratio).
    if ref_logps is not None:
        log_ratio = policy_logps - ref_logps
        kl = (torch.exp(log_ratio) - 1) - log_ratio  # (G, T)
    else:
        # Fall back to a simple prior (uniform) KL.
        kl = -policy_logps - 1.0

    loss = (pg_loss + beta * kl) * mask
    return loss.sum() / mask.sum().clamp(min=1)


class GRPOTrainer:
    """
    Lightweight GRPO trainer.

    Args:
        policy: The trainable policy model (``GPT``).
        ref_model: Frozen reference model (optional).
        reward_fn: Callable ``reward_fn(text) -> float`` (verifiable rewards).
        tokenizer: Tokenizer for decoding completions.
        device: Target device.
        beta / clip_eps: GRPO hyperparameters.
    """

    def __init__(
        self,
        policy: nn.Module,
        reward_fn: Callable[[str], float],
        tokenizer,
        ref_model: Optional[nn.Module] = None,
        device: str = "cpu",
        beta: float = 0.04,
        clip_eps: float = 0.2,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.beta = beta
        self.clip_eps = clip_eps
        self.policy.to(self.device)
        if ref_model is not None:
            ref_model.to(self.device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

    def sample_completions(
        self,
        prompt: str,
        group_size: int,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Sample ``group_size`` completions for a prompt using greedy/temperature
        sampling. In a production system this would call the framework's
        ``inference.generate`` with a sampler; here we keep it self-contained.
        """
        from quantiva.inference.generate import generate

        completions = []
        for _ in range(group_size):
            out = generate(
                self.policy,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=str(self.device),
            )
            completions.append(out)
        return completions

    def _tokenize(self, texts: List[str]) -> torch.Tensor:
        ids = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(i) for i in ids)
        padded = [i + [0] * (max_len - len(i)) for i in ids]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def train_step(
        self,
        prompts: List[str],
        group_size: int = 4,
        max_new_tokens: int = 64,
        lr: float = 1e-5,
    ) -> float:
        """
        Run one GRPO update over a batch of prompts.

        For each prompt, sample ``group_size`` completions, compute rewards,
        normalize advantages within the group, and update the policy with a
        clipped PPO-style objective plus a KL penalty.
        """
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        all_completions: List[str] = []
        all_rewards: List[float] = []
        all_full_texts: List[str] = []

        # --- Rollout ---
        with torch.no_grad():
            for prompt in prompts:
                completions = self.sample_completions(prompt, group_size, max_new_tokens)
                rewards = [self.reward_fn(c) for c in completions]
                all_completions.extend(completions)
                all_rewards.extend(rewards)
                all_full_texts.extend(prompt + c for c in completions)

            # Encode full prompt+completion sequences for old log-probs.
            full_ids = self._tokenize(all_full_texts)

            # Old log-probs under frozen policy (rollout distribution).
            old_logps = _logprobs_from_logits(self.policy(full_ids)["logits"], full_ids)
            if self.ref_model is not None:
                ref_logps = _logprobs_from_logits(self.ref_model(full_ids)["logits"], full_ids)
            else:
                ref_logps = None

        # --- Advantages (group-normalized per prompt) ---
        advantages = []
        n = group_size
        for i in range(0, len(all_rewards), n):
            group = torch.tensor(all_rewards[i : i + n], device=self.device)
            mean = group.mean()
            std = group.std().clamp(min=1e-4)
            advantages.append((group - mean) / std)
        advantages = torch.cat(advantages)  # (G_total,)

        # --- Policy update ---
        self.policy.train()
        optimizer.zero_grad()
        policy_logps = _logprobs_from_logits(self.policy(full_ids)["logits"], full_ids)
        mask = (full_ids != 0).float()  # mask padding

        loss = compute_grpo_loss(
            policy_logps, old_logps, advantages, mask,
            beta=self.beta, ref_logps=ref_logps, clip_eps=self.clip_eps,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        optimizer.step()

        return loss.item()


def build_grpo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantiva GRPO")
    parser.add_argument("--policy_checkpoint", type=str, required=True)
    parser.add_argument("--ref_checkpoint", type=str, default=None)
    parser.add_argument("--prompts_path", type=str, required=True, help="JSONL: {'prompt': str}")
    parser.add_argument("--out_dir", type=str, default="out-grpo")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="")
    return parser


def main() -> None:
    import json

    logging.basicConfig(level=logging.INFO)
    args = build_grpo_parser().parse_args()

    from quantiva.model.config import ModelConfig
    from quantiva.model.gpt import GPT
    from quantiva.tokenizer.factory import get_tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.policy_checkpoint, map_location=device)
    policy = GPT(ModelConfig.from_dict(payload["model_config"]))
    policy.load_state_dict(payload["model"])
    policy.to(device)

    ref_model = None
    if args.ref_checkpoint:
        ref_payload = torch.load(args.ref_checkpoint, map_location=device)
        ref_model = GPT(ModelConfig.from_dict(ref_payload["model_config"]))
        ref_model.load_state_dict(ref_payload["model"])

    tokenizer = get_tokenizer("tiktoken", encoding_name="gpt2")

    prompts = []
    with open(args.prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])

    def reward_fn(text: str) -> float:
        # Simple verifiable reward: count of digits / length + bonus for exact.
        return (sum(c.isdigit() for c in text) / max(len(text), 1)) * 10

    trainer = GRPOTrainer(
        policy, reward_fn, tokenizer, ref_model=ref_model, device=device,
        beta=args.beta, clip_eps=args.clip_eps,
    )

    for it in range(args.max_iters):
        loss = trainer.train_step(
            prompts, group_size=args.group_size,
            max_new_tokens=args.max_new_tokens, lr=args.learning_rate,
        )
        logger.info("iter %d: grpo_loss %.4f", it, loss)

    os.makedirs(args.out_dir, exist_ok=True)
    policy.save(os.path.join(args.out_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()

