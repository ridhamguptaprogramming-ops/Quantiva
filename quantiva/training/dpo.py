"""
Direct Preference Optimization (DPO).

Aligns a policy model to human preferences without a separate reward model or
RL loop. DPO reformulates the RLHF objective as a simple classification loss
over pairs of (chosen, rejected) completions.

Reference: https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

logger = logging.getLogger(__name__)

# Stabilization constant used in the DPO loss.
_DPO_LOG_BETA = 1e-6


@dataclass
class DPODataset:
    """A list of preference pairs: (prompt, chosen, rejected)."""

    prompts: List[List[int]]
    chosen: List[List[int]]
    rejected: List[List[int]]
    max_len: int = 1024

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "prompt": self.prompts[idx],
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx],
        }


def load_dpo_data(path: str, tokenizer, max_len: int = 1024) -> DPODataset:
    """
    Load a JSONL file of preference pairs.

    Each line: ``{"prompt": str, "chosen": str, "rejected": str}``.
    """
    prompts, chosen, rejected = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts.append(tokenizer.encode(rec["prompt"])[:max_len])
            chosen.append(tokenizer.encode(rec["chosen"])[:max_len])
            rejected.append(tokenizer.encode(rec["rejected"])[:max_len])
    return DPODataset(prompts, chosen, rejected, max_len=max_len)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute the DPO loss.

    Args:
        policy_chosen_logps: Sum of log-probs of the chosen completion
            under the policy model.
        policy_rejected_logps: Same for the rejected completion.
        ref_chosen_logps: Sum of log-probs under the (frozen) reference model.
        ref_rejected_logps: Same for the rejected completion.
        beta: Temperature parameter scaling the implicit reward.
        label_smoothing: If > 0, applies label smoothing to the loss.

    Returns:
        Scalar DPO loss (mean over batch).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)
    # Standard DPO loss:
    #   -log sigmoid(beta * (log_ratio_policy - log_ratio_ref))
    loss = -F.logsigmoid(logits)

    if label_smoothing > 0:
        # Rewrite as a soft-labeled BCE.
        loss = (1 - label_smoothing) * loss - label_smoothing * F.logsigmoid(-logits)

    return loss.mean()


def _sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Sum the log-probabilities of ``labels`` under ``logits``.

    Args:
        logits: (B, T, V).
        labels: (B, T) target ids; positions equal to ignore_index are masked.
    Returns:
        (B,) sum of log-probs for each sequence.
    """
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    log_probs = log_probs.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = labels != ignore_index
    return (log_probs * mask).sum(dim=-1)


def dpo_train_step(
    model: nn.Module,
    ref_model: nn.Module,
    batch: dict,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Run one DPO training step (forward only; caller does .backward()).

    Args:
        model: Policy model being trained.
        ref_model: Frozen reference model (SFT init).
        batch: dict with prompt/chosen/rejected token lists.
    """
    device = next(model.parameters()).device

    # Build concatenated sequences: prompt + chosen, prompt + rejected.
    def build(completions: List[List[int]]) -> tuple:
        max_len = max(len(p) + len(c) for p, c in zip(batch["prompt"], completions))
        max_len = min(max_len, model.config.block_size)
        input_ids, labels = [], []
        for p, c in zip(batch["prompt"], completions):
            seq = (p + c)[:max_len]
            lab = [-100] * len(p) + list(c)
            lab = lab[:max_len]
            pad = max_len - len(seq)
            input_ids.append(seq + [0] * pad)
            labels.append(lab + [-100] * pad)
        return (
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
        )

    chosen_ids, chosen_labels = build(batch["chosen"])
    rejected_ids, rejected_labels = build(batch["rejected"])

    with torch.no_grad():
        ref_chosen_out = ref_model(chosen_ids, targets=chosen_labels)
        ref_rejected_out = ref_model(rejected_ids, targets=rejected_labels)

    policy_chosen_out = model(chosen_ids, targets=chosen_labels)
    policy_rejected_out = model(rejected_ids, targets=rejected_labels)

    policy_chosen_logps = _sequence_log_probs(policy_chosen_out["logits"], chosen_labels)
    policy_rejected_logps = _sequence_log_probs(policy_rejected_out["logits"], rejected_labels)
    ref_chosen_logps = _sequence_log_probs(ref_chosen_out["logits"], chosen_labels)
    ref_rejected_logps = _sequence_log_probs(ref_rejected_out["logits"], rejected_labels)

    return dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )


def build_dpo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantiva DPO")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--policy_checkpoint", type=str, required=True)
    parser.add_argument("--ref_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="")
    return parser


def dpo(args: argparse.Namespace) -> None:
    """Run the DPO training loop."""
    from quantiva.model.config import ModelConfig
    from quantiva.model.gpt import GPT
    from quantiva.tokenizer.factory import get_tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy model.
    policy_payload = torch.load(args.policy_checkpoint, map_location=device)
    policy = GPT(ModelConfig.from_dict(policy_payload["model_config"]))
    policy.load_state_dict(policy_payload["model"])
    policy.to(device)
    policy.train()

    # Load frozen reference model.
    ref_payload = torch.load(args.ref_checkpoint, map_location=device)
    ref_model = GPT(ModelConfig.from_dict(ref_payload["model_config"]))
    ref_model.load_state_dict(ref_payload["model"])
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    tokenizer = get_tokenizer("tiktoken", encoding_name="gpt2")
    dataset = load_dpo_data(args.data_path, tokenizer, max_len=policy.config.block_size)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    idx = 0
    for it in range(args.max_iters):
        batch = {"prompt": [], "chosen": [], "rejected": []}
        for _ in range(args.batch_size):
            sample = dataset[idx % len(dataset)]
            idx += 1
            batch["prompt"].append(sample["prompt"])
            batch["chosen"].append(sample["chosen"])
            batch["rejected"].append(sample["rejected"])

        loss = dpo_train_step(policy, ref_model, batch, beta=args.beta, label_smoothing=args.label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if it % args.log_interval == 0:
            logger.info("iter %d: dpo_loss %.4f", it, loss.item())

    os.makedirs(args.out_dir, exist_ok=True)
    policy.save(os.path.join(args.out_dir, "ckpt.pt"))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dpo(build_dpo_parser().parse_args())


if __name__ == "__main__":
    main()

