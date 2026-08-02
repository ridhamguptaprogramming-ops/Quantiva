"""
Supervised Fine-Tuning (SFT).

Fine-tunes a pretrained GPT model on chat / instruction data. The loss is
masked so only assistant tokens are trained on (prompt tokens are ignored).

Supports:
  - Loading a pretrained checkpoint (init_from)
  - Chat-format datasets (JSONL with messages)
  - LoRA-compatible (see ``lora.py`` for wrapping)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

from quantiva.data.dataloader import DataLoader
from quantiva.datasets.preprocessing.formatting import (
    ChatMessage,
    apply_chat_template,
    format_sft_example,
)
from quantiva.model.config import ModelConfig
from quantiva.model.gpt import GPT
from quantiva.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def build_sft_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantiva SFT")
    parser.add_argument("--data_path", type=str, required=True, help="JSONL of chat data")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out-sft")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", choices=["tiktoken", "bpe", "sentencepiece"])
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--lr_decay_iters", type=int, default=10000)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--wandb_log", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="quantiva-sft")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def load_chat_data(path: str) -> List[List[ChatMessage]]:
    """Load a JSONL file where each line is ``{"messages": [...]}``."""
    examples: List[List[ChatMessage]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = [
                ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                for m in record.get("messages", [])
            ]
            if messages:
                examples.append(messages)
    logger.info("Loaded %d SFT examples from %s", len(examples), path)
    return examples


class SFTDataset:
    """In-memory SFT dataset yielding (input_ids, labels) batches."""

    def __init__(self, examples: List[List[ChatMessage]], tokenizer, block_size: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = 1
        self.data: List[tuple] = []
        for msgs in examples:
            input_ids, labels = format_sft_example(msgs, tokenizer)
            if len(input_ids) > 0:
                self.data.append((input_ids, labels))
        self._idx = 0

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, device: str) -> tuple:
        """Return a batch of (inputs, labels) padded to block_size."""
        batch_inputs = []
        batch_labels = []
        for _ in range(self.batch_size):
            if self._idx >= len(self.data):
                self._idx = 0
            input_ids, labels = self.data[self._idx]
            self._idx += 1
            # Truncate / pad to block_size.
            input_ids = input_ids[: self.block_size]
            labels = labels[: self.block_size]
            pad_len = self.block_size - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
        return (
            torch.tensor(batch_inputs, dtype=torch.long, device=device),
            torch.tensor(batch_labels, dtype=torch.long, device=device),
        )


def sft(args: argparse.Namespace) -> None:
    """Run supervised fine-tuning."""
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer.
    from quantiva.tokenizer.factory import get_tokenizer

    if args.tokenizer == "tiktoken":
        tokenizer = get_tokenizer("tiktoken", encoding_name="gpt2")
    elif args.tokenizer == "bpe":
        tokenizer = get_tokenizer("bpe", vocab_size=50304)
    else:
        tokenizer = get_tokenizer("sentencepiece", model_path=args.tokenizer_path)

    # Model.
    payload = torch.load(args.init_checkpoint, map_location=device)
    model_config = ModelConfig.from_dict(payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.train()

    # Data.
    examples = load_chat_data(args.data_path)
    dataset = SFTDataset(examples, tokenizer, model_config.block_size)
    dataset.batch_size = args.batch_size
    val_split = int(len(dataset) * 0.95)
    val_dataset = SFTDataset(examples[val_split:], tokenizer, model_config.block_size)
    val_dataset.batch_size = args.batch_size

    tconfig = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        dtype=args.dtype,
        compile=args.compile,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        out_dir=args.out_dir,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        device=device,
    )

    trainer = Trainer(
        model=model,
        config=tconfig,
        train_batch_fn=lambda: dataset.get_batch(device),
        val_batch_fn=lambda: val_dataset.get_batch(device),
    )
    trainer.train()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_sft_parser().parse_args()
    sft(args)


if __name__ == "__main__":
    main()

