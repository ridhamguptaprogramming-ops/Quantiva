"""
Pretraining entry point.

Trains a GPT-style model from scratch on a tokenized corpus (``train.bin`` /
``val.bin``). Supports the classic nanoGPT workflow plus the new framework's
config-driven design.

Example:
    python -m quantiva.training.pretrain \\
        --train_path data/shakespeare_char/train.bin \\
        --val_path data/shakespeare_char/val.bin \\
        --vocab_size 65 --block_size 256 --n_layer 6 --n_head 6 --n_embd 384 \\
        --max_iters 5000 --device mps --out_dir out-shakespeare-char
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
import torch.nn as nn

from quantiva.data.dataloader import DataLoader
from quantiva.model.config import ModelConfig
from quantiva.model.gpt import GPT
from quantiva.training.distributed import (
    get_rank,
    get_world_size,
    init_process_group,
    wrap_ddp,
)
from quantiva.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def add_pretrain_args(parser: argparse.ArgumentParser) -> None:
    """Add pretraining-specific CLI args."""
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--pos_emb", type=str, default="learned")
    parser.add_argument("--flash_attn", type=bool, default=True)


def build_pretrain_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser for pretraining."""
    parser = argparse.ArgumentParser(description="Quantiva pretraining")
    add_pretrain_args(parser)

    # Training config.
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_iters", type=int, default=600000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--lr_decay_iters", type=int, default=600000)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", type=bool, default=False)

    # Environment.
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--wandb_log", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="quantiva")
    parser.add_argument("--wandb_run_name", type=str, default="pretrain")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--init_from", type=str, default="scratch")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    return parser


def pretrain(args: argparse.Namespace) -> None:
    """Run pretraining."""
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    init_process_group()
    rank = get_rank()
    world_size = get_world_size()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.split(":")[0]

    # Model.
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        pos_emb=args.pos_emb,
        flash_attn=args.flash_attn,
    )
    model = GPT(model_config)

    if args.init_from == "resume" or args.init_checkpoint:
        ckpt_path = args.init_checkpoint or os.path.join(args.out_dir, "ckpt.pt")
        if os.path.exists(ckpt_path):
            payload = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(payload["model"])
            logger.info("Loaded model from %s", ckpt_path)

    # Data loaders (per-rank slices).
    def make_loader(split: str) -> DataLoader:
        path = args.train_path if split == "train" else args.val_path
        return DataLoader(
            path,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
            process_rank=rank,
            num_processes=world_size,
            split=split,
        )

    train_loader = make_loader("train")
    val_loader = make_loader("val") if args.val_path else None

    # Wrap in DDP.
    if world_size > 1:
        model = wrap_ddp(model, torch.device(device))

    # Training config.
    tconfig = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
        dtype=args.dtype,
        compile=args.compile,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        out_dir=args.out_dir,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume=args.resume,
        device=device,
    )

    trainer = Trainer(
        model=model,
        config=tconfig,
        train_batch_fn=train_loader.get_batch,
        val_batch_fn=val_loader.get_batch if val_loader else None,
    )

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(model_config.to_dict(), f, indent=2)

    trainer.train()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_pretrain_parser()
    args = parser.parse_args()
    pretrain(args)


if __name__ == "__main__":
    main()

