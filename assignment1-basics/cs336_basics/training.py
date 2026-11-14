"""Train a TransformerLM on pre-tokenized data.

Example:
    python -m cs336_basics.training \
        --train_data data/ts_train.bin --valid_data data/ts_valid.bin \
        --data_dtype uint16 \
        --vocab_size 10000 --context_length 256 \
        --num_layers 4 --d_model 512 --num_heads 16 --d_ff 1344 --rope_theta 10000 \
        --batch_size 32 --max_iters 5000 \
        --lr_max 1e-3 --lr_min 1e-4 --warmup_iters 200 --cosine_iters 5000 \
        --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --grad_clip 1.0 \
        --log_interval 10 --eval_interval 200 --eval_iters 20 --ckpt_interval 1000 \
        --checkpoint_dir checkpoints/ts_run1
"""

import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.utils import (
    ADAMW,
    cross_entropy,
    data_loading,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)


# --------------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()

    # data / io
    p.add_argument("--train_data", default="data/train.bin")
    p.add_argument("--valid_data", default="data/valid.bin")
    p.add_argument("--data_dtype", default="uint16")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume_from", default=None, help="checkpoint path to resume from")

    # model (TinyStories config from the handout, ~17M non-embedding params)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # optimizer
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--cosine_iters", type=int, default=5000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # loop (32 x 5000 x 256 ~= 40M tokens, the handout's low-resource budget)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--ckpt_interval", type=int, default=1000)

    # system
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2. Data (memory-efficient)
# --------------------------------------------------------------------------- #
def load_tokens(path, dtype):
    """Return a read-only memmap over the raw token ids on disk (no header)."""
    return np.memmap(path, dtype=dtype, mode="r")


# --------------------------------------------------------------------------- #
# 3. Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(model, data, args):
    """Average loss over args.eval_iters random batches from `data`."""
    model.eval()
    losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        x, y = data_loading(data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        losses[k] = cross_entropy(logits, y).item()
    model.train()
    return losses.mean().item()


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # --- setup -------------------------------------------------------------
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- data --------------------------------------------------------------
    train_data = load_tokens(args.train_data, args.data_dtype)
    valid_data = load_tokens(args.valid_data, args.data_dtype)

    # --- model + optimizer -------------------------------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
    )
    optimizer = ADAMW(
        model.parameters(),
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # --- resume ------------------------------------------------------------
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    # --- training loop -----------------------------------------------------
    model.train()
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        # (a) learning rate for this step
        lr = learning_rate_schedule(it, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # (b) batch
        x, y = data_loading(train_data, args.batch_size, args.context_length, args.device)

        # (c) forward + loss
        logits = model(x)
        loss = cross_entropy(logits, y)

        # (d) backward + clip + step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # (e) train logging
        if it % args.log_interval == 0:
            print(f"iter {it:6d} | train loss {loss.item():.4f} | lr {lr:.2e} | {time.time() - t0:7.1f}s")

        # (f) validation
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            val_loss = estimate_loss(model, valid_data, args)
            print(f"iter {it:6d} | val loss   {val_loss:.4f} | {time.time() - t0:7.1f}s")

        # (g) checkpoint (store it + 1 = the next step to run, so resuming doesn't repeat step `it`)
        if it > 0 and it % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, it + 1, os.path.join(args.checkpoint_dir, f"ckpt_{it:06d}.pt"))

    # --- final save --------------------------------------------------------
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.checkpoint_dir, "final.pt"))


if __name__ == "__main__":
    main()
