"""
Training loop for TransformerLM on TinyStories.

Ties together: model, optimizer, cosine LR, gradient clipping, data loading,
checkpointing, and logging.

Usage:
    python -m cs336_basics.train --device mps
    python -m cs336_basics.train --device cpu --compile
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.optimizer import AdamW, save_checkpoint


def load_data(path: str) -> np.ndarray:
    """Load tokenized data as a memory-mapped numpy array."""
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int = 20,
) -> float:
    """Estimate validation loss over a number of batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(eval_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        x, y = x.long(), y.long()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / eval_batches


def main():
    parser = argparse.ArgumentParser(description="Train TransformerLM on TinyStories")

    # Data
    parser.add_argument("--train-data", type=str, default="data/train_tokens.npy")
    parser.add_argument("--val-data", type=str, default="data/valid_tokens.npy")

    # Model architecture
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=4883,
                        help="~40M tokens / (32 * 256)")
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Evaluation & checkpointing
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="output")

    # Device & optimization
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for speedup")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cs336-tinystories")

    args = parser.parse_args()

    # ---- Setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Training for {args.num_steps} steps, batch_size={args.batch_size}, "
          f"context_length={args.context_length}")
    print(f"Total tokens: ~{args.num_steps * args.batch_size * args.context_length:,}")

    # ---- Load data ----
    print("Loading data...")
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Valid: {len(val_data):,} tokens")

    # ---- Build model ----
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    num_non_emb = num_params - model.token_embeddings.weight.numel()
    print(f"Model: {num_params:,} total params, {num_non_emb:,} non-embedding params")

    # ---- torch.compile ----
    if args.compile:
        if args.device == "mps":
            print("Compiling model with backend='aot_eager' (MPS)...")
            model = torch.compile(model, backend="aot_eager")
        else:
            print("Compiling model...")
            model = torch.compile(model)

    # ---- Optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )

    # ---- wandb ----
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ---- Training loop ----
    model.train()
    t0 = time.time()
    best_val_loss = float("inf")

    for step in range(args.num_steps):
        # Update learning rate
        lr = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr,
            args.warmup_steps, args.num_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        x, y = x.long(), y.long()

        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # ---- Logging ----
        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length,
                args.device, args.eval_batches,
            )

            elapsed = time.time() - t0
            tokens_per_sec = (step + 1) * args.batch_size * args.context_length / elapsed
            print(
                f"Step {step:5d}/{args.num_steps} | "
                f"train_loss={loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"tok/s={tokens_per_sec:.0f} | "
                f"elapsed={elapsed:.0f}s"
            )

            if args.wandb:
                import wandb
                wandb.log({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # ---- Checkpointing ----
        if (step > 0 and step % args.checkpoint_interval == 0) or step == args.num_steps - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

    # ---- Summary ----
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final checkpoint: {os.path.join(args.checkpoint_dir, 'checkpoint.pt')}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
