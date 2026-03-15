"""AdamW optimizer and checkpoint utilities for CS336 Assignment 1.

AdamW algorithm follows Loshchilov and Hutter [2019]:
  m ← β1*m + (1-β1)*g
  v ← β2*v + (1-β2)*g²
  αt ← α * √(1-β2^t) / (1-β1^t)
  θ ← θ - αt * m/(√v + ε)
  θ ← θ - α*λ*θ   (weight decay)
"""

from __future__ import annotations

import math
import os
from typing import IO, BinaryIO

import torch
import torch.nn as nn


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialise state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected learning rate (per the assignment spec)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Parameter update: θ ← θ - αt * m/(√v + ε)
                p.addcdiv_(m, torch.sqrt(v) + eps, value=-alpha_t)

                # Decoupled weight decay: θ ← θ - α*λ*θ
                p.add_(p, alpha=-lr * weight_decay)

        return loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer, and iteration to disk."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer from a checkpoint. Returns the iteration number."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
