"""Neural network utility functions: softmax, cross-entropy, gradient clipping, batching, LR schedule."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy loss from unnormalized logits.

    Args:
        inputs: (batch_size, vocab_size) unnormalized logits
        targets: (batch_size,) target class indices
    """
    # Numerically stable log-softmax: cancel log and exp
    x_max = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(inputs - x_max).sum(dim=-1, keepdim=True)) + x_max
    log_probs = inputs - log_sum_exp  # (batch_size, vocab_size)

    # Gather the log prob of the correct class for each example
    nll = -log_probs[torch.arange(inputs.size(0)), targets]
    return nll.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip combined gradient L2 norm in-place."""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    total_norm_sq = sum((p.grad ** 2).sum() for p in params_with_grad)
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in params_with_grad:
            p.grad.mul_(scale)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (x, y) pairs from a 1D dataset. y is x shifted right by 1."""
    max_start = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start, size=(batch_size,))

    x = torch.stack([torch.from_numpy(dataset[i : i + context_length].copy()) for i in start_indices])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + 1 + context_length].copy()) for i in start_indices])

    return x.to(device=device), y.to(device=device)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_iters:
        # Linear warmup
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        # Cosine decay
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    else:
        # Hold at min
        return min_learning_rate
