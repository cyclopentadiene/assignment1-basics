"""Transformer model components for CS336 Assignment 1.

Architecture follows the LLaMA / modern pre-norm decoder-only transformer:
- Embedding → N × TransformerBlock → RMSNorm → LM head
- TransformerBlock = x + MHA(RMSNorm(x)), x + SwiGLU(RMSNorm(x))
- MHA uses causal masking and RoPE

All tensor operations use einops einsum/rearrange for self-documenting shapes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def silu(x: Tensor) -> Tensor:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """Linear layer without bias. y = Wx."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

    def forward(self, x: Tensor) -> Tensor:
        # "... d_in, d_out d_in -> ... d_out"
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding lookup table."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network (Shazeer, 2020)
# FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU FFN: W2(SiLU(W1·x) ⊙ W3·x)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
#
# Attention(Q, K, V) = softmax(Q^T K / √d_k) V   (column-vector math)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Scaled dot-product attention supporting arbitrary batch dims.

    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: (..., queries, keys) boolean mask. True = attend, False = mask out.
    """
    d_k = Q.size(-1)

    # QK^T: "... queries d_k, ... keys d_k -> ... queries keys"
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = attn_weights.nan_to_num(0.0)  # handle fully-masked rows

    # weighted sum: "... queries keys, ... keys d_v -> ... queries d_v"
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — Su et al., 2021
#
# θ_{i,k} = i / Θ^{(2k-2)/d}   for k ∈ {1, ..., d/2}
# Pairwise rotation: (x_{2k-1}, x_{2k}) rotated by θ_{i,k}
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    """Rotary Position Embeddings."""

    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        # Frequency for each pair: 1/Θ^{2i/d} for i in [0, d/2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # (max_seq_len, d_k/2)
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Apply RoPE rotation to x.

        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len) integer positions
        """
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]

        # Split into even/odd pairs
        x1 = x[..., 0::2]  # (..., seq_len, d_k/2) — even indices
        x2 = x[..., 1::2]  # (..., seq_len, d_k/2) — odd indices

        # Apply 2D rotation per pair
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave: "(... seq_len d_half) pair -> ... seq_len (d_half pair)"
        return rearrange(
            torch.stack((out1, out2), dim=-1),
            "... d_half pair -> ... (d_half pair)",
        )


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (Vaswani et al., 2017)
#
# MultiHeadSelfAttention(x) = W_O · Concat(head_1, ..., head_h)
# where head_i = Attention(Q_i, K_i, V_i)
#
# Causal masking is always applied (token i only attends to j ≤ i).
# RoPE is optionally applied to Q and K.
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        rope: RoPE | None = None,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (seq_len, d_model)
            rope: optional RoPE module to apply to Q and K
            token_positions: (batch, seq_len) positions for RoPE
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        batch, seq_len, _ = x.shape

        # Project Q, K, V — single matmul per projection across all heads
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: "batch seq (heads d_k) -> batch heads seq d_k"
        Q = rearrange(Q, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)

        # Apply RoPE to Q and K (same rotation for every head — head is batch dim)
        if rope is not None:
            if token_positions is not None:
                pos = token_positions
                if pos.dim() == 1:
                    pos = pos.unsqueeze(0).expand(batch, -1)
                # expand: "batch seq -> batch heads seq"
                pos = pos.unsqueeze(1).expand(-1, self.num_heads, -1)
            else:
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, -1)
            Q = rope(Q, pos)
            K = rope(K, pos)

        # Causal mask: token i can only attend to positions j ≤ i
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # Scaled dot-product attention with causal mask
        out = scaled_dot_product_attention(Q, K, V, causal_mask)  # (batch, heads, seq, d_k)

        # Concatenate heads: "batch heads seq d_k -> batch seq (heads d_k)"
        out = rearrange(out, "batch heads seq d_k -> batch seq (heads d_k)")

        # Output projection
        out = self.output_proj(out)

        if squeeze:
            out = out.squeeze(0)

        return out


# ---------------------------------------------------------------------------
# Transformer Block (Pre-norm)
#
# y = x + MultiHeadSelfAttention(RMSNorm(x))   (Eq 15)
# z = y + SwiGLU(RMSNorm(y))
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE, RMSNorm, causal MHA, and SwiGLU."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), rope=self.rope, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Language Model
#
# token_ids → Embedding → N × TransformerBlock → RMSNorm → LM head → logits
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """Full decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)
        rope = RoPE(d_model // num_heads, context_length, theta=rope_theta)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (batch, seq_len)

        Returns:
            (batch, seq_len, vocab_size) unnormalized logits
        """
        batch, seq_len = token_ids.shape
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
