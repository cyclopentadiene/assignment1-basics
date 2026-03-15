"""
Text generation / decoding from a trained TransformerLM.

Supports temperature scaling and top-p (nucleus) sampling.

Usage:
    python -m cs336_basics.generate \
        --checkpoint output/checkpoint.pt \
        --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.tokenize_data import load_tokenizer


EOT = "<" + "|endoftext|" + ">"


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Apply top-p (nucleus) filtering to logits.

    Keep the smallest set of tokens whose cumulative probability >= top_p.
    Set all other logits to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Find cutoff: first index where cumulative prob >= top_p
    # Keep all tokens up to and including this index
    sorted_mask = cumulative_probs - probs >= top_p  # mask tokens AFTER the cutoff
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order
    filtered_logits = torch.empty_like(logits)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt using the trained model.

    Args:
        model: Trained TransformerLM in eval mode.
        tokenizer: BPE Tokenizer instance.
        prompt: Text prompt to complete.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Softmax temperature (higher = more random).
        top_p: Nucleus sampling threshold (1.0 = no filtering).
        device: Device to run on.

    Returns:
        Generated text (prompt + completion).
    """
    model.eval()
    context_length = model.context_length

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        token_ids = [0]  # fallback

    # Find EOT token ID
    eot_bytes = EOT.encode("utf-8")
    eot_id = tokenizer.bytes_to_id.get(eot_bytes)

    generated = list(token_ids)

    for _ in range(max_tokens):
        # Truncate to context_length if needed
        context = generated[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        # Forward pass
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Top-p filtering
        if top_p < 1.0:
            next_logits = top_p_filter(next_logits, top_p)

        # Sample
        probs = softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Check for EOT
        if eot_id is not None and next_token == eot_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to complete")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, mps, cuda)")
    # Model config (must match training)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(special_tokens=[EOT])
    print(f"Loaded tokenizer: {len(tokenizer.vocab)} tokens")

    # Build model and load checkpoint
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {num_params:,} parameters")
    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("-" * 60)

    text = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print(text)


if __name__ == "__main__":
    main()
