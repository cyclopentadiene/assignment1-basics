"""
Tokenize raw TinyStories text files into numpy arrays for memmap training.

Usage:
    python -m cs336_basics.tokenize_data
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(
    vocab_path: str = "output/vocab.json",
    merges_path: str = "output/merges.txt",
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Load a BPE tokenizer from serialized vocab and merges files."""
    # Parse vocab: {str(id): list_of_byte_values} -> {int: bytes}
    with open(vocab_path) as f:
        vocab_json = json.load(f)
    vocab = {int(k): bytes(v) for k, v in vocab_json.items()}

    # Parse merges: "b'...' b'...'" per line -> list[tuple[bytes, bytes]]
    # Use regex to find complete bytes literals (handles escape sequences like b'\n')
    bytes_literal_re = re.compile(r"""b(?:'[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")""")
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            matches = bytes_literal_re.findall(line)
            if len(matches) != 2:
                raise ValueError(f"Expected 2 bytes literals, got {len(matches)} in: {line!r}")
            left = ast.literal_eval(matches[0])
            right = ast.literal_eval(matches[1])
            merges.append((left, right))

    return Tokenizer(vocab, merges, special_tokens)


def tokenize_file(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
) -> int:
    """Tokenize a text file and save as numpy array.

    Returns the number of tokens.
    """
    print(f"Tokenizing {input_path} ...")
    t0 = time.time()

    # Stream tokenize line-by-line
    def line_iter():
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    tokens = list(tokenizer.encode_iterable(line_iter()))
    arr = np.array(tokens, dtype=np.uint16)

    # Save
    np.save(output_path, arr)
    elapsed = time.time() - t0

    print(f"  -> {len(arr):,} tokens, saved to {output_path} ({elapsed:.1f}s)")
    print(f"  -> dtype={arr.dtype}, max_id={arr.max()}, file_size={os.path.getsize(output_path) / 1e6:.1f} MB")
    return len(arr)


EOT = "<" + "|endoftext|" + ">"


if __name__ == "__main__":
    tokenizer = load_tokenizer(special_tokens=[EOT])
    print(f"Loaded tokenizer: {len(tokenizer.vocab)} tokens, {len(tokenizer.merge_priority)} merges")

    tokenize_file(
        tokenizer,
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        output_path="data/train_tokens.npy",
    )

    tokenize_file(
        tokenizer,
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        output_path="data/valid_tokens.npy",
    )

    print("\nDone!")
