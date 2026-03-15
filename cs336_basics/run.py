"""
CS336 Assignment 1 - BPE Training Deliverable
Train on TinyStories, measure time/memory, find longest token, serialize results.
"""
import time
import json
import os
import tracemalloc
from cs336_basics.train_bpe_review import train_bpe

EOT = "<" + "|endoftext|" + ">"

if __name__ == "__main__":
    # --- Train ---
    print("Training BPE on TinyStoriesV2-GPT4-train.txt ...")
    print(f"Special tokens: [{EOT}]")

    tracemalloc.start()
    t0 = time.time()

    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=[EOT],
    )

    elapsed = time.time() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Stats ---
    print(f"\n=== Results ===")
    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Peak memory: {peak_mem / 1e9:.2f} GB")

    # Longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token} (len={len(longest_token)})")
    try:
        print(f"Longest token decoded: '{longest_token.decode('utf-8')}'")
    except UnicodeDecodeError:
        print(f"Longest token (hex): {longest_token.hex()}")

    # --- Serialize ---
    os.makedirs("output", exist_ok=True)

    # Vocab as JSON: {id: list of byte values}
    vocab_json = {str(k): list(v) for k, v in vocab.items()}
    with open("output/vocab.json", "w") as f:
        json.dump(vocab_json, f, indent=2)

    # Merges as text file (GPT-2 style)
    with open("output/merges.txt", "w") as f:
        for left, right in merges:
            f.write(f"{left!r} {right!r}\n")

    print(f"\nSerialized to output/vocab.json and output/merges.txt")
