"""
BPE Training Implementation

Key gotchas we debugged:
1. _merge_pair: must skip i+=2 (not i+=1) after merging a pair
2. _merge_pair: when two words merge to the same result, accumulate freq (+=), don't overwrite (=)
3. Special tokens: must split text BEFORE pre-tokenization so their bytes never get merged
4. Tie-breaking: when pairs have equal frequency, break ties by byte content of
   (left_token_bytes, right_token_bytes) -- NOT by token IDs, because compound tokens
   get IDs > 256 that don't correspond to byte ordering
5. Performance: use an inverted index (pair -> words containing it) so each merge
   only touches affected words, not all words
6. Performance: use a lazy max-heap for O(log P) pair selection instead of O(P) max() scan
7. Performance: use multiprocessing to parallelize pre-tokenization across CPU cores
"""

import heapq
import multiprocessing
import os
import regex
from collections import defaultdict


GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize(text: str) -> list[str]:
    return regex.findall(GPT2_PRETOKENIZE_PATTERN, text)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """GOTCHA #3: Special tokens must be removed from text BEFORE pre-tokenization.
    Otherwise their constituent bytes (e.g. '<', '|') participate in merges
    and end up as parts of merged tokens in the vocab."""
    if not special_tokens:
        return [text]
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    escaped = [regex.escape(st) for st in sorted_specials]
    pattern = "|".join(escaped)
    chunks = regex.split(pattern, text)
    return [chunk for chunk in chunks if chunk]


def _pretokenize_chunk(args):
    """Worker: read chunk from disk, split on special tokens, pre-tokenize, return freqs.
    Must read the full chunk to preserve cross-line whitespace like \\n\\n."""
    filepath, start_byte, end_byte, special_tokens = args
    word_freqs: dict[tuple[int, ...], int] = {}

    with open(filepath, "rb") as f:
        f.seek(start_byte)
        raw = f.read(end_byte - start_byte)
    text_chunk = raw.decode("utf-8", errors="replace")
    del raw  # free the bytes copy immediately

    for chunk in _split_on_special_tokens(text_chunk, special_tokens):
        for pretoken in _pretokenize(chunk):
            byte_seq = tuple(pretoken.encode("utf-8"))
            word_freqs[byte_seq] = word_freqs.get(byte_seq, 0) + 1
    return word_freqs


def _find_chunk_boundaries(filepath: str, num_chunks: int, special_token: str) -> list[int]:
    """Find byte offsets to split the file into roughly equal chunks.
    Always splits at a newline boundary so no word is cut in half.
    Prefers splitting right after a special token if one is nearby."""
    file_size = os.path.getsize(filepath)
    chunk_size = file_size // num_chunks
    boundaries = [0]

    with open(filepath, "rb") as f:
        for i in range(1, num_chunks):
            # Seek to approximate boundary
            target = chunk_size * i
            f.seek(target)
            # Read ahead to find the next special token or newline
            search_buf = f.read(min(100_000, file_size - target))
            token_bytes = special_token.encode("utf-8")
            idx = search_buf.find(token_bytes)
            if idx != -1 and idx < 50_000:
                # Split right after the special token
                boundaries.append(target + idx + len(token_bytes))
            else:
                # Fall back to splitting at the nearest newline
                nl_idx = search_buf.find(b"\n")
                if nl_idx != -1:
                    boundaries.append(target + nl_idx + 1)
                else:
                    boundaries.append(target)

    boundaries.append(file_size)
    return boundaries


class _Negated:
    """Wrapper that inverts comparison, turning min-heap into max-heap."""
    __slots__ = ('val',)
    def __init__(self, val):
        self.val = val
    def __lt__(self, other):
        return self.val > other.val
    def __eq__(self, other):
        return self.val == other.val


def _heap_push(heap, pair, count, token_bytes_map):
    """Push a pair onto the max-heap with its count and byte-based tie-breaker."""
    tb = (token_bytes_map[pair[0]], token_bytes_map[pair[1]])
    heapq.heappush(heap, (-count, _Negated(tb), pair))


def _heap_pop_best(heap, pair_counts):
    """Pop the best (highest count, with tie-breaking) valid pair from the heap.
    Stale entries (where count changed) are lazily discarded."""
    while heap:
        neg_count, neg_tb, pair = heapq.heappop(heap)
        # Check if this entry is still valid (count matches current)
        if pair in pair_counts and pair_counts[pair] == -neg_count:
            return pair
    return None


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Step 1: Parallel pre-tokenization using multiprocessing.
    # Split file into chunks at special token boundaries, each worker pre-tokenizes
    # its chunk independently, then merge word_freqs dicts.
    # Use 'fork' context to avoid 'spawn' overhead and __main__ guard requirement.
    num_workers = min(2, os.cpu_count() or 2)
    primary_special = special_tokens[0] if special_tokens else "\n"
    boundaries = _find_chunk_boundaries(str(input_path), num_workers, primary_special)

    # Workers read their own slices from disk independently (byte offset args only).
    chunks_with_args = []
    for i in range(len(boundaries) - 1):
        chunks_with_args.append((str(input_path), boundaries[i], boundaries[i + 1], special_tokens))

    # Parallel pre-tokenization with fork context
    word_freqs: dict[tuple[int, ...], int] = {}
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(num_workers, maxtasksperchild=1) as pool:
        results = pool.map(_pretokenize_chunk, chunks_with_args)

    # Merge results from all workers
    for partial in results:
        for word, freq in partial.items():
            word_freqs[word] = word_freqs.get(word, 0) + freq

    # Step 2: Base vocab = all 256 single bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    num_merges_needed = vocab_size - 256 - len(special_tokens)

    merges: list[tuple[bytes, bytes]] = []
    token_bytes_map: dict[int, bytes] = dict(vocab)

    # Step 3: Build initial pair counts AND inverted index
    pair_index: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    pair_counts: dict[tuple[int, int], int] = {}

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_index[pair].add(word)
            pair_counts[pair] = pair_counts.get(pair, 0) + freq

    # OPTIMIZATION: Build max-heap for O(log P) pair selection instead of O(P)
    # Heap entries: (-count, negated_tie_break_bytes, pair)
    heap = []
    for pair, count in pair_counts.items():
        _heap_push(heap, pair, count, token_bytes_map)

    # Step 4: Greedy merge loop
    for merge_i in range(num_merges_needed):
        # Lazy-pop: skip stale entries until we find a valid one
        best_pair = _heap_pop_best(heap, pair_counts)
        if best_pair is None:
            break
        new_token_id = 256 + merge_i

        left_bytes = token_bytes_map[best_pair[0]]
        right_bytes = token_bytes_map[best_pair[1]]
        merges.append((left_bytes, right_bytes))
        new_token_bytes = left_bytes + right_bytes
        token_bytes_map[new_token_id] = new_token_bytes
        vocab[new_token_id] = new_token_bytes

        left, right = best_pair

        # Only process words that contain the best pair (via inverted index)
        affected_words = list(pair_index.pop(best_pair, []))
        if best_pair in pair_counts:
            del pair_counts[best_pair]

        for word in affected_words:
            if word not in word_freqs:
                continue  # word was already replaced by a previous merge in this batch
            freq = word_freqs[word]

            # GOTCHA #1: i += 2, not i += 1. We consumed BOTH tokens of the pair.
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                    new_word.append(new_token_id)
                    i += 2  # skip BOTH elements of the merged pair
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_t = tuple(new_word)

            if new_word_t == word:
                continue

            # Remove old pair contributions (skip best_pair, already deleted)
            for i in range(len(word) - 1):
                old_pair = (word[i], word[i + 1])
                if old_pair == best_pair:
                    continue  # already removed above
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    if old_pair in pair_counts:
                        del pair_counts[old_pair]
                else:
                    # Re-push with updated (lower) count
                    _heap_push(heap, old_pair, pair_counts[old_pair], token_bytes_map)
                pair_index[old_pair].discard(word)

            # Add new pair contributions from the merged word
            for i in range(len(new_word_t) - 1):
                new_pair = (new_word_t[i], new_word_t[i + 1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
                pair_index[new_pair].add(new_word_t)
                # Push updated count to heap (old entries become stale, lazily skipped)
                _heap_push(heap, new_pair, pair_counts[new_pair], token_bytes_map)

            # GOTCHA #2: When two different pre-token sequences merge into
            # the same result, their frequencies must be SUMMED (+=), not
            # overwritten (=). E.g. "ab" and "a b" could both become "ab".
            del word_freqs[word]
            word_freqs[new_word_t] = word_freqs.get(new_word_t, 0) + freq

    # Step 5: Add special tokens at the end of vocab
    next_id = 256 + len(merges)
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    return vocab, merges
