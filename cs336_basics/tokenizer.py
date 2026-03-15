"""
BPE Tokenizer: encode text → int IDs, decode int IDs → text.

Uses the same GPT-2 pre-tokenization regex as training.
Special tokens are never split and always map to a single ID.
"""

import regex


GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab  # id → bytes
        self.bytes_to_id = {v: k for k, v in vocab.items()}  # bytes → id

        # Merge priority: lower rank = merge first
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

        # Special tokens
        self.special_tokens = special_tokens or []
        # Add special tokens to vocab if not already present
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = st_bytes
                self.bytes_to_id[st_bytes] = new_id

        # Build special token lookup: bytes → str
        self.special_token_ids = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            self.special_token_ids[self.bytes_to_id[st_bytes]] = st

        # Build regex pattern to split on special tokens (longest first)
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [regex.escape(st) for st in sorted_specials]
            self._special_pattern = regex.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pattern = None

    def _apply_bpe(self, token_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token's bytes. Returns list of token IDs."""
        # Start with individual bytes
        parts = [bytes([b]) for b in token_bytes]

        while len(parts) > 1:
            # Find the pair with the lowest merge rank
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break  # No more merges possible

            # Merge the best pair
            merged = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2:]

        return [self.bytes_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        if not text:
            return []

        ids = []

        # Split on special tokens
        if self._special_pattern:
            segments = self._special_pattern.split(text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue

            # Check if this segment is a special token
            if segment in self.special_tokens:
                st_bytes = segment.encode("utf-8")
                ids.append(self.bytes_to_id[st_bytes])
                continue

            # GPT-2 pre-tokenize, then apply BPE to each pre-token
            pretokens = regex.findall(GPT2_PRETOKENIZE_PATTERN, segment)
            for pretoken in pretokens:
                token_bytes = pretoken.encode("utf-8")
                ids.extend(self._apply_bpe(token_bytes))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into text."""
        raw_bytes = b"".join(self.vocab[id] for id in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable) -> "Iterator[int]":
        """Streaming encode: yields token IDs one at a time from an iterable of text chunks.
        Memory-efficient — only one chunk is processed at a time."""
        for chunk in iterable:
            for id in self.encode(chunk):
                yield id
