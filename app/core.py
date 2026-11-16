"""Corpus preparation and the byte-pair-encoding (BPE) tokenizer.

Pure Python, standard library only, so everything here can be tested without PyTorch.

Byte-pair encoding in one paragraph: start with one token per byte (256 tokens).
Find the pair of neighbouring tokens that occurs most often in the training text,
give that pair a new token id, and replace it everywhere. Repeat until the
vocabulary reaches the wanted size. Common words and word parts such as
"cardi", "itis" or " the" end up as single tokens, so the model sees fewer, more
meaningful units than it would with single characters.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Corpus preparation
# --------------------------------------------------------------------------

_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",      # curly single quotes
    "\u201c": '"', "\u201d": '"',      # curly double quotes
    "\u2013": "-", "\u2014": "-",      # en / em dash
    "\u2026": "...",                   # ellipsis
    "\u00a0": " ",                     # non-breaking space
}

_GUTENBERG_START = re.compile(
    r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
_GUTENBERG_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
# Illustration captions, page markers and transcriber notes inside square brackets.
_NOISE = re.compile(
    r"\[(?:Illustration|Image|Pg|Footnote|Transcriber)[^\]]{0,400}\]", re.IGNORECASE | re.DOTALL
)

MIN_PARAGRAPH_CHARS = 10


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg licence header and footer, if present."""
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def prepare_corpus(raw: str) -> str:
    """Turn a raw text file into clean paragraphs separated by blank lines.

    * drops the Project Gutenberg header and footer
    * normalises punctuation and removes illustration / page markers
    * re-flows hard-wrapped lines, so a paragraph is one line of text
    * drops tiny paragraphs such as page numbers

    Running it twice gives the same result.
    """
    text = strip_gutenberg(raw)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NOISE.sub(" ", text)
    text = text.replace("_", "")                 # Gutenberg marks italics with underscores
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        paragraph = " ".join(block.split())
        if len(paragraph) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs) + "\n" if paragraphs else ""


def limit_chars(text: str, max_chars: int | None) -> str:
    """Cut the text to about `max_chars`, ending on a paragraph boundary."""
    if max_chars is None or len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    return text[: cut if cut > 0 else max_chars].rstrip() + "\n"


def split_paragraphs(text: str, val_fraction: float = 0.1, seed: int = 0) -> tuple[str, str]:
    """Shuffle paragraphs (reproducibly) and hold some out as validation text.

    Splitting by paragraph, not by position, puts a bit of every source file in
    both parts.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        raise ValueError("need at least 2 paragraphs to hold some out for validation")
    random.Random(seed).shuffle(paragraphs)
    n_val = min(len(paragraphs) - 1, max(1, round(len(paragraphs) * val_fraction)))
    val, train = paragraphs[:n_val], paragraphs[n_val:]
    return "\n\n".join(train) + "\n", "\n\n".join(val) + "\n"


# --------------------------------------------------------------------------
# Byte-pair-encoding tokenizer
# --------------------------------------------------------------------------

# Split text into word-like chunks (a word keeps its leading space), so merges
# never cross word boundaries. Every character is matched by one alternative,
# so joining the chunks always gives back the original text.
_PRETOKEN = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+")


def _merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE: ids 0-255 are raw bytes, ids 256+ are learned merges."""

    def __init__(self, merges=None):
        self.merges: dict[tuple[int, int], int] = {}     # pair -> new id, in learning order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._cache: dict[str, list[int]] = {}
        for pair in merges or []:
            self._add_merge((int(pair[0]), int(pair[1])))

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _add_merge(self, pair: tuple[int, int]) -> int:
        new_id = 256 + len(self.merges)
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._cache.clear()
        return new_id

    # ---- training ----------------------------------------------------------

    def train(self, text: str, vocab_size: int, min_count: int = 2) -> "BPETokenizer":
        """Learn merges until the vocabulary has `vocab_size` tokens.

        Works on the distinct words of the text (weighted by how often each
        occurs) and updates pair counts incrementally, so it stays fast.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the raw bytes)")
        words: Counter = Counter(tuple(chunk.encode("utf-8")) for chunk in _PRETOKEN.findall(text))

        pair_counts: Counter = Counter()
        index: dict[tuple[int, int], set] = defaultdict(set)     # pair -> words containing it
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
                index[pair].add(word)

        while self.vocab_size < vocab_size and pair_counts:
            # Most frequent pair; ties are broken by the pair itself so runs are reproducible.
            best, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if best_count < min_count:
                break
            new_id = self._add_merge(best)
            for word in list(index.pop(best, ())):
                count = words.pop(word)
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    if pair in index:
                        index[pair].discard(word)
                merged = tuple(_merge(word, best, new_id))
                words[merged] += count
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += count
                    index[pair].add(merged)
        return self

    # ---- encoding and decoding ---------------------------------------------

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge available (lowest new id).
            best = min(zip(ids, ids[1:]), key=lambda pair: self.merges.get(pair, float("inf")))
            if best not in self.merges:
                break
            ids = _merge(ids, best, self.merges[best])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _PRETOKEN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        """The text one token stands for (useful for showing tokens)."""
        return self.decode([token_id])

    def pieces(self, text: str) -> list[str]:
        """Split text into its BPE tokens, as strings."""
        return [self.token_text(i) for i in self.encode(text)]

    # ---- saving ---------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"merges": [[a, b] for a, b in self.merges]}

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        return cls(data["merges"])
