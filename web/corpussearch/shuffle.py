"""Text-shuffling utilities for the demo.

The point of these transforms is pedagogical: they let a user take a text that
is genuinely contained in a corpus and *reorder* its pieces while keeping the
local wording intact. This preserves the set of shared fingerprints (so the
"shared fingerprints" baseline keeps reporting a high score) but destroys the
long, position-coherent fingerprint *chains* that the position-aware method
depends on. The demo can thus show the two methods diverging on the same text.

All functions are pure, dependency-free and deterministic given ``seed``.

Two granularities are offered:

* :func:`shuffle_sentence_blocks` — keep every ``block_size`` consecutive
  sentences together, shuffle the order of those blocks.
* :func:`shuffle_word_blocks` — keep every ``block_size`` consecutive words
  together, shuffle the order of those blocks.

``block_size`` is the knob the user controls: larger blocks preserve longer
intact runs (harder to distinguish from the original), smaller blocks chop the
text more finely (the position-aware chains collapse fastest).
"""

from __future__ import annotations

import random
import re
from typing import Callable

__all__ = [
    "split_sentences",
    "shuffle_sentence_blocks",
    "shuffle_word_blocks",
    "shuffle_text",
    "STRATEGIES",
]

# A deliberately simple sentence splitter: break after ., ! or ? followed by
# whitespace. Keeps the terminator with the sentence. Good enough for the demo;
# we are only reordering, so occasional mis-splits are harmless.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, preserving each sentence's trailing space."""
    parts = [m.group(0) for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]
    return parts if parts else ([text] if text else [])


def _chunk(items: list, block_size: int) -> list[list]:
    """Group ``items`` into consecutive blocks of at most ``block_size``."""
    block_size = max(1, int(block_size))
    return [items[i:i + block_size] for i in range(0, len(items), block_size)]


def _shuffled_order(n: int, seed: int) -> list[int]:
    """A permutation of ``range(n)`` that is guaranteed to move something.

    Falls back to the identity only when ``n < 2`` (nothing can move).
    """
    order = list(range(n))
    if n < 2:
        return order
    rng = random.Random(seed)
    for _ in range(10):
        rng.shuffle(order)
        if any(i != j for i, j in enumerate(order)):
            break
    return order


def shuffle_sentence_blocks(text: str, block_size: int = 1, seed: int = 0) -> str:
    """Shuffle the order of blocks of ``block_size`` consecutive sentences."""
    sentences = split_sentences(text)
    blocks = _chunk(sentences, block_size)
    order = _shuffled_order(len(blocks), seed)
    reordered = ["".join(blocks[i]) for i in order]
    out = " ".join(s.strip() for s in reordered if s.strip())
    return out


def shuffle_word_blocks(text: str, block_size: int = 10, seed: int = 0) -> str:
    """Shuffle the order of blocks of ``block_size`` consecutive words."""
    words = text.split()
    blocks = _chunk(words, block_size)
    order = _shuffled_order(len(blocks), seed)
    reordered = [" ".join(blocks[i]) for i in order]
    return " ".join(reordered)


# Registry so callers (web layer, experiments) can pick a strategy by name.
STRATEGIES: dict[str, Callable[..., str]] = {
    "sentence_blocks": shuffle_sentence_blocks,
    "word_blocks": shuffle_word_blocks,
}


def shuffle_text(
    text: str, strategy: str = "sentence_blocks", block_size: int = 1, seed: int = 0
) -> str:
    """Dispatch to a named shuffle strategy."""
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown shuffle strategy {strategy!r}.")
    return fn(text, block_size=block_size, seed=seed)
