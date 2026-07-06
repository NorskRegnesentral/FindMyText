"""Empirical comparison of text-shuffling strategies for the demo.

Loads a real corpus index and, for each sample text, measures how the two
detection scores respond to different shuffling strategies / block sizes:

* baseline  = shared-fingerprint count for the top match (``find_matches_jaccard``)
* ours      = largest position-aware cluster for the top match
              (``find_matches_clustering``)

The goal we are probing: a good shuffle keeps ``baseline`` high (the material is
still there) while driving ``ours`` down (the contiguous chains are broken), so
the demo can visibly separate the two methods on a single pasted text.

Run:  ../.venv/bin/python shuffle_experiment.py [corpus_id]
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from detector import TextContainmentDetector  # noqa: E402
from corpussearch.shuffle import shuffle_sentence_blocks, shuffle_word_blocks  # noqa: E402

INDEX_ROOT = "/home/jullum/copyai_local"
INDEX_DIRS = {
    "arxiv": "arxiv/index(4,6)",
    "wiki": "wiki/index(4,6)_wikipedia",
    "hplt": "hplt/index(4,6)",
}

# Same clustering params the web app uses (config.json).
CPARAMS = {
    "method": "rectangle",
    "position_threshold": 10,
    "offset_threshold": 10,
    "min_cluster_size": 5,
}
NORM_C = 8.0  # avg winnowed fingerprints per sentence (for readability)


def top_scores(detector: TextContainmentDetector, text: str) -> tuple[int, int]:
    """Return (baseline_top, ours_top) count scores for the best match."""
    jac = detector.find_matches_jaccard(text, score="count")
    clu = detector.find_matches_clustering(text, CPARAMS, score="count")
    j = max(jac.values()) if jac else 0
    c = max(clu.values()) if clu else 0
    return int(j), int(c)


def load_samples(corpus_id: str) -> list[dict]:
    path = os.path.join(_REPO_ROOT, "web", "corpussearch", "samples_data.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw.get(corpus_id, [])


def fmt(j: int, c: int) -> str:
    return f"base={j:>5} ({j / NORM_C:5.1f}s) | ours={c:>5} ({c / NORM_C:5.1f}s)"


def main() -> None:
    corpus_id = sys.argv[1] if len(sys.argv) > 1 else "arxiv"
    index_dir = os.path.join(INDEX_ROOT, INDEX_DIRS[corpus_id])
    print(f"Loading {corpus_id} index: {index_dir}")
    detector = TextContainmentDetector(index_dir, top_k=10, min_fingerprints=5)

    samples = load_samples(corpus_id)
    if not samples:
        print(f"No samples for corpus {corpus_id!r}.")
        return

    for s in samples[:6]:
        text = s["text"]
        label = s.get("label", "?")
        n_sent = len(text.split(". "))
        n_word = len(text.split())
        print("\n" + "=" * 100)
        print(f"SAMPLE: {label}  (~{n_sent} sentences, {n_word} words)")
        j0, c0 = top_scores(detector, text)
        print(f"  original                         {fmt(j0, c0)}")

        print("  -- sentence-block shuffle (keep k consecutive sentences intact) --")
        for k in (1, 2, 3, 5):
            variant = shuffle_sentence_blocks(text, block_size=k, seed=42)
            j, c = top_scores(detector, variant)
            drop = f"ours {100 * (1 - c / c0):4.0f}% down" if c0 else ""
            keep = f"base keeps {100 * j / j0:3.0f}%" if j0 else ""
            print(f"    k={k:<2} sentences/block             {fmt(j, c)}   {keep}, {drop}")

        print("  -- word-block shuffle (keep q consecutive words intact) --")
        for q in (5, 10, 20, 40):
            variant = shuffle_word_blocks(text, block_size=q, seed=42)
            j, c = top_scores(detector, variant)
            drop = f"ours {100 * (1 - c / c0):4.0f}% down" if c0 else ""
            keep = f"base keeps {100 * j / j0:3.0f}%" if j0 else ""
            print(f"    q={q:<2} words/block                 {fmt(j, c)}   {keep}, {drop}")


if __name__ == "__main__":
    main()
