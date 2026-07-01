"""
Standalone demo for :class:`TextContainmentDetector`.

It loads a disk-based winnowing index, takes a few *positive* examples (texts that
are contained in an indexed document) and a few *negative* examples (texts that are
not), and runs every scoring strategy on each one:

- Jaccard, count score
- Jaccard, rarity score
- Clustering with the ``rectangle`` method
- Clustering with the ``euclidean`` single-linkage metric
- Clustering with the ``cityblock`` single-linkage metric
- Clustering with the ``chebyshev`` single-linkage metric

For every detector call it prints the wall-clock time taken and the best (highest
scoring) match, so you can compare both the verdicts and the cost of each strategy.

Run it from the repo root with the project virtual-env active:

    python -m detector_demo

TODO: add the functionallity to get the text from the mathing document,
 run oracle and hightlight the matching text in the document. This will
 be useful for debugging and for understanding why a match was made.
"""

from __future__ import annotations

import time
from typing import Callable, Dict

from detector import TextContainmentDetector
from detector_utils import load_examples_from_jsonl

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
INDEX_DIR = "/cluster/work/projects/nn9850k/wiki/index(4,6)"
POSITIVE_FILE = "/cluster/work/projects/nn9850k/wiki/positive_samples_wiki_sw1000.jsonl"
NEGATIVE_FILE = "/cluster/work/projects/nn9850k/wiki/negative_samples_wiki_sw1000.jsonl"
N_EXAMPLES = 3  # how many positive and negative examples to run


def _timed_call(fn: Callable[[], Dict[str, float]]) -> tuple[Dict[str, float], float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _best_match(scores: Dict[str, float]) -> str:
    """Format the highest-scoring match as ``'doc_id (score)'`` or ``'<no match>'``."""
    if not scores:
        return "<no match>"
    doc_id, score = max(scores.items(), key=lambda kv: kv[1])
    return f"{doc_id} ({score:.3f})"


def _print_top_n(scores: Dict[str, float], n: int = 5) -> None:
    """Print the ``n`` highest-scoring matches, one per line."""
    if not scores:
        print("    <no match>")
        return
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        print(f"    {rank}. doc {doc_id:<10} score {score:.3f}")


def run_demo() -> None:
    print(f"Loading index from {INDEX_DIR} ...")
    load_start = time.perf_counter()
    detector = TextContainmentDetector(INDEX_DIR, top_k=10)
    print(
        f"Index loaded in {time.perf_counter() - load_start:.2f}s "
        f"({detector.num_documents} documents)\n"
    )

    # Each strategy is a (label, callable) pair. The callable takes the query text
    # and returns a Dict[str, float] of external_doc_id -> score.
    strategies: list[tuple[str, Callable[[str], Dict[str, float]]]] = [
        ("jaccard (count)",    lambda t: detector.find_matches_jaccard(t, score="count")),
        ("jaccard (rarity)",   lambda t: detector.find_matches_jaccard(t, score="rarity")),
        ("rectangle (count)",  lambda t: detector.find_matches_clustering(t, {"method": "rectangle", "score": "count"})),
        ("rectangle (rarity)", lambda t: detector.find_matches_clustering(t, {"method": "rectangle", "score": "rarity"})),
        ("euclidean (count)",  lambda t: detector.find_matches_clustering(t, {"method": "euclidean", "score": "count"})),
        ("euclidean (rarity)", lambda t: detector.find_matches_clustering(t, {"method": "euclidean", "score": "rarity"})),
        ("cityblock (count)",  lambda t: detector.find_matches_clustering(t, {"method": "cityblock", "score": "count"})),
        ("cityblock (rarity)", lambda t: detector.find_matches_clustering(t, {"method": "cityblock", "score": "rarity"})),
    ]

    datasets = [
        ("POSITIVE (expected: match)", POSITIVE_FILE),
        ("NEGATIVE (expected: no match)", NEGATIVE_FILE),
    ]

    for label, filepath in datasets:
        print("=" * 78)
        print(label)
        print("=" * 78)
        examples = load_examples_from_jsonl(filepath, verbose=False)[:N_EXAMPLES]

        for i, example in enumerate(examples):
            text = example["text"]
            preview = text[:60].replace("\n", " ")
            print(f"\nExample {i + 1}: {preview!r}...")
            print(f"  {'strategy':<18}{'time (s)':>10}   best match")
            print(f"  {'-' * 18}{'-' * 10:>10}   {'-' * 30}")
            for name, fn in strategies:
                scores, elapsed = _timed_call(lambda t=text, f=fn: f(t))
                print(f"  {name:<18}{elapsed:>10.4f}   {_best_match(scores)}")
        print()

    # ------------------------------------------------------------------ #
    # Top-5 rectangle (count) results for one positive and one negative  #
    # example, to inspect the full ranking rather than just the best hit #
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("TOP 5 rectangle (count) matches")
    print("=" * 78)
    for label, filepath in datasets:
        text = load_examples_from_jsonl(filepath, verbose=False)[0]["text"]
        preview = text[:60].replace("\n", " ")
        scores = detector.find_matches_clustering(text, {"method": "rectangle"}, score="count")
        print(f"\n{label}")
        print(f"  query: {preview!r}...")
        _print_top_n(scores, n=5)
    print()


if __name__ == "__main__":
    run_demo()


"""
Loading index from /cluster/work/projects/nn9850k/wiki/index(4,6) ...
Index loaded in 0.12s (380855 documents)

==============================================================================
POSITIVE (expected: match)
==============================================================================

Example 1: 'NCERT Solutions for Class 8 Chapter 15 Introduction to Graph'...
  strategy            time (s)   best match
  ----------------------------   ------------------------------
  jaccard (count)       8.9878   634114 (38.000)
  jaccard (rarity)      0.0966   634114 (37.990)
  rectangle (count)     0.7459   634114 (38.000)
  rectangle (rarity)    0.1627   634114 (38.000)
  euclidean (count)     0.2382   634114 (38.000)
  euclidean (rarity)    0.1586   634114 (38.000)
  cityblock (count)     0.1798   634114 (38.000)
  cityblock (rarity)    0.1952   634114 (38.000)

Example 2: 'Friday, May 20, 2011 So, this has been knocking around in my'...
  strategy            time (s)   best match
  ----------------------------   ------------------------------
  jaccard (count)       0.8271   421395 (47.000)
  jaccard (rarity)      0.0941   421395 (46.998)
  rectangle (count)     0.1365   421395 (47.000)
  rectangle (rarity)    0.1508   421395 (47.000)
  euclidean (count)     0.2121   421395 (47.000)
  euclidean (rarity)    0.1332   421395 (47.000)
  cityblock (count)     0.1424   421395 (47.000)
  cityblock (rarity)    0.1341   421395 (47.000)

Example 3: 'Milton’s Local bridges the gap between local chefs and farme'...
  strategy            time (s)   best match
  ----------------------------   ------------------------------
  jaccard (count)       0.4440   233701 (32.000)
  jaccard (rarity)      0.1362   233701 (31.980)
  rectangle (count)     0.3437   233701 (32.000)
  rectangle (rarity)    0.3505   233701 (32.000)
  euclidean (count)     0.3227   233701 (32.000)
  euclidean (rarity)    0.2978   233701 (32.000)
  cityblock (count)     0.3372   233701 (32.000)
  cityblock (rarity)    0.3092   233701 (32.000)

==============================================================================
NEGATIVE (expected: no match)
==============================================================================

Example 1: 'SSK was the United States Navy hull classification symbol fo'...
  strategy            time (s)   best match
  ----------------------------   ------------------------------
  jaccard (count)       0.4562   476885 (97.000)
  jaccard (rarity)      0.1164   476885 (96.996)
  rectangle (count)     0.1455   476885 (31.000)
  rectangle (rarity)    0.1335   476885 (31.000)
  euclidean (count)     0.1750   476885 (29.000)
  euclidean (rarity)    0.1395   476885 (29.000)
  cityblock (count)     0.1577   476885 (29.000)
  cityblock (rarity)    0.1320   476885 (29.000)

Example 2: 'Timothy Eugene Scott (born September 19, 1965) is an America'...
  strategy            time (s)   best match
  ----------------------------   ------------------------------
  jaccard (count)       1.2002   751774 (398.000)
  jaccard (rarity)      0.4079   751774 (397.976)
  rectangle (count)     0.5516   751774 (41.000)
  rectangle (rarity)    0.5574   751774 (41.000)
  euclidean (count)     0.6677   751774 (41.000)
  euclidean (rarity)    0.5937   751774 (41.000)
  cityblock (count)     0.5476   751774 (41.000)
  cityblock (rarity)    0.5646   751774 (41.000)

Example 3: 'The Robbers (, ) is the first drama by German playwright Fri'...
  strategy            time (s)   best match
  ----------------------------   ------------------------------
  jaccard (count)       1.1338   59625 (396.000)
  jaccard (rarity)      0.3945   59625 (395.973)
  rectangle (count)     0.4240   59625 (35.000)
  rectangle (rarity)    0.4007   59625 (35.000)
  euclidean (count)     0.4214   59625 (35.000)
  euclidean (rarity)    0.4878   59625 (35.000)
  cityblock (count)     0.4653   59625 (35.000)
  cityblock (rarity)    0.4702   59625 (35.000)

==============================================================================
TOP 5 rectangle (count) matches
==============================================================================

POSITIVE (expected: match)
  query: 'NCERT Solutions for Class 8 Chapter 15 Introduction to Graph'...
    1. doc 634114     score 38.000
    2. doc 616064     score 0.000
    3. doc 740159     score 0.000
    4. doc 408651     score 0.000

NEGATIVE (expected: no match)
  query: 'SSK was the United States Navy hull classification symbol fo'...
    1. doc 476885     score 31.000
"""