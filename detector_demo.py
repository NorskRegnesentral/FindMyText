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
    return f"{doc_id} ({score:.4f})"


def _print_top_n(scores: Dict[str, float], n: int = 5) -> None:
    """Print the ``n`` highest-scoring matches, one per line."""
    if not scores:
        print("    <no match>")
        return
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        print(f"    {rank}. doc {doc_id:<10} score {score:.4f}")


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
        ("jaccard (count)",    lambda t: detector.find_matches_jaccard(text=t, score="count")),
        ("jaccard (rarity)",   lambda t: detector.find_matches_jaccard(text=t, score="rarity")),
        ("rectangle (count)",  lambda t: detector.find_matches_clustering(text=t, clustering_params={"method": "rectangle"}, score="count")),
        ("rectangle (rarity)", lambda t: detector.find_matches_clustering(text=t, clustering_params={"method": "rectangle"}, score="rarity")),
        ("euclidean (count)",  lambda t: detector.find_matches_clustering(text=t, clustering_params={"method": "euclidean"}, score="count")),
        ("euclidean (rarity)", lambda t: detector.find_matches_clustering(text=t, clustering_params={"method": "euclidean"}, score="rarity")),
        ("cityblock (count)",  lambda t: detector.find_matches_clustering(text=t, clustering_params={"method": "cityblock"}, score="count")),
        ("cityblock (rarity)", lambda t: detector.find_matches_clustering(text=t, clustering_params={"method": "cityblock"}, score="rarity")),
    ]

    datasets = [
        ("POSITIVE (expected: match)", POSITIVE_FILE),
        ("NEGATIVE (expected: no match)", NEGATIVE_FILE),
    ]

    label, filepath = datasets[0]
    for label, filepath in datasets:
        print("=" * 78)
        print(label)
        print("=" * 78)
        examples = load_examples_from_jsonl(filepath, verbose=False)[:N_EXAMPLES]

        i, example = list(enumerate(examples))[0]
        for i, example in enumerate(examples):
            text = example["text"]
            preview = text[:60].replace("\n", " ")
            print(f"\nExample {i + 1}: {preview!r}...")
            print(f"  {'strategy':<18}{'time (s)':>10}   ID of best match (score)")
            print(f"  {'-' * 18}{'-' * 10:>10}   {'-' * 30}")
            name, fn = strategies[3] # rectangle (rarity) is the default for the demo   
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
Index loaded in 0.01s (380855 documents)

==============================================================================
POSITIVE (expected: match)
==============================================================================

Example 1: 'NCERT Solutions for Class 8 Chapter 15 Introduction to Graph'...
  strategy            time (s)   ID of best match (score)
  ----------------------------   ------------------------------
  jaccard (count)       0.0819   634114 (38.0000)
  jaccard (rarity)      0.0651   634114 (37.9897)
  rectangle (count)     0.1521   634114 (38.0000)
  rectangle (rarity)    0.1644   634114 (37.9897)
  euclidean (count)     0.1670   634114 (38.0000)
  euclidean (rarity)    0.1648   634114 (37.9897)
  cityblock (count)     0.1529   634114 (38.0000)
  cityblock (rarity)    0.2248   634114 (37.9897)

Example 2: 'Friday, May 20, 2011 So, this has been knocking around in my'...
  strategy            time (s)   ID of best match (score)
  ----------------------------   ------------------------------
  jaccard (count)       0.1357   421395 (47.0000)
  jaccard (rarity)      0.1090   421395 (46.9985)
  rectangle (count)     0.1427   421395 (47.0000)
  rectangle (rarity)    0.1253   421395 (46.9985)
  euclidean (count)     0.1445   421395 (47.0000)
  euclidean (rarity)    0.1437   421395 (46.9985)
  cityblock (count)     0.1287   421395 (47.0000)
  cityblock (rarity)    0.1305   421395 (46.9985)

Example 3: 'Milton’s Local bridges the gap between local chefs and farme'...
  strategy            time (s)   ID of best match (score)
  ----------------------------   ------------------------------
  jaccard (count)       0.1083   233701 (32.0000)
  jaccard (rarity)      0.0959   233701 (31.9796)
  rectangle (count)     0.2906   233701 (32.0000)
  rectangle (rarity)    0.3071   233701 (31.9796)
  euclidean (count)     0.3478   233701 (32.0000)
  euclidean (rarity)    0.2660   233701 (31.9796)
  cityblock (count)     0.3174   233701 (32.0000)
  cityblock (rarity)    0.3462   233701 (31.9796)

==============================================================================
NEGATIVE (expected: no match)
==============================================================================

Example 1: 'SSK was the United States Navy hull classification symbol fo'...
  strategy            time (s)   ID of best match (score)
  ----------------------------   ------------------------------
  jaccard (count)       0.1414   476885 (97.0000)
  jaccard (rarity)      0.1408   476885 (96.9963)
  rectangle (count)     0.1653   476885 (31.0000)
  rectangle (rarity)    0.1616   476885 (30.9987)
  euclidean (count)     0.1707   476885 (29.0000)
  euclidean (rarity)    0.1751   476885 (28.9996)
  cityblock (count)     0.1786   476885 (29.0000)
  cityblock (rarity)    0.1979   476885 (28.9996)

Example 2: 'Timothy Eugene Scott (born September 19, 1965) is an America'...
  strategy            time (s)   ID of best match (score)
  ----------------------------   ------------------------------
  jaccard (count)       0.5235   751774 (398.0000)
  jaccard (rarity)      0.5029   751774 (397.9764)
  rectangle (count)     0.6460   751774 (41.0000)
  rectangle (rarity)    0.6512   751774 (40.9969)
  euclidean (count)     0.6498   751774 (41.0000)
  euclidean (rarity)    0.6183   751774 (40.9969)
  cityblock (count)     0.6322   751774 (41.0000)
  cityblock (rarity)    0.6170   751774 (40.9969)

Example 3: 'The Robbers (, ) is the first drama by German playwright Fri'...
  strategy            time (s)   ID of best match (score)
  ----------------------------   ------------------------------
  jaccard (count)       0.4930   59625 (396.0000)
  jaccard (rarity)      0.4850   59625 (395.9728)
  rectangle (count)     0.5275   59625 (35.0000)
  rectangle (rarity)    0.5078   59625 (34.9998)
  euclidean (count)     0.5246   59625 (35.0000)
  euclidean (rarity)    0.5517   59625 (34.9998)
  cityblock (count)     0.5010   59625 (35.0000)
  cityblock (rarity)    0.5102   59625 (34.9998)

==============================================================================
TOP 5 rectangle (count) matches
==============================================================================

POSITIVE (expected: match)
  query: 'NCERT Solutions for Class 8 Chapter 15 Introduction to Graph'...
    1. doc 634114     score 38.0000
    2. doc 616064     score 0.0000
    3. doc 740159     score 0.0000
    4. doc 408651     score 0.0000

NEGATIVE (expected: no match)
  query: 'SSK was the United States Navy hull classification symbol fo'...
    1. doc 476885     score 31.0000
"""