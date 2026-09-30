"""Document-document similarity computed on top of an already merged index.

Similarities are no longer produced while merging the index; instead they are
computed afterwards from the inverted index (``postings.dat``) and/or the forward
index (``forward_fingerprints.npy``), so that several methods can be run and
compared on the same index.  Each method writes ``similarities_<method>.npy`` in
the index directory: a structured array with uint32 fields ``doc_i < doc_j``
(internal document IDs) and ``count`` (number of unique shared fingerprints),
sorted by ``(doc_i, doc_j)``.

Methods
-------
``inverted``
    Exhaustive.  Streams the inverted index once; every fingerprint adds one count
    to each pair of documents in its posting list.
``forward_topk``
    For every document, reads its fingerprints from the forward index, queries the
    inverted index with them and keeps the ``top_k`` most similar other documents
    sharing at least ``min_similarity`` fingerprints.  The stored pairs are the
    union of those per-document neighbour lists.
"""

from __future__ import annotations

import argparse
import heapq
import os
from collections.abc import Callable, Iterator
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

from .indexing import DiskBasedIndex, similarity_file_name

SIMILARITY_DTYPE = np.dtype(
    [("doc_i", np.uint32), ("doc_j", np.uint32), ("count", np.uint32)]
)
_CHUNK_DTYPE = np.dtype([("key", np.uint64), ("count", np.uint32)])
_LOW_32 = np.uint64(0xFFFFFFFF)
_SHIFT = np.uint64(32)


class _PairAccumulator:
    """Aggregate ``(pair_key, count)`` items into ``similarities_<method>.npy`` in bounded memory.

    ``pair_key = (doc_i << 32) | doc_j``.  Items are buffered, then aggregated
    (``reduce="sum"`` or ``"max"``) into sorted temporary chunks, which are finally
    heap-merged (optionally through bounded-fan-in compaction passes).  Pairs below
    ``min_similarity`` are dropped only after complete aggregation.
    """

    def __init__(
        self,
        output_path: str,
        reduce: str = "sum",
        max_pairs_in_memory: int = 20_000_000,
        min_similarity: int = 1,
        max_chunks_per_merge: Optional[int] = None,
    ):
        if reduce not in {"sum", "max"}:
            raise ValueError("reduce must be 'sum' or 'max'")
        if max_pairs_in_memory <= 0:
            raise ValueError("max_pairs_in_memory must be positive")
        if min_similarity < 1:
            raise ValueError("min_similarity must be at least 1")
        if max_chunks_per_merge is not None and max_chunks_per_merge < 2:
            raise ValueError("max_chunks_per_merge must be at least 2")
        self.output_path = output_path
        self.max_pairs_in_memory = max_pairs_in_memory
        self.min_similarity = min_similarity
        self.max_chunks_per_merge = max_chunks_per_merge
        self._np_reduce = np.add if reduce == "sum" else np.maximum
        self._py_reduce: Callable[[int, int], int] = (
            (lambda a, b: a + b) if reduce == "sum" else max
        )
        self._keys: List[np.ndarray] = []
        self._counts: List[np.ndarray] = []
        self._buffered = 0
        self.chunk_paths: List[str] = []
        self._tmp_prefix = os.path.join(
            os.path.dirname(output_path) or ".",
            "." + os.path.splitext(os.path.basename(output_path))[0],
        )

    def add(self, keys: np.ndarray, counts: np.ndarray) -> None:
        if len(keys) == 0:
            return
        self._keys.append(np.asarray(keys, dtype=np.uint64))
        self._counts.append(np.asarray(counts, dtype=np.uint32))
        self._buffered += len(keys)
        if self._buffered >= self.max_pairs_in_memory:
            self._flush_chunk()

    def _aggregate(
        self, keys: np.ndarray, counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(keys, kind="stable")
        keys, counts = keys[order], counts[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        return keys[starts], self._np_reduce.reduceat(counts, starts)

    def _flush_chunk(self) -> None:
        if not self._keys:
            return
        keys, counts = self._aggregate(
            np.concatenate(self._keys), np.concatenate(self._counts)
        )
        self._keys, self._counts, self._buffered = [], [], 0
        chunk = np.empty(len(keys), dtype=_CHUNK_DTYPE)
        chunk["key"], chunk["count"] = keys, counts
        path = f"{self._tmp_prefix}-chunk-{len(self.chunk_paths):06d}.npy"
        np.save(path, chunk)
        self.chunk_paths.append(path)

    def finish(self) -> int:
        """Write the output file, remove temporary chunks and return the number of pairs."""
        self._flush_chunk()
        paths = self.chunk_paths
        if self.max_chunks_per_merge is not None:
            paths = self._compact_chunks(paths)

        if len(paths) <= 1:
            chunk = np.load(paths[0]) if paths else np.empty(0, dtype=_CHUNK_DTYPE)
            chunk = chunk[chunk["count"] >= self.min_similarity]
            output = np.empty(len(chunk), dtype=SIMILARITY_DTYPE)
            output["doc_i"] = chunk["key"] >> _SHIFT
            output["doc_j"] = chunk["key"] & _LOW_32
            output["count"] = chunk["count"]
            np.save(self.output_path, output)
            n_pairs = len(output)
        else:
            n_pairs = sum(
                1 for _, c in self._merged_items(paths) if c >= self.min_similarity
            )
            output = np.lib.format.open_memmap(
                self.output_path, mode="w+", dtype=SIMILARITY_DTYPE, shape=(n_pairs,)
            )
            index = 0
            for key, count in self._merged_items(paths):
                if count >= self.min_similarity:
                    output[index] = (key >> 32, key & 0xFFFFFFFF, count)
                    index += 1
            output.flush()
            del output

        for path in paths:
            os.remove(path)
        print(f"Saved {n_pairs:,} document similarity pairs to {self.output_path}.")
        return n_pairs

    def _compact_chunks(self, paths: List[str]) -> List[str]:
        """Reduce chunk count through bounded-fan-in merge passes."""
        fan_in = self.max_chunks_per_merge
        pass_index = 0
        while len(paths) > fan_in:
            compacted = []
            for group_index, start in enumerate(range(0, len(paths), fan_in)):
                group = paths[start : start + fan_in]
                if len(group) == 1:
                    compacted.append(group[0])
                    continue
                output_path = (
                    f"{self._tmp_prefix}-merge-{pass_index:03d}-{group_index:06d}.npy"
                )
                n_items = sum(1 for _ in self._merged_items(group))
                merged = np.lib.format.open_memmap(
                    output_path, mode="w+", dtype=_CHUNK_DTYPE, shape=(n_items,)
                )
                for index, item in enumerate(self._merged_items(group)):
                    merged[index] = item
                merged.flush()
                del merged
                for path in group:
                    os.remove(path)
                compacted.append(output_path)
            paths = compacted
            pass_index += 1
        return paths

    def _merged_items(self, paths: List[str]) -> Iterator[Tuple[int, int]]:
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        heap = [
            (int(array["key"][0]), chunk_index, 0)
            for chunk_index, array in enumerate(arrays)
            if len(array)
        ]
        heapq.heapify(heap)

        def _advance(chunk_index: int, position: int) -> None:
            if position + 1 < len(arrays[chunk_index]):
                next_key = int(arrays[chunk_index]["key"][position + 1])
                heapq.heappush(heap, (next_key, chunk_index, position + 1))

        while heap:
            key, chunk_index, position = heapq.heappop(heap)
            count = int(arrays[chunk_index]["count"][position])
            _advance(chunk_index, position)
            while heap and heap[0][0] == key:
                _, other_chunk, other_position = heapq.heappop(heap)
                count = self._py_reduce(
                    count, int(arrays[other_chunk]["count"][other_position])
                )
                _advance(other_chunk, other_position)
            yield key, count


def compute_similarities_inverted(
    index_dir: str,
    min_similarity: int = 1,
    max_posting_length: Optional[int] = None,
    max_pairs_in_memory: int = 20_000_000,
    max_chunks_per_merge: Optional[int] = None,
    batch_entries: int = 10_000_000,
    verbose: bool = True,
) -> str:
    """Exhaustive pair counts from one sequential pass over the inverted index.

    Args:
        index_dir: Merged index directory.
        min_similarity: Only store pairs sharing at least this many fingerprints.
        max_posting_length: If set, ignore fingerprints occurring in more unique
            documents than this (bounds quadratic pair generation; counts become
            approximate).
        max_pairs_in_memory: Pair items buffered before a sorted chunk is flushed.
        max_chunks_per_merge: Optional fan-in bound for merging temporary chunks.
        batch_entries: Posting entries read per batch from ``postings.dat``.
        verbose: Show a progress bar.

    Returns:
        Path to ``similarities_inverted.npy``.
    """
    if max_posting_length is not None and max_posting_length < 2:
        raise ValueError("max_posting_length must be at least 2")
    index = DiskBasedIndex(index_dir)
    output_path = os.path.join(index_dir, similarity_file_name("inverted"))
    accumulator = _PairAccumulator(
        output_path,
        reduce="sum",
        max_pairs_in_memory=max_pairs_in_memory,
        min_similarity=min_similarity,
        max_chunks_per_merge=max_chunks_per_merge,
    )
    triu_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    progress = tqdm.tqdm(
        total=len(index.fingerprints),
        desc="Inverted-index similarities",
        disable=not verbose,
    )
    for batch_fps, group, docs in index.iter_unique_posting_batches(batch_entries):
        sizes = np.bincount(group, minlength=len(batch_fps))
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        eligible = sizes >= 2
        if max_posting_length is not None:
            eligible &= sizes <= max_posting_length

        batch_keys = []
        wide_docs = docs.astype(np.uint64)
        for g in np.flatnonzero(eligible):
            size = int(sizes[g])
            if size not in triu_cache:
                triu_cache[size] = np.triu_indices(size, 1)
            first, second = triu_cache[size]
            # Documents within a fingerprint are sorted, so doc[first] < doc[second].
            group_docs = wide_docs[starts[g] : starts[g] + size]
            batch_keys.append((group_docs[first] << _SHIFT) | group_docs[second])
        if batch_keys:
            keys = np.concatenate(batch_keys)
            accumulator.add(keys, np.ones(len(keys), dtype=np.uint32))
        progress.update(len(batch_fps))
    progress.close()

    accumulator.finish()
    return output_path


def compute_similarities_forward_topk(
    index_dir: str,
    top_k: int = 50,
    min_similarity: int = 5,
    max_pairs_in_memory: int = 20_000_000,
    max_chunks_per_merge: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """Top-k neighbours per document, found by querying each document's forward fingerprints.

    For every internal document ``d``, :meth:`DiskBasedIndex.get_document_candidates`
    returns the ``top_k`` other documents sharing the most unique fingerprints with
    ``d`` (at least ``min_similarity``; ties broken by ascending doc ID).  The
    stored pairs are the union over all documents, so a pair is kept if either
    document is among the other's top-k.  Counts are exact.

    Returns:
        Path to ``similarities_forward_topk.npy``.
    """
    index = DiskBasedIndex(index_dir)
    n_docs = len(index.to_external_doc_id)
    output_path = os.path.join(index_dir, similarity_file_name("forward_topk"))
    accumulator = _PairAccumulator(
        output_path,
        reduce="max",
        max_pairs_in_memory=max_pairs_in_memory,
        min_similarity=min_similarity,
        max_chunks_per_merge=max_chunks_per_merge,
    )

    for doc in tqdm.trange(
        n_docs, desc="Forward top-k similarities", disable=not verbose
    ):
        candidates, counts = index.get_document_candidates(
            doc, top_k=top_k, min_fingerprints=min_similarity
        )
        if len(candidates) == 0:
            continue
        candidates = candidates.astype(np.uint64)
        this_doc = np.uint64(doc)
        low = np.minimum(candidates, this_doc)
        high = np.maximum(candidates, this_doc)
        accumulator.add((low << _SHIFT) | high, counts)

    accumulator.finish()
    return output_path


METHODS: Dict[str, Callable[..., str]] = {
    "inverted": compute_similarities_inverted,
    "forward_topk": compute_similarities_forward_topk,
}


def compute_similarities(index_dir: str, method: str = "inverted", **kwargs) -> str:
    """Compute similarities with ``method`` (see :data:`METHODS`) and return the output path."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    return METHODS[method](index_dir, **kwargs)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def similarities_to_dict(similarities: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Map ``(doc_i, doc_j) -> count`` for a similarity array."""
    return dict(
        zip(
            zip(similarities["doc_i"].tolist(), similarities["doc_j"].tolist()),
            similarities["count"].tolist(),
        )
    )


def top_k_per_document(
    similarities: np.ndarray, top_k: int, min_similarity: int = 1
) -> Dict[Tuple[int, int], int]:
    """Truncate exhaustive similarities to the union of per-document top-k neighbours.

    Uses the same selection rule as ``forward_topk``: neighbours with at least
    ``min_similarity`` shared fingerprints, ordered by descending count, then
    ascending doc ID.
    """
    doc_i = similarities["doc_i"].astype(np.int64)
    doc_j = similarities["doc_j"].astype(np.int64)
    count = similarities["count"].astype(np.int64)
    keep = count >= min_similarity
    source = np.concatenate([doc_i[keep], doc_j[keep]])
    target = np.concatenate([doc_j[keep], doc_i[keep]])
    weight = np.concatenate([count[keep], count[keep]])
    if len(source) == 0:
        return {}

    order = np.lexsort((target, -weight, source))
    source, target, weight = source[order], target[order], weight[order]
    starts = np.flatnonzero(np.r_[True, source[1:] != source[:-1]])
    rank = np.arange(len(source)) - np.repeat(
        starts, np.diff(np.r_[starts, len(source)])
    )
    selected = rank < top_k
    low = np.minimum(source, target)[selected].tolist()
    high = np.maximum(source, target)[selected].tolist()
    return dict(zip(zip(low, high), weight[selected].tolist()))


def compare_with_exhaustive(
    exhaustive: np.ndarray, top_k_result: np.ndarray, top_k: int, min_similarity: int
) -> Dict[str, object]:
    """Compare a top-k result against exhaustive similarities truncated to top-k.

    Returns a report with the number of pairs in each, pairs missing from / extra
    in the top-k result, and pairs whose count differs from the exhaustive count.
    ``identical`` is true when the truncated exhaustive result equals the top-k
    result exactly.
    """
    exhaustive_all = similarities_to_dict(exhaustive)
    expected = top_k_per_document(exhaustive, top_k, min_similarity)
    actual = similarities_to_dict(top_k_result)
    count_mismatches = {
        pair: (exhaustive_all.get(pair, 0), count)
        for pair, count in actual.items()
        if exhaustive_all.get(pair, 0) != count
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    return {
        "n_exhaustive_pairs": len(exhaustive_all),
        "n_expected_top_k_pairs": len(expected),
        "n_top_k_pairs": len(actual),
        "missing": missing,
        "extra": extra,
        "count_mismatches": count_mismatches,
        "identical": not missing and not extra and not count_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute document similarities for a merged index."
    )
    parser.add_argument("index_dir", help="Merged index directory.")
    parser.add_argument("--method", choices=sorted(METHODS), default="inverted")
    parser.add_argument(
        "--min-similarity",
        type=int,
        default=None,
        help="Minimum shared fingerprints per stored pair "
        "(default: 1 for inverted, 5 for forward_topk).",
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="forward_topk: neighbours per document."
    )
    parser.add_argument(
        "--max-posting-length",
        type=int,
        default=None,
        help="inverted: skip fingerprints in more documents than this (approximate).",
    )
    parser.add_argument("--max-pairs-in-memory", type=int, default=20_000_000)
    parser.add_argument("--max-chunks-per-merge", type=int, default=None)
    args = parser.parse_args()

    kwargs = {
        "max_pairs_in_memory": args.max_pairs_in_memory,
        "max_chunks_per_merge": args.max_chunks_per_merge,
    }
    if args.min_similarity is not None:
        kwargs["min_similarity"] = args.min_similarity
    if args.method == "inverted":
        kwargs["max_posting_length"] = args.max_posting_length
    else:
        kwargs["top_k"] = args.top_k
    compute_similarities(args.index_dir, args.method, **kwargs)


if __name__ == "__main__":
    main()
