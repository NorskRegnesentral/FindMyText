"""Single-worker, bounded-subset prototypes for absolute fingerprint overlap.

Source indexes remain read-only. Frequency caps affect discovery, never the
full-forward overlap verifier. This module is not a full-corpus join backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from .overlap_inverted import OverlapInvertedIndex, _source_identity
from .similarity import SIMILARITY_DTYPE

PAIR_DTYPE = np.dtype([("doc_i", np.uint32), ("doc_j", np.uint32)])
_FP_DOC = np.dtype([("fingerprint", np.uint64), ("doc", np.uint32)])
_METRICS = (
    "candidate_visits", "duplicate_discoveries", "verification_calls",
    "length_rejections", "hash_comparisons", "early_rejections",
    "threshold_early_acceptances", "exact_terminations",
)


@dataclass
class JoinSubset:
    document_ids: np.ndarray
    forward_offsets: np.ndarray
    forward_fingerprints: np.ndarray
    fingerprints: np.ndarray
    posting_offsets: np.ndarray
    posting_docs: np.ndarray
    source_df: np.ndarray
    eligible_lengths: np.ndarray
    document_fp_offsets: np.ndarray
    document_fp_ranks: np.ndarray
    memory_budget_bytes: int
    source_identity: dict | None = None

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))


def _require_memory(required: int, budget: int) -> None:
    if required > budget:
        raise MemoryError(
            f"Prototype working arrays require up to {required:,} bytes; "
            f"budget is {budget:,}. Reduce the document subset or raise the budget."
        )


@njit(cache=True, nogil=True)
def _gather_forward(ids, offsets, values, output_offsets, output):
    cursor = 0
    output_offsets[0] = 0
    for local, doc in enumerate(ids):
        first = np.intp(offsets[doc])
        last = np.intp(offsets[doc + 1])
        output[cursor:cursor + last - first] = values[first:last]
        cursor += last - first
        output_offsets[local + 1] = cursor


@njit(cache=True, nogil=True)
def _subset_sizes(offsets, docs, mapping, sizes):
    for rank in range(len(sizes)):
        count = 0
        for entry in range(np.intp(offsets[rank]), np.intp(offsets[rank + 1])):
            if mapping[docs[entry]] >= 0:
                count += 1
        sizes[rank] = count


@njit(cache=True, nogil=True)
def _subset_postings(ranks, offsets, docs, mapping, output_offsets, output):
    for local, rank in enumerate(ranks):
        cursor = np.intp(output_offsets[local])
        for entry in range(np.intp(offsets[rank]), np.intp(offsets[rank + 1])):
            doc = mapping[docs[entry]]
            if doc >= 0:
                output[cursor] = doc
                cursor += 1
        if cursor != output_offsets[local + 1]:
            raise ValueError("Subset posting cardinality changed during preparation")


@njit(cache=True, nogil=True)
def _document_ranks(offsets, docs, cursors, output):
    for rank in range(len(offsets) - 1):
        for entry in range(np.intp(offsets[rank]), np.intp(offsets[rank + 1])):
            doc = docs[entry]
            output[cursors[doc]] = rank
            cursors[doc] += 1


def prepare_subset(
    index_dir: str | Path,
    overlap_dir: str | Path,
    document_ids: np.ndarray,
    memory_budget_bytes: int,
) -> JoinSubset:
    """Retain global rarity/df while restricting postings to selected source IDs."""
    ids = np.asarray(document_ids)
    overlap = OverlapInvertedIndex(overlap_dir, index_dir)
    n_source_docs = overlap.meta["document_count"]
    if (
        ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
        or np.any(ids < 0) or np.any(ids >= n_source_docs)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Document subset must contain unique, in-range integer source IDs")
    ids = np.sort(ids.astype(np.uint32))
    _require_memory(
        n_source_docs * 8 + len(overlap.fingerprints) * 16,
        memory_budget_bytes,
    )
    forward_offsets = np.load(Path(index_dir) / "forward_offsets.npy", mmap_mode="r")
    forward_values = np.load(Path(index_dir) / "forward_fingerprints.npy", mmap_mode="r")
    total = int(np.sum(forward_offsets[ids + 1] - forward_offsets[ids]))
    _require_memory(
        n_source_docs * 8 + len(overlap.fingerprints) * 16 + total * 64,
        memory_budget_bytes,
    )
    local_offsets = np.zeros(len(ids) + 1, dtype=np.uint64)
    local_values = np.empty(total, dtype=np.uint64)
    _gather_forward(ids, forward_offsets, forward_values, local_offsets, local_values)
    mapping = np.full(n_source_docs, -1, dtype=np.int64)
    mapping[ids] = np.arange(len(ids))
    sizes = np.empty(len(overlap.fingerprints), dtype=np.uint32)
    _subset_sizes(overlap.offsets, overlap.doc_ids, mapping, sizes)
    ranks = np.flatnonzero(sizes >= 2)
    if len(ranks) > np.iinfo(np.uint32).max:
        raise ValueError("Prototype fingerprint ranks exceed uint32; reduce the subset")
    offsets = np.zeros(len(ranks) + 1, dtype=np.uint64)
    np.cumsum(sizes[ranks], out=offsets[1:])
    docs = np.empty(int(offsets[-1]), dtype=np.uint32)
    _subset_postings(ranks, overlap.offsets, overlap.doc_ids, mapping, offsets, docs)
    eligible = np.bincount(docs, minlength=len(ids)).astype(np.uint64)
    doc_offsets = np.zeros(len(ids) + 1, dtype=np.uint64)
    np.cumsum(eligible, out=doc_offsets[1:])
    doc_ranks = np.empty(len(docs), dtype=np.uint32)
    _document_ranks(offsets, docs, doc_offsets[:-1].copy(), doc_ranks)
    if _source_identity(Path(index_dir).resolve()) != overlap.meta["source"]:
        raise ValueError("Source index changed during subset preparation")
    return JoinSubset(
        ids, local_offsets, local_values, np.asarray(overlap.fingerprints[ranks]),
        offsets, docs, np.asarray(overlap.offsets[ranks + 1] - overlap.offsets[ranks]),
        eligible, doc_offsets, doc_ranks, memory_budget_bytes,
        source_identity=overlap.meta["source"],
    )


@njit(cache=True, nogil=True)
def _verify(values, offsets, a, b, k, exact):
    i = np.intp(offsets[a])
    j = np.intp(offsets[b])
    end_a = np.intp(offsets[a + 1])
    end_b = np.intp(offsets[b + 1])
    matches = comparisons = 0
    while i < end_a and j < end_b:
        if values[end_a - 1] < values[j] or values[end_b - 1] < values[i]:
            return matches >= k, matches, comparisons, 2
        if matches + min(end_a - i, end_b - j) < k:
            return False, matches, comparisons, 0
        comparisons += 1
        if values[i] == values[j]:
            matches += 1
            i += 1
            j += 1
            if not exact and matches >= k:
                return True, matches, comparisons, 1
        elif values[i] < values[j]:
            i += 1
        else:
            j += 1
    return matches >= k, matches, comparisons, 2


def verify_pair(a: np.ndarray, b: np.ndarray, k: int, output_mode: str = "exact_counts"):
    """Validate sorted-set inputs and return a boolean plus an optional exact count.

    A rejected pair has no advertised count: early rejection can be partial.
    """
    if k < 1 or output_mode not in ("threshold_only", "exact_counts"):
        raise ValueError("Use positive K and threshold_only or exact_counts")
    for values in (a, b):
        if (
            values.ndim != 1 or values.dtype != np.uint64
            or np.any(values[1:] <= values[:-1])
        ):
            raise ValueError("Pair inputs must be strictly sorted unique uint64 arrays")
    joined = np.concatenate((a, b))
    offsets = np.array([0, len(a), len(joined)], dtype=np.uint64)
    accepted, count, _, _ = _verify(joined, offsets, 0, 1, k, output_mode == "exact_counts")
    return bool(accepted), int(count) if accepted and output_mode == "exact_counts" else None


@njit(cache=True, nogil=True)
def _pair_slot(a, b, n):
    a = np.intp(a)
    b = np.intp(b)
    if b < a:
        a, b = b, a
    return a * (2 * n - a - 1) // 2 + b - a - 1


@njit(cache=True, nogil=True)
def _evaluate(a, b, eligible, values, offsets, k, exact, metrics):
    if eligible[a] < k or eligible[b] < k:
        metrics[3] += 1
        return False, 0
    metrics[2] += 1
    accepted, count, comparisons, reason = _verify(values, offsets, a, b, k, exact)
    metrics[4] += comparisons
    metrics[5 if reason == 0 else 6 if reason == 1 else 7] += 1
    return accepted, count


@njit(cache=True, nogil=True)
def _emit(output, used, a, b, count, ids):
    first, second = ids[a], ids[b]
    output[used]["doc_i"] = min(first, second)
    output[used]["doc_j"] = max(first, second)
    output[used]["count"] = count


@njit(cache=True, nogil=True)
def _fingerprint_batch(post_offsets, docs, stop, state, checked, eligible,
                       values, forward_offsets, ids, k, exact, metrics, output):
    rank, left, right = state
    visits = used = 0
    while rank < stop and visits < len(output):
        first = np.intp(post_offsets[rank])
        count = np.intp(post_offsets[rank + 1] - post_offsets[rank])
        if left >= count - 1:
            rank += 1
            left, right = 0, 1
            continue
        if right >= count:
            left += 1
            right = left + 1
            continue
        a, b = docs[first + left], docs[first + right]
        right += 1
        visits += 1
        metrics[0] += 1
        slot = _pair_slot(a, b, len(ids))
        if checked[slot]:
            metrics[1] += 1
            continue
        checked[slot] = True
        accepted, matches = _evaluate(
            a, b, eligible, values, forward_offsets, k, exact, metrics
        )
        if accepted:
            _emit(output, used, a, b, matches, ids)
            used += 1
    state[:] = rank, left, right
    return used


@njit(cache=True, nogil=True)
def _next_epoch(markers, epoch):
    if epoch == np.iinfo(np.uint32).max:
        markers[:] = 0
        return np.uint32(1)
    return np.uint32(epoch + 1)


@njit(cache=True, nogil=True)
def _document_batch(post_offsets, docs, stop, doc_offsets, doc_ranks, order,
                    join_rank, state, markers, eligible, values, forward_offsets,
                    ids, k, exact, metrics, output):
    probe, fp_cursor, doc_cursor, epoch = state
    visits = used = 0
    while probe < len(ids) and visits < len(output):
        a = order[probe]
        first_rank = np.intp(doc_offsets[a])
        last_rank = np.intp(doc_offsets[a + 1])
        if fp_cursor >= last_rank - first_rank:
            probe += 1
            fp_cursor = doc_cursor = 0
            epoch = _next_epoch(markers, epoch)
            continue
        rank = doc_ranks[first_rank + fp_cursor]
        if rank >= stop:
            fp_cursor = last_rank - first_rank
            continue
        first = np.intp(post_offsets[rank])
        count = np.intp(post_offsets[rank + 1] - post_offsets[rank])
        if doc_cursor >= count:
            fp_cursor += 1
            doc_cursor = 0
            continue
        b = docs[first + doc_cursor]
        doc_cursor += 1
        visits += 1
        metrics[0] += 1
        if join_rank[b] >= probe:
            continue
        if markers[b] == epoch:
            metrics[1] += 1
            continue
        markers[b] = epoch
        accepted, matches = _evaluate(
            a, b, eligible, values, forward_offsets, k, exact, metrics
        )
        if accepted:
            _emit(output, used, a, b, matches, ids)
            used += 1
    state[:] = probe, fp_cursor, doc_cursor, epoch
    return used


def run_join(
    data: JoinSubset,
    k: int,
    *,
    scheduler: str,
    output_mode: str = "exact_counts",
    candidate_df_cap: int | None = None,
    batch_size: int = 65_536,
) -> tuple[np.ndarray, dict]:
    """Run one fixed-K experiment; all de-duplication state is private to this run."""
    if k < 1 or batch_size < 1:
        raise ValueError("K and batch_size must be positive")
    if output_mode not in ("threshold_only", "exact_counts"):
        raise ValueError("Unknown overlap-output mode")
    if scheduler not in ("fingerprint", "document"):
        raise ValueError("Unknown candidate scheduler")
    if candidate_df_cap is not None and candidate_df_cap < 2:
        raise ValueError("Candidate df cap must be at least two")
    started = time.perf_counter()
    n_docs = len(data.document_ids)
    stop = len(data.fingerprints) if candidate_df_cap is None else int(
        np.searchsorted(data.source_df, candidate_df_cap, side="right")
    )
    cache_bytes = n_docs * (n_docs - 1) // 2 if scheduler == "fingerprint" else n_docs * 4
    base_bytes = data.nbytes + cache_bytes + n_docs * 20 + len(_METRICS) * 8 + 32 + batch_size * 24
    _require_memory(base_bytes, data.memory_budget_bytes)
    peak_owned_bytes = base_bytes
    metrics = np.zeros(len(_METRICS), dtype=np.uint64)
    output = np.empty(batch_size, dtype=SIMILARITY_DTYPE)
    checked = np.zeros(cache_bytes, dtype=np.bool_) if scheduler == "fingerprint" else None
    markers = np.zeros(n_docs, dtype=np.uint32)
    order = np.lexsort((data.document_ids, data.eligible_lengths)).astype(np.int64)
    join_rank = np.empty(n_docs, dtype=np.int64)
    join_rank[order] = np.arange(n_docs)
    state = np.array([0, 0, 1] if scheduler == "fingerprint" else [0, 0, 0, 1], dtype=np.int64)
    chunks = []
    buffered = 0
    batches = 0
    while state[0] < (stop if scheduler == "fingerprint" else n_docs):
        if scheduler == "fingerprint":
            used = _fingerprint_batch(
                data.posting_offsets, data.posting_docs, stop, state, checked,
                data.eligible_lengths, data.forward_fingerprints, data.forward_offsets,
                data.document_ids, k, output_mode == "exact_counts", metrics, output,
            )
        else:
            used = _document_batch(
                data.posting_offsets, data.posting_docs, stop, data.document_fp_offsets,
                data.document_fp_ranks, order, join_rank, state, markers,
                data.eligible_lengths, data.forward_fingerprints, data.forward_offsets,
                data.document_ids, k, output_mode == "exact_counts", metrics, output,
            )
        buffered += used * SIMILARITY_DTYPE.itemsize
        peak_owned_bytes = max(peak_owned_bytes, base_bytes + buffered * 3)
        _require_memory(
            peak_owned_bytes, data.memory_budget_bytes,
        )
        if used:
            chunks.append(output[:used].copy())
        batches += 1
    result = np.concatenate(chunks) if chunks else np.empty(0, dtype=SIMILARITY_DTYPE)
    result.sort(order=["doc_i", "doc_j"])
    if output_mode == "threshold_only":
        pairs = np.empty(len(result), dtype=PAIR_DTYPE)
        pairs["doc_i"], pairs["doc_j"] = result["doc_i"], result["doc_j"]
        result = pairs
    report = {
        "scheduler": scheduler, "k": k, "output_mode": output_mode,
        "candidate_df_cap": candidate_df_cap,
        "exhaustive_within_subset": candidate_df_cap is None,
        "completed": True, "accepted_pairs": len(result),
        "elapsed_seconds": time.perf_counter() - started,
        "checked_state_backend": "dense_triangular_bool_table" if scheduler == "fingerprint" else "per_probe_uint32_epoch_markers",
        "checked_state_bytes": cache_bytes,
        "prepared_arrays_bytes": data.nbytes, "output_bytes": result.nbytes,
        "estimated_peak_owned_bytes": peak_owned_bytes,
        "batch_size": batch_size, "batches": batches,
        "fingerprint_records_used": stop,
        "fingerprint_records_skipped": len(data.fingerprints) - stop,
        **{name: int(value) for name, value in zip(_METRICS, metrics, strict=True)},
    }
    return result, report


@njit(cache=True, nogil=True)
def _forward_records(offsets, values, records):
    for doc in range(len(offsets) - 1):
        for entry in range(np.intp(offsets[doc]), np.intp(offsets[doc + 1])):
            records[entry]["fingerprint"] = values[entry]
            records[entry]["doc"] = doc


@njit(cache=True, nogil=True)
def _reference_accumulate(records, counts, n_docs):
    first = 0
    updates = 0
    while first < len(records):
        last = first + 1
        while last < len(records) and records[last]["fingerprint"] == records[first]["fingerprint"]:
            last += 1
        for i in range(first, last):
            for j in range(i + 1, last):
                counts[_pair_slot(records[i]["doc"], records[j]["doc"], n_docs)] += 1
                updates += 1
        first = last
    return updates


@njit(cache=True, nogil=True)
def _dense_results(counts, ids, output):
    slot = used = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if counts[slot]:
                _emit(output, used, a, b, counts[slot], ids)
                used += 1
            slot += 1


def reference_counts(data: JoinSubset) -> tuple[np.ndarray, dict]:
    """Independently aggregate full-forward fingerprint groups, without the sidecar."""
    started = time.perf_counter()
    n_docs = len(data.document_ids)
    n_pairs = n_docs * (n_docs - 1) // 2
    _require_memory(
        data.nbytes + n_pairs * 16 + len(data.forward_fingerprints) * 24,
        data.memory_budget_bytes,
    )
    records = np.empty(len(data.forward_fingerprints), dtype=_FP_DOC)
    _forward_records(data.forward_offsets, data.forward_fingerprints, records)
    records.sort(order=["fingerprint", "doc"])
    counts = np.zeros(n_pairs, dtype=np.uint32)
    updates = _reference_accumulate(records, counts, n_docs)
    result = np.empty(np.count_nonzero(counts), dtype=SIMILARITY_DTYPE)
    _dense_results(counts, data.document_ids, result)
    return result, {
        "method": "independent_full_forward_grouped_inverted_count",
        "elapsed_seconds": time.perf_counter() - started,
        "pair_contributions": int(updates),
        "pairs_with_positive_overlap": len(result),
        "dense_counter_bytes": counts.nbytes,
    }


def warm_up() -> None:
    """Compile the same writable-array signatures used in the timed experiments."""
    ids = np.array([0, 1, 2], dtype=np.uint32)
    data = JoinSubset(
        ids, np.array([0, 3, 5, 7], dtype=np.uint64),
        np.array([1, 3, 9, 1, 3, 3, 9], dtype=np.uint64),
        np.array([1, 9, 3], dtype=np.uint64),
        np.array([0, 2, 4, 7], dtype=np.uint64),
        np.array([0, 1, 0, 2, 0, 1, 2], dtype=np.uint32),
        np.array([2, 2, 3], dtype=np.uint64),
        np.array([3, 2, 2], dtype=np.uint64),
        np.array([0, 3, 5, 7], dtype=np.uint64),
        np.array([0, 1, 2, 0, 2, 1, 2], dtype=np.uint32), 1 << 30,
    )
    reference_counts(data)
    for scheduler in ("fingerprint", "document"):
        for mode in ("threshold_only", "exact_counts"):
            run_join(data, 1, scheduler=scheduler, output_mode=mode, batch_size=2)
