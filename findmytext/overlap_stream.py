"""Single-worker document-owned overlap joins with atomic probe checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from filelock import FileLock
from numba import njit

from .overlap_inverted import _close_on_exit, _source_identity
from .overlap_join import (
    PAIR_DTYPE, JoinSubset, _FP_DOC, _METRICS, _emit, _forward_records,
    _require_memory, _verify,
)
from .similarity import SIMILARITY_DTYPE, _PairAccumulator


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def subset_identity(data: JoinSubset) -> dict:
    digest = hashlib.sha256()
    for name, value in vars(data).items():
        if isinstance(value, np.ndarray):
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(np.asarray(value.shape, dtype=np.uint64).tobytes())
            digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return {"sha256": digest.hexdigest(), "documents": len(data.document_ids)}


def save_subset(data: JoinSubset, directory: Path, source: dict) -> None:
    if data.source_identity != source:
        raise ValueError("Subset/source provenance mismatch")
    directory.mkdir()
    for name, value in vars(data).items():
        if isinstance(value, np.ndarray):
            np.save(directory / f"{name}.npy", value)
    atomic_json(directory / "meta.json", {
        "source": source, "identity": subset_identity(data),
        "memory_budget_bytes": data.memory_budget_bytes,
    })


def load_subset(directory: Path) -> JoinSubset:
    meta = json.loads((directory / "meta.json").read_text())
    values = {}
    for name in JoinSubset.__dataclass_fields__:
        if name == "source_identity":
            values[name] = meta["source"]
        elif name == "memory_budget_bytes":
            values[name] = meta[name]
        else:
            values[name] = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    data = JoinSubset(**values)
    if subset_identity(data) != meta["identity"]:
        raise ValueError("Persisted subset arrays do not match their identity")
    return data


@njit(cache=True, nogil=True)
def _filter_postings(offsets, docs, eligible, k, output_offsets, output, prefix_limits=None):
    cursor = 0
    output_offsets[0] = 0
    for rank in range(len(offsets) - 1):
        for entry in range(np.intp(offsets[rank]), np.intp(offsets[rank + 1])):
            doc = docs[entry]
            if eligible[doc] >= k and (prefix_limits is None or rank <= prefix_limits[doc]):
                output[cursor] = doc
                cursor += 1
        output_offsets[rank + 1] = cursor
    return cursor


@njit(cache=True, nogil=True)
def _probe_batch(post_offsets, docs, stop_fp, doc_offsets, doc_ranks, order,
                 join_rank, state, stop_probe, markers, eligible, values,
                 forward_offsets, verify_ids, ids, k, exact, early_pruning,
                 metrics, output, prefix_filter=False):
    probe, fp_cursor, doc_cursor = state
    visits = used = 0
    while probe < stop_probe and visits < len(output):
        a = order[probe]
        if early_pruning and eligible[a] < k:
            probe += 1
            fp_cursor = doc_cursor = 0
            continue
        first_rank = np.intp(doc_offsets[a])
        last_rank = np.intp(doc_offsets[a + 1])
        if prefix_filter:
            last_rank = first_rank + max(0, last_rank - first_rank - k + 1)
        if fp_cursor >= last_rank - first_rank:
            probe += 1
            fp_cursor = doc_cursor = 0
            continue
        rank = doc_ranks[first_rank + fp_cursor]
        if rank >= stop_fp:
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
        epoch = probe + 1
        if markers[b] == epoch:
            metrics[1] += 1
            continue
        markers[b] = epoch
        if eligible[a] < k or eligible[b] < k:
            metrics[3] += 1
            continue
        metrics[2] += 1
        accepted, count, comparisons, reason = _verify(
            values, forward_offsets, verify_ids[a], verify_ids[b], k, exact
        )
        metrics[4] += comparisons
        metrics[5 if reason == 0 else 6 if reason == 1 else 7] += 1
        if accepted:
            _emit(output, used, a, b, count, ids)
            used += 1
    state[:] = probe, fp_cursor, doc_cursor
    return used


@njit(cache=True, nogil=True)
def _prefix_limits(offsets, ranks, eligible, k):
    limits = np.full(len(eligible), -1, dtype=np.int64)
    for doc in range(len(eligible)):
        if eligible[doc] >= k:
            last = np.intp(offsets[doc + 1]) - k
            limits[doc] = ranks[last]
    return limits


def _io_counters() -> dict:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt}
    for line in Path("/proc/self/io").read_text().splitlines():
        name, value = line.split(":")
        if name in ("read_bytes", "write_bytes", "rchar", "wchar"):
            result[name] = int(value)
    return result


def _write_part(path: Path, rows: np.ndarray) -> dict:
    with path.open("wb") as stream:
        np.save(stream, rows, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "file": path.name, "rows": len(rows), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
    }


def _check_parts(root: Path, checkpoint: dict, dtype: np.dtype) -> None:
    for part in checkpoint["parts"]:
        path = root / part["file"]
        if path.parent != root or path.name != part["file"] or path.stat().st_size != part["bytes"]:
            raise ValueError(f"Committed output part is missing or changed: {path}")
        rows = np.load(path, allow_pickle=False)
        if (
            rows.dtype != dtype or rows.ndim != 1 or len(rows) != part["rows"]
            or hashlib.sha256(rows.tobytes()).hexdigest() != part["sha256"]
        ):
            raise ValueError(f"Committed output checksum/schema mismatch: {path}")


def run_document_stream(
    data: JoinSubset,
    output_dir: str | Path,
    k: int,
    *,
    source_index_dir: str | Path,
    output_mode: str = "exact_counts",
    candidate_df_cap: int | None = None,
    batch_size: int = 65_536,
    probes_per_commit: int = 256,
    max_partitions: int | None = None,
    early_pruning: bool = True,
    mmap_forward: bool = True,
    prefix_filter: bool = False,
    probe_start: int = 0,
    probe_stop: int | None = None,
    output_buffer_rows: int | None = None,
    verbose: bool = True,
) -> dict:
    """Stream results and resume only at atomically committed probe boundaries.

    max_partitions is an explicit pause control used for recovery testing, not
    a deadline. An unfinished partition is retried from its start on resume.
    """
    if k < 1 or min(batch_size, probes_per_commit) < 1:
        raise ValueError("K, batch size and probes per commit must be positive")
    if output_mode not in ("exact_counts", "threshold_only"):
        raise ValueError("Unknown overlap-output mode")
    if candidate_df_cap is not None and candidate_df_cap < 2:
        raise ValueError("Candidate df cap must be at least two")
    if max_partitions is not None and max_partitions < 1:
        raise ValueError("max_partitions must be positive")
    if output_buffer_rows is not None and output_buffer_rows < 1:
        raise ValueError("output_buffer_rows must be positive")
    n_docs = len(data.document_ids)
    stop_at = n_docs if probe_stop is None else probe_stop
    if not 0 <= probe_start < stop_at <= n_docs:
        raise ValueError("Probe range must be within the document join order")
    if prefix_filter and not early_pruning:
        raise ValueError("Prefix filtering requires early eligible-length pruning")
    source = Path(source_index_dir).resolve()
    root = Path(output_dir).resolve()
    if root == source or source in root.parents:
        raise ValueError("Stream output must be outside the source index")
    identity = _source_identity(source)
    if data.source_identity != identity:
        raise ValueError("Subset was prepared from a different or changed source index")
    contract = {
        "format": "findmytext-document-overlap-stream-v1",
        "source": identity, "subset": subset_identity(data), "k": k,
        "output_mode": output_mode, "candidate_df_cap": candidate_df_cap,
        "df_scope": "full_source_corpus",
        "ownership": "earlier_rank_in_eligible_length_then_source_id_order",
        "batch_size": batch_size, "probes_per_commit": probes_per_commit,
        "early_pruning": early_pruning, "mmap_forward": mmap_forward,
    }
    if prefix_filter:
        contract["prefix_filter"] = "both_rarity_prefixes_length_L_minus_K_plus_1_v1"
    if probe_start != 0 or stop_at != n_docs:
        contract["probe_range"] = [probe_start, stop_at]
    if output_buffer_rows is not None:
        contract["output_buffer_rows"] = output_buffer_rows
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "runner.lock"), timeout=0):
        checkpoint_path = root / "checkpoint.json"
        dtype = SIMILARITY_DTYPE if output_mode == "exact_counts" else PAIR_DTYPE
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint["contract"] != contract:
                raise ValueError("Checkpoint source/subset/K/mode/cap/settings mismatch")
            _check_parts(root, checkpoint, dtype)
        else:
            if any(path.name != "runner.lock" for path in root.iterdir()):
                raise ValueError("Untracked stream output directory; refusing to reuse it")
            checkpoint = {
                "contract": contract, "status": "partial", "next_probe": probe_start,
                "parts": [], "accepted_pairs": 0, "elapsed_seconds": 0.0,
                "metrics": {name: 0 for name in _METRICS},
                "exhaustive_within_subset": False,
            }
            atomic_json(checkpoint_path, checkpoint)
        # These names belong exclusively to this locked run. Files not in the
        # committed manifest were interrupted before the commit became visible.
        committed = {part["file"] for part in checkpoint["parts"]}
        for path in root.glob("part-*.npy"):
            if path.name not in committed:
                path.unlink()
        if checkpoint["status"] == "complete":
            return checkpoint
        owned_bytes = data.nbytes + len(data.posting_docs) * 4 + len(data.posting_offsets) * 8 + n_docs * 36 + batch_size * 32
        if output_buffer_rows is not None:
            owned_bytes += output_buffer_rows * 32
        _require_memory(owned_bytes, data.memory_budget_bytes)
        started = time.perf_counter()
        io_start = _io_counters()
        order = np.lexsort((data.document_ids, data.eligible_lengths)).astype(np.int64)
        join_rank = np.empty(n_docs, dtype=np.int64)
        join_rank[order] = np.arange(n_docs)
        filtered_offsets = np.empty_like(data.posting_offsets)
        filtered_docs = np.empty_like(data.posting_docs)
        limits = _prefix_limits(
            data.document_fp_offsets, data.document_fp_ranks, data.eligible_lengths, k
        ) if prefix_filter else None
        if early_pruning:
            kept = _filter_postings(
                data.posting_offsets, data.posting_docs, data.eligible_lengths,
                k, filtered_offsets, filtered_docs, limits,
            )
            filtered_docs = filtered_docs[:kept]
        else:
            filtered_offsets = data.posting_offsets
            filtered_docs = data.posting_docs
        stop_fp = len(data.fingerprints) if candidate_df_cap is None else int(
            np.searchsorted(data.source_df, candidate_df_cap, side="right")
        )
        checkpoint["posting_entries_removed_before_enumeration"] = len(data.posting_docs) - len(filtered_docs)
        checkpoint["probes_removed_before_enumeration"] = int(np.count_nonzero(data.eligible_lengths < k)) if early_pruning else 0
        checkpoint["estimated_owned_bytes"] = owned_bytes
        checkpoint["working_allowance_bytes"] = data.memory_budget_bytes
        checkpoint["marker_bytes"] = n_docs * 4
        checkpoint["fingerprint_records_skipped_by_cap"] = len(data.fingerprints) - stop_fp
        previous_io = checkpoint.get("io_counters", {})
        previous_elapsed = checkpoint["elapsed_seconds"]
        with ExitStack() as mappings:
            if mmap_forward:
                values_path = getattr(data.forward_fingerprints, "filename", None)
                offsets_path = getattr(data.forward_offsets, "filename", None)
                reuse_source_maps = (
                    isinstance(values_path, (str, Path)) and isinstance(offsets_path, (str, Path))
                    and Path(values_path).resolve() == source / "forward_fingerprints.npy"
                    and Path(offsets_path).resolve() == source / "forward_offsets.npy"
                    and len(data.forward_fingerprints) == int(data.forward_offsets[-1])
                )
                if reuse_source_maps:
                    values, forward_offsets = data.forward_fingerprints, data.forward_offsets
                else:
                    values = np.load(source / "forward_fingerprints.npy", mmap_mode="r")
                    _close_on_exit(mappings, values)
                    forward_offsets = np.load(source / "forward_offsets.npy", mmap_mode="r")
                    _close_on_exit(mappings, forward_offsets)
                verify_ids = data.document_ids
            else:
                values, forward_offsets = data.forward_fingerprints, data.forward_offsets
                verify_ids = np.arange(n_docs, dtype=np.uint32)
            output = np.empty(batch_size, dtype=SIMILARITY_DTYPE)
            accepted_buffer = np.empty(output_buffer_rows, dtype=SIMILARITY_DTYPE) if output_buffer_rows is not None else None
            markers = np.zeros(n_docs, dtype=np.uint32)
            partitions = 0
            probe_started = time.perf_counter()
            previous_probe_seconds = checkpoint.get("probe_seconds", 0.0)
            checkpoint["setup_seconds_last_session"] = probe_started - started
            while checkpoint["next_probe"] < stop_at:
                start = checkpoint["next_probe"]
                stop = min(start + probes_per_commit, stop_at)
                state = np.array([start, 0, 0], dtype=np.int64)
                metrics = np.zeros(len(_METRICS), dtype=np.uint64)
                parts = []
                buffered = 0

                def persist_rows(rows):
                    if output_mode == "threshold_only":
                        pairs = np.empty(len(rows), dtype=PAIR_DTYPE)
                        pairs["doc_i"], pairs["doc_j"] = rows["doc_i"], rows["doc_j"]
                        rows = pairs
                    path = root / f"part-{start:09d}-{stop:09d}-{len(parts):05d}.npy"
                    parts.append(_write_part(path, rows))

                while state[0] < stop:
                    used = _probe_batch(
                        filtered_offsets, filtered_docs, stop_fp,
                        data.document_fp_offsets, data.document_fp_ranks,
                        order, join_rank, state, stop, markers, data.eligible_lengths,
                        values, forward_offsets, verify_ids, data.document_ids,
                        k, output_mode == "exact_counts", early_pruning, metrics, output, prefix_filter,
                    )
                    if not used:
                        continue
                    if accepted_buffer is None:
                        persist_rows(output[:used])
                    else:
                        cursor = 0
                        while cursor < used:
                            count = min(used - cursor, len(accepted_buffer) - buffered)
                            accepted_buffer[buffered:buffered + count] = output[cursor:cursor + count]
                            buffered += count
                            cursor += count
                            if buffered == len(accepted_buffer):
                                persist_rows(accepted_buffer)
                                buffered = 0
                if accepted_buffer is not None and buffered:
                    persist_rows(accepted_buffer[:buffered])
                checkpoint["parts"].extend(parts)
                checkpoint["accepted_pairs"] += sum(part["rows"] for part in parts)
                for name, value in zip(_METRICS, metrics, strict=True):
                    checkpoint["metrics"][name] += int(value)
                checkpoint["next_probe"] = stop
                checkpoint["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
                checkpoint["probe_seconds"] = previous_probe_seconds + time.perf_counter() - probe_started
                io_end = _io_counters()
                checkpoint["io_counters"] = {
                    key: previous_io.get(key, 0) + io_end[key] - io_start[key]
                    for key in io_end
                }
                if stop == stop_at:
                    if _source_identity(source) != identity:
                        raise ValueError("Source index changed during streamed join")
                    checkpoint["status"] = "complete"
                checkpoint["exhaustive_within_subset"] = (
                    checkpoint["status"] == "complete" and candidate_df_cap is None
                    and probe_start == 0 and stop_at == n_docs
                )
                atomic_json(checkpoint_path, checkpoint)
                partitions += 1
                if verbose and (stop == stop_at or partitions % 20 == 0):
                    print(f"[stream join] K={k} cap={candidate_df_cap} prefix={prefix_filter} {stop:,}/{stop_at:,} probes; {checkpoint['accepted_pairs']:,} pairs", flush=True)
                if max_partitions is not None and partitions >= max_partitions:
                    break
        return checkpoint


@njit(cache=True, nogil=True)
def _reference_pair_batch(records, state, ids, keys):
    first, last, left, right = state
    used = 0
    while first < len(records) and used < len(keys):
        if last <= first:
            last = first + 1
            while last < len(records) and records[last]["fingerprint"] == records[first]["fingerprint"]:
                last += 1
            left, right = first, first + 1
        if left >= last - 1:
            first = last
            continue
        if right >= last:
            left += 1
            right = left + 1
            continue
        a = np.uint64(ids[records[left]["doc"]])
        b = np.uint64(ids[records[right]["doc"]])
        keys[used] = (a << np.uint64(32)) | b
        used += 1
        right += 1
    state[:] = first, last, left, right
    return used


def _external_reference(
    data: JoinSubset, output_dir: str | Path, batch_entries: int = 1_000_000,
) -> dict:
    """Independent full-forward grouped counting with bounded external aggregation."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    if batch_entries < 1:
        raise ValueError("Reference batch size must be positive")
    _require_memory(data.nbytes + len(data.forward_fingerprints) * 24 + batch_entries * 128, data.memory_budget_bytes)
    root.mkdir(parents=True)
    started = time.perf_counter()
    records = np.empty(len(data.forward_fingerprints), dtype=_FP_DOC)
    _forward_records(data.forward_offsets, data.forward_fingerprints, records)
    records.sort(order=["fingerprint", "doc"])
    result_path = root / "counts.npy"
    accumulator = _PairAccumulator(
        str(result_path), max_pair_updates_in_memory=batch_entries,
        min_shared_fingerprints=1, max_chunks_per_merge=32,
    )
    state = np.zeros(4, dtype=np.int64)
    updates = 0
    while state[0] < len(records):
        keys = np.empty(batch_entries, dtype=np.uint64)
        used = _reference_pair_batch(records, state, data.document_ids, keys)
        if used:
            accumulator.add(keys[:used], np.ones(used, dtype=np.uint32))
            updates += used
    pairs = accumulator.finish()
    with ExitStack() as mappings:
        rows = np.load(result_path, mmap_mode="r")
        _close_on_exit(mappings, rows)
        keys = np.lib.format.open_memmap(root / "keys.npy", mode="w+", dtype=np.uint64, shape=(pairs,))
        _close_on_exit(mappings, keys)
        for first in range(0, pairs, batch_entries):
            last = min(first + batch_entries, pairs)
            keys[first:last] = (
                rows["doc_i"][first:last].astype(np.uint64) << np.uint64(32)
            ) | rows["doc_j"][first:last]
        keys.flush()
    report = {
        "method": "independent_full_forward_external_grouped_pair_count",
        "subset": subset_identity(data), "positive_pairs": pairs,
        "pair_contributions": updates, "batch_entries": batch_entries,
        "elapsed_seconds": time.perf_counter() - started,
        "final_bytes": sum(path.stat().st_size for path in root.iterdir() if path.is_file()),
    }
    atomic_json(root / "report.json", report)
    return report


def external_reference(
    data: JoinSubset, output_dir: str | Path, batch_entries: int = 1_000_000,
) -> dict:
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=root.parent) as temporary:
        staging = Path(temporary) / "reference"
        report = _external_reference(data, staging, batch_entries)
        staging.rename(root)
    return report


def validate_stream(output_dir: str | Path, reference_dir: str | Path, batch_entries: int = 1_000_000) -> dict:
    root, reference = Path(output_dir), Path(reference_dir)
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    if checkpoint["status"] != "complete":
        raise ValueError("Cannot advertise an unfinished streamed join as validated complete")
    contract = checkpoint["contract"]
    if "probe_range" in contract:
        raise ValueError("Probe-range pilots require range-aware validation, not a whole-subset reference")
    reference_report = json.loads((reference / "report.json").read_text())
    if reference_report["subset"] != contract["subset"]:
        raise ValueError("Independent reference belongs to a different subset")
    exact = contract["output_mode"] == "exact_counts"
    _check_parts(root, checkpoint, SIMILARITY_DTYPE if exact else PAIR_DTYPE)
    digest = hashlib.sha256()
    with ExitStack() as mappings:
        counts = np.load(reference / "counts.npy", mmap_mode="r")
        _close_on_exit(mappings, counts)
        keys = np.load(reference / "keys.npy", mmap_mode="r")
        _close_on_exit(mappings, keys)
        seen = np.zeros(len(keys), dtype=np.bool_)
        expected = 0
        for start in range(0, len(counts), batch_entries):
            expected += int(np.count_nonzero(counts["count"][start:start + batch_entries] >= contract["k"]))
        accepted = 0
        for part in checkpoint["parts"]:
            rows = np.load(root / part["file"], allow_pickle=False)
            pair_keys = (rows["doc_i"].astype(np.uint64) << np.uint64(32)) | rows["doc_j"]
            positions = np.searchsorted(keys, pair_keys)
            if np.any(positions >= len(keys)) or not np.array_equal(keys[positions], pair_keys):
                raise AssertionError("Stream contains a pair absent from the independent reference")
            if np.any(seen[positions]) or len(np.unique(positions)) != len(positions):
                raise AssertionError("Stream duplicated an accepted pair")
            if np.any(counts["count"][positions] < contract["k"]):
                raise AssertionError("Stream emitted a pair below K")
            if exact and not np.array_equal(rows["count"], counts["count"][positions]):
                raise AssertionError("Stream exact counts differ from the independent reference")
            seen[positions] = True
            digest.update(pair_keys.tobytes())
            accepted += len(rows)
        if contract["candidate_df_cap"] is None and accepted != expected:
            raise AssertionError(f"Uncapped stream missed {expected - accepted} qualifying pairs")
    result = {
        "validated": True, "accepted_pairs": accepted, "reference_pairs_at_k": expected,
        "recall": accepted / expected if expected else None,
        "missing_pairs": expected - accepted, "ordered_pair_digest": digest.hexdigest(),
    }
    atomic_json(root / "validation.json", result)
    return result


def _filter_count_cache(
    stream_dir: str | Path, output_dir: str | Path, k_values: tuple[int, ...],
) -> dict:
    """Reuse exact accepted counts at a floor K for any equal/higher threshold."""
    source = Path(stream_dir).resolve()
    root = Path(output_dir).resolve()
    checkpoint = json.loads((source / "checkpoint.json").read_text())
    contract = checkpoint["contract"]
    if checkpoint["status"] != "complete" or contract["output_mode"] != "exact_counts":
        raise ValueError("Count reuse requires completed exact-count output")
    if not k_values or min(k_values) < contract["k"]:
        raise ValueError("Requested K is below the cache's minimum supported threshold")
    index = Path(contract["source"]["path"])
    if root == index or index in root.parents:
        raise ValueError("Count-cache output must be outside the source index")
    if root.exists():
        raise FileExistsError(root)
    if _source_identity(index) != contract["source"]:
        raise ValueError("Count cache belongs to a changed source index")
    _check_parts(source, checkpoint, SIMILARITY_DTYPE)
    started = time.perf_counter()
    thresholds = sorted(set(k_values), reverse=True)
    counts = {k: 0 for k in thresholds}
    for part in checkpoint["parts"]:
        rows = np.load(source / part["file"])
        for k in thresholds:
            counts[k] += int(np.count_nonzero(rows["count"] >= k))
    root.mkdir(parents=True)
    digests = {k: hashlib.sha256() for k in thresholds}
    with ExitStack() as streams:
        outputs = {}
        cursors = {k: 0 for k in thresholds}
        for k in thresholds:
            outputs[k] = streams.enter_context((root / f"k{k}_exact_counts.npy").open("wb", buffering=8 * 2**20))
            np.lib.format.write_array_header_1_0(outputs[k], {
                "descr": np.lib.format.dtype_to_descr(SIMILARITY_DTYPE),
                "fortran_order": False, "shape": (counts[k],),
            })
        for part in checkpoint["parts"]:
            rows = np.load(source / part["file"])
            for k in thresholds:
                selected = rows[rows["count"] >= k]
                outputs[k].write(selected.tobytes())
                cursors[k] += len(selected)
                keys = (selected["doc_i"].astype(np.uint64) << np.uint64(32)) | selected["doc_j"]
                digests[k].update(keys.tobytes())
        for stream in outputs.values():
            stream.flush()
            os.fsync(stream.fileno())
        if cursors != counts:
            raise ValueError("Exact-count cache changed during filtering")
    if _source_identity(index) != contract["source"]:
        raise ValueError("Source index changed during count-cache filtering")
    report = {
        "status": "passed", "source_contract": contract,
        "minimum_supported_k": contract["k"],
        "candidate_df_cap": contract["candidate_df_cap"],
        "exhaustive_within_subset": checkpoint["exhaustive_within_subset"],
        "input_output_bytes": sum(part["bytes"] for part in checkpoint["parts"]),
        "filter_seconds_including_output_writes": time.perf_counter() - started,
        "output_writer": "bounded_buffered_npy_stream",
        "thresholds": {
            str(k): {
                "pairs": counts[k], "ordered_pair_digest": digests[k].hexdigest(),
                "file": f"k{k}_exact_counts.npy",
            }
            for k in thresholds
        },
    }
    atomic_json(root / "report.json", report)
    return report


def filter_count_cache(
    stream_dir: str | Path, output_dir: str | Path, k_values: tuple[int, ...],
) -> dict:
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    checkpoint = json.loads((Path(stream_dir) / "checkpoint.json").read_text())
    index = Path(checkpoint["contract"]["source"]["path"])
    if root == index or index in root.parents:
        raise ValueError("Count-cache output must be outside the source index")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count-cache-", dir=root.parent) as temporary:
        staging = Path(temporary) / "cache"
        report = _filter_count_cache(stream_dir, staging, k_values)
        staging.rename(root)
    return report
