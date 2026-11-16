"""Check all three persisted indexes and compare exact sampled pair overlaps.

Example:
    .venv/bin/python -m experiments.verify_overlap_inverted \
        /path/to/build/index /path/to/build/overlap_inverted \
        --report /path/to/build/verification.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from numba import njit

from findmytext import utils, winnower
from findmytext.indexing import POSTING_DTYPE
from findmytext.overlap_inverted import (
    OverlapInvertedIndex,
    _close_on_exit,
    _load_array,
    _source_identity,
)


@njit(cache=True)
def _find(values, first, last, value):
    first = np.intp(first)
    last = np.intp(last)
    end = last
    while first < last:
        middle = first + (last - first) // 2
        if values[middle] < value:
            first = middle + 1
        else:
            last = middle
    return first if first < end and values[first] == value else -1


@njit(cache=True)
def _check_forward(offsets, fps, start, stop):
    for doc in range(start, stop):
        first = int(offsets[doc])
        last = int(offsets[doc + 1])
        if last < first or last > len(fps):
            raise ValueError("Invalid forward offsets")
        for entry in range(first + 1, last):
            if fps[entry] <= fps[entry - 1]:
                raise ValueError("Forward fingerprints are not strictly hash-sorted and unique")


@njit(cache=True)
def _check_positional(fps, offsets, lengths, postings, forward_offsets, forward_fps,
                      start, stop, seen, scratch_docs, full_counts, eligible_counts, dfs):
    unique_pairs = singletons = retained = high_hashes = 0
    for rank in range(start, stop):
        if rank and fps[rank] <= fps[rank - 1]:
            raise ValueError("Positional fingerprints are not strictly hash-sorted")
        if fps[rank] >= np.uint64(1) << np.uint64(63):
            high_hashes += 1
        expected = (
            offsets[rank - 1] + np.uint64(lengths[rank - 1]) * 8
            if rank else np.uint64(0)
        )
        if offsets[rank] != expected or lengths[rank] == 0:
            raise ValueError("Invalid positional offsets or zero-length posting list")
        first = int(offsets[rank] // 8)
        last = first + int(lengths[rank])
        if last > len(postings):
            raise ValueError("Positional posting list extends past its payload")
        count = 0
        epoch = np.uint64(rank) + np.uint64(1)
        for entry in range(first, last):
            doc = postings[entry]["doc_id"]
            if doc >= len(seen):
                raise ValueError("Positional document ID is out of range")
            if seen[doc] == epoch:
                continue
            seen[doc] = epoch
            if _find(forward_fps, int(forward_offsets[doc]),
                     int(forward_offsets[doc + 1]), fps[rank]) < 0:
                raise ValueError("Positional fingerprint/document pair is absent from forward index")
            full_counts[doc] += 1
            scratch_docs[count] = doc
            count += 1
        dfs[rank] = count
        unique_pairs += count
        if count == 1:
            singletons += 1
        else:
            retained += 1
            for entry in range(count):
                eligible_counts[scratch_docs[entry]] += 1
    return unique_pairs, singletons, retained, high_hashes


@njit(cache=True)
def _check_overlap(fps, offsets, docs, source_fps, source_dfs,
                   forward_offsets, forward_fps, start, stop, eligible_counts,
                   sample_cursors, sampled_fps):
    for rank in range(start, stop):
        first = int(offsets[rank])
        last = int(offsets[rank + 1])
        count = last - first
        if count < 2 or last > len(docs):
            raise ValueError("Overlap inverted posting list has invalid offsets or df < 2")
        if rank:
            previous_count = int(offsets[rank] - offsets[rank - 1])
            if count < previous_count or (
                count == previous_count and fps[rank] <= fps[rank - 1]
            ):
                raise ValueError("Overlap inverted records are not strictly ordered by (df, hash)")
        source_rank = _find(source_fps, 0, len(source_fps), fps[rank])
        if source_rank < 0 or source_dfs[source_rank] != count:
            raise ValueError("Overlap inverted fingerprint/document frequency differs from positional index")
        for entry in range(first, last):
            doc = docs[entry]
            if doc >= len(eligible_counts):
                raise ValueError("Overlap inverted document ID is out of range")
            if entry > first and doc <= docs[entry - 1]:
                raise ValueError("Overlap inverted document IDs are not ascending and unique")
            if _find(forward_fps, int(forward_offsets[doc]),
                     int(forward_offsets[doc + 1]), fps[rank]) < 0:
                raise ValueError("Overlap inverted fingerprint/document pair is absent from forward index")
            eligible_counts[doc] += 1
            cursor = sample_cursors[doc]
            if cursor >= 0:
                sampled_fps[cursor] = fps[rank]
                sample_cursors[doc] += 1


def _sample_pairs(overlap: OverlapInvertedIndex, n_docs: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    target = min(count, n_docs * (n_docs - 1) // 2)
    pairs = set()
    if len(overlap.fingerprints):
        for _ in range(target // 2):
            docs = overlap.get_postings(int(rng.integers(len(overlap.fingerprints))))
            selected = rng.choice(len(docs), size=2, replace=False)
            pairs.add(tuple(sorted((int(docs[selected[0]]), int(docs[selected[1]])))))
    while len(pairs) < target:
        pairs.add(tuple(sorted(map(int, rng.choice(n_docs, size=2, replace=False)))))
    return sorted(pairs)


@njit(cache=True)
def _check_sample_positions(fps, offsets, lengths, postings, slots,
                            expected_offsets, expected_fps, expected_positions, matched):
    for rank in range(len(fps)):
        first = int(offsets[rank] // 8)
        last = first + int(lengths[rank])
        for entry in range(first, last):
            posting = postings[entry]
            slot = slots[posting["doc_id"]]
            if slot < 0:
                continue
            position = _find(
                expected_positions, expected_offsets[slot], expected_offsets[slot + 1],
                posting["position"],
            )
            if position < 0 or expected_fps[position] != fps[rank]:
                raise ValueError("A sampled positional fingerprint differs from source text")
            if matched[position]:
                raise ValueError("A sampled positional occurrence is duplicated")
            matched[position] = True


def _verify_corpus_sample(source, corpus_dir, source_identity, fps, offsets, lengths,
                          postings, forward_offsets, forward_fps, name_offsets,
                          name_bytes, seed, verbose):
    files = sorted(Path(corpus_dir).glob("*.jsonl.zst"))
    if not files:
        raise FileNotFoundError(f"No compressed corpus shards found in {corpus_dir}")

    def identities():
        return [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for path in files
        ]

    corpus_identity = identities()
    manifest_path = source.parent / "build.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["corpus_files"] != corpus_identity:
            raise ValueError("Corpus shards do not match the index build manifest")
    rng = np.random.default_rng(seed)
    records = {}
    shard_counts = {}
    for path in files:
        reservoir = []
        count = 0
        for count, record in enumerate(
            utils.stream_json_zst(str(path), min_length=100, max_length=100_000), start=1
        ):
            if len(reservoir) < 4:
                reservoir.append(record)
            else:
                selected = int(rng.integers(count))
                if selected < len(reservoir):
                    reservoir[selected] = record
        shard_counts[path.name] = count
        for record in reservoir:
            records[str(record["id"])] = (path.name, record["text"])
        if verbose:
            print(f"[verify] source-text sampling: {path.name}, {count:,} eligible records", flush=True)
    wanted = set(records)
    selected_docs = {}
    for doc in range(len(name_offsets) - 1):
        external_id = bytes(name_bytes[int(name_offsets[doc]):int(name_offsets[doc + 1])]).decode("utf-8")
        if external_id in wanted:
            selected_docs[doc] = external_id
            wanted.remove(external_id)
    if wanted:
        raise ValueError(f"Sampled corpus document IDs are absent from index: {sorted(wanted)}")
    extractor = winnower.Winnower.from_metadata(source_identity["settings"])
    slots = np.full(len(name_offsets) - 1, -1, dtype=np.int64)
    expected_offsets = [0]
    all_fps = []
    all_positions = []
    sample_report = []
    for slot, (doc, external_id) in enumerate(sorted(selected_docs.items())):
        shard, text = records[external_id]
        hashes, positions = extractor.get_winnowed_fingerprints(text)
        expected = np.unique(hashes)
        actual = forward_fps[int(forward_offsets[doc]):int(forward_offsets[doc + 1])]
        if not np.array_equal(expected, actual):
            raise ValueError(f"Forward fingerprints differ from source text for {external_id}")
        slots[doc] = slot
        all_fps.extend(map(int, hashes))
        all_positions.extend(map(int, positions))
        expected_offsets.append(len(all_fps))
        sample_report.append({
            "doc_id": doc, "external_id": external_id, "shard": shard,
            "occurrences": len(hashes), "unique_fingerprints": len(expected),
        })
    matched = np.zeros(len(all_fps), dtype=np.bool_)
    _check_sample_positions(
        fps, offsets, lengths, postings, slots,
        np.array(expected_offsets, dtype=np.uint64),
        np.array(all_fps, dtype=np.uint64),
        np.array(all_positions, dtype=np.uint32), matched,
    )
    if not np.all(matched):
        raise ValueError("Sampled source-text occurrences are missing from positional index")
    if identities() != corpus_identity:
        raise ValueError("Corpus shards changed during source-text verification")
    return {
        "status": "passed", "records_per_shard_target": 4,
        "eligible_records_per_shard": shard_counts,
        "documents": sample_report, "positional_occurrences_checked": len(all_fps),
        "corpus_files": corpus_identity,
    }


def verify_indexes(
    index_dir: str | Path,
    overlap_dir: str | Path,
    *,
    sample_pairs: int = 256,
    seed: int = 41,
    batch_fingerprints: int = 1_000_000,
    report_path: str | Path | None = None,
    corpus_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Validate every posting/membership, then independently count sampled pairs.

    Checking subset membership plus exact per-document/per-fingerprint
    cardinalities establishes equality of the three set representations.
    Document-frequency scratch is disk-backed; it is removed on completion.
    """
    if sample_pairs < 0 or batch_fingerprints < 1:
        raise ValueError("Sample count must be nonnegative and batch size positive")
    source = Path(index_dir).resolve()
    overlap_root = Path(overlap_dir).resolve()
    report_file = Path(report_path).resolve() if report_path is not None else None
    if report_file is not None and (
        source == report_file or source in report_file.parents
        or overlap_root == report_file or overlap_root in report_file.parents
    ):
        raise ValueError("The verification report must be outside the index directories")
    started = time.perf_counter()
    source_identity = _source_identity(source)
    overlap = OverlapInvertedIndex(overlap_root, source)
    fps = _load_array(source / "fingerprints.npy", np.dtype("uint64"))
    offsets = _load_array(source / "offsets.npy", np.dtype("uint64"))
    lengths = _load_array(source / "lengths.npy", np.dtype("uint32"))
    forward_offsets = _load_array(source / "forward_offsets.npy", np.dtype("uint64"))
    forward_fps = _load_array(source / "forward_fingerprints.npy", np.dtype("uint64"))
    name_offsets = _load_array(source / "doc_name_offsets.npy", np.dtype("uint64"))
    name_bytes = _load_array(source / "doc_name_bytes.npy", np.dtype("uint8"))
    n_docs = overlap.meta["document_count"]
    if (
        len(offsets) != len(fps) or len(lengths) != len(fps)
        or len(forward_offsets) != n_docs + 1 or len(name_offsets) != n_docs + 1
        or forward_offsets[0] != 0 or int(forward_offsets[-1]) != len(forward_fps)
        or name_offsets[0] != 0 or int(name_offsets[-1]) != len(name_bytes)
    ):
        raise ValueError("Source array cardinalities or endpoint offsets do not match")
    if np.any(name_offsets[1:] <= name_offsets[:-1]):
        raise ValueError("Source document names have invalid offsets")
    posting_bytes = (source / "postings.dat").stat().st_size
    expected_bytes = int(offsets[-1]) + int(lengths[-1]) * 8 if len(fps) else 0
    if posting_bytes != expected_bytes:
        raise ValueError("Positional payload byte count is inconsistent")
    postings = (
        np.memmap(source / "postings.dat", mode="r", dtype=POSTING_DTYPE)
        if posting_bytes else np.empty(0, dtype=POSTING_DTYPE)
    )
    phase_seconds = {}

    def progress(phase: str, completed: int, total: int) -> None:
        if verbose:
            print(f"[verify] {phase}: {completed:,}/{total:,}", flush=True)

    phase_start = time.perf_counter()
    for start in range(0, n_docs, batch_fingerprints):
        stop = min(start + batch_fingerprints, n_docs)
        _check_forward(forward_offsets, forward_fps, start, stop)
        progress("forward ordering", stop, n_docs)
    phase_seconds["forward_ordering"] = time.perf_counter() - phase_start
    full_counts = np.zeros(n_docs, dtype=np.uint64)
    eligible_counts = np.zeros(n_docs, dtype=np.uint64)
    source_counts = {"unique_pairs": 0, "singletons": 0, "retained": 0, "high_hashes": 0}
    with (
        tempfile.TemporaryDirectory(prefix=".verify-overlap-", dir=source.parent) as tmp,
        ExitStack() as mappings,
    ):
        dfs = np.lib.format.open_memmap(
            Path(tmp) / "document_frequencies.npy", mode="w+",
            dtype=np.uint64, shape=(len(fps),),
        )
        _close_on_exit(mappings, dfs)
        seen = np.zeros(n_docs, dtype=np.uint64)
        scratch_docs = np.empty(n_docs, dtype=np.uint32)
        phase_start = time.perf_counter()
        for start in range(0, len(fps), batch_fingerprints):
            stop = min(start + batch_fingerprints, len(fps))
            counts = _check_positional(
                fps, offsets, lengths, postings, forward_offsets, forward_fps,
                start, stop, seen, scratch_docs, full_counts, eligible_counts, dfs,
            )
            for key, value in zip(source_counts, counts, strict=True):
                source_counts[key] += int(value)
            progress("positional/forward membership", stop, len(fps))
        if not np.array_equal(full_counts, np.diff(forward_offsets)):
            raise ValueError("Forward index has extra or missing document/fingerprint pairs")
        if not np.array_equal(eligible_counts, overlap.eligible_lengths):
            raise ValueError("Persisted eligible lengths differ from positional index")
        if len(overlap.fingerprints) != source_counts["retained"]:
            raise ValueError("Overlap inverted fingerprint coverage is incomplete")
        phase_seconds["positional_forward_membership"] = time.perf_counter() - phase_start
        pairs = _sample_pairs(overlap, n_docs, sample_pairs, seed)
        selected_docs = sorted({doc for pair in pairs for doc in pair})
        sample_starts = np.full(n_docs, -1, dtype=np.int64)
        total = 0
        for doc in selected_docs:
            sample_starts[doc] = total
            total += int(eligible_counts[doc])
        sampled_fps = np.empty(total, dtype=np.uint64)
        sample_cursors = sample_starts.copy()
        overlap_counts = np.zeros(n_docs, dtype=np.uint64)
        phase_start = time.perf_counter()
        for start in range(0, len(overlap.fingerprints), batch_fingerprints):
            stop = min(start + batch_fingerprints, len(overlap.fingerprints))
            _check_overlap(
                overlap.fingerprints, overlap.offsets, overlap.doc_ids, fps, dfs,
                forward_offsets, forward_fps, start, stop, overlap_counts,
                sample_cursors, sampled_fps,
            )
            progress("overlap inverted membership/order", stop, len(overlap.fingerprints))
        if not np.array_equal(overlap_counts, eligible_counts):
            raise ValueError("Overlap inverted per-document coverage differs from positional index")
        phase_seconds["overlap_membership"] = time.perf_counter() - phase_start

    sampled_counts = []
    for first, second in pairs:
        a = forward_fps[int(forward_offsets[first]):int(forward_offsets[first + 1])]
        b = forward_fps[int(forward_offsets[second]):int(forward_offsets[second + 1])]
        expected = len(np.intersect1d(a, b, assume_unique=True))
        overlap_a = sampled_fps[sample_starts[first]:sample_cursors[first]]
        overlap_b = sampled_fps[sample_starts[second]:sample_cursors[second]]
        actual = len(np.intersect1d(overlap_a, overlap_b, assume_unique=True))
        if actual != expected:
            raise ValueError(f"Exact overlap differs for pair {(first, second)}: {expected} vs {actual}")
        sampled_counts.append({"doc_i": first, "doc_j": second, "count": actual})
    progress("sampled exact pair counts", len(pairs), len(pairs))
    stats = overlap.meta["statistics"]
    expected_stats = {
        "input_fingerprints": len(fps),
        "singleton_fingerprints": source_counts["singletons"],
        "retained_fingerprints": source_counts["retained"],
        "zero_document_fingerprints": 0,
        "input_positional_postings": len(postings),
        "unique_doc_fingerprint_pairs": source_counts["unique_pairs"],
        "collapsed_repeats": len(postings) - source_counts["unique_pairs"],
        "singleton_pairs_removed": source_counts["singletons"],
        "retained_doc_id_postings": len(overlap.doc_ids),
        "documents_without_retained_fingerprints": int(np.count_nonzero(eligible_counts == 0)),
    }
    for key, expected in expected_stats.items():
        if stats[key] != expected:
            raise ValueError(f"Persisted build accounting differs for {key}: {stats[key]} vs {expected}")
    corpus_report = None
    if corpus_dir is not None:
        phase_start = time.perf_counter()
        corpus_report = _verify_corpus_sample(
            source, corpus_dir, source_identity, fps, offsets, lengths,
            postings, forward_offsets, forward_fps, name_offsets, name_bytes,
            seed, verbose,
        )
        phase_seconds["source_text_samples"] = time.perf_counter() - phase_start
    files = [*overlap_root.glob("*.npy"), overlap_root / "meta.json"]
    storage = overlap.meta["storage_bytes"]
    if (
        any(storage[path.name] != path.stat().st_size for path in files)
        or storage["total"] != sum(path.stat().st_size for path in files)
    ):
        raise ValueError("Persisted overlap inverted byte accounting is incorrect")
    if _source_identity(source) != source_identity:
        raise ValueError("Source index changed during verification")
    report = {
        "status": "passed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": source_identity,
        "overlap_inverted": str(overlap_root),
        "document_count": n_docs,
        "hash_settings": source_identity["settings"],
        "statistics": expected_stats,
        "full_forward_entries": len(forward_fps),
        "storage_bytes": {
            "positional_inverted": sum(
                entry["size"] for name, entry in source_identity["files"].items()
                if not name.startswith("forward_")
            ),
            "forward": sum(
                entry["size"] for name, entry in source_identity["files"].items()
                if name.startswith("forward_")
            ),
            "overlap_inverted": storage["total"],
        },
        "fingerprints_at_or_above_2_pow_63": source_counts["high_hashes"],
        "all_positional_forward_memberships_checked": True,
        "all_overlap_memberships_and_ordering_checked": True,
        "source_unchanged": True,
        "sample_seed": seed,
        "sampled_exact_pairs": sampled_counts,
        "source_text_verification": corpus_report,
        "phase_seconds": phase_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        partial = report_file.with_suffix(report_file.suffix + ".tmp")
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(report_file)
    if verbose:
        print(
            f"[verify] PASSED: {n_docs:,} documents, {len(fps):,} positional fingerprints, "
            f"{len(forward_fps):,} forward entries, {len(overlap.fingerprints):,} "
            f"overlap inverted fingerprints, {len(pairs):,} exact sampled pairs; "
            f"{report['elapsed_seconds']:.2f}s",
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("overlap_dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--corpus-dir", type=Path, help="Also check four seeded source-text samples per HPLT shard, including positions.")
    parser.add_argument("--sample-pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-fingerprints", type=int, default=1_000_000)
    args = parser.parse_args()
    verify_indexes(
        args.index_dir, args.overlap_dir, report_path=args.report,
        sample_pairs=args.sample_pairs, seed=args.seed,
        batch_fingerprints=args.batch_fingerprints,
        corpus_dir=args.corpus_dir,
    )


if __name__ == "__main__":
    main()
