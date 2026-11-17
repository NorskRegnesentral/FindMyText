"""Build a separate, rare-first, document-only overlap inverted index.

This is not a positional index or an overlap join. Fingerprints are addressed
by rarity rank, never by numeric-hash binary search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from numba import njit

from . import winnower
from .indexing import POSTING_DTYPE, validate_postings_layout

_RECORD = np.dtype(
    [("df", np.uint64), ("fingerprint", np.uint64), ("offset", np.uint64)]
)
_SOURCE_FILES = (
    "meta.json", "fingerprints.npy", "offsets.npy", "lengths.npy",
    "postings.dat", "doc_name_offsets.npy", "doc_name_bytes.npy",
)
_ARRAYS = {
    "fingerprints.npy": np.dtype("uint64"),
    "offsets.npy": np.dtype("uint64"),
    "doc_ids.npy": np.dtype("uint32"),
    "eligible_lengths.npy": np.dtype("uint64"),
}
_ORDER = ["document_frequency", "fingerprint_value"]


def _source_identity(index_dir: Path) -> dict:
    meta_bytes = (index_dir / "meta.json").read_bytes()
    meta = json.loads(meta_bytes)
    winnower.Winnower.from_metadata(meta)
    validate_postings_layout(meta)
    names: list[str] = list(_SOURCE_FILES)
    names.extend(
        name for name in ("forward_offsets.npy", "forward_fingerprints.npy")
        if (index_dir / name).exists()
    )
    files = {}
    for name in names:
        stat = (index_dir / name).stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "path": str(index_dir.resolve()),
        "settings": meta,
        "metadata_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "files": files,
    }


def _load_array(path: Path, dtype: np.dtype) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 1 or array.dtype != dtype:
        raise ValueError(f"{path}: expected a one-dimensional {dtype} array")
    return array


def _close_on_exit(stack: ExitStack, array: np.ndarray) -> None:
    # NFS cannot remove mapped scratch files until their mmap handles are closed.
    if isinstance(array, np.memmap):
        mapping = getattr(array, "_mmap", None)
        if not isinstance(mapping, mmap.mmap):
            raise TypeError("Expected an owned NumPy memory map")
        stack.callback(mapping.close)


@njit(cache=True)
def _batch_stop(lengths, start, batch_entries):
    stop = start
    entries = 0
    while stop < len(lengths):
        length = int(lengths[stop])
        if length == 0:
            raise ValueError("Source contains a zero-document fingerprint")
        if stop > start and entries + length > batch_entries:
            break
        entries += length
        stop += 1
    return stop


@njit(cache=True)
def _scan(fps, offsets, lengths, postings, start, stop, seen, docs,
          records, eligible_lengths, df_histogram, posting_offset):
    used = 0
    retained = 0
    singletons = 0
    unique_pairs = 0
    for rank in range(start, stop):
        if rank and fps[rank] <= fps[rank - 1]:
            raise ValueError("Source fingerprints must be strictly hash-sorted")
        expected = (
            offsets[rank - 1] + np.uint64(lengths[rank - 1]) * 8
            if rank else np.uint64(0)
        )
        if offsets[rank] != expected:
            raise ValueError("Source posting offsets are not contiguous byte offsets")
        first = int(offsets[rank] // 8)
        last = first + int(lengths[rank])
        if last > len(postings):
            raise ValueError("Source posting list extends past postings.dat")
        count = 0
        # Source lists can be unsorted and a hot list can exceed the batch size.
        # Epoch markers bound de-duplication scratch space by the document count.
        epoch = np.uint64(rank) + np.uint64(1)
        for entry in range(first, last):
            doc = postings[entry]["doc_id"]
            if doc >= len(seen):
                raise ValueError("Source posting contains an out-of-range document ID")
            if seen[doc] != epoch:
                seen[doc] = epoch
                docs[used + count] = doc
                count += 1
        unique_pairs += count
        if count == 1:
            singletons += 1
            continue
        docs[used:used + count].sort()
        records[retained]["df"] = count
        records[retained]["fingerprint"] = fps[rank]
        records[retained]["offset"] = posting_offset + used
        for i in range(used, used + count):
            eligible_lengths[docs[i]] += 1
        df_histogram[count] += 1
        used += count
        retained += 1
    return used, retained, singletons, unique_pairs


@njit(cache=True)
def _merge(left, right, output):
    i = 0
    j = 0
    k = 0
    while i < len(left) and j < len(right):
        take_left = (
            left[i]["df"] < right[j]["df"]
            or (
                left[i]["df"] == right[j]["df"]
                and left[i]["fingerprint"] < right[j]["fingerprint"]
            )
        )
        if take_left:
            output[k] = left[i]
            i += 1
        else:
            output[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        output[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        output[k] = right[j]
        j += 1
        k += 1


def _rank_runs(paths: list[Path], scratch: Path, progress) -> Path | None:
    generation = 0
    while len(paths) > 1:
        merged = []
        for i in range(0, len(paths), 2):
            if i + 1 == len(paths):
                merged.append(paths[i])
                continue
            with ExitStack() as mappings:
                left = np.load(paths[i], mmap_mode="r")
                _close_on_exit(mappings, left)
                right = np.load(paths[i + 1], mmap_mode="r")
                _close_on_exit(mappings, right)
                path = scratch / f"merge-{generation}-{i}.npy"
                output = np.lib.format.open_memmap(
                    path, mode="w+", dtype=_RECORD, shape=(len(left) + len(right),)
                )
                _close_on_exit(mappings, output)
                _merge(left, right, output)
                output.flush()
            progress(f"frequency ranking: merged {i + 2}/{len(paths)} runs")
            paths[i].unlink()
            paths[i + 1].unlink()
            merged.append(path)
        paths = merged
        generation += 1
    return paths[0] if paths else None


@njit(cache=True)
def _layout(records, source_docs, fps, offsets, output_docs, batch_entries):
    cursor = 0
    offsets[0] = 0
    for rank in range(len(records)):
        record = records[rank]
        count = int(record["df"])
        start = int(record["offset"])
        fps[rank] = record["fingerprint"]
        for first in range(0, count, batch_entries):
            size = min(batch_entries, count - first)
            output_docs[cursor + first:cursor + first + size] = (
                source_docs[start + first:start + first + size]
            )
        cursor += count
        offsets[rank + 1] = cursor


def _distribution(values: np.ndarray, frequencies: np.ndarray) -> dict:
    count = int(frequencies.sum())
    if not count:
        return {"count": 0, "min": None, "max": None, "mean": None, "quantiles": {}}
    cumulative = np.cumsum(frequencies)
    return {
        "count": count,
        "min": int(values[0]),
        "max": int(values[-1]),
        "mean": float(np.dot(values.astype(np.float64), frequencies) / count),
        "quantiles": {
            str(q): int(values[np.searchsorted(cumulative, max(1, np.ceil(q * count)))])
            for q in (0.5, 0.9, 0.99, 0.999)
        },
    }


def _save_metadata(output: Path, meta: dict) -> None:
    sizes = {name: (output / name).stat().st_size for name in _ARRAYS}
    sizes["lookup_structures"] = 0
    sizes["meta.json"] = 0
    meta["storage_bytes"] = sizes
    while True:
        sizes["total"] = sum(value for key, value in sizes.items() if key != "total")
        data = json.dumps(meta, indent=2).encode("utf-8") + b"\n"
        if sizes["meta.json"] == len(data):
            break
        sizes["meta.json"] = len(data)
    (output / "meta.json").write_bytes(data)


def build_overlap_inverted_index(
    index_dir: str | Path,
    output_dir: str | Path,
    batch_entries: int = 1_000_000,
    verbose: bool = True,
) -> Path:
    """Derive ``df >= 2`` postings without modifying the source.

    Working arrays are O(batch_entries + source document count), including for
    individual posting lists larger than the batch target. Sorting uses disk
    runs and two-way compiled merges, not an in-memory global permutation.
    Source and output memory maps also consume OS-managed resident pages.
    Existing destinations are never overwritten; publication is atomic.
    """
    if batch_entries < 1:
        raise ValueError("batch_entries must be positive")
    source = Path(index_dir).resolve()
    destination = Path(output_dir).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("The overlap inverted index must be outside the source index")
    if destination.exists():
        raise FileExistsError(f"Overlap inverted destination already exists: {destination}")
    identity = _source_identity(source)
    fps = _load_array(source / "fingerprints.npy", np.dtype("uint64"))
    offsets = _load_array(source / "offsets.npy", np.dtype("uint64"))
    lengths = _load_array(source / "lengths.npy", np.dtype("uint32"))
    names = _load_array(source / "doc_name_offsets.npy", np.dtype("uint64"))
    if not len(names) or len(fps) != len(offsets) or len(fps) != len(lengths):
        raise ValueError("Invalid source array lengths")
    n_docs = len(names) - 1
    if n_docs > 2**32:
        raise ValueError("Source document IDs exceed uint32 capacity")
    posting_bytes = (source / "postings.dat").stat().st_size
    if posting_bytes % POSTING_DTYPE.itemsize:
        raise ValueError("Source postings.dat has an incomplete positional record")
    postings = (
        np.memmap(source / "postings.dat", mode="r", dtype=POSTING_DTYPE)
        if posting_bytes else np.empty(0, dtype=POSTING_DTYPE)
    )
    seen = np.zeros(n_docs, dtype=np.uint64)
    docs = np.empty(max(batch_entries, n_docs), dtype=np.uint32)
    eligible = np.zeros(n_docs, dtype=np.uint64)
    df_histogram = np.zeros(n_docs + 1, dtype=np.uint64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    temporary_peak = 0
    phase_seconds = {}

    with (
        tempfile.TemporaryDirectory(prefix=".overlap-inverted-", dir=destination.parent) as tmp,
        ExitStack() as mappings,
    ):
        scratch = Path(tmp)
        output = scratch / "output"
        output.mkdir()

        def progress(message: str) -> None:
            nonlocal temporary_peak
            temporary_peak = max(
                temporary_peak,
                sum(path.stat().st_size for path in scratch.rglob("*") if path.is_file()),
            )
            if verbose:
                print(f"[overlap inverted] {message}", flush=True)

        progress("scanning/de-duplication")
        singletons = unique_pairs = retained = doc_postings = 0
        paths = []
        start = 0
        spool = scratch / "documents.dat"
        with spool.open("wb") as stream:
            while start < len(fps):
                stop = _batch_stop(lengths, start, batch_entries)
                records = np.empty(stop - start, dtype=_RECORD)
                used, kept, removed, unique = _scan(
                    fps, offsets, lengths, postings, start, stop, seen, docs,
                    records, eligible, df_histogram, doc_postings,
                )
                docs[:used].tofile(stream)
                if kept:
                    records = records[:kept]
                    records.sort(order=["df", "fingerprint"])
                    path = scratch / f"run-{start}.npy"
                    np.save(path, records)
                    paths.append(path)
                retained += kept
                singletons += removed
                unique_pairs += unique
                doc_postings += used
                start = stop
                progress(f"scanning: {start:,}/{len(fps):,} fingerprints")
        expected_bytes = (
            int(offsets[-1]) + int(lengths[-1]) * POSTING_DTYPE.itemsize
            if len(fps) else 0
        )
        if expected_bytes != posting_bytes:
            raise ValueError("Source postings.dat size does not match its posting lists")
        if retained != len(fps) - singletons or doc_postings != unique_pairs - singletons:
            raise ValueError("Overlap inverted de-duplication accounting mismatch")
        phase_seconds["scanning"] = time.perf_counter() - started
        progress("frequency ranking")
        phase_start = time.perf_counter()
        ranked_path = _rank_runs(paths, scratch, progress)
        ranked = (
            np.load(ranked_path, mmap_mode="r")
            if ranked_path else np.empty(0, dtype=_RECORD)
        )
        _close_on_exit(mappings, ranked)
        phase_seconds["ranking"] = time.perf_counter() - phase_start
        progress("posting-list layout/write")
        phase_start = time.perf_counter()
        out_fps = np.lib.format.open_memmap(
            output / "fingerprints.npy", mode="w+", dtype=np.uint64, shape=(retained,)
        )
        _close_on_exit(mappings, out_fps)
        out_offsets = np.lib.format.open_memmap(
            output / "offsets.npy", mode="w+", dtype=np.uint64, shape=(retained + 1,)
        )
        _close_on_exit(mappings, out_offsets)
        out_docs = np.lib.format.open_memmap(
            output / "doc_ids.npy", mode="w+", dtype=np.uint32, shape=(doc_postings,)
        )
        _close_on_exit(mappings, out_docs)
        source_docs = (
            np.memmap(spool, mode="r", dtype=np.uint32)
            if doc_postings else np.empty(0, dtype=np.uint32)
        )
        _close_on_exit(mappings, source_docs)
        _layout(ranked, source_docs, out_fps, out_offsets, out_docs, batch_entries)
        for array in (out_fps, out_offsets, out_docs):
            array.flush()
        np.save(output / "eligible_lengths.npy", eligible)
        phase_seconds["layout"] = time.perf_counter() - phase_start
        progress(f"posting-list layout/write: {retained:,} lists written")
        if _source_identity(source) != identity:
            raise ValueError("Source index changed during the overlap inverted build")
        eligible_values, eligible_frequencies = np.unique(eligible, return_counts=True)
        df_values = np.flatnonzero(df_histogram)
        statistics = {
            "input_fingerprints": len(fps),
            "singleton_fingerprints": singletons,
            "singleton_percentage": 100 * singletons / len(fps) if len(fps) else 0,
            "retained_fingerprints": retained,
            "retained_percentage": 100 * retained / len(fps) if len(fps) else 0,
            "zero_document_fingerprints": 0,
            "input_positional_postings": posting_bytes // POSTING_DTYPE.itemsize,
            "unique_doc_fingerprint_pairs": unique_pairs,
            "collapsed_repeats": posting_bytes // POSTING_DTYPE.itemsize - unique_pairs,
            "singleton_pairs_removed": singletons,
            "retained_doc_id_postings": doc_postings,
            "documents_without_retained_fingerprints": int(np.count_nonzero(eligible == 0)),
            "eligible_lengths": _distribution(eligible_values, eligible_frequencies),
            "retained_document_frequency": _distribution(df_values, df_histogram[df_values]),
        }
        meta = {
            "format": "findmytext-overlap-inverted-v1",
            "source": identity,
            "document_count": n_docs,
            "minimum_document_frequency": 2,
            "fingerprint_order": _ORDER,
            "document_order": "ascending_source_document_id",
            "offset_unit": "document_id_entries",
            "batch_entries": batch_entries,
            "statistics": statistics,
            "phase_seconds": phase_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_temporary_bytes": temporary_peak,
        }
        if sys.platform == "linux":
            import resource

            meta["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        _save_metadata(output, meta)
        os.rename(output, destination)
        progress(
            f"complete: {retained:,} fingerprints, {doc_postings:,} document-ID postings; "
            f"removed {singletons:,} singletons ({statistics['singleton_percentage']:.2f}%), "
            f"collapsed {statistics['collapsed_repeats']:,} repeated occurrences; "
            f"{meta['storage_bytes']['total']:,} bytes, {meta['elapsed_seconds']:.2f}s. "
            f"Statistics: {destination / 'meta.json'}"
        )
        if verbose:
            print(json.dumps(meta, indent=2), flush=True)
    return destination


class OverlapInvertedIndex:
    """Memory-mapped rank-based access, explicitly bound to its positional source."""

    def __init__(self, overlap_dir: str | Path, source_index_dir: str | Path):
        root = Path(overlap_dir)
        self.meta = json.loads((root / "meta.json").read_text())
        if (
            self.meta.get("format") != "findmytext-overlap-inverted-v1"
            or self.meta.get("minimum_document_frequency") != 2
            or self.meta.get("fingerprint_order") != _ORDER
            or self.meta.get("document_order") != "ascending_source_document_id"
            or self.meta.get("offset_unit") != "document_id_entries"
        ):
            raise ValueError("Unsupported overlap inverted index format or ordering")
        if self.meta["source"] != _source_identity(Path(source_index_dir).resolve()):
            raise ValueError("Overlap inverted index does not match this source index")
        self.fingerprints = _load_array(root / "fingerprints.npy", _ARRAYS["fingerprints.npy"])
        self.offsets = _load_array(root / "offsets.npy", _ARRAYS["offsets.npy"])
        self.doc_ids = _load_array(root / "doc_ids.npy", _ARRAYS["doc_ids.npy"])
        self.eligible_lengths = _load_array(root / "eligible_lengths.npy", _ARRAYS["eligible_lengths.npy"])
        if (
            len(self.offsets) != len(self.fingerprints) + 1
            or len(self.eligible_lengths) != self.meta["document_count"]
            or self.offsets[0] != 0
            or int(self.offsets[-1]) != len(self.doc_ids)
        ):
            raise ValueError("Invalid overlap inverted index array lengths or offsets")

    def get_postings(self, rank: int) -> np.ndarray:
        """Return the ascending unique source document IDs at a rarity rank."""
        if rank < 0 or rank >= len(self.fingerprints):
            raise IndexError("Overlap inverted fingerprint rank is out of range")
        return self.doc_ids[int(self.offsets[rank]):int(self.offsets[rank + 1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--batch-entries", type=int, default=1_000_000)
    args = parser.parse_args()
    build_overlap_inverted_index(args.index_dir, args.output_dir, args.batch_entries)


if __name__ == "__main__":
    main()
