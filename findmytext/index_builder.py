"""Scripts for building and merging document fingerprint indexes.

Indexing pipeline
-----------------
build_index() turns one corpus file into one index in a single call, and merge_indexes()
combines several indexes into one.  Under the hood this is a two-step process, which
can also be driven separately (e.g. one job per corpus file on a cluster, then one merge):

Step 1 — Build intermediate shards  (Indexer.index_file)
    Each document is tokenised and run through the winnowing algorithm to produce a
    set of (fingerprint, position) pairs.  The pairs are accumulated in a PostingBuffer
    in the main process and periodically flushed as a shard directory in the same
    on-disk format as the final index (see indexing.py), so every shard is itself a
    valid DiskBasedIndex.  Parallel extraction keeps a bounded number of batches in flight.

Step 2 — Merge into a single disk-based index  (Merger.merge)
    The shards are merged by splitting the 64-bit fingerprint space into ranges holding
    about the same number of postings (winnowing keeps the smallest fingerprint of each
    window, so fingerprints are far from uniform) and processing each range in a worker
    process: the postings of a range are a contiguous slice of every shard, so they are
    gathered with a binary search and one read per shard, sorted, and written in place
    into the preallocated final postings file.  Only the small per-fingerprint arrays
    are concatenated afterwards.

"""

import argparse
import multiprocessing as mp
import os
import shutil
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import closing
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import orjson
import psutil

from . import indexing, utils, winnower


def build_index(
    corpus_file: str,
    index_dir: str,
    length: int = 4,
    window_size: int = 6,
    n_workers: Optional[int] = None,
    max_memory_gb: Optional[float] = None,
    stop_after: Optional[int] = None,
    delete_existing: bool = False,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> indexing.DiskBasedIndex:
    """Index one corpus file into a single disk-based index and open it.

    This runs both steps of the pipeline: the shards are written to index_dir/shards,
    merged into index_dir and deleted.  index_dir is complete once it holds a
    meta.json; if a build is interrupted during the merge, the shards are left in
    place and ``merge_indexes(index_dir + "/shards", index_dir)`` resumes it.

    Args:
        corpus_file: Path to the corpus (.jsonl, .jsonl.gz or .jsonl.zst); one JSON
            object per line with a unique "id" and a "text".
        index_dir: Directory of the resulting index.
        length: k-gram length for fingerprinting (default 4).
        window_size: Winnowing window size (default 6).
        n_workers: Number of processes, for fingerprinting and for merging
            (default: the CPUs available to this process).
        max_memory_gb: Memory budget, for the posting buffer and the merge workers
            (default: 80% of the memory currently available; set it explicitly
            inside a SLURM job or a container, whose limits are not detected).
        stop_after: If set, index only the first stop_after documents.
        delete_existing: If True, an existing index_dir is deleted first; otherwise a
            directory already holding an index raises an error.
        min_length: If set, only index documents with at least this many characters.
        max_length: If set, only index documents with at most this many characters.

    Returns:
        The opened index, ready to be passed to a detector.
    """
    if os.path.exists(index_dir):
        if delete_existing:
            shutil.rmtree(index_dir)
        elif os.path.isfile(os.path.join(index_dir, "meta.json")):
            raise ValueError(f"{index_dir} already holds an index; pass delete_existing=True to rebuild it")
    shards_dir = os.path.join(index_dir, "shards")

    indexer = Indexer(length, window_size, n_workers, max_memory_gb)
    shards = indexer.index_file(corpus_file, shards_dir, stop_after, min_length=min_length, max_length=max_length)
    if len(shards) == 1:  # a single shard already is the final index
        for name in os.listdir(shards[0]):
            os.replace(os.path.join(shards[0], name), os.path.join(index_dir, name))
    else:
        Merger(n_workers, max_memory_gb).merge(shards, index_dir)
    shutil.rmtree(shards_dir)
    return indexing.DiskBasedIndex(index_dir)


def merge_indexes(
    sources: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    output_dir: Union[str, os.PathLike],
    n_workers: Optional[int] = None,
    max_memory_gb: Optional[float] = None,
    delete_existing: bool = False,
) -> indexing.DiskBasedIndex:
    """Merge several indexes into a single disk-based index and open it.

    Args:
        sources: Either a list of index directories (shards written by
            Indexer.index_file, outputs of build_index or of a previous merge), or a
            single directory whose subdirectories holding a meta.json are merged, in
            sorted name order.
        output_dir: Directory of the merged index.
        n_workers: Number of processes merging fingerprint ranges in parallel
            (default: the CPUs available to this process).
        max_memory_gb: Memory budget shared by the workers (default: 80% of the
            memory currently available; set it explicitly inside a SLURM job or a
            container, whose limits are not detected).
        delete_existing: If True, an existing output_dir is deleted first; otherwise a
            directory already holding an index raises an error.

    Returns:
        The opened index, ready to be passed to a detector.  Document IDs are
        assumed to be unique across the inputs: no deduplication is performed.
    """
    if isinstance(sources, (str, os.PathLike)):
        index_dirs = _find_indexes(os.fspath(sources))
    else:
        index_dirs = [os.fspath(d) for d in sources]
    output_dir = os.fspath(output_dir)
    for index_dir in index_dirs:
        if not os.path.isfile(os.path.join(index_dir, "meta.json")):
            raise ValueError(f"{index_dir} does not hold an index (no meta.json)")
    if os.path.exists(output_dir):
        if delete_existing:
            shutil.rmtree(output_dir)
        elif os.path.isfile(os.path.join(output_dir, "meta.json")):
            raise ValueError(f"{output_dir} already holds an index; pass delete_existing=True to replace it")
    Merger(n_workers, max_memory_gb).merge(index_dirs, output_dir)
    return indexing.DiskBasedIndex(output_dir)


def _find_indexes(parent_dir: str) -> List[str]:
    """Subdirectories of parent_dir holding a meta.json, in sorted name order."""
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Provided path {parent_dir} is not a directory")
    index_dirs = [
        os.path.join(parent_dir, name)
        for name in sorted(os.listdir(parent_dir))
        if os.path.isfile(os.path.join(parent_dir, name, "meta.json"))
    ]
    if not index_dirs:
        raise ValueError(f"No index shards found in {parent_dir}")
    return index_dirs


#####################################################
# STEP 1: BUILDING OF INDEX SHARDS FROM CORPUS
#####################################################


class Indexer:
    """Builds index shards from corpus files with fixed fingerprinting settings and
    resource limits.

    Args:
        length: k-gram length for fingerprinting (default 4).
        window_size: Winnowing window size (default 6).
        n_workers: Process budget, including the parent that reads and inserts.
            Values above one use n_workers - 1 fingerprint workers; n_workers=1
            fingerprints directly in the parent (default: the CPUs available to
            this process).
        max_memory_gb: Memory budget for the posting buffer; a shard is written
            whenever the buffer would exceed it (default: 80% of the memory
            currently available, see _resolve_resources).
    """

    def __init__(
        self,
        length: int = 4,
        window_size: int = 6,
        n_workers: Optional[int] = None,
        max_memory_gb: Optional[float] = None,
    ):
        if n_workers is None:
            process = psutil.Process()
            n_workers = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else psutil.cpu_count()
        if max_memory_gb is None:
            max_memory_gb = 0.8 * psutil.virtual_memory().available / 2**30
        
        self.n_workers, self.max_memory_gb = n_workers, max_memory_gb
        if self.n_workers < 1 or self.max_memory_gb <= 0:
            raise ValueError("n_workers and max_memory_gb must be positive")
  
        self.winnower = winnower.Winnower(length=length, window_size=window_size)
        self.max_postings = int(self.max_memory_gb * 2**30 / PostingBuffer.PEAK_BYTES_PER_POSTING)

    def index_file(
        self,
        corpus_file: str,
        output_dir: str,
        stop_after: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[str]:
        """Index documents from a corpus file and save index shards to output_dir.

        Reads the corpus, computes winnowed fingerprints for every document, and
        periodically flushes the accumulated postings as a shard directory
        (output_dir/shard-NNNNNN).  Each shard is a complete DiskBasedIndex; shards
        can then be merged into a single index with Merger.merge().

        Args:
            corpus_file: Path to the corpus (.jsonl, .jsonl.gz, or .jsonl.zst).  Each
                line must be a JSON object with at least an "id" field (unique document
                identifier) and a "text" field (document content).
            output_dir: Directory where the shard directories will be created.
            stop_after: If set, stop after this many documents (useful for testing).
            min_length: If set, only index documents with at least this many characters.
            max_length: If set, only index documents with at most this many characters.

        Returns:
            List of paths to the saved shard directories.
        """
        if not os.path.isfile(corpus_file):
            raise ValueError(f"Provided file {corpus_file} does not exist or is not a file")
        if stop_after is not None and stop_after < 0:
            raise ValueError("stop_after must be nonnegative")

        if corpus_file.endswith((".jsonl", ".jsonl.gz")):
            stream = utils.stream_jsonl(corpus_file, min_length=min_length, max_length=max_length)
        elif corpus_file.endswith(".jsonl.zst"):
            stream = utils.stream_json_zst(corpus_file, min_length=min_length, max_length=max_length)
        else:
            raise ValueError(
                "Unsupported file format. Please provide a .jsonl, .jsonl.gz, or .jsonl.zst file."
            )
        print("Indexing documents from", corpus_file)

        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"Output path {output_dir} exists and is not a directory")
        os.makedirs(output_dir, exist_ok=True)
        print("Index shards will be saved to", output_dir)
        print(
            f"Indexing with {self.n_workers} processes, flushing every "
            f"{self.max_postings:,} postings...", flush=True,
        )

        meta = self.winnower.to_metadata()
        buffer = PostingBuffer(meta=meta)
        shard_dirs: List[str] = []

        def flush():
            path = os.path.join(output_dir, f"shard-{len(shard_dirs):06d}")
            buffer.save(path)
            shard_dirs.append(path)

        batch_results = self._get_batched_fingerprints(stream, stop_after)
        total_count = 0
        with closing(stream), closing(batch_results):
            for batch_result in batch_results:
                for doc_id, fp_pos_list in batch_result.items():
                    buffer.add_fingerprints(doc_id, fp_pos_list)
                    total_count += 1

                    if total_count % 100_000 == 0:
                        print(f"Processed {total_count:,} documents... (nb of postings: {buffer.index_size:,})", flush=True)

                    if buffer.index_size >= self.max_postings:
                        flush()
                        buffer = PostingBuffer(meta=meta)

        if buffer.doc_count > 0:
            flush()

        return shard_dirs

    def _get_batched_fingerprints(
        self, stream: Any, stop_after: Optional[int]
    ) -> Iterator[Dict[str, List[Tuple[int, int]]]]:
        """Yield batches of {doc_id: [(fingerprint, position), ...]} from the stream.

        With more than one process, the fingerprinting runs in a pool of worker
        processes while the parent keeps reading; a bounded number of batches is in
        flight at any time.
        """
        batch_winnower = self.winnower.get_winnowed_fingerprints_batch
        batches = utils.generate_batches(stream, stop_after)
        if self.n_workers == 1:
            for batch in batches:
                yield batch_winnower(batch)
            return
        n_extractors = self.n_workers - 1
        # Spawned workers (not forked: the parent may hold numpy/polars threads).  The
        # corpus generator stays in the parent, since it cannot be pickled.
        pool = ProcessPoolExecutor(n_extractors, mp_context=mp.get_context("spawn"))
        try:
            pending: deque = deque()
            for batch in batches:
                if len(pending) >= n_extractors:
                    yield pending.popleft().result()
                pending.append(pool.submit(batch_winnower, batch))
            while pending:
                yield pending.popleft().result()
        except BrokenProcessPool as e:
            raise RuntimeError(
                "A fingerprinting worker died unexpectedly (most likely killed by the kernel for "
                "running out of memory): lower n_workers or raise the job's memory limit."
            ) from e
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


class PostingBuffer:
    """Accumulates (fingerprint, doc_id, position) triples in a flat numpy structured
    array (16 bytes per entry) and writes them out as an index shard directory.

    Optimised for fast insertion; querying is left to DiskBasedIndex, which can open
    the saved shard directly.
    """

    _DTYPE = np.dtype([("fp", np.uint64), ("doc", np.uint32), ("pos", np.uint32)])
    _CHUNK = 10_000_000  # grow in 10 M-entry (~160 MB) steps
    # Peak heap bytes per buffered posting, reached in save(): 16 B entry + 8 B sort
    # index + 16 B sorted entries + 8 B postings, rounded up for the doc-id table and
    # the transient copy made when the buffer grows.
    PEAK_BYTES_PER_POSTING = 56

    def __init__(self, meta: Dict[str, Any]):
        self.meta = winnower.Winnower.from_metadata(meta).to_metadata()
        self.doc_id_map: Dict[str, int] = {}
        self.to_external: List[str] = []
        self.index_array: np.ndarray = np.empty(self._CHUNK, dtype=self._DTYPE)
        self.index_size: int = 0  # number of valid entries in index_array

    def add_fingerprints(self, doc_id: str, fp_pos_list: List[Tuple[int, int]]):
        """Append fingerprint-position pairs for a document to the buffer."""
        int_id = self.doc_id_map.get(doc_id)
        if int_id is None:
            int_id = len(self.to_external)
            self.doc_id_map[doc_id] = int_id
            self.to_external.append(doc_id)

        n = len(fp_pos_list)
        if self.index_size + n > len(self.index_array):
            extra = max(self._CHUNK, n)
            new_buf = np.empty(self.index_size + extra, dtype=self._DTYPE)
            new_buf[: self.index_size] = self.index_array[: self.index_size]
            self.index_array = new_buf  # old buffer freed when refcount drops to 0

        # Slice-assign the whole document at once instead of one Python tuple per entry.
        pairs = np.asarray(fp_pos_list, dtype=np.uint64).reshape(-1, 2)
        end = self.index_size + n
        chunk = self.index_array[self.index_size : end]
        chunk["fp"] = pairs[:, 0]
        chunk["doc"] = int_id
        chunk["pos"] = pairs[:, 1]
        self.index_size = end

    @property
    def doc_count(self) -> int:
        """Return the number of unique documents buffered so far."""
        return len(self.to_external)

    def save(self, index_dir: str) -> None:
        """Write the buffered postings as an index shard directory."""
        print(f"Exporting shard to {index_dir}...", end="", flush=True)

        os.makedirs(index_dir, exist_ok=True)
        self._write_doc_names(index_dir)
        self._write_postings(index_dir)
        # meta.json is written last: its presence marks a complete index directory.
        with open(os.path.join(index_dir, "meta.json"), "wb") as f:
            f.write(orjson.dumps(self.meta))
        print("Done.")

    def _write_doc_names(self, index_dir: str) -> None:
        """Write the internal-id -> external document name table.

        Names are UTF-8 encoded and concatenated into doc_name_bytes.npy; the byte
        range of document i is doc_name_offsets[i]:doc_name_offsets[i + 1].
        """
        encoded = [str(name).encode("utf-8") for name in self.to_external]
        sizes = np.fromiter((len(b) for b in encoded), dtype=np.uint64, count=len(encoded))
        offsets = np.zeros(len(encoded) + 1, dtype=np.uint64)
        np.cumsum(sizes, out=offsets[1:])
        flat = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        np.save(os.path.join(index_dir, "doc_name_offsets.npy"), offsets)
        np.save(os.path.join(index_dir, "doc_name_bytes.npy"), flat)

    def _write_postings(self, index_dir: str) -> int:
        """Write fingerprints.npy (+ fingerprint_blocks.npy), offsets.npy, lengths.npy
        and postings.dat.

        Returns:
            The number of unique fingerprints written.
        """
        view = self.index_array[: self.index_size]

        # Stable sort keeps insertion order within a fingerprint, so output is
        # deterministic.  One gather of whole entries, then split into the two outputs.
        entries = view[np.argsort(view["fp"], kind="stable")]
        fingerprints = entries["fp"]
        postings = np.empty(len(entries), dtype=indexing.POSTING_DTYPE)
        postings["doc_id"] = entries["doc"]
        postings["position"] = entries["pos"]

        # Group boundaries in one vectorised pass; a group starts where the value changes.
        if len(fingerprints):
            changes = np.flatnonzero(fingerprints[1:] != fingerprints[:-1]) + 1
            starts = np.concatenate(([0], changes))
            ends = np.append(starts[1:], len(fingerprints))
        else:
            starts = ends = np.zeros(0, dtype=np.int64)

        np.save(os.path.join(index_dir, "fingerprints.npy"), fingerprints[starts])
        np.save(
            os.path.join(index_dir, indexing.FINGERPRINT_BLOCKS_FILE),
            indexing.compute_fingerprint_blocks(fingerprints[starts]),
        )
        np.save(
            os.path.join(index_dir, "offsets.npy"),
            starts.astype(np.uint64) * np.uint64(indexing.POSTING_DTYPE.itemsize),
        )
        np.save(os.path.join(index_dir, "lengths.npy"), (ends - starts).astype(np.uint32))
        with open(os.path.join(index_dir, "postings.dat"), "wb") as f:
            postings.tofile(f)
            f.flush()
            os.fsync(f.fileno())
            _release_page_cache(f, 0, postings.nbytes)
        return len(starts)

#####################################################
# STEP 2: MERGE SHARDS INTO A SINGLE DISK-BASED INDEX
#####################################################
#
# Every shard is itself a complete index whose postings are sorted by fingerprint.
# Merging therefore reduces to:
#
#   1. Cut the 64-bit fingerprint space into n_parts ranges holding about the same
#      number of postings (Merger._partition_bounds).  Fingerprints are not uniform:
#      winnowing keeps the *smallest* fingerprint of each window, so the stored values
#      crowd the low end of the space (for window size 6, the lowest 1/32 of the space
#      holds ~11% of the postings).
#   2. For each range (one job in a process pool): in every shard, locate the range with
#      a binary search on fingerprints.npy and read the corresponding contiguous slice
#      of postings.dat, shifting the doc ids by that shard's offset in the merged doc
#      table.  Sort the gathered postings by fingerprint (stable, so that the shards'
#      order is kept for equal fingerprints).
#   3. Write the sorted postings directly into the final postings.dat, which the parent
#      has preallocated.  Because the merged file is ordered by fingerprint, the range
#      starts at byte Σ_shards (bytes of postings with fingerprint < lo), which each
#      worker computes itself from the same binary searches.  The per-fingerprint
#      arrays (fingerprints, offsets, lengths) are small and written as part files
#      that the parent concatenates in range order.
#
# Each posting is read and written exactly once.  The number of ranges is derived from
# the memory budget so that the postings of one range fit comfortably in RAM.

# Peak heap bytes per posting of a range inside a worker: the raw postings (8), the
# fingerprint of every posting (8), the sort order (8) and the sorted postings (8), plus
# the sort's temporaries.  The value below leaves headroom for the interpreter and for
# ranges that come out larger than planned.  The page cache of the streamed postings is
# released as we go (_release_page_cache), since it would otherwise count against the
# memory limit of a SLURM job.
_BYTES_PER_POSTING = 80


class Merger:
    """Merges index directories into one, with fixed resource limits.

    Args:
        n_workers: Number of worker processes merging fingerprint ranges in parallel.
            Each worker alternates between reading, sorting and writing, so 1.5 workers
            per CPU keeps both the CPUs and the filesystem busy (default: the CPUs
            available to this process).
        max_memory_gb: Memory budget shared by the workers.  It only determines how
            many fingerprint ranges the merge is split into; a smaller budget means
            more, smaller ranges (default: 80% of the memory currently available,
            see _resolve_resources).
    """

    def __init__(self, n_workers: Optional[int] = None, max_memory_gb: Optional[float] = None):

        if n_workers is None:
            process = psutil.Process()
            n_workers = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else psutil.cpu_count()
        if max_memory_gb is None:
            max_memory_gb = 0.8 * psutil.virtual_memory().available / 2**30
        
        self.n_workers, self.max_memory_gb = n_workers, max_memory_gb
        if self.n_workers < 1 or self.max_memory_gb <= 0:
            raise ValueError("n_workers and max_memory_gb must be positive")


    def merge_dir(self, shards_dir: str, output_dir: str) -> None:
        """Merge every index shard found in shards_dir (subdirectories containing a
        meta.json, in sorted order) into a single index in output_dir."""
        self.merge(_find_indexes(shards_dir), output_dir)

    def merge(self, index_dirs: List[str], output_dir: str) -> None:
        """Merge several index directories into one.

        The inputs can be shards produced by Indexer.index_file() or any DiskBasedIndex
        directory, including the output of a previous merge.  Document IDs are assumed
        to be unique across the inputs: no deduplication is performed.
        """
        if not index_dirs:
            raise ValueError("At least one index directory must be provided")
        print(f"Merging {len(index_dirs)} indexes into {output_dir}")

        meta = _common_meta(index_dirs)
        os.makedirs(output_dir, exist_ok=True)
        doc_bases = _merge_doc_names(index_dirs, output_dir)

        total_postings = sum(
            os.path.getsize(os.path.join(d, "postings.dat")) for d in index_dirs
        ) // indexing.POSTING_DTYPE.itemsize
        bounds = self._partition_bounds(index_dirs, total_postings)
        postings_path = os.path.join(output_dir, "postings.dat")
        with open(postings_path, "wb") as f:  # preallocated; workers fill disjoint byte ranges
            f.truncate(total_postings * indexing.POSTING_DTYPE.itemsize)
        parts_dir = os.path.join(output_dir, "_parts")
        os.makedirs(parts_dir, exist_ok=True)
        tasks = [
            (index_dirs, doc_bases, lo, hi, postings_path, os.path.join(parts_dir, f"{k:05d}"))
            for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]))
        ]

        # Workers write the postings in place and the small per-fingerprint arrays of
        # their range as part files; only counts travel back through the pool.
        print(f"Merging {len(tasks)} fingerprint ranges with {self.n_workers} workers...", flush=True)
        n_postings, n_fingerprints = 0, 0
        progress = _Progress(len(tasks), "ranges")
        # A ProcessPoolExecutor (unlike multiprocessing.Pool) fails every pending task
        # when a worker is killed, e.g. by the OOM killer of a SLURM job, instead of
        # silently replacing the worker and waiting forever for the lost task.
        pool = ProcessPoolExecutor(self.n_workers, mp_context=mp.get_context("spawn"))
        try:
            pending = {pool.submit(_merge_partition, task) for task in tasks}
            while pending:
                done, pending = wait(pending, timeout=progress.seconds_to_next_report(), return_when=FIRST_COMPLETED)
                for future in done:
                    n_fps, n_part = future.result()  # re-raises a worker's exception
                    n_postings += n_part
                    n_fingerprints += n_fps
                progress.update(len(done), f"{n_postings:,} postings")
        except BrokenProcessPool as e:
            raise RuntimeError(
                "A merge worker died unexpectedly (most likely killed by the kernel for "
                "running out of memory): lower n_workers or raise the job's memory limit."
            ) from e
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if n_postings != total_postings:
            raise RuntimeError(
                f"Merged {n_postings:,} postings but the inputs hold {total_postings:,}"
            )

        # Ranges are ordered, so the parts concatenated in range order are the final arrays.
        for name, dtype in (("fingerprints", np.uint64), ("offsets", np.uint64), ("lengths", np.uint32)):
            _concatenate_parts(
                [f"{task[-1]}.{name}" for task in tasks],
                os.path.join(output_dir, f"{name}.npy"), dtype, n_fingerprints,
            )
        indexing.write_fingerprint_blocks(output_dir)
        shutil.rmtree(parts_dir)
        # meta.json is written last: its presence marks a complete index directory.
        with open(os.path.join(output_dir, "meta.json"), "wb") as f:
            f.write(orjson.dumps(meta))
        print(f"Merged index: {n_fingerprints:,} unique fingerprints, {n_postings:,} postings. Done.")

    def _partition_bounds(self, index_dirs: List[str], total_postings: int) -> List[int]:
        """Split the 64-bit fingerprint space into ranges of about equal posting counts.

        The number of ranges is the larger of n_workers (so every worker has a job)
        and the count needed for one range's postings to fit the per-worker share of
        the memory budget.  The bounds are approximate posting-count quantiles: every
        input contributes the fingerprints found at evenly spaced posting ranks (100 per
        range), each standing for that many postings; the bounds are read off the
        sorted, weighted samples.  Returns strictly increasing boundaries from 0 to
        2**64, normally n_parts + 1 of them.
        """
        postings_per_part = self.max_memory_gb * 2**30 / self.n_workers / _BYTES_PER_POSTING
        n_parts = max(self.n_workers, -(-total_postings // int(max(postings_per_part, 1))))
        samples, weights = [], []
        for index_dir in index_dirs:
            fingerprints = np.load(os.path.join(index_dir, "fingerprints.npy"), mmap_mode="r")
            offsets = np.load(os.path.join(index_dir, "offsets.npy"), mmap_mode="r")
            n = os.path.getsize(os.path.join(index_dir, "postings.dat")) // indexing.POSTING_DTYPE.itemsize
            if n:  # fingerprint of the posting at each sampled rank
                ranks = np.linspace(0, n, 100 * n_parts, endpoint=False).astype(np.uint64)
                at = np.searchsorted(offsets, ranks * np.uint64(indexing.POSTING_DTYPE.itemsize), side="right") - 1
                samples.append(fingerprints[at])
                weights.append(np.full(len(at), n / len(at)))
        if not samples:
            return [0, 2**64]
        samples, weights = np.concatenate(samples), np.concatenate(weights)
        order = np.argsort(samples)
        cumulative = np.cumsum(weights[order])
        targets = np.arange(1, n_parts) * (total_postings / n_parts)
        cuts = samples[order][np.minimum(np.searchsorted(cumulative, targets), len(order) - 1)]
        bounds = [0, *sorted(set(cuts.tolist()) - {0}), 2**64]
        print(
            f"Splitting {total_postings:,} postings into {len(bounds) - 1} fingerprint ranges "
            f"(~{total_postings // n_parts:,} postings, ~{total_postings / n_parts * _BYTES_PER_POSTING / 2**30:.1f} GB per worker)"
        )
        return bounds


class _Progress:
    """Timestamped progress lines for a pool of parallel tasks, at most one a minute."""

    def __init__(self, total: int, unit: str, every: float = 60.0):
        self.total, self.unit, self.every = total, unit, every
        self.done = 0
        self.start = self.last_print = time.monotonic()

    def seconds_to_next_report(self) -> float:
        return max(0.0, self.every - (time.monotonic() - self.last_print))

    def update(self, n_done: int, extra: str = "") -> None:
        """Record finished items; print when the interval has elapsed or all are done."""
        self.done += n_done
        if self.seconds_to_next_report() == 0 or self.done == self.total:
            self.last_print = time.monotonic()
            elapsed = int(self.last_print - self.start)
            print(
                f"[{time.strftime('%H:%M:%S')}] {self.done}/{self.total} {self.unit} done, "
                f"elapsed {elapsed // 3600}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}"
                + (f", {extra}" if extra else ""),
                flush=True,
            )


def _common_meta(index_dirs: List[str]) -> Dict[str, Any]:
    """Return the fingerprint parameters shared by all inputs, raising if they differ."""
    meta = None
    for index_dir in index_dirs:
        with open(os.path.join(index_dir, "meta.json"), "rb") as f:
            source_meta = winnower.Winnower.from_metadata(orjson.loads(f.read())).to_metadata()
        if meta is not None and source_meta != meta:
            raise ValueError(f"Incompatible fingerprint metadata in {index_dir}")
        meta = source_meta
    return meta


def _merge_doc_names(index_dirs: List[str], output_dir: str) -> List[int]:
    """Concatenate the document name tables of all inputs into output_dir.

    Returns, for each input, the offset to add to its internal doc ids so that they
    refer to the merged table.
    """
    doc_bases, offset_parts, byte_parts = [], [], []
    n_docs, n_bytes = 0, 0
    for index_dir in index_dirs:
        offsets = np.load(os.path.join(index_dir, "doc_name_offsets.npy"), mmap_mode="r")
        names = np.load(os.path.join(index_dir, "doc_name_bytes.npy"), mmap_mode="r")
        doc_bases.append(n_docs)
        offset_parts.append(offsets[:-1] + np.uint64(n_bytes))
        byte_parts.append(names)
        n_docs += len(offsets) - 1
        n_bytes += len(names)
    if n_docs >= 2**32:
        raise ValueError("Merged index would exceed the 32-bit document id limit")
    offset_parts.append(np.array([n_bytes], dtype=np.uint64))
    np.save(os.path.join(output_dir, "doc_name_offsets.npy"), np.concatenate(offset_parts))
    np.save(os.path.join(output_dir, "doc_name_bytes.npy"), np.concatenate(byte_parts))
    print(f"Merged document table: {n_docs:,} documents.")
    return doc_bases


def _locate_range(index_dir: str, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Locate the postings of one index whose fingerprint lies in [lo, hi).

    Returns the fingerprints of the range and the number of postings of each (views
    on the memory-mapped arrays), the byte offset of the range in postings.dat (i.e.
    the bytes taken by postings with fingerprint < lo) and its number of postings.
    """
    fingerprints = np.load(os.path.join(index_dir, "fingerprints.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(index_dir, "offsets.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(index_dir, "lengths.npy"), mmap_mode="r")

    i = int(np.searchsorted(fingerprints, np.uint64(lo)))
    j = len(fingerprints) if hi >= 2**64 else int(np.searchsorted(fingerprints, np.uint64(hi)))
    bytes_below = (int(offsets[i - 1]) + int(lengths[i - 1]) * indexing.POSTING_DTYPE.itemsize) if i else 0
    return fingerprints[i:j], lengths[i:j], bytes_below, int(lengths[i:j].sum(dtype=np.int64))


def _merge_partition(task) -> Tuple[int, int]:
    """Pool job: gather one fingerprint range from every input, write its postings in
    place into the final postings file and its per-fingerprint arrays as raw part
    files.  Returns the range's number of unique fingerprints and of postings."""
    index_dirs, doc_bases, lo, hi, postings_path, part_prefix = task
    located = [_locate_range(d, lo, hi) for d in index_dirs]
    byte_start = sum(bytes_below for *_, bytes_below, _ in located)

    # Read each shard's slice straight into place, then shift its doc ids.
    raw = np.empty(sum(n for *_, n in located), dtype=indexing.POSTING_DTYPE)
    raw_bytes = raw.view(np.uint8)
    pos = 0
    for index_dir, base, (_, _, bytes_below, n) in zip(index_dirs, doc_bases, located):
        if n:
            with open(os.path.join(index_dir, "postings.dat"), "rb") as f:
                f.seek(bytes_below)
                f.readinto(raw_bytes[pos * raw.itemsize : (pos + n) * raw.itemsize])
                _release_page_cache(f, bytes_below, n * raw.itemsize)
            raw["doc_id"][pos : pos + n] += np.uint32(base)
        pos += n

    # Sort the postings by fingerprint; stable, so equal fingerprints keep shard order.
    fp = np.concatenate([np.repeat(fps, lengths) for fps, lengths, *_ in located])
    del located
    order = np.argsort(fp, kind="stable")
    fp = fp[order]
    postings = raw[order]
    del raw, order
    is_first = np.ones(len(fp), dtype=bool)  # first posting of each fingerprint
    is_first[1:] = fp[1:] != fp[:-1]
    first = np.flatnonzero(is_first)

    with open(postings_path, "r+b") as f:
        f.seek(byte_start)
        postings.tofile(f)
        f.flush()
        os.fsync(f.fileno())  # written back now, so the dirty pages can be dropped below
        _release_page_cache(f, byte_start, postings.nbytes)
    fp[first].tofile(part_prefix + ".fingerprints")
    (byte_start + first * indexing.POSTING_DTYPE.itemsize).astype(np.uint64).tofile(part_prefix + ".offsets")
    np.diff(np.append(first, len(postings))).astype(np.uint32).tofile(part_prefix + ".lengths")
    return len(first), len(postings)


def _release_page_cache(f, offset: int, nbytes: int) -> None:
    """Tell the kernel that this byte range will not be read again.

    Page cache counts against the memory limit of a SLURM job (cgroup), so the gigabytes
    streamed through a merge must not be left cached, or they compete with the workers.
    """
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(f.fileno(), offset, nbytes, os.POSIX_FADV_DONTNEED)


def _concatenate_parts(part_paths: List[str], npy_path: str, dtype: np.dtype, count: int):
    """Concatenate raw array files, in order, into one .npy file of `count` entries,
    through a memory map so that only one part is in memory at a time."""
    output = np.lib.format.open_memmap(npy_path, mode="w+", dtype=dtype, shape=(count,))
    start = 0
    for path in part_paths:
        part = np.fromfile(path, dtype=dtype)
        output[start : start + len(part)] = part
        start += len(part)
    output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and merge fingerprint indexes.")
    subparsers = parser.add_subparsers(dest="task", required=True)

    index_parser = subparsers.add_parser("index", help="Build an index from a corpus file")
    index_parser.add_argument("corpus_file", help="Corpus file (.jsonl, .jsonl.gz or .jsonl.zst)")
    index_parser.add_argument("index_dir", help="Directory of the resulting index")
    index_parser.add_argument("--length", type=int, default=4, help="k-gram length (default: 4)")
    index_parser.add_argument("--window_size", type=int, default=6, help="Winnowing window size (default: 6)")
    index_parser.add_argument("--stop_after", type=int, default=None, help="Index only this many documents")
    index_parser.add_argument("--delete_existing", action="store_true", help="Replace an existing index")

    merge_parser = subparsers.add_parser("merge", help="Merge the indexes found in a directory into one")
    merge_parser.add_argument("indexes_dir", help="Directory whose subdirectories are the indexes to merge")
    merge_parser.add_argument("output_dir", help="Directory of the merged index")
    merge_parser.add_argument("--delete_existing", action="store_true", help="Replace an existing index")

    for p in (index_parser, merge_parser):
        p.add_argument(
            "--n_workers", type=int, default=None,
            help="Number of worker processes (default: the CPUs available to the process)",
        )
        p.add_argument(
            "--max_memory_gb", type=float, default=None,
            help="Memory budget in GB (default: 80%% of the available memory; set it inside SLURM jobs)",
        )

    args = parser.parse_args()
    if args.task == "index":
        build_index(
            args.corpus_file, args.index_dir, args.length, args.window_size, args.n_workers,
            args.max_memory_gb, args.stop_after, args.delete_existing,
        )
    else:
        merge_indexes(
            args.indexes_dir, args.output_dir, args.n_workers, args.max_memory_gb, args.delete_existing
        )
