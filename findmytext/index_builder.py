"""Scripts for building and merging document fingerprint indexes.

Indexing pipeline
-----------------
Building a queryable index from a raw corpus is a two-step process:

Step 1 — Build intermediate indexes  (index_file)
    Each document is tokenised and run through the winnowing algorithm to produce a
    set of (fingerprint, position) pairs.  The pairs are accumulated in a
    MemoryBasedIndex (an in-memory dict) in the main process and periodically flushed
    to a gzipped MsgPack file to keep peak memory bounded.  With nb_workers > 1, a
    dedicated producer process dispatches fingerprinting to a pool of worker processes
    and forwards results to the main process via a bounded queue.

Step 2 — Merge into a disk-based index  (merge_indexes)
    The intermediate .jsonl.gz files from Step 1 are read back, sorted by fingerprint,
    and merged into a compact binary index that can be memory-mapped at query time.  The
    merge itself is parallelised: each file is parsed in a dedicated worker process and
    the sorted streams are heap-merged in the main process.

"""

import argparse
import array
import collections
import heapq
import itertools
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import msgpack
import numpy as np
import orjson
import tqdm

from . import utils, winnower
from .indexing import MemoryBasedIndex

try:
    import isal.igzip as gzip  # 2-4x faster gzip decompression (Intel ISA-L)
except ImportError:
    import gzip


#####################################################
# STEP 1: BUILDING OF MEMORY-BASED INDEX FROM CORPUS
#####################################################


def index_file(
    corpus_file: str,
    output_dir: str,
    length=4,
    window_size=6,
    stop_after: Optional[int] = None,
    delete_existing: bool = False,
    nb_workers: int = 1,
    max_nb_documents_before_flush=1_000_000,
    min_length: int = 100,
    max_length: int = 100_000,
    skip_prob: float = 0.0,
) -> List[str]:
    """Index documents from a corpus file and save intermediate index files to output_dir.

    Reads the corpus, computes winnowed fingerprints for every document, and
    periodically flushes the accumulated in-memory index to a gzipped MsgPack file.
    The resulting intermediate files can then be merged into a final disk-based index
    with merge_indexes().

    Args:
        corpus_file: Path to the corpus (.jsonl, .jsonl.gz, or .jsonl.zst).  Each line
            must be a JSON object with at least an "id" field (unique document
            identifier) and a "text" field (document content).
        output_dir: Directory where the intermediate index files will be saved.
        length: k-gram length for fingerprinting (default 4).
        window_size: Winnowing window size (default 6).
        stop_after: If set, stop after this many documents (useful for testing).
        delete_existing: If True, delete output_dir before starting.
        nb_workers: Total number of processes.  One is used as the producer (stream
            reading + dispatch); the rest are fingerprint workers.  nb_workers=1 runs
            a single worker with no separate producer process.
        max_nb_documents_before_flush: Maximum number of documents to hold in memory before flushing to disk. Default is 1M.
        min_length: Minimum document text length to index; shorter documents are skipped (default 100).
        max_length: Maximum document text length to index; longer documents are skipped (default 100,000).
        skip_prob: Probability of randomly skipping a document while streaming (default 0.0).

    Returns:
        List of paths to the saved intermediate index files.

    """

    if not os.path.isfile(corpus_file):
        raise ValueError(f"Provided file {corpus_file} does not exist or is not a file")

    if corpus_file.endswith((".jsonl", ".jsonl.gz")):
        stream = utils.stream_jsonl(
            corpus_file,
            min_length=min_length,
            max_length=max_length,
            skip_prob=skip_prob,
        )
    elif corpus_file.endswith(".jsonl.zst"):
        stream = utils.stream_json_zst(
            corpus_file,
            min_length=min_length,
            max_length=max_length,
            skip_prob=skip_prob,
        )
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .jsonl, .jsonl.gz, or .jsonl.zst file."
        )
    print("Indexing documents from", corpus_file)

    if os.path.exists(output_dir):
        if delete_existing:
            print(f"Deleting existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        elif not os.path.isdir(output_dir):
            raise ValueError(f"Output path {output_dir} exists and is not a directory")
    os.makedirs(output_dir, exist_ok=True)
    print("Index files will be saved to", output_dir)
    print(f"Indexing stream with {nb_workers} fingerprint workers...", flush=True)

    meta = {
        "length": length,
        "window_size": window_size,
        "base": 256,
        "punctuation": False,
    }

    index = MemoryBasedIndex(meta=meta)

    index_files: List[str] = []

    # Get batched fingerprints from the stream using a producer-consumer pattern.
    batch_results = get_batched_fingerprints(
        stream, stop_after, nb_workers, length, window_size
    )

    total_count = 0
    for batch_result in batch_results:
        for doc_id, fp_pos_list in batch_result.items():
            index.add_fingerprints(doc_id, fp_pos_list)
            total_count += 1

            if total_count % 100_000 == 0:
                print(f"Processed {total_count:,} documents...", flush=True)

            if index.doc_count >= max_nb_documents_before_flush:
                increment_str = str((total_count + 1) // 1000) + "K"
                path = os.path.join(output_dir, f"intermediate-{increment_str}.mpk.gz")
                index.to_msgpack(path)
                index_files.append(path)
                index = MemoryBasedIndex(meta=meta)

    if index.doc_count > 0:
        path = os.path.join(output_dir, "final.mpk.gz")
        index.to_msgpack(path)
        index_files.append(path)

    return index_files


def get_batched_fingerprints(
    stream: Any,
    stop_after: Optional[int],
    nb_workers: int,
    length: int,
    window_size: int,
) -> Iterator[Dict[str, List[Tuple[int, int]]]]:
    """Get batched fingerprints from the stream using a producer-consumer pattern.
    The fingerprinting is extracted with winnowing in parallel using multiple
    worker processes, and the results are yielded in batches.

    Args:
        stream: An iterable stream of documents to be processed.
        stop_after: If set, stop after this many documents (useful for testing).
        nb_workers: Total number of processes. One is used as the producer (stream
        reading + dispatch); the rest are fingerprint workers. nb_workers=1 runs a
        single worker with no separate producer process.
        length: k-gram length for fingerprinting.
        window_size: Winnowing window size.

    Returns:
        Iterator[Dict[str, List[Tuple[int, int]]]]: An iterator over batched
        fingerprint results, where each item is a dictionary mapping document IDs
        to lists of (fingerprint, position) tuples.

    """

    n_workers = max(1, nb_workers - 1)  # reserve one core for the producer process

    # Queue for results from the producer to the main process.
    result_queue: mp.Queue = mp.Queue(maxsize=n_workers * 4)

    producer = mp.Process(
        target=_produce,
        args=(stream, stop_after, n_workers, result_queue, length, window_size),
    )
    producer.start()
    try:
        while True:
            try:
                item = result_queue.get(timeout=2)
            except queue.Empty:
                if not producer.is_alive():
                    raise RuntimeError(
                        f"Producer process died unexpectedly (exitcode={producer.exitcode})"
                    )
                continue
            if item is None:
                return
            yield item
    finally:
        producer.join()


def _produce(
    data_stream: Any,
    stop_after: Optional[int],
    n_workers: int,
    result_queue: mp.Queue,
    length: int,
    window_size: int,
) -> None:
    """Read batches of documents from the stream, dispatch to workers that will run
    the winnowing algorithm, and forward the resulting fingerprints.

    Uses apply_async with a bounded pending list (at most n_workers tasks in flight at
    once) so that when result_queue.put() blocks (consumer busy flushing to disk), task
    submission also stalls and memory usage stays bounded.  A None sentinel signals the
    main process that the stream is exhausted.
    """
    batch_winnower = winnower.Winnower(
        length=length, window_size=window_size
    ).get_winnowed_fingerprints_batch

    batches_gen = utils.generate_batches(data_stream, stop_after)
    try:
        with mp.Pool(processes=n_workers) as pool:
            pending: List[Any] = []
            for batch in batches_gen:
                if len(pending) >= n_workers:
                    # Wait for the oldest task and forward the result before submitting
                    # a new one.  This caps in-flight tasks at n_workers, preventing
                    # unbounded accumulation in the Pool's internal result buffer while
                    # the consumer is busy flushing.
                    result_queue.put(pending.pop(0).get())
                pending.append(pool.apply_async(batch_winnower, (batch,)))
            for async_result in pending:
                result_queue.put(async_result.get())
    finally:
        try:
            result_queue.put(
                None, timeout=1200
            )  # wait up to 20 min; consumer may be busy flushing to disk
        except queue.Full:
            pass


#####################################################
# STEP 2: MERGE INTO SINGLE DISK-BASED INDEX
#####################################################


def merge_indexes_from_dir(
    temp_index_dir: str,
    output_dir: str,
    include_similarities: bool = False,
    similarity_chunk_size: int = 1_000_000,
    max_similarity_posting_length: Optional[int] = None,
    min_similarity: int = 1,
    max_similarity_chunks_per_merge: Optional[int] = None,
):
    """Merge intermediate MsgPack indexes into a disk-based index.
    The intermediate indexes must be in gzipped MsgPack format with the extension ``.mpk.gz``.

    When ``include_similarities`` is true, the output also contains
    ``similarities.npy``. Each row stores ``(doc_i, doc_j, count)``, where the
    document fields are internal uint32 IDs and ``count`` is the number of unique
    fingerprints shared by the pair. Pairs below ``min_similarity`` are omitted.

    Args:
        temp_index_dir (str): Directory containing the intermediate index files to merge.
        output_dir (str): Directory where the merged index files will be saved.
        include_similarities (bool): Whether to create ``similarities.npy``.
        similarity_chunk_size (int): Maximum number of distinct document-pair
            counts held in memory before flushing a sorted temporary chunk.
        max_similarity_posting_length (Optional[int]): If set, exclude a
            fingerprint from similarity counting when it occurs in more than this
            many unique documents. This bounds quadratic pair generation but makes
            the resulting counts approximate rather than exact.
        min_similarity (int): Only persist document pairs sharing at least this
            many counted fingerprints. The default of 1 preserves every non-zero
            pair. Filtering happens after complete aggregation, so this reduces
            final output size but not pair-generation or temporary-chunk work.
        max_similarity_chunks_per_merge (Optional[int]): Maximum number of
            temporary similarity chunks opened in one merge. If set, chunks are
            compacted in multiple passes with this fan-in. The default of None
            opens all chunks in the final merge, preserving the previous behavior.

    Raises:
        ValueError: If the provided temp_index_dir does not exist or contains no .mpk.gz files.
    """

    if not os.path.isdir(temp_index_dir):
        raise ValueError(f"Provided path {temp_index_dir} is not a directory")
    index_files = []
    for f in os.listdir(temp_index_dir):
        if f.endswith(".mpk.gz"):
            index_files.append(os.path.join(temp_index_dir, f))
    if not index_files:
        raise ValueError(f"No .mpk.gz index files found in {temp_index_dir}")
    merge_indexes(
        index_files,
        output_dir,
        include_similarities=include_similarities,
        similarity_chunk_size=similarity_chunk_size,
        max_similarity_posting_length=max_similarity_posting_length,
        min_similarity=min_similarity,
        max_similarity_chunks_per_merge=max_similarity_chunks_per_merge,
    )


def merge_indexes(
    index_files: List[str],
    output_dir: str,
    save_every_n: int = 10_000_000,
    include_similarities: bool = False,
    similarity_chunk_size: int = 1_000_000,
    max_similarity_posting_length: Optional[int] = None,
    min_similarity: int = 1,
    max_similarity_chunks_per_merge: Optional[int] = None,
):
    """Merge gzipped MsgPack intermediate indexes into one disk-based index.

    The merged index will consist of the following files:
    - postings.dat: a binary file containing the concatenated postings lists (with internal integer doc IDs)
    - fingerprints.npy: an array of uint64 containing the fingerprints corresponding to each postings list in postings.dat
    - offsets.npy: an array of uint64 containing the byte offsets for each postings list in postings.dat
    - lengths.npy: an array of uint32 containing the number of postings entries for each fingerprint
    - meta.json: a JSON file containing the index parameters (length, window_size, base, punctuation)
    - doc_name_offsets.npy: an array of uint64 containing the byte offsets for each doc name in the concatenated byte array
    - doc_name_bytes.npy: an array of uint8 containing the UTF-8 encoded doc names concatenated together.
    - similarities.npy (optional): a structured array with uint32 fields
        ``doc_i``, ``doc_j``, and ``count``. It contains one row per document pair
            sharing at least ``min_similarity`` counted fingerprints. This file is
            created only when ``include_similarities=True``.

    Args:
        index_files (List[str]): List of paths to the intermediate index files to merge.
        output_dir (str): Directory where the merged index files will be saved.
        save_every_n (int): Frequency (in number of fingerprints) at which to save intermediate merged index files during merging.
        include_similarities (bool): Whether to create ``similarities.npy``.
        similarity_chunk_size (int): Maximum number of distinct document-pair
            counts held in memory before flushing a sorted temporary chunk.
        max_similarity_posting_length (Optional[int]): If set, exclude a
            fingerprint from similarity counting when it occurs in more than this
            many unique documents. This bounds quadratic pair generation but makes
            the resulting counts approximate rather than exact.
        min_similarity (int): Only persist document pairs sharing at least this
            many counted fingerprints. The default of 1 preserves every non-zero
            pair. Filtering happens after complete aggregation, so this reduces
            final output size but not pair-generation or temporary-chunk work.
        max_similarity_chunks_per_merge (Optional[int]): Maximum number of
            temporary similarity chunks opened in one merge. If set, chunks are
            compacted in multiple passes with this fan-in. The default of None
            opens all chunks in the final merge.
    Raises:
        ValueError: If no index files are provided or if the input files are not in the expected format.

    """

    print("Merging the following index files:", index_files)
    print("Saving merged index to", output_dir)

    if not index_files:
        raise ValueError("At least one JSONL file must be provided to load the index")

    os.makedirs(output_dir, exist_ok=True)

    # Read the metadata record from the first msgpack file and save to meta.json.
    with gzip.open(index_files[0], "rb") as f:
        meta_raw = next(msgpack.Unpacker(f, raw=False))
        meta = {
            k: meta_raw[k]
            for k in ["length", "window_size", "base", "punctuation"]
            if k in meta_raw
        }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        f.write(orjson.dumps(meta).decode("utf-8"))
    print("Meta parameters:", meta)

    # Create mappings between external document IDs and internal integer IDs
    # and save the mapping to the output directory as two NumPy arrays: doc_name_offsets.npy and doc_name_bytes.npy
    to_internal = _create_doc_id_mappings(index_files)
    _write_doc_id_mapping(to_internal, output_dir)

    # Path for the merged postings file (fingerprint-to-postings mapping)
    posting_file_path = os.path.join(output_dir, "postings.dat")

    # ExpandingBuffer to accumulate the sorted fingerprints and their corresponding offsets
    # and lengths using a structured NumPy array with a expandable buffer.
    buf = ExpandingBuffer()
    similarity_accumulator = (
        SimilarityAccumulator(
            output_dir,
            max_pairs_in_memory=similarity_chunk_size,
            max_posting_length=max_similarity_posting_length,
            min_similarity=min_similarity,
            max_chunks_per_merge=max_similarity_chunks_per_merge,
        )
        if include_similarities
        else None
    )

    # Parse all files in parallel processes and merge the sorted streams
    merged_stream = merge_streams_parallel(index_files)

    # Large write buffer (8 MB) amortises syscall overhead across millions of small writes.
    current_offset = 0
    with open(posting_file_path, "wb", buffering=8 * 1024 * 1024) as posting_file:
        for fp, postings_list in merged_stream:
            # Remap external doc IDs to internal integers (pure lookups)
            internal_postings = [(to_internal[eid], pos) for eid, pos in postings_list]
            data = np.array(internal_postings, dtype=np.uint32).tobytes()

            if similarity_accumulator is not None:
                similarity_accumulator.add_posting_list(internal_postings)

            # Add fingerprint with byte offset and postings entry count in postings.dat.
            buf.add(fp, current_offset, len(internal_postings))
            posting_file.write(data)

            current_offset += len(data)

            if buf.count % 1_000_000 == 0:
                print(f"Processed {buf.count:,} fingerprints...", flush=True)

            if save_every_n and buf.count > 0 and buf.count % save_every_n == 0:
                print(
                    f"Saving intermediate index ({buf.count:,} fingerprints)...",
                    end="",
                    flush=True,
                )
                buf.save(output_dir)
                print("Done")

    print(f"Saving final index ({buf.count:,} fingerprints)...", end="", flush=True)
    buf.save(output_dir)
    print("Done")

    if similarity_accumulator is not None:
        similarity_accumulator.finish()


class SimilarityAccumulator:
    """Build an exact sparse document-similarity index in bounded memory.

    Each fingerprint contributes one count to every pair of distinct documents in
    its posting list. Sorted chunk files make the accumulator's memory use bounded;
    the chunks are merged into a single memory-mappable ``similarities.npy`` file.
    Low-count pairs are pruned only after all partial counts have been aggregated.
    Optionally, temporary chunks are compacted through bounded-fan-in merge passes
    before the final output is written.
    """

    _DTYPE = np.dtype([("key", np.uint64), ("count", np.uint32)])
    _OUTPUT_DTYPE = np.dtype(
        [("doc_i", np.uint32), ("doc_j", np.uint32), ("count", np.uint32)]
    )

    def __init__(
        self,
        output_dir: str,
        max_pairs_in_memory: int = 1_000_000,
        max_posting_length: Optional[int] = None,
        min_similarity: int = 1,
        max_chunks_per_merge: Optional[int] = None,
    ):
        if max_pairs_in_memory <= 0:
            raise ValueError("similarity_chunk_size must be positive")
        if max_posting_length is not None and max_posting_length < 2:
            raise ValueError("max_similarity_posting_length must be at least 2")
        if min_similarity < 1:
            raise ValueError("min_similarity must be at least 1")
        if max_chunks_per_merge is not None and max_chunks_per_merge < 2:
            raise ValueError("max_similarity_chunks_per_merge must be at least 2")
        self.output_dir = output_dir
        self.max_pairs_in_memory = max_pairs_in_memory
        self.max_posting_length = max_posting_length
        self.min_similarity = min_similarity
        self.max_chunks_per_merge = max_chunks_per_merge
        self.counts: Dict[int, int] = {}
        self.chunk_paths: List[str] = []

    def add_posting_list(self, postings: List[Tuple[int, int]]) -> None:
        """Add one fingerprint's unique document set to the pair accumulator."""
        document_ids = sorted({doc_id for doc_id, _ in postings})
        if (
            self.max_posting_length is not None
            and len(document_ids) > self.max_posting_length
        ):
            return

        for first, second in itertools.combinations(document_ids, 2):
            key = (first << 32) | second
            self.counts[key] = self.counts.get(key, 0) + 1
            if len(self.counts) >= self.max_pairs_in_memory:
                self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self.counts:
            return
        chunk = np.empty(len(self.counts), dtype=self._DTYPE)
        for i, (key, count) in enumerate(sorted(self.counts.items())):
            chunk[i] = (key, count)
        path = os.path.join(
            self.output_dir, f".similarity-chunk-{len(self.chunk_paths):06d}.npy"
        )
        np.save(path, chunk)
        self.chunk_paths.append(path)
        self.counts.clear()

    def finish(self) -> None:
        self._flush_chunk()
        output_path = os.path.join(self.output_dir, "similarities.npy")
        if not self.chunk_paths:
            np.save(output_path, np.empty(0, dtype=self._OUTPUT_DTYPE))
            return

        if self.max_chunks_per_merge is not None:
            self.chunk_paths = self._compact_chunks(self.chunk_paths)

        n_pairs = sum(
            1
            for _, count in self._merged_items(self.chunk_paths)
            if count >= self.min_similarity
        )
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=self._OUTPUT_DTYPE, shape=(n_pairs,)
        )
        output_index = 0
        for key, count in self._merged_items(self.chunk_paths):
            if count < self.min_similarity:
                continue
            output[output_index] = (key >> 32, key & 0xFFFFFFFF, count)
            output_index += 1
        output.flush()
        del output
        for path in self.chunk_paths:
            os.remove(path)
        print(f"Saved {n_pairs:,} document similarity pairs.")

    def _compact_chunks(self, paths: List[str]) -> List[str]:
        """Reduce chunk count through bounded-fan-in merge passes."""
        fan_in = self.max_chunks_per_merge
        if fan_in is None:
            return paths

        pass_index = 0
        while len(paths) > fan_in:
            compacted = []
            for group_index, start in enumerate(range(0, len(paths), fan_in)):
                group = paths[start : start + fan_in]
                if len(group) == 1:
                    compacted.append(group[0])
                    continue
                output_path = os.path.join(
                    self.output_dir,
                    f".similarity-merge-{pass_index:03d}-{group_index:06d}.npy",
                )
                self._write_merged_chunk(group, output_path)
                for path in group:
                    os.remove(path)
                compacted.append(output_path)
            paths = compacted
            pass_index += 1
        return paths

    def _write_merged_chunk(self, paths: List[str], output_path: str) -> None:
        n_pairs = sum(1 for _ in self._merged_items(paths))
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=self._DTYPE, shape=(n_pairs,)
        )
        for index, item in enumerate(self._merged_items(paths)):
            output[index] = item
        output.flush()
        del output

    def _merged_items(self, paths: List[str]) -> Iterator[Tuple[int, int]]:
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        heap = []
        for chunk_index, array_data in enumerate(arrays):
            if len(array_data):
                heapq.heappush(heap, (int(array_data[0]["key"]), chunk_index, 0))

        while heap:
            key, chunk_index, position = heapq.heappop(heap)
            count = int(arrays[chunk_index][position]["count"])
            next_position = position + 1
            if next_position < len(arrays[chunk_index]):
                next_key = int(arrays[chunk_index][next_position]["key"])
                heapq.heappush(heap, (next_key, chunk_index, next_position))

            while heap and heap[0][0] == key:
                _, other_chunk, other_position = heapq.heappop(heap)
                count += int(arrays[other_chunk][other_position]["count"])
                next_position = other_position + 1
                if next_position < len(arrays[other_chunk]):
                    next_key = int(arrays[other_chunk][next_position]["key"])
                    heapq.heappush(heap, (next_key, other_chunk, next_position))
            yield key, count


class ExpandingBuffer:
    """Helper class to accumulate sorted fingerprints and their corresponding offsets
    and lengths using a structured NumPy array with an expandable buffer.

    This class maintains an internal NumPy array with a specified initial size and a
    maximum increment size. When adding new entries, if the buffer is full, it
    automatically expands by creating a new array with increased size and copying the
    existing data over.
    """

    def __init__(self, start_size=10_000_000, max_increment=100_000_000):
        """Initialize the ExpandingBuffer with a specified initial size and maximum
        increment size.

        The buffer is implemented as a structured NumPy array with three fields: 'fingerprints' (uint64), 'offsets' (uint64),
         and 'lengths' (uint32).
        """
        _DTYPE = [
            ("fingerprints", np.uint64),
            ("offsets", np.uint64),
            ("lengths", np.uint32),
        ]

        self.buf: np.ndarray = np.empty(start_size, dtype=_DTYPE)
        self.max_increment = max_increment
        self.count = 0

    def add(self, fingerprint: int, offset: int, length: int):
        """Add a new entry with the given fingerprint, offset, and length to the
        buffer."""
        if self.count == len(self.buf):
            new_size = len(self.buf) + min(len(self.buf), self.max_increment)
            new = np.empty(new_size, dtype=self.buf.dtype)
            new[: len(self.buf)] = self.buf
            self.buf = new

        self.buf[self.count]["fingerprints"] = fingerprint
        self.buf[self.count]["offsets"] = offset
        self.buf[self.count]["lengths"] = length

        self.count += 1

    def save(self, output_dir: str):
        """Save the accumulated fingerprints, offsets, and lengths to NumPy files in the
        specified output directory.

        The files are saved as 'fingerprints.npy', 'offsets.npy', and 'lengths.npy',
        containing only the valid entries up to the current count.
        """
        np.save(
            os.path.join(output_dir, "fingerprints.npy"),
            self.buf["fingerprints"][: self.count],
        )
        np.save(
            os.path.join(output_dir, "offsets.npy"), self.buf["offsets"][: self.count]
        )
        np.save(
            os.path.join(output_dir, "lengths.npy"), self.buf["lengths"][: self.count]
        )


def _create_doc_id_mappings(index_files: List[str]) -> Dict[str, int]:
    """Assign a stable internal integer ID to every unique external document ID.

    Doc IDs are stored in a compact header block at the start of each file, so
    this scan is fast enough to run sequentially — no worker pool needed.
    """
    print(f"Collecting document IDs from {len(index_files)} files...", flush=True)
    to_internal: Dict[str, int] = {}
    for file_path in tqdm.tqdm(
        index_files, total=len(index_files), desc="Scanning doc IDs"
    ):
        with gzip.open(file_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=0)
            next(unpacker)  # skip metadata
            n_docs = next(unpacker)
            for _ in range(n_docs):
                doc_id = next(unpacker)
                if doc_id not in to_internal:
                    to_internal[doc_id] = len(to_internal)
    print(f"  Found {len(to_internal)} unique documents.", flush=True)
    return to_internal


def _write_doc_id_mapping(to_internal: Dict[str, int], output_dir: str):
    """Save the mapping from internal integer IDs to external document IDs as two NumPy arrays:
    - doc_name_offsets.npy: uint64 array of byte offsets for each document name in the concatenated byte array
    - doc_name_bytes.npy: uint8 array containing the UTF-8 encoded document names concatenated together.
    The offset array allows retrieval of each document name by slicing the byte array accordingly."""
    to_external = {i: str(doc_id) for doc_id, i in to_internal.items()}
    n_docs = len(to_external)
    encoded = [to_external[i].encode("utf-8") for i in range(n_docs)]
    doc_offsets = np.zeros(n_docs + 1, dtype=np.uint64)
    np.cumsum([len(b) for b in encoded], out=doc_offsets[1:])
    flat = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    np.save(os.path.join(output_dir, "doc_name_offsets.npy"), doc_offsets)
    np.save(os.path.join(output_dir, "doc_name_bytes.npy"), flat)
    print(f"Saved document ID mapping for {n_docs} documents.")


def _stream_file_to_queue(file_path: str, q: mp.Queue, batch_size: int = 200):
    """Worker process: parse one gzipped msgpack index file and push (fingerprint,
    postings_list) items in batches to a queue.

    Batching reduces IPC overhead; a None sentinel signals end of stream. Module-level
    so it can be pickled by multiprocessing.
    """
    batch = []
    with gzip.open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=0)
        next(unpacker)  # skip metadata
        # Load the doc-ID table from the file header.
        n_docs = next(unpacker)
        to_external = [next(unpacker) for _ in range(n_docs)]
        while True:
            try:
                fp = next(unpacker)
                n = next(unpacker)
                docs = np.frombuffer(next(unpacker), dtype=np.uint32).tolist()
                poss = np.frombuffer(next(unpacker), dtype=np.uint32).tolist()
                postings_list = [(to_external[d], p) for d, p in zip(docs, poss)]
            except StopIteration:
                break
            batch.append((fp, postings_list))
            if len(batch) == batch_size:
                q.put(batch)
                batch = []
    if batch:
        q.put(batch)
    q.put(None)  # sentinel


def merge_streams_parallel(
    file_paths: List[str], queue_depth: int = 2000
) -> Generator[Tuple[int, List[Tuple[str, int]]], None, None]:
    """Parse each index file in a dedicated worker process and heap-merge the results.

    Each worker decompresses and parses independently (no GIL), so this scales close to
    min(n_files, n_cores)x versus the single-threaded version.
    """
    queues = [mp.Queue(maxsize=queue_depth) for _ in file_paths]
    procs = [
        mp.Process(target=_stream_file_to_queue, args=(f, q), daemon=True)
        for f, q in zip(file_paths, queues)
    ]
    for p in procs:
        p.start()

    def _iter_queue(q: mp.Queue) -> Generator:
        """Drain a queue, unbatching items as they arrive."""
        while True:
            batch = q.get()
            if batch is None:
                return
            yield from batch

    yield from merge_streams([_iter_queue(q) for q in queues])

    for p in procs:
        p.join()


def merge_streams(
    streams: List[Generator[Tuple[int, List[Tuple[str, int]]], None, None]],
) -> Generator[Tuple[int, List[Tuple[str, int]]], None, None]:
    """Merge multiple sorted streams of (fingerprint, postings_list) tuples into a
    single sorted stream, yielding one (fingerprint, postings_list) tuple at a time.

    This function uses a heap to efficiently merge the streams while maintaining the
    sorted order based on fingerprints.
    """
    heap = []
    for i, stream in enumerate(streams):
        try:
            fp, postings_list = next(stream)
            heapq.heappush(heap, (fp, i, postings_list))
        except StopIteration:
            continue

    while heap:
        fp, stream_idx, postings_list = heapq.heappop(heap)
        merged = list(postings_list)

        try:
            next_fp, next_postings_list = next(streams[stream_idx])
            heapq.heappush(heap, (next_fp, stream_idx, next_postings_list))
        except StopIteration:
            pass

        # Merge postings from any other streams that share the same fingerprint
        while heap and heap[0][0] == fp:
            _, other_idx, other_postings = heapq.heappop(heap)
            merged.extend(other_postings)
            try:
                next_fp, next_postings_list = next(streams[other_idx])
                heapq.heappush(heap, (next_fp, other_idx, next_postings_list))
            except StopIteration:
                pass

        yield fp, merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for building and merging indexes."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    # --- index task ---
    index_parser = subparsers.add_parser(
        "index", help="Build an index from a corpus file"
    )
    index_parser.add_argument(
        "corpus_file",
        type=str,
        help="Path to the corpus file (.jsonl, .jsonl.gz, or .jsonl.zst)",
    )
    index_parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory in which to save the index files",
    )
    index_parser.add_argument(
        "--nb_workers",
        type=int,
        default=1,
        help="Number of worker processes to use for indexing (default: 1)",
    )
    index_parser.add_argument(
        "--length", type=int, default=5, help="k-gram length (default: 5)"
    )
    index_parser.add_argument(
        "--window_size", type=int, default=6, help="Winnowing window size (default: 6)"
    )
    index_parser.add_argument(
        "--stop_after", type=int, default=None, help="Stop after this many samples"
    )
    index_parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum document text length to index (default: 100)",
    )
    index_parser.add_argument(
        "--max_length",
        type=int,
        default=100_000,
        help="Maximum document text length to index (default: 100,000)",
    )
    index_parser.add_argument(
        "--skip_prob",
        type=float,
        default=0.0,
        help="Probability of randomly skipping a document (default: 0.0)",
    )

    # --- merge task ---
    merge_parser = subparsers.add_parser(
        "merge", help="Merge intermediate index files into a DiskBasedIndex"
    )
    merge_parser.add_argument(
        "temp_index_dir",
        type=str,
        help="Directory containing the intermediate index files to merge",
    )
    merge_parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where the merged index will be saved.",
    )
    merge_parser.add_argument(
        "--include-similarities",
        action="store_true",
        help="Also write similarities.npy with sparse document-pair overlap counts.",
    )
    merge_parser.add_argument(
        "--similarity-chunk-size",
        type=int,
        default=1_000_000,
        help="Maximum pair counts held in memory before a similarity chunk is flushed.",
    )
    merge_parser.add_argument(
        "--max-similarity-posting-length",
        type=int,
        default=None,
        help="Skip fingerprints occurring in more unique documents than this; bounds pair generation but makes similarity counts approximate.",
    )
    merge_parser.add_argument(
        "--min-similarity",
        type=int,
        default=1,
        help="Only store document pairs sharing at least this many counted fingerprints (default: 1).",
    )
    merge_parser.add_argument(
        "--max-similarity-chunks-per-merge",
        type=int,
        default=None,
        help="Maximum temporary similarity chunks opened per merge pass; default opens all chunks together.",
    )

    args = parser.parse_args()

    if args.task == "index":
        index_file(
            args.corpus_file,
            length=args.length,
            window_size=args.window_size,
            output_dir=args.output_dir,
            stop_after=args.stop_after,
            nb_workers=args.nb_workers,
            min_length=args.min_length,
            max_length=args.max_length,
            skip_prob=args.skip_prob,
        )

    elif args.task == "merge":
        merge_indexes_from_dir(
            args.temp_index_dir,
            args.output_dir,
            include_similarities=args.include_similarities,
            similarity_chunk_size=args.similarity_chunk_size,
            max_similarity_posting_length=args.max_similarity_posting_length,
            min_similarity=args.min_similarity,
            max_similarity_chunks_per_merge=args.max_similarity_chunks_per_merge,
        )
