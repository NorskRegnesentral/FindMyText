"""Scripts for building and merging document fingerprint indexes.

Indexing pipeline
-----------------
Building a queryable index from a raw corpus is a two-step process:

Step 1 — Build intermediate indexes  (index_file)
    Each document is tokenised and run through the winnowing algorithm to produce a
    set of (fingerprint, position) pairs.  The pairs are accumulated in a
    MemoryBasedIndex in the main process and periodically flushed to gzipped MsgPack.
    Parallel extraction keeps a bounded number of batches in flight.

Step 2 — Merge into a disk-based index  (merge_indexes)
    The intermediate .mpk.gz files from Step 1 are read back, sorted by fingerprint,
    and merged into a compact binary index that can be memory-mapped at query time.  The
    merge itself is parallelised: each file is parsed in a dedicated worker process and
    the sorted streams are heap-merged in the main process.

"""

import argparse
import heapq
import multiprocessing as mp
import os
import queue
import shutil
from contextlib import closing
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
        nb_workers: Process budget, including the parent that reads and inserts.
            Values above one use max(1, nb_workers - 1) fingerprint workers.
            nb_workers=1 fingerprints directly in the parent.
        max_nb_documents_before_flush: Maximum number of documents to hold in memory before flushing to disk. Default is 1M.

    Returns:
        List of paths to the saved intermediate index files.

    """

    if not os.path.isfile(corpus_file):
        raise ValueError(f"Provided file {corpus_file} does not exist or is not a file")
    if nb_workers < 1 or max_nb_documents_before_flush < 1:
        raise ValueError("Worker count and flush size must be positive")
    if stop_after is not None and stop_after < 0:
        raise ValueError("stop_after must be nonnegative")
    configured_winnower = winnower.Winnower(length=length, window_size=window_size)

    if corpus_file.endswith(".jsonl"):
        stream = utils.stream_jsonl(corpus_file)
    elif corpus_file.endswith(".jsonl.gz"):
        stream = utils.stream_jsonl(corpus_file)
    elif corpus_file.endswith(".jsonl.zst"):
        stream = utils.stream_json_zst(corpus_file)
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

    meta = configured_winnower.to_metadata()

    index = MemoryBasedIndex(meta=meta)

    index_files: List[str] = []

    # Get batched fingerprints from the stream using a producer-consumer pattern.
    batch_results = get_batched_fingerprints(
        stream, stop_after, nb_workers, length, window_size
    )

    total_count = 0
    with closing(stream), closing(batch_results):
        for batch_result in batch_results:
            for doc_id, fp_pos_list in batch_result.items():
                index.add_fingerprints(doc_id, fp_pos_list)
                total_count += 1

                if total_count % 100_000 == 0:
                    print(f"Processed {total_count:,} documents...", flush=True)

                if index.doc_count >= max_nb_documents_before_flush:
                    path = os.path.join(output_dir, f"intermediate-{len(index_files):06d}.mpk.gz")
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
        nb_workers: Process budget including the reading/inserting parent.
        length: k-gram length for fingerprinting.
        window_size: Winnowing window size.

    Returns:
        Iterator[Dict[str, List[Tuple[int, int]]]]: An iterator over batched
        fingerprint results, where each item is a dictionary mapping document IDs
        to lists of (fingerprint, position) tuples.

    """

    if nb_workers < 1:
        raise ValueError("Worker count must be positive")
    batch_winnower = winnower.Winnower(
        length=length, window_size=window_size
    ).get_winnowed_fingerprints_batch
    batches = utils.generate_batches(stream, stop_after)
    if nb_workers == 1:
        for batch in batches:
            yield batch_winnower(batch)
        return
    n_workers = nb_workers - 1
    # Keep the corpus generator in the parent: spawn/forkserver cannot pickle it.
    with mp.Pool(processes=n_workers) as pool:
        pending: List[Any] = []
        for batch in batches:
            if len(pending) >= n_workers:
                yield pending.pop(0).get()
            pending.append(pool.apply_async(batch_winnower, (batch,)))
        for result in pending:
            yield result.get()


#####################################################
# STEP 2: MERGE INTO SINGLE DISK-BASED INDEX
#####################################################


def merge_indexes_from_dir(temp_index_dir: str, output_dir: str):
    """Merge the index files in the specified directory and store the result
    in output_dir. The index files in temp_index_dir must be in .mpk.gz format."""

    if not os.path.isdir(temp_index_dir):
        raise ValueError(f"Provided path {temp_index_dir} is not a directory")
    index_files = []
    for f in sorted(os.listdir(temp_index_dir)):
        if f.endswith(".mpk.gz"):
            index_files.append(os.path.join(temp_index_dir, f))
    if not index_files:
        raise ValueError(f"No .mpk.gz index files found in {temp_index_dir}")
    merge_indexes(index_files, output_dir)


def merge_indexes(
    index_files: List[str], output_dir: str, save_every_n: int = 10_000_000
):
    """Merge multiple intermediate index files (in gzipped MsgPack format) into a single
    disk-based index.

    The merged index will consist of the following files:
    - postings.dat: a binary file containing the concatenated postings lists (with internal integer doc IDs)
    - fingerprints.npy: an array of uint64 containing the fingerprints corresponding to each postings list in postings.dat
    - offsets.npy: an array of uint64 containing the byte offsets for each postings list in postings.dat
    - lengths.npy: an array of uint32 containing the number of postings entries for each fingerprint
    - meta.json: a JSON file containing the index parameters (length, window_size, base, punctuation)
    - doc_name_offsets.npy: an array of uint64 containing the byte offsets for each doc name in the concatenated byte array
    - doc_name_bytes.npy: an array of uint8 containing the UTF-8 encoded doc names concatenated together.

    Args:
        index_files (List[str]): List of paths to the intermediate index files to merge.
        output_dir (str): Directory where the merged index files will be saved.
        save_every_n (int): Frequency (in number of fingerprints) at which to save intermediate merged index files during merging
    Raises:
        ValueError: If no index files are provided or if the input files are not in the expected format.

    """

    print("Merging the following index files:", index_files)
    print("Saving merged index to", output_dir)

    if not index_files:
        raise ValueError("At least one MsgPack index file must be provided")
    meta = None
    for path in index_files:
        with gzip.open(path, "rb") as f:
            source_meta = winnower.Winnower.from_metadata(
                next(msgpack.Unpacker(f, raw=False))
            ).to_metadata()
        if meta is not None and source_meta != meta:
            raise ValueError(f"Incompatible fingerprint metadata in {path}")
        meta = source_meta
    os.makedirs(output_dir, exist_ok=True)
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

    # Parse all files in parallel processes and merge the sorted streams
    merged_stream = merge_streams_parallel(index_files)

    # Large write buffer (8 MB) amortises syscall overhead across millions of small writes.
    current_offset = 0
    with closing(merged_stream), open(posting_file_path, "wb", buffering=8 * 1024 * 1024) as posting_file:
        for fp, postings_list in merged_stream:
            # Remap external doc IDs to internal integers (pure lookups)
            internal_postings = [(to_internal[eid], pos) for eid, pos in postings_list]
            data = np.array(internal_postings, dtype=np.uint32).tobytes()

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


def _iter_index_file(file_path: str):
    """Read one intermediate file without discarding incomplete posting records."""
    with gzip.open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=0)
        next(unpacker)  # skip metadata
        # Load the doc-ID table from the file header.
        n_docs = next(unpacker)
        to_external = [next(unpacker) for _ in range(n_docs)]
        for fp in unpacker:
            try:
                n = next(unpacker)
                docs = np.frombuffer(next(unpacker), dtype=np.uint32).tolist()
                poss = np.frombuffer(next(unpacker), dtype=np.uint32).tolist()
            except StopIteration as error:
                raise ValueError(f"Truncated posting record in {file_path}") from error
            if len(docs) != n or len(poss) != n:
                raise ValueError(f"Invalid posting length in {file_path}")
            postings_list = [(to_external[d], p) for d, p in zip(docs, poss)]
            yield fp, postings_list


def _stream_file_to_queue(file_path: str, q: mp.Queue, batch_size: int = 200):
    """Parse in a child, forwarding bounded batches and a completion sentinel."""
    batch = []
    for row in _iter_index_file(file_path):
        batch.append(row)
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
    if len(file_paths) == 1:
        yield from _iter_index_file(file_paths[0])
        return
    queues = [mp.Queue(maxsize=queue_depth) for _ in file_paths]
    procs = [
        mp.Process(target=_stream_file_to_queue, args=(f, q), daemon=True)
        for f, q in zip(file_paths, queues)
    ]
    for p in procs:
        p.start()

    def _iter_queue(q: mp.Queue, process) -> Generator:
        """Drain a queue, unbatching items as they arrive."""
        while True:
            try:
                batch = q.get(timeout=1)
            except queue.Empty:
                if not process.is_alive():
                    raise RuntimeError(
                        f"Index reader failed for {file_paths[procs.index(process)]} "
                        f"(exitcode={process.exitcode})"
                    )
                continue
            if batch is None:
                return
            yield from batch

    try:
        yield from merge_streams([_iter_queue(q, p) for q, p in zip(queues, procs)])
        for p in procs:
            p.join()
            if p.exitcode:
                raise RuntimeError(f"Index reader failed (exitcode={p.exitcode})")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join()
        for q in queues:
            q.close()
            q.cancel_join_thread()


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
        "--length", type=int, default=4, help="k-gram length (default: 4)"
    )
    index_parser.add_argument(
        "--window_size", type=int, default=6, help="Winnowing window size (default: 6)"
    )
    index_parser.add_argument(
        "--stop_after", type=int, default=None, help="Stop after this many samples"
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

    args = parser.parse_args()

    if args.task == "index":
        index_file(
            args.corpus_file,
            length=args.length,
            window_size=args.window_size,
            output_dir=args.output_dir,
            stop_after=args.stop_after,
            nb_workers=args.nb_workers,
        )

    elif args.task == "merge":
        merge_indexes_from_dir(args.temp_index_dir, args.output_dir)
