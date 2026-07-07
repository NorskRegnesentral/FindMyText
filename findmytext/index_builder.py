"""Scripts for building indexes from .jsonl or .jsonl.zst files containing documents,
and for merging intermediate indexes into a final disk-based index.

The indexing process consists of two main steps:

1. Building memory-based indexes from the corpus: The `do_indexing` function reads
documents from the input data file (which can be in .jsonl or .jsonl.zst format),
processes them to create an index based on k-grams and winnowing, and saves intermediate
indexes to gzipped JSONL files after processing a specified number of documents. Each
intermediate index file contains a mapping from fingerprints to postings lists,
where each postings list includes document IDs and positions.

2. Merging intermediate indexes into a single disk-based index: The `merge_indexes` function
takes the intermediate index files generated in the first step, reads and merges their contents,
and saves the final merged index to a specified output directory. The merged index consists
of several files, including a binary postings file, NumPy arrays for fingerprints, offsets,
and lengths, and a JSON file containing index metadata.
"""

import argparse
import concurrent.futures
import heapq
import multiprocessing as mp
import os
import shutil
import sys
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import datasets
import numpy as np
import orjson

from . import indexing, utils

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
    intermediate_save_freq: int = 300_000,
    stop_after: Optional[int] = None,
    delete_existing: bool = False,
    nb_workers: int = 1,
) -> List[str]:
    """Index documents from a data file (currently supported formats: .jsonl,
    .jsonl.zst) and save intermediate indexes (every intermediate_save_freq documents).

    Args:
        corpus_file (str): Path to the input data file.  The corpus file should be in .jsonl format
         (optionally gzipped or zst-compressed), where each line is a JSON object with at least two fields:
        "id" (a unique document identifier) and "text" (the document content to be indexed).
        output_dir (str): Directory where the index files will be saved.
        length (int): Length of k-grams to use for indexing.
        window_size (int): Window size for winnowing.
        intermediate_save_freq (int): Frequency (in number of documents) at which to save intermediate indexes.
        stop_after (Optional[int]): If provided, stop indexing after this many documents (for testing purposes).
        delete_existing (bool): whether to delete the output_dir if it already exists
        nb_workers (int): Number of parallel worker processes to use for indexing. If set to 1, indexing will
         be done in a single process.

    Returns:
        List of paths to the saved intermediate index files.

    """

    if not os.path.isfile(corpus_file):
        raise ValueError(f"Provided file {corpus_file} does not exist or is not a file")

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

    if nb_workers == 1:
        index_files = index_data(
            data_stream=stream,
            output_dir=output_dir,
            length=length,
            window_size=window_size,
            intermediate_save_freq=intermediate_save_freq,
            stop_after=stop_after,
            delete_existing=delete_existing,
        )
    else:
        if stop_after is not None:
            raise ValueError(
                "stop_after is not supported when using multiple workers. Please set nb_workers=1 to use stop_after."
            )

        index_files = index_data_parallel(
            data=stream,
            output_dir=output_dir,
            n_workers=nb_workers,
            length=length,
            window_size=window_size,
            intermediate_save_freq=intermediate_save_freq,
            delete_existing=delete_existing,
        )
    return index_files


def index_data(
    data_stream: Any,
    output_dir: str,
    length=4,
    window_size=6,
    intermediate_save_freq: int = 300_000,
    stop_after: Optional[int] = None,
    delete_existing: bool = False,
    file_suffix: str = "",
) -> List[str]:
    """Index documents from a corpus and save intermediate indexes (every intermediate_save_freq documents).

    Args:
        data_stream (Iterator[Dict[str, Any]]): An iterator over the input data. Each item should be a dictionary
         with at least two fields: "id" (a unique document identifier) and "text" (the document content to be indexed).
        output_dir (str): Directory where the index files will be saved.
        length (int): Length of k-grams to use for indexing.
        window_size (int): Window size for winnowing.
        intermediate_save_freq (int): Frequency (in number of documents) at which to save intermediate indexes.
        stop_after (Optional[int]): If provided, stop indexing after this many documents (for testing purposes).
        delete_existing (bool): whether to delete the output_dir if it already exists
        file_suffix (str): A suffix to add to the index files. Default is no suffix

    Returns:
        List of paths to the saved intermediate index files.

    """

    if os.path.exists(output_dir):
        if delete_existing:
            print(f"Deleting existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        elif not os.path.isdir(output_dir):
            raise ValueError(f"Output path {output_dir} exists and is not a directory")
    os.makedirs(output_dir, exist_ok=True)
    print("Saving index files to", output_dir)

    index = indexing.MemoryBasedIndex(length=length, window_size=window_size)
    index_files = []
    for i, result in enumerate(data_stream):
        result = utils.normalise_json(result)

        index.add_doc(result["text"], result["id"])
        if stop_after is not None and i + 1 >= stop_after:
            break
        if (i + 1) % intermediate_save_freq == 0:
            increment_str = str(i + 1)[:-3] + "K"
            if file_suffix:
                index_filename = "intermediate-%s-%s.jsonl.gz" % (
                    file_suffix,
                    increment_str,
                )
            else:
                index_filename = "intermediate-%s.jsonl.gz" % (increment_str)
            output_file = os.path.join(output_dir, index_filename)
            print("Saving intermediate index to", output_file, end="...", flush=True)
            index.to_jsonl(output_file)
            index_files.append(output_file)
            print("Done.")
            del index
            index = indexing.MemoryBasedIndex(length=length, window_size=window_size)

    if file_suffix:
        index_filename = "final-%s.jsonl.gz" % (file_suffix)
    else:
        index_filename = "final.jsonl.gz"
    final_file = os.path.join(output_dir, index_filename)
    print("Saving final index to", final_file, end="...", flush=True)
    index.to_jsonl(final_file)
    index_files.append(final_file)
    print("Done.")

    return index_files


def index_data_parallel(
    data: Any,
    output_dir: str,
    n_workers: int,
    length=4,
    window_size=6,
    intermediate_save_freq: int = 300_000,
    delete_existing: bool = False,
) -> List[str]:
    """Index documents in parallel by splitting data across n_workers processes.

    Args:
        data (List[Dict[str, Any]]): The full dataset to index. Each item must have "id" and "text" fields.
        output_dir (str): Directory where the index files will be saved.
        n_workers (int): Number of parallel worker processes.
        length (int): Length of k-grams to use for indexing.
        window_size (int): Window size for winnowing.
        intermediate_save_freq (int): Frequency at which to save intermediate indexes per worker.
        delete_existing (bool): whether to delete the output_dir if it already exists

    Returns:
        Flat list of all intermediate index file paths produced by all workers.

    """

    if os.path.exists(output_dir):
        if delete_existing:
            print(f"Deleting existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        elif not os.path.isdir(output_dir):
            raise ValueError(f"Output path {output_dir} exists and is not a directory")
    os.makedirs(output_dir, exist_ok=True)
    print("And saving index files to", output_dir)

    if isinstance(data, datasets.Dataset):
        data = list(data)

    split_size = (len(data) + n_workers - 1) // n_workers
    print(
        "Indexing %d documents across %d workers..." % (len(data), n_workers),
        flush=True,
    )
    worker_args = [
        (
            i + 1,
            data[i * split_size : (i + 1) * split_size],
            output_dir,
            length,
            window_size,
            intermediate_save_freq,
        )
        for i in range(n_workers)
    ]

    all_files = []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for files in pool.imap_unordered(_parallel_worker, worker_args):
            all_files.extend(files)
            print("  indexed -> %s" % files[-1], flush=True)
    return all_files


def _parallel_worker(args: Tuple) -> List[str]:
    """Module-level worker for index_data_parallel (must be picklable)."""
    worker_idx, split, output_dir, length, window_size, intermediate_save_freq = args
    sys.stdout = open(os.devnull, "w")
    return index_data(
        iter(split),
        output_dir,
        length=length,
        window_size=window_size,
        intermediate_save_freq=intermediate_save_freq,
        file_suffix=f"worker{worker_idx:02d}",
    )


#####################################################
# STEP 2: MERGE INTO SINGLE DISK-BASED INDEX
#####################################################


def merge_indexes_from_dir(temp_index_dir: str, output_dir: str):
    """Merge the index files in the specified directory and store the result
    in output_dir. The index files in temp_index_dir must be in .jsonl.gz format."""

    if not os.path.isdir(temp_index_dir):
        raise ValueError(f"Provided path {temp_index_dir} is not a directory")
    index_files = []
    for f in os.listdir(temp_index_dir):
        if f.endswith(".jsonl.gz"):
            index_files.append(os.path.join(temp_index_dir, f))
    if not index_files:
        raise ValueError(f"No .jsonl.gz index files found in {temp_index_dir}")
    merge_indexes(index_files, output_dir)


def merge_indexes(
    index_files: List[str], output_dir: str, save_every_n: int = 10_000_000
):
    """Merge multiple intermediate index files (in gzipped JSONL format) into a single
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
        raise ValueError("At least one JSONL file must be provided to load the index")

    os.makedirs(output_dir, exist_ok=True)

    # Peek at the first line of the first JSONL file to read the index parameters,
    # and save them to the output directory in a meta.json file.
    with gzip.open(index_files[0], "rt", encoding="utf-8") as f:
        meta = orjson.loads(f.readline())
        meta = {
            k: meta[k]
            for k in ["length", "window_size", "base", "punctuation"]
            if k in meta
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

    # Parse all files in parallel processes and merge the sorted streams
    merged_stream = merge_streams_parallel(index_files)

    # Large write buffer (8 MB) amortises syscall overhead across millions of small writes.
    current_offset = 0
    with open(posting_file_path, "wb", buffering=8 * 1024 * 1024) as posting_file:
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
    """Scan all JSONL files in parallel and assign a stable internal integer ID to every
    unique external document ID."""
    n_workers = min(len(index_files), os.cpu_count() or 1)
    print(
        f"Collecting document IDs from {len(index_files)} files ({n_workers} workers)...",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        per_file = list(pool.map(_collect_doc_ids_from_jsonl, index_files))

    to_internal: Dict[str, int] = {}
    for doc_ids in per_file:
        for doc_id in doc_ids:
            if doc_id not in to_internal:
                internal_id = len(to_internal)
                to_internal[doc_id] = internal_id
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


def _collect_doc_ids_from_jsonl(file_path: str) -> List[str]:
    """Return all unique document IDs found in a gzipped JSONL index file.

    Module-level so it can be pickled for ProcessPoolExecutor.
    """
    seen: dict = {}
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        f.readline()  # skip metadata line
        for line in f:
            try:
                for posting in orjson.loads(line)["postings"]:
                    seen.setdefault(posting["doc_id"], None)
            except (orjson.JSONDecodeError, KeyError):
                continue
    return list(seen)


def _stream_file_to_queue(file_path: str, q: mp.Queue, batch_size: int = 200):
    """Worker process: parse one gzipped JSONL index file and push (fingerprint,
    postings_list) items in batches to a queue.

    Batching reduces IPC overhead; a None sentinel signals end of stream. Module-level
    so it can be pickled by multiprocessing.
    """
    batch = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        f.readline()  # skip metadata line
        for line in f:
            try:
                data = orjson.loads(line)
                fp = int(data["fingerprint"])
                postings_list = [
                    (posting["doc_id"], int(posting["position"]))
                    for posting in data["postings"]
                ]
            except (orjson.JSONDecodeError, KeyError, ValueError):
                print("Corrupted line in JSONL file, skipping:", line)
                continue
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
