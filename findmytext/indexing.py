"""Disk-based index of winnowed document fingerprints.

An index directory contains:
- meta.json: the fingerprint parameters (see winnower.Winnower.to_metadata)
- fingerprints.npy: sorted uint64 array with one entry per unique fingerprint
- offsets.npy / lengths.npy: byte offset in postings.dat and number of postings for
  each fingerprint
- postings.dat: concatenated postings lists, POSTING_DTYPE records (doc_id, position)
- doc_name_offsets.npy / doc_name_bytes.npy: UTF-8 document names, concatenated,
  indexed by internal doc_id
- fingerprint_blocks.npy: every FINGERPRINT_BLOCK-th fingerprint, the in-RAM top level
  of the lookup (derived from fingerprints.npy; rebuilt and cached if missing)

Directories in this format are written by index_builder.PostingBuffer.
"""

import heapq
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import orjson  # 3-5x faster JSON parsing
import tqdm

from . import winnower

POSTING_DTYPE = np.dtype([("doc_id", np.uint32), ("position", np.uint32)])

# Fingerprints per block of the two-level lookup: 512 uint64 = one 4 KiB page.
FINGERPRINT_BLOCK = 512
FINGERPRINT_BLOCKS_FILE = "fingerprint_blocks.npy"


def compute_fingerprint_blocks(
    fingerprints: np.ndarray, block: int = FINGERPRINT_BLOCK
) -> np.ndarray:
    """Return the first fingerprint of every block-sized block.

    Reads the (possibly memory-mapped) array in large sequential chunks: a strided
    ``fingerprints[::block]`` on a memmap would instead fault in every page one at a
    time, which takes minutes on a network filesystem.
    """
    chunk = block * (1 << 15)  # ~128 MiB of uint64 for the default block size
    parts = [
        np.asarray(fingerprints[i : i + chunk])[::block]
        for i in range(0, len(fingerprints), chunk)
    ]
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    return np.ascontiguousarray(np.concatenate(parts))


def write_fingerprint_blocks(index_dir: str) -> None:
    """Derive fingerprint_blocks.npy from fingerprints.npy in index_dir."""
    fingerprints = np.load(os.path.join(index_dir, "fingerprints.npy"), mmap_mode="r")
    np.save(
        os.path.join(index_dir, FINGERPRINT_BLOCKS_FILE),
        compute_fingerprint_blocks(fingerprints),
    )


class DiskBasedIndex:
    """Disk-based implementation of the inverted index for document fingerprinting.

    The index data (fingerprints, offsets, lengths, and document ID mapping) is stored
    on disk as memory-mapped files, and postings lists are stored in a separate binary
    file that is accessed on demand.
    """

    def __init__(self, index_dir: str, posting_read_threads: int = 1):
        """Load a positional index; one posting-read thread avoids creating a pool.

        Explicit larger counts are available for workloads that benefit from
        concurrent reads. This setting does not control index-build processes.
        """
        if posting_read_threads < 1:
            raise ValueError("posting_read_threads must be at least 1")
        self.index_dir = index_dir
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = orjson.loads(f.read())
        # Round-tripping through Winnower validates hash_method and normalises the keys.
        self._meta = winnower.Winnower.from_metadata(meta).to_metadata()

        # Load index arrays as memory-mapped files
        print("Loading index arrays...", end="", flush=True)
        self.fingerprints = np.load(
            os.path.join(index_dir, "fingerprints.npy"), mmap_mode="r"
        )
        self._lookup = _TwoLevelSearch(
            self.fingerprints, self._load_fingerprint_blocks(index_dir)
        )
        self.offsets = np.load(os.path.join(index_dir, "offsets.npy"), mmap_mode="r")
        self.lengths = np.load(os.path.join(index_dir, "lengths.npy"), mmap_mode="r")
        print("done.", flush=True)

        # Load doc-ID mapping as memory-mapped arrays wrapped in a lazy accessor
        print("Loading doc-ID mapping...", end="", flush=True)
        doc_name_offsets = np.load(
            os.path.join(index_dir, "doc_name_offsets.npy"), mmap_mode="r"
        )
        doc_name_bytes = np.load(
            os.path.join(index_dir, "doc_name_bytes.npy"), mmap_mode="r"
        )
        self.to_external_doc_id = _DocIdMap(doc_name_offsets, doc_name_bytes)
        print("done.", flush=True)

        # Raw fd (not a file object) so os.pread can be called concurrently from
        # multiple threads without a shared seek position.
        self._posting_fd = os.open(os.path.join(index_dir, "postings.dat"), os.O_RDONLY)
        self._io_pool = (
            ThreadPoolExecutor(max_workers=posting_read_threads)
            if posting_read_threads > 1 else None
        )

    def _load_fingerprint_blocks(self, index_dir: str) -> np.ndarray:
        """Top level of the fingerprint lookup, read from fingerprint_blocks.npy.

        Indexes built before that file existed get it computed (one sequential pass
        over fingerprints.npy) and cached next to the index if the directory is
        writable.
        """
        path = os.path.join(index_dir, FINGERPRINT_BLOCKS_FILE)
        expected = -(-len(self.fingerprints) // FINGERPRINT_BLOCK)
        if os.path.isfile(path):
            blocks = np.load(path)
            if len(blocks) == expected:
                return blocks
        blocks = compute_fingerprint_blocks(self.fingerprints)
        try:
            np.save(path, blocks)
        except OSError:
            pass  # read-only index directory: recompute on every load
        return blocks

    def get_metadata(self) -> dict:
        """Fingerprint parameters the index was built with (see ``Winnower.to_metadata``).

        Pass the result to ``Winnower.from_metadata`` to hash queries the same way
        as the indexed corpus.
        """
        return dict(self._meta)

    @property
    def num_documents(self) -> int:
        """Number of documents in the index."""
        return len(self.to_external_doc_id)

    @property
    def num_fingerprints(self) -> int:
        """Number of distinct fingerprints (i.e. posting lists) in the index."""
        return len(self.fingerprints)

    def get_closest_matches(
        self, fingerprints: np.ndarray, min_fingerprints=5, top_k: int = 5
    ) -> List[str]:
        """Given an array of query fingerprints, retrieve the closest matching
        documents from the index based on shared fingerprints.

        Arguments:
        fingerprints: Winnowed fingerprints of the query document.
        min_fingerprints: The minimum number of shared fingerprints required for a document to be
            considered a match (default: 5).
        top_k: The number of top matching documents to return based on shared fingerprint counts
            (default: 5).

        Returns:
        A list of document IDs corresponding to the closest matches in the index.

        """
        match_counts = self.get_match_counts(
            fingerprints, min_fingerprints=min_fingerprints, top_k=top_k
        )

        return list(match_counts.keys())

    def get_match_counts(
        self, fingerprints: np.ndarray, min_fingerprints: int = 5, top_k: int = 5
    ) -> Dict[str, int]:
        """Given an array of query fingerprints, retrieve the counts of shared
        fingerprints for each document in the index, and return the top-k matches that
        have at least the minimum number of shared fingerprints."""
        if self.to_external_doc_id is None or self.fingerprints is None:
            raise ValueError("Index not initialized")

        # Retrieve the postings lists for each fingerprint in the query
        # (NB: we ignore the positions in the postings here)
        postings = self._get_postings(fingerprints, only_doc_ids=True)

        if not postings:
            return {}  # Return an empty dictionary if no postings are found

        # Sparse counting: work only over doc_ids that actually appeared in postings.
        # Avoids allocating a full n_docs-length array (400 MB for 50M docs) on every query.
        all_doc_ids = np.concatenate(
            [np.unique(posting_arr) for posting_arr in postings.values()]
        )
        unique_docs, doc_counts = np.unique(all_doc_ids, return_counts=True)

        # Filter documents that have at least the minimum number of shared fingerprints
        # and return the top-k matches
        mask = doc_counts >= min_fingerprints
        candidates = unique_docs[mask]
        if len(candidates) == 0:
            return {}
        candidate_counts = doc_counts[mask]
        top_idx = heapq.nlargest(
            min(top_k, len(candidates)),
            range(len(candidates)),
            key=lambda i: candidate_counts[i],
        )
        return {
            self.to_external_doc_id[int(candidates[i])]: int(candidate_counts[i])
            for i in top_idx
        }

    def get_closest_matches_with_positions(
        self,
        query: np.ndarray,
        min_fingerprints=5,
        top_k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Dict[int, List[int]]]:
        """Given an array of query fingerprints, retrieve the closest matching documents
        from the index based on shared fingerprints, along with the positions of the
        shared fingerprints in the index for each matching document.

        Args:
            query: Winnowed fingerprints of the query document.
            min_fingerprints: The minimum number of shared fingerprints required for a document to be
            considered a match (default: 5).
            top_k: The number of top matching documents to return based on shared fingerprint counts
            (default: 5).
            verbose: Whether to display a progress bar when retrieving postings for the query fingerprints.

        Returns:
            A dictionary mapping external document IDs of the closest matches in the index to another
        dictionary that maps shared fingerprint values to lists of positions where those fingerprints
        occur in the index for that document.

        """
        if self.to_external_doc_id is None or self.fingerprints is None:
            raise ValueError("Index not initialized")

        if isinstance(query, np.ndarray):
            query_fps = query
        elif hasattr(query, "to_numpy"):
            query_fps = query.to_numpy()
        else:
            raise ValueError("query must be an array of fingerprints")

        # Retrieve the postings lists for each fingerprint in the query
        # (NB: we ignore the positions in the postings here)
        postings = self._get_postings(query_fps, only_doc_ids=False, verbose=verbose)

        if not postings:
            return {}  # Return an empty dictionary if no postings are found

        # Sparse counting: work only over doc_ids that actually appeared in postings.
        all_doc_ids = np.concatenate(
            [np.unique(posting_arr["doc_id"]) for posting_arr in postings.values()]
        )
        unique_docs, doc_counts = np.unique(all_doc_ids, return_counts=True)

        # Filter documents that have at least the minimum number of shared fingerprints
        # and return the top-k matches
        mask = doc_counts >= min_fingerprints
        candidates = unique_docs[mask]
        if len(candidates) == 0:
            return {}
        candidate_counts = doc_counts[mask]
        top_idx = heapq.nlargest(
            min(top_k, len(candidates)),
            range(len(candidates)),
            key=lambda i: candidate_counts[i],
        )
        top_docs = [int(candidates[i]) for i in top_idx]

        # For each of the top matching documents, retrieve the positions of the shared fingerprints in the index
        top_docs_external = {}
        for doc_id in top_docs:
            doc_id_external = self.to_external_doc_id[doc_id]
            top_docs_external[doc_id_external] = {}

            for fp, postings_array in postings.items():
                postings_with_doc = postings_array[postings_array["doc_id"] == doc_id]
                if len(postings_with_doc["position"]) > 0:
                    top_docs_external[doc_id_external][int(fp)] = postings_with_doc[
                        "position"
                    ].tolist()

        return top_docs_external

    def get_posting_counts(self, fingerprints: np.ndarray) -> np.ndarray:
        """Number of postings of each fingerprint in the index (0 when absent).

        Only the fingerprint and length arrays are consulted, so this is much
        cheaper than reading the postings themselves.
        """
        queries = np.asarray(fingerprints, dtype=np.uint64)
        counts = np.zeros(len(queries), dtype=np.int64)
        if not len(self.fingerprints):
            return counts
        idx = self._lookup(queries)
        idx_clipped = np.clip(idx, 0, len(self.fingerprints) - 1)
        found = (idx < len(self.fingerprints)) & (self.fingerprints[idx_clipped] == queries)
        counts[found] = self.lengths[idx_clipped[found]]
        return counts

    def _get_postings(
        self,
        fingerprints: np.ndarray,
        only_doc_ids: bool = False,
        verbose: bool = False,
    ) -> Dict[np.uint64, np.ndarray]:
        """Retrieve the postings lists for a given array of fingerprints from the index,
        returning a list of Numpy arrays containing document IDs and positions for each
        fingerprint.

        This method performs a vectorized lookup of the fingerprints in the index,
        retrieves the corresponding offsets and lengths for the postings in the posting
        file, and extracts the relevant postings for each fingerprint.
        """
        queries = np.asarray(fingerprints, dtype=np.uint64)
        if not len(self.fingerprints):
            return {}  # an index of fingerprint-less documents has nothing to look up

        # 1. vectorized lookup (two-level search keeps page faults to ~1 per query)
        idx = self._lookup(queries)

        # 2. keep only matches — clip first so the index lookup is always in bounds
        idx_clipped = np.clip(idx, 0, len(self.fingerprints) - 1)
        mask = (idx < len(self.fingerprints)) & (
            self.fingerprints[idx_clipped] == queries
        )
        idx = idx_clipped[mask]

        # 3. get offsets and lengths
        offsets = self.offsets[idx]
        lengths = self.lengths[idx]

        found_fingerprints = queries[mask]

        # 4. sort by file offset to favour sequential I/O and reduce random seeks
        order = np.argsort(offsets)
        found_fingerprints = found_fingerprints[order]
        offsets = offsets[order]
        lengths = lengths[order]

        # os.pread avoids a shared seek position in the optional threaded path.
        def _read_one(args):
            fp, offset, length = args
            data = os.pread(self._posting_fd, int(length) * 8, int(offset))
            return fp, data

        results = {}
        read_args = zip(found_fingerprints, offsets, lengths)
        read_iter = (
            self._io_pool.map(_read_one, read_args)
            if self._io_pool is not None else map(_read_one, read_args)
        )
        if verbose:
            read_iter = tqdm.tqdm(
                read_iter, total=len(found_fingerprints), desc="Retrieving postings"
            )
        for fp, data in read_iter:
            postings_array = np.frombuffer(
                data, dtype=[("doc_id", np.uint32), ("position", np.uint32)]
            )
            if only_doc_ids:
                results[fp] = postings_array["doc_id"]
            else:
                results[fp] = postings_array

        return results

    def __del__(self):
        """Close the posting file when the DiskBasedIndex instance is deleted to free
        up system resources."""
        if hasattr(self, "_posting_fd") and self._posting_fd >= 0:
            os.close(self._posting_fd)
            self._posting_fd = -1
        if hasattr(self, "_io_pool") and self._io_pool is not None:
            self._io_pool.shutdown(wait=False)


class _TwoLevelSearch:
    """searchsorted over a memory-mapped sorted array with one page fault per query.

    A plain binary search over a 16 GB fingerprint array touches ~30 random pages per
    query.  Here the array is split into blocks of `block` entries (one 4 KiB page for
    uint64); the first entry of each block is kept in RAM (0.2% of the array), located
    with an in-memory binary search, and only the matching block is then searched on
    disk.  Positions returned are identical to np.searchsorted(array, queries).
    """

    def __init__(self, array: np.ndarray, block_starts: np.ndarray = None,
                 block: int = FINGERPRINT_BLOCK):
        self.array = array
        self.block = block
        if block_starts is None:
            block_starts = compute_fingerprint_blocks(array, block)
        self.block_starts = block_starts  # in RAM, 0.2% of the array

    def __call__(self, queries: np.ndarray) -> np.ndarray:
        # Block b holds array[b*block:(b+1)*block]; the query belongs to the last block
        # whose first entry is <= query, i.e. searchsorted(side="right") - 1.
        blocks = np.searchsorted(self.block_starts, queries, side="right") - 1
        blocks = np.maximum(blocks, 0)
        result = np.empty(len(queries), dtype=np.int64)
        for b in np.unique(blocks):  # queries are grouped by block: one page read each
            in_block = blocks == b
            start = int(b) * self.block
            chunk = self.array[start : start + self.block]
            result[in_block] = start + np.searchsorted(chunk, queries[in_block])
        return result


class _DocIdMap:
    """Disk-backed mapping from internal integer document IDs to external string
    document IDs, backed by a uint64 offset table and a flat uint8 UTF-8 byte buffer
    that can be memory-mapped so they consume no additional RAM beyond the OS page
    cache."""

    def __init__(self, offsets: np.ndarray, flat: np.ndarray):
        self._offsets = offsets  # uint64, shape (n_docs + 1,)
        self._flat = flat  # uint8, flat UTF-8 byte buffer

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, i: int) -> str:
        start = int(self._offsets[i])
        end = int(self._offsets[i + 1])
        return self._flat[start:end].tobytes().decode("utf-8")
