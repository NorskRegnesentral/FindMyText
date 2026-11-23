"""Disk-based index of winnowed document fingerprints.

An index directory contains:
- meta.json: the fingerprint parameters (see winnower.Winnower.to_metadata)
- fingerprints.npy: sorted uint64 array with one entry per unique fingerprint
- offsets.npy / lengths.npy: byte offset in postings.dat and number of postings for
  each fingerprint
- postings.dat: concatenated postings lists, POSTING_DTYPE records (doc_id, position)
- doc_name_offsets.npy / doc_name_bytes.npy: UTF-8 document names, concatenated,
  indexed by internal doc_id
- fingerprint_blocks.npy: every FINGERPRINT_BLOCK-th entry of fingerprints.npy, the
  in-RAM top level of the fingerprint lookup

Directories in this format are written by index_builder.PostingBuffer.
"""

import heapq
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

import numpy as np
import orjson  # 3-5x faster JSON parsing
import tqdm

from . import winnower

POSTING_DTYPE = np.dtype([("doc_id", np.uint32), ("position", np.uint32)])

# One row of Hits.table: a query fingerprint found at a position of a candidate doc.
HIT_DTYPE = np.dtype([("doc", np.uint32), ("hash", np.uint64), ("position", np.uint32)])

# Fingerprints per block of the two-level lookup: 512 uint64 = one 4 KiB page.
FINGERPRINT_BLOCK = 512
FINGERPRINT_BLOCKS_FILE = "fingerprint_blocks.npy"


@dataclass(frozen=True, eq=False)
class Hits:
    """Occurrences of query fingerprints in the closest candidate documents.

    Rows are grouped by document in closeness order (most shared fingerprints first) 
    and sorted by ``(hash, position)`` within a document, so the hits of candidate 
    ``i`` are ``table[starts[i]:starts[i + 1]]`` (a view, do not mutate) 
    and ``doc_ids[i]`` is its external id.

    Attributes
    ----------
    table : np.ndarray
        Structured array of :data:`HIT_DTYPE` (``doc``, ``hash``, ``position``).
    starts : np.ndarray
        int64 block boundaries, ``len(doc_ids) + 1`` entries, from 0 to ``len(table)``.
    doc_ids : list[str]
        External ids of the candidate documents, in closeness order.

    """

    table: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=HIT_DTYPE))
    starts: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    doc_ids: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def block(self, i: int) -> np.ndarray:
        """Hits of the ``i``-th candidate document (a view into ``table``)."""
        return self.table[self.starts[i] : self.starts[i + 1]]

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over ``(doc_id, block)`` pairs in closeness order."""
        for i, doc_id in enumerate(self.doc_ids):
            yield doc_id, self.block(i)

    def to_dict(self) -> Dict[str, Dict[int, List[int]]]:
        """The nested ``{doc_id: {hash: [positions]}}`` form of the same hits."""
        result: Dict[str, Dict[int, List[int]]] = {}
        for doc_id, block in self:
            hashes = block["hash"]
            # Blocks are sorted by hash: a run of equal hashes is one fingerprint.
            bounds = np.flatnonzero(np.r_[True, hashes[1:] != hashes[:-1], True])
            positions = block["position"]
            result[doc_id] = {
                int(hashes[lo]): positions[lo:hi].tolist()
                for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist())
            }
        return result


class DiskBasedIndex:
    """Disk-based implementation of the inverted index for document fingerprinting.

    The index data (fingerprints, offsets, lengths, and document ID mapping) is stored
    on disk as memory-mapped files, and postings lists are stored in a separate binary
    file that is accessed on demand.
    """

    def __init__(self, index_dir: str, n_workers: int = 1):
        """Load a positional index; ``n_workers`` threads read posting lists in
        parallel, and the default of one avoids creating a pool.

        Larger counts are for workloads that benefit from concurrent reads (cold
        or network storage).  This setting does not control index-build processes.
        """
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
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
            self.fingerprints, np.load(os.path.join(index_dir, FINGERPRINT_BLOCKS_FILE))
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
            ThreadPoolExecutor(max_workers=n_workers)
            if n_workers > 1 else None
        )

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

    def get_closest_hits(
        self,
        query: np.ndarray,
        min_fingerprints: int = 5,
        top_k: int = 5,
        verbose: bool = True,
    ) -> Hits:
        """Retrieve the documents sharing the most fingerprints with ``query``, with
        every occurrence of a shared fingerprint, as one flat :class:`Hits` table.

        Args:
            query: Winnowed fingerprints of the query document (uint64 array, or an
                object with ``to_numpy``).
            min_fingerprints: Minimum number of distinct shared fingerprints for a
                document to be a candidate (default: 5).
            top_k: Number of candidates to return, by descending shared-fingerprint
                count (default: 5).
            verbose: Display a progress bar while reading the posting lists.

        Returns:
            The hits of the top-k candidates (empty :class:`Hits` when none qualifies).

        """
        if self.to_external_doc_id is None or self.fingerprints is None:
            raise ValueError("Index not initialized")

        if isinstance(query, np.ndarray):
            query_fps = query
        elif hasattr(query, "to_numpy"):
            query_fps = query.to_numpy()
        else:
            raise ValueError("query must be an array of fingerprints")

        postings = self._get_postings(query_fps, only_doc_ids=False, verbose=verbose)
        if not postings:
            return Hits()

        # One flat (doc, hash, position) table over every posting read, sorted by
        # (doc, hash, position) so that each document, and each fingerprint within
        # it, is a contiguous run.
        arrays = list(postings.values())
        fps = np.fromiter(postings.keys(), dtype=np.uint64, count=len(arrays))
        lengths = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
        flat = np.concatenate(arrays)
        hashes = np.repeat(fps, lengths)
        order = np.lexsort((flat["position"], hashes, flat["doc_id"]))
        docs, hashes, positions = flat["doc_id"][order], hashes[order], flat["position"][order]

        # Shared-fingerprint count per document = number of distinct (doc, hash)
        # runs; a fingerprint occurring several times in a document counts once.
        new_run = np.r_[True, (docs[1:] != docs[:-1]) | (hashes[1:] != hashes[:-1])]
        unique_docs, doc_counts = np.unique(docs[new_run], return_counts=True)

        mask = doc_counts >= min_fingerprints
        candidates = unique_docs[mask]
        if len(candidates) == 0:
            return Hits()
        candidate_counts = doc_counts[mask]
        top_idx = heapq.nlargest(
            min(top_k, len(candidates)),
            range(len(candidates)),
            key=lambda i: candidate_counts[i],
        )
        top_docs = candidates[top_idx]

        # Gather the (already sorted) run of each top document, in closeness order.
        lo = np.searchsorted(docs, top_docs, side="left")
        hi = np.searchsorted(docs, top_docs, side="right")
        starts = np.zeros(len(top_docs) + 1, dtype=np.int64)
        np.cumsum(hi - lo, out=starts[1:])
        rows = np.concatenate([np.arange(a, b) for a, b in zip(lo.tolist(), hi.tolist())])
        table = np.empty(len(rows), dtype=HIT_DTYPE)
        table["doc"] = docs[rows]
        table["hash"] = hashes[rows]
        table["position"] = positions[rows]
        doc_ids = [self.to_external_doc_id[int(d)] for d in top_docs]
        return Hits(table, starts, doc_ids)

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

        This is :meth:`get_closest_hits` in nested-dictionary form.

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
        return self.get_closest_hits(
            query, min_fingerprints=min_fingerprints, top_k=top_k, verbose=verbose
        ).to_dict()

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

    ``block_starts`` must equal ``array[::block]`` (index_builder writes it to
    fingerprint_blocks.npy).
    """

    def __init__(self, array: np.ndarray, block_starts: np.ndarray,
                 block: int = FINGERPRINT_BLOCK):
        self.array = array
        self.block = block
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
