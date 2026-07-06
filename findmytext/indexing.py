"""Module for in-memory and disk-based indexing of documents using winnowed
fingerprints."""

import gzip
import heapq
import os
from concurrent.futures import ThreadPoolExecutor

import orjson  # 3-5x faster JSON parsing
import tqdm

try:
    import isal.igzip as igzip  # 2-4x faster gzip decompression (Intel ISA-L)
except ImportError:
    igzip = gzip  # type: ignore[assignment]

from typing import Dict, List, Union

import numpy as np

from . import winnower


class MemoryBasedIndex:
    """Memory-based implementation of the inverted index for document fingerprinting.

    This class stores the index entirely in memory using Python dictionaries.
    """

    def __init__(
        self, length: int = 5, window_size: int = 6, base: int = 256, punctuation=False
    ):
        """Initialize the MemoryBasedIndex with specified winnowing parameters."""

        # We use two winnowers: one for indexing and one for runtime queries
        # (the latter with a window size of 1, since we want to get as many
        # fingerprints as possible for matching).
        self.indexing_winnower = winnower.Winnower(
            length=length, window_size=window_size, base=base, punctuation=punctuation
        )
        self.runtime_winnower = winnower.Winnower(
            length=length, window_size=1, base=base, punctuation=punctuation
        )

        self.index = {}

    def add_doc(self, text: str, unique_id: str):
        """Add a document to the inverted index with a unique identifier."""
        # Get the winnowed fingerprints and their positions for the document text
        fingerprints, positions = self.indexing_winnower.get_winnowed_fingerprints(text)

        # Add each fingerprint and its position to the index
        for fp, pos in zip(fingerprints, positions):
            fp = int(fp)
            pos = int(pos)
            if fp not in self.index:
                self.index[fp] = [(unique_id, pos)]
            else:
                self.index[fp].append((unique_id, pos))
        return self

    def get_closest_matches(
        self, query_text: str, min_fingerprints=5, top_k: int = 5
    ) -> List[str]:
        """Given a query text, compute its winnowed fingerprints and retrieve the
        closest matching documents from the index based on shared fingerprints.

        Args:
            query_text: The text of the query document to find matches for.
            min_fingerprints: The minimum number of shared fingerprints required for a document to be
            considered a match (default: 5).
            top_k: The number of top matching documents to return based on shared fingerprint counts
            (default: 5).

        Returns:
        - A list of document IDs corresponding to the closest matches in the index.

        """
        # Compute the winnowed fingerprints for the query text
        query_fps, _ = self.runtime_winnower.get_winnowed_fingerprints(query_text)

        # Retrieve the counts of shared fingerprints for each document in the index, and return the top-k
        # matches that have at least the minimum number of shared fingerprints
        match_counts = self.get_match_counts(
            query_fps, min_fingerprints=min_fingerprints, top_k=top_k
        )

        return list(match_counts.keys())

    def get_match_counts(
        self, fingerprints: np.ndarray, min_fingerprints: int = 5, top_k: int = 5
    ) -> Dict[str, int]:
        """Given an array of query fingerprints, retrieve the counts of shared
        fingerprints for each document in the index, and return the top-k matches that
        have at least the minimum number of shared fingerprints."""
        # Retrieve the postings lists for each fingerprint in the query
        # (NB: we ignore the positions in the postings here)
        doc_counts = {}
        for fp in fingerprints:
            if fp in self.index:
                for doc_id, _ in self.index[fp]:
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        # Filter documents that have at least the minimum number of shared fingerprints
        # and return the top-k matches
        candidates = [
            doc_id for doc_id, count in doc_counts.items() if count >= min_fingerprints
        ]
        top_docs = heapq.nlargest(top_k, candidates, key=lambda x: doc_counts[x])

        return {doc_id: doc_counts[doc_id] for doc_id in top_docs}

    def get_fingerprint_positions(
        self, fingerprints: np.ndarray, doc_id: str
    ) -> Dict[int, List[int]]:
        """Given an array of query fingerprints and a document ID, retrieve the
        positions of the shared fingerprints for that document in the index."""
        positions = {}
        for fp in fingerprints:
            if fp in self.index:
                for doc_id2, pos in self.index[fp]:
                    if doc_id2 == doc_id:
                        if fp not in positions:
                            positions[fp] = []
                        positions[fp].append(pos)

        return positions

    def to_jsonl(self, output_file: str):
        """Serialize the in-memory index to a gzipped JSONL file at the specified output
        path."""
        # Sort the fingerprints in the index to ensure a consistent order in the output file
        sorted_fps = np.array(list(self.index.keys()), dtype=np.uint64)
        sorted_fps.sort()

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            # The first line of the JSONL file contains metadata about the index parameters and statistics,
            metadata_line = orjson.dumps(
                {
                    "length": self.indexing_winnower.length,
                    "window_size": self.indexing_winnower.window_size,
                    "base": self.indexing_winnower.base,
                    "punctuation": self.indexing_winnower.punctuation,
                    "num_fingerprints": len(self.index),
                }
            ).decode("utf-8")
            f.write(metadata_line + "\n")

            # Write each fingerprint and its postings list to the JSONL file
            for fp in sorted_fps:
                postings = self.index[fp]
                postings_list = [
                    {"doc_id": doc_id, "position": int(pos)} for doc_id, pos in postings
                ]
                json_line = orjson.dumps(
                    {"fingerprint": int(fp), "postings": postings_list}
                ).decode("utf-8")
                f.write(json_line + "\n")

    @classmethod
    def from_jsonl(cls, input_file: str, only_meta_data=False) -> "MemoryBasedIndex":
        """Deserialize an in-memory index from a gzipped JSONL file at the specified
        input path, and create a MemoryBasedIndex instance with the loaded data."""
        with igzip.open(input_file, "rt", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError("Input file is empty")

            # The first line of the JSONL file contains metadata about the index parameters and statistics
            metadata = orjson.loads(first_line)
            index = cls(
                length=metadata["length"],
                window_size=metadata["window_size"],
                base=metadata["base"],
                punctuation=metadata["punctuation"],
            )

            if only_meta_data:
                return index

            num_fingerprints = metadata.get("num_fingerprints", None)
            # Load each fingerprint and its postings list from the JSONL file into the index
            for line in tqdm.tqdm(
                f, total=num_fingerprints, desc="Loading index from JSONL"
            ):
                data = orjson.loads(line)
                fp = int(data["fingerprint"])
                postings_list = [
                    (posting["doc_id"], int(posting["position"]))
                    for posting in data["postings"]
                ]
                index.index[fp] = postings_list

            return index


class DiskBasedIndex:
    """Disk-based implementation of the inverted index for document fingerprinting.

    The index data (fingerprints, offsets, lengths, and document ID mapping) is stored
    on disk as memory-mapped files, and postings lists are stored in a separate binary
    file that is accessed on demand.
    """

    def __init__(self, index_dir: str):
        """Initialize the disk-based index by loading the metadata and memory-mapped
        files from the specified index directory."""
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = orjson.loads(f.read())
            self.indexing_winnower = winnower.Winnower(
                length=meta["length"],
                window_size=meta["window_size"],
                base=meta["base"],
                punctuation=meta["punctuation"],
            )
            self.runtime_winnower = winnower.Winnower(
                length=meta["length"],
                window_size=1,
                base=meta["base"],
                punctuation=meta["punctuation"],
            )

        # Load index arrays as memory-mapped files (zero RAM cost; served from OS page cache)
        self.fingerprints = np.load(
            os.path.join(index_dir, "fingerprints.npy"), mmap_mode="r"
        )
        self.offsets = np.load(os.path.join(index_dir, "offsets.npy"), mmap_mode="r")
        self.lengths = np.load(os.path.join(index_dir, "lengths.npy"), mmap_mode="r")

        # Load doc-ID mapping as memory-mapped arrays wrapped in a lazy accessor
        doc_name_offsets = np.load(
            os.path.join(index_dir, "doc_name_offsets.npy"), mmap_mode="r"
        )
        doc_name_bytes = np.load(
            os.path.join(index_dir, "doc_name_bytes.npy"), mmap_mode="r"
        )
        self.to_external_doc_id = _DocIdMap(doc_name_offsets, doc_name_bytes)

        # Raw fd (not a file object) so os.pread can be called concurrently from
        # multiple threads without a shared seek position.
        self._posting_fd = os.open(os.path.join(index_dir, "postings.dat"), os.O_RDONLY)
        self._io_pool = ThreadPoolExecutor(max_workers=8)

    def get_closest_matches(
        self, query_text: str, min_fingerprints=5, top_k: int = 5
    ) -> List[str]:
        """Given a query text, compute its winnowed fingerprints and retrieve the
        closest matching documents from the index based on shared fingerprints.

        Arguments:
        query_text: The text of the query document to find matches for.
        min_fingerprints: The minimum number of shared fingerprints required for a document to be
            considered a match (default: 5).
        top_k: The number of top matching documents to return based on shared fingerprint counts
            (default: 5).

        Returns:
        A list of document IDs corresponding to the closest matches in the index.

        """
        # Compute the winnowed fingerprints for the query text
        query_fps, _ = self.runtime_winnower.get_winnowed_fingerprints(query_text)

        # Retrieve the counts of shared fingerprints for each document in the index, and return the top-k
        # matches that have at least the minimum number of shared fingerprints
        match_counts = self.get_match_counts(
            query_fps, min_fingerprints=min_fingerprints, top_k=top_k
        )

        return list(match_counts.keys())

    def get_match_counts(
        self, fingerprints: np.ndarray, min_fingerprints: int = 5, top_k: int = 5
    ) -> Dict[str, int]:
        """Given an array of query fingerprints, retrieve the counts of shared
        fingerprints for each document in the index, and return the top-k matches that
        have at least the minimum number of shared fingerprints."""
        if (
            self.runtime_winnower is None
            or self.to_external_doc_id is None
            or self.fingerprints is None
        ):
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
        query: Union[str, np.ndarray],
        min_fingerprints=5,
        top_k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Dict[int, List[int]]]:
        """Given a query text, compute its winnowed fingerprints and retrieve the
        closest matching documents from the index based on shared fingerprints, along
        with the positions of the shared fingerprints in the index for each matching
        document.

        Args:
            query: The text of the query document to find matches for, or an array of precomputed
             fingerprints for that text.
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
        if (
            self.runtime_winnower is None
            or self.to_external_doc_id is None
            or self.fingerprints is None
        ):
            raise ValueError("Index not initialized")

        # Compute the winnowed fingerprints for the query text
        if isinstance(query, str):
            query_fps, _ = self.runtime_winnower.get_winnowed_fingerprints(query)
        elif isinstance(query, np.ndarray):
            query_fps = query
        elif hasattr(query, "to_numpy"):
            query_fps = query.to_numpy()
        else:
            raise ValueError("query must be a string or an array of fingerprints")

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

        # 1. vectorized lookup
        idx = np.searchsorted(self.fingerprints, queries)

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

        # 5. parallel reads: os.pread releases the GIL so threads overlap I/O;
        #    no shared seek position means reads are safe to issue concurrently.
        def _read_one(args):
            fp, offset, length = args
            data = os.pread(self._posting_fd, int(length) * 8, int(offset))
            return fp, data

        results = {}
        read_iter = self._io_pool.map(
            _read_one, zip(found_fingerprints, offsets, lengths)
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
        if hasattr(self, "_io_pool"):
            self._io_pool.shutdown(wait=False)


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
