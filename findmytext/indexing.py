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

from typing import Any, Dict, List, Tuple

import msgpack
import numpy as np

from . import winnower


class MemoryBasedIndex:
    """In-memory index that accumulates all (fingerprint, doc_id, position) triples in
    a single flat numpy structured array (16 bytes per entry).

    The implementation is optimised for fast insertion of fingerprints, and to
    efficiently export the index to disk (in MsgPack format). It is, however,
    not suitable for very large datasets that exceed available RAM. For larger
    datasets, use the DiskBasedIndex.
    """

    _DTYPE = np.dtype([("fp", np.uint64), ("doc", np.uint32), ("pos", np.uint32)])
    _CHUNK = 10_000_000  # grow in 10 M-entry (~160 MB) steps

    def __init__(self, meta: Dict[str, Any]):
        self.doc_id_map: Dict[str, int] = {}
        self.to_external: List[str] = []
        self.meta = meta

        # Preallocate a large numpy structured array to hold the index entries.
        # # The array will grow in chunks as needed.
        self.index_array: np.ndarray = np.empty(self._CHUNK, dtype=self._DTYPE)
        self.index_size: int = 0  # number of valid entries in index_array

    def add_fingerprints(self, doc_id: str, fp_pos_list: List[Tuple[int, int]]):
        """Append fingerprint-position pairs for a document to the posting buffer."""

        # We use a mapping from external document IDs (strings) to internal integer IDs
        # to save space in the index.
        int_id = self.doc_id_map.get(doc_id)
        if int_id is None:
            int_id = len(self.to_external)
            self.doc_id_map[doc_id] = int_id
            self.to_external.append(doc_id)

        # If the current index is not large enough to hold the new entries, we grow it
        n = len(fp_pos_list)
        if self.index_size + n > len(self.index_array):
            extra = max(self._CHUNK, n)
            new_buf = np.empty(self.index_size + extra, dtype=self._DTYPE)
            new_buf[: self.index_size] = self.index_array[: self.index_size]
            self.index_array = new_buf  # old buffer freed when refcount drops to 0

        # Append the new fingerprint-position pairs to the index array.
        for fp, pos in fp_pos_list:
            self.index_array[self.index_size] = (int(fp), int_id, int(pos))
            self.index_size += 1

    @property
    def doc_count(self) -> int:
        """Return the number of unique documents indexed so far."""
        return len(self.to_external)

    def get_closest_matches(
        self, fingerprints: np.ndarray, min_fingerprints=5, top_k: int = 5
    ) -> List[str]:
        """Given an array of query fingerprints, retrieve the closest matching
        documents from the index based on shared fingerprints.

        Args:
            fingerprints: Winnowed fingerprints of the query document.
            min_fingerprints: The minimum number of shared fingerprints required for a document to be
            considered a match (default: 5).
            top_k: The number of top matching documents to return based on shared fingerprint counts
            (default: 5).

        Returns:
        - A list of document IDs corresponding to the closest matches in the index.

        """
        return list(
            self.get_match_counts(
                fingerprints, min_fingerprints=min_fingerprints, top_k=top_k
            ).keys()
        )

    def get_match_counts(
        self, fingerprints: np.ndarray, min_fingerprints: int = 5, top_k: int = 5
    ) -> Dict[str, int]:
        """Given an array of query fingerprints, retrieve the counts of shared
        fingerprints for each document in the index, and return the top-k matches that
        have at least the minimum number of shared fingerprints."""
        if self.index_size == 0:
            return {}
        buf = self.index_array[: self.index_size]
        mask = np.isin(buf["fp"], np.asarray(fingerprints, dtype=np.uint64))
        if not mask.any():
            return {}
        unique_docs, counts = np.unique(buf["doc"][mask], return_counts=True)
        keep = counts >= min_fingerprints
        unique_docs, counts = unique_docs[keep], counts[keep]
        if len(unique_docs) == 0:
            return {}
        order = np.argsort(-counts)[:top_k]
        return {self.to_external[int(unique_docs[i])]: int(counts[i]) for i in order}

    def get_fingerprint_positions(
        self, fingerprints: np.ndarray, doc_id: str
    ) -> Dict[int, List[int]]:
        """Given an array of query fingerprints and a document ID, retrieve the
        positions of the shared fingerprints for that document in the index."""
        int_id = self.doc_id_map.get(doc_id)
        if int_id is None:
            return {}
        buf = self.index_array[: self.index_size]
        mask = (buf["doc"] == int_id) & np.isin(
            buf["fp"], np.asarray(fingerprints, dtype=np.uint64)
        )
        hits = buf[mask]
        positions: Dict[int, List[int]] = {}
        for fp_val, pos_val in zip(hits["fp"].tolist(), hits["pos"].tolist()):
            positions.setdefault(fp_val, []).append(pos_val)
        return positions

    def to_msgpack(self, path: str) -> None:
        """Write sorted fingerprint postings to a gzipped msgpack file (flat-stream format).

        Format: metadata dict, doc-ID table (n_docs: int + n_docs × str), then for each
        unique fingerprint: fp (int), n_postings (int), doc_ids (bytes: n_postings ×
        uint32), positions (bytes: n_postings × uint32).

        Group boundaries are found with a single numpy vectorised pass; posting data is
        written as raw byte blobs rather than individual msgpack values, reducing Python-
        level work from O(total_postings) to O(unique_fingerprints).
        """
        print(f"Exporting index to {path}...", end="", flush=True)
        view = self.index_array[: self.index_size]
        view.sort(order="fp")  # in-place
        print("Done sorting. Now writing to file...", end="", flush=True)
        fps = view["fp"]
        docs = view["doc"]
        positions = view["pos"]

        # Find all positions where the fingerprint value changes (one C-level pass).
        fp_changes = np.flatnonzero(fps[:-1] != fps[1:]) + 1

        packer = msgpack.Packer()
        with gzip.open(path, "wb", compresslevel=1) as f:
            f.write(packer.pack(self.meta))
            f.write(packer.pack(len(self.to_external)))
            for doc_id in self.to_external:
                f.write(packer.pack(doc_id))
            i = 0
            for boundary in fp_changes:
                j = int(boundary)
                f.write(packer.pack(int(fps[i])))
                f.write(packer.pack(j - i))
                f.write(packer.pack(docs[i:j].tobytes()))
                f.write(packer.pack(positions[i:j].tobytes()))
                i = j
            # Write the final group.
            if i < self.index_size:
                f.write(packer.pack(int(fps[i])))
                f.write(packer.pack(self.index_size - i))
                f.write(packer.pack(docs[i:].tobytes()))
                f.write(packer.pack(positions[i:].tobytes()))
        print("Done.")

    @classmethod
    def from_msgpack(cls, path: str) -> "MemoryBasedIndex":
        """Load a MemoryBasedIndex from a gzipped msgpack file (flat-stream format)."""
        with gzip.open(path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            meta = next(unpacker)
            index = cls(meta)
            n_docs = next(unpacker)
            for i in range(n_docs):
                doc_id = next(unpacker)
                index.doc_id_map[doc_id] = i
                index.to_external.append(doc_id)
            for fp in unpacker:
                n_postings = next(unpacker)
                docs_raw = next(unpacker)  # bytes: n_postings × uint32
                pos_raw = next(unpacker)  # bytes: n_postings × uint32
                if index.index_size + n_postings > len(index.index_array):
                    extra = max(cls._CHUNK, n_postings)
                    new_buf = np.empty(index.index_size + extra, dtype=cls._DTYPE)
                    new_buf[: index.index_size] = index.index_array[: index.index_size]
                    index.index_array = new_buf
                end = index.index_size + n_postings
                index.index_array["fp"][index.index_size : end] = fp
                index.index_array["doc"][index.index_size : end] = np.frombuffer(
                    docs_raw, dtype=np.uint32
                )
                index.index_array["pos"][index.index_size : end] = np.frombuffer(
                    pos_raw, dtype=np.uint32
                )
                index.index_size = end
        return index


class DiskBasedIndex:
    """Disk-based implementation of the inverted index for document fingerprinting.

    The index data (fingerprints, offsets, lengths, and document ID mapping) is stored
    on disk as memory-mapped files, and postings lists are stored in a separate binary
    file that is accessed on demand. If the index was merged with
    ``include_similarities=True``, ``similarities.npy`` is also memory-mapped as
    ``similarities``; otherwise that attribute is ``None``.
    """

    def __init__(self, index_dir: str):
        """Initialize the disk-based index by loading the metadata and memory-mapped
        files from the specified index directory."""
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = orjson.loads(f.read())
            self.winnower = winnower.Winnower(
                length=meta["length"],
                window_size=meta["window_size"],
                base=meta["base"],
                punctuation=meta["punctuation"],
            )

        # Load index arrays as memory-mapped files (zero RAM cost; served from OS page cache)
        self.fingerprints = np.load(
            os.path.join(index_dir, "fingerprints.npy"), mmap_mode="r"
        )
        self.offsets = np.load(os.path.join(index_dir, "offsets.npy"), mmap_mode="r")
        self.lengths = np.load(os.path.join(index_dir, "lengths.npy"), mmap_mode="r")
        similarities_path = os.path.join(index_dir, "similarities.npy")
        self.similarities = (
            np.load(similarities_path, mmap_mode="r")
            if os.path.exists(similarities_path)
            else None
        )

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

    def get_document_similarities(
        self, doc_id: str, min_shared_fingerprints: int = 1
    ) -> Dict[str, int]:
        """Return stored exact fingerprint overlaps for one document.

        Similarities are available only when the index was merged with
        ``include_similarities=True``. The persisted pairs use internal integer
        document IDs and are memory-mapped on demand.
        """
        if self.similarities is None:
            raise ValueError("This index was built without document similarities")

        internal_id = None
        for candidate in range(len(self.to_external_doc_id)):
            if self.to_external_doc_id[candidate] == doc_id:
                internal_id = candidate
                break
        if internal_id is None:
            return {}

        matches: Dict[str, int] = {}
        for pair in self.similarities:
            first = int(pair["doc_i"])
            second = int(pair["doc_j"])
            count = int(pair["count"])
            if count < min_shared_fingerprints:
                continue
            if first == internal_id:
                matches[self.to_external_doc_id[second]] = count
            elif second == internal_id:
                matches[self.to_external_doc_id[first]] = count
        return matches

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
