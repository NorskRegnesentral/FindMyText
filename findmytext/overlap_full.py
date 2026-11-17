"""K-independent full-corpus join views reusing the original read-only arrays."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .overlap_inverted import OverlapInvertedIndex, _close_on_exit, _source_identity
from .overlap_join import JoinSubset, _document_ranks, _require_memory
from .overlap_stream import atomic_json, subset_identity


def load_full_corpus(
    directory: str | Path, memory_budget_bytes: int = 32 * 2**30,
) -> JoinSubset:
    root = Path(directory)
    meta = json.loads((root / "meta.json").read_text())
    if meta["format"] != "findmytext-full-overlap-view-v1":
        raise ValueError("Unsupported full-corpus view format")
    source = Path(meta["source"]["path"])
    if _source_identity(source) != meta["source"]:
        raise ValueError("Full-corpus view source changed")
    overlap_root = Path(meta["overlap_dir"])
    if hashlib.sha256((overlap_root / "meta.json").read_bytes()).hexdigest() != meta["overlap_metadata_sha256"]:
        raise ValueError("Full-corpus overlap metadata changed")
    overlap = OverlapInvertedIndex(overlap_root, source)
    data = JoinSubset(
        np.load(root / "document_ids.npy", mmap_mode="r"),
        np.load(source / "forward_offsets.npy", mmap_mode="r"),
        np.load(source / "forward_fingerprints.npy", mmap_mode="r"),
        overlap.fingerprints, overlap.offsets, overlap.doc_ids,
        np.load(root / "source_df.npy", mmap_mode="r"),
        overlap.eligible_lengths,
        np.load(root / "document_fp_offsets.npy", mmap_mode="r"),
        np.load(root / "document_fp_ranks.npy", mmap_mode="r"),
        memory_budget_bytes, source_identity=meta["source"],
    )
    _require_memory(data.nbytes + len(data.posting_docs) * 4 + len(data.posting_offsets) * 8, memory_budget_bytes)
    if subset_identity(data) != meta["identity"]:
        raise ValueError("Full-corpus view arrays do not match their identity")
    return data


def _prepare_full_corpus_files(
    index_dir: str | Path, overlap_dir: str | Path, output_dir: str | Path,
    memory_budget_bytes: int = 32 * 2**30,
) -> None:
    source, overlap_root, root = map(lambda value: Path(value).resolve(), (index_dir, overlap_dir, output_dir))
    if any(root == index or index in root.parents for index in (source, overlap_root)):
        raise ValueError("Full-corpus helper views must be outside source indexes")
    if root.exists():
        meta = json.loads((root / "meta.json").read_text())
        if meta["source"] != _source_identity(source) or meta["overlap_dir"] != str(overlap_root):
            raise ValueError("Existing full-corpus view belongs to a different source")
        return
    identity = _source_identity(source)
    overlap = OverlapInvertedIndex(overlap_root, source)
    n_docs = overlap.meta["document_count"]
    if not n_docs:
        raise ValueError("Cannot prepare a full-corpus join with no documents")
    if len(overlap.fingerprints) > np.iinfo(np.uint32).max:
        raise ValueError("Full-corpus rarity ranks exceed uint32 capacity")
    forward_offsets = np.load(source / "forward_offsets.npy", mmap_mode="r")
    forward_values = np.load(source / "forward_fingerprints.npy", mmap_mode="r")
    required = (
        forward_offsets.nbytes + forward_values.nbytes + overlap.fingerprints.nbytes
        + overlap.offsets.nbytes + overlap.doc_ids.nbytes + overlap.eligible_lengths.nbytes
        + n_docs * 32 + len(overlap.doc_ids) * 8 + len(overlap.fingerprints) * 16
    )
    _require_memory(required, memory_budget_bytes)
    started = time.perf_counter()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix=".full-overlap-view-", dir=root.parent) as temporary,
        ExitStack() as mappings,
    ):
        staging = Path(temporary) / "view"
        staging.mkdir()
        ids = np.lib.format.open_memmap(
            staging / "document_ids.npy", mode="w+", dtype=np.uint32, shape=(n_docs,),
        )
        _close_on_exit(mappings, ids)
        for first in range(0, n_docs, 1_000_000):
            last = min(first + 1_000_000, n_docs)
            ids[first:last] = np.arange(first, last, dtype=np.uint32)
        doc_offsets = np.lib.format.open_memmap(
            staging / "document_fp_offsets.npy", mode="w+", dtype=np.uint64, shape=(n_docs + 1,),
        )
        _close_on_exit(mappings, doc_offsets)
        doc_offsets[0] = 0
        np.cumsum(overlap.eligible_lengths, out=doc_offsets[1:])
        if int(doc_offsets[-1]) != len(overlap.doc_ids):
            raise ValueError("Eligible-length accounting mismatch")
        doc_ranks = np.lib.format.open_memmap(
            staging / "document_fp_ranks.npy", mode="w+", dtype=np.uint32, shape=(len(overlap.doc_ids),),
        )
        _close_on_exit(mappings, doc_ranks)
        _document_ranks(overlap.offsets, overlap.doc_ids, doc_offsets[:-1].copy(), doc_ranks)
        dfs = np.lib.format.open_memmap(
            staging / "source_df.npy", mode="w+", dtype=np.uint64, shape=(len(overlap.fingerprints),),
        )
        _close_on_exit(mappings, dfs)
        for first in range(0, len(dfs), 1_000_000):
            last = min(first + 1_000_000, len(dfs))
            dfs[first:last] = overlap.offsets[first + 1:last + 1] - overlap.offsets[first:last]
        for array in (ids, doc_offsets, doc_ranks, dfs):
            array.flush()
        data = JoinSubset(
            ids, forward_offsets, forward_values, overlap.fingerprints,
            overlap.offsets, overlap.doc_ids, dfs, overlap.eligible_lengths,
            doc_offsets, doc_ranks, memory_budget_bytes, source_identity=identity,
        )
        meta = {
            "format": "findmytext-full-overlap-view-v1",
            "source": identity, "overlap_dir": str(overlap_root),
            "overlap_metadata_sha256": hashlib.sha256((overlap_root / "meta.json").read_bytes()).hexdigest(),
            "identity": subset_identity(data),
            "memory_budget_bytes": memory_budget_bytes,
            "logical_view_bytes": data.nbytes,
            "preparation_seconds": time.perf_counter() - started,
            "source_forward_storage_reused_bytes": forward_offsets.nbytes + forward_values.nbytes,
            "helper_array_bytes": sum(path.stat().st_size for path in staging.iterdir()),
            "no_forward_or_overlap_payload_copy": True,
        }
        if _source_identity(source) != identity:
            raise ValueError("Source index changed during full-corpus preparation")
        atomic_json(staging / "meta.json", meta)
        staging.rename(root)


def prepare_full_corpus(
    index_dir: str | Path, overlap_dir: str | Path, output_dir: str | Path,
    memory_budget_bytes: int = 32 * 2**30,
) -> JoinSubset:
    _prepare_full_corpus_files(index_dir, overlap_dir, output_dir, memory_budget_bytes)
    return load_full_corpus(output_dir, memory_budget_bytes)
