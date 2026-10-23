import multiprocessing as mp
import os
import json
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.compare_similarity_methods import (
    build_index,
    compare_methods,
    write_synthetic_corpus,
)
from findmytext import index_builder, indexing, similarity


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    if os.name != "nt":
        mp.set_start_method("fork", force=True)
    root = str(tmp_path_factory.mktemp("similarity"))
    corpus_path = os.path.join(root, "corpus.jsonl")
    write_synthetic_corpus(corpus_path, n_documents=40, seed=3)
    return build_index(corpus_path, root)


@pytest.mark.parametrize("top_k,min_shared_fingerprints", [(50, 5), (3, 2), (1, 1)])
def test_forward_topk_matches_truncated_exhaustive(
    index_dir, top_k, min_shared_fingerprints
):
    report = compare_methods(
        index_dir,
        top_k=top_k,
        min_shared_fingerprints=min_shared_fingerprints,
        verbose=False,
    )
    assert report["n_top_k_pairs"] > 0
    assert report["identical"], report


def test_forward_topk_single_thread_and_threaded_match(index_dir):
    single_thread_index = indexing.DiskBasedIndex(index_dir, posting_read_threads=1)
    assert single_thread_index._io_pool is None
    similarity.compute_similarities_forward_topk(
        index_dir,
        top_k=10,
        min_shared_fingerprints=2,
        verbose=False,
        posting_read_threads=1,
    )
    sequential = (
        indexing.DiskBasedIndex(index_dir).load_similarities("forward_topk").copy()
    )

    similarity.compute_similarities_forward_topk(
        index_dir,
        top_k=10,
        min_shared_fingerprints=2,
        verbose=False,
        posting_read_threads=2,
    )
    threaded = indexing.DiskBasedIndex(index_dir).load_similarities("forward_topk")

    assert np.array_equal(sequential, threaded)


def test_interleaved_remerge_preserves_postings_forward_and_similarities(index_dir, tmp_path):
    rebuilt_dir = str(tmp_path / "rebuilt_index")
    intermediate_dir = os.path.join(os.path.dirname(index_dir), "intermediate")
    index_builder.merge_indexes_from_dir(
        intermediate_dir,
        rebuilt_dir,
    )
    original = indexing.DiskBasedIndex(index_dir, posting_read_threads=1)
    remade = indexing.DiskBasedIndex(rebuilt_dir, posting_read_threads=1)
    assert remade.postings_layout == indexing.POSTINGS_LAYOUT_INTERLEAVED
    assert remade.fingerprints.dtype == np.uint64
    assert indexing.POSTING_DTYPE.itemsize == 8
    assert json.loads((Path(rebuilt_dir) / "meta.json").read_text())["postings_layout"] == "interleaved"

    for doc in range(len(original.to_external_doc_id)):
        query = original.get_document_fingerprints(doc)
        original_candidates = original.get_document_candidates(doc, top_k=10)
        remade_candidates = remade.get_document_candidates(doc, top_k=10)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(original_candidates, remade_candidates)
        )

        if len(query):
            for only_doc_ids in (False, True):
                original_postings = original._get_postings(query[:8], only_doc_ids=only_doc_ids)
                remade_postings = remade._get_postings(query[:8], only_doc_ids=only_doc_ids)
                assert original_postings.keys() == remade_postings.keys()
                assert all(
                    np.array_equal(original_postings[fp], remade_postings[fp])
                    for fp in original_postings
                )
            assert original.get_closest_matches_with_positions(
                query[:8], min_fingerprints=1, top_k=5, verbose=False,
            ) == remade.get_closest_matches_with_positions(
                query[:8], min_fingerprints=1, top_k=5, verbose=False,
            )

    original_batches = original.iter_unique_posting_batches(batch_entries=31)
    remade_batches = remade.iter_unique_posting_batches(batch_entries=31)
    for original_batch, remade_batch in zip(original_batches, remade_batches, strict=True):
        assert all(
            np.array_equal(left, right)
            for left, right in zip(original_batch, remade_batch)
        )

    index_builder.build_forward_index(rebuilt_dir, batch_entries=31)
    rebuilt = indexing.DiskBasedIndex(rebuilt_dir, posting_read_threads=1)
    assert original.forward_offsets is not None
    assert original.forward_fingerprints is not None
    assert rebuilt.forward_offsets is not None
    assert rebuilt.forward_fingerprints is not None
    assert np.array_equal(original.forward_offsets, rebuilt.forward_offsets)
    assert np.array_equal(original.forward_fingerprints, rebuilt.forward_fingerprints)

    similarity.compute_similarities_inverted(index_dir, verbose=False)
    similarity.compute_similarities_inverted(rebuilt_dir, verbose=False)
    similarity.compute_similarities_forward_topk(
        index_dir, verbose=False, posting_read_threads=1
    )
    similarity.compute_similarities_forward_topk(
        rebuilt_dir, verbose=False, posting_read_threads=1
    )
    for method in ("inverted", "forward_topk"):
        original_result = indexing.DiskBasedIndex(index_dir).load_similarities(
            method
        )
        remade_result = indexing.DiskBasedIndex(rebuilt_dir).load_similarities(
            method
        )
        assert np.array_equal(original_result, remade_result)


@pytest.mark.parametrize("layout", ["columnar", "unknown", None])
def test_unsupported_layout_is_rejected_before_posting_io(tmp_path, layout):
    meta = indexing.winnower.Winnower(length=4).to_metadata()
    meta["postings_layout"] = layout
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="only interleaved"):
        indexing.DiskBasedIndex(str(tmp_path))
    with pytest.raises(ValueError, match="only interleaved"):
        index_builder.build_forward_index(str(tmp_path))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["meta.json"]


def test_layout_selection_is_removed_from_builder_and_cli(tmp_path):
    target = tmp_path / "not-created"
    with pytest.raises(TypeError, match="postings_layout"):
        inspect.signature(index_builder.merge_indexes).bind(
            [], str(target), postings_layout="columnar",
        )
    result = subprocess.run(
        [sys.executable, "-m", "findmytext.index_builder", "merge",
         str(tmp_path / "shards"), str(target), "--postings-layout", "columnar"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert not target.exists()


def test_posting_batch_iterator_rejects_plain_column_data():
    with pytest.raises(ValueError, match="interleaved POSTING_DTYPE"):
        list(indexing.iter_unique_posting_batches(
            np.array([1], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([1], dtype=np.uint32),
            np.array([0, 10], dtype=np.uint32),
        ))


def test_inverted_chunked_merge_matches_single_chunk(index_dir):
    similarity.compute_similarities_inverted(index_dir, verbose=False)
    single = indexing.DiskBasedIndex(index_dir).load_similarities("inverted").copy()

    similarity.compute_similarities_inverted(
        index_dir,
        max_pair_updates_in_memory=7,
        max_chunks_per_merge=2,
        posting_batch_entries=13,
        verbose=False,
    )
    chunked = indexing.DiskBasedIndex(index_dir).load_similarities("inverted")
    assert np.array_equal(single, chunked)
    assert not [f for f in os.listdir(index_dir) if f.startswith(".similarities")]
