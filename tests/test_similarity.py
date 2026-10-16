import multiprocessing as mp
import os

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


def test_columnar_postings_match_interleaved_index(index_dir, tmp_path):
    columnar_dir = str(tmp_path / "columnar_index")
    intermediate_dir = os.path.join(os.path.dirname(index_dir), "intermediate")
    index_builder.merge_indexes_from_dir(
        intermediate_dir,
        columnar_dir,
        postings_layout=indexing.POSTINGS_LAYOUT_COLUMNAR,
    )
    original = indexing.DiskBasedIndex(index_dir, posting_read_threads=1)
    columnar = indexing.DiskBasedIndex(columnar_dir, posting_read_threads=1)
    assert columnar.postings_layout == indexing.POSTINGS_LAYOUT_COLUMNAR

    for doc in range(len(original.to_external_doc_id)):
        query = original.get_document_fingerprints(doc)
        original_candidates = original.get_document_candidates(doc, top_k=10)
        columnar_candidates = columnar.get_document_candidates(doc, top_k=10)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(original_candidates, columnar_candidates)
        )

        if len(query):
            original_postings = original._get_postings(query[:8], only_doc_ids=False)
            columnar_postings = columnar._get_postings(query[:8], only_doc_ids=False)
            assert original_postings.keys() == columnar_postings.keys()
            assert all(
                np.array_equal(original_postings[fp], columnar_postings[fp])
                for fp in original_postings
            )

    original_batches = original.iter_unique_posting_batches(batch_entries=31)
    columnar_batches = columnar.iter_unique_posting_batches(batch_entries=31)
    for original_batch, columnar_batch in zip(original_batches, columnar_batches):
        assert all(
            np.array_equal(left, right)
            for left, right in zip(original_batch, columnar_batch)
        )

    index_builder.build_forward_index(columnar_dir, batch_entries=31)
    rebuilt = indexing.DiskBasedIndex(columnar_dir, posting_read_threads=1)
    assert original.forward_offsets is not None
    assert original.forward_fingerprints is not None
    assert rebuilt.forward_offsets is not None
    assert rebuilt.forward_fingerprints is not None
    assert np.array_equal(original.forward_offsets, rebuilt.forward_offsets)
    assert np.array_equal(original.forward_fingerprints, rebuilt.forward_fingerprints)

    similarity.compute_similarities_inverted(index_dir, verbose=False)
    similarity.compute_similarities_inverted(columnar_dir, verbose=False)
    similarity.compute_similarities_forward_topk(
        index_dir, verbose=False, posting_read_threads=1
    )
    similarity.compute_similarities_forward_topk(
        columnar_dir, verbose=False, posting_read_threads=1
    )
    for method in ("inverted", "forward_topk"):
        original_result = indexing.DiskBasedIndex(index_dir).load_similarities(
            method
        )
        columnar_result = indexing.DiskBasedIndex(columnar_dir).load_similarities(
            method
        )
        assert np.array_equal(original_result, columnar_result)


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
