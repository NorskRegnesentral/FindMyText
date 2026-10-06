import multiprocessing as mp
import os

import numpy as np
import pytest

from experiments.compare_similarity_methods import (
    build_index,
    compare_methods,
    write_synthetic_corpus,
)
from findmytext import indexing, similarity


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


def test_forward_topk_threaded_and_sequential_match(index_dir):
    similarity.compute_similarities_forward_topk(
        index_dir,
        top_k=10,
        min_shared_fingerprints=2,
        verbose=False,
        posting_read_mode="sequential",
    )
    sequential = indexing.DiskBasedIndex(index_dir).load_similarities(
        "forward_topk"
    ).copy()

    similarity.compute_similarities_forward_topk(
        index_dir,
        top_k=10,
        min_shared_fingerprints=2,
        verbose=False,
        posting_read_mode="threaded",
    )
    threaded = indexing.DiskBasedIndex(index_dir).load_similarities("forward_topk")

    assert np.array_equal(sequential, threaded)


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
