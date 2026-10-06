"""Compare document-similarity methods on the same merged index: runtime and agreement.

Runs the exhaustive ``inverted`` method and the ``forward_topk`` method, then checks
that the top-k result equals the exhaustive result truncated to the union of
per-document top-k neighbours (same ``min_shared_fingerprints`` threshold and tie-breaking),
and that every top-k count equals the exhaustive count.

Examples::

    # Existing merged index (the forward index is built if missing)
    python -m experiments.compare_similarity_methods --index-dir my_index

    # Build a fresh index from a JSONL corpus, or from a synthetic corpus by default
    python -m experiments.compare_similarity_methods --corpus corpus.jsonl
    python -m experiments.compare_similarity_methods --n-documents 200
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time

import orjson

from findmytext import index_builder, indexing, similarity, synthetic


def build_index(corpus_path: str, root: str, nb_workers: int = 1) -> str:
    """Index and merge ``corpus_path`` under ``root``; return the merged index dir."""
    temporary_dir = os.path.join(root, "intermediate")
    merged_dir = os.path.join(root, "merged")
    index_builder.index_file(
        corpus_path,
        temporary_dir,
        nb_workers=nb_workers,
        min_length=0,
        max_length=10_000_000,
        delete_existing=True,
    )
    index_builder.merge_indexes_from_dir(temporary_dir, merged_dir)
    return merged_dir


def write_synthetic_corpus(path: str, n_documents: int, seed: int = 0) -> None:
    corpus = synthetic.generate_controlled_corpus(
        n_documents=n_documents,
        n_hub_fingerprints=4,
        hub_inclusion_probability=[0.9, 0.7, 0.5, 0.3],
        seed=seed,
    )
    with open(path, "wb") as output:
        for document in corpus.documents:
            output.write(orjson.dumps(document))
            output.write(b"\n")


def compare_methods(
    index_dir: str,
    top_k: int = 50,
    min_shared_fingerprints: int = 5,
    verbose: bool = True,
) -> dict:
    """Run both methods on ``index_dir`` and return timings plus the comparison report."""
    index = indexing.DiskBasedIndex(index_dir)
    if index.forward_offsets is None:
        index_builder.build_forward_index(index_dir)
    del index

    timings = {}
    started = time.perf_counter()
    similarity.compute_similarities_inverted(index_dir, verbose=verbose)
    timings["inverted"] = time.perf_counter() - started

    started = time.perf_counter()
    similarity.compute_similarities_forward_topk(
        index_dir,
        top_k=top_k,
        min_shared_fingerprints=min_shared_fingerprints,
        verbose=verbose,
    )
    timings["forward_topk"] = time.perf_counter() - started

    index = indexing.DiskBasedIndex(index_dir)
    report = similarity.compare_with_exhaustive(
        index.load_similarities("inverted"),
        index.load_similarities("forward_topk"),
        top_k=top_k,
        min_shared_fingerprints=min_shared_fingerprints,
    )
    report["timings"] = timings
    report["n_documents"] = len(index.to_external_doc_id)
    return report


def main() -> None:
    if os.name != "nt":
        mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index-dir", help="Existing merged index directory.")
    source.add_argument("--corpus", help="JSONL corpus with 'id' and 'text' fields.")
    parser.add_argument(
        "--n-documents",
        type=int,
        default=100,
        help="Synthetic corpus size when neither --index-dir nor --corpus is given.",
    )
    parser.add_argument("--output-dir", help="Keep built index files here.")
    parser.add_argument("--nb-workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-shared-fingerprints", type=int, default=5)
    args = parser.parse_args()

    root = None
    if args.index_dir:
        index_dir = args.index_dir
    else:
        root = args.output_dir or tempfile.mkdtemp(prefix="findmytext-compare-")
        os.makedirs(root, exist_ok=True)
        corpus_path = args.corpus
        if corpus_path is None:
            corpus_path = os.path.join(root, "synthetic.jsonl")
            write_synthetic_corpus(corpus_path, args.n_documents)
        index_dir = build_index(corpus_path, root, nb_workers=args.nb_workers)

    try:
        report = compare_methods(index_dir, args.top_k, args.min_shared_fingerprints)
    finally:
        if root is not None and args.output_dir is None:
            shutil.rmtree(root)

    timings = report["timings"]
    print()
    print(f"Documents:                    {report['n_documents']:,}")
    print(f"inverted (exhaustive) time:   {timings['inverted']:.3f}s")
    print(f"forward_topk time:            {timings['forward_topk']:.3f}s")
    print(f"Exhaustive pairs:             {report['n_exhaustive_pairs']:,}")
    print(f"Exhaustive truncated to top-k:{report['n_expected_top_k_pairs']:>8,}")
    print(f"forward_topk pairs:           {report['n_top_k_pairs']:,}")
    print(
        f"Missing / extra pairs:        {len(report['missing'])} / {len(report['extra'])}"
    )
    print(f"Count mismatches:             {len(report['count_mismatches'])}")
    print(f"Identical:                    {report['identical']}")
    if not report["identical"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
