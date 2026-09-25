"""Force many index flushes and verify exact merged similarity counts."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile

import orjson

from findmytext import index_builder, indexing, synthetic, winnower


def _write_jsonl(path: str, documents: list[dict[str, str]]) -> None:
    with open(path, "wb") as output:
        for document in documents:
            output.write(orjson.dumps(document))
            output.write(b"\n")


def _stored_counts(index: indexing.DiskBasedIndex) -> dict[tuple[str, str], int]:
    if index.similarities is None:
        raise AssertionError("similarities.npy was not created")
    counts = {}
    for row in index.similarities:
        first = index.to_external_doc_id[int(row["doc_i"])]
        second = index.to_external_doc_id[int(row["doc_j"])]
        counts[tuple(sorted((first, second)))] = int(row["count"])
    return counts


def _expected_counts(
    documents: list[dict[str, str]], fingerprint_length: int, window_size: int
) -> dict[tuple[str, str], int]:
    generator = winnower.Winnower(length=fingerprint_length, window_size=window_size)
    fingerprint_sets = {}
    for document in documents:
        hashes, _ = generator.get_winnowed_fingerprints(document["text"])
        fingerprint_sets[document["id"]] = set(map(int, hashes))
    return {
        tuple(sorted((first, second))): len(
            fingerprint_sets[first] & fingerprint_sets[second]
        )
        for index, first in enumerate(fingerprint_sets)
        for second in list(fingerprint_sets)[index + 1 :]
        if fingerprint_sets[first] & fingerprint_sets[second]
    }


def run_verification(
    output_dir: str,
    *,
    n_documents: int = 18,
    documents_per_index: int = 3,
    similarity_chunk_size: int = 2,
) -> None:
    """Build many shards/chunks and assert every similarity count is exact."""
    corpus = synthetic.generate_controlled_corpus(
        n_documents=n_documents,
        n_hub_fingerprints=4,
        hub_inclusion_probability=[0.9, 0.7, 0.5, 0.3],
        seed=41,
    )
    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    temporary_dir = os.path.join(output_dir, "intermediate")
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)
    _write_jsonl(corpus_path, corpus.documents)

    index_files = index_builder.index_file(
        corpus_path,
        temporary_dir,
        nb_workers=1,
        max_nb_documents_before_flush=documents_per_index,
        min_length=0,
        max_length=1_000_000,
        delete_existing=True,
    )
    if len(index_files) < 2:
        raise AssertionError(
            "The experiment did not create multiple intermediate indexes"
        )

    index_builder.merge_indexes_from_dir(
        temporary_dir,
        merged_dir,
        include_similarities=True,
        similarity_chunk_size=similarity_chunk_size,
        max_similarity_chunks_per_merge=2,
    )
    merged = indexing.DiskBasedIndex(merged_dir)
    stored = _stored_counts(merged)
    expected = _expected_counts(corpus.documents, 4, 6)
    if stored != expected:
        mismatches = {
            pair: (expected.get(pair), stored.get(pair))
            for pair in expected.keys() | stored.keys()
            if expected.get(pair) != stored.get(pair)
        }
        raise AssertionError(
            f"Similarity counts differ for {len(mismatches)} pair(s): "
            f"{list(mismatches.items())[:5]}"
        )
    print(
        f"Verified {len(stored)} similarity pairs from {len(index_files)} "
        "intermediate indexes with forced two-pair chunks."
    )


def main() -> None:
    if os.name != "nt":
        mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--n-documents", type=int, default=18)
    parser.add_argument("--documents-per-index", type=int, default=3)
    parser.add_argument("--similarity-chunk-size", type=int, default=2)
    args = parser.parse_args()

    root = args.output_dir or tempfile.mkdtemp(prefix="findmytext-multifile-")
    try:
        run_verification(
            root,
            n_documents=args.n_documents,
            documents_per_index=args.documents_per_index,
            similarity_chunk_size=args.similarity_chunk_size,
        )
    finally:
        if args.output_dir is None:
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
