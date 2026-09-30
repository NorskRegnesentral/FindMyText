"""Benchmark and verify post-merge exhaustive document similarities on Gutenberg."""

import argparse
import multiprocessing as mp
import os
import resource
import shutil
import tempfile
import time
from typing import List, Tuple

import orjson
from datasets import load_dataset

from findmytext import index_builder, indexing, similarity, winnower


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _write_corpus(path: str, records: List[Tuple[str, str]]) -> None:
    with open(path, "wb") as output:
        for doc_id, text in records:
            output.write(orjson.dumps({"id": doc_id, "text": text}))
            output.write(b"\n")


def _load_gutenberg_records(n_books: int) -> List[Tuple[str, str]]:
    dataset = load_dataset("efederici/gutenberg-it", split="train")
    dataset = dataset.select(range(min(n_books, len(dataset))))
    return [(str(row["text_id"]), str(row["text"])) for row in dataset]


def _timed(function, *args, **kwargs) -> Tuple[float, float]:
    started = time.perf_counter()
    function(*args, **kwargs)
    return time.perf_counter() - started, _rss_mib()


def _verify(output_dir: str, records: List[Tuple[str, str]]) -> None:
    disk_index = indexing.DiskBasedIndex(output_dir)
    stored = {
        (int(row["doc_i"]), int(row["doc_j"])): int(row["count"])
        for row in disk_index.load_similarities("inverted")
    }

    # Some records may have been skipped by the existing corpus filters (e.g. the
    # "sex"/"porn" word check in json_line_reader); only verify indexed documents.
    indexed_ids = {
        disk_index.to_external_doc_id[i]
        for i in range(len(disk_index.to_external_doc_id))
    }
    kept_records = [(doc_id, text) for doc_id, text in records if doc_id in indexed_ids]
    if len(kept_records) != len(records):
        print(
            f"Note: {len(records) - len(kept_records)} of {len(records)} records were "
            "skipped by the corpus filters and excluded from verification."
        )

    fingerprint_sets = []
    fingerprint_generator = winnower.Winnower(length=4, window_size=6)
    for _, text in kept_records:
        fingerprints, _ = fingerprint_generator.get_winnowed_fingerprints(text)
        fingerprint_sets.append(set(fingerprints.tolist()))

    expected = {
        (first, second): len(fingerprint_sets[first] & fingerprint_sets[second])
        for first in range(len(kept_records))
        for second in range(first + 1, len(kept_records))
        if fingerprint_sets[first] & fingerprint_sets[second]
    }
    if stored != expected:
        missing = set(expected) - set(stored)
        extra = set(stored) - set(expected)
        raise AssertionError(
            f"Similarity verification failed: missing={len(missing)}, extra={len(extra)}"
        )

    print(f"Verified {len(stored):,} stored pairs against independent winnowing.")
    for pair, count in list(sorted(stored.items()))[:5]:
        print(f"  {kept_records[pair[0]][0]} <-> {kept_records[pair[1]][0]}: {count}")
    del disk_index


def main() -> None:
    if os.name != "nt":
        mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=None, help="Keep benchmark output under this directory."
    )
    parser.add_argument("--nb-workers", type=int, default=1)
    parser.add_argument("--n-books", type=int, default=50)
    args = parser.parse_args()

    records = _load_gutenberg_records(n_books=args.n_books)
    root = args.output_dir or tempfile.mkdtemp(prefix="findmytext-similarity-")
    temporary_index = os.path.join(root, "temporary")
    merged_index = os.path.join(root, "merged")
    os.makedirs(root, exist_ok=True)
    corpus_path = os.path.join(root, f"gutenberg-{args.n_books}.jsonl")
    _write_corpus(corpus_path, records)

    index_started = time.perf_counter()
    index_builder.index_file(
        corpus_path,
        temporary_index,
        length=4,
        window_size=6,
        nb_workers=args.nb_workers,
        delete_existing=True,
        max_length=10_000_000,
    )
    index_time = time.perf_counter() - index_started
    merge_time, merge_rss = _timed(
        index_builder.merge_indexes_from_dir, temporary_index, merged_index
    )
    similarity_time, similarity_rss = _timed(
        similarity.compute_similarities_inverted, merged_index
    )

    print(f"Fingerprint/index-file time: {index_time:.3f}s")
    print(
        f"Merge (incl. forward index): {merge_time:.3f}s, peak RSS: {merge_rss:.1f} MiB"
    )
    print(
        f"Inverted similarities:       {similarity_time:.3f}s, peak RSS: {similarity_rss:.1f} MiB"
    )
    _verify(merged_index, records)

    if args.output_dir is None:
        shutil.rmtree(root)
    else:
        print(f"Benchmark files retained in {root}")


if __name__ == "__main__":
    main()
