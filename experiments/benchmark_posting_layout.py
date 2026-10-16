"""Convert an index to columnar postings and compare forward-topk performance."""

import argparse
import os
import shutil
import statistics
import tempfile
import time

import numpy as np
import orjson

from findmytext import indexing, similarity


def convert_index_layout(source_dir: str, output_dir: str) -> None:
    """Copy an index and store all doc IDs before all positions in postings.dat."""
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir)

    excluded = {"meta.json", "offsets.npy", "postings.dat"}
    try:
        for name in os.listdir(source_dir):
            source_path = os.path.join(source_dir, name)
            if name in excluded or name.startswith("similarities_"):
                continue
            if os.path.isfile(source_path):
                shutil.copy2(source_path, os.path.join(output_dir, name))

        lengths = np.load(os.path.join(source_dir, "lengths.npy"), mmap_mode="r")
        total_entries = int(lengths.sum())
        source_postings = np.memmap(
            os.path.join(source_dir, "postings.dat"),
            dtype=indexing.POSTING_DTYPE,
            mode="r",
        )
        destination_postings = np.memmap(
            os.path.join(output_dir, "postings.dat"),
            dtype=np.uint32,
            mode="w+",
            shape=(2 * total_entries,),
        )
        chunk_entries = 10_000_000
        for start in range(0, total_entries, chunk_entries):
            stop = min(start + chunk_entries, total_entries)
            destination_postings[start:stop] = source_postings["doc_id"][start:stop]
            destination_postings[
                total_entries + start : total_entries + stop
            ] = source_postings["position"][start:stop]
        destination_postings.flush()
        del destination_postings
        del source_postings

        entry_offsets = np.empty(len(lengths) + 1, dtype=np.uint64)
        entry_offsets[0] = 0
        np.cumsum(lengths, dtype=np.uint64, out=entry_offsets[1:])
        np.save(
            os.path.join(output_dir, "offsets.npy"),
            entry_offsets[:-1] * np.uint64(4),
        )

        with open(os.path.join(source_dir, "meta.json"), "rb") as source_meta:
            meta = orjson.loads(source_meta.read())
        meta["postings_layout"] = indexing.POSTINGS_LAYOUT_COLUMNAR
        with open(os.path.join(output_dir, "meta.json"), "wb") as output_meta:
            output_meta.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    except Exception:
        shutil.rmtree(output_dir)
        raise

    print(
        f"Created columnar index at {output_dir} "
        f"({total_entries:,} posting entries).",
        flush=True,
    )


def verify_index_apis(source_dir: str, columnar_dir: str, batch_entries: int = 1_000_000):
    """Check posting reads, position reads, candidates, and posting-batch iteration."""
    original = indexing.DiskBasedIndex(source_dir, posting_read_threads=1)
    columnar = indexing.DiskBasedIndex(columnar_dir, posting_read_threads=1)
    n_docs = len(original.to_external_doc_id)

    sample_docs = sorted({0, n_docs // 3, n_docs // 2, n_docs - 1})
    for doc in sample_docs:
        fingerprints = original.get_document_fingerprints(doc)
        if not np.array_equal(fingerprints, columnar.get_document_fingerprints(doc)):
            raise AssertionError(f"Forward fingerprints differ for document {doc}")
        query = fingerprints[:: max(1, len(fingerprints) // 16)][:16]
        for only_doc_ids in (True, False):
            left = original._get_postings(query, only_doc_ids=only_doc_ids)
            right = columnar._get_postings(query, only_doc_ids=only_doc_ids)
            if left.keys() != right.keys():
                raise AssertionError("Posting lookup returned different fingerprints")
            for fingerprint in left:
                if not np.array_equal(left[fingerprint], right[fingerprint]):
                    raise AssertionError(
                        f"Posting data differ for fingerprint {fingerprint}"
                    )
        left_candidates = original.get_document_candidates(doc, top_k=20)
        right_candidates = columnar.get_document_candidates(doc, top_k=20)
        if not all(
            np.array_equal(left, right)
            for left, right in zip(left_candidates, right_candidates)
        ):
            raise AssertionError(f"Candidate results differ for document {doc}")

    query = original.get_document_fingerprints(sample_docs[0])[:32]
    left_positions = original.get_closest_matches_with_positions(
        query, min_fingerprints=1, top_k=5, verbose=False
    )
    right_positions = columnar.get_closest_matches_with_positions(
        query, min_fingerprints=1, top_k=5, verbose=False
    )
    if left_positions != right_positions:
        raise AssertionError("Position-returning lookup differs between layouts")

    original_batches = original.iter_unique_posting_batches(batch_entries)
    columnar_batches = columnar.iter_unique_posting_batches(batch_entries)
    for batch_number, (left_batch, right_batch) in enumerate(
        zip(original_batches, columnar_batches)
    ):
        if not all(
            np.array_equal(left, right)
            for left, right in zip(left_batch, right_batch)
        ):
            raise AssertionError(f"Posting batch {batch_number} differs")
    try:
        next(original_batches)
    except StopIteration:
        pass
    else:
        raise AssertionError("columnar index has fewer fingerprint groups")
    try:
        next(columnar_batches)
    except StopIteration:
        pass
    else:
        raise AssertionError("interleaved index has fewer fingerprint groups")

    print("Posting, position, candidate, and batch APIs match.", flush=True)


def benchmark_forward_topk(
    layouts: dict[str, str],
    max_documents: int,
    repeats: int,
) -> None:
    """Time both layouts and require byte-for-byte equal similarity outputs."""
    timings = {label: [] for label in layouts}
    original_name = similarity.similarity_file_name
    parent_dir = os.path.dirname(os.path.abspath(next(iter(layouts.values()))))
    try:
        with tempfile.TemporaryDirectory(
            prefix=".posting-layout-benchmark-", dir=parent_dir
        ) as output_dir:
            expected = None
            for label, index_dir in layouts.items():
                output_path = os.path.join(output_dir, f"warmup-{label}.npy")
                similarity.similarity_file_name = (
                    lambda method, path=output_path: path
                )
                print(f"Warming {label} layout...", flush=True)
                similarity.compute_similarities_forward_topk(
                    index_dir=index_dir,
                    top_k=50,
                    min_shared_fingerprints=5,
                    max_pair_updates_in_memory=20_000_000,
                    max_chunks_per_merge=None,
                    verbose=False,
                    posting_read_threads=1,
                    max_documents=max_documents,
                )
                warm_result = np.load(output_path)
                if expected is None:
                    expected = warm_result.copy()
                elif not np.array_equal(expected, warm_result):
                    raise AssertionError(
                        f"Warmup similarity output differs for {label} layout"
                    )
            if expected is None:
                raise AssertionError("No layouts were benchmarked")
            reference_result = expected

            for repeat in range(repeats):
                ordered_layouts = list(layouts.items())
                if repeat % 2:
                    ordered_layouts.reverse()
                for label, index_dir in ordered_layouts:
                    output_path = os.path.join(
                        output_dir, f"{label}-{repeat}.npy"
                    )
                    similarity.similarity_file_name = (
                        lambda method, path=output_path: path
                    )
                    started = time.perf_counter()
                    similarity.compute_similarities_forward_topk(
                        index_dir=index_dir,
                        top_k=50,
                        min_shared_fingerprints=5,
                        max_pair_updates_in_memory=20_000_000,
                        max_chunks_per_merge=None,
                        verbose=False,
                        posting_read_threads=1,
                        max_documents=max_documents,
                    )
                    elapsed = time.perf_counter() - started
                    result = np.load(output_path)
                    timings[label].append(elapsed)
                    if not np.array_equal(reference_result, result):
                        raise AssertionError(
                            f"Similarity output differs for {label}, repeat {repeat}"
                        )
                    print(
                        f"repeat={repeat} {label}: {elapsed:.2f}s "
                        f"({len(result):,} pairs; identical=True)",
                        flush=True,
                    )
    finally:
        similarity.similarity_file_name = original_name

    for label, values in timings.items():
        print(
            f"{label} median: {statistics.median(values):.2f}s "
            f"({values})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_index")
    parser.add_argument("columnar_index")
    parser.add_argument("--max-documents", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    convert_index_layout(args.source_index, args.columnar_index)
    verify_index_apis(args.source_index, args.columnar_index)
    layouts = {
        "interleaved": args.source_index,
        "columnar": args.columnar_index,
    }
    benchmark_forward_topk(
        layouts,
        max_documents=args.max_documents,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()