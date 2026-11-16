"""Read-only comparison of sequential and eight-thread posting retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from findmytext import indexing, utils  # noqa: E402


def source_identity(index_dir):
    names = (
        "meta.json", "fingerprints.npy", "offsets.npy", "lengths.npy",
        "postings.dat", "doc_name_offsets.npy", "doc_name_bytes.npy",
    )
    return {
        "metadata_sha256": hashlib.sha256((index_dir / "meta.json").read_bytes()).hexdigest(),
        "files": {
            name: {"bytes": (index_dir / name).stat().st_size, "mtime_ns": (index_dir / name).stat().st_mtime_ns}
            for name in names
        },
    }


def result_digest(result, *, positional):
    digest = hashlib.sha256()
    if positional:
        for fingerprint, values in result.items():
            digest.update(int(fingerprint).to_bytes(8, "little"))
            digest.update(str(values.dtype.descr).encode())
            digest.update(len(values).to_bytes(8, "little"))
            digest.update(values.tobytes())
    else:
        digest.update(json.dumps(list(result.items())).encode())
    return digest.hexdigest()


def benchmark(index_dir, corpus, output, *, documents=200, repeats=5, seed=20261016):
    if documents < 1 or repeats < 1:
        raise ValueError("documents and repeats must be positive")
    index_dir, corpus, output = Path(index_dir).resolve(), Path(corpus).resolve(), Path(output).resolve()
    if output == index_dir or index_dir in output.parents or output == corpus:
        raise ValueError("Benchmark output must be outside the source index and corpus file")
    if output.exists():
        raise FileExistsError(output)
    identity = source_identity(index_dir)
    readers = {}
    try:
        readers = {
            threads: indexing.DiskBasedIndex(str(index_dir), posting_read_threads=threads)
            for threads in (1, 8)
        }
        configured = readers[1].winnower
        stream = (
            utils.stream_json_zst(str(corpus), max_length=100_000)
            if str(corpus).endswith(".zst") else utils.stream_jsonl(str(corpus))
        )
        queries = []
        input_digest = hashlib.sha256()
        try:
            for record in stream:
                fingerprints, _ = configured.get_winnowed_fingerprints(record["text"])
                queries.append(np.unique(fingerprints))
                input_digest.update(json.dumps({"id": str(record["id"]), "text": record["text"]}, sort_keys=True).encode())
                input_digest.update(b"\n")
                if len(queries) == documents:
                    break
        finally:
            stream.close()
        if len(queries) != documents or any(not len(query) for query in queries):
            raise ValueError("Require the requested number of nonempty document fingerprint queries")
        query_stats = []
        for query in queries:
            ranks = np.searchsorted(readers[1].fingerprints, query)
            if np.any(ranks >= len(readers[1].fingerprints)) or not np.array_equal(readers[1].fingerprints[ranks], query):
                raise ValueError("A corpus query contains fingerprints missing from this saved index")
            entries = int(readers[1].lengths[ranks].astype(np.uint64).sum())
            query_stats.append({"fingerprints": len(query), "posting_entries": entries, "posting_payload_bytes": entries * 8})
        order = np.random.default_rng(seed).permutation(documents)
        runs = []
        for mode in ("positional_postings", "candidate_counts_top50"):
            reference = []

            def invoke(reader, query):
                if mode == "positional_postings":
                    return reader._get_postings(query)
                return reader.get_match_counts(query, top_k=50, min_fingerprints=5)

            # Warm the selected pages and each reader's execution path; never drop shared caches.
            for threads in (1, 8):
                for offset, position in enumerate(order):
                    result = invoke(readers[threads], queries[position])
                    digest = result_digest(result, positional=mode == "positional_postings")
                    if threads == 1:
                        reference.append(digest)
                    elif reference[offset] != digest:
                        raise AssertionError("Warm-up thread configurations return different results")
            for repeat in range(repeats):
                configs = (8, 1) if repeat % 2 == 0 else (1, 8)
                for threads in configs:
                    seconds = []
                    pass_digest = hashlib.sha256()
                    for offset, position in enumerate(order):
                        started = time.perf_counter()
                        result = invoke(readers[threads], queries[position])
                        seconds.append(time.perf_counter() - started)
                        digest = result_digest(result, positional=mode == "positional_postings")
                        if digest != reference[offset]:
                            raise AssertionError("Timed results differ from the sequential reference")
                        pass_digest.update(bytes.fromhex(digest))
                    row = {
                        "mode": mode, "threads": threads, "repeat": repeat + 1,
                        "query_seconds": seconds, "total_seconds": sum(seconds),
                        "median_query_seconds": statistics.median(seconds),
                        "p95_query_seconds": float(np.percentile(seconds, 95)),
                        "all_queries_identical": True, "queries_verified": documents,
                        "ordered_results_sha256": pass_digest.hexdigest(),
                    }
                    runs.append(row)
                    print(f"{mode}: threads={threads} repeat={repeat + 1}: {row['total_seconds']:.4f}s; all {documents} query results identical", flush=True)
        if identity != source_identity(index_dir):
            raise ValueError("Source index changed during the read-only benchmark")
        summary = {
            mode: {
                str(threads): {
                    "median_total_seconds": statistics.median(
                        row["total_seconds"] for row in runs if row["mode"] == mode and row["threads"] == threads
                    ),
                    "median_query_seconds": statistics.median(
                        value for row in runs if row["mode"] == mode and row["threads"] == threads for value in row["query_seconds"]
                    ),
                } for threads in (1, 8)
            } for mode in ("positional_postings", "candidate_counts_top50")
        }
        for row in runs:
            del row["query_seconds"]
        report = {
            "documents": documents, "repeats": repeats, "seed": seed,
            "query_selection": "first N eligible corpus records, de-duplicated stored-window fingerprints, seeded common query order",
            "normalized_input_sha256": input_digest.hexdigest(),
            "query_fingerprint_sha256": hashlib.sha256(b"".join(query.tobytes() for query in queries)).hexdigest(),
            "index_source": identity, "source_unchanged": True, "hash_settings": configured.to_metadata(),
            "index_documents": len(readers[1].to_external_doc_id),
            "index_fingerprints": len(readers[1].fingerprints),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "query_statistics": query_stats,
            "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "timing_scope": "per-query calls only; excludes corpus reading, fingerprinting, reader creation, warm-up and equality verification",
            "modes": {
                "positional_postings": "lookup, file reads and positional-array decoding",
                "candidate_counts_top50": "lookup, file reads, per-fingerprint document deduplication, counting and top-50 ranking; no passage clustering",
            },
            "caveats": [
                "Warm-cache, shared-host workload on one saved index; no cold-cache or full-build speedup claim.",
                "The threaded path is the previous eight-thread behavior; sequential is the new default.",
                "This reads complete per-hash posting lists, not only the query document's occurrences.",
            ],
            "runs": runs, "summary": summary,
            "implementation_sha256": hashlib.sha256(Path(indexing.__file__).read_bytes()).hexdigest(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        for reader in readers.values():
            if reader._io_pool is not None:
                reader._io_pool.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    benchmark(args.index_dir, args.corpus, args.output, documents=args.documents, repeats=args.repeats, seed=args.seed)
