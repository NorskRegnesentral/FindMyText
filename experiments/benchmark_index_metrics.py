"""Measure fresh-index build time, final size and end-to-end detector latency."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from findmytext import utils  # noqa: E402
from findmytext.detectors import FingerprintChainDetector  # noqa: E402


def select_queries(detector, records, *, count, characters, seed):
    eligible = [record for record in records if len(record["text"]) >= characters]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    queries, skipped = [], Counter()
    for record in eligible:
        start = rng.randrange(len(record["text"]) - characters + 1)
        text = record["text"][start:start + characters]
        frame = detector.get_fingerprints(text)
        counts = Counter(frame["hash"].to_list())
        if not counts:
            skipped["no_fingerprints"] += 1
            continue
        candidates = detector._get_candidates(frame["hash"])
        points = Counter()
        for doc, fingerprint in candidates.select("doc_match_id", "hash").iter_rows():
            points[doc] += counts[fingerprint]
        if not points:
            skipped["no_candidates"] += 1
            continue
        if max(points.values()) > 5000:
            skipped["over_5000_alignment_points_in_a_candidate"] += 1
            continue
        queries.append({
            "text": text, "source_id": record["id"],
            "max_candidate_alignment_points": max(points.values()),
        })
        if len(queries) == count:
            break
    if len(queries) != count:
        raise ValueError(f"Only {len(queries)} eligible bounded queries found; skipped={dict(skipped)}")
    return queries, {"eligible_documents": len(eligible), "skipped": dict(skipped)}


def benchmark_metrics(corpus, output, *, documents=5000, queries=50, characters=2000, repeats=3, seed=20261116):
    if min(documents, queries, characters, repeats) < 1 or characters < 1000:
        raise ValueError("Use positive counts and query fragments of at least 1000 characters")
    if "fork" not in mp.get_all_start_methods():
        raise ValueError("The startup-inclusive build harness requires a platform supporting fork")
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    if output == corpus or output in corpus.parents:
        raise ValueError("Benchmark output must be separate from the source corpus")
    output.mkdir()
    stream = (
        utils.stream_json_zst(str(corpus), max_length=100_000)
        if str(corpus).endswith(".zst") else utils.stream_jsonl(str(corpus))
    )
    records = []
    sample = output / "input.jsonl"
    try:
        with sample.open("w") as target:
            for record in stream:
                row = {"id": str(record["id"]), "text": record["text"]}
                records.append(row)
                target.write(json.dumps(row) + "\n")
                if len(records) == documents:
                    break
    finally:
        stream.close()
    if len(records) != documents:
        raise ValueError(f"Requested {documents} records but found only {len(records)}")
    builds = []
    for repeat in range(repeats):
        destination = output / f"build-{repeat + 1}"
        command = [
            sys.executable, str(ROOT / "experiments/benchmark_index_build.py"),
            "--checkout", str(ROOT), "--corpus", str(sample), "--output", str(destination),
            "--workers", "1", "--start-method", "fork", "--cli",
        ]
        with (output / f"build-{repeat + 1}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        result = json.loads((destination / "result.json").read_text())
        if not result["all_source_occurrences_verified"]:
            raise AssertionError("A measured build failed source-occurrence verification")
        builds.append(result)
        print(f"[metrics] build {repeat + 1}/{repeats}: {result['total_seconds']:.3f}s; {result['index_bytes']:,} bytes", flush=True)
    if len({(row["index_bytes"], row["fingerprints"], row["occurrences"]) for row in builds}) != 1:
        raise AssertionError("Repeated builds have different storage/count accounting")
    index = output / f"build-{repeats}/index"
    detector = FingerprintChainDetector(str(index))
    selected, selection = select_queries(detector, records, count=queries, characters=characters, seed=seed)
    reference = []
    for query in selected:
        scores = detector.get_containment_scores(query["text"])
        if not all(isinstance(doc, str) and np.isfinite(score) and 0 <= score <= 1 for doc, score in scores.items()):
            raise AssertionError("Detector returned an invalid document/score mapping")
        reference.append(scores)
    timings = []
    for repeat in range(repeats):
        seconds = []
        for query, expected in zip(selected, reference):
            started = time.perf_counter()
            scores = detector.get_containment_scores(query["text"])
            seconds.append(time.perf_counter() - started)
            if scores != expected:
                raise AssertionError("Timed detector scores differ from the warm-up reference")
        timings.append(seconds)
        print(f"[metrics] query pass {repeat + 1}/{repeats}: {sum(seconds):.3f}s; all {queries} score mappings identical", flush=True)
    flat = [value for row in timings for value in row]
    report = {
        "documents": documents, "input_jsonl_bytes": sample.stat().st_size,
        "input_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "hash_settings": detector.index.winnower.to_metadata(),
        "index_fingerprints": builds[-1]["fingerprints"], "index_occurrences": builds[-1]["occurrences"],
        "index_bytes": builds[-1]["index_bytes"],
        "index_size_scope": "Final index files only, including metadata/document mapping; excludes intermediate shards, sample text and logs.",
        "build_seconds_median": statistics.median(row["total_seconds"] for row in builds),
        "build_runs": builds,
        "build_scope": "Fresh index and merge CLI processes, one worker, default flushing; includes startup/imports/JIT with existing caches; excludes sampling and verification; normal buffered writes without durability fsync.",
        "query_workload": {
            "queries": queries, "characters_per_query": characters, "repeats": repeats, "seed": seed,
            "selection": "Seeded positive contiguous character fragments from the indexed sample; no negative queries.",
            **selection,
            "safety": "Exclude fragments with >5000 alignment points in any candidate during preflight; detector defaults are not changed.",
            "source_document_found_in_results": sum(query["source_id"] in scores for query, scores in zip(selected, reference)),
            "query_sha256": [hashlib.sha256(query["text"].encode()).hexdigest() for query in selected],
            "max_candidate_alignment_points": [query["max_candidate_alignment_points"] for query in selected],
        },
        "detector_defaults": {
            "class": type(detector).__name__, "top_k": detector.top_k,
            "min_fingerprints": detector.min_fingerprints, "rarity_weighted": detector.rarity_weighted,
            "method": detector.method, "position_threshold": detector.position_threshold,
            "offset_threshold": detector.offset_threshold, "min_cluster_size": detector.min_cluster_size,
            "min_fragment_length": 1000, "posting_read_threads": 1,
        },
        "query_latency_seconds": {
            "median": statistics.median(flat), "mean": statistics.mean(flat),
            "p95": float(np.percentile(flat, 95)), "minimum": min(flat), "maximum": max(flat),
            "per_pass": timings, "all_timed_score_mappings_identical": True,
        },
        "query_scope": "Warm loaded-index get_containment_scores call: includes query tokenization/hashing, candidate lookup, positional reads, clustering and scoring; excludes detector initialization, query preparation/preflight, warm-up and verification.",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "polars")},
            "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "POLARS_MAX_THREADS")},
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((ROOT / "findmytext").glob("*.py")), ROOT / "experiments/benchmark_index_build.py", Path(__file__)]
        },
        "caveats": [
            "Bounded 5000-document-style workload, not full-corpus paper results or a universal latency guarantee.",
            "Shared host and warm filesystem/JIT caches; positive source fragments only.",
            "Skipping resource-heavy fragments changes the workload; skipped counts are reported explicitly.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--query-characters", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark_metrics(args.corpus, args.output, documents=args.documents, queries=args.queries, characters=args.query_characters, repeats=args.repeats)
