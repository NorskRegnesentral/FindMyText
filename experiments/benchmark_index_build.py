"""Compare warmed API or startup-inclusive CLI builds on the same bounded JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def build_one(args):
    sys.path.insert(0, str(args.checkout))
    import numpy as np
    from findmytext import index_builder, indexing
    from findmytext.winnower import Winnower

    mp.set_start_method(args.start_method, force=True)
    records = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    configured = Winnower(length=4, window_size=6)
    warmup = 0.0
    if not args.cli:
        started = time.perf_counter()
        configured.get_winnowed_fingerprints(records[0]["text"])
        warmup = time.perf_counter() - started
    output = args.output
    output.mkdir()

    def run_cli(arguments):
        wrapper = (
            "import multiprocessing as mp, runpy; "
            f"mp.set_start_method({args.start_method!r}); "
            "runpy.run_module('findmytext.index_builder', run_name='__main__')"
        )
        subprocess.run(
            [sys.executable, "-c", wrapper, *arguments],
            cwd=args.checkout, check=True, timeout=600,
        )

    build_started = time.perf_counter()
    if args.cli:
        run_cli([
            "index", str(args.corpus), str(output / "shards"),
            "--length", "4", "--window_size", "6", "--nb_workers", str(args.workers),
        ])
    else:
        shards = index_builder.index_file(
            str(args.corpus), str(output / "shards"), length=4, window_size=6,
            nb_workers=args.workers, max_nb_documents_before_flush=2000,
        )
    extracted = time.perf_counter()
    if args.cli:
        run_cli(["merge", str(output / "shards"), str(output / "index")])
    else:
        index_builder.merge_indexes(shards, str(output / "index"))
    merged = time.perf_counter()
    if args.cli:
        shards = list(map(str, sorted((output / "shards").glob("*.mpk.gz"))))
    reader = indexing.DiskBasedIndex(str(output / "index"))
    names = list(reader.to_external_doc_id)
    if len(names) != len(records) or set(names) != {row["id"] for row in records}:
        raise AssertionError("Indexed document IDs differ from the benchmark input")
    to_internal = {name: i for i, name in enumerate(names)}
    dtype = np.dtype([("hash", "<u8"), ("doc", "<u4"), ("position", "<u4")])
    expected = []
    for row in records:
        hashes, positions = configured.get_winnowed_fingerprints(row["text"])
        block = np.empty(len(hashes), dtype=dtype)
        block["hash"], block["doc"], block["position"] = hashes, to_internal[row["id"]], positions
        expected.append(block)
    expected = np.concatenate(expected)
    actual_postings = np.memmap(
        output / "index/postings.dat", dtype=[("doc", "<u4"), ("position", "<u4")], mode="r",
    )
    actual = np.empty(len(actual_postings), dtype=dtype)
    actual["hash"] = np.repeat(reader.fingerprints, reader.lengths.astype(np.int64))
    actual["doc"], actual["position"] = actual_postings["doc"], actual_postings["position"]
    expected.sort(order=["hash", "doc", "position"])
    actual.sort(order=["hash", "doc", "position"])
    np.testing.assert_array_equal(actual, expected)
    hash_times = []
    for _ in range(3):
        started = time.perf_counter()
        for row in records:
            configured.get_winnowed_fingerprints(row["text"])
        hash_times.append(time.perf_counter() - started)
    result = {
        "extraction_seconds": extracted - build_started,
        "total_seconds": merged - build_started,
        "merge_seconds": merged - extracted, "documents": len(records),
        "fingerprints": len(reader.fingerprints), "occurrences": len(actual),
        "all_source_occurrences_verified": True,
        "index_bytes": sum(path.stat().st_size for path in (output / "index").iterdir()),
        "intermediate_bytes": sum(Path(path).stat().st_size for path in shards),
        "tokenization_to_winnowing_seconds": statistics.median(hash_times),
        "jit_warmup_seconds_excluded": warmup,
        "timing_mode": "startup_inclusive_cli" if args.cli else "warmed_api",
        "intermediate_shards": len(shards),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cli", action="store_true",
        help="Time separate CLI processes with startup/JIT and default flushing (requires --start-method fork)",
    )
    parser.add_argument("--start-method", choices=mp.get_all_start_methods(), default=mp.get_start_method())
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cli and args.start_method != "fork":
        parser.error("--cli comparison requires --start-method fork to run unchanged original main")
    if args.checkout:
        result = build_one(args)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        return
    if args.baseline is None or min(args.documents, args.repeats, args.workers) < 1:
        parser.error("Provide --baseline and positive documents/repeats/workers")
    output = args.output.resolve()
    output.mkdir()
    sys.path.insert(0, str(args.candidate.resolve()))
    from findmytext import utils

    stream = (
        utils.stream_json_zst(str(args.corpus), max_length=100_000)
        if str(args.corpus).endswith(".zst") else utils.stream_jsonl(str(args.corpus))
    )
    sample = output / "input.jsonl"
    count = 0
    with sample.open("w") as target:
        try:
            for record in stream:
                target.write(json.dumps({"id": str(record["id"]), "text": record["text"]}) + "\n")
                count += 1
                if count == args.documents:
                    break
        finally:
            stream.close()
    if count != args.documents:
        raise ValueError(f"Only {count} eligible benchmark documents found")
    runs = []
    for repeat in range(args.repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for label in order:
            checkout = getattr(args, label).resolve()
            destination = output / f"{label}-{repeat + 1}"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--checkout", str(checkout), "--corpus", str(sample), "--output", str(destination),
                "--workers", str(args.workers), "--start-method", args.start_method,
            ]
            if args.cli:
                command.append("--cli")
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            with (output / f"{label}-{repeat + 1}.log").open("w") as log:
                subprocess.run(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            runs.append({"label": label, "repeat": repeat + 1, **json.loads((destination / "result.json").read_text())})
            print(label, repeat + 1, runs[-1], flush=True)
    result = {
        "documents": count, "workers_argument": args.workers, "start_method": args.start_method,
        "python": platform.python_version(), "platform": platform.platform(),
        "input_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "scope": "Extraction + positional merge on the sampled JSONL, not a full-corpus build.",
        "timing_mode": "startup_inclusive_cli" if args.cli else "warmed_api",
        "flush_documents": 1_000_000 if args.cli else 2000,
        "timing": (
            "Separate CLI index/merge processes, including their interpreter/import/startup/JIT time; no explicit JIT warm-up, normal existing caches retained."
            if args.cli else "Fresh child per build, warmed parent JIT; interpreter/import/JIT warm-up excluded."
        ) + " Alternate old/new order; input sampling, occurrence verification and separate hashing loop excluded. Timed writes use normal OS buffering, without a durability fsync.",
        "runs": runs,
        "medians": {
            label: {
                field: statistics.median(row[field] for row in runs if row["label"] == label)
                for field in ("extraction_seconds", "merge_seconds", "total_seconds", "tokenization_to_winnowing_seconds")
            } for label in ("baseline", "candidate")
        },
    }
    (output / "report.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
