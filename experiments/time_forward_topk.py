"""Time sample runs before full forward-top-k runs, preserving source indexes.

Run from the repository root with PYTHONPATH=. .venv/bin/python
experiments/time_forward_topk.py --output-dir /path/to/new/run-directory
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from findmytext.similarity import compute_similarities_forward_topk


ROOT = Path("/nr/project/stat/COPY.AI/data/Lars")
INPUTS = (
    "meta.json", "fingerprints.npy", "offsets.npy", "lengths.npy",
    "postings.dat", "doc_name_offsets.npy", "doc_name_bytes.npy",
    "forward_offsets.npy", "forward_fingerprints.npy",
)


def announce(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def prepare_directory(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        (target / name).symlink_to(source_file)


def timed_run(directory, documents, args):
    started = perf_counter()
    result = compute_similarities_forward_topk(
        index_dir=str(directory),
        top_k=args.top_k,
        min_shared_fingerprints=args.min_shared_fingerprints,
        posting_read_threads=1,
        max_documents=documents,
        max_chunks_per_merge=64,
        verbose=True,
    )
    elapsed = perf_counter() - started
    pairs = len(np.load(result, mmap_mode="r"))
    return elapsed, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", nargs="+", help="Optional name=index_dir entries.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-documents", type=int, default=25_000)
    parser.add_argument("--warmup-documents", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-shared-fingerprints", type=int, default=5)
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    if min(args.sample_documents, args.warmup_documents, args.top_k,
           args.min_shared_fingerprints) <= 0:
        parser.error("Document counts, top-k, and minimum shared fingerprints must be positive")

    entries = args.indexes or [
        f"{name}={ROOT / ('hplt_' + name) / 'findmytext_build/index'}"
        for name in ("nynorsk", "icelandic", "latvian", "bokmal")
    ]
    indexes = []
    for entry in entries:
        name, path = entry.split("=", 1)
        if not name or Path(name).name != name or name in {".", ".."}:
            parser.error("Index names must be simple directory names")
        source = Path(path).resolve()
        for filename in INPUTS:
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
        offsets = np.load(source / "doc_name_offsets.npy", mmap_mode="r")
        indexes.append((len(offsets) - 1, name, source))
    if len({name for _, name, _ in indexes}) != len(indexes):
        parser.error("Index names must be unique")
    indexes.sort()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "run.json"
    with manifest.open("x") as output:
        json.dump({
            "settings": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
            "indexes": [{"name": name, "documents": count, "source": str(source)}
                        for count, name, source in indexes],
        }, output, indent=2)

    rows = []
    estimates = {}
    stages = ("sample",) if args.sample_only else ("sample", "full")
    for stage in stages:
        for total, name, source in indexes:
            documents = min(args.sample_documents, total) if stage == "sample" else total
            if not documents:
                raise ValueError(f"Empty index: {source}")
            target = args.output_dir / name / stage
            warmup = target / "warmup"
            prepare_directory(source, target)
            prepare_directory(source, warmup)
            announce(f"{name} {stage}: warming up {min(args.warmup_documents, total):,} documents")
            timed_run(warmup, min(args.warmup_documents, total), args)
            announce(f"{name} {stage}: timing {documents:,}/{total:,} documents")
            elapsed, pairs = timed_run(target, documents, args)
            if stage == "sample":
                estimates[name] = elapsed * total / documents
            row = {
                "language": name, "stage": stage, "documents": documents,
                "total_documents": total, "seconds": elapsed, "pairs": pairs,
                "estimated_full_seconds": estimates[name],
                "output": str(target / "similarities_forward_topk.npy"),
            }
            rows.append(row)
            with (args.output_dir / "timings.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            announce(f"{name} {stage}: {elapsed:.1f}s; {pairs:,} pairs; "
                     f"sample-based full estimate {estimates[name] / 3600:.2f} hours")
    announce("All requested runs completed")


if __name__ == "__main__":
    main()