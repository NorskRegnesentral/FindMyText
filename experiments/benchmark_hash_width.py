"""Throughput comparison of production hash methods and independent reference.

Compare production rolling64-v1 with an independent BLAKE2b-64 reference.
Legacy benchmark results are frozen in HASHING_FINDINGS.md; legacy execution
is no longer supported.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.hash_collision_study import ROOT, frame_tokens, wide_fingerprints, winnow_uint64
from findmytext import utils
from findmytext.winnower import (
    ROLLING64_HASH, ROLLING_BASE64, Winnower,
    winnow,
)

METHODS = ("rolling64_production", "blake2b64")


def fingerprint_tokens(tokens, method, length, window_size):
    if method == "blake2b64":
        _, values = wide_fingerprints(frame_tokens(tokens), length)
        return winnow_uint64(values, window_size)
    if method == "rolling64_production":
        configured = Winnower(length=length, window_size=window_size, hash_method=ROLLING64_HASH)
        return winnow(configured.get_token_fingerprints(tokens), window_size)
    raise ValueError(f"Unknown or retired hashing method: {method}")


def run_benchmark(corpus_dir, documents, repeats, output_path):
    if min(documents, repeats) < 1:
        raise ValueError("Documents and repeats must be positive")
    texts = []
    sources = []
    for shard in sorted(corpus_dir.glob("*.jsonl.zst")):
        count = 0
        stream = utils.stream_json_zst(str(shard), min_length=100, max_length=100_000)
        try:
            for record in stream:
                texts.append(record["text"])
                count += 1
                if len(texts) == documents:
                    break
        finally:
            stream.close()
        sources.append({"shard": str(shard), "documents": count})
        if len(texts) == documents:
            break
    if len(texts) != documents:
        raise ValueError(f"Only {len(texts)} documents available; requested {documents}")
    winnower = Winnower(length=4, window_size=6, punctuation=False)
    token_count = sum(len(winnower.tokenize(text)) for text in texts)
    kgram_count = sum(max(0, len(winnower.tokenize(text)) - 3) for text in texts)
    measurements = {method: [] for method in METHODS}
    selected_counts = {}
    before = perf_counter()
    for method in METHODS:
        fingerprint_tokens(winnower.tokenize(texts[0]), method, 4, 6)
    warmup_seconds = perf_counter() - before
    # Rotate execution order to reduce fixed ordering effects; all methods see identical text.
    for repeat in range(repeats):
        order = METHODS[repeat % len(METHODS):] + METHODS[:repeat % len(METHODS)]
        for method in order:
            selected = 0
            before = perf_counter()
            for text in texts:
                hashes, _ = fingerprint_tokens(winnower.tokenize(text), method, 4, 6)
                selected += len(hashes)
            elapsed = perf_counter() - before
            measurements[method].append(elapsed)
            selected_counts[method] = selected
            print(f"{method}: {elapsed:.3f}s, {kgram_count / elapsed:,.0f} k-grams/s", flush=True)
    current_time = median(measurements["rolling64_production"])
    report = {
        "documents": documents, "tokens": token_count, "kgram_occurrences": kgram_count,
        "repeats": repeats, "sources": sources, "jit_warmup_seconds_excluded": warmup_seconds,
        "scope": "First eligible Nynorsk documents in sorted shard order; a throughput workload, "
                 "not a uniform sample or a full-corpus collision study.",
        "timing_includes": "Tokenization, UTF-8 preparation, token/k-gram hashing and winnowing. "
                           "Excludes corpus decompression, index I/O, grouping and JIT warmup.",
        "rolling64_design": {
            "token_hash": "Unreduced legacy FNV-1a-style uint64 token hash, compiled with Numba",
            "rolling": "Polynomial arithmetic modulo 2**64 with unsigned wraparound",
            "base": ROLLING_BASE64,
            "output_mixing": "MurmurHash3 fmix64 (bijective); mix only emitted k-gram state",
            "production_hash_method": ROLLING64_HASH,
            "status": "Versioned opt-in non-cryptographic production method; "
                      "collision rates must be measured on each corpus.",
        },
        "methods": {
            method: {
                "seconds": times, "median_seconds": median(times),
                "median_kgrams_per_second": kgram_count / median(times),
                "speedup_vs_rolling64": current_time / median(times),
                "selected_occurrences": selected_counts[method],
            }
            for method, times in measurements.items()
        },
    }
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "nno_Latn")
    parser.add_argument("--documents", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_benchmark(args.corpus_dir, args.documents, args.repeats, args.output)


if __name__ == "__main__":
    main()
