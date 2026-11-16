"""Compare exact single-worker overlap joins on a seeded indexed subset.

This is a bounded Nynorsk-first benchmark, not a full-corpus join command.
No source index is modified. Candidate caps use full-corpus document frequency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_join import (
    PAIR_DTYPE,
    prepare_subset,
    reference_counts,
    run_join,
    warm_up,
)


def choose_memory_budget() -> tuple[int, int]:
    """Use the agreed 32 GiB working allowance, checking host headroom."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            budget = 32 * 2**30
            if available < budget:
                raise MemoryError("Less than 32 GiB available; choose an explicit smaller --memory-gib allowance")
            return budget, available
    raise RuntimeError("Cannot determine available RAM; specify --memory-gib")


def _pairs(result: np.ndarray) -> np.ndarray:
    pairs = np.empty(len(result), dtype=PAIR_DTYPE)
    pairs["doc_i"], pairs["doc_j"] = result["doc_i"], result["doc_j"]
    return pairs


def _keys(result: np.ndarray) -> np.ndarray:
    return (result["doc_i"].astype(np.uint64) << np.uint64(32)) | result["doc_j"]


def _validate(result: np.ndarray, expected: np.ndarray, exact: bool, exhaustive: bool) -> None:
    keys = _keys(result)
    if np.any(keys[1:] <= keys[:-1]):
        raise AssertionError("Join emitted duplicate or noncanonical/unordered pairs")
    if exhaustive:
        np.testing.assert_array_equal(result, expected if exact else _pairs(expected))
        return
    expected_keys = _keys(expected)
    positions = np.searchsorted(expected_keys, keys)
    if np.any(positions >= len(expected_keys)) or not np.array_equal(expected_keys[positions], keys):
        raise AssertionError("Capped run emitted a pair not qualifying in the exact reference")
    if exact:
        np.testing.assert_array_equal(result["count"], expected["count"][positions])


def benchmark(
    index_dir: str | Path,
    overlap_dir: str | Path,
    output_dir: str | Path,
    *,
    n_documents: int = 10_000,
    seed: int = 41,
    k_values: tuple[int, ...] = (50, 30, 20, 10, 5),
    candidate_df_cap: int = 1000,
    memory_budget_bytes: int | None = None,
    batch_size: int = 65_536,
    repetitions: int = 3,
) -> dict:
    if n_documents < 2 or not k_values or any(k < 1 for k in k_values):
        raise ValueError("Use at least two documents and positive overlap thresholds")
    if candidate_df_cap < 2 or batch_size < 1 or repetitions < 1:
        raise ValueError("Use a df cap of at least two and positive batch size/repetitions")
    source = Path(index_dir).resolve()
    overlap = Path(overlap_dir).resolve()
    output = Path(output_dir).resolve()
    if source == output or source in output.parents or overlap == output or overlap in output.parents:
        raise ValueError("Benchmark output must be outside both source indexes")
    if output.exists():
        raise FileExistsError(f"Benchmark destination already exists: {output}")
    identity = _source_identity(source)
    n_source_docs = len(np.load(source / "doc_name_offsets.npy", mmap_mode="r")) - 1
    if n_documents > n_source_docs:
        raise ValueError("Requested subset exceeds source document count")
    available = None
    if memory_budget_bytes is None:
        memory_budget_bytes, available = choose_memory_budget()
    if memory_budget_bytes < 1:
        raise ValueError("Memory budget must be positive")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": identity,
        "overlap_index": str(overlap),
        "source_document_count": n_source_docs,
        "subset_documents": n_documents,
        "subset_sampling": "seeded_uniform_without_replacement_sorted_source_ids",
        "seed": seed,
        "k_values": sorted(set(k_values), reverse=True),
        "candidate_df_cap": candidate_df_cap,
        "df_scope": "full_source_corpus",
        "eligible_length_scope": "fingerprints_shared_with_at_least_one_other_selected_document",
        "memory_budget_bytes": memory_budget_bytes,
        "host_available_bytes_at_start": available,
        "memory_budget_scope": "prototype-owned arrays/state/output buffers; not mapped-page RSS or JIT/imports",
        "deadline": None,
        "timing_scope": "median of repeated warmed runs; prepared subset is in RAM; not full-corpus mmap/I/O throughput",
        "timing_repetitions": repetitions,
        "runs": [],
    }

    def save_report():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    save_report()
    try:
        print(
            f"[join benchmark] {n_documents:,} uniform documents, "
            f"{memory_budget_bytes / 2**30:.2f} GiB working budget; no deadline",
            flush=True,
        )
        phase = time.perf_counter()
        warm_up()
        report["jit_warmup_seconds"] = time.perf_counter() - phase
        ids = np.sort(np.random.default_rng(seed).choice(
            n_source_docs, size=n_documents, replace=False
        )).astype(np.uint32)
        np.save(output / "document_ids.npy", ids)
        report["document_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
        phase = time.perf_counter()
        data = prepare_subset(source, overlap, ids, memory_budget_bytes)
        report["preparation_seconds_including_view_kernel_jit"] = time.perf_counter() - phase
        report["prepared_arrays_bytes"] = data.nbytes
        report["subset_overlap_fingerprints"] = len(data.fingerprints)
        report["subset_overlap_doc_entries"] = len(data.posting_docs)
        print("[join benchmark] deriving independent exact reference from full forward sets", flush=True)
        reference, reference_report = reference_counts(data)
        report["reference"] = reference_report
        np.save(output / "reference_exact_counts.npy", reference)
        print(
            f"[join benchmark] reference: {len(reference):,} positive pairs, "
            f"{reference_report['pair_contributions']:,} pair contributions",
            flush=True,
        )
        cache, cache_report = run_join(data, 1, scheduler="fingerprint", batch_size=batch_size)
        np.testing.assert_array_equal(cache, reference)
        np.save(output / "reusable_exact_counts.npy", cache)
        report["reusable_exact_count_sweep"] = {
            "discovery_run": cache_report,
            "scope": "all discovered pairs in the complete uncapped subset, including counts below sweep thresholds",
            "validity": "same source, selected document IDs and full-forward fingerprint semantics; counts are independent of K",
            "thresholds": [],
        }
        for k in report["k_values"]:
            phase = time.perf_counter()
            selected = cache[cache["count"] >= k]
            np.testing.assert_array_equal(selected, reference[reference["count"] >= k])
            report["reusable_exact_count_sweep"]["thresholds"].append({
                "k": k, "accepted_pairs": len(selected),
                "filter_seconds": time.perf_counter() - phase,
            })
        del cache
        pair_digests = {}
        for cap in (None, candidate_df_cap):
            for k in report["k_values"]:
                expected = reference[reference["count"] >= k]
                for scheduler in ("fingerprint", "document"):
                    for mode in ("exact_counts", "threshold_only"):
                        result, metrics = run_join(
                            data, k, scheduler=scheduler, output_mode=mode,
                            candidate_df_cap=cap, batch_size=batch_size,
                        )
                        _validate(result, expected, mode == "exact_counts", cap is None)
                        timings = [metrics["elapsed_seconds"]]
                        for _ in range(1, repetitions):
                            del result
                            result, metrics = run_join(
                                data, k, scheduler=scheduler, output_mode=mode,
                                candidate_df_cap=cap, batch_size=batch_size,
                            )
                            timings.append(metrics["elapsed_seconds"])
                            _validate(result, expected, mode == "exact_counts", cap is None)
                        metrics["elapsed_seconds_samples"] = timings
                        metrics["elapsed_seconds"] = float(np.median(timings))
                        digest = hashlib.sha256(_pairs(result).tobytes()).hexdigest()
                        key = cap, k
                        if key in pair_digests and digest != pair_digests[key]:
                            raise AssertionError("Schedulers or output modes returned different pair sets")
                        pair_digests[key] = digest
                        label = f"{scheduler}_k{k}_{mode}_df{'all' if cap is None else cap}"
                        np.save(output / f"{label}.npy", result)
                        metrics.update({
                            "validated": True,
                            "pair_set_sha256": digest,
                            "reference_qualifying_pairs": len(expected),
                            "missing_reference_pairs": len(expected) - len(result),
                            "recall_against_subset_reference": len(result) / len(expected) if len(expected) else None,
                            "result_file": f"{label}.npy",
                        })
                        report["runs"].append(metrics)
                        save_report()
                        print(
                            f"[join benchmark] {label}: {len(result):,} pairs, "
                            f"{metrics['verification_calls']:,} verifications, "
                            f"{metrics['elapsed_seconds']:.3f}s; validated",
                            flush=True,
                        )
                        del result
        if _source_identity(source) != identity:
            raise ValueError("Source index changed during the benchmark")
        report["source_unchanged"] = True
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["status"] = "passed"
        save_report()
        print(f"[join benchmark] PASSED; report: {output / 'report.json'}", flush=True)
        return report
    finally:
        if report["status"] != "passed":
            report["status"] = "failed_or_interrupted"
            report["elapsed_seconds"] = time.perf_counter() - started
            save_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("overlap_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-documents", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 30, 20, 10, 5])
    parser.add_argument("--candidate-df-cap", type=int, default=1000)
    parser.add_argument("--memory-gib", type=float)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    benchmark(
        args.index_dir, args.overlap_dir, args.output_dir,
        n_documents=args.n_documents, seed=args.seed, k_values=tuple(args.k),
        candidate_df_cap=args.candidate_df_cap,
        memory_budget_bytes=int(args.memory_gib * 2**30) if args.memory_gib is not None else None,
        batch_size=args.batch_size,
        repetitions=args.repetitions,
    )


if __name__ == "__main__":
    main()
