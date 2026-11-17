"""Progressive Nynorsk subset benchmarks with disk outputs and resume support."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from experiments.overlap.benchmark_overlap_join import _validate, choose_memory_budget
from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_join import prepare_subset, run_join, warm_up
from findmytext.overlap_stream import (
    atomic_json, external_reference, load_subset, run_document_stream,
    save_subset, subset_identity, validate_stream,
)


def _reference_summary(root: Path, thresholds: list[int]) -> dict:
    rows = np.load(root / "counts.npy", mmap_mode="r")
    result = {str(k): 0 for k in thresholds}
    started = time.perf_counter()
    for start in range(0, len(rows), 1_000_000):
        counts = rows["count"][start:start + 1_000_000]
        for k in thresholds:
            result[str(k)] += int(np.count_nonzero(counts >= k))
    return {"pairs_per_k": result, "filter_all_k_seconds": time.perf_counter() - started}


def scale_benchmark(
    index_dir: str | Path, overlap_dir: str | Path, output_dir: str | Path,
    sizes: tuple[int, ...] = (25_000, 50_000, 100_000),
    k_values: tuple[int, ...] = (50, 30, 20, 10, 5),
    seed: int = 41, candidate_df_cap: int = 1000,
    memory_budget_bytes: int | None = None,
    reference_batch_entries: int = 1_000_000,
    fingerprint_control: bool = True,
    prefix_filter: bool = False,
    run_controls: bool = True,
) -> dict:
    source = Path(index_dir).resolve()
    overlap = Path(overlap_dir).resolve()
    root = Path(output_dir).resolve()
    if any(root == path or path in root.parents for path in (source, overlap)):
        raise ValueError("Experiments must be outside both source indexes")
    if not sizes or min(sizes) < 2 or not k_values or min(k_values) < 1 or candidate_df_cap < 2:
        raise ValueError("Invalid subset sizes, K thresholds, or candidate cap")
    sizes = tuple(sorted(set(sizes)))
    thresholds = sorted(set(k_values), reverse=True)
    identity = _source_identity(source)
    n_docs = len(np.load(source / "doc_name_offsets.npy", mmap_mode="r")) - 1
    if max(sizes) > n_docs:
        raise ValueError("Subset size exceeds the source document count")
    if memory_budget_bytes is None:
        memory_budget_bytes, available = choose_memory_budget()
    else:
        available = None
    contract = {
        "format": "findmytext-progressive-overlap-v1", "source": identity,
        "overlap": str(overlap), "sizes": list(sizes), "k_values": thresholds,
        "seed": seed, "candidate_df_cap": candidate_df_cap,
        "memory_budget_bytes": memory_budget_bytes,
        "reference_batch_entries": reference_batch_entries,
        "fingerprint_control": fingerprint_control,
    }
    if prefix_filter:
        contract["prefix_filter"] = "both_rarity_prefixes_length_L_minus_K_plus_1_v1"
    if not run_controls:
        contract["run_controls"] = False
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["contract"] != contract:
            raise ValueError("Progressive benchmark contract mismatch")
    else:
        if any(root.iterdir()):
            raise ValueError("Untracked benchmark directory")
        report = {
            "contract": contract, "status": "running", "sizes": {},
            "host_available_bytes_at_start": available,
            "deadline": None, "parallel_workers": 1,
            "timing_repetitions": 1,
            "io_scope": "actual full-source forward memmaps with naturally warm OS/NFS caches; no cache flushing",
            "memory_scope": "owned arrays/buffers bounded separately from OS-managed mapped pages and JIT/imports",
        }
        atomic_json(report_path, report)
    master_ids = np.random.default_rng(seed).choice(n_docs, size=max(sizes), replace=False)
    warm_up()
    report["status"] = "running"
    atomic_json(report_path, report)
    suite_start = time.perf_counter()
    try:
        for size in sizes:
            label = str(size)
            stage = root / label
            stage.mkdir(exist_ok=True)
            stage_report = report["sizes"].setdefault(label, {"status": "running", "runs": []})
            if stage_report["status"] == "passed":
                continue
            print(f"[scale benchmark] starting {size:,} documents", flush=True)
            ids = np.sort(master_ids[:size]).astype(np.uint32)
            subset_dir = stage / "subset"
            phase = time.perf_counter()
            if subset_dir.exists():
                data = load_subset(subset_dir)
                np.testing.assert_array_equal(data.document_ids, ids)
            else:
                data = prepare_subset(source, overlap, ids, memory_budget_bytes)
                save_subset(data, subset_dir, identity)
                data = load_subset(subset_dir)
            stage_report["preparation_seconds"] = time.perf_counter() - phase
            stage_report["prepared_view_bytes"] = data.nbytes
            stage_report["subset"] = subset_identity(data)
            stage_report["subset_df_scope"] = "full-corpus rarity and cap; subset-eligible length"
            reference_dir = stage / "reference"
            if (reference_dir / "report.json").exists():
                stage_report["reference"] = json.loads((reference_dir / "report.json").read_text())
                if stage_report["reference"]["subset"] != subset_identity(data):
                    raise ValueError("Reference/subset identity mismatch")
            else:
                stage_report["reference"] = external_reference(
                    data, reference_dir, batch_entries=reference_batch_entries,
                )
            stage_report["reference_count_sweep"] = _reference_summary(reference_dir, thresholds)
            atomic_json(report_path, report)
            for cap in (None, candidate_df_cap):
                for k in thresholds:
                    for mode in ("exact_counts", "threshold_only"):
                        name = f"k{k}_{mode}_df{'all' if cap is None else cap}"
                        output = stage / name
                        checkpoint = run_document_stream(
                            data, output, k, source_index_dir=source, output_mode=mode,
                            candidate_df_cap=cap, mmap_forward=True,
                            prefix_filter=prefix_filter,
                        )
                        validation = validate_stream(output, reference_dir)
                        entry = {
                            "name": name, "k": k, "mode": mode, "cap": cap,
                            "prefix_filter": prefix_filter,
                            "elapsed_seconds": checkpoint["elapsed_seconds"],
                            "metrics": checkpoint["metrics"],
                            "estimated_owned_bytes": checkpoint["estimated_owned_bytes"],
                            "marker_bytes": checkpoint["marker_bytes"],
                            "posting_entries_removed_before_enumeration": checkpoint["posting_entries_removed_before_enumeration"],
                            "probes_removed_before_enumeration": checkpoint["probes_removed_before_enumeration"],
                            "output_bytes": sum(part["bytes"] for part in checkpoint["parts"]),
                            "io_counters": checkpoint["io_counters"],
                            **validation,
                        }
                        previous = next((x for x in stage_report["runs"] if x["name"] == name), None)
                        if previous is not None:
                            stage_report["runs"].remove(previous)
                        stage_report["runs"].append(entry)
                        counterpart = next((
                            x for x in stage_report["runs"]
                            if x["k"] == k and x["cap"] == cap and x["mode"] != mode
                        ), None)
                        if counterpart and counterpart["ordered_pair_digest"] != entry["ordered_pair_digest"]:
                            raise AssertionError("Exact-count and threshold-only streamed pair sets differ")
                        atomic_json(report_path, report)
                        print(f"[scale benchmark] {size:,} {name}: {entry['accepted_pairs']:,} pairs, {entry['elapsed_seconds']:.2f}s, recall={entry['recall']}", flush=True)
            if size == min(sizes) and run_controls:
                stage_report["pruning_controls"] = []
                for k in (max(thresholds), min(thresholds)):
                    checkpoint = run_document_stream(
                        data, stage / f"late_pruning_k{k}", k,
                        source_index_dir=source, early_pruning=False,
                    )
                    validation = validate_stream(stage / f"late_pruning_k{k}", reference_dir)
                    stage_report["pruning_controls"].append({
                        "k": k, "elapsed_seconds": checkpoint["elapsed_seconds"],
                        "metrics": checkpoint["metrics"], **validation,
                    })
                if fingerprint_control:
                    stage_report["scheduler_ram_controls"] = []
                    reference = np.load(reference_dir / "counts.npy", mmap_mode="r")
                    for k in (max(thresholds), min(thresholds)):
                        expected = reference[reference["count"] >= k]
                        for cap in (None, candidate_df_cap):
                            for mode in ("exact_counts", "threshold_only"):
                                rows, metrics = run_join(
                                    data, k, scheduler="fingerprint", output_mode=mode,
                                    candidate_df_cap=cap,
                                )
                                _validate(rows, expected, mode == "exact_counts", cap is None)
                                path = stage / f"fingerprint_ram_k{k}_{mode}_df{cap}.npy"
                                np.save(path, rows)
                                streamed = run_document_stream(
                                    data, stage / f"document_ram_k{k}_{mode}_df{cap}", k,
                                    source_index_dir=source, output_mode=mode,
                                    candidate_df_cap=cap, mmap_forward=False,
                                )
                                v = validate_stream(stage / f"document_ram_k{k}_{mode}_df{cap}", reference_dir)
                                stage_report["scheduler_ram_controls"].append({
                                    "k": k, "cap": cap, "mode": mode,
                                    "fingerprint_prepared_ram": metrics,
                                    "document_prepared_ram_seconds": streamed["elapsed_seconds"],
                                    "document_validation": v,
                                    "caveat": "document time includes checkpoint/fsync output; not an isolated kernel timing",
                                })
            if _source_identity(source) != identity:
                raise ValueError("Source changed during progressive benchmark")
            stage_report["status"] = "passed"
            stage_report["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            atomic_json(report_path, report)
            print(f"[scale benchmark] {size:,} PASSED", flush=True)
        report["status"] = "passed"
        report["source_unchanged"] = True
        report["latest_session_seconds"] = time.perf_counter() - suite_start
        atomic_json(report_path, report)
        return report
    finally:
        if report["status"] != "passed":
            report["status"] = "failed_or_interrupted"
            atomic_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("overlap_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[25000, 50000, 100000])
    parser.add_argument("--k", type=int, nargs="+", default=[50, 30, 20, 10, 5])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--candidate-df-cap", type=int, default=1000)
    parser.add_argument("--memory-gib", type=float)
    parser.add_argument("--prefix-filter", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-fingerprint-control", action="store_true")
    args = parser.parse_args()
    scale_benchmark(
        args.index_dir, args.overlap_dir, args.output_dir, sizes=tuple(args.sizes),
        k_values=tuple(args.k), seed=args.seed, candidate_df_cap=args.candidate_df_cap,
        memory_budget_bytes=int(args.memory_gib * 2**30) if args.memory_gib else None,
        prefix_filter=args.prefix_filter, run_controls=not args.no_controls,
        fingerprint_control=not args.no_fingerprint_control,
    )


if __name__ == "__main__":
    main()
