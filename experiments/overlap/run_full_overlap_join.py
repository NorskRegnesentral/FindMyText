"""Validate exact rarity prefixes, pilot full partners, then join one full corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout
from numba import njit

from findmytext.overlap_full import load_full_corpus, prepare_full_corpus
from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_join import _verify, warm_up
from findmytext.overlap_stream import (
    _check_parts, atomic_json, filter_count_cache, load_subset,
    run_document_stream, validate_stream,
)
from findmytext.similarity import SIMILARITY_DTYPE


def validate_prefix(source: Path, reference_stage: Path, root: Path, ks: tuple[int, ...], budget: int) -> dict:
    data = load_subset(reference_stage / "subset")
    data.memory_budget_bytes = budget
    reference = reference_stage / "reference"
    report = {"status": "running", "runs": [], "subset_documents": len(data.document_ids)}
    root.mkdir(parents=True, exist_ok=True)
    # Measure the lowest K first: it is the intended full-run cache floor.
    order = [min(ks), *[k for k in sorted(set(ks), reverse=True) if k != min(ks)]]
    for k in order:
        for mode in ("exact_counts", "threshold_only"):
            output = root / f"k{k}_{mode}"
            cp = run_document_stream(
                data, output, k, source_index_dir=source, output_mode=mode, prefix_filter=True,
            )
            result = validate_stream(output, reference)
            baseline = json.loads((reference_stage / f"k{k}_{mode}_dfall/checkpoint.json").read_text())
            entry = {
                "k": k, "mode": mode, "elapsed_seconds": cp["elapsed_seconds"],
                "metrics": cp["metrics"], "baseline_metrics": baseline["metrics"],
                "baseline_seconds": baseline["elapsed_seconds"], **result,
                "baseline_used_prefix_filter": bool(baseline["contract"].get("prefix_filter")),
            }
            report["runs"].append(entry)
            atomic_json(root / "report.json", report)
            print(
                f"[prefix validation] K={k} {mode}: {result['accepted_pairs']:,} pairs; "
                f"{cp['metrics']['verification_calls']:,} verifications vs "
                f"{baseline['metrics']['verification_calls']:,}; {cp['elapsed_seconds']:.2f}s",
                flush=True,
            )
    report["status"] = "passed"
    atomic_json(root / "report.json", report)
    return report


@njit(cache=True, nogil=True)
def _probe_reference_counts(a, probe, join_rank, doc_offsets, doc_ranks, offsets, docs, n):
    counts = np.zeros(n, dtype=np.uint32)
    for entry in range(np.intp(doc_offsets[a]), np.intp(doc_offsets[a + 1])):
        rank = doc_ranks[entry]
        for position in range(np.intp(offsets[rank]), np.intp(offsets[rank + 1])):
            b = docs[position]
            if join_rank[b] < probe:
                counts[b] += 1
    return counts


def _read_rows(directory: Path) -> np.ndarray:
    cp = json.loads((directory / "checkpoint.json").read_text())
    chunks = [np.load(directory / part["file"]) for part in cp["parts"]]
    rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=SIMILARITY_DTYPE)
    rows.sort(order=["doc_i", "doc_j"])
    return rows


def full_partner_pilot(data, source: Path, root: Path, k: int, seed: int = 41) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    n = len(data.document_ids)
    order = np.lexsort((data.document_ids, data.eligible_lengths))
    join_rank = np.empty(n, dtype=np.int64)
    join_rank[order] = np.arange(n)
    active = np.flatnonzero(data.eligible_lengths[order] >= k)
    if not len(active):
        raise ValueError("No document can reach the pilot threshold")
    rng = np.random.default_rng(seed)
    report = {
        "status": "running", "full_document_count": n, "k": k,
        "selection": "four-owner probe ranges near active-length rank quantiles 10%,50%,90%,99%",
        "partner_scope": "all earlier-owned documents from the complete corpus",
        "exhaustive_corpus_join": False, "strata": [],
    }
    for quantile in (0.1, 0.5, 0.9, 0.99):
        position = min(int(quantile * len(active)), max(0, len(active) - 4))
        start = int(active[position])
        stop = min(start + 4, n)
        expected_chunks = []
        checks = 0
        for probe in range(start, stop):
            a = int(order[probe])
            counts = _probe_reference_counts(
                a, probe, join_rank, data.document_fp_offsets, data.document_fp_ranks,
                data.posting_offsets, data.posting_docs, n,
            )
            candidates = np.flatnonzero(counts >= k)
            rows = np.empty(len(candidates), dtype=SIMILARITY_DTYPE)
            rows["doc_i"] = np.minimum(data.document_ids[a], data.document_ids[candidates])
            rows["doc_j"] = np.maximum(data.document_ids[a], data.document_ids[candidates])
            rows["count"] = counts[candidates]
            expected_chunks.append(rows)
            earlier = order[:probe]
            sample = rng.choice(earlier, size=min(64, len(earlier)), replace=False)
            if len(candidates):
                sample = np.unique(np.r_[sample, rng.choice(candidates, size=min(64, len(candidates)), replace=False)])
            for b in sample:
                _, actual, _, _ = _verify(
                    data.forward_fingerprints, data.forward_offsets, a, int(b), 1, True,
                )
                if actual != counts[b]:
                    raise AssertionError("Pilot posting-count oracle differs from full-forward intersection")
                checks += 1
        expected = np.concatenate(expected_chunks)
        expected.sort(order=["doc_i", "doc_j"])
        entry = {
            "quantile": quantile, "probe_range": [start, stop],
            "eligible_lengths": data.eligible_lengths[order[start:stop]].tolist(),
            "expected_pairs": len(expected), "full_forward_crosschecks": checks,
            "runs": [],
        }
        for prefix in (False, True):
            output = root / f"q{quantile}_{'prefix' if prefix else 'unfiltered'}"
            cp = run_document_stream(
                data, output, k, source_index_dir=source, prefix_filter=prefix,
                probe_start=start, probe_stop=stop,
            )
            np.testing.assert_array_equal(_read_rows(output), expected)
            entry["runs"].append({
                "prefix": prefix, "elapsed_seconds": cp["elapsed_seconds"],
                "probe_seconds": cp["probe_seconds"],
                "setup_seconds": cp["setup_seconds_last_session"],
                "metrics": cp["metrics"], "accepted_pairs": cp["accepted_pairs"],
                "matches_independent_unfiltered_posting_oracle": True,
            })
            print(
                f"[full pilot] q={quantile} prefix={prefix}: {len(expected):,} pairs, "
                f"{cp['metrics']['verification_calls']:,} verifications, "
                f"{cp['probe_seconds']:.2f}s probe time",
                flush=True,
            )
        report["strata"].append(entry)
        atomic_json(root / "report.json", report)
    report["status"] = "passed"
    atomic_json(root / "report.json", report)
    return report


@njit(cache=True, nogil=True)
def _check_full_rows(rows, join_rank, eligible, markers, previous_owner, floor):
    for row in rows:
        a, b, count = row["doc_i"], row["doc_j"], row["count"]
        if a >= b or b >= len(join_rank) or count < floor or count > min(eligible[a], eligible[b]):
            raise ValueError("Invalid full-run pair IDs or count bounds")
        if join_rank[a] > join_rank[b]:
            owner, partner = join_rank[a], b
        else:
            owner, partner = join_rank[b], a
        if owner < previous_owner:
            raise ValueError("Full-run owner order moved backwards")
        if markers[partner] == owner + 1:
            raise ValueError("Full-run output contains a duplicate pair")
        markers[partner] = owner + 1
        previous_owner = owner
    return previous_owner


def verify_full_output(data, source: Path, directory: Path, seed: int = 41) -> dict:
    cp = json.loads((directory / "checkpoint.json").read_text())
    if cp["status"] != "complete" or not cp["exhaustive_within_subset"] or cp["next_probe"] != len(data.document_ids):
        raise ValueError("Full-run output is incomplete or capped")
    if cp["contract"]["source"] != _source_identity(source):
        raise ValueError("Full-run source identity mismatch")
    source_documents = len(np.load(source / "doc_name_offsets.npy", mmap_mode="r")) - 1
    if len(data.document_ids) != source_documents:
        raise ValueError("Full-run verification requires all source documents")
    _check_parts(directory, cp, SIMILARITY_DTYPE)
    n = len(data.document_ids)
    np.testing.assert_array_equal(data.document_ids, np.arange(n, dtype=np.uint32))
    order = np.lexsort((data.document_ids, data.eligible_lengths))
    join_rank = np.empty(n, dtype=np.int64)
    join_rank[order] = np.arange(n)
    markers = np.zeros(n, dtype=np.uint32)
    total = cp["accepted_pairs"]
    sample_indices = np.sort(np.random.default_rng(seed).choice(total, size=min(total, 10_000), replace=False))
    samples = []
    cursor = 0
    previous_owner = -1
    distribution = {str(k): 0 for k in (50, 30, 20, 10, 5)}
    digest = hashlib.sha256()
    for part in cp["parts"]:
        rows = np.load(directory / part["file"])
        previous_owner = _check_full_rows(
            rows, join_rank, data.eligible_lengths, markers, previous_owner, cp["contract"]["k"],
        )
        first, last = np.searchsorted(sample_indices, [cursor, cursor + len(rows)])
        if last > first:
            samples.append(rows[sample_indices[first:last] - cursor].copy())
        for k in distribution:
            distribution[k] += int(np.count_nonzero(rows["count"] >= int(k)))
        keys = (rows["doc_i"].astype(np.uint64) << np.uint64(32)) | rows["doc_j"]
        digest.update(keys.tobytes())
        cursor += len(rows)
    if cursor != total:
        raise AssertionError("Full-run part-row accounting mismatch")
    checked = 0
    for chunk in samples:
        for row in chunk:
            a, b = int(row["doc_i"]), int(row["doc_j"])
            first = data.forward_fingerprints[int(data.forward_offsets[a]):int(data.forward_offsets[a + 1])]
            second = data.forward_fingerprints[int(data.forward_offsets[b]):int(data.forward_offsets[b + 1])]
            count = len(np.intersect1d(first, second, assume_unique=True))
            if count < cp["contract"]["k"] or count != row["count"]:
                raise AssertionError("Full-run sampled count differs from numeric full-forward intersection")
            checked += 1
    report = {
        "status": "passed", "source": cp["contract"]["source"],
        "documents": n, "accepted_pairs": total, "counts_per_k": distribution,
        "all_output_checksums_bounds_ownership_and_duplicates_checked": True,
        "independent_numeric_forward_samples": checked, "sample_seed": seed,
        "numeric_sample_method": "numpy_intersect1d_assume_unique",
        "ordered_pair_digest": digest.hexdigest(),
        "completeness_basis": "lossless prefix bound, source-view integrity, subset oracle equality and independent full-partner pilots",
        "independent_exhaustive_full_corpus_oracle_run": False,
        "source_unchanged": _source_identity(source) == cp["contract"]["source"],
    }
    if not report["source_unchanged"]:
        raise ValueError("Source changed during full-run verification")
    atomic_json(directory / "verification.json", report)
    return report


def _run_pipeline(
    index_dir: str | Path, overlap_dir: str | Path, reference_stage: str | Path,
    output_dir: str | Path, memory_budget_bytes: int = 32 * 2**30,
) -> dict:
    source, overlap, reference, root = map(
        lambda path: Path(path).resolve(), (index_dir, overlap_dir, reference_stage, output_dir)
    )
    if any(root == path or path in root.parents for path in (source, overlap)):
        raise ValueError("Full-run output must be outside source indexes")
    root.mkdir(parents=True, exist_ok=True)
    source_identity = _source_identity(source)
    ks = (50, 30, 20, 10, 5)
    contract = {
        "format": "findmytext-full-overlap-pipeline-v1", "source": source_identity,
        "overlap_dir": str(overlap), "reference_stage": str(reference),
        "memory_budget_bytes": memory_budget_bytes, "floor_k": 5, "k_values": list(ks),
        "candidate_df_cap": None, "output_mode": "exact_counts", "parallel_workers": 1,
    }
    report_path = root / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["contract"] != contract:
            raise ValueError("Full-run pipeline contract mismatch")
    else:
        report = {"contract": contract, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    report["status"] = "running"
    atomic_json(report_path, report)
    try:
        warm_up()
        report["phase"] = "prefix_validation"
        atomic_json(report_path, report)
        reference_size = len(np.load(reference / "subset/document_ids.npy", mmap_mode="r"))
        report["prefix_validation"] = validate_prefix(
            source, reference, root / f"prefix-validation-{reference_size}", ks, memory_budget_bytes,
        )
        report["phase"] = "full_corpus_preparation"
        atomic_json(report_path, report)
        data = prepare_full_corpus(source, overlap, root / "full-view", memory_budget_bytes)
        report["full_view"] = json.loads((root / "full-view/meta.json").read_text())
        report["phase"] = "full_partner_pilot"
        atomic_json(report_path, report)
        report["pilot"] = full_partner_pilot(data, source, root / "pilot", min(ks))
        report["phase"] = "complete_exact_join"
        atomic_json(report_path, report)
        join_checkpoint = root / "join-k5/checkpoint.json"
        buffer_rows = (
            json.loads(join_checkpoint.read_text())["contract"].get("output_buffer_rows")
            if join_checkpoint.exists() else 1_000_000
        )
        report["join"] = run_document_stream(
            data, root / "join-k5", min(ks), source_index_dir=source, prefix_filter=True,
            output_buffer_rows=buffer_rows,
        )
        report["phase"] = "output_verification"
        atomic_json(report_path, report)
        report["verification"] = verify_full_output(data, source, root / "join-k5")
        report["phase"] = "derive_k_sweep"
        atomic_json(report_path, report)
        if (root / "k-sweep/report.json").exists():
            report["k_sweep"] = json.loads((root / "k-sweep/report.json").read_text())
        else:
            report["k_sweep"] = filter_count_cache(root / "join-k5", root / "k-sweep", ks)
        for k, item in report["k_sweep"]["thresholds"].items():
            if item["pairs"] != report["verification"]["counts_per_k"][k]:
                raise AssertionError("Derived K-sweep accounting mismatch")
        if _source_identity(source) != source_identity:
            raise ValueError("Source index changed during the full pipeline")
        report["status"] = "passed"
        report["phase"] = "complete"
        report["source_unchanged"] = True
        report["latest_session_seconds"] = time.perf_counter() - started
        report["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
        print(f"[full overlap] PASSED: {report['verification']['accepted_pairs']:,} pairs; {report_path}", flush=True)
        return report
    finally:
        if report["status"] != "passed":
            report["status"] = "failed_or_interrupted"
            report["latest_session_seconds"] = time.perf_counter() - started
            atomic_json(report_path, report)


def run_pipeline(
    index_dir: str | Path, overlap_dir: str | Path, reference_stage: str | Path,
    output_dir: str | Path, memory_budget_bytes: int = 32 * 2**30,
) -> dict:
    root = Path(output_dir).resolve()
    if any(root == Path(value).resolve() or Path(value).resolve() in root.parents for value in (index_dir, overlap_dir)):
        raise ValueError("Full-run output must be outside source indexes")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "pipeline.lock"), timeout=0):
        return _run_pipeline(index_dir, overlap_dir, reference_stage, root, memory_budget_bytes)


def finish_full_run(output_dir: str | Path, poll_seconds: int = 30) -> dict:
    """Wait for the persistent job, then independently check counts and every K file."""
    if poll_seconds < 1:
        raise ValueError("Completion polling interval must be positive")
    root = Path(output_dir).resolve()
    report_path = root / "report.json"
    while True:
        report = json.loads(report_path.read_text())
        if report["status"] == "passed":
            break
        if "failed" in report["status"] or report["status"] == "failed_or_interrupted":
            raise RuntimeError(f"Full pipeline failed: {report.get('phase')}; inspect run.log")
        try:
            with FileLock(str(root / "pipeline.lock"), timeout=0):
                current = json.loads(report_path.read_text())
                if current["status"] != "passed":
                    current["status"] = "failed_or_interrupted"
                    current["failure_reason"] = "Pipeline lock is free before completion"
                    atomic_json(report_path, current)
                    raise RuntimeError("Pipeline is no longer running but has not completed")
                break
        except Timeout:
            time.sleep(poll_seconds)
    with FileLock(str(root / "pipeline.lock"), timeout=120):
        report = json.loads(report_path.read_text())
        if report["status"] != "passed":
            raise RuntimeError("Pipeline state changed before independent verification")
        report["primary_pipeline_status"] = "passed"
        report["status"] = "running_independent_verification"
        report["phase"] = "independent_verification"
        atomic_json(report_path, report)
        started = time.perf_counter()
        complete = False
        try:
            data = load_full_corpus(root / "full-view", report["contract"]["memory_budget_bytes"])
            source = Path(report["contract"]["source"]["path"])
            verification = verify_full_output(data, source, root / "join-k5")
            with ExitStack() as streams:
                threshold_files = {}
                cursors = {}
                lengths = {}
                for k, entry in report["k_sweep"]["thresholds"].items():
                    stream = streams.enter_context((root / "k-sweep" / entry["file"]).open("rb", buffering=8 * 2**20))
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported derived NumPy format version: {version}")
                    if dtype != SIMILARITY_DTYPE or fortran or shape != (verification["counts_per_k"][k],):
                        raise AssertionError("Derived threshold file schema/cardinality mismatch")
                    threshold_files[k] = stream
                    lengths[k] = shape[0]
                    cursors[k] = 0
                checkpoint = json.loads((root / "join-k5/checkpoint.json").read_text())
                compared = 0
                for part in checkpoint["parts"]:
                    rows = np.load(root / "join-k5" / part["file"])
                    for k, target in threshold_files.items():
                        expected = rows[rows["count"] >= int(k)]
                        payload = target.read(expected.nbytes)
                        if len(payload) != expected.nbytes:
                            raise AssertionError("Derived threshold file was truncated")
                        np.testing.assert_array_equal(np.frombuffer(payload, dtype=SIMILARITY_DTYPE), expected)
                        cursors[k] += len(expected)
                        compared += len(expected)
                if any(cursors[k] != lengths[k] or stream.read(1) for k, stream in threshold_files.items()):
                    raise AssertionError("Derived threshold file row accounting mismatch")
            report["verification"] = verification
            report["additional_independent_checks"] = {
                "status": "passed",
                "numeric_sample_method": verification["numeric_sample_method"],
                "numeric_forward_samples": verification["independent_numeric_forward_samples"],
                "derived_threshold_rows_compared_to_exact_floor": compared,
                "elapsed_seconds": time.perf_counter() - started,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
            atomic_json(root / "independent_verification.json", {
                **verification, "additional_checks": report["additional_independent_checks"],
            })
            report["status"] = "passed"
            report["phase"] = "complete"
            report["independent_verification_finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(report_path, report)
            complete = True
            print(json.dumps({
                "status": "passed", "documents": verification["documents"],
                "counts_per_k": verification["counts_per_k"],
                "independent_checks": report["additional_independent_checks"],
                "report": str(report_path),
            }, indent=2), flush=True)
            return report
        finally:
            if not complete:
                report["status"] = "independent_verification_failed"
                atomic_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("overlap_dir", type=Path)
    parser.add_argument("reference_stage", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--memory-gib", type=float, default=32)
    parser.add_argument("--finish-only", action="store_true", help="Wait for an existing run and independently validate its completed output.")
    args = parser.parse_args()
    if args.finish_only:
        finish_full_run(args.output_dir)
    else:
        run_pipeline(args.index_dir, args.overlap_dir, args.reference_stage, args.output_dir, int(args.memory_gib * 2**30))


if __name__ == "__main__":
    main()
