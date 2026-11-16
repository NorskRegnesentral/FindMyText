"""Gated sequential full joins after independently validated overnight pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
from filelock import FileLock

from experiments.overlap.overnight_prefix_benchmarks import (
    _available_memory, run_logged_command, timestamp,
)
from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_stream import atomic_json, validate_stream


def full_working_bounds(index_dir: str | Path, overlap_dir: str | Path) -> dict:
    source, overlap = Path(index_dir), Path(overlap_dir)
    meta = json.loads((overlap / "meta.json").read_text())
    n = meta["document_count"]
    f = meta["statistics"]["retained_fingerprints"]
    postings = meta["statistics"]["retained_doc_id_postings"]
    forward = sum(np.load(source / name, mmap_mode="r").nbytes for name in (
        "forward_offsets.npy", "forward_fingerprints.npy",
    ))
    side = f * 8 + (f + 1) * 8 + postings * 4 + n * 8
    logical = forward + side + n * 4 + f * 8 + (n + 1) * 8 + postings * 4
    preparation = forward + side + n * 32 + postings * 8 + f * 16
    join = logical + postings * 4 + (f + 1) * 8 + n * 36 + 65536 * 32 + 1_000_000 * 32
    return {
        "documents": n, "logical_view_bytes": logical,
        "preparation_bound_bytes": preparation, "join_bound_bytes": join,
        "required_working_bytes": max(preparation, join),
        "helper_array_bytes": n * 4 + f * 8 + (n + 1) * 8 + postings * 4,
    }


def review_pilot(job: dict, pilot_job: dict, budget: int) -> dict:
    if pilot_job["status"] != "passed":
        return {"status": "held_pilot_not_passed", "reason": pilot_job["status"]}
    largest = max(pilot_job["completed_stages"], key=lambda item: item["documents"])
    path = Path(largest["report"])
    report = json.loads(path.read_text())
    stage = report["sizes"][str(largest["documents"])]
    if (
        report["status"] != "passed" or stage["status"] != "passed"
        or len(stage["runs"]) != 20 or not all(row["validated"] and row["prefix_filter"] for row in stage["runs"])
        or report["contract"]["source"] != _source_identity(Path(job["index_dir"]))
    ):
        return {"status": "held_pilot_validation", "reason": "Pilot validation/settings/source checks did not pass"}
    stage_dir = path.parent / str(largest["documents"])
    checks = []
    for k, mode in ((5, "exact_counts"), (5, "threshold_only"), (50, "exact_counts")):
        checks.append(validate_stream(stage_dir / f"k{k}_{mode}_dfall", stage_dir / "reference"))
    n, sample = job["preflight"]["documents"], largest["documents"]
    denominator = sample * (sample - 1)
    forecasts = {}
    for row in stage["runs"]:
        if row["cap"] is None and row["mode"] == "exact_counts":
            forecasts[str(row["k"])] = (
                row["accepted_pairs"] * n * (n - 1) + denominator - 1
            ) // denominator
    disk = 12 * (forecasts["5"] + sum(forecasts.values())) + job["preflight"]["helper_array_bytes"]
    floor_run = next(row for row in stage["runs"] if row["k"] == 5 and row["cap"] is None and row["mode"] == "exact_counts")
    return {
        "status": "passed", "reference_stage": str(stage_dir),
        "subset_documents": sample, "rechecked_results": checks,
        "projected_pairs_per_k": forecasts, "projected_result_bytes": disk,
        "projection_caveat": "uniform pair-space estimate, not a runtime or size guarantee",
        "rough_join_seconds_pair_space": floor_run["elapsed_seconds"] * n * (n - 1) / denominator,
        "memory_budget_bytes": budget,
    }


def run_full_queue(run_dir: str | Path) -> dict:
    root = Path(run_dir).resolve()
    config = json.loads((root / "run.json").read_text())
    with FileLock(str(root / "queue.lock"), timeout=0):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(root / "queue.log")],
        )
        (root / "logs").mkdir(exist_ok=True)
        status = {
            "status": "waiting_for_all_pilots", "started_at": timestamp(),
            "heartbeat_at": timestamp(), "launcher_pid": os.getpid(), "child_pid": None,
            "memory_budget_bytes": config["memory_budget_bytes"], "jobs": [],
        }
        for job in config["jobs"]:
            bounds = full_working_bounds(job["index_dir"], job["overlap_dir"])
            held = bounds["required_working_bytes"] > config["memory_budget_bytes"]
            status["jobs"].append({
                **job, "preflight": bounds,
                "status": "held_memory_allowance" if held else "waiting_pilots",
                "hold_reason": (
                    f"Current full-view/join bound {bounds['required_working_bytes']/2**30:.1f} GiB "
                    f"exceeds {config['memory_budget_bytes']/2**30:.0f} GiB; needs out-of-core work or explicit allowance change"
                ) if held else None,
                "returncodes": {},
            })
        atomic_json(root / "status.json", status)
        finished = False
        try:
            for relative, expected in config["source_snapshot_sha256"].items():
                actual = hashlib.sha256((root / "source_snapshot" / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Frozen full-run source changed: {relative}")
            logging.info("Waiting for all pilot corpora to finish; full runs are sequential and gated")
            terminal = {"passed", "completed_with_failures", "completed_with_workload_limits", "launcher_failed_or_interrupted"}
            while True:
                pilots = json.loads(Path(config["pilot_status"]).read_text())
                status["pilot_queue_status"] = pilots["status"]
                if pilots["status"] in terminal:
                    break
                status["heartbeat_at"] = timestamp()
                atomic_json(root / "status.json", status)
                time.sleep(30)
            by_name = {job["name"]: job for job in pilots["jobs"]}
            status["status"] = "running"
            atomic_json(root / "status.json", status)
            for job in status["jobs"]:
                if job["status"].startswith("held_"):
                    logging.warning("%s: %s", job["name"], job["hold_reason"])
                    continue
                review = review_pilot(job, by_name[job["name"]], config["memory_budget_bytes"])
                job["pilot_review"] = review
                if review["status"] != "passed":
                    job["status"] = review["status"]
                    atomic_json(root / "status.json", status)
                    continue
                if (
                    _available_memory() < config["minimum_available_memory_bytes"]
                    or shutil.disk_usage(root).free < 2 * review["projected_result_bytes"] + config["minimum_free_disk_bytes"]
                ):
                    job["status"] = "held_resources"
                    job["hold_reason"] = "Insufficient available RAM or projected disk headroom"
                    atomic_json(root / "status.json", status)
                    continue
                output = root / "results" / job["name"]
                environment = os.environ.copy()
                environment.update(config["environment"])
                environment["PYTHONPATH"] = str(root / "source_snapshot")
                base = [
                    config["python"], "-u", "-m", "experiments.overlap.run_full_overlap_join",
                    job["index_dir"], job["overlap_dir"], review["reference_stage"], str(output),
                    "--memory-gib", str(config["memory_budget_bytes"] / 2**30),
                ]
                job["status"] = "running"
                job["started_at"] = timestamp()
                job["output_dir"] = str(output)
                for phase, command in (("full_pipeline", base), ("independent_verification", base + ["--finish-only"])):
                    job["phase"] = phase
                    log_path = root / "logs" / f"{job['name']}.{phase}.log"
                    job["log"] = str(log_path)
                    atomic_json(root / "status.json", status)
                    logging.info("Starting %s %s; may extend beyond one night; log=%s", job["name"], phase, log_path)
                    code = run_logged_command(root, command, log_path, environment, status)
                    job["returncodes"][phase] = code
                    if code:
                        job["status"] = "failed"
                        logging.error("%s %s failed: exit %d", job["name"], phase, code)
                        break
                if job["status"] != "failed":
                    result = json.loads((output / "report.json").read_text())
                    if result["status"] != "passed" or result.get("additional_independent_checks", {}).get("status") != "passed":
                        raise RuntimeError("Full run exited without independently verified results")
                    job["status"] = "passed"
                    job["counts_per_k"] = result["verification"]["counts_per_k"]
                job["finished_at"] = timestamp()
                atomic_json(root / "status.json", status)
            statuses = [job["status"] for job in status["jobs"]]
            status["status"] = (
                "completed_with_failures" if "failed" in statuses
                else "completed_with_holds" if any(value.startswith("held_") for value in statuses)
                else "passed"
            )
            status["finished_at"] = timestamp()
            atomic_json(root / "status.json", status)
            finished = True
            return status
        finally:
            if not finished:
                status["status"] = "launcher_failed_or_interrupted"
                status["finished_at"] = timestamp()
                atomic_json(root / "status.json", status)
                logging.error("Full-run queue failed/interrupted; inspect status and logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    status = run_full_queue(args.run_dir)
    if status["status"] in {"completed_with_failures", "launcher_failed_or_interrupted"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
