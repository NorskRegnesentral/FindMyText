"""Sequential, safeguarded cross-corpus prefix experiments using existing indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from findmytext.overlap_stream import atomic_json


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_larger_stage(screen_size: int, reference: dict, policy: dict) -> dict:
    """Limit estimated reference work, not runtime or the meaning of exact results."""
    denominator = screen_size * (screen_size - 1)
    if denominator <= 0:
        raise ValueError("Screening subset must have at least two documents")
    forecasts = []
    for size in policy["larger_sizes_descending"]:
        numerator = size * (size - 1)
        updates = (reference["pair_contributions"] * numerator + denominator - 1) // denominator
        pairs = (reference["positive_pairs"] * numerator + denominator - 1) // denominator
        forecast = {
            "documents": size, "projected_pair_contributions": updates,
            "projected_positive_pairs": pairs, "projection": "pair-space scaling; estimate, not a guarantee",
        }
        forecasts.append(forecast)
        if (
            updates <= policy["max_projected_pair_contributions"]
            and pairs <= policy["max_projected_positive_pairs"]
        ):
            return {"documents": size, "forecasts": forecasts, "reason": "within reference-work safeguards"}
    return {
        "documents": None, "forecasts": forecasts,
        "reason": "larger reference work exceeds safeguards; screening results retained",
    }


def _available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available memory")


def run_logged_command(root: Path, command: list[str], log_path: Path, environment: dict, status: dict) -> int:
    with log_path.open("a", buffering=1) as stream:
        stream.write(f"\n=== {timestamp()} {json.dumps(command)} ===\n")
        with subprocess.Popen(
            command, cwd=root / "source_snapshot", env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        ) as process:
            status["child_pid"] = process.pid
            status["heartbeat_at"] = timestamp()
            atomic_json(root / "status.json", status)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    status["heartbeat_at"] = timestamp()
                    atomic_json(root / "status.json", status)
    status["child_pid"] = None
    status["heartbeat_at"] = timestamp()
    atomic_json(root / "status.json", status)
    return code


def run_stage(root: Path, config: dict, status: dict, job: dict, size: int) -> bool:
    stage_name = f"{job['name']}-{size}"
    output = root / "results" / job["name"] / str(size)
    log_path = root / "logs" / f"{stage_name}.log"
    if _available_memory() < config["minimum_available_memory_bytes"]:
        job["error"] = "Insufficient available RAM; stage not launched"
        logging.error("%s: %s", stage_name, job["error"])
        return False
    if shutil.disk_usage(root).free < config["minimum_free_disk_bytes"]:
        job["error"] = "Insufficient free disk space; stage not launched"
        logging.error("%s: %s", stage_name, job["error"])
        return False
    command = [
        config["python"], "-u", "-m", "experiments.overlap.scale_overlap_join",
        job["index_dir"], job["overlap_dir"], str(output),
        "--sizes", str(size), "--seed", str(config["seed"]),
        "--k", *map(str, config["k_values"]),
        "--candidate-df-cap", str(config["candidate_df_cap"]),
        "--memory-gib", str(config["memory_budget_bytes"] / 2**30),
        "--prefix-filter", "--no-controls", "--no-fingerprint-control",
    ]
    job["phase"] = stage_name
    job["phase_started_at"] = timestamp()
    job["current_output"] = str(output)
    job["current_log"] = str(log_path)
    environment = os.environ.copy()
    environment.update(config["environment"])
    environment["PYTHONPATH"] = str(root / "source_snapshot")
    logging.info("Starting %s; log=%s", stage_name, log_path)
    code = run_logged_command(root, command, log_path, environment, status)
    job["returncodes"][str(size)] = code
    status["heartbeat_at"] = timestamp()
    atomic_json(root / "status.json", status)
    if code:
        job["error"] = f"{stage_name} failed with exit code {code}; inspect {log_path}"
        logging.error(job["error"])
        return False
    report_path = output / "report.json"
    if not report_path.is_file():
        job["error"] = f"{stage_name} exited without a report"
        logging.error(job["error"])
        return False
    report = json.loads(report_path.read_text())
    entry = report["sizes"][str(size)]
    if (
        report["status"] != "passed" or entry["status"] != "passed"
        or len(entry["runs"]) != len(config["k_values"]) * 4
        or not all(row["validated"] and row["prefix_filter"] for row in entry["runs"])
    ):
        job["error"] = f"{stage_name} exited without the required complete prefix validation"
        logging.error(job["error"])
        return False
    job["completed_stages"].append({
        "documents": size, "report": str(output / "report.json"),
        "configurations": len(entry["runs"]),
        "reference": entry["reference"],
        "peak_process_rss_bytes": entry["process_lifetime_peak_rss_bytes"],
    })
    atomic_json(root / "status.json", status)
    logging.info("%s passed %d configurations", stage_name, len(entry["runs"]))
    return True


def run_queue(run_dir: str | Path) -> dict:
    root = Path(run_dir).resolve()
    config = json.loads((root / "run.json").read_text())
    with FileLock(str(root / "queue.lock"), timeout=0):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(root / "queue.log")],
        )
        (root / "logs").mkdir(exist_ok=True)
        status = {
            "run_id": config["run_id"], "status": "waiting_for_nynorsk_verification",
            "started_at": timestamp(), "heartbeat_at": timestamp(),
            "launcher_pid": os.getpid(), "child_pid": None,
            "memory_budget_bytes": config["memory_budget_bytes"],
            "parallel_workers": 1,
            "jobs": [
                {
                    **job, "status": "pending", "completed_stages": [],
                    "returncodes": {},
                }
                for job in config["jobs"]
            ],
        }
        atomic_json(root / "status.json", status)
        finished = False
        try:
            for relative, expected in config["source_snapshot_sha256"].items():
                actual = hashlib.sha256((root / "source_snapshot" / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Frozen overnight source changed: {relative}")
            logging.info("Waiting for Nynorsk's final independent verification before starting the sequential queue")
            while True:
                nynorsk = json.loads(Path(config["nynorsk_report"]).read_text())
                status["nynorsk_phase"] = nynorsk.get("phase")
                if nynorsk.get("additional_independent_checks", {}).get("status") == "passed":
                    status["nynorsk_dependency"] = "passed"
                    break
                if "failed" in nynorsk["status"]:
                    status["nynorsk_dependency"] = "failed; independent corpus experiments still proceed"
                    logging.error("Nynorsk reported failure; retaining its reports and proceeding only with independent bounded corpora")
                    break
                status["heartbeat_at"] = timestamp()
                atomic_json(root / "status.json", status)
                time.sleep(30)
            status["status"] = "running"
            atomic_json(root / "status.json", status)
            for job in status["jobs"]:
                job["status"] = "running"
                job["started_at"] = timestamp()
                atomic_json(root / "status.json", status)
                screen = config["screen_documents"]
                if not run_stage(root, config, status, job, screen):
                    job["status"] = "failed"
                    job["finished_at"] = timestamp()
                    atomic_json(root / "status.json", status)
                    continue
                decision = choose_larger_stage(screen, job["completed_stages"][-1]["reference"], config["policy"])
                job["larger_stage_decision"] = decision
                atomic_json(root / "status.json", status)
                if decision["documents"] is None:
                    job["status"] = "completed_screen_only_workload_limit"
                    logging.warning("%s: %s", job["name"], decision["reason"])
                elif run_stage(root, config, status, job, decision["documents"]):
                    job["status"] = "passed"
                else:
                    job["status"] = "failed"
                job["finished_at"] = timestamp()
                atomic_json(root / "status.json", status)
            failures = [job["name"] for job in status["jobs"] if job["status"] == "failed"]
            limited = [job["name"] for job in status["jobs"] if job["status"].startswith("completed_screen_only")]
            status["status"] = (
                "completed_with_failures" if failures or status["nynorsk_dependency"].startswith("failed")
                else "completed_with_workload_limits" if limited
                else "passed"
            )
            status["finished_at"] = timestamp()
            status["heartbeat_at"] = timestamp()
            atomic_json(root / "status.json", status)
            logging.info("Queue finished: %s; failures=%s; workload-limited=%s", status["status"], failures, limited)
            finished = True
            return status
        finally:
            if not finished:
                status["status"] = "launcher_failed_or_interrupted"
                status["finished_at"] = timestamp()
                atomic_json(root / "status.json", status)
                logging.error("Queue launcher failed/interrupted; inspect status.json and stage logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    status = run_queue(args.run_dir)
    if status["status"] == "completed_with_failures":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
