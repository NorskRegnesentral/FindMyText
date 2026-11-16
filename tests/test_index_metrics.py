import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from experiments.benchmark_index_metrics import select_queries


def test_query_selection_reports_resource_exclusions():
    class Detector:
        def get_fingerprints(self, text):
            return pl.DataFrame({"hash": [1] * 100})

        def _get_candidates(self, fingerprints):
            return pl.DataFrame({"doc_match_id": ["candidate"] * 51, "hash": [1] * 51})

    with pytest.raises(ValueError, match="over_5000_alignment_points"):
        select_queries(
            Detector(), [{"id": "source", "text": "a" * 1000}],
            count=1, characters=1000, seed=41,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Build measurement uses the fork-based CLI harness")
def test_metrics_measure_same_fresh_index_and_complete_detector_call(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "input.jsonl"
    text = " ".join(f"metricword{i}" for i in range(220))
    source.write_text("".join(json.dumps({"id": str(i), "text": text}) + "\n" for i in range(2)))
    output = tmp_path / "metrics"
    result = subprocess.run(
        [
            sys.executable, str(root / "experiments/benchmark_index_metrics.py"),
            str(source), "--output", str(output), "--documents", "2",
            "--queries", "1", "--query-characters", "1000", "--repeats", "1",
        ],
        cwd=root, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["documents"] == 2
    assert report["index_bytes"] == sum(path.stat().st_size for path in (output / "build-1/index").iterdir())
    assert report["build_runs"][0]["all_source_occurrences_verified"]
    assert report["query_workload"]["source_document_found_in_results"] == 1
    assert report["query_workload"]["skipped"] == {}
    assert len(report["query_latency_seconds"]["per_pass"]) == 1
    assert len(report["query_latency_seconds"]["per_pass"][0]) == 1
    assert report["query_latency_seconds"]["all_timed_score_mappings_identical"]
