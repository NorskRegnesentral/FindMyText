import json
import subprocess
import sys
from pathlib import Path

import pytest
import zstandard as zstd

pytest.importorskip("plotly")

from experiments.profile_index_build import analyze_profiles, execute_build, write_report


@pytest.mark.parametrize("diagnostic", [False, True])
def test_fresh_process_build_profiles_and_validates_all_stages(tmp_path, diagnostic):
    method = "rolling64-v1"
    source = tmp_path / "source.jsonl.zst"
    records = [
        {"id": str(i), "text": " ".join(f"token_{i}_{j}" for j in range(60))}
        for i in range(20)
    ]
    source.write_bytes(zstd.ZstdCompressor().compress(
        "\n".join(json.dumps(row) for row in records).encode()
    ))
    directory = tmp_path / "build"
    config = {
        "source": str(source), "directory": str(directory), "hash_method": method,
        "diagnostic": diagnostic, "documents": len(records), "nb_workers": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    subprocess.run(
        [sys.executable, "-m", "experiments.profile_index_build", "--run-config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    result = json.loads((directory / "build_result.json").read_text())
    assert result["counts"]["documents"] == 20
    assert result["validated_forward_documents"] == 20
    assert result["counts"]["postings"] > 0
    assert all(seconds > 0 for seconds in result["phases"].values())
    assert sum(result["extraction_main_process_breakdown"].values()) == pytest.approx(
        result["phases"]["extraction"], abs=0.01
    )
    assert result["total_build_seconds"] >= sum(result["phases"].values())
    if diagnostic:
        profile = analyze_profiles(directory)
        assert profile["summed_profiled_process_cpu_seconds"] > 0
        assert profile["functions"]
        assert "hash-worker" in profile["process_roles_cpu_seconds"]
        assert "merge-reader" in profile["process_roles_cpu_seconds"]
    report = {"documents": 20, "timed_builds": {"rolling64-v1": result}}
    write_report(report, tmp_path)
    assert (tmp_path / "build_profile.html").is_file()
    assert len((tmp_path / "wall_time_phases.csv").read_text().splitlines()) == 4


def test_legacy_build_profiling_is_retired_before_output_creation(tmp_path):
    with pytest.raises(ValueError, match="retired"):
        execute_build({"hash_method": "legacy31", "directory": str(tmp_path / "not-created")})
    assert not (tmp_path / "not-created").exists()
