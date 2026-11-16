import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Original-main comparison uses Linux fork")
@pytest.mark.parametrize("cli", [False, True])
def test_build_benchmark_modes_build_and_verify_outputs(tmp_path, cli):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({
        "id": "source", "text": " ".join(f"benchmarkword{i}" for i in range(120)),
    }) + "\n")
    output = tmp_path / "result"
    completed = subprocess.run(
        [
            sys.executable, str(root / "experiments/benchmark_index_build.py"),
            "--checkout", str(root), "--corpus", str(source), "--output", str(output),
            "--workers", "1", "--start-method", "fork",
        ] + (["--cli"] if cli else []),
        cwd=root, text=True, capture_output=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["documents"] == 1 and result["occurrences"] > 0
    assert result["all_source_occurrences_verified"]
    assert result["timing_mode"] == ("startup_inclusive_cli" if cli else "warmed_api")
    if cli:
        assert result["jit_warmup_seconds_excluded"] == 0
    assert result["total_seconds"] == pytest.approx(result["extraction_seconds"] + result["merge_seconds"])
    assert (output / "index/postings.dat").stat().st_size == result["occurrences"] * 8
