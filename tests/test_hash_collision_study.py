import hashlib
import json

import numpy as np
import polars as pl
import pytest
import subprocess
import sys

pytest.importorskip("plotly")

from experiments import hash_collision_study as study


def test_wide_hash_uses_unambiguous_exact_token_encoding():
    tokens = ["blå", "a\0b", "c"]
    frames = study.frame_tokens(tokens)
    grams, hashes = study.wide_fingerprints(frames, 2)
    assert study.decode_kgram(grams[0]) == tokens[:2]
    assert study.decode_kgram(grams[1]) == tokens[1:]
    expected = int.from_bytes(hashlib.blake2b(
        grams[0], digest_size=8, person=study.PERSONALIZATION,
    ).digest(), "little")
    assert int(hashes[0]) == expected
    assert hashes.dtype == np.uint64
    left, _ = study.wide_fingerprints(study.frame_tokens(["ab", "c"]), 2)
    right, _ = study.wide_fingerprints(study.frame_tokens(["a", "bc"]), 2)
    assert left != right


@pytest.mark.parametrize("window", [1, 2, 6, 101])
def test_unsigned_winnowing_matches_independent_reference(window):
    rng = np.random.default_rng(7)
    values = rng.integers(0, np.iinfo(np.uint64).max, 100, dtype=np.uint64)
    values[3:7] = np.uint64(2**63 + 1)
    expected_positions = []
    for start in range(max(0, len(values) - window + 1)):
        position = min(range(start, start + window), key=lambda p: (int(values[p]), -p))
        if not expected_positions or expected_positions[-1] != position:
            expected_positions.append(position)
    hashes, positions = study.winnow_uint64(values, window)
    assert list(positions) == expected_positions
    assert np.array_equal(hashes, values[positions])
    assert hashes.dtype == np.uint64
    if window == 1:
        assert any(int(value) > 2**63 for value in hashes)


def test_exact_kgram_multiplicity_does_not_trust_64bit_signature(tmp_path):
    raw = tmp_path / "occurrences"
    raw.mkdir()
    a = b"".join(study.frame_tokens(["a", "b"]))
    b = b"".join(study.frame_tokens(["c", "d"]))
    c = b"".join(study.frame_tokens(["same", "phrase"]))
    rows = [
        (7, 100, a, 0, 0), (7, 100, a, 0, 1), (7, 100, b, 1, 0),
        (8, 200, c, 0, 2), (8, 200, c, 1, 2),
    ]
    study._occurrence_frame(rows).write_parquet(raw / "old-0001.parquet")
    summary, exact, examples = study.summarize_pipeline(raw, "old", tmp_path)
    assert summary["n_unique_hashes"] == 2
    assert summary["n_exact_distinct_kgrams"] == 3
    assert summary["n_collided_hashes"] == 1
    assert summary["n_distinct_kgrams_merged_by_hashing"] == 1
    assert summary["n_unique_independent64_signatures"] == 2
    assert summary["n_kgrams_merged_by_independent64_signatures"] == 1
    assert summary["n_selected_occurrences"] == 5
    assert summary["n_occurrences_in_collided_hashes"] == 3
    assert summary["n_shared_hashes_without_any_actually_shared_kgram"] == 1
    assert summary["singleton_hash_fraction"] == 0
    assert exact["singleton_fraction"] == pytest.approx(2 / 3)
    assert examples[0]["hash"] == 7
    assert examples[0]["n_distinct_kgrams"] == 2
    assert examples[0]["n_distinct_signatures64"] == 1
    per_hash = pl.read_csv(tmp_path / "old_per_hash.csv")
    assert per_hash["n_distinct_kgrams"].to_list() == [2, 1]
    assert summary["collision_multiplicity"] == [
        {"n_distinct_kgrams": 1, "n_hashes": 1, "n_occurrences": 2},
        {"n_distinct_kgrams": 2, "n_hashes": 1, "n_occurrences": 3},
    ]
    partitioned, partitioned_exact, _ = study.summarize_partitioned(
        raw, "old", tmp_path, partitions=4,
    )
    assert partitioned == summary
    assert partitioned_exact == exact
    rows.append((9, 100, b"".join(study.frame_tokens(["another", "gram"])), 2, 0))
    study._occurrence_frame(rows).write_parquet(raw / "old-0001.parquet")
    cross_output = tmp_path / "cross_partition_collision"
    cross_output.mkdir()
    cross_summary, _, _ = study.summarize_partitioned(
        raw, "old", cross_output, partitions=4,
    )
    assert cross_summary["n_exact_distinct_kgrams"] == 4
    assert cross_summary["n_unique_independent64_signatures"] == 2
    assert cross_summary["n_kgrams_merged_by_independent64_signatures"] == 2


def test_uniform_sampling_is_seeded_and_scans_all_shards(monkeypatch):
    shards = [study.Path("small"), study.Path("large")]

    def records(path, **kwargs):
        size = 3 if path == "small" else 50
        return iter({"id": f"{path}-{i}", "text": f"text {i}"} for i in range(size))

    monkeypatch.setattr(study.utils, "stream_json_zst", records)
    first, populations = study.sample_documents(shards, 20, 123, 0, 1000)
    second, _ = study.sample_documents(shards, 20, 123, 0, 1000)
    assert first == second
    assert len(first) == 20
    assert sum(row["eligible_documents"] for row in populations) == 53
    assert sum(row["sampled_documents"] for row in populations) == 20
    with pytest.raises(ValueError, match="fewer than"):
        study.sample_documents(shards, 54, 123, 0, 1000)


def test_legacy_collection_is_retired_and_frozen_reports_still_render(tmp_path):
    with pytest.raises(ValueError, match="retired"):
        study.run_study(None)
    raw = tmp_path / "raw"
    raw.mkdir()
    gram = b"".join(study.frame_tokens(["one", "two"]))
    study._occurrence_frame([(7, 10, gram, 0, 0)]).write_parquet(raw / "old-one.parquet")
    summary, exact, examples = study.summarize_pipeline(raw, "old", tmp_path)
    report = {
        "scope": "sample_only_not_a_full_corpus_census",
        "settings": {"wide_hash_method": "rolling64-v1"},
        "counters": {"documents": 1, "validated_forward_documents": 1},
        "fixed_selection": {"old_hashes": summary, "exact_kgrams": exact},
        "full_64bit": {"hashes": summary},
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "collision_examples.json").write_text(json.dumps({"old": examples, "wide": []}))
    subprocess.run(
        [sys.executable, "-m", "experiments.hash_collision_study", "--report-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "collision_study.html").is_file()
    assert json.loads((tmp_path / "report.json").read_text()) == report

