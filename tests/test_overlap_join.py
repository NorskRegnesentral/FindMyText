import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.benchmark_overlap_join import benchmark
from findmytext import index_builder, similarity, winnower
from findmytext.indexing import POSTING_DTYPE
from findmytext.overlap_inverted import build_overlap_inverted_index
from findmytext.overlap_join import (
    JoinSubset,
    PAIR_DTYPE,
    _next_epoch,
    _pair_slot,
    prepare_subset,
    reference_counts,
    run_join,
    verify_pair,
)


HIGH = 2**64 - 1
SETS = [
    {1, 3, 7, 9, HIGH},
    {1, 3, 7, 9, 31, HIGH},
    {3, 11, 13, HIGH},
    {3, 11, 17, HIGH},
    set(),
    {3, 23},
]


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "index"
    root.mkdir()
    meta = winnower.Winnower(length=4, window_size=6).to_metadata()
    meta["postings_layout"] = "interleaved"
    (root / "meta.json").write_text(json.dumps(meta))
    values = sorted(set().union(*SETS))
    lengths = []
    records = []
    for fp in values:
        postings = [(doc, fp % 100) for doc in reversed(range(len(SETS))) if fp in SETS[doc]]
        if fp == 3:
            postings.append((0, 200))
        lengths.append(len(postings))
        records.extend(postings)
    lengths = np.array(lengths, dtype=np.uint32)
    np.save(root / "fingerprints.npy", np.array(values, dtype=np.uint64))
    np.save(root / "lengths.npy", lengths)
    np.save(root / "offsets.npy", np.r_[0, np.cumsum(lengths[:-1], dtype=np.uint64) * 8].astype(np.uint64))
    np.array(records, dtype=POSTING_DTYPE).tofile(root / "postings.dat")
    np.save(root / "doc_name_offsets.npy", np.arange(len(SETS) + 1, dtype=np.uint64))
    np.save(root / "doc_name_bytes.npy", np.frombuffer(b"abcdef", dtype=np.uint8))
    index_builder.build_forward_index(str(root), batch_entries=3)
    overlap = tmp_path / "overlap_inverted"
    build_overlap_inverted_index(root, overlap, batch_entries=3, verbose=False)
    return root, overlap


def expected(ids, k):
    return {
        (int(a), int(b)): len(SETS[a] & SETS[b])
        for a, b in itertools.combinations(sorted(ids), 2)
        if len(SETS[a] & SETS[b]) >= k
    }


def as_dict(rows):
    return {(int(row["doc_i"]), int(row["doc_j"])): int(row["count"]) for row in rows}


@pytest.mark.parametrize("a,b", [
    ([], []), ([], [1]), ([1], [1]), ([1], [2]),
    ([1, 3, 9], [2, 3, 9]), ([3, 9, 15], [1, 2, 9, 18]),
    ([1, 2, 3, 4, 5], [5]), ([1, HIGH], [HIGH]),
    ([1, 3, 5], [6, 7, 8]), ([5], [1, 2, 3, 4, 5]),
])
@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_merge_verifier_boundaries(a, b, k):
    first = np.array(a, dtype=np.uint64)
    second = np.array(b, dtype=np.uint64)
    count = len(set(a) & set(b))
    assert verify_pair(first, second, k) == (count >= k, count if count >= k else None)
    assert verify_pair(first, second, k, "threshold_only") == (count >= k, None)


def test_merge_verifier_rejects_bad_inputs():
    good = np.array([1, 3], dtype=np.uint64)
    for bad in (np.array([1, 1], dtype=np.uint64), np.array([3, 1], dtype=np.uint64),
                np.array([1, 3], dtype=np.int64)):
        with pytest.raises(ValueError, match="strictly sorted"):
            verify_pair(good, bad, 1)
    with pytest.raises(ValueError):
        verify_pair(good, good, 0)
    with pytest.raises(ValueError):
        verify_pair(good, good, 1, "unknown")


@pytest.mark.parametrize("batch_size", [1, 3, 1000])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 6])
def test_both_schedulers_and_modes_match_reference(source, batch_size, k):
    root, overlap = source
    ids = np.array([5, 2, 0, 4, 1, 3], dtype=np.uint32)
    data = prepare_subset(root, overlap, ids, 1 << 28)
    reference, _ = reference_counts(data)
    assert as_dict(reference) == expected(ids, 1)
    target = expected(ids, k)
    decisions = len(expected(ids, 1))
    for scheduler in ("fingerprint", "document"):
        rows, metrics = run_join(data, k, scheduler=scheduler, batch_size=batch_size)
        assert as_dict(rows) == target
        assert metrics["completed"] and metrics["exhaustive_within_subset"]
        assert metrics["verification_calls"] + metrics["length_rejections"] == decisions
        assert metrics["duplicate_discoveries"] > 0
        pairs, threshold_metrics = run_join(
            data, k, scheduler=scheduler, output_mode="threshold_only", batch_size=batch_size
        )
        assert pairs.dtype == PAIR_DTYPE
        assert {(int(row["doc_i"]), int(row["doc_j"])) for row in pairs} == set(target)
        assert threshold_metrics["hash_comparisons"] <= metrics["hash_comparisons"]
    if batch_size == 1:
        assert metrics["batches"] > 1


def test_reference_matches_existing_inverted_implementation(source):
    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS), dtype=np.uint32), 1 << 28)
    rows, _ = reference_counts(data)
    similarity.compute_similarities_inverted(str(root), verbose=False, posting_batch_entries=3)
    original = np.load(root / "similarities_inverted.npy")
    np.testing.assert_array_equal(rows, original)


def test_frequency_cap_changes_discovery_only_and_uses_global_df(source):
    root, overlap = source
    ids = np.array([0, 1, 2, 3], dtype=np.uint32)
    data = prepare_subset(root, overlap, ids, 1 << 28)
    counts = []
    for scheduler in ("fingerprint", "document"):
        rows, metrics = run_join(data, 2, scheduler=scheduler, candidate_df_cap=2, batch_size=1)
        counts.append(as_dict(rows))
        assert not metrics["exhaustive_within_subset"]
        assert metrics["fingerprint_records_skipped"] == 2
        assert as_dict(rows) == {(0, 1): 5, (2, 3): 3}
    assert counts[0] == counts[1]
    # Both omitted fingerprints are common globally, even though the subset
    # has only four documents. Their contributions still appear in the counts.
    assert expected(ids, 2)[(0, 2)] == 2
    assert (0, 2) not in counts[0]


def test_exact_count_cache_can_be_filtered_after_lowering_k(source):
    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS), dtype=np.uint32), 1 << 28)
    cache, _ = run_join(data, 1, scheduler="fingerprint", batch_size=1)
    for k in (50, 30, 20, 10, 5, 3, 2, 1):
        assert as_dict(cache[cache["count"] >= k]) == expected(range(len(SETS)), k)


def test_pair_addressing_and_epoch_rollover():
    for n in (2, 5, 20):
        slots = [_pair_slot(a, b, n) for a, b in itertools.combinations(range(n), 2)]
        assert slots == list(range(n * (n - 1) // 2))
        for a, b in itertools.combinations(range(n), 2):
            assert _pair_slot(a, b, n) == _pair_slot(b, a, n)
    markers = np.array([4, 2**32 - 1, 1], dtype=np.uint32)
    assert _next_epoch(markers, np.uint32(3)) == 4
    np.testing.assert_array_equal(markers, [4, 2**32 - 1, 1])
    assert _next_epoch(markers, np.uint32(2**32 - 1)) == 1
    np.testing.assert_array_equal(markers, [0, 0, 0])


def test_subset_and_resource_validation(source):
    root, overlap = source
    for ids in (np.array([0, 0]), np.array([-1, 2]), np.array([0, 6]),
                np.array([], dtype=np.int64), np.array([0.5, 1.5])):
        with pytest.raises(ValueError):
            prepare_subset(root, overlap, ids, 1 << 28)
    with pytest.raises(MemoryError, match="budget"):
        prepare_subset(root, overlap, np.arange(len(SETS)), 1)
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    with pytest.raises(ValueError):
        run_join(data, 1, scheduler="unknown")
    with pytest.raises(ValueError):
        run_join(data, 1, scheduler="document", output_mode="bad")
    with pytest.raises(ValueError):
        run_join(data, 1, scheduler="fingerprint", candidate_df_cap=1)
    with pytest.raises(ValueError):
        run_join(data, 1, scheduler="document", batch_size=0)
    data.memory_budget_bytes = 1
    with pytest.raises(MemoryError, match="budget"):
        run_join(data, 1, scheduler="fingerprint")


def test_seeded_benchmark_reports_exact_and_capped_results(source, tmp_path):
    root, overlap = source
    report = benchmark(
        root, overlap, tmp_path / "benchmark", n_documents=6, seed=41,
        k_values=(5, 3, 2, 1), candidate_df_cap=2,
        memory_budget_bytes=1 << 28, batch_size=1,
    )
    assert report["status"] == "passed"
    assert report["source_unchanged"]
    assert report["deadline"] is None
    assert len(report["runs"]) == 32
    assert all(len(row["elapsed_seconds_samples"]) == 3 for row in report["runs"])
    assert all(row["validated"] for row in report["runs"])
    assert all(row["missing_reference_pairs"] == 0 for row in report["runs"] if row["candidate_df_cap"] is None)
    assert any(row["missing_reference_pairs"] > 0 for row in report["runs"] if row["candidate_df_cap"] == 2)
    persisted = json.loads((tmp_path / "benchmark/report.json").read_text())
    assert persisted == report
    assert len(report["reusable_exact_count_sweep"]["thresholds"]) == 4
    assert np.load(tmp_path / "benchmark/document_k2_threshold_only_dfall.npy").dtype == PAIR_DTYPE
    with pytest.raises(FileExistsError):
        benchmark(root, overlap, tmp_path / "benchmark", n_documents=6)


@pytest.mark.parametrize("seed", range(8))
def test_random_sets_against_direct_set_intersections(seed):
    rng = np.random.default_rng(seed)
    universe = np.array([1, 3, 5, 8, 11, 19, 2**63 + 5, HIGH], dtype=np.uint64)
    sets = [set(rng.choice(universe, size=int(rng.integers(9)), replace=False).tolist()) for _ in range(15)]
    values = np.concatenate([np.array(sorted(values), dtype=np.uint64) for values in sets])
    forward_offsets = np.r_[0, np.cumsum([len(values) for values in sets])].astype(np.uint64)
    records = sorted(
        (sum(fp in values for values in sets), fp)
        for fp in set().union(*sets) if sum(fp in values for values in sets) >= 2
    )
    fps = np.array([fp for _, fp in records], dtype=np.uint64)
    post_offsets = np.r_[0, np.cumsum([df for df, _ in records])].astype(np.uint64)
    docs = np.array([doc for _, fp in records for doc, values in enumerate(sets) if fp in values], dtype=np.uint32)
    eligible = np.bincount(docs, minlength=len(sets)).astype(np.uint64)
    doc_ranks = [[rank for rank, (_, fp) in enumerate(records) if fp in values] for values in sets]
    data = JoinSubset(
        np.arange(len(sets), dtype=np.uint32) * 7 + 2, forward_offsets, values,
        fps, post_offsets, docs, np.array([df for df, _ in records], dtype=np.uint64),
        eligible, np.r_[0, np.cumsum(eligible)].astype(np.uint64),
        np.array([rank for ranks in doc_ranks for rank in ranks], dtype=np.uint32),
        1 << 28,
    )
    target = {
        (int(data.document_ids[a]), int(data.document_ids[b])): len(sets[a] & sets[b])
        for a, b in itertools.combinations(range(len(sets)), 2) if sets[a] & sets[b]
    }
    reference, _ = reference_counts(data)
    assert as_dict(reference) == target
    for k in (1, 3, 5, 9):
        for scheduler in ("fingerprint", "document"):
            rows, _ = run_join(data, k, scheduler=scheduler, batch_size=3)
            assert as_dict(rows) == {pair: count for pair, count in target.items() if count >= k}


def stream_rows(directory):
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    chunks = [np.load(directory / part["file"]) for part in checkpoint["parts"]]
    if chunks:
        return np.concatenate(chunks)
    from findmytext.similarity import SIMILARITY_DTYPE
    return np.empty(0, dtype=SIMILARITY_DTYPE)


@pytest.mark.parametrize("mode", ["exact_counts", "threshold_only"])
@pytest.mark.parametrize("cap", [None, 2])
def test_streamed_join_matches_external_reference_and_early_pruning(source, tmp_path, mode, cap):
    from findmytext.overlap_stream import external_reference, run_document_stream, validate_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    reference_dir = tmp_path / "external"
    report = external_reference(data, reference_dir, batch_entries=2)
    reference, dense_report = reference_counts(data)
    np.testing.assert_array_equal(np.load(reference_dir / "counts.npy"), reference)
    assert report["pair_contributions"] == dense_report["pair_contributions"]
    runs = []
    for early in (False, True):
        output = tmp_path / f"stream-{early}"
        checkpoint = run_document_stream(
            data, output, 3, source_index_dir=root, output_mode=mode,
            candidate_df_cap=cap, batch_size=1, probes_per_commit=2,
            early_pruning=early, verbose=False,
        )
        validation = validate_stream(output, reference_dir, batch_entries=2)
        assert validation["validated"]
        assert checkpoint["status"] == "complete"
        assert checkpoint["exhaustive_within_subset"] == (cap is None)
        rows = stream_rows(output)
        keys = {(int(row["doc_i"]), int(row["doc_j"])) for row in rows}
        runs.append((keys, checkpoint))
        if mode == "threshold_only":
            assert rows.dtype == PAIR_DTYPE
        else:
            assert all(int(row["count"]) == expected(range(len(SETS)), 1)[(int(row["doc_i"]), int(row["doc_j"]))] for row in rows)
        if cap is None:
            assert keys == set(expected(range(len(SETS)), 3))
    assert runs[0][0] == runs[1][0]
    assert runs[1][1]["metrics"]["candidate_visits"] <= runs[0][1]["metrics"]["candidate_visits"]
    if cap is None:
        assert runs[1][1]["metrics"]["candidate_visits"] < runs[0][1]["metrics"]["candidate_visits"]
    assert runs[1][1]["posting_entries_removed_before_enumeration"] > 0


def test_pause_and_resume_without_duplicate_or_missing_pairs(source, tmp_path):
    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    output = tmp_path / "stream"
    partial = run_document_stream(
        data, output, 1, source_index_dir=root, probes_per_commit=1,
        batch_size=1, max_partitions=1, verbose=False,
    )
    assert partial["status"] == "partial" and partial["next_probe"] == 1
    assert not partial["exhaustive_within_subset"]
    final = run_document_stream(
        data, output, 1, source_index_dir=root, probes_per_commit=1,
        batch_size=1, verbose=False,
    )
    rows = stream_rows(output)
    assert as_dict(rows) == expected(range(len(SETS)), 1)
    assert len(rows) == len(as_dict(rows))
    assert final["metrics"]["verification_calls"] == len(rows)
    again = run_document_stream(
        data, output, 1, source_index_dir=root, probes_per_commit=1,
        batch_size=1, verbose=False,
    )
    assert again == final
    with pytest.raises(ValueError, match="mismatch"):
        run_document_stream(data, output, 2, source_index_dir=root, probes_per_commit=1, batch_size=1, verbose=False)
    with pytest.raises(ValueError, match="mismatch"):
        run_document_stream(data, output, 1, source_index_dir=root, probes_per_commit=1, batch_size=1, output_mode="threshold_only", verbose=False)
    with pytest.raises(ValueError, match="mismatch"):
        run_document_stream(data, output, 1, source_index_dir=root, probes_per_commit=1, batch_size=1, candidate_df_cap=2, verbose=False)


def test_interruption_after_output_write_before_commit_recovers(source, tmp_path, monkeypatch):
    from findmytext import overlap_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    atomic = overlap_stream.atomic_json
    output = tmp_path / "interrupted"

    def interrupt(path, checkpoint):
        if path.name == "checkpoint.json" and checkpoint["parts"]:
            raise RuntimeError("interrupted before checkpoint publication")
        atomic(path, checkpoint)

    monkeypatch.setattr(overlap_stream, "atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        overlap_stream.run_document_stream(
            data, output, 1, source_index_dir=root, probes_per_commit=1,
            batch_size=1, verbose=False,
        )
    assert list(output.glob("part-*.npy"))
    monkeypatch.setattr(overlap_stream, "atomic_json", atomic)
    final = overlap_stream.run_document_stream(
        data, output, 1, source_index_dir=root, probes_per_commit=1,
        batch_size=1, verbose=False,
    )
    rows = stream_rows(output)
    assert final["status"] == "complete"
    assert as_dict(rows) == expected(range(len(SETS)), 1)
    assert len(rows) == len(as_dict(rows))


def test_subset_persistence_and_changed_committed_output_are_rejected(source, tmp_path):
    from findmytext.overlap_inverted import _source_identity
    from findmytext.overlap_stream import load_subset, run_document_stream, save_subset, subset_identity

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    save_subset(data, tmp_path / "subset", _source_identity(root))
    loaded = load_subset(tmp_path / "subset")
    assert subset_identity(data) == subset_identity(loaded)
    output = tmp_path / "stream"
    checkpoint = run_document_stream(loaded, output, 1, source_index_dir=root, verbose=False)
    part = output / checkpoint["parts"][0]["file"]
    with part.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        run_document_stream(loaded, output, 1, source_index_dir=root, verbose=False)


def test_progressive_benchmark_resumes_and_checks_all_modes(source, tmp_path):
    from experiments.scale_overlap_join import scale_benchmark

    root, overlap = source
    report = scale_benchmark(
        root, overlap, tmp_path / "scale", sizes=(4, 6), k_values=(5, 3, 1),
        candidate_df_cap=2, memory_budget_bytes=1 << 28, reference_batch_entries=2,
    )
    assert report["status"] == "passed"
    assert all(stage["status"] == "passed" for stage in report["sizes"].values())
    assert all(len(stage["runs"]) == 12 for stage in report["sizes"].values())
    assert len(report["sizes"]["4"]["scheduler_ram_controls"]) == 8
    assert len(report["sizes"]["4"]["pruning_controls"]) == 2
    resumed = scale_benchmark(
        root, overlap, tmp_path / "scale", sizes=(4, 6), k_values=(5, 3, 1),
        candidate_df_cap=2, memory_budget_bytes=1 << 28, reference_batch_entries=2,
    )
    assert resumed["status"] == "passed"
    with pytest.raises(ValueError, match="mismatch"):
        scale_benchmark(
            root, overlap, tmp_path / "scale", sizes=(4, 6), k_values=(5, 3, 1),
            candidate_df_cap=3, memory_budget_bytes=1 << 28, reference_batch_entries=2,
        )


def test_explicit_pair_passage_bypasses_topk_and_bounds_resources(source, monkeypatch):
    from findmytext.detectors import FingerprintChainDetector

    root, _ = source
    detector = FingerprintChainDetector(str(root), top_k=1)

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit pair inspection must not retrieve or truncate candidates")

    monkeypatch.setattr(detector, "_get_candidates", forbidden)
    result = detector.get_pair_passage(0, 1)
    assert result["status"] == "cluster"
    assert result["shared_fingerprints"] == 5
    assert result["best_passage"]["distinct_fingerprint_support"] == 5
    assert result["best_passage"]["score"] == 5
    assert result["best_passage"]["token_span_i"] == [1, 19]
    assert detector.get_pair_passage(0, 4)["status"] == "no_shared_fingerprints"
    with pytest.raises(MemoryError, match="alignment"):
        detector.get_pair_passage(0, 1, max_alignment_points=1)
    with pytest.raises(MemoryError, match="posting"):
        detector.get_pair_passage(0, 1, max_posting_bytes=1)
    with pytest.raises(ValueError):
        detector.get_pair_passage(0, 0)
    with pytest.raises(ValueError):
        detector.get_pair_passage(-1, 1)
    weighted = FingerprintChainDetector(str(root), rarity_weighted=True)
    evidence = weighted.get_pair_passage(0, 1)
    assert evidence["parameters"]["score"] == "rarity_weighted_distinct_hashes"
    assert evidence["best_passage"]["distinct_fingerprint_support"] == 5
    assert evidence["best_passage"]["score"] > 0


def test_synthetic_passage_controls_distinguish_scattered_and_aligned_text(tmp_path):
    from experiments.inspect_overlap_passages import synthetic_controls

    result = synthetic_controls(tmp_path / "controls")
    assert result["aligned_positive"]["status"] == "cluster"
    assert result["scattered_shared_grams_negative"]["status"] == "no_cluster"
    assert result["scattered_shared_grams_negative"]["shared_fingerprints"] >= 5
    assert result["contiguous_template_semantic_negative"]["status"] == "cluster"


def test_real_source_passage_inspection_end_to_end(tmp_path):
    import zstandard

    from experiments.inspect_overlap_passages import inspect_passages, synthetic_controls
    from experiments.scale_overlap_join import scale_benchmark

    setup = tmp_path / "setup"
    synthetic_controls(setup)
    source = setup / "merged"
    overlap = tmp_path / "overlap"
    build_overlap_inverted_index(source, overlap, verbose=False)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sample.jsonl.zst").write_bytes(
        zstandard.ZstdCompressor().compress((setup / "controls.jsonl").read_bytes())
    )
    scale_benchmark(
        source, overlap, tmp_path / "scale", sizes=(6,), k_values=(50, 30, 20, 10, 5),
        memory_budget_bytes=1 << 28, reference_batch_entries=20,
        fingerprint_control=False,
    )
    report = inspect_passages(source, corpus, tmp_path / "scale/6", tmp_path / "inspection")
    assert report["status"] == "passed"
    assert report["source_unchanged"]
    assert report["summary"]["no_cluster"] >= 1
    assert report["summary"]["clusters"] >= 2
    for row in report["pairs"]:
        if row["best_passage"]:
            assert row["best_passage"]["source_positions_and_exact_kgrams_verified"]
    with pytest.raises(FileExistsError):
        inspect_passages(source, corpus, tmp_path / "scale/6", tmp_path / "inspection")


def test_stream_rejects_mixed_source_before_creating_output(source, tmp_path):
    import shutil

    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    other = tmp_path / "other-index"
    shutil.copytree(root, other)
    with pytest.raises(ValueError, match="different or changed"):
        run_document_stream(data, tmp_path / "mixed", 1, source_index_dir=other)
    assert not (tmp_path / "mixed").exists()


def test_explicit_pair_keeps_only_the_highest_scoring_surviving_passage(tmp_path):
    from experiments.compare_similarity_methods import build_index
    from findmytext.detectors import FingerprintChainDetector

    short = " ".join(f"shortshared{i}" for i in range(60))
    long = " ".join(f"longshared{i}" for i in range(180))
    padding_a = " ".join(f"padding_a{i}" for i in range(200))
    padding_b = " ".join(f"padding_b{i}" for i in range(250))
    records = [
        {"id": "a", "text": short + " " + padding_a + " " + long},
        {"id": "b", "text": "prefix " + short + " " + padding_b + " " + long},
    ]
    corpus = tmp_path / "pairs.jsonl"
    corpus.write_text("".join(json.dumps(record) + "\n" for record in records))
    index = build_index(str(corpus), str(tmp_path))
    detector = FingerprintChainDetector(index)
    result = detector.get_pair_passage(0, 1)
    assert result["status"] == "cluster"
    assert result["surviving_clusters"] >= 2
    assert result["best_passage"]["token_span_i"][0] >= 260
    assert result["best_passage"]["token_span_j"][0] >= 311
    assert result["best_passage"]["score"] == result["best_passage"]["distinct_fingerprint_support"]


def test_floor_k_cache_reuses_only_supported_thresholds(source, tmp_path):
    from findmytext.overlap_stream import filter_count_cache, run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    stream = tmp_path / "floor"
    run_document_stream(data, stream, 3, source_index_dir=root, batch_size=1, verbose=False)
    report = filter_count_cache(stream, tmp_path / "cache", (50, 5, 3))
    assert report["minimum_supported_k"] == 3
    for k in (50, 5, 3):
        rows = np.load(tmp_path / "cache" / report["thresholds"][str(k)]["file"])
        assert as_dict(rows) == expected(range(len(SETS)), k)
    with pytest.raises(ValueError, match="below"):
        filter_count_cache(stream, tmp_path / "bad-cache", (2,))


@pytest.mark.parametrize("k", [1, 2, 3, 5, 6, 50])
@pytest.mark.parametrize("batch_size", [1, 1000])
def test_exact_prefix_pair_sets_and_counts_equal_unfiltered_reference(source, tmp_path, k, batch_size):
    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    for mode in ("exact_counts", "threshold_only"):
        output = tmp_path / mode
        checkpoint = run_document_stream(
            data, output, k, source_index_dir=root, prefix_filter=True,
            output_mode=mode, batch_size=batch_size, probes_per_commit=2, verbose=False,
        )
        rows = stream_rows(output)
        if mode == "exact_counts":
            assert as_dict(rows) == expected(range(len(SETS)), k)
        else:
            assert {(int(r["doc_i"]), int(r["doc_j"])) for r in rows} == set(expected(range(len(SETS)), k))
        assert checkpoint["exhaustive_within_subset"]
        assert checkpoint["contract"]["prefix_filter"]
        assert checkpoint["metrics"]["verification_calls"] <= len(expected(range(len(SETS)), 1))


def test_full_view_reuses_source_payloads_and_matches_subset_factory(source, tmp_path):
    from findmytext.overlap_full import load_full_corpus, prepare_full_corpus
    from findmytext.overlap_stream import subset_identity

    root, overlap = source
    view = tmp_path / "full-view"
    data = prepare_full_corpus(root, overlap, view, 1 << 28)
    ordinary = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    assert subset_identity(data) == subset_identity(ordinary)
    assert Path(data.forward_fingerprints.filename) == root / "forward_fingerprints.npy"
    assert Path(data.posting_docs.filename) == overlap / "doc_ids.npy"
    assert not (view / "forward_fingerprints.npy").exists()
    assert not (view / "doc_ids.npy").exists()
    assert subset_identity(load_full_corpus(view, 1 << 28)) == subset_identity(data)
    assert subset_identity(prepare_full_corpus(root, overlap, view, 1 << 28)) == subset_identity(data)
    with pytest.raises(MemoryError):
        load_full_corpus(view, 1)


def test_prefix_checkpoint_and_partial_probe_coverage(source, tmp_path):
    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    output = tmp_path / "prefix"
    partial = run_document_stream(
        data, output, 2, source_index_dir=root, prefix_filter=True,
        max_partitions=1, probes_per_commit=1, batch_size=1, verbose=False,
    )
    assert partial["status"] == "partial"
    complete = run_document_stream(
        data, output, 2, source_index_dir=root, prefix_filter=True,
        probes_per_commit=1, batch_size=1, verbose=False,
    )
    assert as_dict(stream_rows(output)) == expected(range(len(SETS)), 2)
    assert complete["status"] == "complete"
    with pytest.raises(ValueError, match="mismatch"):
        run_document_stream(data, output, 2, source_index_dir=root, probes_per_commit=1, batch_size=1)
    pilot = run_document_stream(
        data, tmp_path / "pilot", 2, source_index_dir=root, prefix_filter=True,
        probe_start=2, probe_stop=4, verbose=False,
    )
    assert pilot["status"] == "complete"
    assert pilot["next_probe"] == 4
    assert not pilot["exhaustive_within_subset"]
    assert pilot["contract"]["probe_range"] == [2, 4]


def test_default_working_allowance_is_32_gib():
    from unittest.mock import patch

    from experiments.benchmark_overlap_join import choose_memory_budget

    with patch.object(Path, "read_text", return_value=f"MemAvailable: {128 * 2**20} kB\n"):
        budget, available = choose_memory_budget()
    assert budget == 32 * 2**30
    assert available == 128 * 2**30
    with patch.object(Path, "read_text", return_value="MemAvailable: 1024 kB\n"):
        with pytest.raises(MemoryError, match="32 GiB"):
            choose_memory_budget()


def test_prefix_full_pipeline_with_independent_pilot_and_sample_verification(source, tmp_path):
    from experiments.run_full_overlap_join import finish_full_run, run_pipeline
    from experiments.scale_overlap_join import scale_benchmark

    root, overlap = source
    scale_benchmark(
        root, overlap, tmp_path / "reference", sizes=(6,),
        k_values=(50, 30, 20, 10, 5), memory_budget_bytes=1 << 28,
        reference_batch_entries=2, fingerprint_control=False,
    )
    report = run_pipeline(
        root, overlap, tmp_path / "reference/6", tmp_path / "full-run",
        memory_budget_bytes=1 << 28,
    )
    assert report["status"] == "passed"
    assert report["source_unchanged"]
    assert len(report["prefix_validation"]["runs"]) == 10
    assert report["pilot"]["status"] == "passed"
    assert len(report["pilot"]["strata"]) == 4
    assert report["verification"]["accepted_pairs"] == 1
    assert report["verification"]["independent_numeric_forward_samples"] == 1
    assert not report["verification"]["independent_exhaustive_full_corpus_oracle_run"]
    assert report["k_sweep"]["thresholds"]["5"]["pairs"] == 1
    assert report["k_sweep"]["thresholds"]["50"]["pairs"] == 0
    resumed = run_pipeline(
        root, overlap, tmp_path / "reference/6", tmp_path / "full-run",
        memory_budget_bytes=1 << 28,
    )
    assert resumed["status"] == "passed"
    assert resumed["verification"] == report["verification"]
    finished = finish_full_run(tmp_path / "full-run")
    assert finished["status"] == "passed"
    assert finished["additional_independent_checks"]["numeric_sample_method"] == "numpy_intersect1d_assume_unique"
    assert finished["additional_independent_checks"]["derived_threshold_rows_compared_to_exact_floor"] == 1


def test_full_output_verifier_rejects_a_complete_but_partial_document_subset(source, tmp_path):
    from experiments.run_full_overlap_join import verify_full_output
    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(3), 1 << 28)
    output = tmp_path / "only-three-documents"
    run_document_stream(data, output, 5, source_index_dir=root, prefix_filter=True, verbose=False)
    with pytest.raises(ValueError, match="all source documents"):
        verify_full_output(data, root, output)


def test_prefix_only_cross_corpus_benchmark_uses_oracle_without_extra_controls(source, tmp_path):
    from experiments.scale_overlap_join import scale_benchmark

    root, overlap = source
    report = scale_benchmark(
        root, overlap, tmp_path / "prefix-only", sizes=(6,), k_values=(5, 3, 1),
        candidate_df_cap=2, memory_budget_bytes=1 << 28,
        reference_batch_entries=2, fingerprint_control=False,
        prefix_filter=True, run_controls=False,
    )
    assert report["status"] == "passed"
    assert report["contract"]["prefix_filter"]
    assert report["contract"]["run_controls"] is False
    stage = report["sizes"]["6"]
    assert len(stage["runs"]) == 12
    assert all(row["validated"] and row["prefix_filter"] for row in stage["runs"])
    assert "pruning_controls" not in stage
    assert "scheduler_ram_controls" not in stage


def test_overnight_workload_policy_selects_a_bounded_larger_stage():
    from experiments.overnight_prefix_benchmarks import choose_larger_stage

    policy = {
        "larger_sizes_descending": [100000, 50000],
        "max_projected_pair_contributions": 250_000_000,
        "max_projected_positive_pairs": 100_000_000,
    }
    assert choose_larger_stage(25000, {"pair_contributions": 2_000_000, "positive_pairs": 1_000_000}, policy)["documents"] == 100000
    decision = choose_larger_stage(25000, {"pair_contributions": 20_000_000, "positive_pairs": 5_000_000}, policy)
    assert decision["documents"] == 50000
    assert len(decision["forecasts"]) == 2
    decision = choose_larger_stage(25000, {"pair_contributions": 100_000_000, "positive_pairs": 50_000_000}, policy)
    assert decision["documents"] is None
    with pytest.raises(ValueError):
        choose_larger_stage(1, {"pair_contributions": 1, "positive_pairs": 1}, policy)


def test_overnight_queue_continues_other_languages_after_stage_failure(tmp_path, monkeypatch):
    from experiments import overnight_prefix_benchmarks as overnight

    root = tmp_path / "night"
    root.mkdir()
    nynorsk = tmp_path / "nynorsk.json"
    nynorsk.write_text(json.dumps({"status": "passed", "additional_independent_checks": {"status": "passed"}}))
    config = {
        "run_id": "test", "memory_budget_bytes": 1 << 28,
        "nynorsk_report": str(nynorsk), "source_snapshot_sha256": {},
        "screen_documents": 4,
        "policy": {
            "larger_sizes_descending": [8, 6],
            "max_projected_pair_contributions": 1000,
            "max_projected_positive_pairs": 1000,
        },
        "jobs": [{"name": name} for name in ("icelandic", "latvian", "bokmal")],
    }
    (root / "run.json").write_text(json.dumps(config))

    def stage(root, config, status, job, size):
        if job["name"] == "icelandic" and size == 8:
            return False
        value = 10000 if job["name"] == "latvian" else 10
        job["completed_stages"].append({
            "documents": size,
            "reference": {"pair_contributions": value, "positive_pairs": value},
        })
        return True

    monkeypatch.setattr(overnight, "run_stage", stage)
    result = overnight.run_queue(root)
    assert result["status"] == "completed_with_failures"
    assert result["nynorsk_dependency"] == "passed"
    assert [job["status"] for job in result["jobs"]] == [
        "failed", "completed_screen_only_workload_limit", "passed",
    ]


@pytest.mark.parametrize("mode", ["exact_counts", "threshold_only"])
def test_accepted_output_buffer_coalesces_parts_without_changing_results(source, tmp_path, mode):
    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    configs = []
    for buffer_rows in (None, 4):
        output = tmp_path / f"buffer-{buffer_rows}"
        cp = run_document_stream(
            data, output, 1, source_index_dir=root, prefix_filter=True,
            output_mode=mode, batch_size=1, probes_per_commit=6,
            output_buffer_rows=buffer_rows, verbose=False,
        )
        configs.append((cp, stream_rows(output)))
    np.testing.assert_array_equal(configs[0][1], configs[1][1])
    assert len(configs[1][0]["parts"]) < len(configs[0][0]["parts"])
    assert configs[1][0]["contract"]["output_buffer_rows"] == 4


def test_buffered_output_flushes_before_pause_and_resumes_exactly(source, tmp_path):
    from findmytext.overlap_stream import run_document_stream

    root, overlap = source
    data = prepare_subset(root, overlap, np.arange(len(SETS)), 1 << 28)
    output = tmp_path / "buffered-resume"
    first = run_document_stream(
        data, output, 1, source_index_dir=root, prefix_filter=True,
        batch_size=1, probes_per_commit=4, output_buffer_rows=2,
        max_partitions=1, verbose=False,
    )
    assert first["status"] == "partial" and first["accepted_pairs"] > 0
    final = run_document_stream(
        data, output, 1, source_index_dir=root, prefix_filter=True,
        batch_size=1, probes_per_commit=4, output_buffer_rows=2, verbose=False,
    )
    assert final["status"] == "complete"
    assert as_dict(stream_rows(output)) == expected(range(len(SETS)), 1)


def test_full_followup_preflight_matches_logical_view_and_reviews_real_pilot(source, tmp_path):
    from experiments.queue_full_overlap import full_working_bounds, review_pilot
    from experiments.scale_overlap_join import scale_benchmark
    from findmytext.overlap_full import prepare_full_corpus

    root, overlap = source
    data = prepare_full_corpus(root, overlap, tmp_path / "view", 1 << 28)
    bounds = full_working_bounds(root, overlap)
    assert bounds["documents"] == len(SETS)
    assert bounds["logical_view_bytes"] == data.nbytes
    assert bounds["required_working_bytes"] >= bounds["logical_view_bytes"]
    pilot_root = tmp_path / "pilot"
    scale_benchmark(
        root, overlap, pilot_root, sizes=(6,), k_values=(50, 30, 20, 10, 5),
        memory_budget_bytes=1 << 28, prefix_filter=True, run_controls=False,
        fingerprint_control=False, reference_batch_entries=2,
    )
    job = {"index_dir": str(root), "overlap_dir": str(overlap), "preflight": bounds}
    pilot = {
        "status": "passed",
        "completed_stages": [{"documents": 6, "report": str(pilot_root / "report.json")}],
    }
    checked = review_pilot(job, pilot, 1 << 28)
    assert checked["status"] == "passed"
    assert checked["projected_pairs_per_k"]["5"] == 1
    assert len(checked["rechecked_results"]) == 3
    assert review_pilot(job, {"status": "failed"}, 1 << 28)["status"] == "held_pilot_not_passed"


def test_full_followup_queue_holds_a_corpus_exceeding_working_allowance(source, tmp_path):
    from experiments.queue_full_overlap import run_full_queue

    root, overlap = source
    queue = tmp_path / "full-queue"
    queue.mkdir()
    pilot_status = tmp_path / "pilots.json"
    pilot_status.write_text(json.dumps({
        "status": "passed", "jobs": [{"name": "test", "status": "passed"}],
    }))
    config = {
        "memory_budget_bytes": 1, "source_snapshot_sha256": {},
        "pilot_status": str(pilot_status),
        "jobs": [{"name": "test", "index_dir": str(root), "overlap_dir": str(overlap)}],
    }
    (queue / "run.json").write_text(json.dumps(config))
    result = run_full_queue(queue)
    assert result["status"] == "completed_with_holds"
    assert result["jobs"][0]["status"] == "held_memory_allowance"
    assert result["child_pid"] is None


def test_shared_logged_command_records_exit_and_releases_child_pid(tmp_path):
    import os
    import sys

    from experiments.overnight_prefix_benchmarks import run_logged_command

    (tmp_path / "source_snapshot").mkdir()
    status = {"child_pid": None}
    log = tmp_path / "child.log"
    code = run_logged_command(
        tmp_path, [sys.executable, "-c", "print('child completed')"],
        log, os.environ.copy(), status,
    )
    assert code == 0
    assert status["child_pid"] is None
    assert "heartbeat_at" in status
    assert "child completed" in log.read_text()
