import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from findmytext import detectors, index_builder, indexing
from findmytext.winnower import HASH_METHOD, ROLLING_BASE64, Winnower, winnow


def reference_fingerprints(tokens, length, base):
    mask = 2**64 - 1
    values = []
    for token in tokens:
        value = 1469598103934665603
        for byte in token.encode("utf-8"):
            value = ((value ^ byte) * 1099511628211) & mask
        values.append(value)
    result = []
    for start in range(max(0, len(tokens) - length + 1)):
        value = 0
        for token in values[start:start + length]:
            value = (value * base + token) & mask
        value = (value * base) & mask
        value ^= value >> 33
        value = value * 0xFF51AFD7ED558CCD & mask
        value ^= value >> 33
        value = value * 0xC4CEB9FE1A85EC53 & mask
        value ^= value >> 33
        result.append(value)
    return result


def test_hashing_matches_independent_unsigned_reference_and_rightmost_winnowing():
    tokens = ["blå", "", "東京", "punctuation", "repeated", "repeated"] * 8
    for base in (ROLLING_BASE64, 1_000_003):
        configured = Winnower(length=4, window_size=6, base=base)
        actual = configured.get_token_fingerprints(tokens)
        assert actual.dtype == np.uint64
        assert list(map(int, actual)) == reference_fingerprints(tokens, 4, base)
    values = np.array([2**63 + 9] * 10, dtype=np.uint64)
    hashes, positions = winnow(values, 3)
    assert hashes.dtype == np.uint64
    assert hashes.tolist() == [2**63 + 9] * 8
    assert positions.tolist() == list(range(2, 10))
    for text in ("", "too short"):
        assert Winnower(length=4).get_winnowed_fingerprints(text)[0].dtype == np.uint64
        assert not len(Winnower(length=4).get_winnowed_fingerprints(text)[0])
    configured = Winnower(length=4)
    assert Winnower.from_metadata(configured.to_metadata()).to_metadata() == configured.to_metadata()
    with pytest.raises(TypeError, match="hash_method"):
        Winnower(hash_method="legacy31")


def make_corpus(path, count=7):
    shared = " ".join(f"sharedword{i}" for i in range(90))
    records = [
        {"id": f"document-{i}", "text": shared + f" unique{i} " + shared[:120]}
        for i in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return records


def assert_index_matches_source(directory, records):
    reader = indexing.DiskBasedIndex(str(directory))
    index_winnower = Winnower.from_metadata(reader.get_metadata())
    expected = {}
    for row in records:
        values, positions = index_winnower.get_winnowed_fingerprints(row["text"])
        for value, position in zip(values, positions):
            expected.setdefault(int(value), []).append((row["id"], int(position)))
    assert list(map(int, reader.fingerprints)) == sorted(expected)
    postings = reader._get_postings(reader.fingerprints)
    for value, entries in expected.items():
        actual = sorted(
            (reader.to_external_doc_id[int(row["doc_id"])], int(row["position"]))
            for row in postings[np.uint64(value)]
        )
        assert actual == sorted(entries)
    assert set(reader.to_external_doc_id) == {row["id"] for row in records}
    return reader


def test_multiple_sub_thousand_flushes_preserve_every_occurrence_and_query(tmp_path, monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source)
    # A budget of one posting flushes after every document.
    shards = index_builder.Indexer(max_memory_gb=1e-9).index_file(str(source), str(tmp_path / "shards"))
    assert len(shards) == len(set(shards)) == len(records)
    target = tmp_path / "index"
    index_builder.Merger().merge(shards, str(target))
    reader = assert_index_matches_source(target, records)
    assert reader.get_metadata()["hash_method"] == HASH_METHOD
    assert not list(target.glob("forward*"))
    query = records[0]["text"]
    for cls in (detectors.NbSharedFingerprintsDetector, detectors.FingerprintChainDetector):
        for rarity in (False, True):
            detector = cls(str(target), rarity_weighted=rarity)
            hashes, positions = detector.get_fingerprints(query)
            assert hashes.dtype == np.uint64 and len(hashes) == len(positions) > 0
            scores = detector.search(query, min_fragment_length=100)
            assert scores[records[0]["id"]] > 0.5
            assert min(scores.values()) >= detector.min_probability
        # Far too short a match for a 5000-char fragment: nothing is returned, whether
        # the candidates are pruned at retrieval (derived threshold) or scored and
        # then dropped below min_probability (explicit min_fingerprints=1).
        with pytest.warns(UserWarning, match="shorter than min_fragment_length"):
            assert cls(str(target)).search(query, min_fragment_length=5000) == {}
        with pytest.warns(UserWarning, match="shorter than min_fragment_length"):
            assert cls(str(target), min_fingerprints=1).search(query, min_fragment_length=5000) == {}
    # Normalisation (unweighted detector, so the dispersion is exactly the default 0.5):
    # N0 = 0.9^4 * (1000 / 5.5 - 4 + 1) * 2 / 7 ~ 33.5 for k=4, w=6 and a 10% word-edit
    # rate; sigma = sqrt(0.5 * N0).
    plain = detectors.NbSharedFingerprintsDetector(str(target))
    n0 = 0.9**4 * (1000 / 5.5 - 4 + 1) * 2 / 7
    assert plain._expected_fingerprints(1000) == pytest.approx(n0)
    probs = plain._normalise_scores({"mid": n0 - 0.5, "high": n0 + 2 * (0.5 * n0) ** 0.5, "low": 0}, 1000)
    assert probs["mid"] == pytest.approx(0.5) and probs["high"] > 0.97 and probs["low"] < 1e-6
    # Retrieval threshold: the smallest count whose probability reaches the default
    # min_probability of 0.01 (z_0.01 = -2.326), so it inverts the normalisation
    # exactly at the boundary.
    assert plain.min_probability == 0.01
    threshold = plain._min_shared_fingerprints(1000)
    assert threshold == int(np.ceil(n0 - 0.5 - 2.3263 * (0.5 * n0) ** 0.5)) == 24
    boundary = plain._normalise_scores({"at": threshold, "below": threshold - 1}, 1000)
    assert boundary["at"] >= plain.min_probability > boundary["below"]
    assert plain._min_shared_fingerprints(100) < threshold
    # At min_probability=0.5 the threshold is N0 - 0.5 itself (z = 0), i.e. ceil(33.0) = 34.
    assert detectors.NbSharedFingerprintsDetector(str(target), min_probability=0.5)._min_shared_fingerprints(1000) == 34
    assert detectors.NbSharedFingerprintsDetector(str(target), min_fingerprints=1)._min_shared_fingerprints(1000) == 1
    assert detector.rarity_weighted and detector._min_shared_fingerprints(1000) < threshold
    for bad in ({"min_probability": 0}, {"min_probability": 1}, {"min_fingerprints": 0}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            detectors.NbSharedFingerprintsDetector(str(target), **bad)
    # Rarity weighting widens the transition: same scores, less extreme probabilities.
    weighted_probs = detector._normalise_scores({"high": n0 + 2 * (0.5 * n0) ** 0.5}, 1000)
    assert detector.rarity_weighted and 0.5 < weighted_probs["high"] < probs["high"]
    steep = detectors.NbSharedFingerprintsDetector(str(target), dispersion=0.2)
    steep_probs = steep._normalise_scores({"above": n0 + 3, "below": n0 - 3}, 1000)
    flat_probs = plain._normalise_scores({"above": n0 + 3, "below": n0 - 3}, 1000)
    assert steep_probs["above"] > flat_probs["above"] and steep_probs["below"] < flat_probs["below"]
    with pytest.raises(ValueError, match="dispersion"):
        detectors.NbSharedFingerprintsDetector(str(target), dispersion=0)


def test_rarity_weights_are_idf_like_and_normalised_to_unit_occurrence_mean(tmp_path):
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source)
    target = tmp_path / "index"
    index_builder.Merger().merge(
        index_builder.Indexer().index_file(str(source), str(tmp_path / "shards")), str(target)
    )
    detector = detectors.NbSharedFingerprintsDetector(str(target), rarity_weighted=True, dispersion=0.5)
    weighting = detector.weighting
    n_docs = len(records)
    counts = np.asarray(detector.index.lengths, dtype=float)
    idf = np.log1p(n_docs / counts)
    assert weighting.mean == pytest.approx(np.average(idf, weights=counts))
    assert weighting.extra_dispersion == pytest.approx(
        np.average(idf**2, weights=counts) / weighting.mean**2 - 1
    )
    assert weighting.extra_dispersion > 0  # unequal weights add variance
    assert detector._score_dispersion == pytest.approx(0.5 + weighting.extra_dispersion)
    assert weighting.max == pytest.approx(np.log1p(n_docs) / weighting.mean)
    plain = detectors.NbSharedFingerprintsDetector(str(target))
    assert plain.weighting is None and plain._score_dispersion == 0.5

    # Weights follow log(1 + N / cf) / mean and so decrease with the posting count;
    # the corpus has fingerprints seen 7 and 14 times (the shared text is repeated).
    hashes = np.unique(detector.get_fingerprints(records[0]["text"])[0])
    values = detector._fingerprint_weights(hashes)
    posting_counts = weighting.posting_counts(hashes)
    assert posting_counts.min() >= 1 and posting_counts.min() < posting_counts.max()
    assert values == pytest.approx(np.log1p(n_docs / posting_counts) / weighting.mean)
    assert np.array_equal(values, weighting(hashes))
    order = np.argsort(posting_counts)
    assert np.all(np.diff(values[order]) <= 1e-12)
    # Unknown fingerprints have a posting count of 0 and weight 0; uniform weights are 1.
    unknown = np.array([0], dtype=np.uint64)
    assert weighting.posting_counts(unknown).tolist() == [0]
    assert detector._fingerprint_weights(unknown).tolist() == [0.0]
    assert plain._fingerprint_weights(unknown).tolist() == [1.0]


def test_rectangle_clustering_matches_chebyshev_linkage_and_dense_reference():
    # Four dense clusters plus noise on an integer grid, so that many pairwise
    # distances fall exactly on the thresholds.
    rng = np.random.default_rng(0)
    points = np.concatenate(
        [c + rng.integers(-20, 21, size=(60, 2)) for c in rng.integers(0, 2000, size=(4, 2))]
        + [rng.integers(0, 2000, size=(40, 2))]
    )
    pos, offset = points[:, 0], points[:, 1]

    def labels(method, position_threshold, offset_threshold, min_cluster_size=5):
        if method == "rectangle":
            return detectors.cluster_rectangle(pos, offset, position_threshold, offset_threshold, min_cluster_size)
        return detectors.cluster_linkage(pos, offset, method, position_threshold, min_cluster_size)

    def dense_reference(position_threshold, offset_threshold, min_cluster_size=5):
        diff = np.abs(points[:, None, :] - points[None, :, :])
        adjacency = (diff[..., 0] <= position_threshold) & (diff[..., 1] <= offset_threshold)
        _, raw = connected_components(csr_matrix(adjacency), directed=False)
        return detectors._relabel_small_clusters(raw, min_cluster_size)

    def assert_same_partition(a, b):  # equal up to the numbering of the valid clusters
        np.testing.assert_array_equal(a == -1, b == -1)
        pairs = set(zip(a[a != -1].tolist(), b[b != -1].tolist()))
        assert len(pairs) == len(set(a.tolist()) - {-1}) == len(set(b.tolist()) - {-1})

    for t in (10, 30):
        rect = labels("rectangle", t, t)
        assert rect.dtype == np.int32 and rect.min() == -1 and rect.max() >= 1
        np.testing.assert_array_equal(rect, dense_reference(t, t))
        assert_same_partition(rect, labels("chebyshev", t, t))
    for thresholds in ((10, 40), (40, 10)):
        np.testing.assert_array_equal(labels("rectangle", *thresholds), dense_reference(*thresholds))
    assert labels("rectangle", 1, 1, min_cluster_size=1).max() > labels("rectangle", 1, 1).max()
    assert detectors.cluster_rectangle(np.array([3]), np.array([5]), 30, 30, 1).tolist() == [0]
    assert detectors.cluster_linkage(np.array([3]), np.array([5]), "euclidean", 30, 1).tolist() == [0]
    # The detector dispatches on its method and rejects unknown ones.
    chain = detectors.FingerprintChainDetector.__new__(detectors.FingerprintChainDetector)
    chain.method, chain.position_threshold, chain.offset_threshold = "rectangle", 30, 30
    chain.distance_threshold, chain.min_cluster_size = 30, 5
    np.testing.assert_array_equal(chain.cluster(pos, offset), labels("rectangle", 30, 30))
    chain.method = "dbscan"
    with pytest.raises(ValueError, match="Unknown clustering method"):
        chain.cluster(pos, offset)

    relabelled = detectors._relabel_small_clusters(np.array([7, 3, 7, 3, 9, 7]), 2)
    assert relabelled.dtype == np.int32 and relabelled.tolist() == [1, 0, 1, 0, -1, 1]
    assert detectors._relabel_small_clusters(np.empty(0, dtype=np.int32), 1).tolist() == []


def test_candidates_join_expands_repeated_hashes_and_cluster_weights_count_hashes_once():
    # Hash 10 occurs twice in the query (positions 0 and 7), hash 20 once, hash 30 once
    # but never in the document.
    candidates = detectors.Candidates.from_query(
        np.array([10, 20, 10, 30], dtype=np.uint64), np.array([0, 1, 7, 2])
    )
    assert candidates.is_empty and len(candidates.hits) == 0
    assert candidates.hashes.tolist() == [10, 10, 20, 30] and candidates.hashes.dtype == np.uint64
    assert candidates.positions.tolist() == [0, 7, 1, 2]
    weights = candidates.hashes.astype(float) / 10  # one weight per query k-gram

    block = np.array(
        [(0, 10, 100), (0, 10, 105), (0, 20, 101)], dtype=indexing.HIT_DTYPE
    )
    # A fingerprint is identified by the index of its first query occurrence.
    assert candidates.shared_fingerprints(block).tolist() == [0, 2]
    fingerprint, pos_query, pos_doc = candidates.join(block)
    # Every (query occurrence x document occurrence) pair, as an inner join would give.
    pairs = sorted(zip(fingerprint.tolist(), pos_query.tolist(), pos_doc.tolist()))
    assert pairs == [(0, 0, 100), (0, 0, 105), (0, 7, 100), (0, 7, 105), (2, 1, 101)]
    assert pos_query.dtype == pos_doc.dtype == np.int64

    # One cluster holding hash 10 (several pairs) and hash 20: each hash counts once.
    labels = np.zeros(5, dtype=np.int32)
    assert detectors.largest_cluster_weight(labels, fingerprint, weights) == pytest.approx(3.0)
    # Two clusters: the largest weight wins, noise (-1) is ignored.
    labels = np.array([1, 1, -1, 0, 0], dtype=np.int32)
    expected = max(
        weights[np.unique(fingerprint[labels == c])].sum() for c in (0, 1)
    )
    assert detectors.largest_cluster_weight(labels, fingerprint, weights) == pytest.approx(expected)
    assert detectors.largest_cluster_weight(np.full(5, -1, dtype=np.int32), fingerprint, weights) == 0.0

    empty = candidates.join(np.empty(0, dtype=indexing.HIT_DTYPE))
    assert all(len(part) == 0 for part in empty)


@pytest.mark.parametrize("start_method", [name for name in ("spawn", "forkserver") if name in mp.get_all_start_methods()])
def test_parallel_build_accepts_generator_with_modern_start_methods(tmp_path, start_method):
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=4)
    script = (
        "import multiprocessing as mp\n"
        "from findmytext import index_builder\n"
        "if __name__ == '__main__':\n"
        f"    mp.set_start_method({start_method!r})\n"
        f"    shards = index_builder.Indexer(n_workers=2, max_memory_gb=1e-9).index_file({str(source)!r}, {str(tmp_path / 'shards')!r})\n"
        f"    index_builder.Merger().merge(shards, {str(tmp_path / 'index')!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert_index_matches_source(tmp_path / "index", records)


def test_single_worker_avoids_process_pool_and_errors_are_not_success(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Single-worker extraction must not start a pool")

    monkeypatch.setattr(index_builder, "ProcessPoolExecutor", forbidden)
    stream = iter([{"id": "a", "text": "one two three four five six seven eight"}])
    assert len(list(index_builder.Indexer()._get_batched_fingerprints(stream, None))) == 1
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "word " * 30}) + "\nnot JSON\n")
    with pytest.raises(json.JSONDecodeError):
        index_builder.Indexer().index_file(str(source), str(tmp_path / "bad-shards"))
    assert not list((tmp_path / "bad-shards").glob("shard-*"))


def test_parallel_fingerprint_errors_reach_the_caller():
    stream = iter([{"id": "bad", "text": 123}])
    with pytest.raises(TypeError):
        list(index_builder.Indexer(n_workers=2)._get_batched_fingerprints(stream, None))


def write_shard(path, meta):
    """Create a shard directory holding only meta.json: enough for validation tests,
    since Merger.merge checks the metadata before touching any array."""
    path.mkdir()
    (path / "meta.json").write_text(json.dumps(meta))


def test_merge_rejects_unknown_and_mixed_settings_before_writing(tmp_path):
    invalid = {**Winnower(length=4).to_metadata(), "hash_method": "unknown"}
    invalid_shard = tmp_path / "invalid"
    write_shard(invalid_shard, invalid)
    with pytest.raises(ValueError, match="Unsupported"):
        index_builder.Merger().merge([str(invalid_shard)], str(tmp_path / "merged"))
    assert not (tmp_path / "merged").exists()
    with pytest.raises(ValueError, match="Unsupported"):
        indexing.DiskBasedIndex(str(invalid_shard))
    with pytest.raises(ValueError, match="Unsupported"):
        index_builder.PostingBuffer(invalid)
    for different in ("base", "length"):
        configurations = [
            Winnower(length=4, base=1_000_003),
            Winnower(
                length=5 if different == "length" else 4,
                base=1_000_033 if different == "base" else 1_000_003,
            ),
        ]
        shards = [tmp_path / f"{different}-{i}" for i in range(2)]
        for shard, configured in zip(shards, configurations):
            write_shard(shard, configured.to_metadata())
        target = tmp_path / f"mixed-{different}"
        with pytest.raises(ValueError, match="Incompatible"):
            index_builder.Merger().merge(list(map(str, shards)), str(target))
        assert not target.exists()


@pytest.mark.parametrize("method", [None, "legacy31", "rolling64-v1"])
def test_old_hash_formats_require_rebuild_without_writing_outputs(tmp_path, method):
    metadata = Winnower(length=4).to_metadata()
    if method is None:
        del metadata["hash_method"]
    else:
        metadata["hash_method"] = method
    shard = tmp_path / "old"
    write_shard(shard, metadata)
    with pytest.raises(ValueError, match="Rebuild"):
        indexing.DiskBasedIndex(str(shard))
    with pytest.raises(ValueError, match="Rebuild"):
        index_builder.PostingBuffer(metadata)
    target = tmp_path / "output"
    with pytest.raises(ValueError, match="Rebuild"):
        index_builder.Merger().merge([str(shard)], str(target))
    assert not target.exists()


def test_hash_method_selection_is_not_available_in_api_or_cli(tmp_path):
    with pytest.raises(TypeError, match="hash_method"):
        index_builder.Indexer(hash_method="legacy31")
    completed = subprocess.run(
        [
            sys.executable, "-m", "findmytext.index_builder", "index", "unused.jsonl",
            str(tmp_path / "shards"), "--hash-method", "legacy31",
        ],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode != 0 and "unrecognized arguments" in completed.stderr
    assert not (tmp_path / "shards").exists()


def test_partitioned_merge_matches_source_and_accepts_merged_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=5)
    shards = index_builder.Indexer(max_memory_gb=1e-9).index_file(str(source), str(tmp_path / "shards"))
    assert len(shards) == len(records)
    # A tiny memory budget forces many fingerprint ranges, most of them empty.
    merged = tmp_path / "merged"
    index_builder.Merger(n_workers=2, max_memory_gb=1e-6).merge(shards, str(merged))
    reader = assert_index_matches_source(merged, records)
    assert not (merged / "_parts").exists()
    assert reader.get_metadata()["hash_method"] == HASH_METHOD
    # A merged index is itself a valid input for a further merge.
    extra_source = tmp_path / "extra.jsonl"
    extra = [{"id": "extra-doc", "text": " ".join(f"extraword{i}" for i in range(80))}]
    extra_source.write_text(json.dumps(extra[0]) + "\n")
    extra_shards = index_builder.Indexer().index_file(str(extra_source), str(tmp_path / "extra-shards"))
    # merge_indexes: a parent directory or a list of directories (str or PathLike),
    # returning the opened index; the engine's merge_dir is still available.
    again = index_builder.merge_indexes(tmp_path / "shards", tmp_path / "again")
    assert isinstance(again, indexing.DiskBasedIndex) and again.num_documents == len(records)
    assert_index_matches_source(tmp_path / "again", records)
    index_builder.Merger().merge_dir(str(tmp_path / "shards"), str(tmp_path / "again-engine"))
    assert_index_matches_source(tmp_path / "again-engine", records)
    final = index_builder.merge_indexes([merged] + extra_shards, str(tmp_path / "final"), n_workers=2)
    assert final.num_documents == len(records) + 1
    assert_index_matches_source(tmp_path / "final", records + extra)
    with pytest.raises(ValueError, match="already holds an index"):
        index_builder.merge_indexes(tmp_path / "shards", tmp_path / "final")
    index_builder.merge_indexes([merged], tmp_path / "final", delete_existing=True)
    assert_index_matches_source(tmp_path / "final", records)
    with pytest.raises(ValueError, match="does not hold an index"):
        index_builder.merge_indexes([merged, tmp_path / "shards"], tmp_path / "y")
    with pytest.raises(ValueError, match="No index shards"):
        index_builder.merge_indexes(str(tmp_path / "shards" / "shard-000000"), str(tmp_path / "x"))


def test_build_index_produces_one_complete_index_and_feeds_detectors(tmp_path, monkeypatch):
    import findmytext

    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=4)
    # Several shards (one per document), merged inside the index directory.
    index = findmytext.build_index(str(source), str(tmp_path / "index"), n_workers=2, max_memory_gb=1e-9)
    assert isinstance(index, indexing.DiskBasedIndex) and index.index_dir == str(tmp_path / "index")
    assert index.num_documents == len(records) == 4
    assert index.num_fingerprints == len(index.fingerprints) == len(index.offsets) > 0
    assert_index_matches_source(tmp_path / "index", records)
    assert sorted(os.listdir(tmp_path / "index")) == sorted(
        ["meta.json", "fingerprints.npy", "offsets.npy", "lengths.npy", "postings.dat",
         "doc_name_offsets.npy", "doc_name_bytes.npy", "fingerprint_blocks.npy"]
    )
    # A single shard is moved into place, with the same layout.
    single = findmytext.build_index(str(source), str(tmp_path / "single"))
    assert sorted(os.listdir(tmp_path / "single")) == sorted(os.listdir(tmp_path / "index"))
    assert_index_matches_source(tmp_path / "single", records)
    # The opened index goes straight into a detector, and so does its path.
    for detector in (findmytext.FingerprintChainDetector(index),
                     findmytext.NbSharedFingerprintsDetector(str(tmp_path / "index"), min_fingerprints=1)):
        assert detector.index_dir == str(tmp_path / "index")
        assert detector.search(records[0]["text"], min_fragment_length=100)[records[0]["id"]] > 0.5
    assert findmytext.FingerprintChainDetector(index).index is index
    # n_workers becomes the posting-read thread count when a path is given, and
    # is ignored (with a warning) when the index is already open.
    threaded = findmytext.NbSharedFingerprintsDetector(str(tmp_path / "index"), n_workers=2)
    assert threaded.n_workers == 2 and threaded.index._io_pool._max_workers == 2
    assert findmytext.FingerprintChainDetector(str(tmp_path / "index")).index._io_pool is None
    with pytest.warns(UserWarning, match="n_workers is ignored"):
        assert findmytext.FingerprintChainDetector(index, n_workers=2).index is index
    with pytest.raises(ValueError, match="n_workers"):
        findmytext.FingerprintChainDetector(str(tmp_path / "index"), n_workers=0)
    with pytest.raises(ValueError, match="already holds an index"):
        findmytext.build_index(str(source), str(tmp_path / "index"))
    findmytext.build_index(str(source), str(tmp_path / "index"), stop_after=1, delete_existing=True)
    assert len(indexing.DiskBasedIndex(str(tmp_path / "index")).to_external_doc_id) == 1


def test_partition_bounds_balance_postings_of_skewed_fingerprints(tmp_path, monkeypatch):
    """Winnowed fingerprints crowd the low end of the 64-bit space; the ranges must
    follow the postings, not the value space, or the first worker gets nearly all."""
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    rng = np.random.default_rng(1)
    meta = Winnower().to_metadata()
    shards, expected = [], {}
    for s in range(3):
        buffer = index_builder.PostingBuffer(meta)
        for d in range(4):
            # Minimum of six draws, like winnowing with window 6; a few repeated values.
            values = rng.integers(0, 2**64, (200, 6), dtype=np.uint64).min(axis=1)
            values[:5] = np.arange(5, dtype=np.uint64)
            pairs = [(int(v), p) for p, v in enumerate(values)]
            buffer.add_fingerprints(f"doc-{s}-{d}", pairs)
            for v, p in pairs:
                expected.setdefault(v, []).append((f"doc-{s}-{d}", p))
        shards.append(str(tmp_path / f"shard-{s}"))
        buffer.save(shards[-1])
    total = sum(len(v) for v in expected.values())
    # A budget of 250 postings per worker share: 2400 postings -> 10 ranges.
    merger = index_builder.Merger(n_workers=2, max_memory_gb=2 * 250 * index_builder._BYTES_PER_POSTING / 2**30)
    bounds = merger._partition_bounds(shards, total)
    assert bounds[0] == 0 and bounds[-1] == 2**64 and bounds == sorted(set(bounds))
    assert len(bounds) - 1 >= 8
    counts = [
        sum(len(expected[v]) for v in expected if lo <= v < hi)
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]
    assert sum(counts) == total
    assert max(counts) <= 2 * total / (len(bounds) - 1)
    assert all(hi - lo < 2**64 // 4 for lo, hi in zip(bounds[:-2], bounds[1:-1]))  # low values are dense
    merged = tmp_path / "merged"
    merger.merge(shards, str(merged))
    reader = indexing.DiskBasedIndex(str(merged))
    assert list(map(int, reader.fingerprints)) == sorted(expected)
    postings = reader._get_postings(reader.fingerprints)
    for value, entries in expected.items():
        actual = [(reader.to_external_doc_id[int(r["doc_id"])], int(r["position"])) for r in postings[np.uint64(value)]]
        assert sorted(actual) == sorted(entries)


def test_merge_fails_promptly_when_a_worker_is_killed(tmp_path):
    """A worker killed by the kernel (OOM) must surface as an error, not a hang."""
    source = tmp_path / "corpus.jsonl"
    make_corpus(source, count=2)
    # Run from a file so that spawn/forkserver children can import `killed` from __main__.
    script = tmp_path / "kill_worker.py"
    script.write_text(
        "import os, signal\n"
        "from findmytext import index_builder\n"
        "def killed(task):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "if __name__ == '__main__':\n"
        f"    shards = index_builder.Indexer(max_memory_gb=1e-9).index_file({str(source)!r}, {str(tmp_path / 'shards')!r})\n"
        "    index_builder._merge_partition = killed\n"
        "    try:\n"
        f"        index_builder.Merger(n_workers=2).merge(shards, {str(tmp_path / 'index')!r})\n"
        "    except RuntimeError as e:\n"
        "        print('CAUGHT', e)\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    # Prepend rather than replace: the inherited PYTHONPATH may provide numpy etc.
    pythonpath = os.pathsep.join(p for p in (str(repo_root), os.environ.get("PYTHONPATH", "")) if p)
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=repo_root,
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CAUGHT" in completed.stdout and "died unexpectedly" in completed.stdout
    assert not (tmp_path / "index" / "meta.json").exists()


def test_text_without_fingerprints_can_be_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    target = tmp_path / "empty-fingerprints"
    buffer = index_builder.PostingBuffer(Winnower().to_metadata())
    buffer.add_fingerprints("short", [])
    buffer.save(str(target))
    reader = indexing.DiskBasedIndex(str(target))
    assert not len(reader.fingerprints)
    assert list(reader.to_external_doc_id) == ["short"]


def test_shard_directory_is_a_complete_index_across_buffer_growth(tmp_path, monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=3)
    shards = index_builder.Indexer().index_file(str(source), str(tmp_path / "shards"))
    assert shards == [str(tmp_path / "shards" / "shard-000000")]
    reader = assert_index_matches_source(shards[0], records)
    assert reader.get_metadata()["hash_method"] == HASH_METHOD
    # Postings for one fingerprint keep insertion (document) order.
    for value in reader.fingerprints:
        docs = reader._get_postings(np.array([value]))[value]["doc_id"]
        assert list(docs) == sorted(docs)


def test_two_level_search_matches_searchsorted_and_touches_only_needed_blocks():
    array = np.unique(np.random.default_rng(0).integers(0, 2**64, 10_000, dtype=np.uint64))
    search = indexing._TwoLevelSearch(array, array[::64], block=64)
    queries = np.concatenate([
        array[::7], array[::64], array[63::64],  # present values, incl. block edges
        array[::7] + np.uint64(1), array[::64] - np.uint64(1),  # just above / below
        np.array([0, 2**64 - 1], dtype=np.uint64),  # beyond both ends
    ])
    np.testing.assert_array_equal(search(queries), np.searchsorted(array, queries))
    empty = np.zeros(0, dtype=np.uint64)
    assert len(indexing._TwoLevelSearch(empty, empty)(queries)) == len(queries)

    class Counting(np.ndarray):
        reads = 0

        def __getitem__(self, item):
            if isinstance(item, slice):
                Counting.reads += 1
            return np.ndarray.__getitem__(self, item)

    counted = indexing._TwoLevelSearch(array.view(Counting), array[::64], block=64)
    assert counted(array[:3]).tolist() == [0, 1, 2]
    assert Counting.reads == 1  # all three queries fall in the first block


def test_fingerprint_blocks_are_written_by_the_builder_and_required_by_the_reader(tmp_path):
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=3)
    index_dir = tmp_path / "index"
    index_builder.build_index(str(source), str(index_dir))
    blocks_path = index_dir / indexing.FINGERPRINT_BLOCKS_FILE
    fingerprints = np.load(index_dir / "fingerprints.npy")
    expected = fingerprints[:: indexing.FINGERPRINT_BLOCK]
    np.testing.assert_array_equal(np.load(blocks_path), expected)

    # The reader does not derive the file itself: it is loaded like every other array.
    blocks_path.unlink()
    with pytest.raises(FileNotFoundError):
        indexing.DiskBasedIndex(str(index_dir))

    # The builder provides the repair path for an index directory lacking the file.
    index_builder.write_fingerprint_blocks(str(index_dir))
    np.testing.assert_array_equal(np.load(blocks_path), expected)
    reader = indexing.DiskBasedIndex(str(index_dir))
    np.testing.assert_array_equal(reader._lookup.block_starts, expected)
    assert_index_matches_source(index_dir, records)

    # Sequential chunked computation equals the strided sample for any block size.
    array = np.sort(np.random.default_rng(1).integers(0, 2**64, 5_000, dtype=np.uint64))
    for block in (1, 7, 64, indexing.FINGERPRINT_BLOCK):
        np.testing.assert_array_equal(
            index_builder.compute_fingerprint_blocks(array, block), array[::block]
        )
    assert len(index_builder.compute_fingerprint_blocks(array[:0])) == 0


def test_default_posting_reads_avoid_pool_and_equal_explicit_threaded_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=3)
    shards = index_builder.Indexer().index_file(str(source), str(tmp_path / "shards"))
    target = tmp_path / "index"
    index_builder.Merger().merge(shards, str(target))
    threaded = indexing.DiskBasedIndex(str(target), n_workers=8)
    queries, _ = Winnower.from_metadata(threaded.get_metadata()).get_winnowed_fingerprints(
        records[0]["text"]
    )
    queries = np.r_[queries[::-1], queries[:2], np.uint64(2**64 - 1)]

    def forbidden(*args, **kwargs):
        raise AssertionError("Default posting reads must not create a thread pool")

    monkeypatch.setattr(indexing, "ThreadPoolExecutor", forbidden)
    sequential = indexing.DiskBasedIndex(str(target))
    for only_doc_ids in (False, True):
        expected = threaded._get_postings(queries, only_doc_ids=only_doc_ids)
        actual = sequential._get_postings(queries, only_doc_ids=only_doc_ids)
        assert list(actual) == list(expected)
        for fingerprint in expected:
            np.testing.assert_array_equal(actual[fingerprint], expected[fingerprint])
    assert sequential.get_match_counts(queries) == threaded.get_match_counts(queries)
    assert sequential._get_postings(np.empty(0, dtype=np.uint64)) == {}
    with pytest.raises(ValueError, match="at least 1"):
        indexing.DiskBasedIndex(str(target), n_workers=0)


def test_resource_defaults_come_from_available_cpus_and_memory(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        Process=lambda: types.SimpleNamespace(cpu_affinity=lambda: [0, 2, 5]),
        cpu_count=lambda: 64,
        virtual_memory=lambda: types.SimpleNamespace(available=10 * 2**30),
    )
    monkeypatch.setattr(index_builder, "psutil", fake)
    for cls in (index_builder.Indexer, index_builder.Merger):
        auto = cls()
        assert auto.n_workers == 3 and auto.max_memory_gb == pytest.approx(8.0)
        explicit = cls(n_workers=2, max_memory_gb=0.5)
        assert explicit.n_workers == 2 and explicit.max_memory_gb == 0.5
    # Without an affinity mask (macOS), the logical CPU count is used instead.
    fake.Process = lambda: types.SimpleNamespace()
    assert index_builder.Indexer(max_memory_gb=1.0).n_workers == 64
    merger = index_builder.Merger()
    assert merger.n_workers == 64 and merger.max_memory_gb == pytest.approx(8.0)
    for bad in ({"n_workers": 0}, {"max_memory_gb": 0}):
        with pytest.raises(ValueError, match="must be positive"):
            index_builder.Indexer(**bad)
        with pytest.raises(ValueError, match="must be positive"):
            index_builder.Merger(**bad)
