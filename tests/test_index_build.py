import gzip
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys

import msgpack
import numpy as np
import polars as pl
import pytest

from findmytext import detectors, index_builder, indexing, web
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
    frame = detectors.convert_closest_matches_with_positions_to_df({"document": {2**63 + 9: [3]}})
    assert frame["hash"].dtype == pl.UInt64 and frame["hash"][0] == 2**63 + 9


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
    expected = {}
    for row in records:
        values, positions = reader.winnower.get_winnowed_fingerprints(row["text"])
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
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source)
    shards = index_builder.index_file(
        str(source), str(tmp_path / "shards"), nb_workers=1,
        max_nb_documents_before_flush=2,
    )
    assert len(shards) == len(set(shards)) == 4
    target = tmp_path / "index"
    index_builder.merge_indexes(shards, str(target))
    reader = assert_index_matches_source(target, records)
    assert reader.winnower.to_metadata()["hash_method"] == HASH_METHOD
    assert not list(target.glob("forward*"))
    query = records[0]["text"]
    for cls in (detectors.NbSharedFingerprintsDetector, detectors.FingerprintChainDetector):
        for rarity in (False, True):
            detector = cls(str(target), min_fingerprints=1, rarity_weighted=rarity)
            assert detector.get_fingerprints(query).schema["hash"] == pl.UInt64
            scores = detector.get_containment_scores(query, min_fragment_length=100)
            assert scores[records[0]["id"]] > 0
    website = web.TextContainmentDetector(str(target), min_fingerprints=1)
    assert website.find_matches_jaccard(query, score="rarity")[records[0]["id"]] > 0
    assert website.find_matches_clustering(query)[records[0]["id"]] > 0


@pytest.mark.parametrize("start_method", [name for name in ("spawn", "forkserver") if name in mp.get_all_start_methods()])
def test_parallel_build_accepts_generator_with_modern_start_methods(tmp_path, start_method):
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=4)
    script = (
        "import multiprocessing as mp\n"
        "from findmytext import index_builder\n"
        "if __name__ == '__main__':\n"
        f"    mp.set_start_method({start_method!r})\n"
        f"    shards = index_builder.index_file({str(source)!r}, {str(tmp_path / 'shards')!r}, nb_workers=2, max_nb_documents_before_flush=2)\n"
        f"    index_builder.merge_indexes(shards, {str(tmp_path / 'index')!r})\n"
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

    monkeypatch.setattr(index_builder.mp, "Pool", forbidden)
    stream = iter([{"id": "a", "text": "one two three four five six seven eight"}])
    assert len(list(index_builder.get_batched_fingerprints(stream, None, 1, 4, 6))) == 1
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "word " * 30}) + "\nnot JSON\n")
    with pytest.raises(json.JSONDecodeError):
        index_builder.index_file(str(source), str(tmp_path / "bad-shards"), nb_workers=1)
    assert not list((tmp_path / "bad-shards").glob("*.mpk.gz"))


def test_parallel_fingerprint_errors_reach_the_caller():
    stream = iter([{"id": "bad", "text": 123}])
    with pytest.raises(TypeError):
        list(index_builder.get_batched_fingerprints(stream, None, 2, 4, 6))


def write_shard(path, meta, records=()):
    with gzip.open(path, "wb") as stream:
        stream.write(msgpack.packb(meta))
        for record in records:
            stream.write(msgpack.packb(record))


def test_merge_rejects_unknown_and_mixed_settings_before_writing(tmp_path):
    invalid = {**Winnower(length=4).to_metadata(), "hash_method": "unknown"}
    invalid_shard = tmp_path / "invalid.mpk.gz"
    write_shard(invalid_shard, invalid)
    with pytest.raises(ValueError, match="Unsupported"):
        index_builder.merge_indexes([str(invalid_shard)], str(tmp_path / "merged"))
    assert not (tmp_path / "merged").exists()
    (tmp_path / "meta.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="Unsupported"):
        indexing.DiskBasedIndex(str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported"):
        indexing.MemoryBasedIndex(invalid)
    shards = [tmp_path / f"part{i}.mpk.gz" for i in range(2)]
    for different in ("base", "length"):
        configurations = [
            Winnower(length=4, base=1_000_003),
            Winnower(
                length=5 if different == "length" else 4,
                base=1_000_033 if different == "base" else 1_000_003,
            ),
        ]
        for shard, configured in zip(shards, configurations):
            write_shard(shard, configured.to_metadata())
        target = tmp_path / f"mixed-{different}"
        with pytest.raises(ValueError, match="Incompatible"):
            index_builder.merge_indexes(list(map(str, shards)), str(target))
        assert not target.exists()


@pytest.mark.parametrize("method", [None, "legacy31", "rolling64-v1"])
def test_old_hash_formats_require_rebuild_without_writing_outputs(tmp_path, method):
    metadata = Winnower(length=4).to_metadata()
    if method is None:
        del metadata["hash_method"]
    else:
        metadata["hash_method"] = method
    (tmp_path / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Rebuild"):
        indexing.DiskBasedIndex(str(tmp_path))
    with pytest.raises(ValueError, match="Rebuild"):
        indexing.MemoryBasedIndex(metadata)
    shard = tmp_path / "old.mpk.gz"
    write_shard(shard, metadata)
    target = tmp_path / "output"
    with pytest.raises(ValueError, match="Rebuild"):
        index_builder.merge_indexes([str(shard)], str(target))
    assert not target.exists()


def test_hash_method_selection_is_not_available_in_api_or_cli(tmp_path):
    with pytest.raises(TypeError, match="hash_method"):
        index_builder.index_file("unused.jsonl", str(tmp_path / "shards"), hash_method="legacy31")
    completed = subprocess.run(
        [
            sys.executable, "-m", "findmytext.index_builder", "index", "unused.jsonl",
            str(tmp_path / "shards"), "--hash-method", "legacy31",
        ],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode != 0 and "unrecognized arguments" in completed.stderr
    assert not (tmp_path / "shards").exists()


def test_truncated_merge_reader_raises_instead_of_hanging_or_finishing(tmp_path):
    path = tmp_path / "broken.mpk.gz"
    write_shard(path, Winnower(length=4).to_metadata(), [1, "doc", 123, 1])
    with pytest.raises(ValueError, match="Truncated"):
        list(index_builder.merge_streams_parallel([str(path)]))
    with pytest.raises(RuntimeError, match="reader failed"):
        list(index_builder.merge_streams_parallel([str(path), str(path)], queue_depth=1))


def test_text_without_fingerprints_can_be_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    path = tmp_path / "empty-fingerprints.mpk.gz"
    memory = indexing.MemoryBasedIndex(Winnower().to_metadata())
    memory.add_fingerprints("short", [])
    memory.to_msgpack(str(path))
    target = tmp_path / "index"
    index_builder.merge_indexes([str(path)], str(target))
    reader = indexing.DiskBasedIndex(str(target))
    assert not len(reader.fingerprints)
    assert list(reader.to_external_doc_id) == ["short"]


def test_buffered_msgpack_roundtrip_crosses_compression_buffer_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    memory = indexing.MemoryBasedIndex(Winnower().to_metadata())
    values = np.arange(2**63, 2**63 + 80_000, dtype=np.uint64)
    memory.add_fingerprints("source", list(zip(values.tolist(), range(len(values)))))
    path = tmp_path / "large.mpk.gz"
    memory.to_msgpack(str(path))
    restored = indexing.MemoryBasedIndex.from_msgpack(str(path))
    assert restored.meta == memory.meta
    assert restored.to_external == ["source"]
    np.testing.assert_array_equal(
        restored.index_array[:restored.index_size], memory.index_array[:memory.index_size],
    )


def test_default_posting_reads_avoid_pool_and_equal_explicit_threaded_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    records = make_corpus(source, count=3)
    shards = index_builder.index_file(str(source), str(tmp_path / "shards"), nb_workers=1)
    target = tmp_path / "index"
    index_builder.merge_indexes(shards, str(target))
    threaded = indexing.DiskBasedIndex(str(target), posting_read_threads=8)
    queries, _ = threaded.winnower.get_winnowed_fingerprints(records[0]["text"])
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
        indexing.DiskBasedIndex(str(target), posting_read_threads=0)


def test_posting_read_benchmark_compares_both_modes_without_writing_index(tmp_path, monkeypatch):
    from experiments.benchmark_posting_reads import benchmark

    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    source = tmp_path / "corpus.jsonl"
    make_corpus(source, count=3)
    shards = index_builder.index_file(str(source), str(tmp_path / "shards"), nb_workers=1)
    target = tmp_path / "index"
    index_builder.merge_indexes(shards, str(target))
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    output = tmp_path / "reads.json"
    report = benchmark(target, source, output, documents=3, repeats=2)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert json.loads(output.read_text()) == report
    for mode in ("positional_postings", "candidate_counts_top50"):
        runs = [row for row in report["runs"] if row["mode"] == mode]
        assert {row["threads"] for row in runs} == {1, 8}
        assert len({row["ordered_results_sha256"] for row in runs}) == 1
        assert all(row["all_queries_identical"] and row["queries_verified"] == 3 for row in runs)
    with pytest.raises(ValueError, match="outside"):
        benchmark(target, source, target / "invalid.json", documents=3, repeats=1)
