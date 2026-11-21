"""Tests for the read side of the index: findmytext.indexing.DiskBasedIndex.

Indexes are built from hand-picked (fingerprint, position) pairs through
index_builder.PostingBuffer, so every expected posting is known exactly without
depending on the hashing; one test goes end to end through the Winnower.
"""

import os

import numpy as np
import polars as pl
import pytest

from findmytext import index_builder, indexing
from findmytext.winnower import Winnower

U64_MAX = 2**64 - 1


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)


def build_index(path, docs, meta=None):
    """Write `docs` ({doc_name: [(fingerprint, position), ...]}, in insertion order)
    as an index directory and open it."""
    buffer = index_builder.PostingBuffer(meta or Winnower().to_metadata())
    for name, postings in docs.items():
        buffer.add_fingerprints(name, postings)
    buffer.save(str(path))
    return indexing.DiskBasedIndex(str(path))


def u64(*values):
    return np.array(values, dtype=np.uint64)


# Four documents whose shared fingerprints 1-4 are all in the query below: A shares
# four distinct fingerprints (one of them at two positions), B two, C one, D none.
RANKED_DOCS = {
    "A": [(1, 0), (2, 1), (3, 2), (4, 3), (4, 8)],
    "B": [(1, 0), (2, 1), (99, 2)],
    "C": [(3, 0), (98, 1)],
    "D": [(97, 0)],
}
RANKED_QUERY = u64(1, 2, 3, 4, 5)


def test_reader_exposes_metadata_names_and_exact_postings(tmp_path):
    meta = Winnower(length=3, window_size=4, base=1_000_003, punctuation=True).to_metadata()
    docs = {
        "blå/東京": [(10, 0), (20, 1), (U64_MAX, 2)],
        "": [(20, 5), (2**63, 6)],
        42: [(0, 3), (20, 4), (20, 9)],
    }
    reader = build_index(tmp_path / "index", docs, meta)

    assert reader.get_metadata() == meta
    assert len(reader.to_external_doc_id) == 3
    assert [reader.to_external_doc_id[i] for i in range(3)] == ["blå/東京", "", "42"]
    assert list(reader.to_external_doc_id) == ["blå/東京", "", "42"]

    assert reader.fingerprints.dtype == np.uint64
    assert reader.fingerprints.tolist() == [0, 10, 20, 2**63, U64_MAX]
    assert reader.lengths.tolist() == [1, 1, 4, 1, 1]
    assert reader.offsets.tolist() == [0, 8, 16, 48, 56]  # 8 bytes per posting

    # Duplicates and an absent value in the query; result keys are the present values.
    query = u64(20, 7, U64_MAX, 20, 0, 2**63)
    postings = reader._get_postings(query)
    assert sorted(int(key) for key in postings) == [0, 20, 2**63, U64_MAX]
    assert all(array.dtype == indexing.POSTING_DTYPE for array in postings.values())
    # Postings of one fingerprint keep insertion order: (doc_id, position).
    assert postings[np.uint64(20)].tolist() == [(0, 1), (1, 5), (2, 4), (2, 9)]
    assert postings[np.uint64(0)].tolist() == [(2, 3)]
    assert postings[np.uint64(2**63)].tolist() == [(1, 6)]
    assert postings[np.uint64(U64_MAX)].tolist() == [(0, 2)]

    doc_ids = reader._get_postings(query, only_doc_ids=True)
    assert doc_ids[np.uint64(20)].tolist() == [0, 1, 2, 2]
    assert doc_ids[np.uint64(U64_MAX)].tolist() == [0]

    verbose = reader._get_postings(query, verbose=True)
    assert {int(k): v.tolist() for k, v in verbose.items()} == {
        int(k): v.tolist() for k, v in postings.items()
    }


def test_match_counts_count_distinct_fingerprints_and_rank_documents(tmp_path):
    reader = build_index(tmp_path / "index", RANKED_DOCS)

    counts = reader.get_match_counts(RANKED_QUERY, min_fingerprints=1, top_k=5)
    assert list(counts.items()) == [("A", 4), ("B", 2), ("C", 1)]
    assert reader.get_closest_matches(RANKED_QUERY, min_fingerprints=1, top_k=5) == ["A", "B", "C"]

    assert reader.get_match_counts(RANKED_QUERY, min_fingerprints=2, top_k=5) == {"A": 4, "B": 2}
    assert reader.get_match_counts(RANKED_QUERY, min_fingerprints=1, top_k=1) == {"A": 4}
    assert reader.get_closest_matches(RANKED_QUERY, min_fingerprints=1, top_k=2) == ["A", "B"]
    # A has four distinct shared fingerprints, so the default threshold of 5 excludes it.
    assert reader.get_match_counts(RANKED_QUERY) == {}
    assert reader.get_match_counts(RANKED_QUERY, min_fingerprints=5, top_k=5) == {}
    assert reader.get_match_counts(np.empty(0, dtype=np.uint64), min_fingerprints=1) == {}
    assert reader.get_match_counts(u64(5, 6, U64_MAX), min_fingerprints=1) == {}
    assert reader.get_closest_matches(u64(5, 6), min_fingerprints=1) == []

    tied = reader.get_match_counts(u64(1, 2), min_fingerprints=1, top_k=5)
    assert tied == {"A": 2, "B": 2}
    top = reader.get_match_counts(u64(1, 2), min_fingerprints=1, top_k=1)
    assert len(top) == 1 and set(top) <= {"A", "B"}


def test_closest_matches_with_positions_report_each_documents_positions(tmp_path):
    reader = build_index(tmp_path / "index", RANKED_DOCS)

    matches = reader.get_closest_matches_with_positions(
        RANKED_QUERY, min_fingerprints=1, top_k=5, verbose=False
    )
    assert matches == {
        "A": {1: [0], 2: [1], 3: [2], 4: [3, 8]},
        "B": {1: [0], 2: [1]},
        "C": {3: [0]},
    }
    assert list(matches) == ["A", "B", "C"]
    assert all(type(fp) is int for doc in matches.values() for fp in doc)

    series = pl.Series(RANKED_QUERY)
    assert series.dtype == pl.UInt64
    assert reader.get_closest_matches_with_positions(series, min_fingerprints=1, top_k=5, verbose=False) == matches
    assert reader.get_closest_matches_with_positions(RANKED_QUERY, min_fingerprints=1, top_k=5, verbose=True) == matches

    with pytest.raises(ValueError, match="query must be"):
        reader.get_closest_matches_with_positions([1, 2, 3], min_fingerprints=1)

    assert reader.get_closest_matches_with_positions(RANKED_QUERY, min_fingerprints=1, top_k=1, verbose=False) == {
        "A": matches["A"]
    }
    assert reader.get_closest_matches_with_positions(RANKED_QUERY, min_fingerprints=3, verbose=False) == {
        "A": matches["A"]
    }
    assert reader.get_closest_matches_with_positions(RANKED_QUERY, verbose=False) == {}
    assert reader.get_closest_matches_with_positions(u64(5, 6), min_fingerprints=1, verbose=False) == {}
    assert reader.get_closest_matches_with_positions(np.empty(0, dtype=np.uint64), min_fingerprints=1) == {}


def test_lookups_across_many_blocks_and_posting_counts(tmp_path):
    rng = np.random.default_rng(1)
    # Strictly inside the range so that 0 and 2**64 - 1 are guaranteed absent.
    values = np.unique(rng.integers(1, U64_MAX, 3000, dtype=np.uint64))
    n = len(values)
    expected_counts = np.arange(n) % 3 + 1
    docs = {f"doc{d}": [] for d in range(5)}
    for i, value in enumerate(values):
        docs[f"doc{i % 5}"].extend((int(value), i + p) for p in range(expected_counts[i]))
    reader = build_index(tmp_path / "index", docs)

    assert len(reader.fingerprints) == n
    assert len(reader._lookup.block_starts) > 2  # queries span several 512-entry blocks
    assert reader.fingerprints.tolist() == values.tolist()

    np.testing.assert_array_equal(reader.get_posting_counts(values), expected_counts)
    shuffled = rng.permutation(n)
    np.testing.assert_array_equal(reader.get_posting_counts(values[shuffled]), expected_counts[shuffled])
    np.testing.assert_array_equal(reader.get_posting_counts(values.tolist()[:10]), expected_counts[:10])

    gaps = values[:-1] + np.uint64(1)
    absent = gaps[gaps != values[1:]]
    assert len(absent) > 0
    assert not reader.get_posting_counts(absent).any()
    assert reader.get_posting_counts(u64(0, U64_MAX)).tolist() == [0, 0]
    mixed = np.concatenate([values[shuffled[:50]], absent[:50]])
    mixed_counts = reader.get_posting_counts(mixed)
    np.testing.assert_array_equal(mixed_counts[:50], expected_counts[shuffled[:50]])
    assert not mixed_counts[50:].any()

    subset = shuffled[:300]
    postings = reader._get_postings(np.concatenate([values[subset], absent[:20]]))
    assert len(postings) == len(subset)
    for i in subset:
        records = postings[values[i]]
        assert records["doc_id"].tolist() == [i % 5] * expected_counts[i]
        assert records["position"].tolist() == list(range(i, i + expected_counts[i]))

    totals = reader.get_match_counts(values, min_fingerprints=1, top_k=5)
    assert totals == {f"doc{d}": len(range(d, n, 5)) for d in range(5)}


def test_index_without_fingerprints_answers_every_query_with_nothing(tmp_path):
    reader = build_index(tmp_path / "empty", {"lonely": []})
    assert len(reader.fingerprints) == 0
    assert list(reader.to_external_doc_id) == ["lonely"]

    query = u64(0, 7, U64_MAX)
    assert reader.get_posting_counts(query).tolist() == [0, 0, 0]
    assert reader.get_posting_counts(np.empty(0, dtype=np.uint64)).tolist() == []
    assert reader._get_postings(query) == {}
    assert reader.get_match_counts(query, min_fingerprints=1) == {}
    assert reader.get_closest_matches(query, min_fingerprints=1) == []
    assert reader.get_closest_matches_with_positions(query, min_fingerprints=1, verbose=False) == {}


def test_text_indexed_through_winnower_is_found_with_shifted_positions(tmp_path):
    winnower = Winnower(length=4, window_size=5)
    shared = " ".join(f"shared{i}" for i in range(40))
    filler = " ".join(f"filler{i}" for i in range(10))
    other = " ".join(f"other{i}" for i in range(40))
    texts = {"A": filler + " " + shared, "B": shared, "C": other}
    reader = build_index(
        tmp_path / "index", winnower.get_winnowed_fingerprints_batch(texts), winnower.to_metadata()
    )
    assert list(reader.to_external_doc_id) == ["A", "B", "C"]
    assert not hasattr(reader, "winnower")  # callers rebuild one from get_metadata()
    query_winnower = Winnower.from_metadata(reader.get_metadata())
    assert query_winnower.to_metadata() == winnower.to_metadata()

    query_fps, query_pos = query_winnower.get_winnowed_fingerprints(shared)
    n = len(query_fps)
    assert n >= 5 and len(np.unique(query_fps)) == n  # all tokens differ

    # Every winnowed k-gram of the query is a k-gram of B (identical) and of A
    # (shifted by the ten filler tokens); C shares nothing.
    assert reader.get_posting_counts(query_fps).tolist() == [2] * n
    assert reader.get_match_counts(query_fps, min_fingerprints=1) == {"A": n, "B": n}
    assert reader.get_closest_matches(query_fps, min_fingerprints=n) == ["A", "B"]
    matches = reader.get_closest_matches_with_positions(query_fps, min_fingerprints=1, verbose=False)
    assert matches == {
        "A": {int(fp): [int(pos) + 10] for fp, pos in zip(query_fps, query_pos)},
        "B": {int(fp): [int(pos)] for fp, pos in zip(query_fps, query_pos)},
    }

    other_fps, _ = query_winnower.get_winnowed_fingerprints(other)
    assert reader.get_closest_matches(other_fps, min_fingerprints=1) == ["C"]
    unrelated_fps, _ = query_winnower.get_winnowed_fingerprints(
        " ".join(f"nothing{i}" for i in range(40))
    )
    assert len(unrelated_fps) > 0
    assert reader.get_match_counts(unrelated_fps, min_fingerprints=1) == {}


def test_deleting_the_reader_closes_the_posting_file(tmp_path):
    reader = build_index(tmp_path / "index", {"a": [(1, 0)]})
    fd = reader._posting_fd
    os.fstat(fd)  # open while the reader is alive
    reader.__del__()
    assert reader._posting_fd == -1
    with pytest.raises(OSError):
        os.fstat(fd)
    reader.__del__()  # a second call must not raise
