import hashlib
import itertools
import json
from collections import Counter

import numpy as np
import pytest

from findmytext import index_builder, similarity, winnower
from findmytext import overlap_inverted
from findmytext.indexing import DiskBasedIndex, POSTING_DTYPE
from findmytext.overlap_inverted import OverlapInvertedIndex, build_overlap_inverted_index
from experiments.indexing.verify_overlap_inverted import verify_indexes


def write_source(root, lists, n_docs=5):
    root.mkdir()
    meta = winnower.Winnower(length=4, window_size=6).to_metadata()
    meta["postings_layout"] = "interleaved"
    (root / "meta.json").write_text(json.dumps(meta))
    np.save(root / "fingerprints.npy", np.array(list(lists), dtype=np.uint64))
    lengths = np.array([len(docs) for docs in lists.values()], dtype=np.uint32)
    np.save(root / "lengths.npy", lengths)
    np.save(root / "offsets.npy", np.r_[0, np.cumsum(lengths[:-1], dtype=np.uint64) * 8].astype(np.uint64) if len(lengths) else np.empty(0, dtype=np.uint64))
    records = [(doc, pos) for docs in lists.values() for pos, doc in enumerate(docs)]
    np.array(records, dtype=POSTING_DTYPE).tofile(root / "postings.dat")
    names = [f"doc-{i}".encode() for i in range(n_docs)]
    np.save(root / "doc_name_offsets.npy", np.r_[0, np.cumsum([len(n) for n in names])].astype(np.uint64))
    np.save(root / "doc_name_bytes.npy", np.frombuffer(b"".join(names), dtype=np.uint8))
    return root


def snapshot(root):
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in root.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("batch_entries", [1, 3, 100])
def test_sidecar_preserves_exact_overlap_and_full_indexes(tmp_path, batch_entries, capsys):
    high = 2**63 + 17
    source = write_source(
        tmp_path / "index",
        {1: [0, 0, 0], 3: [2, 0, 2], 5: [3, 0, 2, 3],
         high: [1, 0, 1], 2**64 - 1: [2]},
    )
    index_builder.build_forward_index(str(source), batch_entries=3)
    before = snapshot(source)
    destination = tmp_path / "overlap_inverted"
    build_overlap_inverted_index(source, destination, batch_entries=batch_entries)
    assert snapshot(source) == before
    sidecar = OverlapInvertedIndex(destination, source)
    assert sidecar.fingerprints.tolist() == [3, high, 5]
    assert sidecar.offsets.tolist() == [0, 2, 4, 7]
    assert sidecar.eligible_lengths.tolist() == [3, 1, 2, 1, 0]
    assert [sidecar.get_postings(i).tolist() for i in range(3)] == [[0, 2], [0, 1], [0, 2, 3]]
    counts = Counter()
    for i in range(3):
        counts.update(itertools.combinations(sidecar.get_postings(i).tolist(), 2))
    similarity.compute_similarities_inverted(str(source), verbose=False)
    reference = similarity.similarities_to_dict(
        DiskBasedIndex(str(source)).load_similarities("inverted")
    )
    assert counts == reference
    for threshold in (1, 2, 3):
        assert {p for p, c in counts.items() if c >= threshold} == {
            p for p, c in reference.items() if c >= threshold
        }
    original = DiskBasedIndex(str(source))
    assert 1 in original.get_document_fingerprints(0)
    assert 2**64 - 1 in original.get_document_fingerprints(2)
    for doc in range(5):
        values = original.get_document_fingerprints(doc)
        assert np.all(values[1:] > values[:-1])
    stats = sidecar.meta["statistics"]
    assert stats["input_fingerprints"] == 5
    assert stats["retained_fingerprints"] == 3
    assert stats["singleton_fingerprints"] == 2
    assert stats["singleton_percentage"] == 40
    assert stats["input_positional_postings"] == 14
    assert stats["unique_doc_fingerprint_pairs"] == 9
    assert stats["collapsed_repeats"] == 5
    assert stats["retained_doc_id_postings"] == 7
    assert stats["documents_without_retained_fingerprints"] == 1
    assert stats["retained_document_frequency"]["quantiles"]["0.5"] == 2
    assert stats["eligible_lengths"]["min"] == 0
    assert sidecar.meta["storage_bytes"]["total"] == sum(
        p.stat().st_size for p in destination.iterdir()
    )
    output = capsys.readouterr().out
    for phase in ("scanning", "frequency ranking", "posting-list layout/write", "complete"):
        assert phase in output
    assert not list(tmp_path.glob(".overlap-inverted-*"))
    with pytest.raises(IndexError):
        sidecar.get_postings(-1)


@pytest.mark.parametrize("n_docs,lists", [(0, {}), (3, {}), (3, {5: [0, 0], 6: [2]})])
def test_empty_or_singleton_only_sources(tmp_path, n_docs, lists):
    source = write_source(tmp_path / "index", lists, n_docs)
    destination = build_overlap_inverted_index(source, tmp_path / "overlap_inverted", batch_entries=1)
    sidecar = OverlapInvertedIndex(destination, source)
    assert sidecar.fingerprints.tolist() == []
    assert sidecar.offsets.tolist() == [0]
    assert sidecar.doc_ids.tolist() == []
    assert sidecar.eligible_lengths.tolist() == [0] * n_docs
    assert sidecar.meta["statistics"]["retained_document_frequency"]["count"] == 0


def test_hot_fingerprint_and_random_unsorted_occurrences(tmp_path):
    rng = np.random.default_rng(17)
    lists = {i: rng.integers(0, 25, size=int(rng.integers(1, 200))).tolist() for i in range(1, 50)}
    source = write_source(tmp_path / "index", lists, 27)
    sidecar = OverlapInvertedIndex(
        build_overlap_inverted_index(source, tmp_path / "overlap_inverted", batch_entries=3),
        source,
    )
    expected = sorted(
        ((len(set(docs)), fp, sorted(set(docs))) for fp, docs in lists.items() if len(set(docs)) >= 2)
    )
    assert sidecar.fingerprints.tolist() == [fp for _, fp, _ in expected]
    for rank, (_, _, docs) in enumerate(expected):
        assert sidecar.get_postings(rank).tolist() == docs
    assert int(sidecar.eligible_lengths.sum()) == len(sidecar.doc_ids)


@pytest.mark.parametrize("corruption", ["zero", "doc", "offset", "truncated", "extra", "unsorted", "layout", "legacy"])
def test_bad_source_is_rejected_without_publishing(tmp_path, corruption):
    source = write_source(tmp_path / "index", {3: [0, 0, 1], 5: [2, 1]})
    if corruption == "zero":
        np.save(source / "lengths.npy", np.array([0, 2], dtype=np.uint32))
    elif corruption == "doc":
        postings = np.memmap(source / "postings.dat", mode="r+", dtype=POSTING_DTYPE)
        postings[0]["doc_id"] = 5
        postings.flush()
    elif corruption == "offset":
        np.save(source / "offsets.npy", np.array([8, 24], dtype=np.uint64))
    elif corruption == "truncated":
        (source / "postings.dat").write_bytes(b"bad")
    elif corruption == "extra":
        with (source / "postings.dat").open("ab") as stream:
            stream.write(b"\0" * POSTING_DTYPE.itemsize)
    elif corruption == "unsorted":
        np.save(source / "fingerprints.npy", np.array([5, 3], dtype=np.uint64))
    else:
        meta = json.loads((source / "meta.json").read_text())
        if corruption == "layout":
            meta["postings_layout"] = "columnar"
        else:
            del meta["hash_method"]
        (source / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        build_overlap_inverted_index(source, tmp_path / "overlap_inverted", batch_entries=1)
    assert not (tmp_path / "overlap_inverted").exists()
    assert not list(tmp_path.glob(".overlap-inverted-*"))


def test_existing_destination_source_binding_and_bad_arguments(tmp_path):
    source = write_source(tmp_path / "index", {3: [0, 1]})
    destination = build_overlap_inverted_index(source, tmp_path / "overlap_inverted")
    with pytest.raises(FileExistsError):
        build_overlap_inverted_index(source, destination)
    with pytest.raises(ValueError, match="outside"):
        build_overlap_inverted_index(source, source / "overlap_inverted")
    with pytest.raises(ValueError, match="positive"):
        build_overlap_inverted_index(source, tmp_path / "other", batch_entries=0)
    other_source = write_source(tmp_path / "other-index", {3: [0, 1]})
    with pytest.raises(ValueError, match="does not match"):
        OverlapInvertedIndex(destination, other_source)
    (source / "doc_name_bytes.npy").touch()
    with pytest.raises(ValueError, match="does not match"):
        OverlapInvertedIndex(destination, source)


def test_source_change_during_build_is_rejected(tmp_path, monkeypatch):
    source = write_source(tmp_path / "index", {3: [0, 1]})
    rank_runs = overlap_inverted._rank_runs

    def change_source(*args):
        result = rank_runs(*args)
        (source / "doc_name_bytes.npy").touch()
        return result

    monkeypatch.setattr(overlap_inverted, "_rank_runs", change_source)
    with pytest.raises(ValueError, match="changed during"):
        build_overlap_inverted_index(source, tmp_path / "overlap_inverted")
    assert not (tmp_path / "overlap_inverted").exists()
    assert not list(tmp_path.glob(".overlap-inverted-*"))


def test_full_verification_and_exact_sampled_counts(tmp_path):
    source = write_source(
        tmp_path / "index",
        {1: [0, 0], 3: [2, 0, 2], 5: [3, 0, 2, 3],
         2**63 + 17: [1, 0, 1], 2**64 - 1: [2]},
    )
    index_builder.build_forward_index(str(source), batch_entries=3)
    destination = build_overlap_inverted_index(source, tmp_path / "overlap_inverted", batch_entries=3)
    report_path = tmp_path / "verification.json"
    before = snapshot(source)
    report = verify_indexes(
        source, destination, sample_pairs=100, batch_fingerprints=1,
        report_path=report_path, verbose=False,
    )
    assert report["status"] == "passed"
    assert report["source_unchanged"]
    assert report["fingerprints_at_or_above_2_pow_63"] == 2
    assert report["full_forward_entries"] == 9
    assert len(report["sampled_exact_pairs"]) == 10
    expected = {(0, 1): 1, (0, 2): 2, (0, 3): 1, (2, 3): 1}
    assert {
        (row["doc_i"], row["doc_j"]): row["count"]
        for row in report["sampled_exact_pairs"] if row["count"]
    } == expected
    assert json.loads(report_path.read_text()) == report
    assert snapshot(source) == before
    assert not list(tmp_path.glob(".verify-overlap-*"))


@pytest.mark.parametrize("n_docs,lists", [(0, {}), (3, {}), (3, {1: [0, 0], 3: [2]})])
def test_verification_without_retained_fingerprints(tmp_path, n_docs, lists):
    source = write_source(tmp_path / "index", lists, n_docs)
    index_builder.build_forward_index(str(source), batch_entries=3)
    destination = build_overlap_inverted_index(source, tmp_path / "overlap_inverted")
    report = verify_indexes(source, destination, verbose=False)
    assert report["status"] == "passed"
    assert all(row["count"] == 0 for row in report["sampled_exact_pairs"])


@pytest.mark.parametrize("corruption", ["forward_order", "forward_membership", "overlap_docs", "eligible_lengths"])
def test_full_verification_rejects_inconsistent_indexes(tmp_path, corruption):
    source = write_source(tmp_path / "index", {3: [0, 2], 5: [0, 1], 9: [0]})
    index_builder.build_forward_index(str(source))
    destination = build_overlap_inverted_index(source, tmp_path / "overlap_inverted")
    if corruption.startswith("forward"):
        data = np.load(source / "forward_fingerprints.npy", mmap_mode="r+")
        data[1] = data[0] if corruption == "forward_order" else 6
        data.flush()
        meta_path = destination / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["source"] = overlap_inverted._source_identity(source)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    else:
        name = "doc_ids.npy" if corruption == "overlap_docs" else "eligible_lengths.npy"
        data = np.load(destination / name, mmap_mode="r+")
        data[1] = 0
        data.flush()
    with pytest.raises(ValueError):
        verify_indexes(source, destination, batch_fingerprints=1, verbose=False)


@pytest.mark.parametrize("failure", [None, "layout", "merge"])
def test_scratch_mmaps_are_closed_before_cleanup(tmp_path, monkeypatch, failure):
    source = write_source(tmp_path / "index", {3: [0, 2], 5: [0, 1], 9: [1, 2]})
    maps = []
    layout = overlap_inverted._layout
    merge = overlap_inverted._merge

    def capture_layout(*args):
        maps.extend(array for array in args if isinstance(array, np.memmap))
        if failure == "layout":
            raise RuntimeError("layout failed")
        return layout(*args)

    def capture_merge(*args):
        maps.extend(array for array in args if isinstance(array, np.memmap))
        if failure == "merge":
            raise RuntimeError("merge failed")
        return merge(*args)

    monkeypatch.setattr(overlap_inverted, "_layout", capture_layout)
    monkeypatch.setattr(overlap_inverted, "_merge", capture_merge)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            build_overlap_inverted_index(source, tmp_path / "overlap_inverted", batch_entries=1)
    else:
        build_overlap_inverted_index(source, tmp_path / "overlap_inverted", batch_entries=1)
    assert maps
    assert all(array._mmap.closed for array in maps)
    assert not list(tmp_path.glob(".overlap-inverted-*"))
