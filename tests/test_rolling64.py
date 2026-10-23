import json
import gzip

import msgpack
import numpy as np
import polars as pl
import pytest

from findmytext import detectors, index_builder, indexing, similarity, web
from findmytext.multi_document import MultiDocumentFingerprintChainAnalyzer
from findmytext.winnower import (
    ROLLING64_HASH, ROLLING_BASE64, Winnower, winnow,
)


def test_rolling64_is_the_only_default_and_legacy_metadata_requires_rebuild():
    legacy_meta = {"length": 4, "window_size": 6, "base": 256, "punctuation": False}
    with pytest.raises(ValueError, match="Rebuild"):
        Winnower.from_metadata(legacy_meta)
    with pytest.raises(ValueError, match="Rebuild"):
        indexing.MemoryBasedIndex(legacy_meta)
    wide = Winnower(length=4)
    assert wide.hash_method == ROLLING64_HASH
    assert wide.base == ROLLING_BASE64
    assert Winnower.from_metadata(wide.to_metadata()).to_metadata() == wide.to_metadata()
    for invalid in ("legacy31", "rolling32-v1", "unknown", "rolling64-v2"):
        with pytest.raises(ValueError, match="Unsupported hash method"):
            Winnower(hash_method=invalid)
        with pytest.raises(ValueError, match="Rebuild"):
            Winnower.from_metadata({**legacy_meta, "hash_method": invalid})
    for base in (0, 1, 256, 2**64, -3):
        with pytest.raises(ValueError, match="odd base"):
            Winnower(hash_method=ROLLING64_HASH, base=base)


def test_production_hashes_deterministic_with_rightmost_ties_and_empty_input():
    configured = Winnower(length=4, window_size=6)
    text = "bl\u00e5 \u00e6 \U0001f600 punctuation, repeated " * 30
    first = configured.get_winnowed_fingerprints(text)
    second = configured.get_winnowed_fingerprints(text)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert first[0].dtype == np.uint64
    assert len(configured.get_winnowed_fingerprints("")[0]) == 0
    assert len(configured.get_winnowed_fingerprints("too short")[0]) == 0
    values = np.array([2**63 + 9] * 10, dtype=np.uint64)
    hashes, positions = winnow(values, 3)
    assert list(map(int, hashes)) == [2**63 + 9] * 8
    assert list(positions) == list(range(2, 10))


@pytest.mark.parametrize("layout", ["interleaved", "columnar"])
def test_build_merge_query_and_detectors_preserve_high_bits(tmp_path, layout):
    method = ROLLING64_HASH
    text = " ".join(f"source_token_{i}" for i in range(150))
    records = [{"id": "a", "text": text}, {"id": "b", "text": text},
               {"id": "c", "text": " ".join(f"other_token_{i}" for i in range(150))}]
    source = tmp_path / "corpus.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records))
    shards = index_builder.index_file(
        str(source), str(tmp_path / "shards"), length=4,
        window_size=6, nb_workers=2, max_nb_documents_before_flush=2,
    )
    target = tmp_path / "index"
    index_builder.merge_indexes(shards, str(target), postings_layout=layout)
    index = indexing.DiskBasedIndex(str(target))
    assert index.winnower.hash_method == method
    assert json.loads((target / "meta.json").read_text())["hash_method"] == method
    hashes, _ = index.winnower.get_winnowed_fingerprints(text)
    assert any(int(value) > 2**63 for value in hashes)
    for internal in range(3):
        doc_id = index.to_external_doc_id[internal]
        record = next(row for row in records if row["id"] == doc_id)
        expected, _ = index.winnower.get_winnowed_fingerprints(record["text"])
        assert np.array_equal(index.get_document_fingerprints(internal), np.unique(expected))
    similarity.compute_similarities_inverted(str(target), verbose=False)
    pairs = index.load_similarities()
    expected_count = len(np.unique(hashes))
    a = list(index.to_external_doc_id).index("a")
    b = list(index.to_external_doc_id).index("b")
    stored = pairs[(pairs["doc_i"] == min(a, b)) & (pairs["doc_j"] == max(a, b))]
    assert int(stored[0]["count"]) == expected_count

    for detector_type in (detectors.NbSharedFingerprintsDetector, detectors.FingerprintChainDetector):
        detector = detector_type(str(target), min_fingerprints=1)
        assert detector.runtime_winnower.hash_method == method
        assert detector.get_fingerprints(text).schema["hash"] == pl.UInt64
        scores = detector.get_containment_scores(text, min_fragment_length=100)
        assert scores["a"] > 0 and scores["b"] > 0
    website = web.TextContainmentDetector(str(target), min_fingerprints=1)
    assert website.runtime_winnower.hash_method == method
    assert website.find_matches_jaccard(text, score="rarity")["a"] > 0


def test_merge_refuses_different_rolling_bases_before_writing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    shards = []
    for base in (ROLLING_BASE64, 1_000_003):
        configured = Winnower(length=4, base=base)
        memory = indexing.MemoryBasedIndex(configured.to_metadata())
        hashes, positions = configured.get_winnowed_fingerprints("one two three four five six " * 5)
        memory.add_fingerprints("doc", list(zip(hashes, positions)))
        shard = tmp_path / f"{base}.mpk.gz"
        memory.to_msgpack(str(shard))
        shards.append(str(shard))
    target = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Incompatible fingerprint metadata"):
        index_builder.merge_indexes(shards, str(target))
    assert not target.exists()


def test_legacy_disk_indexes_and_shards_are_rejected_without_writes(tmp_path):
    meta = {"length": 4, "window_size": 6, "base": 256, "punctuation": False}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Rebuild"):
        indexing.DiskBasedIndex(str(tmp_path))
    shard = tmp_path / "legacy.mpk.gz"
    with gzip.open(shard, "wb") as output:
        output.write(msgpack.packb(meta))
    target = tmp_path / "merged"
    with pytest.raises(ValueError, match="Rebuild"):
        index_builder.merge_indexes([str(shard)], str(target))
    assert not target.exists()
    for name in ("fingerprints", "offsets", "lengths"):
        np.save(tmp_path / f"{name}.npy", np.array([], dtype=np.uint64))
    with pytest.raises(ValueError, match="Rebuild"):
        index_builder.build_forward_index(str(tmp_path))
    assert not (tmp_path / "forward_offsets.npy").exists()


def test_multi_document_analysis_accepts_unsigned_hashes():
    text = " ".join(f"source_token_{i}" for i in range(120))
    analyzer = MultiDocumentFingerprintChainAnalyzer(
        winnower=Winnower(length=4, hash_method=ROLLING64_HASH), min_cluster_size=2,
    )
    result = analyzer.analyze({"a": text, "b": text})
    assert result.passages
    assert result.points.schema["hash"] == pl.UInt64
