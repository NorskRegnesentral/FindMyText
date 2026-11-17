import json
import numpy as np
import pytest
import zstandard

from findmytext import document_store
from findmytext.document_store import CorpusDocumentStore, build_document_store
from hplt_build import build_index


def write_records(directory, records):
    directory.mkdir()
    data = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
    (directory / "sample.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(data))


@pytest.fixture
def records():
    return [
        {
            "id": "alpha", "text": " ".join(f"alphaword{i}" for i in range(40)),
            "xml": "<doc>all original XML</doc>", "u": "https://example.org/a",
            "ts": "2018-01-02T03:04:05Z", "doc_scores": [1.5, 2],
            "web-register": {"NA": 0.75}, "seg_langs": ["nno_Latn"],
            "nested": {"null": None, "list": [True, "æøå"]},
        },
        {
            "id": "βeta", "text": " ".join(f"betaword{i}" for i in range(40)),
            "u": "https://example.org/b", "ts": "2017-01-02T03:04:05Z",
        },
        {"id": "not-indexed", "text": "tiny", "metadata": {"keep": "this"}},
    ]


def test_complete_records_metadata_and_internal_ids_roundtrip(tmp_path, records):
    corpus = tmp_path / "corpus"
    write_records(corpus, records)
    index = build_index(corpus, tmp_path / "build", 1, 2)
    database = build_document_store(corpus, tmp_path / "documents.sqlite", index_dir=index, batch_records=1)
    before = database.stat().st_mtime_ns
    with CorpusDocumentStore(database, index_dir=index) as store:
        assert store.meta["records"] == 3
        assert store.meta["indexed_document_ids_checked"] == 2
        actual = store.get_many(["βeta", "alpha", "βeta", "not-indexed"])
        assert list(actual) == ["βeta", "alpha", "not-indexed"]
        assert actual == {record["id"]: record for record in records}
        metadata = store.get_many(["alpha"], metadata_only=True)["alpha"]
        assert "text" not in metadata and "xml" not in metadata
        assert metadata["web-register"] == records[0]["web-register"]
        assert metadata["doc_scores"] == records[0]["doc_scores"]
        assert metadata["seg_langs"] == records[0]["seg_langs"]
        indexed = store.get_many([np.uint32(0), 1], id_kind="internal")
        assert indexed[0] == records[0] and indexed[1] == records[1]
        assert store.get_many([]) == {}
        with pytest.raises(KeyError, match="missing"):
            store.get_many(["alpha", "missing"])
        with pytest.raises(ValueError, match="External"):
            store.get_many([0])
        with pytest.raises(ValueError, match="out of range"):
            store.get_many([True], id_kind="internal")
        with pytest.raises(ValueError, match="out of range"):
            store.get_many([1, True], id_kind="internal")
        with pytest.raises(ValueError, match="out of range"):
            store.get_many([2], id_kind="internal")
    assert build_document_store(corpus, database, index_dir=index) == database
    assert database.stat().st_mtime_ns == before
    with CorpusDocumentStore(database) as store:
        with pytest.raises(ValueError, match="matching index_dir"):
            store.get_many([0], id_kind="internal")


def test_resume_replays_only_unfinished_shard_and_preserves_metadata(tmp_path, records, monkeypatch):
    corpus = tmp_path / "corpus"
    write_records(corpus, records)
    output = tmp_path / "documents.sqlite"
    real = document_store.zstd.ZstdCompressor

    class Interrupted:
        def __init__(self, **kwargs):
            self.compressor = real(**kwargs)
            self.calls = 0

        def compress(self, data):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("interrupted after first committed record")
            return self.compressor.compress(data)

    monkeypatch.setattr(document_store.zstd, "ZstdCompressor", Interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_document_store(corpus, output, batch_records=1)
    assert not output.exists()
    assert output.with_suffix(".sqlite.building").exists()
    monkeypatch.setattr(document_store.zstd, "ZstdCompressor", real)
    build_document_store(corpus, output, batch_records=1)
    with CorpusDocumentStore(output) as store:
        assert store.get_many([record["id"] for record in records]) == {record["id"]: record for record in records}
    assert not output.with_suffix(".sqlite.building").exists()


def test_duplicate_raw_ids_and_source_changes_are_explicit(tmp_path, records):
    duplicate = tmp_path / "duplicates"
    write_records(duplicate, [records[0], {**records[0], "u": "different provenance"}])
    with pytest.raises(ValueError, match="Duplicate"):
        build_document_store(duplicate, tmp_path / "duplicates.sqlite", batch_records=1)
    corpus = tmp_path / "corpus"
    write_records(corpus, records)
    output = build_document_store(corpus, tmp_path / "documents.sqlite")
    (corpus / "sample.jsonl.zst").touch()
    with pytest.raises(ValueError, match="different source"):
        build_document_store(corpus, output)


def test_many_ids_are_batched_beyond_sqlite_parameter_limit(tmp_path):
    corpus = tmp_path / "corpus"
    records = [{"id": f"d{i}", "text": f"text {i}", "metadata": i} for i in range(1200)]
    write_records(corpus, records)
    output = build_document_store(corpus, tmp_path / "documents.sqlite")
    with CorpusDocumentStore(output) as store:
        result = store.get_many([record["id"] for record in reversed(records)])
    assert len(result) == len(records)
    assert list(result)[:2] == ["d1199", "d1198"]
    assert result["d601"] == records[601]


@pytest.mark.parametrize("record", [
    {"id": "", "text": "text"}, {"id": 1, "text": "text"},
    {"id": "valid", "text": None},
])
def test_invalid_raw_fields_prevent_publication(tmp_path, record):
    corpus = tmp_path / "corpus"
    write_records(corpus, [record])
    output = tmp_path / "documents.sqlite"
    with pytest.raises(ValueError, match="Expected raw string id/text"):
        build_document_store(corpus, output)
    assert not output.exists()


def test_index_binding_rejects_a_changed_snapshot(tmp_path, records):
    corpus = tmp_path / "corpus"
    write_records(corpus, records)
    index = build_index(corpus, tmp_path / "build", 1, 2)
    output = build_document_store(corpus, tmp_path / "documents.sqlite", index_dir=index)
    (index / "doc_name_bytes.npy").touch()
    with pytest.raises(ValueError, match="snapshots do not match"):
        CorpusDocumentStore(output, index_dir=index)


def test_corrupt_json_reports_source_line(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "broken.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(b"not json\n"))
    output = tmp_path / "documents.sqlite"
    with pytest.raises(ValueError, match="Invalid JSON in broken.jsonl.zst:1"):
        build_document_store(corpus, output)
    assert not output.exists()
