import json
import sys

import numpy as np
import pytest
import zstandard as zstd

import hplt_build
import hplt_nynorsk
from findmytext import index_builder
from findmytext.indexing import DiskBasedIndex
from findmytext.overlap_inverted import OverlapInvertedIndex, _source_identity
from experiments.indexing.verify_overlap_inverted import verify_indexes


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "hplt_nynorsk" / "nno_Latn"
    root.mkdir(parents=True)
    shared = " ".join(f"sharedword{i}" for i in range(50))
    # Exactly two accepted records per shard: extraction flushes without
    # producing final.mpk.gz, which must still be resumable.
    shards = [
        [{"id": "a", "text": shared}, {"id": "b", "text": shared + " suffix"},
         {"id": "short", "text": "too short"}],
        [{"id": "c", "text": "prefix " + shared}, {"id": "no-fps", "text": "a" * 150}],
    ]
    for i, records in enumerate(shards):
        data = "".join(json.dumps(record) + "\n" for record in records).encode()
        (root / f"{i}.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(data))
    return root


@pytest.mark.parametrize("workers", [1, 2])
def test_local_cli_build_and_resume(corpus, tmp_path, monkeypatch, capsys, workers):
    monkeypatch.setattr(hplt_build, "ROOT_PATH", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "hplt_build.py", "nno_Latn", "hplt_nynorsk", "--skip-download",
        "--overlap-inverted", "--nb-workers", str(workers), "--batch-entries", "3",
        "--max-documents-before-flush", "2",
    ])
    monkeypatch.setattr(hplt_build, "download", lambda *args: pytest.fail("Must not download"))
    hplt_build.main()
    build = corpus.parent / "findmytext_build"
    index = DiskBasedIndex(str(build / "index"))
    sidecar = OverlapInvertedIndex(build / "overlap_inverted", build / "index")
    assert index.winnower.hash_method == "rolling64-v1"
    assert index.winnower.length == 4
    assert index.winnower.window_size == 6
    assert len(index.to_external_doc_id) == 4
    assert len(sidecar.eligible_lengths) == 4
    assert len(sidecar.fingerprints) > 0
    assert any(len(index.get_document_fingerprints(i)) == 0 for i in range(4))
    assert all(len(sidecar.get_postings(i)) >= 2 for i in range(len(sidecar.fingerprints)))
    manifest = json.loads((build / "build.json").read_text())
    assert (manifest["min_length"], manifest["max_length"]) == (100, 100_000)
    assert len(list((build / "shards").glob("*/complete.json"))) == 2
    assert not list((build / "shards").glob("*/final.mpk.gz"))
    report = verify_indexes(
        build / "index", build / "overlap_inverted", corpus_dir=corpus,
        batch_fingerprints=3, verbose=False,
    )
    assert report["source_text_verification"]["status"] == "passed"
    assert len(report["source_text_verification"]["documents"]) == 4
    monkeypatch.setattr(index_builder, "index_file", lambda *a, **k: pytest.fail("Must reuse index"))
    hplt_build.main()
    output = capsys.readouterr().out
    assert "Using existing index" in output
    assert "Using existing overlap inverted index" in output

    # Removing only the publication (not the shard checkpoints) must reuse
    # both completed shards even though neither has a final.mpk.gz file.
    (build / "index").rename(tmp_path / "preserved-index")
    hplt_build.build_index(corpus, build, 1, 2)
    rebuilt = DiskBasedIndex(str(build / "index"))
    assert np.array_equal(index.fingerprints, rebuilt.fingerprints)
    assert index.forward_fingerprints is not None
    assert rebuilt.forward_fingerprints is not None
    assert np.array_equal(index.forward_fingerprints, rebuilt.forward_fingerprints)


def test_interrupted_merge_does_not_publish_and_can_resume(corpus, tmp_path, monkeypatch):
    build = tmp_path / "build"
    merge = index_builder.merge_indexes

    def fail_merge(files, output):
        from pathlib import Path

        path = Path(output)
        path.mkdir()
        (path / "meta.json").write_text("{}")
        raise RuntimeError("Interrupted merge")

    monkeypatch.setattr(index_builder, "merge_indexes", fail_merge)
    with pytest.raises(RuntimeError, match="Interrupted merge"):
        hplt_build.build_index(corpus, build, 1, 2)
    assert not (build / "index").exists()
    assert not list(build.glob(".merge-*"))
    monkeypatch.setattr(index_builder, "merge_indexes", merge)
    monkeypatch.setattr(index_builder, "index_file", lambda *a, **k: pytest.fail("Must reuse shards"))
    assert hplt_build.build_index(corpus, build, 1, 2).is_dir()


def test_changed_corpus_and_legacy_paths_are_not_reused(corpus, tmp_path):
    legacy = tmp_path / "findmytext_build_31_bit"
    legacy.mkdir()
    (legacy / "meta.json").write_text('{"base": 256}')
    with pytest.raises(ValueError, match="Untracked/legacy"):
        hplt_build.build_index(corpus, legacy, 1)
    assert (legacy / "meta.json").read_text() == '{"base": 256}'
    build = tmp_path / "build"
    hplt_build.build_index(corpus, build, 1, 2)
    (corpus / "0.jsonl.zst").touch()
    with pytest.raises(ValueError, match="Source files or settings changed"):
        hplt_build.build_index(corpus, build, 1, 2)


def test_nynorsk_build_routes_to_local_rebuild_and_sidecar(corpus, monkeypatch):
    build = corpus.parent / "findmytext_build"
    monkeypatch.setattr(hplt_nynorsk, "DATA_PATH", corpus.parent)
    monkeypatch.setattr(hplt_nynorsk, "BUILD_PATH", build)
    monkeypatch.setattr(sys, "argv", [
        "hplt_nynorsk.py", "--build", "--nb-workers", "1",
        "--batch-entries", "3", "--max-documents-before-flush", "2",
    ])
    monkeypatch.setattr(hplt_nynorsk, "inspect_metadata", lambda *a: pytest.fail("Must not inspect"))
    hplt_nynorsk.main()
    assert OverlapInvertedIndex(build / "overlap_inverted", build / "index").meta["document_count"] == 4


@pytest.mark.parametrize("option", ["--nb-workers", "--batch-entries", "--max-documents-before-flush"])
def test_invalid_cli_limits_fail_before_download_or_build(option, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hplt_build.py", "nno_Latn", "hplt_nynorsk", option, "0"])
    monkeypatch.setattr(hplt_build, "download", lambda *a: pytest.fail("Must not download"))
    monkeypatch.setattr(hplt_build, "build_index", lambda *a: pytest.fail("Must not build"))
    with pytest.raises(SystemExit) as exc:
        hplt_build.main()
    assert exc.value.code == 2


def test_changed_completed_shard_is_rejected(corpus, tmp_path, monkeypatch):
    build = tmp_path / "build"

    def stop_merge(*args):
        raise RuntimeError("stop")

    monkeypatch.setattr(index_builder, "merge_indexes", stop_merge)
    with pytest.raises(RuntimeError, match="stop"):
        hplt_build.build_index(corpus, build, 1, 2)
    shard = next((build / "shards").glob("*/*.mpk.gz"))
    shard.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="Completed shard file is missing or changed"):
        hplt_build.build_index(corpus, build, 1, 2)


def test_source_text_verification_rejects_wrong_positions(corpus, tmp_path):
    build = tmp_path / "build"
    source = hplt_build.build_index(corpus, build, 1, 2)
    destination = hplt_build.build_overlap_inverted(source, build, 3)
    postings = np.memmap(
        source / "postings.dat", mode="r+",
        dtype=[("doc_id", np.uint32), ("position", np.uint32)],
    )
    postings[0]["position"] += 1
    postings.flush()
    meta_path = destination / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["source"] = _source_identity(source)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    with pytest.raises(ValueError, match="differs from source text"):
        verify_indexes(source, destination, corpus_dir=corpus, verbose=False)
