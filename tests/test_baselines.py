import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"fmt_{name}", _EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baselines = _load("baselines")


def write_fbin(path, vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(vectors.shape, dtype=np.int32).tofile(f)
        vectors.tofile(f)


@pytest.fixture
def db_dir(tmp_path):
    """A database of 6 orthogonal-ish document vectors, doc id = row index."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 8)).astype(np.float32)
    vectors[0] *= 10  # not normalised on disk; search must still rank by cosine
    write_fbin(tmp_path / baselines.FBIN_NAME, vectors)
    baselines.write_meta(str(tmp_path), repo_id="test/repo", model_name="fake-model", n_docs=6, dim=8)
    return tmp_path, vectors


def test_read_fbin_returns_header_shape_and_values(db_dir):
    path, vectors = db_dir
    read = baselines.read_fbin(str(path / baselines.FBIN_NAME))
    assert read.shape == (6, 8)
    np.testing.assert_array_equal(np.asarray(read), vectors)


def test_read_fbin_rejects_truncated_or_inconsistent_files(tmp_path):
    path = tmp_path / "base.fbin"
    write_fbin(path, np.zeros((4, 3)))
    assert baselines._is_valid_fbin(str(path))
    with open(path, "r+b") as f:
        f.truncate(8 + 4 * 3 * 4 - 4)
    assert not baselines._is_valid_fbin(str(path))
    with pytest.raises(ValueError, match="header announces 4 x 3"):
        baselines.read_fbin(str(path))
    assert not baselines._is_valid_fbin(str(tmp_path / "missing.fbin"))


class FakeModel:
    """Maps a text ``"doc<i>"`` to the stored vector i (plus noise), and anything else
    to a direction orthogonal to all stored vectors (6 vectors in 8 dimensions)."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.unrelated = np.linalg.svd(vectors)[2][-1]
        self.max_seq_length = None

    def encode(self, texts, **kwargs):
        out = []
        for text in texts:
            if text.startswith("doc"):
                v = self.vectors[int(text[3:])] + 0.01
            else:
                v = self.unrelated
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


@pytest.fixture
def detector(db_dir, monkeypatch):
    pytest.importorskip("faiss")
    path, vectors = db_dir
    monkeypatch.setattr(
        baselines.DenseRetrievalDetector,
        "_load_model",
        lambda self, device, max_seq_length: FakeModel(vectors),
    )
    return baselines.DenseRetrievalDetector(str(path), top_k=3)


def test_search_returns_closest_doc_ids_by_cosine(detector):
    assert detector.model_name == "fake-model"  # taken from meta.json
    assert detector.n_docs == 6 and detector.dim == 8
    for doc_id in range(6):
        result = detector.search(f"doc{doc_id}")
        assert len(result) == 3
        assert all(isinstance(k, int) and isinstance(v, float) for k, v in result.items())
        best = max(result, key=result.get)
        assert best == doc_id
        assert result[best] == pytest.approx(1.0, abs=1e-3)
    # The signature of the fingerprint detectors is accepted.
    assert detector.search("doc2", 1000.0) == detector.search("doc2")


def test_top_k_is_capped_by_database_size(db_dir, monkeypatch):
    pytest.importorskip("faiss")
    path, vectors = db_dir
    monkeypatch.setattr(
        baselines.DenseRetrievalDetector,
        "_load_model",
        lambda self, device, max_seq_length: FakeModel(vectors),
    )
    detector = baselines.DenseRetrievalDetector(str(path), top_k=100)
    assert len(detector.search("doc0")) == 6


def test_instruction_wraps_query_in_qwen_prompt(detector, monkeypatch):
    seen = []

    def fake_encode(texts, **kwargs):
        seen.extend(texts)
        return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(detector.model, "encode", fake_encode)
    detector.search("hello")
    detector.instruction = "Find the source article"
    detector.search("hello")
    assert seen == ["hello", "Instruct: Find the source article\nQuery: hello"]


def test_build_database_renames_download_and_writes_meta(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    calls = []

    def fake_download(repo_id, filename, repo_type, local_dir):
        calls.append(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_fbin(dest, vectors)
        return str(dest)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    db = tmp_path / "db"
    meta_path = baselines.DenseRetrievalDetector.build_database(
        str(db), repo_id="test/repo", model_name="some-model", verify=False
    )
    meta = json.loads(Path(meta_path).read_text())
    assert meta == {
        "repo_id": "test/repo", "model_name": "some-model", "n_docs": 3, "dim": 4,
        "source": "fbin/base.fbin",
    }
    assert (db / baselines.FBIN_NAME).is_file() and not (db / "fbin").exists()
    np.testing.assert_array_equal(np.asarray(baselines.read_fbin(str(db / baselines.FBIN_NAME))), vectors)

    # A second build reuses the complete file instead of downloading again.
    baselines.DenseRetrievalDetector.build_database(str(db), repo_id="test/repo", verify=False)
    assert calls == ["fbin/base.fbin"]


def test_dense_detector_is_accepted_by_evaluator(tmp_path, detector):
    pytest.importorskip("sklearn")
    from findmytext import detectors as fmt_detectors, index_builder

    fmt_eval = _load("eval")
    docs = [{"id": str(i), "text": " ".join(f"w{i}_{j}" for j in range(150))} for i in range(6)]
    (tmp_path / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    index = index_builder.build_index(str(tmp_path / "corpus.jsonl"), str(tmp_path / "index"), n_workers=1)

    # Positives query "doc<i>" -> dense detector retrieves doc i; negatives do not match.
    examples = [{"text": f"doc{i}", "type": "positive", "source_doc_id": i} for i in range(6)]
    examples += [{"text": f"unrelated text {i}", "type": "negative"} for i in range(6)]
    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))

    evaluator = fmt_eval.Evaluator(
        index,
        detectors={"dense": detector, "baseline": fmt_detectors.NbSharedFingerprintsDetector(index)},
    )
    assert evaluator._separate == ["dense"] and evaluator._shared == ["baseline"]
    metrics = evaluator.run(str(tmp_path / "eval.jsonl"), verbose=False)
    dense = metrics["dense"]
    assert dense["top1_acc"] == 1.0
    assert dense["roc_auc"] == 1.0  # positives score ~1, negatives ~0 (orthogonal query)
    assert np.isnan(dense["mean_retrieval_time"])
