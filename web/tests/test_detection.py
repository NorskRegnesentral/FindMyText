import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from findmytext import detectors, index_builder
from web.corpussearch import create_app
from web.corpussearch.detection import _ranking_from_scores


def test_web_adapter_scores_and_highlights_after_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(index_builder.PostingBuffer, "_CHUNK", 100)
    text = " ".join(f"sharedword{i}" for i in range(90))
    records = [
        {"id": "source", "text": text},
        {"id": "other", "text": " ".join(f"otherword{i}" for i in range(90))},
    ]
    source = tmp_path / "corpus.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    shards = index_builder.Indexer().index_file(str(source), str(tmp_path / "shards"))
    index = tmp_path / "index"
    index_builder.Merger().merge(shards, str(index))
    (tmp_path / "samples_data.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "corpora": [{"id": "fixture", "label": "Fixture", "index_dir": str(index)}],
    }))
    monkeypatch.setenv("FINDMYTEXT_CONFIG", str(config))
    app = create_app()
    app.testing = True
    client = app.test_client()
    response = client.post("/api/detect", json={
        "corpus": "fixture", "text": text,
        "algorithms": ["jaccard", "connected_components"],
        "min_fragment_length": 500,
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[-1]["type"] == "result", events
    assert events[-1]["min_fragment_length"] == 500
    for ranking in events[-1]["results"].values():
        assert ranking["top"]["doc_id"] == "source"
        assert 0 < ranking["top"]["score"] <= 1
    cfg = app.config["APP_CONFIG"]
    detector, _ = app.config["INDEX_MANAGER"].get(cfg.corpus("fixture"), cfg.params)
    baseline = detectors.NbSharedFingerprintsDetector(detector.index, top_k=cfg.params.top_k)
    chain = detectors.FingerprintChainDetector(
        detector.index, top_k=cfg.params.top_k,
        position_threshold=cfg.params.cc_doc1_position_threshold,
        offset_threshold=cfg.params.cc_offset_threshold,
        min_cluster_size=cfg.params.cc_min_cluster_size,
    )
    assert events[-1]["results"]["jaccard"]["top"]["score"] == baseline.search(text, 500)["source"]
    assert events[-1]["results"]["connected_components"]["top"]["score"] == chain.search(text, 500)["source"]
    for key, core in [("jaccard", baseline), ("connected_components", chain)]:
        candidates = core.retrieve(text, 500)
        assert events[-1]["results"][key]["top"]["raw_score"] == core.raw_scores(candidates)["source"]
        assert events[-1]["results"][key]["ranking"][0]["raw_score"] == core.raw_scores(candidates)["source"]

    shorter_text = text[:250]
    response = client.post("/api/detect", json={
        "corpus": "fixture", "text": shorter_text, "algorithms": ["connected_components"],
    })
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[-1]["type"] == "result", events
    assert events[-1]["min_fragment_length"] == len(shorter_text.strip())

    response = client.post("/api/detect", json={
        "corpus": "fixture", "text": text, "algorithms": ["connected_components"],
        "min_fragment_length": 1000,
    })
    longer_threshold = [json.loads(line) for line in response.data.splitlines()][-1]
    assert longer_threshold["results"]["connected_components"]["top"]["raw_score"] == (
        chain.raw_scores(chain.retrieve(text, 500))["source"]
    )

    partial = " ".join(text.split()[:35] + [f"unrelatedword{i}" for i in range(120)])
    partial_results = []
    for passage_length in (500, 1000):
        response = client.post("/api/detect", json={
            "corpus": "fixture", "text": partial,
            "algorithms": ["jaccard", "connected_components"],
            "min_fragment_length": passage_length,
        })
        result = [json.loads(line) for line in response.data.splitlines()][-1]
        assert result["type"] == "result", result
        partial_results.append(result["results"])
    for key in ("jaccard", "connected_components"):
        short_target = partial_results[0][key]["top"]
        long_target = partial_results[1][key]["top"]
        assert long_target["doc_id"] == "source"
        assert 0 < long_target["score"] < 0.01
        assert long_target["raw_score"] == short_target["raw_score"]
        assert long_target["score"] < short_target["score"]
    assert "source" not in chain.search(partial, 1000)

    with monkeypatch.context() as patch:
        patch.setattr(
            detectors.FingerprintChainDetector, "_normalise_scores",
            lambda self, scores, length: {doc: 0.0 for doc in scores},
        )
        response = client.post("/api/detect", json={
            "corpus": "fixture", "text": text,
            "algorithms": ["connected_components"], "min_fragment_length": 1000,
        })
        result = [json.loads(line) for line in response.data.splitlines()][-1]
        assert result["results"]["connected_components"]["top"]["score"] == 0.0
        assert result["results"]["connected_components"]["top"]["raw_score"] > 0

    assert detector.find_matches_jaccard(text, score="rarity")["source"] > 0
    assert detector.find_matches_clustering(text)["source"] > 0
    response = client.post("/api/highlight", json={
        "corpus": "fixture", "text": text, "doc_id": "source",
        "modes": ["cc_largest", "cc_all", "jaccard"],
    })
    assert response.status_code == 200
    layers = response.get_json()["layers"]
    assert {layer["mode"] for layer in layers} == {"cc_largest", "cc_all", "jaccard"}
    for layer in layers:
        spans = [span for cluster in layer["clusters"] for span in cluster["spans"]]
        assert spans
        assert all(0 <= start < end <= len(text) for start, end in spans)
        assert layer["matched_chars"] == sum(end - start for start, end in spans)

    candidates = baseline.retrieve(text, 500, min_fingerprints=cfg.params.min_fingerprints)
    block = candidates.hits.block(candidates.hits.doc_ids.index("source"))
    _, positions, _ = candidates.join(block)
    positions = positions.tolist()
    middle_position = next(pos for pos in positions if 35 <= pos < 55)
    words = text.split()
    middle = " ".join(words[middle_position:middle_position + detector.runtime_winnower.length])
    filler_a = " ".join(f"fillera{i}" for i in range(40))
    filler_b = " ".join(f"fillerb{i}" for i in range(40))
    scattered = f"😀 {' '.join(words[:30])} {filler_a} {middle} {filler_b} {' '.join(words[60:])}"
    middle_start = scattered.index(middle)
    response = client.post("/api/highlight", json={
        "corpus": "fixture", "text": scattered, "doc_id": "source",
        "modes": ["cc_largest", "cc_all", "jaccard"],
    })
    assert response.status_code == 200
    layers = {layer["mode"]: layer for layer in response.get_json()["layers"]}
    def covered(mode):
        return {
            pos for cluster in layers[mode]["clusters"]
            for start, end in cluster["spans"] for pos in range(start, end)
        }
    largest, all_chains, shared = (covered(mode) for mode in ("cc_largest", "cc_all", "jaccard"))
    assert largest <= all_chains <= shared
    assert middle_start in shared and middle_start not in all_chains
    first_position = min(pos for pos in positions if pos <= 25)
    assert scattered.index(words[first_position]) in shared
    assert 0 not in shared
    for invalid in [0, -1, 1.5, True, len(text) + 1]:
        response = client.post("/api/detect", json={
            "corpus": "fixture", "text": text, "algorithms": ["jaccard"],
            "min_fragment_length": invalid,
        })
        assert response.status_code == 400
    response = client.post("/api/detect", json={
        "corpus": "fixture", "text": "!!!", "algorithms": ["jaccard"],
        "min_fragment_length": 3,
    })
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[-1]["min_fragment_length"] == 3
    assert events[-1]["results"]["jaccard"] == {"top": None, "ranking": []}


def test_saturated_probability_ranking_uses_raw_fingerprint_count():
    assert _ranking_from_scores({"smaller": 1.0, "larger": 1.0}, {
        "smaller": 30, "larger": 50,
    }) == [("larger", 1.0), ("smaller", 1.0)]


def test_home_page_paper_link_configuration(tmp_path, monkeypatch):
    (tmp_path / "samples_data.json").write_text("{}")
    config = tmp_path / "config.json"
    monkeypatch.setenv("FINDMYTEXT_CONFIG", str(config))
    for settings, expected_url in [
        ({"corpora": []}, "https://arxiv.org/abs/2607.10020"),
        ({"corpora": [], "paper_url": ""}, ""),
    ]:
        config.write_text(json.dumps(settings))
        page = create_app().test_client().get("/").get_data(as_text=True)
        public_config = re.search(
            r'<script id="public-config" type="application/json">(.*?)</script>', page
        )
        assert public_config is not None
        assert json.loads(public_config.group(1))["paper_url"] == expected_url


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not available")
def test_highlight_ui():
    completed = subprocess.run(
        [shutil.which("node"), str(Path(__file__).with_name("highlight_ui.test.js"))],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_wsgi_imports_from_web_working_directory(tmp_path, monkeypatch):
    (tmp_path / "samples_data.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_text('{"corpora": []}')
    monkeypatch.setenv("FINDMYTEXT_CONFIG", str(config))
    completed = subprocess.run(
        [sys.executable, "-c", (
            "from wsgi import app; "
            "assert app.test_client().get('/').status_code == 200"
        )],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
