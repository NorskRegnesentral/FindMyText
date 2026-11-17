import json
from pathlib import Path
import subprocess
import sys

from findmytext import index_builder, indexing
from web.corpussearch import create_app


def test_web_adapter_scores_and_highlights_after_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    text = " ".join(f"sharedword{i}" for i in range(90))
    records = [
        {"id": "source", "text": text},
        {"id": "other", "text": " ".join(f"otherword{i}" for i in range(90))},
    ]
    source = tmp_path / "corpus.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    shards = index_builder.index_file(
        str(source), str(tmp_path / "shards"), nb_workers=1,
    )
    index = tmp_path / "index"
    index_builder.merge_indexes(shards, str(index))
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
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[-1]["type"] == "result", events
    for ranking in events[-1]["results"].values():
        assert ranking["top"]["doc_id"] == "source"
        assert ranking["top"]["score"] > 0
    cfg = app.config["APP_CONFIG"]
    detector, _ = app.config["INDEX_MANAGER"].get(cfg.corpus("fixture"), cfg.params)
    assert detector.find_matches_jaccard(text, score="rarity")["source"] > 0
    assert detector.find_matches_clustering(text)["source"] > 0
    response = client.post("/api/highlight", json={
        "corpus": "fixture", "text": text, "doc_id": "source",
        "modes": ["cc_largest", "cc_all"],
    })
    assert response.status_code == 200
    layers = response.get_json()["layers"]
    assert {layer["mode"] for layer in layers} == {"cc_largest", "cc_all"}
    for layer in layers:
        spans = [span for cluster in layer["clusters"] for span in cluster["spans"]]
        assert spans
        assert all(0 <= start < end <= len(text) for start, end in spans)
        assert layer["matched_chars"] == sum(end - start for start, end in spans)


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
