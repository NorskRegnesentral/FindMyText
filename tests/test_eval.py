import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from findmytext import detectors, index_builder  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "fmt_eval", Path(__file__).resolve().parents[1] / "experiments" / "eval.py"
)
fmt_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fmt_eval)


def random_words(rng, n):
    return [f"tok{rng.randrange(3000)}" for _ in range(n)]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def corpus_and_eval_set(tmp_path):
    """A small index of distinct random-word documents, and an evaluation set whose
    positives embed a passage of one document between unrelated filler words."""
    rng = random.Random(0)
    docs = [{"id": f"doc-{i}", "text": " ".join(random_words(rng, 200))} for i in range(6)]
    write_jsonl(tmp_path / "corpus.jsonl", docs)
    index = index_builder.build_index(str(tmp_path / "corpus.jsonl"), str(tmp_path / "index"), n_workers=1)

    examples = []
    for doc in docs:
        words = doc["text"].split()
        passage = " ".join(words[40:160])
        text = " ".join(random_words(rng, 60) + [passage] + random_words(rng, 60))
        examples.append({"text": text, "type": "positive", "source_doc_id": doc["id"]})
    for _ in range(6):
        examples.append({"text": " ".join(random_words(rng, 200)), "type": "negative"})
    rng.shuffle(examples)
    write_jsonl(tmp_path / "eval.jsonl", examples)
    return index, docs, examples


class SearchOnlyDetector:
    """A detector exposing nothing but search(); never finds anything."""

    def search(self, text, min_fragment_length=1000.0):
        return {}


class TextOnlyDetector:
    """A detector whose search() takes the text alone; wraps a real detector."""

    def __init__(self, inner):
        self.inner = inner

    def search(self, text):
        return self.inner.search(text, 300)


def test_precision_at_recall_and_metrics_on_known_scores():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.5, 0.1, 0.05])
    # Recall 0.75 is reached with perfect precision; full recall needs the 0.2
    # positive, which lets in two negatives (4 / 6).
    assert fmt_eval.precision_at_recall(labels, scores, 0.75) == 1.0
    assert fmt_eval.precision_at_recall(labels, scores, 0.90) == pytest.approx(4 / 6)
    metrics = fmt_eval.compute_metrics(labels, scores)
    assert metrics["roc_auc"] == pytest.approx(14 / 16)
    assert metrics["p_at_r90"] == metrics["p_at_r99"] == pytest.approx(4 / 6)
    assert metrics["n_pos_missed"] == metrics["n_neg_empty"] == 0 and metrics["max_recall"] == 1.0
    assert fmt_eval.floor_warnings({"det": {**metrics, "n_pos": 4, "n_neg": 4}}) == []


def test_compute_metrics_reports_the_score_floor():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.0, 0.6, 0.5, 0.1, 0.0])
    m = fmt_eval.compute_metrics(labels, scores)
    assert m["n_pos_missed"] == 1 and m["n_neg_empty"] == 1 and m["max_recall"] == 0.75
    # Recall above 0.75 needs the positive at 0, i.e. everything: the positive rate.
    assert m["p_at_r90"] == m["p_at_r99"] == 0.5
    m.update(n_pos=4, n_neg=4, top1_acc=0.75, mean_score_time=0.0, mean_retrieval_time=0.0)
    (warning,) = fmt_eval.floor_warnings({"det": m})
    assert warning.startswith("det: P@R=90, P@R=99 is the positive rate (4/8)")
    assert "1 positives (25.0%)" in warning and "1 negatives" in warning
    table = fmt_eval.format_results({"det": m})
    assert "missed" in table.splitlines()[1] and table.endswith(warning)


def test_evaluator_separates_classes_and_matches_plain_search(tmp_path, corpus_and_eval_set):
    index, docs, examples = corpus_and_eval_set
    evaluator = fmt_eval.Evaluator(index)
    assert list(evaluator.detectors) == list(fmt_eval.DEFAULT_DETECTOR_NAMES)
    assert evaluator._shared == list(fmt_eval.DEFAULT_DETECTOR_NAMES) and evaluator._separate == []

    scores_path = tmp_path / "scores.jsonl"
    metrics = evaluator.run(str(tmp_path / "eval.jsonl"), min_fragment_length=300,
                            output_scores=str(scores_path), verbose=False)
    for name, m in metrics.items():
        assert m["roc_auc"] == 1.0 and m["p_at_r90"] == 1.0 and m["p_at_r99"] == 1.0, name
        assert m["top1_acc"] == 1.0 and m["n_pos"] == m["n_neg"] == 6
        assert m["mean_score_time"] > 0 and m["mean_retrieval_time"] > 0

    rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert len(rows) == len(examples)
    for row, ex in zip(rows, examples):
        assert row["type"] == ex["type"] and row["text_length"] == len(ex["text"])
        for name in fmt_eval.DEFAULT_DETECTOR_NAMES:
            d = row["detectors"][name]
            if ex["type"] == "positive":
                assert d["top_doc"] == ex["source_doc_id"] and d["score"] == d["top_prob"] > 0.9
            else:
                assert d["top_doc"] is None and d["score"] == 0.0

    # The shared-retrieval fast path gives exactly what each detector's search() gives.
    for ex in examples:
        results, _, _ = evaluator._search_all(ex["text"], 300)
        for name, detector in evaluator.detectors.items():
            assert results[name] == pytest.approx(detector.search(ex["text"], 300))

    # A positive assigned to the wrong document scores 0, even if some document matches.
    wrong = dict(next(e for e in examples if e["type"] == "positive"))
    wrong["source_doc_id"] = "doc-does-not-exist"
    write_jsonl(tmp_path / "wrong.jsonl", [wrong] + [e for e in examples if e["type"] == "negative"][:1])
    assert evaluator.run(str(tmp_path / "wrong.jsonl"), min_fragment_length=300, verbose=False)["chain"]["roc_auc"] == 0.5

    # The printed table mentions every detector.
    table = fmt_eval.format_results(metrics)
    assert all(name in table for name in fmt_eval.DEFAULT_DETECTOR_NAMES)


def test_raw_scores_are_what_score_candidates_normalises(corpus_and_eval_set):
    index, docs, examples = corpus_and_eval_set
    positive = next(e for e in examples if e["type"] == "positive")
    for cls in (detectors.NbSharedFingerprintsDetector, detectors.FingerprintChainDetector):
        detector = cls(index, min_probability=1e-3)
        candidates = detector.retrieve(positive["text"], 300)
        assert isinstance(candidates, detectors.Candidates)
        raw = detector.raw_scores(candidates)
        assert raw[positive["source_doc_id"]] > 0
        probs = detector.score_candidates(candidates, 300)
        assert probs == pytest.approx(detector._to_probabilities(raw, 300))
        assert probs == pytest.approx(detector.search(positive["text"], 300))
        empty = detectors.Candidates(candidates.hashes, candidates.positions)
        assert empty.is_empty and detector.raw_scores(empty) == {}
        # A wider retrieval (as shared between detectors) only adds candidates,
        # which scoring then drops: same probabilities.
        wide = detector.retrieve(positive["text"], 300, top_k=1000, min_fingerprints=1)
        assert np.array_equal(wide.hashes, candidates.hashes)
        assert np.array_equal(wide.positions, candidates.positions)
        assert set(wide.hits.doc_ids) >= set(candidates.hits.doc_ids)
        assert detector.score_candidates(wide, 300) == pytest.approx(probs)


def test_run_reports_missed_positives_on_the_score_floor(tmp_path, corpus_and_eval_set):
    index, docs, examples = corpus_and_eval_set
    positives = [e for e in examples if e["type"] == "positive"]

    # A query where a longer passage of doc-1 outranks the source doc-0: top_k=1
    # drops the source; a source id absent from the index is never returned.
    rng = random.Random(1)
    words0, words1 = docs[0]["text"].split(), docs[1]["text"].split()
    outranked = {
        "text": " ".join(random_words(rng, 30) + words1 + random_words(rng, 30)
                         + words0[40:140] + random_words(rng, 30)),
        "type": "positive", "source_doc_id": "doc-0",
    }
    unknown = {**positives[0], "source_doc_id": "doc-unknown"}
    misses_path = tmp_path / "misses.jsonl"
    write_jsonl(misses_path, [outranked, unknown] + examples)

    small = fmt_eval.Evaluator(index, top_k=1)
    metrics = small.run(str(misses_path), min_fragment_length=300, verbose=False)
    assert metrics["chain"]["n_pos_missed"] == 2 and metrics["chain"]["max_recall"] == 0.75
    assert metrics["chain"]["p_at_r90"] == metrics["chain"]["p_at_r99"] == pytest.approx(8 / 14)
    assert "is the positive rate (8/14)" in fmt_eval.format_results(metrics)


def test_add_detector_accepts_search_only_detectors_and_index_paths(tmp_path, corpus_and_eval_set):
    index, docs, examples = corpus_and_eval_set
    defaults = list(fmt_eval.DEFAULT_DETECTOR_NAMES)
    evaluator = fmt_eval.Evaluator(str(tmp_path / "index"))
    assert list(evaluator.detectors) == defaults
    # `plain` is bound to a different DiskBasedIndex object, so it takes the search()
    # path, like the dummy; the defaults share the evaluator's own index.
    plain = detectors.NbSharedFingerprintsDetector(index, min_probability=1e-3)
    evaluator.add_detector("plain", plain)
    evaluator.add_detector("dummy", SearchOnlyDetector())
    evaluator.add_detector("text_only", TextOnlyDetector(plain))
    assert evaluator._shared == defaults and evaluator._separate == ["plain", "dummy", "text_only"]
    metrics = evaluator.run(str(tmp_path / "eval.jsonl"), min_fragment_length=300, verbose=False, limit=8)
    assert metrics["plain"]["roc_auc"] == 1.0 and np.isnan(metrics["plain"]["mean_retrieval_time"])
    assert metrics["plain"]["mean_score_time"] > metrics["baseline"]["mean_score_time"] > 0
    assert metrics["dummy"]["roc_auc"] == 0.5 and metrics["dummy"]["top1_acc"] == 0.0
    assert metrics["text_only"]["roc_auc"] == 1.0 and metrics["text_only"]["top1_acc"] == 1.0
    assert metrics["plain"]["n_pos"] + metrics["plain"]["n_neg"] == 8

    with pytest.raises(TypeError, match="search"):
        evaluator.add_detector("bad", object())
    with pytest.raises(ValueError, match="already registered"):
        evaluator.add_detector("chain", plain)


def test_run_rejects_malformed_examples(tmp_path, corpus_and_eval_set):
    index, docs, examples = corpus_and_eval_set
    evaluator = fmt_eval.Evaluator(index)
    negative = next(e for e in examples if e["type"] == "negative")
    bad = tmp_path / "bad.jsonl"
    write_jsonl(bad, [{"text": "a b c", "type": "positive"}, negative])
    with pytest.raises(ValueError, match="source_doc_id"):
        evaluator.run(str(bad), verbose=False)
    write_jsonl(bad, [{"text": "a b c", "type": "maybe"}])
    with pytest.raises(ValueError, match="'type' must be"):
        evaluator.run(str(bad), verbose=False)
    write_jsonl(bad, [negative, negative])
    with pytest.raises(ValueError, match="both classes"):
        evaluator.run(str(bad), verbose=False)
