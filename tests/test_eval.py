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


def test_create_eval_set_samples_shuffles_and_keeps_only_light_fields(tmp_path):
    pos = tmp_path / "pos.jsonl"
    neg = tmp_path / "neg.jsonl"
    write_jsonl(pos, [
        {"text": f"p{i}", "type": "positive", "source_doc_id": i, "source_text": "x" * 100,
         "oracle_score": 1200.0, "before": {"doc_id": "a"}}
        for i in range(5)
    ])
    write_jsonl(neg, [{"text": f"n{i}", "source_text": "y" * 100, "oracle_score": 90.0} for i in range(4)])
    out = tmp_path / "mix.jsonl"
    fmt_eval.create_eval_set(str(pos), str(neg), 3, 2, str(out), seed=1)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5
    assert sorted(r["type"] for r in rows) == ["negative"] * 2 + ["positive"] * 3
    for row in rows:
        assert set(row) <= set(fmt_eval._KEPT_FIELDS) and "source_text" not in row
        assert row["text"].startswith("p" if row["type"] == "positive" else "n")
        assert (row["type"] == "positive") == ("source_doc_id" in row)
    # Deterministic for a given seed, and not simply positives-then-negatives.
    fmt_eval.Evaluator.create_eval_set(str(pos), str(neg), 3, 2, str(tmp_path / "again.jsonl"), seed=1)
    assert (tmp_path / "again.jsonl").read_text() == out.read_text()
    fmt_eval.create_eval_set(str(pos), str(neg), 5, 4, str(out), seed=3)
    types = [json.loads(line)["type"] for line in out.read_text().splitlines()]
    assert types != ["positive"] * 5 + ["negative"] * 4
    with pytest.raises(ValueError, match="only 4 examples"):
        fmt_eval.create_eval_set(str(pos), str(neg), 1, 5, str(out))


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
        empty = detectors.Candidates(candidates.query, candidates.matches.clear())
        assert detector.raw_scores(empty) == {}
        # A wider retrieval (as shared between detectors) only adds candidates,
        # which scoring then drops: same probabilities.
        wide = detector.retrieve(positive["text"], 300, top_k=1000, min_fingerprints=1)
        assert wide.query.equals(candidates.query)
        assert set(wide.matches["doc_match_id"]) >= set(candidates.matches["doc_match_id"])
        assert detector.score_candidates(wide, 300) == pytest.approx(probs)


def test_diagnose_misses_classifies_each_cut(tmp_path, corpus_and_eval_set):
    index, docs, examples = corpus_and_eval_set
    eval_path = str(tmp_path / "eval.jsonl")
    positives = [e for e in examples if e["type"] == "positive"]

    evaluator = fmt_eval.Evaluator(index)
    report = evaluator.diagnose_misses(eval_path, min_fragment_length=300, verbose=False)
    assert report["misses"] == [] and set(report["summary"]) == set(fmt_eval.DEFAULT_DETECTOR_NAMES)
    assert all(s["n_misses"] == 0 and s["to_recover_all"] is None for s in report["summary"].values())
    assert fmt_eval.format_misses([], ["chain"]) == "No missed positive."
    with pytest.raises(ValueError, match="unknown detector"):
        evaluator.diagnose_misses(eval_path, detector_names=["nope"])
    evaluator.add_detector("dummy", SearchOnlyDetector())
    with pytest.raises(ValueError, match="cannot be diagnosed"):
        evaluator.diagnose_misses(eval_path, detector_names=["dummy"])

    # A query where a longer passage of doc-1 outranks the source doc-0: top_k=1
    # drops the source, and a source id absent from the index shares nothing.
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
    out = tmp_path / "misses_out.jsonl"
    report = small.diagnose_misses(str(misses_path), min_fragment_length=300,
                                   detector_names=["baseline", "chain"],
                                   output_path=str(out), verbose=False)
    by_id = {m["source_doc_id"]: m for m in report["misses"]}
    assert set(by_id) == {"doc-0", "doc-unknown"} and set(report["summary"]) == {"baseline", "chain"}
    m = by_id["doc-0"]
    assert m["rank"] == 2 and m["top_doc"] == "doc-1" and m["top_count"] > m["n_shared"] > 0
    assert m["n_candidates"] >= 2 and m["text_length"] == len(outranked["text"])
    for d in m["detectors"].values():
        assert d["reason"] == fmt_eval.OUTSIDE_TOP_K and d["probability"] > 0.9
    assert m["detectors"]["baseline"]["raw_score"] == m["n_shared"]
    m = by_id["doc-unknown"]
    assert m["rank"] is None and m["n_shared"] == 0 and m["top_doc"] == positives[0]["source_doc_id"]
    for d in m["detectors"].values():
        assert d["reason"] == fmt_eval.NO_SHARED_FINGERPRINTS and d["raw_score"] is None

    summary = report["summary"]["chain"]
    assert summary["n_pos"] == 8 and summary["n_misses"] == 2 and summary["n_recoverable"] == 1
    assert summary["reasons"] == {
        fmt_eval.NO_SHARED_FINGERPRINTS: 1, fmt_eval.BELOW_MIN_FINGERPRINTS: 0,
        fmt_eval.OUTSIDE_TOP_K: 1, fmt_eval.BELOW_MIN_PROBABILITY: 0,
    }
    assert summary["to_recover_all"]["top_k"] == 2 and summary["to_recover_all"]["achievable"]
    # With 8 positives, 99 % recall tolerates no miss, and one is unrecoverable.
    assert summary["to_reach_recall_99"]["n_to_recover"] == 2
    assert not summary["to_reach_recall_99"]["achievable"]
    text = fmt_eval.format_miss_summary(report["summary"])
    assert "top_k >= 2" in text and "not enough: 1 of the 2" in text
    assert "doc-unknown" in fmt_eval.format_misses(report["misses"], ["baseline", "chain"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["source_doc_id"] for r in rows] == [m["source_doc_id"] for m in report["misses"]]

    # run() misses exactly these positives, and its P@R values sit on the floor.
    metrics = small.run(str(misses_path), min_fragment_length=300, verbose=False)
    assert metrics["chain"]["n_pos_missed"] == 2 and metrics["chain"]["max_recall"] == 0.75
    assert metrics["chain"]["p_at_r90"] == metrics["chain"]["p_at_r99"] == pytest.approx(8 / 14)
    assert "is the positive rate (8/14)" in fmt_eval.format_results(metrics)

    # Cuts set on the detectors themselves are honoured: a retrieval threshold no
    # positive reaches, and a cluster size that leaves the chain detector no cluster.
    strict = fmt_eval.Evaluator(index)
    strict.detectors["baseline"].min_fingerprints = 500
    strict.detectors["chain"].min_cluster_size = 10_000
    report = strict.diagnose_misses(eval_path, min_fragment_length=300,
                                    detector_names=["baseline", "chain"], verbose=False)
    assert report["thresholds"]["baseline"]["min_fingerprints"] == 500
    assert len(report["misses"]) == len(positives)
    for m in report["misses"]:
        assert m["rank"] == 1
        assert m["detectors"]["baseline"]["reason"] == fmt_eval.BELOW_MIN_FINGERPRINTS
        chain = m["detectors"]["chain"]
        assert chain["reason"] == fmt_eval.BELOW_MIN_PROBABILITY and chain["raw_score"] == 0.0
        assert chain["probability"] < strict.detectors["chain"].min_probability
    assert report["summary"]["baseline"]["to_recover_all"]["min_fingerprints"] == min(
        m["n_shared"] for m in report["misses"]
    )
    assert report["summary"]["chain"]["to_recover_all"]["min_probability"] == min(
        m["detectors"]["chain"]["probability"] for m in report["misses"]
    )


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
