"""Tests of the negative-example generation in experiments/benchmark.py.

The negatives must (a) no longer align with their source above the oracle threshold
(Smith-Waterman), while (b) retaining most of the source wording, so that they still share
many word n-grams / fingerprints with it.
"""

import collections
import importlib.util
from pathlib import Path

import numpy as np
import orjson
import pytest

pytest.importorskip("pyphen")

_SPEC = importlib.util.spec_from_file_location(
    "fmt_benchmark", Path(__file__).resolve().parents[1] / "experiments" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def random_text(rng, nb_sentences, words_per_sentence=12):
    """Prose-like text of short random words with sentence punctuation, so that both
    the sentence-boundary splitting and the word n-grams are meaningful."""
    sentences = []
    for _ in range(nb_sentences):
        words = [f"w{rng.integers(10_000)}" for _ in range(words_per_sentence)]
        sentences.append(" ".join(words).capitalize() + ".")
    return " ".join(sentences)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def generator():
    """Generator without LLM: every aligned region is permuted."""
    return benchmark.NegativeExampleGenerator(
        benchmark.Corpus(), llm_model=None, min_oracle_score=100
    )


# --- word_ngram_overlap ------------------------------------------------------------


def test_ngram_overlap_bounds(rng):
    text = random_text(rng, 10)
    assert benchmark.word_ngram_overlap(text, text) == 1.0
    assert benchmark.word_ngram_overlap(text, random_text(rng, 10)) == 0.0
    assert benchmark.word_ngram_overlap(text, "only four words here") == 0.0


def test_ngram_overlap_of_shuffled_sentences_is_high(rng):
    text = random_text(rng, 20)
    sentences = text.split(". ")
    rng.shuffle(sentences)
    # Only the 5-grams straddling a sentence boundary (4 per boundary) can be broken,
    # i.e. at most 4 * 19 of the 236 5-grams
    assert benchmark.word_ngram_overlap(text, ". ".join(sentences)) > 0.6


# --- permute_chunks ---------------------------------------------------------------


def test_permute_chunks_keeps_words_and_breaks_alignment(generator, rng):
    np.random.seed(0)
    text = random_text(rng, 40)
    permuted = generator.permute_chunks(text, target_chunk_size=70, max_chunks=64)

    assert permuted != text
    assert sorted(permuted.split()) == sorted(text.split())
    # ~42 chunks of ~70 characters: the largest aligned region is a single chunk, and
    # each chunk boundary breaks at most 4 word 5-grams (out of ~480)
    assert generator.aligner.get_largest_region_score(text, permuted) < len(text) / 3
    assert benchmark.word_ngram_overlap(text, permuted) > 0.6


def test_permute_chunks_does_not_keep_adjacent_chunks_in_order(generator, monkeypatch):
    np.random.seed(0)
    sentences = [f"Sentence number {i} is here." for i in range(12)]
    monkeypatch.setattr(benchmark, "chunk_proportion", lambda text, chunk_proportions: list(sentences))

    permuted = generator.permute_chunks(" ".join(sentences))

    order = [int(s.split()[2]) for s in permuted.split(". ")]
    assert sorted(order) == list(range(12))
    assert all(b != a + 1 for a, b in zip(order, order[1:]))


# --- window / edit helpers ---------------------------------------------------------


def test_merge_windows_merges_overlaps_and_prefers_permutation(generator):
    windows = [(50, 80, "paraphrase"), (0, 30, "paraphrase"), (25, 40, "permute"),
               (80, 90, "paraphrase")]
    assert generator._merge_windows(windows) == [(0, 40, "permute"), (50, 90, "paraphrase")]


def test_apply_edits_replaces_windows_from_the_end(generator):
    text = "aaa bbb ccc ddd eee"
    edits = [
        {"start": 0, "end": 3, "edited": "X"},
        {"start": 8, "end": 11, "edited": "Y Y"},
    ]
    assert generator._apply_edits(text, edits) == "X bbb Y Y ddd eee"


def test_check_edits_rejects_edits_that_do_not_reduce_alignment(generator, rng):
    source = random_text(rng, 30)
    window = source[200:500]
    edits = [
        {"start": 200, "end": 500, "original": window, "operation": "paraphrase",
         "edited": window},
        {"start": 200, "end": 500, "original": window, "operation": "paraphrase",
         "edited": random_text(rng, 4)},
    ]
    applied, rejected = generator._check_edits(source, edits)
    assert applied == [edits[1]]
    assert rejected == [edits[0]]


def test_get_regions_to_edit_finds_all_copied_blocks(generator, rng):
    block_a, block_b, block_c = (random_text(rng, 6) for _ in range(3))
    source = " ".join([block_a, block_b, block_c])
    filler = random_text(rng, 6)
    edited = " ".join([block_c, filler, block_a])

    regions = generator._get_regions_to_edit(source, edited, min_score=100)

    assert len(regions) == 2
    assert regions[0][2] >= regions[1][2] >= 100
    # The aligned spans may extend by a few characters into the neighbouring text
    spans = sorted((start, end) for start, end, _ in regions)
    assert spans[0][0] <= 10 and abs(spans[0][1] - len(block_c)) <= 10
    assert abs(spans[1][0] - (len(edited) - len(block_a))) <= 10
    assert spans[1][1] >= len(edited) - 10


def test_select_windows_permutes_long_regions_and_paraphrases_short_ones(generator, rng):
    text = random_text(rng, 40)
    regions = [(0, 400, 400.0), (1200, 1320, 120.0)]

    # Without an LLM, everything is permuted over the whole region
    assert [op for _, _, op in generator._select_windows(text, regions, stuck=False)] == [
        "permute", "permute"]

    generator.model = object()
    windows = generator._select_windows(text, regions, stuck=False)
    assert [op for _, _, op in windows] == ["permute", "paraphrase"]
    assert windows[1][1] - windows[1][0] <= generator.max_chunk_size + 2 * 150


def test_edit_regions_retries_rejected_paraphrases_with_aggressive_prompt(
    generator, rng, monkeypatch
):
    source = random_text(rng, 30)
    # A long copied block (to be permuted) and a short one (to be paraphrased),
    # separated by unrelated text
    filler = random_text(rng, 5)
    short_block = source[1000:1120]
    text = " ".join([source[:600], filler, short_block, random_text(rng, 5)])
    short_start = 600 + 1 + len(filler) + 1
    regions = [(0, 600, 600.0), (short_start, short_start + 120, 120.0)]
    calls = []

    def fake_paraphrases(chunks, temperature, style):
        calls.append(style)
        # The conservative prompt returns the text unchanged, the aggressive one
        # produces unrelated text
        return list(chunks) if style == "conservative" else [random_text(rng, 2) for _ in chunks]

    generator.model = object()
    monkeypatch.setattr(generator, "get_paraphrases", fake_paraphrases)

    edits = generator._edit_regions(source, text, regions)

    assert calls == ["conservative", "aggressive"]
    assert collections.Counter(e["operation"] for e in edits) == {
        "permute": 1, "paraphrase_aggressive": 1}
    for edit in edits:
        assert generator.aligner.get_largest_region_score(
            source, edit["edited"]
        ) < generator.aligner.get_largest_region_score(source, edit["original"])


# --- end to end --------------------------------------------------------------------


def test_sample_negative_breaks_alignment_but_keeps_ngrams(generator, rng):
    np.random.seed(7)
    source = random_text(rng, 50)
    generator.corpus.data = [{"id": "doc-1", "text": source}]

    result = generator.sample_negative(max_nb_attempts=30)

    assert result["text1"] == source
    assert result["pair_type"] == "negative"
    assert result["meta"] == {"id": "doc-1"}
    assert result["operations"]["permute"] >= 1

    # No segment aligns with the source above the threshold any more...
    assert result["score"] < generator.min_oracle_score
    assert generator.aligner.get_largest_region_score(source, result["text2"]) < 100
    # ...but the wording is largely intact: the same words, and most 5-grams preserved
    assert set(source.split()) <= set(result["text2"].split())
    assert result["ngram_overlap"] == pytest.approx(
        benchmark.word_ngram_overlap(source, result["text2"])
    )
    assert result["ngram_overlap"] > 0.4
    assert result["nb_steps"] >= 1
    assert result["operations"].get("paraphrase", 0) == 0


def test_paraphrase_fraction_paraphrases_only_the_requested_share(generator, rng, monkeypatch):
    np.random.seed(3)
    text = random_text(rng, 40)
    chunks = benchmark.chunk_max(text, max_chunk_size=generator.max_chunk_size)
    sent = []

    def fake_paraphrases(chunks, temperature, style):
        sent.extend(chunks)
        assert style == "conservative"
        return ["<P>"] * len(chunks)

    # Without an LLM the step is a no-op
    assert generator._paraphrase_fraction(text) == (text, 0)

    generator.model = object()
    monkeypatch.setattr(generator, "get_paraphrases", fake_paraphrases)
    generator.paraphrase_fraction = 0.5
    new_text, nb = generator._paraphrase_fraction(text)

    assert nb == len(sent) == round(0.5 * len(chunks))
    assert new_text.count("<P>") == nb
    # The paraphrased chunks are real chunks, and the others are kept verbatim, in order
    assert set(sent) <= set(chunks)
    position = 0
    for chunk in chunks:
        if chunk not in sent:
            position = new_text.index(chunk, position) + len(chunk)


def test_sample_negative_with_paraphrase_fraction_gives_intermediate_overlap(
    generator, rng, monkeypatch
):
    np.random.seed(11)
    source = random_text(rng, 50)
    generator.corpus.data = [{"id": "doc-3", "text": source}]
    generator.model = object()
    generator.paraphrase_fraction = 0.5
    # The fake LLM rewrites every chunk into unrelated text of similar length
    monkeypatch.setattr(
        generator,
        "get_paraphrases",
        lambda chunks, temperature, style: [random_text(rng, 3) for _ in chunks],
    )

    result = generator.sample_negative(max_nb_attempts=30)

    assert result["score"] < generator.min_oracle_score
    assert result["operations"]["paraphrase"] >= 1
    # About half of the words were rewritten: the overlap is neither ~1 nor ~0
    assert 0.15 < result["ngram_overlap"] < 0.7


def test_sample_negative_raises_when_threshold_cannot_be_reached(generator):
    # A single long "word" cannot be split by permutation, and without an LLM there is
    # no other way to edit it
    generator.corpus.data = [{"id": "doc-2", "text": "x" * 400}]
    with pytest.raises(RuntimeError, match="still above threshold"):
        generator.sample_negative(max_nb_attempts=6)


# --- benchmark pipeline ------------------------------------------------------------


def write_jsonl(path, rows):
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")
    return str(path)


def test_sample_corpus_from_file_and_from_list(tmp_path, rng):
    docs_a = [{"id": f"a{i}", "text": random_text(rng, 20)} for i in range(30)]
    docs_b = [{"id": f"b{i}", "text": random_text(rng, 20)} for i in range(30)]
    # Too short and too long documents must be filtered out
    docs_a.append({"id": "short", "text": "tiny"})
    docs_b.append({"id": "long", "text": "x" * 100_000})
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    file_a = write_jsonl(corpus_dir / "a.jsonl", docs_a)
    file_b = write_jsonl(corpus_dir / "b.jsonl", docs_b)
    out = tmp_path / "sample.jsonl"

    benchmark.sample_corpus(file_a, 5, str(out), skip_prob=0.0, min_length=100, max_length=10_000)
    sample = [orjson.loads(l) for l in out.read_bytes().splitlines()]
    assert len(sample) == 5 and all(d["id"].startswith("a") for d in sample)

    # Collection of files: the quota is spread over the files, filtered docs excluded
    benchmark.sample_corpus([file_a, file_b], 7, str(out), skip_prob=0.0, min_length=100, max_length=10_000)
    ids = [orjson.loads(l)["id"] for l in out.read_bytes().splitlines()]
    assert len(ids) == 7
    assert sum(i.startswith("a") for i in ids) == 4 and sum(i.startswith("b") for i in ids) == 3
    assert "short" not in ids and "long" not in ids

    # A directory is expanded to its corpus files
    assert benchmark._list_corpus_files(str(corpus_dir)) == sorted([file_a, file_b])
    with pytest.raises(FileNotFoundError):
        benchmark._list_corpus_files(str(corpus_dir / "missing.jsonl"))


def test_merge_examples_splits_evenly_and_drops_bulky_fields(tmp_path):
    positives = [
        {"text": f"pos {i}", "source_doc_id": f"d{i}", "source_text": "S" * 100, "before": [], "oracle_score": 1200}
        for i in range(10)
    ]
    negatives = [
        {"text": f"neg {i}", "source_doc_id": f"d{i}", "source_text": "S" * 100, "oracle_score": 50}
        for i in range(10)
    ]
    pos_path = write_jsonl(tmp_path / "pos.jsonl", positives)
    neg_path = write_jsonl(tmp_path / "neg.jsonl", negatives)
    out = tmp_path / "bench.jsonl"

    benchmark.merge_examples(pos_path, neg_path, str(out), nb_examples=7, seed=0)
    rows = [orjson.loads(l) for l in out.read_bytes().splitlines()]
    assert len(rows) == 7
    types = collections.Counter(r["type"] for r in rows)
    assert types == {benchmark.POSITIVE: 4, benchmark.NEGATIVE: 3}
    for r in rows:
        assert not set(benchmark.BULKY_FIELDS) & set(r)
        assert r["text"].startswith("pos" if r["type"] == benchmark.POSITIVE else "neg")
        assert "source_doc_id" in r and "oracle_score" in r

    # Same seed, same output; bulky fields can be kept on request
    benchmark.merge_examples(pos_path, neg_path, str(out), nb_examples=7, seed=0, keep_source_texts=True)
    rows2 = [orjson.loads(l) for l in out.read_bytes().splitlines()]
    assert [r["text"] for r in rows2] == [r["text"] for r in rows]
    assert all("source_text" in r for r in rows2)

    # The legacy alias keeps separate counts
    benchmark.merge_positive_and_negative_samples(pos_path, neg_path, 2, 5, str(out), seed=1)
    rows3 = [orjson.loads(l) for l in out.read_bytes().splitlines()]
    assert collections.Counter(r["type"] for r in rows3) == {benchmark.POSITIVE: 2, benchmark.NEGATIVE: 5}

    # Positives without a source document are rejected
    bad_path = write_jsonl(tmp_path / "bad.jsonl", [{"text": "no source"}])
    with pytest.raises(ValueError, match="source_doc_id"):
        benchmark.merge_examples(bad_path, neg_path, str(out), nb_examples=2)


def test_median_text_length_sets_positive_target(tmp_path, monkeypatch):
    neg_path = write_jsonl(
        tmp_path / "neg.jsonl", [{"text": "x" * n} for n in (100, 300, 500, 900, 2000)]
    )
    assert benchmark.median_text_length(neg_path) == 500

    # generate_positives resolves a path into the median length of that file
    captured = {}

    class FakeGenerator:
        def __init__(self, corpus, other_corpus, min_oracle_score, target_text_length):
            captured["target"] = target_text_length
            captured["threshold"] = min_oracle_score

        def sample_positive(self):
            return {"text1": "src", "text2": "edited", "score": 1000.0, "meta": {"id": "d1"}, "before": []}

    monkeypatch.setattr(benchmark, "PositiveExampleGenerator", FakeGenerator)
    monkeypatch.setattr(benchmark, "Corpus", lambda path: None)
    out = tmp_path / "pos.jsonl"
    benchmark.generate_positives("c.jsonl", "o.jsonl", str(out), nb_examples=2, sw_threshold=1000, target_length=neg_path)

    assert captured == {"target": 500, "threshold": 1000}
    rows = [orjson.loads(l) for l in out.read_bytes().splitlines()]
    assert len(rows) == 2
    assert rows[0] == {
        "text": "edited", "type": benchmark.POSITIVE, "source_doc_id": "d1",
        "source_text": "src", "oracle_score": 1000.0, "before": [],
    }


def test_append_until_resumes_and_skips_failures(tmp_path):
    out = tmp_path / "examples.jsonl"
    write_jsonl(out, [{"n": -1}, {"n": -2}])
    calls = iter([RuntimeError("bad doc"), None, {"n": 1}, {"n": 2}, {"n": 3}])

    def make_example():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    # Two examples present, two more needed: the failure and the None are skipped
    benchmark._append_until(str(out), 4, make_example)
    rows = [orjson.loads(l)["n"] for l in out.read_bytes().splitlines()]
    assert rows == [-1, -2, 1, 2]

    # Nothing to do when the file is already full: make_example is not called
    benchmark._append_until(str(out), 4, lambda: pytest.fail("should not generate"))
    assert benchmark.count_lines(str(out)) == 4

    # overwrite starts from scratch
    benchmark._append_until(str(out), 1, make_example, overwrite=True)
    assert [orjson.loads(l)["n"] for l in out.read_bytes().splitlines()] == [3]
