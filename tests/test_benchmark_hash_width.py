import numpy as np
import pytest

from experiments.benchmark_hash_width import (
    ROLLING_BASE64,
    fingerprint_tokens,
    run_benchmark,
)
from findmytext.winnower import get_fingerprints64 as rolling_fingerprints64, tokens2uint64


def test_unreduced_token_hashes_match_independent_integer_reference():
    tokens = ["", "abc", "bl\u00e5", "\u00e6", "\U0001f600", "a\0b", "a" * 10000]
    hashes = tokens2uint64(tokens)
    assert hashes.dtype == np.uint64
    for token, value in zip(tokens, hashes):
        reference = 1469598103934665603
        for byte in token.encode("utf-8"):
            reference = ((reference ^ byte) * 1099511628211) & ((1 << 64) - 1)
        assert int(value) == reference


@pytest.mark.parametrize("length", [0, 1, 4, 6, 33])
@pytest.mark.parametrize("base", [ROLLING_BASE64, 1_000_003])
def test_rolling64_matches_independent_big_integer_reference(length, base):
    tokens = np.random.default_rng(42).integers(
        0, np.iinfo(np.uint64).max, size=32, dtype=np.uint64,
    )
    tokens[:3] = [0, 2**63, 2**64 - 1]
    mask = (1 << 64) - 1
    expected = []
    if length > 0:
        for start in range(len(tokens) - length + 1):
            state = 0
            for token in tokens[start:start + length]:
                state = (state * base + int(token)) & mask
            value = (state * base) & mask
            value ^= value >> 33
            value = (value * 0xFF51AFD7ED558CCD) & mask
            value ^= value >> 33
            value = (value * 0xC4CEB9FE1A85EC53) & mask
            value ^= value >> 33
            expected.append(value)
    actual = rolling_fingerprints64(tokens, length, base)
    assert actual.dtype == np.uint64
    assert list(map(int, actual)) == expected


def test_benchmark_outputs_measured_timings_and_equivalence(tmp_path, monkeypatch):
    from experiments import benchmark_hash_width

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.jsonl.zst").touch()

    def source(*_args, **_kwargs):
        for number in range(4):
            yield {"id": str(number), "text": "a b c d e f g h i j " * 20}

    monkeypatch.setattr(benchmark_hash_width.utils, "stream_json_zst", source)
    report = run_benchmark(corpus, 3, 2, tmp_path / "speed.json")
    assert report["documents"] == 3
    assert set(report["methods"]) == {"rolling64_production", "blake2b64"}
    for method in report["methods"].values():
        assert len(method["seconds"]) == 2
        assert method["median_kgrams_per_second"] > 0
    with pytest.raises(ValueError, match="retired"):
        fingerprint_tokens(["a"] * 10, "current31", 4, 6)
