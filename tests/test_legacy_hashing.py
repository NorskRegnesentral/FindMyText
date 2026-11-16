import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from findmytext import detectors, index_builder, indexing, web
from findmytext.winnower import HASH_METHOD, LEGACY_HASH, ROLLING_BASE64, Winnower


def original_signed_reference(tokens, length, base):
    """Emulate the paper kernel's signed int64 intermediates with Python integers."""
    def signed(value):
        return (value + 2**63) % 2**64 - 2**63

    modulus = 2**31 - 1
    values = []
    for token in tokens:
        value = 1469598103934665603
        for byte in token.encode("utf-8"):
            value = ((value ^ byte) * 1099511628211) % 2**64
        values.append(value % modulus)
    if len(values) < length:
        return []
    power = 1
    for _ in range(length):
        power = signed(power * base) % modulus
    current = 0
    for value in values[:length]:
        current = signed(signed(current * base) + value) % modulus
    current = signed(current * base) % modulus
    result = [current]
    for position in range(1, len(values) - length + 1):
        removed = signed(values[position - 1] * power)
        current = signed(
            signed(signed(current - removed) + values[position + length - 1]) * base
        ) % modulus
        result.append(current)
    return result


@pytest.mark.parametrize("length,window", [(1, 1), (4, 6), (7, 3)])
def test_legacy_hashes_and_selected_positions_match_original_signed_reference(length, window):
    texts = [
        "", "short", "one two three four five " * 20,
        "Blå, æøå 東京 punctuation-with-hyphens! 123 Å\n" * 25,
    ]
    for base in (256, 257):
        for punctuation in (False, True):
            configured = Winnower(
                length=length, window_size=window, base=base,
                punctuation=punctuation, hash_method=LEGACY_HASH,
            )
            for text in texts:
                normalized = text if punctuation else re.sub(r"[^\w\s]", "", text)
                expected = original_signed_reference(normalized.split(), length, base)
                np.testing.assert_array_equal(configured.get_token_fingerprints(normalized.split()), expected)
                positions = []
                for start in range(max(0, len(expected) - window + 1)):
                    position = min(range(start, start + window), key=lambda p: (expected[p], -p))
                    if not positions or position != positions[-1]:
                        positions.append(position)
                hashes, actual_positions = configured.get_winnowed_fingerprints(text)
                assert hashes.dtype == np.int64
                assert hashes.tolist() == [expected[p] for p in positions]
                assert actual_positions.tolist() == positions


def test_method_defaults_metadata_and_old_unlabelled_index_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.MemoryBasedIndex, "_CHUNK", 100)
    assert Winnower().base == ROLLING_BASE64
    assert Winnower().hash_method == HASH_METHOD
    legacy = Winnower(length=4, hash_method=LEGACY_HASH)
    assert legacy.base == 256
    metadata = legacy.to_metadata()
    assert Winnower.from_metadata(metadata).to_metadata() == metadata
    del metadata["hash_method"]
    assert Winnower.from_metadata(metadata).hash_method == LEGACY_HASH
    text = " ".join(f"paperword{i}" for i in range(150))
    values, positions = legacy.get_winnowed_fingerprints(text)
    memory = indexing.MemoryBasedIndex(metadata)
    memory.add_fingerprints("paper-document", list(zip(values.tolist(), positions.tolist())))
    shard = tmp_path / "old-unlabelled.mpk.gz"
    memory.to_msgpack(str(shard))
    target = tmp_path / "old-index"
    index_builder.merge_indexes([str(shard)], str(target))
    assert json.loads((target / "meta.json").read_text())["hash_method"] == LEGACY_HASH
    (target / "meta.json").write_text(json.dumps(metadata))
    for cls in (detectors.NbSharedFingerprintsDetector, detectors.FingerprintChainDetector):
        detector = cls(str(target), min_fingerprints=1)
        assert detector.runtime_winnower.hash_method == LEGACY_HASH
        assert detector.get_containment_scores(text, min_fragment_length=100)["paper-document"] > 0
    website = web.TextContainmentDetector(str(target), min_fingerprints=1)
    assert website.runtime_winnower.hash_method == LEGACY_HASH
    assert website.find_matches_jaccard(text)["paper-document"] > 0


def test_cli_selects_legacy_and_queries_follow_metadata(tmp_path):
    root = Path(__file__).resolve().parents[1]
    text = " ".join(f"paperword{i}" for i in range(100))
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "paper", "text": text}) + "\n")
    for args in (
        ["index", str(source), str(tmp_path / "shards"), "--hash-method", LEGACY_HASH, "--nb_workers", "2"],
        ["merge", str(tmp_path / "shards"), str(tmp_path / "index")],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "findmytext.index_builder", *args],
            cwd=root, text=True, capture_output=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    reader = indexing.DiskBasedIndex(str(tmp_path / "index"))
    assert reader.winnower.hash_method == LEGACY_HASH and reader.winnower.base == 256
    expected, _ = Winnower(length=4, hash_method=LEGACY_HASH).get_winnowed_fingerprints(text)
    np.testing.assert_array_equal(reader.fingerprints, np.unique(expected))
