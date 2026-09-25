"""Deterministic synthetic corpora with known passage and fingerprint overlap."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from . import winnower


@dataclass(frozen=True)
class ControlledCorpus:
    """Generated JSONL-style records and their exact construction ground truth."""

    documents: list[dict[str, str]]
    ground_truth: dict[str, Any]


def _tokens(namespace: str, count: int) -> list[str]:
    return [f"{namespace}_t{index:05d}" for index in range(count)]


def _normalise_probabilities(
    n_hub_fingerprints: int, probability: float | Sequence[float]
) -> np.ndarray:
    if n_hub_fingerprints < 0:
        raise ValueError("n_hub_fingerprints must be non-negative")
    values = np.asarray(probability, dtype=float)
    if values.ndim == 0:
        values = np.full(n_hub_fingerprints, float(values))
    elif values.ndim != 1 or len(values) != n_hub_fingerprints:
        raise ValueError(
            "hub_inclusion_probability must be a scalar or have "
            "n_hub_fingerprints entries"
        )
    if np.any((values < 0) | (values > 1)):
        raise ValueError("hub inclusion probabilities must be between 0 and 1")
    return values


def _make_hub_kgrams(
    count: int, configured_winnower: winnower.Winnower
) -> tuple[list[list[str]], list[int]]:
    """Choose non-overlapping k-grams with very low hashes from a fixed pool."""
    if count == 0:
        return [], []
    pool_size = max(20_000, count * 2_000)
    candidate_tokens = _tokens(
        "hub_candidate", pool_size + configured_winnower.length - 1
    )
    candidate_hashes = winnower.get_fingerprints(
        winnower.tokens2unicode(candidate_tokens),
        configured_winnower.base,
        configured_winnower.length,
    )
    selected_positions: list[int] = []
    for position in np.argsort(candidate_hashes):
        position = int(position)
        if all(
            abs(position - selected) >= configured_winnower.length
            for selected in selected_positions
        ):
            selected_positions.append(position)
            if len(selected_positions) == count:
                break
    hub_kgrams = [
        candidate_tokens[position : position + configured_winnower.length]
        for position in selected_positions
    ]
    return hub_kgrams, [
        int(candidate_hashes[position]) for position in selected_positions
    ]


def _make_hub_carrier(
    doc_id: str,
    hub_index: int,
    hub_kgram: list[str],
    hub_hash: int,
    configured_winnower: winnower.Winnower,
) -> list[str]:
    """Surround a target k-gram with unique guards and verify its selection."""
    guard_length = configured_winnower.window_size + configured_winnower.length
    for attempt in range(1_000):
        left = _tokens(f"guard_{doc_id}_{hub_index}_{attempt}_left", guard_length)
        right = _tokens(f"guard_{doc_id}_{hub_index}_{attempt}_right", guard_length)
        carrier = left + hub_kgram + right
        hashes, _ = configured_winnower.get_winnowed_fingerprints(" ".join(carrier))
        if hub_hash in hashes:
            return carrier
    raise RuntimeError(f"Could not construct a stable carrier for hub {hub_index}")


def generate_controlled_corpus(
    n_documents: int = 50,
    *,
    n_hub_fingerprints: int = 0,
    hub_inclusion_probability: float | Sequence[float] = 0.0,
    source_intervals: Mapping[str, tuple[int, int]] | None = None,
    source_token_count: int = 180,
    filler_token_count: int = 70,
    seed: int = 0,
    fingerprint_length: int = 4,
    window_size: int = 6,
) -> ControlledCorpus:
    """Create documents with controlled source passages and common hash noise.

    ``source_intervals`` maps derivative document IDs to half-open token intervals
    in ``source-alpha``. Intersecting intervals therefore create exactly known
    passage overlap. ``n_hub_fingerprints`` creates shared k-grams that are
    independently inserted into each document according to one scalar probability
    or one probability per hub k-gram.

    The ground truth distinguishes intended hub k-grams from those retained by
    winnowing. It also reports exact pairwise shared-fingerprint counts measured
    with the configured :class:`~findmytext.winnower.Winnower`.
    """
    if source_token_count <= 0 or filler_token_count < 0:
        raise ValueError(
            "source_token_count must be positive and filler_token_count non-negative"
        )
    if fingerprint_length <= 0 or window_size <= 0:
        raise ValueError("fingerprint_length and window_size must be positive")

    intervals = dict(
        source_intervals
        or {
            "alpha-copy-a": (0, 90),
            "alpha-copy-b": (60, 150),
            "alpha-copy-c": (120, 180),
        }
    )
    for doc_id, (start, end) in intervals.items():
        if not 0 <= start < end <= source_token_count:
            raise ValueError(
                f"Invalid source interval {(start, end)!r} for {doc_id!r}; "
                f"expected 0 <= start < end <= {source_token_count}"
            )

    minimum_documents = 1 + len(intervals)
    if n_documents < minimum_documents:
        raise ValueError(
            f"n_documents must be at least {minimum_documents} for the requested intervals"
        )

    probabilities = _normalise_probabilities(
        n_hub_fingerprints, hub_inclusion_probability
    )
    rng = np.random.default_rng(seed)
    source_tokens = _tokens("shared_source", source_token_count)
    configured_winnower = winnower.Winnower(
        length=fingerprint_length, window_size=window_size
    )
    hub_kgrams, raw_hub_hashes = _make_hub_kgrams(
        n_hub_fingerprints, configured_winnower
    )

    document_specs: list[tuple[str, tuple[int, int] | None]] = [
        ("source-alpha", (0, source_token_count)),
        *intervals.items(),
    ]
    document_specs.extend(
        (f"control-{index:02d}", None)
        for index in range(n_documents - len(document_specs))
    )

    documents: list[dict[str, str]] = []
    passage_spans: dict[str, dict[str, Any]] = {}
    intended_hubs: dict[str, list[int]] = {}
    for doc_id, source_interval in document_specs:
        pieces: list[tuple[str, list[str]]] = [
            ("filler", _tokens(f"unique_{doc_id}", filler_token_count))
        ]
        if source_interval is not None:
            start, end = source_interval
            pieces.append(("source-alpha", source_tokens[start:end]))

        included = [
            index
            for index, probability in enumerate(probabilities)
            if rng.random() < probability
        ]
        intended_hubs[doc_id] = included
        for hub_index in included:
            insertion_index = int(rng.integers(0, len(pieces) + 1))
            pieces.insert(
                insertion_index,
                (
                    f"hub-{hub_index}",
                    _make_hub_carrier(
                        doc_id,
                        hub_index,
                        hub_kgrams[hub_index],
                        raw_hub_hashes[hub_index],
                        configured_winnower,
                    ),
                ),
            )

        all_tokens: list[str] = []
        for piece_name, piece_tokens in pieces:
            token_start = len(all_tokens)
            all_tokens.extend(piece_tokens)
            if piece_name == "source-alpha":
                char_start = len(" ".join(all_tokens[:token_start]))
                if token_start:
                    char_start += 1
                char_end = len(" ".join(all_tokens))
                passage_spans[doc_id] = {
                    "source_interval": source_interval,
                    "token_span": (token_start, len(all_tokens)),
                    "char_span": (char_start, char_end),
                }
        documents.append({"id": doc_id, "text": " ".join(all_tokens)})

    fingerprints_by_document: dict[str, set[int]] = {}
    positions_by_document: dict[str, dict[int, list[int]]] = {}
    realized_hubs: dict[str, list[int]] = {}
    for record in documents:
        hashes, positions = configured_winnower.get_winnowed_fingerprints(
            record["text"]
        )
        fingerprints_by_document[record["id"]] = set(map(int, hashes))
        position_map: dict[int, list[int]] = {}
        for fingerprint, position in zip(hashes, positions):
            position_map.setdefault(int(fingerprint), []).append(int(position))
        positions_by_document[record["id"]] = position_map
        realized_hubs[record["id"]] = [
            index
            for index in intended_hubs[record["id"]]
            if raw_hub_hashes[index] in fingerprints_by_document[record["id"]]
        ]

    pairwise: dict[tuple[str, str], dict[str, int]] = {}
    specs_by_id = dict(document_specs)
    for first, second in combinations((record["id"] for record in documents), 2):
        first_interval = specs_by_id[first]
        second_interval = specs_by_id[second]
        source_overlap = 0
        if first_interval is not None and second_interval is not None:
            source_overlap = max(
                0,
                min(first_interval[1], second_interval[1])
                - max(first_interval[0], second_interval[0]),
            )
        pairwise[(first, second)] = {
            "source_overlap_tokens": source_overlap,
            "shared_winnowed_fingerprints": len(
                fingerprints_by_document[first] & fingerprints_by_document[second]
            ),
        }

    return ControlledCorpus(
        documents=documents,
        ground_truth={
            "seed": seed,
            "winnower": {
                "length": fingerprint_length,
                "window_size": window_size,
                "base": configured_winnower.base,
                "punctuation": configured_winnower.punctuation,
            },
            "source_intervals": specs_by_id,
            "passage_spans": passage_spans,
            "pairwise": pairwise,
            "hub_fingerprints": [
                {
                    "index": index,
                    "hash": fingerprint_hash,
                    "probability": float(probabilities[index]),
                    "tokens": hub_kgrams[index],
                }
                for index, fingerprint_hash in enumerate(raw_hub_hashes)
            ],
            "intended_hubs_by_document": intended_hubs,
            "realized_hubs_by_document": realized_hubs,
            "fingerprint_positions_by_document": positions_by_document,
        },
    )
