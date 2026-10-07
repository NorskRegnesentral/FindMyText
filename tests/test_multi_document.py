import numpy as np

from findmytext.multi_document import (
    MultiDocumentFingerprintChainAnalyzer,
    transform_positions,
)
from findmytext.synthetic import generate_controlled_corpus


def test_coordinate_representations_are_available_and_centered_is_symmetric():
    positions = np.array([[10, 20, 30], [14, 24, 34]])
    assert transform_positions(
        positions,
        "raw",
        position_threshold=10,
        offset_threshold=5,
    ).shape == (2, 3)
    assert transform_positions(
        positions,
        "anchor_offsets",
        position_threshold=10,
        offset_threshold=5,
    ).shape == (2, 3)

    centered = transform_positions(
        positions,
        "centered_offsets",
        position_threshold=10,
        offset_threshold=5,
    )
    permuted = transform_positions(
        positions[:, [2, 0, 1]],
        "centered_offsets",
        position_threshold=10,
        offset_threshold=5,
    )
    np.testing.assert_allclose(centered[:, 0], permuted[:, 0])
    np.testing.assert_allclose(centered[:, 1:][:, [2, 0, 1]], permuted[:, 1:])


def test_analyzer_finds_all_document_and_classifies_pairwise_passages():
    corpus = generate_controlled_corpus(n_documents=4, seed=3)
    documents = {record["id"]: record["text"] for record in corpus.documents}
    selected = ["source-alpha", "alpha-copy-a", "alpha-copy-b"]
    analyzer = MultiDocumentFingerprintChainAnalyzer(
        position_threshold=20,
        offset_threshold=20,
        min_cluster_size=4,
    )
    analysis = analyzer.analyze(documents, document_ids=selected, pairwise_mode="all")

    assert any(p.relationship == "all_documents" for p in analysis.passages)
    assert any(p.relationship == "pairwise" for p in analysis.passages)
    assert any(
        p.classification in {"subsumed", "partial_overlap"}
        for p in analysis.passages
        if p.relationship == "pairwise"
    )
    assert set(analysis.document_ids) == set(selected)


def test_analyzer_generalizes_beyond_three_documents():
    shared = " ".join(f"shared_{index}" for index in range(120))
    documents = {f"doc-{index}": f"unique_{index} " * 20 + shared for index in range(4)}
    analysis = MultiDocumentFingerprintChainAnalyzer(min_cluster_size=4).analyze(
        documents, pairwise_mode="none"
    )
    assert analysis.passages
    assert all(len(p.document_ids) == 4 for p in analysis.passages)
