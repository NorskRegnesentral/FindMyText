from findmytext.synthetic import generate_controlled_corpus
from findmytext.winnower import Winnower


def test_controlled_corpus_reports_exact_ground_truth():
    result = generate_controlled_corpus(
        n_documents=8,
        n_hub_fingerprints=3,
        hub_inclusion_probability=[1.0, 0.0, 1.0],
        seed=7,
    )
    repeated = generate_controlled_corpus(
        n_documents=8,
        n_hub_fingerprints=3,
        hub_inclusion_probability=[1.0, 0.0, 1.0],
        seed=7,
    )

    assert result.documents == repeated.documents
    assert result.ground_truth["pairwise"] == repeated.ground_truth["pairwise"]
    assert (
        result.ground_truth["pairwise"][("alpha-copy-a", "alpha-copy-b")][
            "source_overlap_tokens"
        ]
        == 30
    )
    assert all(
        hubs == [0, 2]
        for hubs in result.ground_truth["intended_hubs_by_document"].values()
    )

    configured = result.ground_truth["winnower"]
    fingerprint_sets = {
        record["id"]: set(
            map(
                int,
                Winnower(**configured).get_winnowed_fingerprints(record["text"])[0],
            )
        )
        for record in result.documents
    }
    for (first, second), truth in result.ground_truth["pairwise"].items():
        assert truth["shared_winnowed_fingerprints"] == len(
            fingerprint_sets[first] & fingerprint_sets[second]
        )


def test_controlled_corpus_validates_probability_shape():
    try:
        generate_controlled_corpus(
            n_hub_fingerprints=2, hub_inclusion_probability=[0.5]
        )
    except ValueError as error:
        assert "n_hub_fingerprints entries" in str(error)
    else:
        raise AssertionError("Expected an invalid probability vector to fail")
