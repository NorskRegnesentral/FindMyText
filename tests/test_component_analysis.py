import numpy as np
import pytest

from experiments.analyze_overlap_component import (
    _common_signatures, _excluded_shared, component_statistics, marker_regions,
    layout_census,
)
from findmytext.winnower import Winnower


DTYPE = [("doc_i", "<u4"), ("doc_j", "<u4"), ("count", "<u4")]


def test_filtered_graph_counts_retained_isolates_and_non_cliques():
    edges = np.array([(1, 3, 50), (3, 8, 60)], dtype=DTYPE)
    report = component_statistics([1, 3, 8, 11], edges)
    assert report["documents"] == 4
    assert report["edges"] == 2
    assert report["largest_components"] == [3, 1]
    assert report["isolates"] == 1
    assert report["nontrivial_components"] == 1
    with pytest.raises(ValueError, match="endpoint"):
        component_statistics([1, 3], edges)
    assert component_statistics([], np.empty(0, dtype=DTYPE))["components_including_isolates"] == 0


def test_common_hash_subtraction_matches_direct_set_intersections_across_words():
    sets = [np.arange(0, 130, dtype=np.uint64), np.arange(60, 160, dtype=np.uint64), np.array([3, 62, 90, 129, 150], dtype=np.uint64)]
    offsets = np.array([0] + list(np.cumsum([len(values) for values in sets])), dtype=np.uint64)
    hashes = np.concatenate(sets)
    excluded = np.arange(0, 160, 2, dtype=np.uint64)
    signatures = _common_signatures(np.arange(3), offsets, hashes, excluded)
    edges = np.array([(0, 1, 70), (0, 2, 4), (1, 2, 4)], dtype=DTYPE)
    counts = _excluded_shared(edges, np.arange(3), signatures)
    expected = [len(set(sets[int(row["doc_i"])]) & set(sets[int(row["doc_j"])]) & set(excluded)) for row in edges]
    assert counts.tolist() == expected
    assert _excluded_shared(edges, np.arange(3), _common_signatures(np.arange(3), offsets, hashes, np.empty(0, dtype=np.uint64))).tolist() == [0, 0, 0]


def test_marker_diagnostic_uses_standalone_lines_not_every_word():
    winnower = Winnower(length=4, window_size=6)
    text = "First example\nDetaljer\nOne more Detaljer word\n  Detaljer  \nLast words"
    assert marker_regions(text, winnower) == [2, 7]


def test_layout_census_measures_lines_without_claiming_content_classification():
    class Store:
        def iter_many(self, ids, *, id_kind):
            assert list(ids) == [3, 8] and id_kind == "internal"
            yield 3, {"text": "word Detaljer word\nDetaljer\n"}
            yield 8, {"text": "Details\n" + "Detaljer\n" * 50}

    report = layout_census(Store(), [3, 8])
    assert report["documents"] == 2
    assert report["documents_with_at_least_3_standalone_detaljer_lines"] == 1
    assert report["documents_with_at_least_50_standalone_detaljer_lines"] == 1
    assert report["standalone_detaljer_lines_min_median_max"] == [1, 25.5, 50]
