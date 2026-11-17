import numpy as np
import pytest

from findmytext.overlap_groups import OverlapGroups, build_overlap_groups

DTYPE = np.dtype([("doc_i", "<u4"), ("doc_j", "<u4"), ("count", "<u4")])


def test_complete_components_before_bounded_selection(tmp_path):
    edges = np.array([(0, 1, 50), (1, 2, 51), (4, 5, 70), (5, 6, 60), (7, 8, 50)], dtype=DTYPE)
    np.save(tmp_path / "edges.npy", edges)
    report = build_overlap_groups(tmp_path / "edges.npy", 10, 50, tmp_path / "groups")
    assert report["edges"] == 5
    assert report["documents_with_partners"] == 8
    assert report["isolates"] == 2
    assert report["nontrivial_components"] == 3
    assert report["largest_group_documents"] == 3
    with OverlapGroups(tmp_path / "groups") as groups:
        assert groups.labels.tolist() == [0, 0, 0, 3, 4, 4, 4, 7, 7, 9]
        assert groups.degrees.tolist() == [1, 2, 1, 0, 1, 2, 1, 1, 1, 0]
        assert groups.select(limit=2)["group_id"].tolist() == [0, 4]
        assert groups.select(mode="all_above_min", min_documents=3)["group_id"].tolist() == [0, 4]
        assert groups.members(0).tolist() == [0, 1, 2]
        assert groups.groups[0]["edges"] == 2  # A-B-C is not a clique.
        with pytest.raises(KeyError):
            groups.members(2)
        with pytest.raises(ValueError):
            groups.select(min_documents=1)
        labels = groups.labels.copy()
    np.save(tmp_path / "reverse.npy", edges[::-1])
    build_overlap_groups(tmp_path / "reverse.npy", 10, 50, tmp_path / "reversed")
    with OverlapGroups(tmp_path / "reversed") as groups:
        np.testing.assert_array_equal(groups.labels, labels)
    assert build_overlap_groups(tmp_path / "edges.npy", 10, 50, tmp_path / "groups") == report


@pytest.mark.parametrize("documents", [0, 4])
def test_empty_edges_account_for_all_isolates(tmp_path, documents):
    np.save(tmp_path / "edges.npy", np.empty(0, dtype=DTYPE))
    report = build_overlap_groups(tmp_path / "edges.npy", documents, 20, tmp_path / "groups")
    assert report["nontrivial_components"] == 0
    assert report["isolates"] == documents
    with OverlapGroups(tmp_path / "groups") as groups:
        assert len(groups.select()) == 0
        if documents:
            assert groups.members(2).tolist() == [2]


@pytest.mark.parametrize("row", [(1, 1, 30), (0, 5, 30), (0, 1, 19)])
def test_bad_edges_are_not_published(tmp_path, row):
    np.save(tmp_path / "edges.npy", np.array([row], dtype=DTYPE))
    with pytest.raises(ValueError, match="canonical"):
        build_overlap_groups(tmp_path / "edges.npy", 4, 20, tmp_path / "groups")
    assert not (tmp_path / "groups/report.json").exists()
