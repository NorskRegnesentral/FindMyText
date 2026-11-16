import numpy as np

from experiments.review.analyze_overlap_hosts import choose_edges, repeated_lines, skogholt_page_role


def test_host_pair_selection_is_deterministic_and_keeps_selection_provenance():
    dtype = [("doc_i", "<u4"), ("doc_j", "<u4"), ("count", "<u4")]
    rows = np.array([(0, 1, 50), (1, 2, 55), (1, 3, 90), (2, 4, 80), (2, 5, 60)], dtype=dtype)
    same_url = np.array([True, False, False, False, False])
    external = np.array([False, False, False, True, True])
    docplayer = np.array([False, False, False, True, False])
    result = choose_edges(rows, same_url, external, docplayer, np.random.default_rng(41))
    assert result == choose_edges(rows, same_url, external, docplayer, np.random.default_rng(41))
    by_pair = {(row["doc_i"], row["doc_j"]): row for row in result}
    assert len(result) == 5
    assert "random_same_url" in by_pair[(0, 1)]["categories"]
    assert "random_near_k50" in by_pair[(1, 2)]["categories"]
    assert "strongest_different_urls" in by_pair[(1, 3)]["categories"]
    assert "random_docplayer_bridge" in by_pair[(2, 4)]["categories"]
    assert "random_other_host_bridge" in by_pair[(2, 5)]["categories"]
    assert choose_edges(rows[:0], same_url[:0], external[:0], docplayer[:0], np.random.default_rng(41)) == []


def test_repeated_lines_count_documents_not_repetitions_in_one_record():
    line = "A sufficiently long standard report heading"
    result = repeated_lines([
        {"text": f"{line}\n{line}\nTiny"},
        {"text": f"{line}\nA different and sufficiently long line"},
    ])
    assert result[0] == {"line": line, "documents": 2}


def test_skogholt_url_roles_distinguish_article_archive_and_unknown():
    for path in ("/", "/tema/politikk/page/2/", "/merke/buss/", "/author/user/page/3/", "/2023/08/", "/2023/", "/2023/08/01/"):
        assert skogholt_page_role("https://www.skogholt.org" + path) == "archive_or_listing"
    assert skogholt_page_role("https://www.skogholt.org/2023/08/a-post/?amp") == "dated_article"
    assert skogholt_page_role("https://www.skogholt.org/about/") == "other_or_unknown"
