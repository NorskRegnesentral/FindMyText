import json

import numpy as np
import zstandard

from experiments.review.review_overlap_groups import prepare_groups, review_groups, select_review_pairs
from findmytext.document_store import build_document_store
from findmytext.overlap_inverted import _source_identity
from hplt_build import build_index


def test_review_uses_default_indexed_pair_detector_and_all_original_metadata(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    common = " ".join(f"commonword{i}" for i in range(300))
    records = [
        {
            "id": f"doc{i}", "text": f"prefix{i} {common} suffix{i}",
            "u": f"https://example.org/{i}", "ts": f"201{i}-01-01T00:00:00Z",
            "web-register": {"NA": 0.7}, "doc_scores": {"quality": i}, "seg_langs": ["nno_Latn"],
        } for i in range(3)
    ]
    data = "".join(json.dumps(row) + "\n" for row in records).encode()
    (corpus / "part.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(data))
    index = build_index(corpus, tmp_path / "build", 1, 2)
    store = build_document_store(corpus, tmp_path / "corpus.sqlite", index_dir=index)
    identity = _source_identity(index)
    join = tmp_path / "join"
    (join / "k-sweep").mkdir(parents=True)
    (join / "report.json").write_text(json.dumps({
        "status": "passed", "contract": {"source": identity, "candidate_df_cap": None},
    }))
    (join / "independent_verification.json").write_text(json.dumps({"status": "passed", "source": identity}))
    offsets = np.load(index / "forward_offsets.npy")
    hashes = np.load(index / "forward_fingerprints.npy")
    all_pairs = []
    for first in range(3):
        for second in range(first + 1, 3):
            count = len(np.intersect1d(hashes[offsets[first]:offsets[first + 1]], hashes[offsets[second]:offsets[second + 1]]))
            all_pairs.append((first, second, count))
    dtype = [("doc_i", "<u4"), ("doc_j", "<u4"), ("count", "<u4")]
    for k in (50, 30, 20):
        np.save(join / f"k-sweep/k{k}_exact_counts.npy", np.array([row for row in all_pairs if row[2] >= k], dtype=dtype))
    output = tmp_path / "review"
    prepare_groups(index, join, output)
    selected, views, strata = select_review_pairs(index, join, output)
    assert (selected, views, strata) == select_review_pairs(index, join, output)
    assert sum(row["population"] for row in strata) == 3
    assert len(selected) == 3
    report = review_groups(index, join, store, output)
    assert report["summary"] == {"pairs": 3, "documents": 3, "cluster": 3, "no_cluster": 0, "resource_limit": 0}
    assert report["component_verification"]["status"] == "passed"
    assert report["baseline"]["position_threshold"] == 30
    assert report["baseline"]["offset_threshold"] == 30
    assert report["baseline"]["min_cluster_size"] == 5
    assert report["baseline"]["stored_winnowing_window"] == 6
    assert report["baseline"]["rarity_weighted"] is False
    assert report["metadata"]["0"]["web-register"] == records[0]["web-register"]
    assert report["metadata"]["0"]["doc_scores"] == records[0]["doc_scores"]
    assert report["pairs"][0]["best_passage"]["source_positions_and_exact_kgrams_verified"]
    assert report["pairs"][0]["human_label"] == ""
    assert report["retrieval"]["indexed_document_ids_checked"] == 3
    assert all(view["all_displayed_edges_reviewed"] for view in report["views"])
    assert (output / "human_review.csv").read_text().count("\n") == 4
