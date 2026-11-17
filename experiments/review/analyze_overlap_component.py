"""Explain a complete overlap component with exact diagnostics and bounded text review."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time
from typing import TypedDict
from urllib.parse import urlsplit

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from findmytext.detectors import FingerprintChainDetector
from findmytext.document_store import CorpusDocumentStore
from findmytext.overlap_groups import OverlapGroups
from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_stream import atomic_json


class SelectedEdge(TypedDict):
    doc_i: int
    doc_j: int
    count: int
    edge_class: str


def layout_census(store, document_ids):
    counts, lengths = [], []
    for _, record in store.iter_many(document_ids, id_kind="internal"):
        counts.append(sum(line.strip() == "Detaljer" for line in record["text"].splitlines()))
        lengths.append(len(record["text"]))
    return {
        "documents": len(counts),
        "documents_with_at_least_3_standalone_detaljer_lines": sum(count >= 3 for count in counts),
        "documents_with_at_least_50_standalone_detaljer_lines": sum(count >= 50 for count in counts),
        "standalone_detaljer_lines_min_median_max": list(map(float, np.percentile(counts, [0, 50, 100]))) if counts else [],
        "raw_characters_min_median_max": list(map(float, np.percentile(lengths, [0, 50, 100]))) if lengths else [],
        "interpretation": "Exact raw-text layout counts, not validated classification of main content versus recommendations.",
    }


def component_statistics(nodes, edges):
    """Include isolated retained vertices in an induced/filtered graph."""
    nodes = np.asarray(nodes, dtype=np.uint32)
    if len(nodes) and np.any(nodes[1:] <= nodes[:-1]):
        raise ValueError("Component nodes must be strictly sorted and unique")
    positions_i = np.searchsorted(nodes, edges["doc_i"])
    positions_j = np.searchsorted(nodes, edges["doc_j"])
    if len(edges) and (
        np.any(positions_i >= len(nodes)) or np.any(positions_j >= len(nodes))
        or not np.array_equal(nodes[positions_i], edges["doc_i"])
        or not np.array_equal(nodes[positions_j], edges["doc_j"])
    ):
        raise ValueError("An edge endpoint is absent from retained nodes")
    graph = csr_matrix(
        (np.ones(len(edges), dtype=np.bool_), (positions_i, positions_j)),
        shape=(len(nodes), len(nodes)),
    )
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    return {
        "documents": len(nodes), "edges": len(edges),
        "components_including_isolates": count, "isolates": int(np.count_nonzero(sizes == 1)),
        "nontrivial_components": int(np.count_nonzero(sizes >= 2)),
        "largest_components": sorted(map(int, sizes), reverse=True)[:10],
    }


@njit(cache=True)
def _common_signatures(members, offsets, hashes, excluded):
    signatures = np.zeros((len(members), (len(excluded) + 63) // 64), dtype=np.uint64)
    for row, doc in enumerate(members):
        for position in range(offsets[doc], offsets[doc + 1]):
            fingerprint = hashes[position]
            rank = np.searchsorted(excluded, fingerprint)
            if rank < len(excluded) and excluded[rank] == fingerprint:
                signatures[row, rank // 64] |= np.uint64(1) << np.uint64(rank % 64)
    return signatures


@njit(cache=True)
def _excluded_shared(edges, local_ids, signatures):
    counts = np.zeros(len(edges), dtype=np.uint32)
    for i, edge in enumerate(edges):
        first, second = local_ids[edge["doc_i"]], local_ids[edge["doc_j"]]
        count = 0
        for word in range(signatures.shape[1]):
            bits = signatures[first, word] & signatures[second, word]
            while bits:
                bits &= bits - np.uint64(1)
                count += 1
        counts[i] = count
    return counts


def marker_regions(text, winnower):
    """Token boundaries of standalone Detaljer lines, not validated main-text extraction."""
    marker_starts = []
    token_cursor = 0
    for line in text.splitlines(keepends=True):
        if line.strip() == "Detaljer":
            marker_starts.append(token_cursor)
        token_cursor += len(winnower.tokenize(line))
    if token_cursor != len(winnower.tokenize(text)):
        raise ValueError("Line-wise and full-text tokenization do not agree")
    return marker_starts


def analyze_component(index_dir, review_dir, store_file, output_dir, *, k=50, seed=20261026, pairs_per_class=20):
    index_dir, review_dir, output = Path(index_dir).resolve(), Path(review_dir).resolve(), Path(output_dir).resolve()
    if output == index_dir or index_dir in output.parents:
        raise ValueError("Analysis output must be outside the fingerprint index")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    source = _source_identity(index_dir)
    with OverlapGroups(review_dir / f"k{k}") as groups:
        largest = groups.select(limit=1)[0]
        group_id = int(largest["group_id"])
        members = groups.members(group_id).copy()
        labels = groups.labels.copy()
        degree = groups.degrees[members].copy()
        parent_report = groups.report
        source_edges = Path(groups.report["contract"]["edge_file"])
        accounting = {
            "all_documents": len(groups.labels), "isolates": groups.report["isolates"],
            "documents_with_partners": groups.report["documents_with_partners"],
            "nontrivial_groups": groups.report["nontrivial_components"],
            "exactly_two_document_groups": int(np.count_nonzero(groups.groups["documents"] == 2)),
            "groups_with_at_least_three": int(np.count_nonzero(groups.groups["documents"] >= 3)),
            "documents_in_groups_of_at_least_three": int(groups.groups["documents"][groups.groups["documents"] >= 3].sum()),
        }
    accounting["documents_in_two_document_groups"] = 2 * accounting["exactly_two_document_groups"]
    domains, urls, crawl_ids = {}, {}, Counter()
    with CorpusDocumentStore(store_file, index_dir=index_dir) as store:
        for doc, metadata in store.iter_many(members, id_kind="internal", metadata_only=True):
            urls[doc] = metadata.get("u", "")
            domains[doc] = urlsplit(urls[doc]).hostname or "(missing)"
            crawl_ids[str(metadata.get("crawl_id", "(missing)"))] += 1
        print(f"[component] metadata census complete: {len(domains):,} documents", flush=True)
        outside = np.flatnonzero(labels != group_id)
        rng = np.random.default_rng(seed)
        comparison_ids = np.sort(rng.choice(outside, size=min(2000, len(outside)), replace=False))
        comparison = Counter(
            urlsplit(meta.get("u", "")).hostname or "(missing)"
            for _, meta in store.iter_many(comparison_ids, id_kind="internal", metadata_only=True)
        )
        rows = np.load(source_edges, mmap_mode="r")
        edges = rows[labels[rows["doc_i"]] == group_id]
        if len(edges) != largest["edges"] or not np.all(labels[edges["doc_j"]] == group_id):
            raise AssertionError("Complete group edge accounting differs")
        host_counts = Counter(domains.values())
        host_names = sorted(host_counts)
        host_rank = {name: i for i, name in enumerate(host_names)}
        codes = np.zeros(len(labels), dtype=np.uint32)
        for doc, host in domains.items():
            codes[doc] = host_rank[host]
        first, second = codes[edges["doc_i"]], codes[edges["doc_j"]]
        host_keys = np.minimum(first, second).astype(np.uint64) * len(host_names) + np.maximum(first, second)
        keys, key_counts = np.unique(host_keys, return_counts=True)
        order = np.argsort(-key_counts, kind="stable")[:20]
        edge_domains = [
            {"host_i": host_names[int(keys[i]) // len(host_names)], "host_j": host_names[int(keys[i]) % len(host_names)], "edges": int(key_counts[i])}
            for i in order
        ]
        is_docplayer = np.zeros(len(labels), dtype=np.bool_)
        for doc, host in domains.items():
            is_docplayer[doc] = host == "docplayer.me"
        classes = is_docplayer[edges["doc_i"]].astype(np.uint8) + is_docplayer[edges["doc_j"]].astype(np.uint8)
        class_names = {2: "docplayer_docplayer", 1: "docplayer_other", 0: "other_other"}
        class_counts = {class_names[i]: int(np.count_nonzero(classes == i)) for i in (2, 1, 0)}
        without_docplayer = component_statistics(members[~is_docplayer[members]], edges[classes == 0])
        offsets = np.load(index_dir / "forward_offsets.npy", mmap_mode="r")
        hashes = np.load(index_dir / "forward_fingerprints.npy", mmap_mode="r")
        lengths = np.diff(offsets)[members]
        memberships = np.concatenate([hashes[offsets[doc]:offsets[doc + 1]] for doc in members])
        fingerprints, df = np.unique(memberships, return_counts=True)
        del memberships
        top = np.lexsort((fingerprints, -df))[:30]
        common = fingerprints[df > 1000]
        signatures = _common_signatures(members, offsets, hashes, common)
        local_ids = np.full(len(labels), -1, dtype=np.int32)
        local_ids[members] = np.arange(len(members))
        excluded_counts = _excluded_shared(edges, local_ids, signatures)
        if np.any(excluded_counts > edges["count"]):
            raise AssertionError("Excluded intersection exceeds verified full overlap")
        validation_rng = np.random.default_rng(seed + 1)
        sample_positions = validation_rng.choice(len(edges), size=min(1000, len(edges)), replace=False)
        for position in sample_positions:
            first_doc, second_doc = int(edges[position]["doc_i"]), int(edges[position]["doc_j"])
            shared = np.intersect1d(
                hashes[offsets[first_doc]:offsets[first_doc + 1]],
                hashes[offsets[second_doc]:offsets[second_doc + 1]], assume_unique=True,
            )
            removed = len(np.intersect1d(shared, common, assume_unique=True))
            if len(shared) != edges[position]["count"] or removed != excluded_counts[position]:
                raise AssertionError("Bit-signature diagnostic differs from independent NumPy intersections")
        keep = edges["count"] - excluded_counts >= k
        common_sensitivity = component_statistics(members, edges[keep])
        common_sensitivity.update({
            "excluded_fingerprints": len(common),
            "rule": "exclude hashes present in >1000 documents within this component; not a full-corpus df cap",
            "edges_retained_fraction": float(np.count_nonzero(keep) / len(edges)),
            "median_excluded_shared_fingerprints_per_edge": float(np.median(excluded_counts)),
            "changes_production_indexes_or_graph": False,
            "independent_numpy_intersection_samples_checked": len(sample_positions),
        })
        del signatures, excluded_counts, keep
        detector = FingerprintChainDetector(str(index_dir))
        top_hashes = fingerprints[top]
        posting_lists = detector.index._get_postings(top_hashes)
        examples, needed = [], set()
        for rank in top:
            fingerprint = int(fingerprints[rank])
            postings = posting_lists[np.uint64(fingerprint)]
            in_group = postings[labels[postings["doc_id"]] == group_id]
            example_doc = int(in_group[0]["doc_id"])
            example_position = int(in_group[0]["position"])
            examples.append({
                "hash": fingerprint, "component_df": int(df[rank]),
                "full_corpus_df": len(np.unique(postings["doc_id"])),
                "example_doc": example_doc, "example_position": example_position,
            })
            needed.add(example_doc)
        selections: list[SelectedEdge] = []
        for category in (2, 1, 0):
            eligible = np.flatnonzero(classes == category)
            chosen = rng.choice(eligible, size=min(pairs_per_class, len(eligible)), replace=False)
            for edge in edges[chosen]:
                row: SelectedEdge = {
                    "doc_i": int(edge["doc_i"]), "doc_j": int(edge["doc_j"]),
                    "count": int(edge["count"]), "edge_class": class_names[category],
                }
                selections.append(row)
                needed.update((row["doc_i"], row["doc_j"]))
        raw = store.get_many(sorted(needed), id_kind="internal")
        layout = layout_census(store, members[is_docplayer[members]])
        print("[component] full Docplayer raw-text layout census complete", flush=True)
    tokens, selected_positions, markers = {}, {}, {}
    for key, record in raw.items():
        if not isinstance(key, int):
            raise TypeError("Internal-ID document retrieval returned a non-integer key")
        doc = key
        tokens[doc] = detector.index.winnower.tokenize(record["text"])
        values, positions = detector.index.winnower.get_winnowed_fingerprints(record["text"])
        if not np.array_equal(np.unique(values), detector.index.get_document_fingerprints(doc)):
            raise AssertionError(f"Raw text and forward index differ for {doc}")
        selected_positions[doc] = (values, positions)
        markers[doc] = marker_regions(record["text"], detector.index.winnower)
    for example in examples:
        doc, position = example["example_doc"], example["example_position"]
        example["kgram"] = " ".join(tokens[doc][position:position + detector.index.winnower.length])
        values, positions = selected_positions[doc]
        if not np.any((positions == position) & (values == example["hash"])):
            raise AssertionError("Top fingerprint source position differs")
        example["source_url"] = urls[doc]
    reviewed = []
    for selection in selections:
        first, second = selection["doc_i"], selection["doc_j"]
        try:
            evidence = detector.get_pair_passage(first, second)
        except MemoryError as error:
            evidence = {"status": "resource_limit", "best_passage": None, "error": str(error)}
        else:
            if evidence["shared_fingerprints"] != selection["count"]:
                raise AssertionError("Sampled edge's exact overlap differs")
        evidence.update(selection)
        evidence["urls"] = [urls[first], urls[second]]
        evidence["standalone_detaljer_lines"] = [len(markers[first]), len(markers[second])]
        evidence["token_lengths"] = [len(tokens[first]), len(tokens[second])]
        prefix_hashes = []
        for doc in (first, second):
            values, positions = selected_positions[doc]
            cutoff = markers[doc][0] if domains[doc] == "docplayer.me" and len(markers[doc]) >= 3 else len(tokens[doc])
            prefix_hashes.append(np.unique(values[positions + detector.index.winnower.length <= cutoff]))
        evidence["first_block_only_overlap_diagnostic"] = len(np.intersect1d(*prefix_hashes, assume_unique=True))
        passage = evidence["best_passage"]
        if passage:
            for point in passage["aligned_hashes"]:
                a, b = point["position_doc1"], point["position_doc2"]
                length = detector.index.winnower.length
                a_values, a_positions = selected_positions[first]
                b_values, b_positions = selected_positions[second]
                if (
                    tokens[first][a:a + length] != tokens[second][b:b + length]
                    or not np.any((a_values == point["hash"]) & (a_positions == a))
                    or not np.any((b_values == point["hash"]) & (b_positions == b))
                ):
                    raise AssertionError("Aligned source k-grams differ")
            for doc, suffix in ((first, "i"), (second, "j")):
                a, b = passage["token_span_" + suffix]
                passage["preview_" + suffix] = " ".join(tokens[doc][a:b])[:400]
                passage["detaljer_markers_before_start_" + suffix] = int(np.searchsorted(markers[doc], a, side="right"))
                passage["start_fraction_" + suffix] = a / len(tokens[doc])
        reviewed.append(evidence)
        print(f"[component] reviewed {len(reviewed)}/{len(selections)} {selection['edge_class']}: {evidence['status']}", flush=True)
    by_class = {}
    for category in class_names.values():
        chosen = [row for row in reviewed if row["edge_class"] == category]
        by_class[category] = {
            "sampled": len(chosen),
            "clusters": sum(row["status"] == "cluster" for row in chosen),
            "unresolved": sum(row["status"] == "resource_limit" for row in chosen),
            "below_k_after_first_block_only_diagnostic": sum(row["first_block_only_overlap_diagnostic"] < k for row in chosen),
            "clusters_starting_after_detaljer_in_at_least_one_doc": sum(
                row["best_passage"] is not None and any(
                    domains[row["doc_" + suffix]] == "docplayer.me"
                    and row["best_passage"]["detaljer_markers_before_start_" + suffix] > 0
                    for suffix in ("i", "j")
                ) for row in chosen
            ),
        }
    if _source_identity(index_dir) != source or source_edges.stat().st_mtime_ns != parent_report["contract"]["edge_mtime_ns"]:
        raise ValueError("Source index or edges changed during component analysis")
    report = {
        "status": "complete", "source": source, "source_unchanged": True,
        "k": k, "group_id": group_id, "accounting": accounting,
        "component": {
            "documents": len(members), "edges": len(edges),
            "fraction_of_all_k_edges": len(edges) / parent_report["edges"],
            "density": 2 * len(edges) / (len(members) * (len(members) - 1)),
            "degree_min_median_max": list(map(float, np.percentile(degree, [0, 50, 100]))),
            "fingerprint_set_size_min_median_max": list(map(float, np.percentile(lengths, [0, 50, 100]))),
        },
        "domains": [{"host": host, "documents": count} for host, count in host_counts.most_common()],
        "outside_comparison": {"selection": "uniform seeded document sample outside giant component, including isolates", "documents": len(comparison_ids), "domains": comparison.most_common(20)},
        "crawl_ids": crawl_ids.most_common(),
        "edge_classes": class_counts, "top_domain_pairs": edge_domains,
        "without_docplayer_induced_graph": without_docplayer,
        "docplayer_layout_census": layout,
        "fingerprints": {
            "unique": len(fingerprints), "memberships": int(lengths.sum()),
            "maximum_component_df": int(df.max()),
            "present_in_every_component_document": int(np.count_nonzero(df == len(members))),
            "top": examples,
        },
        "common_hash_removal_diagnostic": common_sensitivity,
        "text_review": {
            "seed": seed, "selection": "uniform edge samples within each of three host-pair classes; not a prevalence estimate",
            "by_edge_class": by_class, "pairs": reviewed,
            "first_block_only_caveat": "Diagnostic only: retain indexed hashes fully before the first standalone Detaljer line on Docplayer records with >=3 such lines. This is not validated main-content extraction; a record may contain recommendation snippets only.",
            "defaults_unchanged": True, "stored_window": detector.index.winnower.window_size,
        },
        "seconds": time.perf_counter() - started,
    }
    atomic_json(output / "report.json", report)
    print("[component] completed:", output / "report.json", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--store-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze_component(args.index_dir, args.review_dir, args.store_file, args.output_dir)
