"""Bounded default-detector review after complete corpus component analysis."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
from contextlib import ExitStack
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from findmytext.detectors import FingerprintChainDetector
from findmytext.document_store import CorpusDocumentStore
from findmytext.overlap_groups import OverlapGroups, build_overlap_groups
from findmytext.overlap_inverted import _close_on_exit, _source_identity
from findmytext.overlap_stream import atomic_json

K_VALUES = (50, 30, 20)
SEED = 20261026


def prepare_groups(index_dir: Path, join_dir: Path, output: Path) -> dict:
    if output.resolve() == index_dir.resolve() or index_dir.resolve() in output.resolve().parents:
        raise ValueError("Group review output must be outside the fingerprint index")
    identity = _source_identity(index_dir)
    join = json.loads((join_dir / "report.json").read_text())
    check = json.loads((join_dir / "independent_verification.json").read_text())
    if (
        join["status"] != "passed" or check["status"] != "passed"
        or join["contract"]["source"] != identity or check["source"] != identity
        or join["contract"].get("candidate_df_cap") is not None
    ):
        raise ValueError("Require verified, uncapped exact overlaps for this index snapshot")
    with ExitStack() as lifetime:
        offsets = np.load(index_dir / "forward_offsets.npy", mmap_mode="r")
        _close_on_exit(lifetime, offsets)
        documents = len(offsets) - 1
    return {
        str(k): build_overlap_groups(
            join_dir / f"k-sweep/k{k}_exact_counts.npy", documents, k, output / f"k{k}",
        ) for k in K_VALUES
    }


def verify_components(join_dir: Path, output: Path) -> dict:
    """Independent full-edge SciPy check; temporary sparse graphs stay off disk."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    checks = {}
    for k in K_VALUES:
        with OverlapGroups(output / f"k{k}") as groups, ExitStack() as lifetime:
            rows = np.load(join_dir / f"k-sweep/k{k}_exact_counts.npy", mmap_mode="r")
            _close_on_exit(lifetime, rows)
            documents = len(groups.labels)
            graph = csr_matrix(
                (np.ones(len(rows), dtype=np.bool_), (rows["doc_i"], rows["doc_j"])),
                shape=(documents, documents),
            )
            count, labels = connected_components(graph, directed=False)
            minimum = np.full(count, documents, dtype=np.uint32)
            np.minimum.at(minimum, labels, np.arange(documents, dtype=np.uint32))
            np.testing.assert_array_equal(minimum[labels], groups.labels)
            expected_degree = np.diff(graph.indptr).astype(np.uint64)
            expected_degree += np.bincount(graph.indices, minlength=documents).astype(np.uint64)
            np.testing.assert_array_equal(expected_degree, groups.degrees)
            if count != len(groups.groups) or graph.nnz != len(rows):
                raise AssertionError("Independent component/unique-edge accounting differs")
            checks[str(k)] = {
                "status": "passed", "edges_checked": len(rows), "document_labels_checked": documents,
                "degrees_checked": documents, "components": count,
            }
            del graph
    result = {"status": "passed", "method": "SciPy sparse connected_components and independent CSR degree counts", "checks": checks}
    atomic_json(output / "component_verification.json", result)
    return result


def select_review_pairs(index_dir: Path, join_dir: Path, output: Path, *, per_stratum=8, seed=SEED):
    """Uniform random-priority samples per stratum, supplemented by bounded views.

    Strata use exact overlap and the smaller full-forward fingerprint-set size.
    Supplements are purposive, not part of a prevalence/precision estimate.
    """
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    selected = {}

    def add(row, category):
        first, second = int(row["doc_i"]), int(row["doc_j"])
        entry = selected.setdefault((first, second), {
            "doc_i": first, "doc_j": second, "exact_overlap": int(row["count"]),
            "categories": [],
        })
        if category not in entry["categories"]:
            entry["categories"].append(category)

    views = []
    with ExitStack() as lifetime:
        offsets = np.load(index_dir / "forward_offsets.npy", mmap_mode="r")
        _close_on_exit(lifetime, offsets)
        lengths = np.diff(offsets)
        floor = np.load(join_dir / "k-sweep/k20_exact_counts.npy", mmap_mode="r")
        _close_on_exit(lifetime, floor)
        populations = np.zeros(15, dtype=np.uint64)
        reservoirs = [[] for _ in range(15)]
        strongest = []
        rng = np.random.default_rng(seed)
        for start in range(0, len(floor), 250_000):
            rows = floor[start:start + 250_000]
            bands = np.searchsorted([30, 50, 100, 200], rows["count"], side="right")
            smaller = np.minimum(lengths[rows["doc_i"]], lengths[rows["doc_j"]])
            strata = bands * 3 + np.searchsorted([100, 500], smaller, side="right")
            for cell in np.unique(strata):
                eligible = rows[strata == cell]
                populations[cell] += len(eligible)
                priorities = rng.random(len(eligible))
                count = min(per_stratum, len(eligible))
                chosen = np.argpartition(priorities, count - 1)[:count]
                combined = reservoirs[cell] + [
                    (float(priorities[i]), int(eligible[i]["doc_i"]), int(eligible[i]["doc_j"]), int(eligible[i]["count"]))
                    for i in chosen
                ]
                reservoirs[cell] = sorted(combined)[:per_stratum]
            count = min(5, len(rows))
            if count:
                chosen = np.argpartition(rows["count"], len(rows) - count)[-count:]
                for row in rows[chosen]:
                    heapq.heappush(strongest, (int(row["count"]), -int(row["doc_i"]), -int(row["doc_j"])))
                    if len(strongest) > 5:
                        heapq.heappop(strongest)
        strata_report = []
        for cell, reservoir in enumerate(reservoirs):
            name = f"overlap_{['20-29', '30-49', '50-99', '100-199', '200+'][cell // 3]}__smaller_set_{['under100', '100-499', '500+'][cell % 3]}"
            strata_report.append({"stratum": name, "population": int(populations[cell]), "random_pairs": len(reservoir)})
            for _, first, second, count in reservoir:
                add({"doc_i": first, "doc_j": second, "count": count}, "random:" + name)
        for count, first, second in sorted(strongest, reverse=True):
            add({"doc_i": -first, "doc_j": -second, "count": count}, "purposive:strongest_overlap")
        for k in K_VALUES:
            with OverlapGroups(output / f"k{k}") as groups:
                largest = groups.select(limit=1)
                if len(largest):
                    group_id = int(largest[0]["group_id"])
                    members = groups.members(group_id)
                    anchor = int(members[np.argmax(groups.degrees[members])])
                    neighbors = []
                    for start in range(0, len(floor), 1_000_000):
                        rows = floor[start:start + 1_000_000]
                        matches = rows[
                            (rows["count"] >= k)
                            & ((rows["doc_i"] == anchor) | (rows["doc_j"] == anchor))
                        ]
                        for row in matches:
                            other = int(row["doc_j"] if row["doc_i"] == anchor else row["doc_i"])
                            heapq.heappush(neighbors, (int(row["count"]), -other))
                            if len(neighbors) > 5:
                                heapq.heappop(neighbors)
                    nodes = sorted([anchor] + [-other for _, other in neighbors])
                    views.append({
                        "view_id": f"k{k}_largest_ego", "k": k, "group_id": group_id,
                        "component_documents": int(largest[0]["documents"]), "nodes": nodes,
                        "anchor": anchor, "complete_component": len(nodes) == len(members),
                        "selection": "highest-degree anchor and up to five strongest neighbors; induced view",
                    })
                small = groups.select(min_documents=3)
                small = small[small["documents"] <= 5]
                if len(small):
                    group_id = int(small[0]["group_id"])
                    views.append({
                        "view_id": f"k{k}_small_complete", "k": k, "group_id": group_id,
                        "component_documents": int(small[0]["documents"]),
                        "nodes": groups.members(group_id).tolist(), "complete_component": True,
                        "selection": "largest size <=5 and >=3, ties by minimum document ID",
                    })
        for view in views:
            view["edges"] = []
        for start in range(0, len(floor), 250_000):
            rows = floor[start:start + 250_000]
            for view in views:
                matches = rows[
                    (rows["count"] >= view["k"])
                    & np.isin(rows["doc_i"], view["nodes"])
                    & np.isin(rows["doc_j"], view["nodes"])
                ]
                for row in matches:
                    add(row, "view:" + view["view_id"])
                    view["edges"].append({name: int(row[name]) for name in ("doc_i", "doc_j", "count")})
    return list(selected.values()), views, strata_report


def review_groups(index_dir: str | Path, join_dir: str | Path, store_file: str | Path, output_dir: str | Path):
    index, join, output = Path(index_dir).resolve(), Path(join_dir).resolve(), Path(output_dir).resolve()
    if (output / "passage_review.json").exists():
        raise FileExistsError("Passage report already exists; choose a new output or read the existing report")
    started = time.perf_counter()
    identity = _source_identity(index)
    components = prepare_groups(index, join, output)
    component_verification = verify_components(join, output)
    selections, views, strata = select_review_pairs(index, join, output)
    detector = FingerprintChainDetector(str(index))
    documents = sorted({row[key] for row in selections for key in ("doc_i", "doc_j")})
    with CorpusDocumentStore(store_file, index_dir=index) as store:
        lookup_start = time.perf_counter()
        raw = store.get_many(documents, id_kind="internal")
        lookup_seconds = time.perf_counter() - lookup_start
        lookup_start = time.perf_counter()
        metadata = store.get_many(documents, id_kind="internal", metadata_only=True)
        metadata_seconds = time.perf_counter() - lookup_start
        lookup_plan = [
            row[-1] for row in store.connection.execute(
                "EXPLAIN QUERY PLAN SELECT metadata,content FROM documents WHERE id=?", ("example",)
            )
        ]
        if not any("SEARCH" in plan for plan in lookup_plan):
            raise AssertionError("Batch retrieval is not using the document-ID index")
        store_metadata = store.meta
    tokens, positions = {}, {}
    for doc in documents:
        tokens[doc] = detector.index.winnower.tokenize(raw[doc]["text"])
        hashes, starts = detector.index.winnower.get_winnowed_fingerprints(raw[doc]["text"])
        if not np.array_equal(np.unique(hashes), detector.index.get_document_fingerprints(doc)):
            raise AssertionError(f"Source/forward mismatch for document {doc}")
        positions[doc] = dict(zip(map(int, starts), map(int, hashes)))
    results = []
    for selection in selections:
        first, second = selection["doc_i"], selection["doc_j"]
        try:
            evidence = detector.get_pair_passage(first, second)
        except MemoryError as error:
            print(f"[review] RESOURCE LIMIT {first},{second}: {error}", flush=True)
            evidence = {"status": "resource_limit", "error": str(error), "best_passage": None}
        else:
            if evidence["shared_fingerprints"] != selection["exact_overlap"]:
                raise AssertionError("Cached overlap differs from full numeric forward intersection")
        evidence.update(selection)
        evidence["token_lengths"] = [len(tokens[first]), len(tokens[second])]
        evidence["human_label"] = ""
        passage = evidence["best_passage"]
        if passage is not None:
            length = detector.index.winnower.length
            for point in passage["aligned_hashes"]:
                a, b, fingerprint = point["position_doc1"], point["position_doc2"], point["hash"]
                if (
                    positions[first].get(a) != fingerprint or positions[second].get(b) != fingerprint
                    or tokens[first][a:a + length] != tokens[second][b:b + length]
                ):
                    raise AssertionError("Aligned hash/position does not match exact normalized source k-grams")
            a_start, a_stop = passage["token_span_i"]
            b_start, b_stop = passage["token_span_j"]
            a_tokens, b_tokens = tokens[first][a_start:a_stop], tokens[second][b_start:b_stop]
            passage["source_positions_and_exact_kgrams_verified"] = True
            passage["span_preview_i"] = " ".join(a_tokens)[:400]
            passage["span_preview_j"] = " ".join(b_tokens)[:400]
            if max(len(a_tokens), len(b_tokens)) <= 10_000:
                passage["longest_exact_token_run"] = SequenceMatcher(None, a_tokens, b_tokens, autojunk=False).find_longest_match().size
            else:
                passage["sequence_comparison_status"] = "over_10000_token_diagnostic_limit"
        results.append(evidence)
        print(f"[review] {len(results)}/{len(selections)} {first},{second}: {evidence['status']}", flush=True)
    by_pair = {(row["doc_i"], row["doc_j"]): row for row in results}
    for view in views:
        for edge in view["edges"]:
            edge["baseline_status"] = by_pair[(edge["doc_i"], edge["doc_j"])]["status"]
        view["all_displayed_edges_reviewed"] = True
        view["resource_limited_edges"] = sum(edge["baseline_status"] == "resource_limit" for edge in view["edges"])
    if _source_identity(index) != identity:
        raise ValueError("Source index changed during group/passage review")
    report = {
        "status": "completed", "source": identity, "source_unchanged": True,
        "k_values": list(K_VALUES), "components": components,
        "component_verification": component_verification,
        "sampling": {
            "seed": SEED, "per_nonempty_stratum": 8, "strata": strata,
            "strategy": "uniform random-priority samples within overlap/set-size strata, plus purposive view and strongest edges",
            "caveat": "Unweighted, supplemented sample; automatic cluster rate is not precision or a corpus prevalence estimate.",
        },
        "baseline": {
            "method": detector.method, "position_threshold": detector.position_threshold,
            "offset_threshold": detector.offset_threshold, "distance_threshold": detector.distance_threshold,
            "min_cluster_size": detector.min_cluster_size, "rarity_weighted": detector.rarity_weighted,
            "stored_kgram_length": detector.index.winnower.length,
            "stored_winnowing_window": detector.index.winnower.window_size,
            "explicit_pairs_no_top_k": True, "best_surviving_cluster_only": True,
            "production_acceptance_calibrated": False,
        },
        "summary": {
            "pairs": len(results), "documents": len(documents),
            **{status: sum(row["status"] == status for row in results) for status in ("cluster", "no_cluster", "resource_limit")},
        },
        "retrieval": {
            "store_file": str(Path(store_file).resolve()), "raw_records": store_metadata["records"],
            "indexed_document_ids_checked": store_metadata["indexed_document_ids_checked"],
            "batch_documents": len(documents), "full_record_seconds": lookup_seconds,
            "metadata_only_seconds": metadata_seconds, "query_plan": lookup_plan,
            "timing_caveat": "Local filesystem/cache observation, not a cold-cache or production SLA.",
        },
        "metadata": {str(doc): metadata[doc] for doc in documents},
        "views": views, "pairs": results, "seconds": time.perf_counter() - started,
        "direction": "Undirected; raw ts is crawl/provenance time, not established publication time or proof of copying.",
        "future_experiment": "Rehash selected pairs with winnowing window=1 (all 4-grams), then compare passage coverage; not executed.",
    }
    atomic_json(output / "passage_review.json", report)
    fields = ["doc_i", "doc_j", "exact_overlap", "status", "human_label", "human_notes"]
    with (output / "human_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--join-dir", type=Path, required=True)
    parser.add_argument("--store-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = review_groups(args.index_dir, args.join_dir, args.store_file, args.output_dir)
    print(json.dumps({"summary": report["summary"], "retrieval": report["retrieval"]}, indent=2))


if __name__ == "__main__":
    main()
