"""Bounded explicit-pair passage evidence, not calibrated copyright decisions."""

from __future__ import annotations

import argparse
import heapq
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from experiments.similarity.compare_similarity_methods import build_index
from findmytext import utils
from findmytext.detectors import FingerprintChainDetector
from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_stream import atomic_json


def select_pairs(stage: Path, thresholds=(50, 30, 20, 10, 5)) -> list[dict]:
    reference = np.load(stage / "reference/counts.npy", mmap_mode="r")
    keys = np.load(stage / "reference/keys.npy", mmap_mode="r")
    checkpoint = json.loads((stage / "k5_exact_counts_df1000/checkpoint.json").read_text())
    if checkpoint["status"] != "complete":
        raise ValueError("Passage selection requires a completed capped run")
    seen = np.zeros(len(reference), dtype=np.bool_)
    for part in checkpoint["parts"]:
        rows = np.load(stage / "k5_exact_counts_df1000" / part["file"])
        pair_keys = (rows["doc_i"].astype(np.uint64) << np.uint64(32)) | rows["doc_j"]
        positions = np.searchsorted(keys, pair_keys)
        if np.any(positions >= len(keys)) or not np.array_equal(keys[positions], pair_keys):
            raise ValueError("Capped pair is absent from the independent reference")
        seen[positions] = True
    selections = {}

    def add(row, category):
        pair = int(row["doc_i"]), int(row["doc_j"])
        entry = selections.setdefault(pair, {
            "doc_i": pair[0], "doc_j": pair[1], "exact_overlap": int(row["count"]),
            "categories": [],
        })
        if category not in entry["categories"]:
            entry["categories"].append(category)

    top = []
    omitted_top = []
    boundaries = {k: [] for k in thresholds}
    omitted_boundaries = {k: [] for k in thresholds}
    for start in range(0, len(reference), 1_000_000):
        rows = reference[start:start + 1_000_000]
        counts = rows["count"]
        if not len(rows):
            continue
        n = min(5, len(rows))
        for index in np.argpartition(counts, len(rows) - n)[-n:]:
            row = rows[index]
            item = (int(row["count"]), -int(row["doc_i"]), -int(row["doc_j"]))
            heapq.heappush(top, item)
            if len(top) > 5:
                heapq.heappop(top)
        missing = ~seen[start:start + len(rows)]
        eligible_missing = np.flatnonzero(missing & (counts >= 5))
        if len(eligible_missing):
            count = min(3, len(eligible_missing))
            selected = eligible_missing[np.argpartition(
                counts[eligible_missing], len(eligible_missing) - count
            )[-count:]]
            for index in selected:
                row = rows[index]
                heapq.heappush(omitted_top, (int(row["count"]), -int(row["doc_i"]), -int(row["doc_j"])))
                if len(omitted_top) > 3:
                    heapq.heappop(omitted_top)
        for k in thresholds:
            for mask, target in ((counts >= k, boundaries[k]), (missing & (counts >= k), omitted_boundaries[k])):
                indices = np.flatnonzero(mask)
                if len(indices):
                    selected = indices[np.argsort(counts[indices], kind="stable")[:2]]
                    target.extend((int(rows[i]["count"]), int(rows[i]["doc_i"]), int(rows[i]["doc_j"])) for i in selected)
                    target.sort()
                    del target[2:]
    for count, first, second in top:
        add({"doc_i": -first, "doc_j": -second, "count": count}, "high_overlap")
    for count, first, second in omitted_top:
        add({"doc_i": -first, "doc_j": -second, "count": count}, "high_overlap_omitted_by_cap")
    for k in thresholds:
        for count, first, second in boundaries[k]:
            add({"doc_i": first, "doc_j": second, "count": count}, f"near_k{k}")
        for count, first, second in omitted_boundaries[k]:
            add({"doc_i": first, "doc_j": second, "count": count}, f"cap_omitted_near_k{k}")
    return sorted(selections.values(), key=lambda row: (-row["exact_overlap"], row["doc_i"], row["doc_j"]))


def synthetic_controls(output_dir: Path) -> dict:
    output_dir.mkdir()
    aligned = " ".join(f"alignedword{i}" for i in range(180))
    template = " ".join(f"templateword{i}" for i in range(100))
    blocks = [f"boiler{i} shared{i} notice{i} word{i}" for i in range(64)]
    scattered_a = " ".join(
        block + " " + " ".join(f"uniqa{i}_{j}" for j in range(60))
        for i, block in enumerate(blocks)
    )
    scattered_b = " ".join(
        block + " " + " ".join(f"uniqb{i}_{j}" for j in range(60))
        for i, block in reversed(list(enumerate(blocks)))
    )
    records = [
        {"id": "aligned_a", "text": "differentprefix " + aligned + " endinga"},
        {"id": "aligned_b", "text": "anotherprefix " + aligned + " endingb"},
        {"id": "scattered_a", "text": scattered_a},
        {"id": "scattered_b", "text": scattered_b},
        {"id": "template_a", "text": "uniquea " + template + " lasta"},
        {"id": "template_b", "text": "uniqueb " + template + " lastb"},
    ]
    corpus = output_dir / "controls.jsonl"
    corpus.write_text("".join(json.dumps(record) + "\n" for record in records))
    index_dir = build_index(str(corpus), str(output_dir))
    detector = FingerprintChainDetector(index_dir)
    mapping = {detector.index.to_external_doc_id[i]: i for i in range(detector.num_documents)}
    report = {}
    for name, first, second in (
        ("aligned_positive", "aligned_a", "aligned_b"),
        ("scattered_shared_grams_negative", "scattered_a", "scattered_b"),
        ("contiguous_template_semantic_negative", "template_a", "template_b"),
    ):
        report[name] = detector.get_pair_passage(mapping[first], mapping[second])
    assert report["aligned_positive"]["status"] == "cluster"
    assert report["scattered_shared_grams_negative"]["shared_fingerprints"] >= 5
    assert report["scattered_shared_grams_negative"]["status"] == "no_cluster"
    assert report["contiguous_template_semantic_negative"]["status"] == "cluster"
    report["interpretation"] = (
        "Scattered shared 4-grams are rejected; contiguous template text still forms "
        "a real aligned passage. Alignment alone does not establish interesting reuse or copyright."
    )
    atomic_json(output_dir / "report.json", report)
    del detector
    return report


def inspect_passages(
    index_dir: str | Path, corpus_dir: str | Path,
    benchmark_stage: str | Path, output_dir: str | Path,
) -> dict:
    source = Path(index_dir).resolve()
    output = Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError("Inspection output must be outside the source index")
    if output.exists():
        raise FileExistsError(output)
    identity = _source_identity(source)
    suite = json.loads((Path(benchmark_stage).parent / "report.json").read_text())
    if suite["contract"]["source"] != identity:
        raise ValueError("Passage benchmark and source index do not match")
    output.mkdir(parents=True)
    started = time.perf_counter()
    selected = select_pairs(Path(benchmark_stage))
    detector = FingerprintChainDetector(str(source), min_cluster_size=5)
    wanted = {
        detector.index.to_external_doc_id[doc]: doc
        for row in selected for doc in (row["doc_i"], row["doc_j"])
    }
    loaded = {}
    print(f"[passage inspection] {len(selected)} pairs, {len(wanted)} unique source documents", flush=True)
    for shard in sorted(Path(corpus_dir).glob("*.jsonl.zst")):
        stream = utils.stream_json_zst(str(shard), min_length=100, max_length=100_000)
        try:
            for record in stream:
                external = str(record["id"])
                if external in wanted:
                    doc = wanted[external]
                    tokens = detector.index.winnower.tokenize(record["text"])
                    values, positions = detector.index.winnower.get_winnowed_fingerprints(record["text"])
                    if not np.array_equal(np.unique(values), detector.index.get_document_fingerprints(doc)):
                        raise AssertionError(f"Source/forward mismatch for document {doc}")
                    loaded[doc] = {
                        "tokens": tokens,
                        "selected_positions": dict(zip(map(int, positions), map(int, values))),
                        "external_id": external, "shard": shard.name,
                    }
                    if len(loaded) == len(wanted):
                        break
        finally:
            stream.close()
        print(f"[passage inspection] loaded {len(loaded)}/{len(wanted)} after {shard.name}", flush=True)
        if len(loaded) == len(wanted):
            break
    if len(loaded) != len(wanted):
        raise ValueError("Selected source documents were not found in the local corpus")
    results = []
    for selection in selected:
        first, second = selection["doc_i"], selection["doc_j"]
        try:
            evidence = detector.get_pair_passage(first, second)
        except MemoryError as error:
            print(f"[passage inspection] pair {first},{second}: RESOURCE LIMIT: {error}", flush=True)
            results.append({**selection, "status": "resource_limit", "error": str(error)})
            continue
        if evidence["shared_fingerprints"] != selection["exact_overlap"]:
            raise AssertionError("Passage pair's shared hashes differ from the independent reference")
        evidence.update(selection)
        evidence["source_documents"] = [
            {key: loaded[doc][key] for key in ("external_id", "shard")} for doc in (first, second)
        ]
        if evidence["best_passage"] is not None:
            passage = evidence["best_passage"]
            kgram_length = detector.index.winnower.length
            for point in passage["aligned_hashes"]:
                a, b, fingerprint = point["position_doc1"], point["position_doc2"], point["hash"]
                if (
                    loaded[first]["selected_positions"].get(a) != fingerprint
                    or loaded[second]["selected_positions"].get(b) != fingerprint
                    or loaded[first]["tokens"][a:a + kgram_length] != loaded[second]["tokens"][b:b + kgram_length]
                ):
                    raise AssertionError("Stored aligned fingerprint/position differs from exact source tokens")
            a_start, a_stop = passage["token_span_i"]
            b_start, b_stop = passage["token_span_j"]
            a_tokens = loaded[first]["tokens"][a_start:a_stop]
            b_tokens = loaded[second]["tokens"][b_start:b_stop]
            if max(len(a_tokens), len(b_tokens)) <= 10_000:
                matcher = SequenceMatcher(None, a_tokens, b_tokens, autojunk=False)
                passage["longest_exact_token_run"] = matcher.find_longest_match().size
                passage["normalized_span_similarity"] = matcher.ratio()
            else:
                passage["sequence_comparison_status"] = "span_exceeds_10000_token_inspection_limit"
            passage["source_positions_and_exact_kgrams_verified"] = True
            passage["short_span_prefix_i"] = " ".join(a_tokens)[:80]
            passage["short_span_prefix_j"] = " ".join(b_tokens)[:80]
            evidence["exploratory_label"] = (
                "aligned_text_evidence" if passage.get("longest_exact_token_run", 0) >= 20
                else "cluster_without_20_token_exact_run"
            )
        else:
            evidence["exploratory_label"] = "no_surviving_positional_cluster"
        results.append(evidence)
        print(f"[passage inspection] {first},{second}: overlap={selection['exact_overlap']} status={evidence['status']}", flush=True)
    controls = synthetic_controls(output / "synthetic_controls")
    if _source_identity(source) != identity:
        raise ValueError("Source index changed during passage inspection")
    report = {
        "status": "completed_with_resource_limits" if any(row["status"] == "resource_limit" for row in results) else "passed",
        "source": identity, "source_unchanged": True,
        "benchmark_stage": str(Path(benchmark_stage).resolve()),
        "sampling": "bounded purposive high-overlap, near-K and cap-omitted pairs; not a precision sample",
        "selection_k_values": [50, 30, 20, 10, 5],
        "pairs_selected": len(selected), "source_documents_checked": len(loaded),
        "parameters": {
            "method": "rectangle", "position_threshold": 30, "offset_threshold": 30,
            "minimum_distinct_support": 5, "exploratory_exact_token_run": 20,
            "production_acceptance_calibrated": False,
        },
        "summary": {
            "clusters": sum(row["status"] == "cluster" for row in results),
            "no_cluster": sum(row["status"] == "no_cluster" for row in results),
            "resource_limited": sum(row["status"] == "resource_limit" for row in results),
        },
        "pairs": results,
        "synthetic_controls": {key: value["status"] for key, value in controls.items() if isinstance(value, dict)},
        "interpretation": controls["interpretation"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("benchmark_stage", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inspect_passages(args.index_dir, args.corpus_dir, args.benchmark_stage, args.output_dir)


if __name__ == "__main__":
    main()
