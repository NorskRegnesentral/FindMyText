"""Exact host-level giant-component statistics and bounded source-verified examples."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import numpy as np

from experiments.analyze_overlap_component import (
    _common_signatures, _excluded_shared, component_statistics,
)
from findmytext.detectors import FingerprintChainDetector
from findmytext.document_store import CorpusDocumentStore
from findmytext.overlap_groups import OverlapGroups
from findmytext.overlap_inverted import _source_identity
from findmytext.overlap_stream import atomic_json
from findmytext.web import tokenize_with_offsets

HOSTS = ("www.helsetilsynet.no", "www.skogholt.org", "www.regjeringen.no")
SEED = 20261026


def skogholt_page_role(url):
    path = urlsplit(url).path
    if (
        path == "/" or path.startswith(("/tema/", "/merke/", "/author/", "/page/"))
        or re.fullmatch(r"/\d{4}(?:/\d{2}){0,2}/?", path)
    ):
        return "archive_or_listing"
    if re.match(r"^/\d{4}/\d{2}/[^/]+", path):
        return "dated_article"
    return "other_or_unknown"


def choose_edges(rows, same_url, external, partner_is_docplayer, rng):
    """Keep random, strongest and near-threshold examples distinct in provenance."""
    selections = {}

    def add(indices, category):
        for position in indices:
            row = rows[position]
            key = int(row["doc_i"]), int(row["doc_j"])
            entry = selections.setdefault(key, {
                "doc_i": key[0], "doc_j": key[1], "count": int(row["count"]), "categories": [],
            })
            if category not in entry["categories"]:
                entry["categories"].append(category)

    def random(mask, count, category):
        eligible = np.flatnonzero(mask)
        add(rng.choice(eligible, size=min(count, len(eligible)), replace=False), category)

    def strongest(mask, count, category):
        eligible = np.flatnonzero(mask)
        order = np.lexsort((rows["doc_j"][eligible], rows["doc_i"][eligible], -rows["count"][eligible].astype(np.int64)))
        add(eligible[order[:count]], category)

    within = ~external
    random(within & same_url, 5, "random_same_url")
    random(within & ~same_url, 20, "random_different_urls")
    strongest(within & ~same_url, 3, "strongest_different_urls")
    random(within & ~same_url & (rows["count"] < 80), 5, "random_near_k50")
    random(external & partner_is_docplayer, 5, "random_docplayer_bridge")
    strongest(external & partner_is_docplayer, 1, "strongest_docplayer_bridge")
    random(external & ~partner_is_docplayer, 5, "random_other_host_bridge")
    strongest(external & ~partner_is_docplayer, 1, "strongest_other_host_bridge")
    return list(selections.values())


def repeated_lines(records):
    counts = Counter()
    for record in records:
        counts.update({
            line.strip() for line in record["text"].splitlines()
            if 20 <= len(line.strip()) <= 220 and len(line.split()) >= 4
        })
    return [{"line": line, "documents": count} for line, count in counts.most_common(15)]


def analyze_hosts(index_dir, review_dir, store_file, output_dir):
    index, review, output = Path(index_dir).resolve(), Path(review_dir).resolve(), Path(output_dir).resolve()
    if output == index or index in output.parents:
        raise ValueError("Host analysis must be outside the fingerprint index")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    identity = _source_identity(index)
    with OverlapGroups(review / "k50") as groups:
        group_id = int(groups.select(limit=1)[0]["group_id"])
        members, labels = groups.members(group_id).copy(), groups.labels.copy()
        edge_file = Path(groups.report["contract"]["edge_file"])
        edge_identity = groups.report["contract"]
    offsets = np.load(index / "forward_offsets.npy", mmap_mode="r")
    hashes = np.load(index / "forward_fingerprints.npy", mmap_mode="r")
    rows = np.load(edge_file, mmap_mode="r")
    edges = rows[labels[rows["doc_i"]] == group_id]
    if not np.all(labels[edges["doc_j"]] == group_id):
        raise ValueError("Edge endpoints disagree about component membership")
    metadata = {}
    with CorpusDocumentStore(store_file, index_dir=index) as store:
        for doc, record in store.iter_many(members, id_kind="internal", metadata_only=True):
            metadata[doc] = {
                "host": urlsplit(record.get("u", "")).hostname or "(missing)",
                "url": record.get("u", ""), "ts": record.get("ts"), "crawl_id": record.get("crawl_id"),
            }
        print("[host study] complete giant-component metadata census", flush=True)
        global_selections = {}
        reports = {}
        top_requests = []
        needed = set()
        for host_number, host in enumerate(HOSTS):
            mine = np.array([doc for doc in members if metadata[int(doc)]["host"] == host], dtype=np.uint32)
            mine_mask = np.zeros(len(labels), dtype=np.bool_)
            mine_mask[mine] = True
            left, right = mine_mask[edges["doc_i"]], mine_mask[edges["doc_j"]]
            host_edges = edges[left | right]
            within = mine_mask[host_edges["doc_i"]] & mine_mask[host_edges["doc_j"]]
            external = ~within
            target = np.where(mine_mask[host_edges["doc_i"]], host_edges["doc_i"], host_edges["doc_j"])
            partner = np.where(mine_mask[host_edges["doc_i"]], host_edges["doc_j"], host_edges["doc_i"])
            same_url = np.array([
                bool(metadata[int(row["doc_i"])]["url"])
                and metadata[int(row["doc_i"])]["url"] == metadata[int(row["doc_j"])]["url"]
                for row in host_edges
            ])
            partner_docplayer = np.array([metadata[int(doc)]["host"] == "docplayer.me" for doc in partner])
            partner_counts = Counter(metadata[int(doc)]["host"] for doc in partner[external])
            degree = np.zeros(len(labels), dtype=np.uint32)
            np.add.at(degree, host_edges["doc_i"], 1)
            np.add.at(degree, host_edges["doc_j"], 1)
            values, df = np.unique(
                np.concatenate([hashes[offsets[doc]:offsets[doc + 1]] for doc in mine]), return_counts=True,
            )
            common = values[df >= math.ceil(len(mine) / 2)]
            local_nodes = np.unique(np.r_[host_edges["doc_i"], host_edges["doc_j"]])
            local_ids = np.full(len(labels), -1, dtype=np.int32)
            local_ids[local_nodes] = np.arange(len(local_nodes))
            signatures = _common_signatures(local_nodes, offsets, hashes, common)
            removed = _excluded_shared(host_edges, local_ids, signatures)
            if np.any(removed > host_edges["count"]):
                raise AssertionError("Host-common intersection exceeds full overlap")
            rng = np.random.default_rng(SEED + host_number)
            validation_positions = rng.choice(len(host_edges), size=min(100, len(host_edges)), replace=False)
            for position in validation_positions:
                a, b = int(host_edges[position]["doc_i"]), int(host_edges[position]["doc_j"])
                shared = np.intersect1d(hashes[offsets[a]:offsets[a + 1]], hashes[offsets[b]:offsets[b + 1]], assume_unique=True)
                if len(shared) != host_edges[position]["count"] or len(np.intersect1d(shared, common, assume_unique=True)) != removed[position]:
                    raise AssertionError("Host-common counterfactual differs from independent intersections")
            keep = host_edges["count"] - removed >= 50
            selections = choose_edges(host_edges, same_url, external, partner_docplayer, rng)
            for selection in selections:
                key = selection["doc_i"], selection["doc_j"]
                entry = global_selections.setdefault(key, {
                    "doc_i": key[0], "doc_j": key[1], "count": selection["count"], "selections": {},
                })
                entry["selections"][host] = selection["categories"]
                needed.update(key)
            top = np.lexsort((values, -df))[:15]
            for rank in top:
                top_requests.append((host, int(values[rank]), int(df[rank])))
            reports[host] = {
                "documents": len(mine), "document_ids": list(map(int, mine)),
                "unique_captured_urls": len({metadata[int(doc)]["url"] for doc in mine}),
                "within_host_edges": int(np.count_nonzero(within)),
                "same_url_within_host_edges": int(np.count_nonzero(within & same_url)),
                "external_edges": int(np.count_nonzero(external)),
                "docplayer_bridge_edges": int(np.count_nonzero(external & partner_docplayer)),
                "documents_directly_touching_docplayer": len(np.unique(target[external & partner_docplayer])),
                "external_partner_hosts": [{"host": name, "edges": count} for name, count in partner_counts.most_common()],
                "within_host_components": component_statistics(mine, host_edges[within]),
                "degree_min_median_max": list(map(float, np.percentile(degree[mine], [0, 50, 100]))),
                "half_host_df_counterfactual": {
                    "rule": f"subtract hashes occurring in >= {math.ceil(len(mine) / 2)} of these host members, then require count >=50",
                    "excluded_hashes": len(common),
                    "remaining_within_host_edges": int(np.count_nonzero(keep & within)),
                    "remaining_external_edges": int(np.count_nonzero(keep & external)),
                    "within_host_components": component_statistics(mine, host_edges[keep & within]),
                    "independent_numpy_samples": len(validation_positions),
                    "interpretation": "Frequency diagnostic, not proof these hashes are boilerplate or a production filter.",
                },
                "selected_pairs": [[row["doc_i"], row["doc_j"]] for row in selections],
                "top_fingerprints": [],
            }
            if host == "www.skogholt.org":
                roles = {int(doc): skogholt_page_role(metadata[int(doc)]["url"]) for doc in mine}
                role_pairs = Counter(
                    " / ".join(sorted((roles[int(row["doc_i"])], roles[int(row["doc_j"])])))
                    for row in host_edges[within]
                )
                reports[host]["url_role_diagnostic"] = {
                    "document_roles": dict(Counter(roles.values())),
                    "within_host_edge_roles": dict(role_pairs),
                    "caveat": "URL-pattern diagnostic, not a complete content classifier; archive/listing pages can contain full articles.",
                }
            needed.update(map(int, mine))
            print(f"[host study] {host}: {len(mine)} documents, {np.count_nonzero(within)} internal / {np.count_nonzero(external)} external edges", flush=True)
        detector = FingerprintChainDetector(str(index))
        top_values = np.array(sorted({value for _, value, _ in top_requests}), dtype=np.uint64)
        posting_lists = detector.index._get_postings(top_values)
        for host, value, df in top_requests:
            postings = posting_lists[np.uint64(value)]
            candidates = [row for row in postings if int(row["doc_id"]) in reports[host]["document_ids"]]
            first = candidates[0]
            doc, position = int(first["doc_id"]), int(first["position"])
            needed.add(doc)
            reports[host]["top_fingerprints"].append({
                "hash": value, "host_member_df": df, "full_corpus_df": len(np.unique(postings["doc_id"])),
                "example_doc": doc, "position": position,
            })
        raw = store.get_many(sorted(needed), id_kind="internal")
    prepared = {}
    selected_documents = {doc for key in global_selections for doc in key}
    selected_documents.update(row["example_doc"] for host in HOSTS for row in reports[host]["top_fingerprints"])
    for doc in selected_documents:
        text = raw[doc]["text"]
        values, positions = detector.index.winnower.get_winnowed_fingerprints(text)
        if not np.array_equal(np.unique(values), detector.index.get_document_fingerprints(doc)):
            raise AssertionError("Raw text does not reproduce the source forward index")
        prepared[doc] = {
            "tokens": detector.index.winnower.tokenize(text),
            "offsets": tokenize_with_offsets(text, punctuation=detector.index.winnower.punctuation),
            "positions": dict(zip(map(int, positions), map(int, values))),
        }
    for host in HOSTS:
        documents = reports[host]["document_ids"]
        text_lengths = [len(raw[doc]["text"]) for doc in documents]
        reports[host]["raw_characters_min_median_max"] = list(map(float, np.percentile(text_lengths, [0, 50, 100])))
        reports[host]["repeated_raw_lines"] = repeated_lines(raw[doc] for doc in documents)
        reports[host]["documents_with_at_least_3_detaljer_lines"] = sum(
            sum(line.strip() == "Detaljer" for line in raw[doc]["text"].splitlines()) >= 3 for doc in documents
        )
        for row in reports[host]["top_fingerprints"]:
            tokens = prepared[row["example_doc"]]["tokens"]
            row["kgram"] = " ".join(tokens[row["position"]:row["position"] + detector.index.winnower.length])
            if prepared[row["example_doc"]]["positions"].get(row["position"]) != row["hash"]:
                raise AssertionError("Host-common fingerprint position differs from source")
    results = []
    for selection in global_selections.values():
        a, b = selection["doc_i"], selection["doc_j"]
        try:
            evidence = detector.get_pair_passage(a, b)
        except MemoryError as error:
            print(f"[host study] RESOURCE LIMIT {a},{b}: {error}", flush=True)
            evidence = {"status": "resource_limit", "error": str(error), "best_passage": None}
        else:
            if evidence["shared_fingerprints"] != selection["count"]:
                raise AssertionError("Selected pair count differs from numeric forward intersection")
        evidence.update(selection)
        evidence["documents"] = []
        passage = evidence["best_passage"]
        for doc, suffix in ((a, "i"), (b, "j")):
            item = {
                "doc_id": doc, **metadata[doc], "characters": len(raw[doc]["text"]),
                "tokens": len(prepared[doc]["tokens"]),
                "raw_text_prefix": raw[doc]["text"][:350],
                "standalone_detaljer_lines": sum(line.strip() == "Detaljer" for line in raw[doc]["text"].splitlines()),
            }
            if passage:
                start, stop = passage["token_span_" + suffix]
                offsets_for_doc = prepared[doc]["offsets"]
                char_start, char_stop = offsets_for_doc[start][0], offsets_for_doc[stop - 1][1]
                item["passage"] = {
                    "token_span": [start, stop], "character_span": [char_start, char_stop],
                    "start_fraction": start / len(prepared[doc]["tokens"]),
                    "text": raw[doc]["text"][char_start:char_stop],
                    "before": raw[doc]["text"][max(0, char_start - 180):char_start],
                    "after": raw[doc]["text"][char_stop:char_stop + 180],
                }
            xml = raw[doc].get("xml")
            if isinstance(xml, str):
                tree = ET.fromstring(xml)
                item["xml_tags"] = dict(Counter(node.tag for node in tree.iter()))
            evidence["documents"].append(item)
        if passage:
            length = detector.index.winnower.length
            for point in passage["aligned_hashes"]:
                p, q, value = point["position_doc1"], point["position_doc2"], point["hash"]
                if (
                    prepared[a]["positions"].get(p) != value or prepared[b]["positions"].get(q) != value
                    or prepared[a]["tokens"][p:p + length] != prepared[b]["tokens"][q:q + length]
                ):
                    raise AssertionError("Aligned point differs from exact normalized source k-grams")
        results.append(evidence)
    for host in HOSTS:
        selected = [row for row in results if host in row["selections"]]
        reports[host]["sample_statuses"] = dict(Counter(row["status"] for row in selected))
    if _source_identity(index) != identity or edge_file.stat().st_mtime_ns != edge_identity["edge_mtime_ns"]:
        raise ValueError("Source index or overlap edges changed during host analysis")
    report = {
        "status": "complete", "source": identity, "source_unchanged": True,
        "k": 50, "component_id": group_id, "component_documents": len(members),
        "hosts": reports, "pairs": results,
        "all_component_document_hosts": {str(doc): row["host"] for doc, row in metadata.items()},
        "selection": "Seeded bounded random within-host same/different-URL, near-K and external pairs, plus purposive strongest edges. Not precision or prevalence.",
        "seed": SEED, "detector_defaults_unchanged": True,
        "stored_kgram_length": detector.index.winnower.length,
        "stored_winnowing_window": detector.index.winnower.window_size,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(output / "report.json", report)
    print("[host study] complete:", output / "report.json", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--store-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    analyze_hosts(args.index_dir, args.review_dir, args.store_file, args.output_dir)
