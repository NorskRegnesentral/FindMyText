"""Export complete original pair text with source-verified fingerprint highlights."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
from pathlib import Path

import numpy as np

from findmytext.detectors import FingerprintChainDetector
from findmytext.document_store import CorpusDocumentStore
from findmytext.overlap_inverted import _source_identity
from findmytext.web import tokenize_with_offsets


def highlight_text(text, offsets, shared_positions, cluster_positions, length, *, anchor_id=None):
    """Keep original characters; mark supported k-grams, not whole cluster envelopes."""
    shared_positions, cluster_positions = set(shared_positions), set(cluster_positions)
    if length < 1 or not cluster_positions <= shared_positions:
        raise ValueError("Use positive k-gram length and cluster positions contained in shared positions")
    levels = np.zeros(len(text), dtype=np.uint8)
    for positions, level in ((shared_positions, 1), (cluster_positions, 2)):
        for position in positions:
            if position < 0 or position + length > len(offsets):
                raise ValueError("Fingerprint token position lies outside the original text")
            start, stop = offsets[position][0], offsets[position + length - 1][1]
            if not 0 <= start < stop <= len(text):
                raise ValueError("Token character offsets lie outside the original text")
            levels[start:stop] = np.maximum(levels[start:stop], level)
    boundaries = np.r_[0, np.flatnonzero(levels[1:] != levels[:-1]) + 1, len(text)]
    rendered, anchored = [], False
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start == stop:
            continue
        piece = escape(text[start:stop])
        level = levels[start]
        if level:
            anchor = ""
            if level == 2 and anchor_id and not anchored:
                anchor = f' id="{escape(anchor_id, quote=True)}"'
                anchored = True
            if level == 2:
                piece = "<strong>" + piece + "</strong>"
            piece = f"<mark{anchor}>" + piece + "</mark>"
        rendered.append(piece)
    return "".join(rendered)


def export_pair_review(index_dir, store_file, doc_i, doc_j, output_file):
    source = Path(index_dir).resolve()
    output = Path(output_file).resolve()
    if output == source or source in output.parents:
        raise ValueError("Text review must be outside the source index")
    if output.exists():
        raise FileExistsError(output)
    identity = _source_identity(source)
    detector = FingerprintChainDetector(str(source))
    evidence = detector.get_pair_passage(doc_i, doc_j)
    shared = np.intersect1d(
        detector.index.get_document_fingerprints(doc_i),
        detector.index.get_document_fingerprints(doc_j), assume_unique=True,
    )
    postings = detector.index._get_postings(shared, only_doc_ids=True)
    global_df = {int(value): len(np.unique(docs)) for value, docs in postings.items()}
    if len(global_df) != len(shared):
        raise ValueError("Shared forward fingerprints are missing from the positional index")
    filtered_overlap = sum(frequency <= 1000 for frequency in global_df.values())
    with CorpusDocumentStore(store_file, index_dir=source) as store:
        documents = store.get_many([doc_i, doc_j], id_kind="internal")
    prepared = {}
    normalized_kgrams = {}
    for doc in (doc_i, doc_j):
        record = documents[doc]
        text = record["text"]
        values, positions = detector.index.winnower.get_winnowed_fingerprints(text)
        if not np.array_equal(np.unique(values), detector.index.get_document_fingerprints(doc)):
            raise ValueError(f"Original source text does not reproduce document {doc}'s forward index")
        tokens = detector.index.winnower.tokenize(text)
        offsets = tokenize_with_offsets(text, punctuation=detector.index.winnower.punctuation)
        if len(offsets) != len(tokens):
            raise ValueError("Character offsets do not match indexed tokenization")
        selected = np.isin(values, shared)
        length = detector.index.winnower.length
        for value, position in zip(values[selected], positions[selected]):
            kgram = tuple(tokens[int(position):int(position) + length])
            previous = normalized_kgrams.setdefault(int(value), kgram)
            if previous != kgram:
                raise ValueError("A shared fingerprint represents different normalized source k-grams")
        prepared[doc] = {
            "text": text, "offsets": offsets,
            "shared_positions": list(map(int, positions[selected])),
            "selected": dict(zip(map(int, positions), map(int, values))),
        }
    passage = evidence["best_passage"]
    cluster_positions = {doc_i: set(), doc_j: set()}
    if passage:
        for point in passage["aligned_hashes"]:
            for doc, key in ((doc_i, "position_doc1"), (doc_j, "position_doc2")):
                position = point[key]
                if prepared[doc]["selected"].get(position) != point["hash"]:
                    raise ValueError("Best-cluster position does not reproduce its stored fingerprint")
                cluster_positions[doc].add(position)
    support = passage["distinct_fingerprint_support"] if passage else 0
    lines = [
        f"# Complete corpus text: documents {doc_i} and {doc_j}", "",
        "These are the **complete original `text` fields** preserved in the corpus, not normalized reconstructions or live web pages.",
        "Whitespace and punctuation inside each text block are retained. HTML escaping and highlighting are presentation only.", "",
        "## Reading the highlights", "",
        "- **Yellow:** original character ranges supporting any of the pair's shared stored fingerprints.",
        "- **Bold yellow:** supporting ranges belonging to the detector's best surviving positional cluster.",
        "- Unhighlighted text can still match; this display marks stored winnowed 4-grams, not every possible shared word or 4-gram.",
        "- Matching ignores punctuation under the index tokenization; punctuation and whitespace inside a highlight need not be identical.",
        "- Blank gaps inside a cluster are not automatically highlighted. K counts distinct shared fingerprints over the complete pair, not just the best cluster.",
        "- Open this file in Markdown Preview to see the highlights. The full text is embedded below; nothing is fetched from the old website.", "",
        "## Pair summary", "",
        "| Measurement | Value |", "|---|---:|",
        f"| Whole-pair distinct shared fingerprints | {len(shared)} |",
        f"| Best-cluster distinct support | {support} |",
        f"| Stored k-gram length / winnowing window | {detector.index.winnower.length} / {detector.index.winnower.window_size} |",
        f"| Diagnostic overlap counting only hashes with full-corpus df <= 1,000 | {filtered_overlap} |", "",
        "The last row is a **counterfactual count filter**, not a candidate-discovery cap and not a change to the source indexes.",
        "Crawl timestamps below are not established document publication dates.", "",
        f"[Document A](#document-{doc_i}) | [Document B](#document-{doc_j})", "",
    ]
    if passage:
        lines.extend([
            f"[Best cluster in A](#document-{doc_i}-best) | [Best cluster in B](#document-{doc_j}-best)", "",
        ])
    for label, doc, suffix in (("A", doc_i, "i"), ("B", doc_j, "j")):
        record, data = documents[doc], prepared[doc]
        lines.extend([
            f'<a id="document-{doc}"></a>', f"## Document {label}: internal ID {doc}", "",
            f"- Original corpus ID: `{record['id']}`",
            f"- Historical captured URL: `{record.get('u', '(not present)')}`",
            f"- Crawl timestamp: `{record.get('ts', '(not present)')}`",
            f"- Crawl collection: `{record.get('crawl_id', '(not present)')}`",
            f"- Original text characters: **{len(data['text']):,}**",
            f"- Original text SHA-256 (UTF-8): `{sha256(data['text'].encode()).hexdigest()}`",
        ])
        if passage:
            lines.append(f"- Best-cluster token envelope (zero-based, end-exclusive): `{passage['token_span_' + suffix]}`")
        lines.extend([
            "", '<pre style="white-space: pre-wrap; overflow-wrap: anywhere;">' + highlight_text(
                data["text"], data["offsets"], data["shared_positions"],
                cluster_positions[doc], detector.index.winnower.length,
                anchor_id=f"document-{doc}-best",
            ) + "</pre>", "",
        ])
    if _source_identity(source) != identity:
        raise ValueError("Source index changed during text export")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return {
        "output": str(output), "shared_fingerprints": len(shared),
        "best_cluster_support": support, "full_corpus_df1000_filtered_overlap": filtered_overlap,
        "text_characters": {doc: len(prepared[doc]["text"]) for doc in prepared},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--store-file", type=Path, required=True)
    parser.add_argument("--doc-i", type=int, required=True)
    parser.add_argument("--doc-j", type=int, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    print(export_pair_review(args.index_dir, args.store_file, args.doc_i, args.doc_j, args.output_file))
