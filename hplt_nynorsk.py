"""Build and inspect the HPLT Nynorsk fingerprint index."""

import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import zstandard as zstd

from findmytext import index_builder, indexing, similarity, utils, winnower

# Paths for the HPLT Nynorsk dataset and the index build directory.
DATA_PATH = Path("/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk")
BUILD_PATH = DATA_PATH / "findmytext_build"
INDEX_PATH = BUILD_PATH / "index"


def build_index():
    if INDEX_PATH.is_dir():
        if not all(
            (INDEX_PATH / name).is_file()
            for name in ("forward_offsets.npy", "forward_fingerprints.npy")
        ):
            index_builder.build_forward_index(str(INDEX_PATH))
        print(f"Using existing index: {INDEX_PATH}")
        return

    corpus_files = sorted((DATA_PATH / "nno_Latn").glob("*.jsonl.zst"))
    if not corpus_files:
        raise FileNotFoundError(f"No corpus shards found in {DATA_PATH / 'nno_Latn'}")

    intermediate_files = []
    for corpus_file in corpus_files:
        shard_name = corpus_file.name.removesuffix(".jsonl.zst")
        intermediate_files.extend(
            index_builder.index_file(
                str(corpus_file),
                str(BUILD_PATH / "shards" / shard_name),
                delete_existing=True,
                nb_workers=8,
                length=4,
                window_size=6,
                min_length=100,
                max_length=100_000,
            )
        )

    index_builder.merge_indexes(intermediate_files, str(INDEX_PATH))
    print(f"Built index: {INDEX_PATH}")


def inspect_metadata(disk_index):
    print(f"Documents: {len(disk_index.to_external_doc_id):,}")
    print(f"Unique fingerprints: {len(disk_index.fingerprints):,}")
    print(f"Total inverted postings: {int(disk_index.lengths.sum()):,}")
    print(f"Forward index available: {disk_index.forward_offsets is not None}")
    print(f"Winnower settings: {disk_index.winnower}")


def verify_forward_index(disk_index):
    if disk_index.forward_offsets is None:
        raise AssertionError("Forward index is missing")

    forward_counts = []
    total_documents = len(disk_index.to_external_doc_id)
    next_percent = 10
    document_id = 0
    for document_id in range(total_documents):
        fingerprints = disk_index.get_document_fingerprints(document_id)
        # if len(fingerprints) != len(np.unique(fingerprints)):
        #     raise AssertionError(
        #         f"Duplicate forward fingerprints in document {document_id}"
        #     )
        if len(fingerprints) and not np.all(fingerprints[:-1] <= fingerprints[1:]):
            raise AssertionError(
                f"Forward fingerprints are not sorted in document {document_id}"
            )
        forward_counts.append(len(fingerprints))
        while (
            next_percent <= 100
            and (document_id + 1) * 100 >= next_percent * total_documents
        ):
            print(
                f"Forward index: {next_percent}% ({document_id + 1:,}/{total_documents:,} documents)",
                end="\r",
                flush=True,
            )
            next_percent += 10
    print()

    print(
        f"Fingerprints per document: min={min(forward_counts):,}, median={np.median(forward_counts):,.0f}, max={max(forward_counts):,}"
    )
    fig = go.Figure(
        go.Histogram(
            x=forward_counts,
            xbins={
                "start": min(forward_counts) - 0.5,
                "end": max(forward_counts) + 0.5,
                "size": 1,
            },
        )
    )
    fig.update_layout(
        xaxis_title="Number of unique fingerprints per document",
        yaxis_title="Number of documents",
        title="Forward-index fingerprint counts per document",
    )
    fig.show()


def plot_fingerprint_document_distribution(documents_per_fingerprint, bin_width=None):
    documents_per_fingerprint = np.asarray(documents_per_fingerprint)
    quantile_levels = (0.5, 0.9, 0.99, 0.999)
    quantile_values = np.quantile(documents_per_fingerprint, quantile_levels)

    if bin_width is None:
        document_frequencies, fingerprint_counts = np.unique(
            documents_per_fingerprint, return_counts=True
        )
        fig = go.Figure(
            go.Bar(
                x=document_frequencies,
                y=fingerprint_counts,
                width=1,
                hovertemplate=(
                    "Documents per fingerprint: %{x}<br>"
                    "Number of fingerprints: %{y}<extra></extra>"
                ),
            )
        )
        title = "Fingerprint counts by document frequency"
    else:
        if not isinstance(bin_width, (int, np.integer)) or bin_width <= 0:
            raise ValueError("bin_width must be a positive integer")
        bin_width = int(bin_width)
        bin_end = (
            0.5
            + np.ceil((documents_per_fingerprint.max() - 0.5) / bin_width) * bin_width
        )
        fig = go.Figure(
            go.Histogram(
                x=documents_per_fingerprint,
                xbins={"start": 0.5, "end": bin_end, "size": bin_width},
                hovertemplate=(
                    "Documents per fingerprint: %{x}<br>"
                    "Number of fingerprints: %{y}<extra></extra>"
                ),
            )
        )
        title = f"Fingerprint counts by document frequency (bin width {bin_width})"

    labels_by_value = {}
    for level, value in zip(quantile_levels, quantile_values):
        labels_by_value.setdefault(float(value), []).append(
            f"P{level * 100:g}: {value:,.0f}"
        )
    for value, labels in labels_by_value.items():
        fig.add_vline(
            x=value,
            line_dash="dash",
            annotation_text=" / ".join(labels),
            annotation_position="top",
        )

    fig.update_layout(
        xaxis={
            # "type": "log",
            "title": "Number of documents containing a fingerprint",
        },
        yaxis={"type": "log", "title": "Number of fingerprints at that frequency"},
        title=title,
    )
    fig.show()


def get_documents_per_fingerprint(disk_index):
    """Count distinct documents per fingerprint in one sequential index pass."""
    documents_per_fingerprint = np.empty(len(disk_index.fingerprints), dtype=np.uint32)
    fingerprint_offset = 0
    total_fingerprints = len(disk_index.fingerprints)
    next_percent = 1
    for fingerprints, groups, _ in disk_index.iter_unique_posting_batches():
        batch_counts = np.bincount(groups, minlength=len(fingerprints))
        next_offset = fingerprint_offset + len(fingerprints)
        documents_per_fingerprint[fingerprint_offset:next_offset] = batch_counts
        fingerprint_offset = next_offset
        while (
            next_percent <= 100
            and fingerprint_offset * 100 >= next_percent * total_fingerprints
        ):
            print(
                f"Fingerprint counts: {next_percent}% ({fingerprint_offset:,}/{total_fingerprints:,} fingerprints)",
                end="\r",
                flush=True,
            )
            next_percent += 1
    print()
    return documents_per_fingerprint


def _find_corpus_documents(external_doc_ids, corpus_files):
    wanted_ids = set(external_doc_ids)
    documents = {}
    for corpus_file in corpus_files:
        for document in utils.stream_json_zst(
            str(corpus_file), min_length=100, max_length=1_000_000
        ):
            external_doc_id = document["id"]
            if external_doc_id in wanted_ids:
                documents[external_doc_id] = document["text"]
                wanted_ids.remove(external_doc_id)
                if not wanted_ids:
                    return documents
    return documents


def _fingerprint_document_id(disk_index, fingerprint, match=0):
    postings = disk_index._get_postings(np.array([fingerprint], dtype=np.uint64)).get(
        np.uint64(fingerprint)
    )
    if postings is None or not len(postings):
        return None
    return int(postings[match]["doc_id"])


def get_fingerprint_string(disk_index, fingerprint, corpus_files=None):
    """Return a normalized token span that produces ``fingerprint``, if found."""
    document_id = _fingerprint_document_id(disk_index, fingerprint, match=0)
    if document_id is None:
        return None

    external_doc_id = disk_index.to_external_doc_id[document_id]
    if corpus_files is None:
        corpus_files = sorted((DATA_PATH / "nno_Latn").glob("*.jsonl.zst"))
    documents = _find_corpus_documents([external_doc_id], corpus_files)
    text = documents.get(external_doc_id)
    if text is None:
        return None

    extractor = winnower.Winnower(
        length=disk_index.winnower.length,
        window_size=1,
        base=disk_index.winnower.base,
        punctuation=disk_index.winnower.punctuation,
    )
    fingerprints, positions = extractor.get_winnowed_fingerprints(text)
    matches = np.flatnonzero(fingerprints == fingerprint)
    if not len(matches):
        return None

    tokens = extractor.tokenize(text)
    start = int(positions[matches[0]])
    return " ".join(tokens[start : start + extractor.length])


def get_top_fingerprints_with_text(
    disk_index, documents_per_fingerprint, top_k=5, corpus_files=None, match=0
):
    """Show and return the most widespread fingerprints with one matching token span."""
    if top_k <= 0:
        return []
    if corpus_files is None:
        corpus_files = sorted((DATA_PATH / "nno_Latn").glob("*.jsonl.zst"))

    order = np.lexsort(
        (disk_index.fingerprints, -np.asarray(documents_per_fingerprint))
    )[:top_k]
    selected = []
    rank, index = list(enumerate(order, start=1))[0]
    for rank, index in enumerate(order, start=1):
        fingerprint = int(disk_index.fingerprints[index])
        document_id = _fingerprint_document_id(disk_index, fingerprint, match=match)
        external_doc_id = (
            disk_index.to_external_doc_id[document_id]
            if document_id is not None
            else None
        )
        selected.append(
            {
                "rank": rank,
                "fingerprint": fingerprint,
                "documents": int(documents_per_fingerprint[index]),
                "document_id": external_doc_id,
            }
        )

    documents = _find_corpus_documents(
        [row["document_id"] for row in selected if row["document_id"] is not None],
        corpus_files,
    )
    extractor = winnower.Winnower(
        length=disk_index.winnower.length,
        window_size=1,
        base=disk_index.winnower.base,
        punctuation=disk_index.winnower.punctuation,
    )

    # Iterate over the selected fingerprints and find the matching token spans in the corresponding documents.
    row = selected[1] if selected else None
    for row in selected:
        text = documents.get(row["document_id"])
        row["text"] = None
        if text is None:
            continue
        fingerprints, positions = extractor.get_winnowed_fingerprints(text)
        matches = np.flatnonzero(fingerprints == row["fingerprint"])
        if len(matches):
            tokens = extractor.tokenize(text)
            start = int(positions[matches[0]])
            row["text"] = " ".join(tokens[start : start + extractor.length])

    fig = go.Figure(
        go.Table(
            domain={"x": [0, 1], "y": [0, 1]},
            columnwidth=[70, 150, 150, 400, 300],
            header={
                "values": [
                    "Rank",
                    "Fingerprint",
                    "Documents",
                    "Source document",
                    "Matching token span",
                ],
                "fill_color": "#263238",
                "font": {"color": "white", "size": 12},
                "align": "left",
                "height": 32,
            },
            cells={
                "values": [
                    [row["rank"] for row in selected],
                    [row["fingerprint"] for row in selected],
                    [row["documents"] for row in selected],
                    [row["document_id"] or "Not found" for row in selected],
                    [row["text"] or "Not found" for row in selected],
                ],
                "fill_color": "#f3f6f7",
                "align": "left",
                "height": 30,
            },
        )
    )
    fig.update_layout(
        title={
            "text": f"Top {len(selected)} fingerprints by document count",
            "x": 0,
            "xanchor": "left",
        },
        autosize=True,
        height=50 + 32 + 30 * len(selected),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    fig.show()
    return selected


def verify_inverted_index(disk_index):
    forward_sets = {
        document_id: set(map(int, disk_index.get_document_fingerprints(document_id)))
        for document_id in range(len(disk_index.to_external_doc_id))
    }
    checked_postings = 0
    checked_fingerprints = 0
    total_fingerprints = len(disk_index.fingerprints)
    next_percent = 1
    for fingerprint in disk_index.fingerprints:
        postings = disk_index._get_postings(np.array([fingerprint], dtype=np.uint64))[
            int(fingerprint)
        ]
        for posting in postings:
            document_id = int(posting["doc_id"])
            position = int(posting["position"])
            if document_id not in forward_sets:
                raise AssertionError(f"Invalid document ID {document_id}")
            if int(fingerprint) not in forward_sets[document_id]:
                raise AssertionError(
                    f"Fingerprint missing from forward index: {fingerprint}"
                )
            if position < 0:
                raise AssertionError(f"Invalid position for fingerprint {fingerprint}")
            checked_postings += 1
        checked_fingerprints += 1
        while (
            next_percent <= 100
            and checked_fingerprints * 100 >= next_percent * total_fingerprints
        ):
            print(
                f"Inverted index: {next_percent}% ({checked_fingerprints:,}/{total_fingerprints:,} fingerprints)",
                flush=True,
            )
            next_percent += 1

    print(f"Checked {checked_fingerprints:,} inverted fingerprints")
    print(f"Checked {checked_postings:,} inverted posting positions")
    print("Forward/inverted membership verification passed.")


def inspect_sample(disk_index):
    sample_document_id = 0
    sample_fingerprints = disk_index.get_document_fingerprints(sample_document_id)
    print("Sample document:", disk_index.to_external_doc_id[sample_document_id])
    print("Forward fingerprints:", sample_fingerprints[:10].tolist(), "...")

    sample_fingerprint = int(sample_fingerprints[0])
    sample_postings = disk_index._get_postings(
        np.array([sample_fingerprint], dtype=np.uint64)
    )[sample_fingerprint]
    print(f"Inverted postings for fingerprint {sample_fingerprint}:")
    for posting in sample_postings[:10]:
        document_id = int(posting["doc_id"])
        print(
            " ",
            document_id,
            disk_index.to_external_doc_id[document_id],
            int(posting["position"]),
        )


def build_fingerprint_statistics(disk_index, documents_per_fingerprint):
    """Build a sorted table of document counts and similarity-update counts."""
    return (
        pl.DataFrame(
            {
                "fingerprint": disk_index.fingerprints,
                "document_count": documents_per_fingerprint,
            }
        )
        .with_columns(
            (
                pl.col("document_count").cast(pl.UInt64)
                * (pl.col("document_count").cast(pl.UInt64) - 1)
                // 2
            ).alias("N_similarity_updates"),
        )
        .sort("document_count", descending=True)
    )


def summarize_fingerprint_quantiles(documents_per_fingerprint_df):
    """Summarize fingerprint document-count quantiles and update counts."""
    quantile_levels = [0.5, 0.8, 0.9, 0.99, 0.999, 0.9995, 0.9999, 0.99999, 0.999999]
    quantile_expressions = []
    for index, level in enumerate(quantile_levels):
        threshold = pl.col("document_count").quantile(level, interpolation="nearest")
        quantile_expressions.extend(
            [
                threshold.cast(pl.UInt64).alias(f"document_count_{index}"),
                (pl.col("document_count") <= threshold)
                .sum()
                .cast(pl.UInt64)
                .alias(f"keep_{index}"),
                (pl.col("document_count") > threshold)
                .sum()
                .cast(pl.UInt64)
                .alias(f"remove_{index}"),
            ]
        )
    quantile_statistics = documents_per_fingerprint_df.select(quantile_expressions).row(
        0
    )
    DF_result = pl.DataFrame(
        {
            "quantile": quantile_levels,
            "document_count_quantile": [
                quantile_statistics[index * 3] for index in range(len(quantile_levels))
            ],
            "N_fingerprints_keep": [
                quantile_statistics[index * 3 + 1]
                for index in range(len(quantile_levels))
            ],
            "N_fingerprints_remove": [
                quantile_statistics[index * 3 + 2]
                for index in range(len(quantile_levels))
            ],
        }
    ).with_columns(
        (
            pl.col("document_count_quantile")
            * (pl.col("document_count_quantile") - 1)
            // 2
        ).alias("N_similarity_updates")
    )
    return DF_result


def inspect_corpus_document(external_doc_id, output_directory=Path("json")):
    """Print and save the complete corpus record for an external document ID."""
    record = None
    source_path = None

    for corpus_path in sorted((DATA_PATH / "nno_Latn").glob("*.jsonl.zst")):
        with (
            corpus_path.open("rb") as compressed_file,
            zstd.ZstdDecompressor().stream_reader(compressed_file) as reader,
        ):
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if line.strip():
                    candidate = json.loads(line)
                    if candidate.get("id") == external_doc_id:
                        record = candidate
                        source_path = corpus_path
                        break
        if record is not None:
            break

    if record is None:
        raise LookupError(f"Document not found: {external_doc_id}")

    print(f"Source shard: {source_path}")
    print("Metadata:")
    print(
        json.dumps(
            {key: value for key, value in record.items() if key != "text"},
            indent=2,
            ensure_ascii=False,
        )
    )

    text_cleaner = winnower.Winnower(
        length=4, window_size=6, base=256, punctuation=False
    )
    cleaned_text = " ".join(text_cleaner.tokenize(record["text"]))
    print("Text after index cleaning:")
    print(cleaned_text)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{external_doc_id}.json"
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Complete record saved to: {output_path.resolve()}")
    return output_path


def main():
    # Build the index if it doesn't exist
    # build_index()

    # Inspect the index metadata before verifying the forward and inverted indices
    disk_index = indexing.DiskBasedIndex(str(INDEX_PATH))

    # Inspect the index metadata
    inspect_metadata(disk_index)

    # Verify the forward index before the inverted index
    verify_forward_index(disk_index)

    # Verify the inverted index after the forward index
    verify_inverted_index(disk_index)

    # Count distinct documents for each fingerprint without running verification.
    documents_per_fingerprint = get_documents_per_fingerprint(disk_index)
    documents_per_fingerprint_df = build_fingerprint_statistics(
        disk_index, documents_per_fingerprint
    )
    print("Fingerprint statistics (first 10 rows):")
    print(documents_per_fingerprint_df.head(10))

    quantile_summary_df = summarize_fingerprint_quantiles(documents_per_fingerprint_df)
    print("Document count quantiles and similarity-update counts:")
    print(quantile_summary_df)

    # Plot the fingerprint document distribution with the default bin width
    plot_fingerprint_document_distribution(documents_per_fingerprint)
    plot_fingerprint_document_distribution(documents_per_fingerprint, bin_width=1000)

    # Inspect the most common fingerprints and their matching token spans
    selected = get_top_fingerprints_with_text(
        disk_index=disk_index,
        documents_per_fingerprint=documents_per_fingerprint,
        top_k=20,
        corpus_files=sorted((DATA_PATH / "nno_Latn").glob("*.jsonl.zst")),
        match=1,
    )

    # Inspect a sample document and its fingerprints
    inspect_sample(disk_index)

    # Inspect specific external documents from the corpus
    EXTERNAL_DOC_IDS = (
        "ea6590d6f45ccc0ca9baa2c952cfc4f9",
        "0ab593c6fca13daeabfe73a9a9559c43",
        "a67e4a093a3c9e7c11687764c453f68b",
    )
    for external_doc_id in EXTERNAL_DOC_IDS:
        inspect_corpus_document(external_doc_id)

    disk_index.forward_fingerprints
    disk_index.to_external_doc_id.__len__()

    # Get all the external document IDs.
    all_external_doc_ids = list(disk_index.to_external_doc_id)

    # Loop over all of them and get the corresponding forward fingerprints.
    external_doc_id = all_external_doc_ids[0] if all_external_doc_ids else None
    for external_doc_id in all_external_doc_ids:
        forward_fingerprints = disk_index.forward_fingerprints[
            disk_index.to_external_doc_id.inverse[external_doc_id]
        ]

    # I want to make an alternative way for the

    # Compute and inspect document similarity scores.
    similarity.compute_similarities_inverted(
        index_dir=str(INDEX_PATH),
        min_shared_fingerprints=1,
        max_fingerprint_document_frequency=None,
        max_pair_updates_in_memory=20_000_000,
        max_chunks_per_merge=None,
        posting_batch_entries=10_000_000,
        verbose=True,
    )
    similarity.compute_similarities_forward_topk(
        str(INDEX_PATH), top_k=50, min_shared_fingerprints=5
    )

    similarities = disk_index.load_similarities("inverted")
    forward_topk = disk_index.load_similarities("forward_topk")
    report = similarity.compare_with_exhaustive(
        similarities, forward_topk, top_k=50, min_shared_fingerprints=5
    )
    print(
        f"Document pairs with shared fingerprints: {len(similarities):,}; "
        f"forward_topk pairs: {report['n_top_k_pairs']:,}; "
        f"top-k matches exhaustive: {report['identical']}"
    )

    order = np.argsort(-similarities["count"])
    print(f"{'Document A':>40}  {'Document B':>40}  {'Shared fingerprints':>20}")
    for index in order[:10]:
        row = similarities[index]
        document_a = disk_index.to_external_doc_id[int(row["doc_i"])]
        document_b = disk_index.to_external_doc_id[int(row["doc_j"])]
        print(f"{document_a:>40}  {document_b:>40}  {int(row['count']):>20,}")


if __name__ == "__main__":
    main()
