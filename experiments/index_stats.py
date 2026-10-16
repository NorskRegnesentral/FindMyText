"""Summary statistics and histograms for one or more merged fingerprint indexes.

Distributions are kept as value -> frequency tables (np.bincount), so quantiles and
histograms stay cheap even for billions of fingerprints.  One sequential pass over
``postings.dat`` is needed for the number of distinct documents per fingerprint.

Example:
    python experiments/index_stats.py nynorsk=/path/to/index bokmal=/path/to/index \
        --output-dir /path/to/analysis
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from findmytext.indexing import POSTING_DTYPE, DiskBasedIndex

QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)


def _add_counts(table, values):
    counts = np.bincount(values)
    if len(counts) > len(table):
        counts[: len(table)] += table
        return counts
    table[: len(counts)] += counts
    return table


def _summarize(table, prefix):
    """Mean, quantiles and max of a distribution given as value -> frequency."""
    table = np.asarray(table, dtype=np.int64)
    values = np.flatnonzero(table)
    frequencies = table[values]
    n = int(frequencies.sum())
    cumulative = np.cumsum(frequencies)
    summary = {
        f"{prefix}_mean": float((values * frequencies).sum() / n) if n else 0.0,
        f"{prefix}_max": int(values[-1]) if n else 0,
    }
    for level in QUANTILES:
        position = int(np.searchsorted(cumulative, level * n, side="left"))
        summary[f"{prefix}_p{level * 100:g}"] = int(values[min(position, len(values) - 1)]) if n else 0
    return summary


def _histogram(table, title, x_title, summary, prefix):
    values = np.flatnonzero(table)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=table[values],
            width=1,
            hovertemplate=f"{x_title}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )
    labels_by_value = {}
    for level in QUANTILES:
        value = summary[f"{prefix}_p{level * 100:g}"]
        labels_by_value.setdefault(value, []).append(f"P{level * 100:g}: {value:,}")
    labels_by_value.setdefault(summary[f"{prefix}_max"], []).append(
        f"max: {summary[f'{prefix}_max']:,}"
    )
    for value, labels in labels_by_value.items():
        fig.add_vline(
            x=value,
            line_dash="dash",
            annotation_text=" / ".join(labels),
            annotation_position="top",
        )
    fig.update_layout(
        title=f"{title} (mean {summary[f'{prefix}_mean']:,.2f})",
        xaxis={"type": "log", "title": f"{x_title} (log scale)"},
        yaxis={"type": "log", "title": "Count (log scale)"},
    )
    return fig


def index_statistics(name, index_dir, batch_entries):
    index = DiskBasedIndex(index_dir)
    n_docs = len(index.to_external_doc_id)
    n_fingerprints = len(index.fingerprints)
    lengths = np.asarray(index.lengths, dtype=np.int64)
    total_postings = int(lengths.sum())

    # Fingerprints per document (unique, from the forward index).
    fingerprints_per_doc = np.diff(np.asarray(index.forward_offsets, dtype=np.int64))
    fingerprints_per_doc_table = np.bincount(fingerprints_per_doc)

    # Posting entries per fingerprint (includes repeats within a document) = one pread.
    postings_per_fp_table = np.bincount(lengths)

    # Distinct documents per fingerprint, and the forward_topk read volume:
    # every document holding fingerprint f reads f's whole posting list once.
    docs_per_fp_table = np.zeros(1, dtype=np.int64)
    pair_updates = 0
    forward_topk_entries_read = 0
    fp_offset = 0
    for batch_fps, group, _ in index.iter_unique_posting_batches(batch_entries):
        docs_per_fp = np.bincount(group, minlength=len(batch_fps)).astype(np.int64)
        batch_lengths = lengths[fp_offset : fp_offset + len(batch_fps)]
        docs_per_fp_table = _add_counts(docs_per_fp_table, docs_per_fp)
        pair_updates += int((docs_per_fp * (docs_per_fp - 1) // 2).sum())
        forward_topk_entries_read += int((docs_per_fp * batch_lengths).sum())
        fp_offset += len(batch_fps)
        print(
            f"{name}: {fp_offset * 100 / n_fingerprints:5.1f}% of fingerprints",
            end="\r",
            flush=True,
        )
    print()

    forward_entries = int(np.asarray(index.forward_offsets[-1]))
    stats = {
        "name": name,
        "index_dir": index_dir,
        "n_documents": n_docs,
        "n_documents_without_fingerprints": int(fingerprints_per_doc_table[0]),
        "n_unique_fingerprints": n_fingerprints,
        "n_postings": total_postings,
        "n_unique_doc_fingerprint_pairs": forward_entries,
        "repeated_postings_fraction": 1 - forward_entries / total_postings,
        "singleton_fingerprints": int(docs_per_fp_table[1]),
        "singleton_fingerprints_fraction": int(docs_per_fp_table[1]) / n_fingerprints,
        "inverted_pair_updates": pair_updates,
        "forward_topk_gb_read": forward_topk_entries_read * POSTING_DTYPE.itemsize / 1e9,
        "forward_topk_mb_read_per_document": forward_topk_entries_read
        * POSTING_DTYPE.itemsize
        / 1e6
        / n_docs,
        "postings_dat_gb": os.path.getsize(os.path.join(index_dir, "postings.dat")) / 1e9,
        "index_gb": sum(p.stat().st_size for p in Path(index_dir).iterdir() if p.is_file())
        / 1e9,
    }
    stats.update(_summarize(fingerprints_per_doc_table, "fingerprints_per_doc"))
    stats.update(_summarize(docs_per_fp_table, "docs_per_fingerprint"))
    stats.update(_summarize(postings_per_fp_table, "postings_per_fingerprint"))

    figures = [
        _histogram(
            fingerprints_per_doc_table,
            f"{name}: unique fingerprints per document",
            "Fingerprints per document",
            stats,
            "fingerprints_per_doc",
        ),
        _histogram(
            docs_per_fp_table,
            f"{name}: distinct documents per fingerprint",
            "Documents per fingerprint",
            stats,
            "docs_per_fingerprint",
        ),
        _histogram(
            postings_per_fp_table,
            f"{name}: posting-list length per fingerprint (entries read per lookup)",
            "Posting entries per fingerprint",
            stats,
            "postings_per_fingerprint",
        ),
    ]
    return stats, figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indexes", nargs="+", help="name=index_dir pairs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-entries", type=int, default=50_000_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "index_stats.json"
    # Keep results from earlier runs so indexes can be added one at a time.
    all_stats = json.loads(json_path.read_text()) if json_path.exists() else []
    for item in args.indexes:
        name, index_dir = item.split("=", 1)
        stats, figures = index_statistics(name, index_dir, args.batch_entries)
        all_stats = [s for s in all_stats if s["name"] != name] + [stats]
        print(json.dumps(stats, indent=2), flush=True)
        with open(output_dir / f"index_stats_{name}.html", "w", encoding="utf-8") as f:
            for i, fig in enumerate(figures):
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        json_path.write_text(json.dumps(all_stats, indent=2))

    table = pl.DataFrame(all_stats).drop("index_dir")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=50):
        print(table.transpose(include_header=True, column_names="name"))
    table.write_csv(output_dir / "index_stats.csv")
    print(f"Saved statistics and histograms to {output_dir}")


if __name__ == "__main__":
    main()
