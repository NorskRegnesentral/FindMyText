"""Summary statistics and histograms for one or more merged fingerprint indexes.

Distributions are kept as value -> frequency tables (np.bincount), so quantiles and
histograms stay cheap even for billions of fingerprints.  One sequential pass over
``postings.dat`` is needed for the number of distinct documents per fingerprint.

Run from the repository root:
    python -m experiments.index_stats
    python -m experiments.index_stats --repair-plots
    python -m experiments.index_stats /path/to/corpora --output-dir /path/to/analysis
    python -m experiments.index_stats nynorsk=/path/to/index bokmal=/path/to/index \
        --output-dir /path/to/analysis
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from findmytext.indexing import POSTING_DTYPE, DiskBasedIndex

QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
ROOT_PATH = Path("/nr/project/stat/COPY.AI/data/Lars")
DEFAULT_INDEXES = [
    f"{name}={ROOT_PATH / ('hplt_' + name) / 'findmytext_build/index'}"
    for name in ("nynorsk", "bokmal", "icelandic", "latvian")
]
DEFAULT_OUTPUT_DIR = ROOT_PATH / "index_analysis"


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
        f"{prefix}_min": int(values[0]) if n else 0,
        f"{prefix}_max": int(values[-1]) if n else 0,
    }
    for level in QUANTILES:
        position = int(np.searchsorted(cumulative, level * n, side="left"))
        summary[f"{prefix}_p{level * 100:g}"] = int(values[min(position, len(values) - 1)]) if n else 0
    return summary


def _distribution_markers(summary, prefix, x_scale):
    labels_by_value = {}
    for level in QUANTILES:
        value = summary[f"{prefix}_p{level * 100:g}"]
        labels_by_value.setdefault(value, []).append(f"P{level * 100:g}: {value:,}")
    labels_by_value.setdefault(summary[f"{prefix}_max"], []).append(
        f"max: {summary[f'{prefix}_max']:,}"
    )
    shapes = []
    annotations = []
    for value, labels in labels_by_value.items():
        if x_scale == "log" and value <= 0:
            raise ValueError("Logarithmic distribution markers must be positive")
        shapes.append({
            "type": "line", "x0": value, "x1": value, "xref": "x",
            "y0": 0, "y1": 1, "yref": "paper", "line": {"dash": "dash"},
        })
        # Plotly annotations use log10 coordinates on log axes; shapes use counts.
        annotations.append({
            "x": float(np.log10(value)) if x_scale == "log" else value,
            "xref": "x", "y": 1, "yref": "paper", "text": " / ".join(labels),
            "showarrow": False,
            "xanchor": "right" if value == summary[f"{prefix}_max"] else "left",
            "yanchor": "bottom",
        })
    return shapes, annotations


def _histogram(table, title, x_title, summary, prefix):
    values = np.flatnonzero(table)
    # Zero-fingerprint documents must remain visible.
    x_scale = "linear" if not len(values) or values[0] == 0 else "log"
    fig = go.Figure(
        go.Bar(
            x=values,
            y=table[values],
            width=1,
            hovertemplate=f"{x_title}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )
    shapes, annotations = _distribution_markers(summary, prefix, x_scale)
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        title=f"{title} (mean {summary[f'{prefix}_mean']:,.2f})",
        xaxis={"type": x_scale, "title": f"{x_title} ({x_scale} scale)"},
        yaxis={"type": "log", "title": "Count (log scale)"},
    )
    return fig


def comparison_figure(all_stats):
    """One point per corpus, with singleton fraction encoded by marker color."""
    fig = go.Figure(go.Scatter(
        x=[row["n_unique_fingerprints"] for row in all_stats],
        y=[row["singleton_fingerprints"] for row in all_stats],
        mode="markers+text",
        text=[row["name"] for row in all_stats],
        textposition="top center",
        marker={
            "size": 16,
            "color": [row["singleton_fingerprints_fraction"] for row in all_stats],
            "colorscale": "Viridis",
            "cmin": 0, "cmax": 1,
            "colorbar": {"title": "Singleton fraction", "tickformat": ".0%"},
        },
        customdata=[
            [row["name"], row["n_documents"], row["n_postings"],
             row["n_unique_doc_fingerprint_pairs"],
             row["singleton_fingerprints_fraction"], row["index_gb"],
             row["singleton_fingerprints"], row["index_dir"]]
            for row in all_stats
        ],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Documents: %{customdata[1]:,}<br>"
            "Distinct fingerprints: %{x:,}<br>"
            "Singleton fingerprints: %{customdata[6]:,}<br>"
            "Singleton fraction: %{customdata[4]:.2%}<br>"
            "Positional postings: %{customdata[2]:,}<br>"
            "Document/fingerprint pairs: %{customdata[3]:,}<br>"
            "Index size: %{customdata[5]:.2f} GB<br>"
            "Index: %{customdata[7]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Distinct versus singleton fingerprints by corpus",
        xaxis={"title": "Distinct fingerprints (n_unique_fingerprints)",
               "rangemode": "tozero", "tickformat": "~s"},
        yaxis={"title": "Fingerprints in one document (singleton_fingerprints)",
               "rangemode": "tozero", "tickformat": "~s"},
        margin={"t": 80},
    )
    return fig


def singleton_fraction_figure(all_stats):
    fig = comparison_figure(all_stats)
    fig.update_traces(
        y=[row["singleton_fingerprints_fraction"] for row in all_stats],
    )
    fig.update_layout(
        title="Distinct fingerprints versus singleton fraction by corpus",
        yaxis={"title": "Singleton fraction", "tickformat": ".0%", "range": [0, 1]},
    )
    return fig


def _write_comparison(all_stats, output_dir):
    path = output_dir / "index_stats_comparison.html"
    figures = [comparison_figure(all_stats), singleton_fraction_figure(all_stats)]
    with path.open("w", encoding="utf-8") as output:
        output.write("<!doctype html><html><head><meta charset='utf-8'>"
                     "<title>Corpus fingerprint comparisons</title></head><body>")
        for i, figure in enumerate(figures):
            output.write(figure.to_html(full_html=False, include_plotlyjs=i == 0))
        output.write("</body></html>")
    print(f"Saved corpus comparison to {path}")
    return path


def _saved_plot_layouts(html):
    """Locate JSON layouts without evaluating JavaScript or rewriting trace data."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*', html):
        _, data_end = decoder.raw_decode(html, match.end())
        separator = re.compile(r"\s*,\s*").match(html, data_end)
        if separator is None:
            raise ValueError("Expected a JSON layout after Plotly trace data")
        layout_start = separator.end()
        layout, layout_end = decoder.raw_decode(html, layout_start)
        if not isinstance(layout, dict):
            raise ValueError("Expected a Plotly layout object")
        yield layout_start, layout_end, layout


def repair_saved_reports(output_dir):
    """Repair saved HTML and build the comparison using only existing reports."""
    all_stats = json.loads((output_dir / "index_stats.json").read_text(encoding="utf-8"))
    if not all_stats:
        raise ValueError("No saved index statistics found")
    prefixes = ("fingerprints_per_doc", "docs_per_fingerprint", "postings_per_fingerprint")
    repairs = []
    for stats in all_stats:
        path = (output_dir / "index_stats.json").with_name(
            f"index_stats_{stats['name']}.html"
        )
        html = path.read_text(encoding="utf-8")
        layouts = list(_saved_plot_layouts(html))
        if len(layouts) != len(prefixes):
            raise ValueError(f"Expected three histogram charts in {path}; found {len(layouts)}")
        replacements = []
        for (start, end, layout), prefix in zip(layouts, prefixes):
            x_scale = layout["xaxis"]["type"]
            if x_scale not in {"linear", "log"}:
                raise ValueError(f"Unsupported x-axis scale {x_scale!r} in {path}")
            layout["shapes"], layout["annotations"] = _distribution_markers(
                stats, prefix, x_scale
            )
            layout["xaxis"].pop("range", None)
            layout["xaxis"]["autorange"] = True
            replacements.append((start, end, json.dumps(layout, ensure_ascii=True)))
        for start, end, replacement in reversed(replacements):
            html = html[:start] + replacement + html[end:]
        repairs.append((path, html))

    # Validate every report before overwriting any, retaining original HTML backups.
    for path, html in repairs:
        backup = path.with_suffix(".html.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(html, encoding="utf-8")
        print(f"Repaired {path} (original backup: {backup})")
    _write_comparison(all_stats, output_dir)


def index_statistics(name, index_dir, batch_entries):
    if batch_entries < 1:
        raise ValueError("batch_entries must be positive")
    index_dir = str(index_dir)
    index = DiskBasedIndex(index_dir)
    n_docs = len(index.to_external_doc_id)
    n_fingerprints = len(index.fingerprints)
    lengths = np.asarray(index.lengths, dtype=np.int64)
    total_postings = int(lengths.sum())

    # Without a forward index, accumulate these during the same posting pass.
    has_forward_index = index.forward_offsets is not None
    fingerprints_per_doc = (
        np.diff(np.asarray(index.forward_offsets, dtype=np.int64))
        if has_forward_index
        else np.zeros(n_docs, dtype=np.int64)
    )

    # Posting entries per fingerprint (includes repeats within a document) = one pread.
    postings_per_fp_table = np.bincount(lengths)

    # Distinct documents per fingerprint, and the forward_topk read volume:
    # every document holding fingerprint f reads f's whole posting list once.
    docs_per_fp_table = np.zeros(2, dtype=np.int64)
    pair_updates = 0
    forward_topk_entries_read = 0
    fp_offset = 0
    for batch_fps, group, doc_ids in index.iter_unique_posting_batches(batch_entries):
        docs_per_fp = np.bincount(group, minlength=len(batch_fps)).astype(np.int64)
        if not has_forward_index:
            np.add.at(fingerprints_per_doc, doc_ids, 1)
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

    fingerprints_per_doc_table = np.bincount(fingerprints_per_doc, minlength=1)
    forward_entries = int(fingerprints_per_doc.sum())
    singletons = int(docs_per_fp_table[1])
    stats = {
        "name": name,
        "index_dir": index_dir,
        "n_documents": n_docs,
        "n_documents_without_fingerprints": int(fingerprints_per_doc_table[0]),
        "n_unique_fingerprints": n_fingerprints,
        "n_postings": total_postings,
        "n_unique_doc_fingerprint_pairs": forward_entries,
        "has_forward_index": has_forward_index,
        "postings_layout": index.postings_layout,
        "n_repeated_postings": total_postings - forward_entries,
        "repeated_postings_fraction": (
            1 - forward_entries / total_postings if total_postings else 0.0
        ),
        "singleton_fingerprints": singletons,
        "singleton_fingerprints_fraction": (
            singletons / n_fingerprints if n_fingerprints else 0.0
        ),
        "shared_fingerprints": int(docs_per_fp_table[2:].sum()),
        "fingerprints_in_exactly_two_documents": (
            int(docs_per_fp_table[2]) if len(docs_per_fp_table) > 2 else 0
        ),
        "fingerprints_in_all_documents": (
            int(docs_per_fp_table[n_docs]) if 0 < n_docs < len(docs_per_fp_table) else 0
        ),
        "inverted_pair_updates": pair_updates,
        "forward_topk_gb_read": forward_topk_entries_read * POSTING_DTYPE.itemsize / 1e9,
        "forward_topk_mb_read_per_document": forward_topk_entries_read
        * POSTING_DTYPE.itemsize
        / 1e6
        / n_docs if n_docs else 0.0,
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


def resolve_indexes(items):
    """Accept index paths, corpus folders, a parent folder, or name=path pairs."""
    indexes = []

    def find_index(folder):
        return next(
            (
                candidate
                for candidate in (folder, folder / "index", folder / "findmytext_build/index")
                if (candidate / "meta.json").is_file()
            ),
            None,
        )

    for item in items:
        name, separator, path = item.partition("=")
        folder = Path(path if separator else item).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")
        index_dir = find_index(folder)
        if separator or index_dir is not None:
            if index_dir is None:
                raise ValueError(f"No merged index found in {folder}")
            indexes.append((name if separator else folder.name, index_dir))
        else:
            discovered = [
                (child.name, found)
                for child in sorted(folder.iterdir())
                if child.is_dir() and (found := find_index(child)) is not None
            ]
            if not discovered:
                raise ValueError(f"No merged indexes found in {folder} or its corpus folders")
            indexes.extend(discovered)
    names = [name for name, _ in indexes]
    if any(not name or Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Index names must be simple file names")
    if len(set(names)) != len(names):
        raise ValueError("Index names must be unique; use name=path to disambiguate")
    return indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "indexes", nargs="*", default=DEFAULT_INDEXES,
        help="Index paths, corpus folders, a parent folder containing corpora, or name=path pairs. "
             "Defaults to the Nynorsk, Bokmal, Icelandic, and Latvian HPLT indexes "
             f"under {ROOT_PATH}.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--batch-entries", type=int, default=1_000_000)
    parser.add_argument(
        "--repair-plots", action="store_true",
        help="Repair saved HTML and create the corpus comparison from existing reports only; "
             "does not read indexes. Uses --output-dir.",
    )
    args = parser.parse_args()
    if args.repair_plots:
        if args.indexes != DEFAULT_INDEXES:
            parser.error("--repair-plots uses saved reports; omit index arguments")
        repair_saved_reports(args.output_dir)
        return
    if args.batch_entries < 1:
        parser.error("--batch-entries must be positive")
    try:
        indexes = resolve_indexes(args.indexes)
    except ValueError as error:
        parser.error(str(error))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "index_stats.json"
    # Keep results from earlier runs so indexes can be added one at a time.
    all_stats = json.loads(json_path.read_text()) if json_path.exists() else []
    print(f"Found {len(indexes)} index(es). Each requires one sequential posting scan.")
    for name, index_dir in indexes:
        print(f"\nAnalyzing {name}: {index_dir}", flush=True)
        stats, figures = index_statistics(name, index_dir, args.batch_entries)
        all_stats = [s for s in all_stats if s["name"] != name] + [stats]
        print(json.dumps(stats, indent=2), flush=True)
        with open(output_dir / f"index_stats_{name}.html", "w", encoding="utf-8") as f:
            f.write("<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>Index statistics</title></head><body>")
            for i, fig in enumerate(figures):
                f.write(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
            f.write("</body></html>")
        json_path.write_text(json.dumps(all_stats, indent=2))

    table = pl.DataFrame(all_stats).drop("index_dir")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=50):
        print(table.transpose(include_header=True, column_names="name"))
    table.write_csv(output_dir / "index_stats.csv")
    _write_comparison(all_stats, output_dir)
    print(f"Saved statistics and histograms to {output_dir}")


if __name__ == "__main__":
    main()
