"""Time compute_similarities_forward_topk with sequential vs. threaded posting reads.

With ``--profile-documents N`` it also times ``get_document_candidates`` for N random
documents per configuration, to show how posting-list volume affects read time.

Example:
    python experiments/benchmark_posting_threads.py \
        /nr/project/stat/COPY.AI/data/Lars/hplt_icelandic/findmytext_build/index \
        --threads 1 2 4 8 16 32 --max-documents 100000 --profile-documents 2000
"""

import argparse
import os
import time

import numpy as np
import plotly.graph_objects as go
import polars as pl

from findmytext import similarity
from findmytext.indexing import DiskBasedIndex, similarity_file_name


def profile_documents(index_dir, args, configs):
    """Per-document query time versus the number of posting entries read."""
    rng = np.random.default_rng(args.seed)
    probe = DiskBasedIndex(index_dir)
    n_docs = len(probe.to_external_doc_id)
    documents = rng.choice(n_docs, size=min(args.profile_documents, n_docs), replace=False)

    features = []
    for doc in documents:
        fingerprints = probe.get_document_fingerprints(int(doc))
        idx = np.searchsorted(probe.fingerprints, fingerprints)
        idx = np.clip(idx, 0, len(probe.fingerprints) - 1)
        found = idx[probe.fingerprints[idx] == fingerprints]
        lengths = np.asarray(probe.lengths[found], dtype=np.int64)
        features.append(
            {
                "document": int(doc),
                "n_fingerprints": len(fingerprints),
                "posting_entries_read": int(lengths.sum()),
                "max_posting_length": int(lengths.max()) if len(lengths) else 0,
            }
        )
    del probe

    rows = []
    # The first (sequential) pass only warms the page cache.
    passes = [1] + [1] + configs
    for pass_index, threads in enumerate(passes):
        index = DiskBasedIndex(index_dir, posting_read_threads=threads)
        for feature in features:
            started = time.perf_counter()
            index.get_document_candidates(
                feature["document"],
                top_k=args.top_k,
                min_fingerprints=args.min_shared_fingerprints,
            )
            if pass_index > 0:
                rows.append(
                    {
                        "index_dir": index_dir,
                        "threads": threads,
                        **feature,
                        "seconds": time.perf_counter() - started,
                    }
                )
        del index
    return rows


def plot_profile(profile, path):
    """Median query time per posting-volume bin, one line per configuration."""
    top = np.log10(max(int(profile["posting_entries_read"].max()), 10)) + 1e-9
    edges = np.unique(np.logspace(0, top, 25).astype(np.int64))
    binned = (
        profile.with_columns(
            pl.Series(
                "bin",
                np.searchsorted(edges, profile["posting_entries_read"].to_numpy(), side="right"),
            ),
            (pl.lit("threads ") + pl.col("threads").cast(pl.String)).alias("config"),
        )
        .group_by("index_dir", "config", "threads", "bin")
        .agg(
            pl.col("posting_entries_read").median().alias("entries"),
            pl.col("seconds").median().alias("median_seconds"),
            pl.len().alias("n"),
        )
        .sort("index_dir", "threads", "config", "bin")
    )
    fig = go.Figure()
    for (index_dir, config), group in binned.group_by(
        "index_dir", "config", maintain_order=True
    ):
        dataset = os.path.basename(os.path.dirname(os.path.dirname(index_dir.rstrip("/"))))
        fig.add_trace(
            go.Scatter(
                x=group["entries"],
                y=group["median_seconds"],
                mode="lines+markers",
                name=f"{dataset}: {config}",
                customdata=group["n"],
                hovertemplate="entries=%{x:,}<br>median=%{y:.4f}s<br>docs=%{customdata}",
            )
        )
    fig.update_layout(
        title="get_document_candidates time vs. posting entries read",
        xaxis={"type": "log", "title": "Posting entries read for the document (log)"},
        yaxis={"type": "log", "title": "Median seconds per document (log)"},
    )
    fig.write_html(path)


def run(index_dir, args, threads):
    started = time.perf_counter()
    path = similarity.compute_similarities_forward_topk(
        index_dir=index_dir,
        top_k=args.top_k,
        min_shared_fingerprints=args.min_shared_fingerprints,
        verbose=args.verbose,
        posting_read_threads=threads,
        max_documents=args.max_documents,
    )
    elapsed = time.perf_counter() - started
    return elapsed, np.load(path)


def benchmark(index_dir, args):
    output_path = os.path.join(index_dir, similarity_file_name("forward_topk"))
    backup_path = output_path + ".benchmark_backup"
    # Keep an existing (possibly full) result from being overwritten by the benchmark.
    if os.path.exists(output_path):
        os.replace(output_path, backup_path)

    configs = args.threads
    rows = []
    reference = None
    try:
        if args.warmup:
            print("Warm-up run (sequential) to populate the page cache...", flush=True)
            run(index_dir, args, 1)
        for repeat in range(args.repeats):
            for threads in configs:
                elapsed, result = run(index_dir, args, threads)
                if reference is None:
                    reference = result
                identical = np.array_equal(result, reference)
                print(
                    f"threads={threads:>3} repeat={repeat}: "
                    f"{elapsed:,.1f}s (identical={identical})",
                    flush=True,
                )
                rows.append(
                    {
                        "index_dir": index_dir,
                        "threads": threads,
                        "repeat": repeat,
                        "seconds": elapsed,
                        "n_pairs": len(result),
                        "identical": identical,
                    }
                )
    finally:
        if os.path.exists(backup_path):
            os.replace(backup_path, output_path)
        elif os.path.exists(output_path):
            os.remove(output_path)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dirs", nargs="+", help="Merged index directories.")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-shared-fingerprints", type=int, default=5)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Only query the first N documents (default: all).",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--verbose", action="store_true", help="Show progress bars.")
    parser.add_argument("--output", default=None, help="Optional CSV for results.")
    parser.add_argument(
        "--profile-documents",
        type=int,
        default=0,
        help="Also time N random documents individually per configuration.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.profile_documents:
        configs = args.threads
        profile = pl.DataFrame(
            [
                row
                for index_dir in args.index_dirs
                for row in profile_documents(index_dir, args, configs)
            ]
        )
        with pl.Config(tbl_rows=-1, fmt_str_lengths=200):
            print(
                profile.group_by("index_dir", "threads")
                .agg(
                    pl.col("seconds").sum().alias("total_seconds"),
                    pl.col("seconds").median().alias("median_seconds"),
                    pl.corr(
                        pl.col("posting_entries_read").log1p(), pl.col("seconds").log()
                    ).alias("corr_log_entries_log_seconds"),
                )
                .sort("index_dir", "threads")
            )
        if args.output:
            stem = os.path.splitext(args.output)[0]
            profile.write_csv(f"{stem}_per_document.csv")
            plot_profile(profile, f"{stem}_per_document.html")
            print(f"Saved per-document timings to {stem}_per_document.csv/.html")

    rows = [row for index_dir in args.index_dirs for row in benchmark(index_dir, args)]
    results = pl.DataFrame(rows)
    summary = (
        results.group_by("index_dir", "threads")
        .agg(
            pl.col("seconds").median().alias("median_seconds"),
            pl.col("seconds").min().alias("min_seconds"),
            pl.col("identical").all(),
        )
        .sort("index_dir", "threads")
    )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=200):
        print(summary)
    if args.output:
        results.write_csv(args.output)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
