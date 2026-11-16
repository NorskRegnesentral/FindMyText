"""Read frozen collision reports and regroup saved occurrence data.

Legacy31 collection was retired when production became rolling64-only.
Use --report-dir to render an existing completed report without reading indexes.
See HASHING_FINDINGS.md for the historical experiment results and methodology.
"""

import argparse
import hashlib
import json
import random
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from findmytext import utils
from findmytext.winnower import winnow

ROOT = Path("/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk")
PERSONALIZATION = b"FMT-kgram64-v1"


def frame_tokens(tokens):
    """Length-prefix UTF-8 tokens so k-gram identity is unambiguous."""
    encoded = [token.encode("utf-8") for token in tokens]
    return [len(token).to_bytes(4, "little") + token for token in encoded]


def decode_kgram(value):
    tokens = []
    offset = 0
    while offset < len(value):
        length = int.from_bytes(value[offset:offset + 4], "little")
        offset += 4
        tokens.append(value[offset:offset + length].decode("utf-8"))
        offset += length
    return tokens


def wide_fingerprints(framed_tokens, length):
    grams = [
        b"".join(framed_tokens[i:i + length])
        for i in range(max(0, len(framed_tokens) - length + 1))
    ]
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(
            gram, digest_size=8, person=PERSONALIZATION,
        ).digest(), "little") for gram in grams),
        dtype=np.uint64,
        count=len(grams),
    )
    return grams, hashes


winnow_uint64 = winnow


def sample_documents(shards, sample_size, seed, min_length, max_length):
    if sample_size < 1:
        raise ValueError("Sample size must be positive")
    reservoir = []
    rng = random.Random(seed)
    total_population = 0
    populations = []
    for shard in shards:
        population = 0
        for population, record in enumerate(utils.stream_json_zst(
            str(shard), min_length=min_length, max_length=max_length,
        ), start=1):
            total_population += 1
            item = {"id": str(record["id"]), "text": record["text"], "shard": str(shard)}
            if total_population <= sample_size:
                reservoir.append(item)
            else:
                replacement = rng.randrange(total_population)
                if replacement < sample_size:
                    reservoir[replacement] = item
        populations.append({"shard": str(shard), "eligible_documents": population})
        print(f"Scanned {population:,} eligible documents from {shard.name}", flush=True)
    if total_population < sample_size:
        raise ValueError(f"Only {total_population} eligible documents, fewer than {sample_size}")
    for population in populations:
        population["sampled_documents"] = sum(
            row["shard"] == population["shard"] for row in reservoir
        )
    ids = [row["id"] for row in reservoir]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample contains duplicate source document IDs across shards")
    return sorted(reservoir, key=lambda row: (row["shard"], row["id"])), populations


def _occurrence_frame(rows):
    return pl.DataFrame(rows, schema={
        "hash": pl.UInt64, "signature64": pl.UInt64, "kgram": pl.Binary,
        "document": pl.UInt32, "position": pl.UInt32,
    }, orient="row")


def summarize_pipeline(raw_dir, pipeline, output_dir, input_paths=None):
    print(f"Grouping {pipeline} hashes by exact normalized k-gram...", flush=True)
    occurrences = pl.scan_parquet(
        input_paths if input_paths is not None else raw_dir / f"{pipeline}-*.parquet"
    )
    variants = occurrences.group_by("hash", "kgram").agg(
        pl.col("document").n_unique().alias("n_documents"),
        pl.len().cast(pl.UInt64).alias("n_occurrences"),
        pl.col("signature64").first(),
        pl.col("document").first().alias("example_document"),
        pl.col("position").first().alias("example_position"),
    ).collect(engine="streaming")
    variants_path = output_dir / f"{pipeline}_kgram_variants.parquet"
    variants.write_parquet(variants_path)
    multiplicities = variants.group_by("hash").agg(
        pl.len().alias("n_distinct_kgrams"),
        pl.col("signature64").n_unique().alias("n_distinct_signatures64"),
        pl.col("n_documents").max().alias("max_kgram_document_frequency"),
        (pl.col("n_documents") == 1).sum().alias("n_single_document_kgrams"),
    )
    frequencies = occurrences.group_by("hash").agg(
        pl.col("document").n_unique().alias("n_documents"),
        pl.len().cast(pl.UInt64).alias("n_occurrences"),
    ).collect(engine="streaming")
    buckets = frequencies.join(multiplicities, on="hash").sort("hash")
    buckets.write_csv(output_dir / f"{pipeline}_per_hash.csv")
    buckets.write_parquet(output_dir / f"{pipeline}_per_hash.parquet")
    histogram = buckets.group_by("n_distinct_kgrams").agg(
        pl.len().alias("n_hashes"),
        pl.col("n_occurrences").sum().alias("n_occurrences"),
    ).sort("n_distinct_kgrams")
    histogram.write_csv(output_dir / f"{pipeline}_collision_multiplicity.csv")
    collided = buckets.filter(pl.col("n_distinct_kgrams") > 1)
    unique_hashes = buckets.height
    n_occurrences = int(buckets["n_occurrences"].sum())
    summary = {
        "n_unique_hashes": unique_hashes,
        "n_exact_distinct_kgrams": variants.height,
        "n_selected_occurrences": n_occurrences,
        "n_unique_document_hash_pairs": int(buckets["n_documents"].sum()),
        "n_singleton_hashes": int((buckets["n_documents"] == 1).sum()),
        "singleton_hash_fraction": (
            float((buckets["n_documents"] == 1).sum() / unique_hashes) if unique_hashes else 0.0
        ),
        "n_collided_hashes": collided.height,
        "collided_hash_fraction": collided.height / unique_hashes if unique_hashes else 0.0,
        "n_distinct_kgrams_in_collided_hashes": int(collided["n_distinct_kgrams"].sum()),
        "n_occurrences_in_collided_hashes": int(collided["n_occurrences"].sum()),
        "occurrence_fraction_in_collided_hashes": (
            float(collided["n_occurrences"].sum() / n_occurrences) if n_occurrences else 0.0
        ),
        "n_distinct_kgrams_merged_by_hashing": variants.height - unique_hashes,
        "n_unique_independent64_signatures": variants["signature64"].n_unique(),
        "n_kgrams_merged_by_independent64_signatures": (
            variants.height - variants["signature64"].n_unique()
        ),
        "max_distinct_kgrams_per_hash": int(
            buckets["n_distinct_kgrams"].to_numpy().max(initial=0)
        ),
        "n_shared_hashes_without_any_actually_shared_kgram": collided.filter(
            (pl.col("n_documents") > 1) & (pl.col("max_kgram_document_frequency") == 1)
        ).height,
        "collision_multiplicity": histogram.to_dicts(),
    }
    exact_identity = {
        "n_unique_fingerprints": variants.height,
        "n_singleton_fingerprints": int((variants["n_documents"] == 1).sum()),
        "singleton_fraction": (
            float((variants["n_documents"] == 1).sum() / variants.height)
            if variants.height else 0.0
        ),
        "n_unique_document_fingerprint_pairs": int(variants["n_documents"].sum()),
    }
    # Bound examples while preserving the complete per-hash and variant tables.
    example_hashes = collided.sort(
        ["n_distinct_kgrams", "hash"], descending=[True, False]
    ).head(5)
    examples = []
    for bucket in example_hashes.to_dicts():
        items = variants.filter(pl.col("hash") == bucket["hash"]).sort("kgram").head(3)
        examples.append({
            **bucket,
            "variants": [
                {**{key: value for key, value in item.items() if key != "kgram"},
                 "tokens": decode_kgram(item["kgram"])}
                for item in items.to_dicts()
            ],
        })
    return summary, exact_identity, examples


def _partition_rows(scan, key, destination):
    if (destination / "_SUCCESS").is_file():
        return
    if destination.exists():
        raise ValueError(f"Incomplete partition directory: {destination}; "
                         "remove this derived directory before retrying aggregation")
    destination.mkdir(parents=True)
    scan.sink_parquet(
        pl.PartitionBy(destination, key=key, include_key=False,
                       approximate_bytes_per_file=128_000_000),
        mkdir=True,
    )
    (destination / "_SUCCESS").touch()


def summarize_partitioned(raw_dir, pipeline, output_dir, partitions):
    """Group exact k-grams in hash-disjoint partitions, retaining restart markers."""
    work = output_dir / f"{pipeline}_aggregation_{partitions}"
    partitioned = work / "input"
    scan = pl.scan_parquet(raw_dir / f"{pipeline}-*.parquet")
    _partition_rows(scan, {"partition": pl.col("hash") % partitions}, partitioned)
    partials = []
    for part in sorted(path for path in partitioned.iterdir() if path.is_dir()):
        result_dir = work / part.name
        result_dir.mkdir(exist_ok=True, parents=True)
        checkpoint = result_dir / "summary.json"
        if checkpoint.exists():
            summary, exact, examples = json.loads(checkpoint.read_text())
        else:
            print(f"Aggregating {pipeline} {part.name}", flush=True)
            summary, exact, examples = summarize_pipeline(
                raw_dir, pipeline, result_dir, sorted(part.glob("*.parquet")),
            )
            checkpoint.write_text(json.dumps([summary, exact, examples]))
        partials.append((result_dir, summary, exact, examples))
    if not partials:
        return summarize_pipeline(raw_dir, pipeline, output_dir)

    summed_fields = (
        "n_unique_hashes", "n_exact_distinct_kgrams", "n_selected_occurrences",
        "n_unique_document_hash_pairs", "n_singleton_hashes", "n_collided_hashes",
        "n_distinct_kgrams_in_collided_hashes", "n_occurrences_in_collided_hashes",
        "n_distinct_kgrams_merged_by_hashing",
        "n_shared_hashes_without_any_actually_shared_kgram",
    )
    totals = {field: sum(item[1][field] for item in partials) for field in summed_fields}
    combined: dict[str, int | float | list[dict[str, int]]] = dict(totals)
    for fraction, numerator, denominator in (
        ("singleton_hash_fraction", "n_singleton_hashes", "n_unique_hashes"),
        ("collided_hash_fraction", "n_collided_hashes", "n_unique_hashes"),
        ("occurrence_fraction_in_collided_hashes", "n_occurrences_in_collided_hashes",
         "n_selected_occurrences"),
    ):
        combined[fraction] = totals[numerator] / totals[denominator] if totals[denominator] else 0.0
    combined["max_distinct_kgrams_per_hash"] = max(
        item[1]["max_distinct_kgrams_per_hash"] for item in partials
    )
    histogram = {}
    for _, summary, _, _ in partials:
        for row in summary["collision_multiplicity"]:
            target = histogram.setdefault(row["n_distinct_kgrams"], {
                "n_distinct_kgrams": row["n_distinct_kgrams"], "n_hashes": 0, "n_occurrences": 0,
            })
            target["n_hashes"] += row["n_hashes"]
            target["n_occurrences"] += row["n_occurrences"]
    combined["collision_multiplicity"] = [histogram[key] for key in sorted(histogram)]
    pl.DataFrame(combined["collision_multiplicity"]).write_csv(
        output_dir / f"{pipeline}_collision_multiplicity.csv"
    )

    variant_paths = [part / f"{pipeline}_kgram_variants.parquet" for part, _, _, _ in partials]
    variants = pl.scan_parquet(variant_paths)
    # A signature collision can cross old-hash partitions; count globally by signature.
    signature_parts = work / "signatures"
    _partition_rows(
        variants.select("signature64"),
        {"partition": pl.col("signature64") % partitions}, signature_parts,
    )
    n_signatures = sum(
        int(pl.scan_parquet(sorted(part.glob("*.parquet"))).select(
            pl.col("signature64").n_unique()
        ).collect(engine="streaming").item())
        for part in signature_parts.iterdir() if part.is_dir()
    )
    combined["n_unique_independent64_signatures"] = n_signatures
    combined["n_kgrams_merged_by_independent64_signatures"] = (
        totals["n_exact_distinct_kgrams"] - n_signatures
    )
    variants.sink_parquet(output_dir / f"{pipeline}_kgram_variants.parquet")
    bucket_scan = pl.scan_parquet([
        part / f"{pipeline}_per_hash.parquet" for part, _, _, _ in partials
    ])
    bucket_scan.sink_parquet(output_dir / f"{pipeline}_per_hash.parquet")
    bucket_scan.sink_csv(output_dir / f"{pipeline}_per_hash.csv")
    exact_totals = {
        key: sum(part[2][key] for part in partials)
        for key in ("n_unique_fingerprints", "n_singleton_fingerprints",
                    "n_unique_document_fingerprint_pairs")
    }
    exact = {**exact_totals, "singleton_fraction": (
        exact_totals["n_singleton_fingerprints"] / exact_totals["n_unique_fingerprints"]
        if exact_totals["n_unique_fingerprints"] else 0.0
    )}
    examples = sorted(
        (example for part in partials for example in part[3]),
        key=lambda row: (-row["n_distinct_kgrams"], row["hash"]),
    )[:5]
    return combined, exact, examples


def write_figures(report, output_dir):
    old = report["fixed_selection"]["old_hashes"]
    exact = report["fixed_selection"]["exact_kgrams"]
    wide = report["full_64bit"]["hashes"]
    full_corpus = report["scope"] == "full_corpus_census"
    scope = "Full-corpus" if full_corpus else "Sample-only"
    population = "indexed corpus" if full_corpus else "sampled documents"
    wide_name = report["settings"].get("wide_hash_method", "blake2b64")
    labels = ["Current hashes", "Exact k-grams, same positions", f"{wide_name}, re-winnowed"]
    figures = [
        go.Figure(go.Bar(
            x=labels,
            y=[old["singleton_hash_fraction"], exact["singleton_fraction"],
               wide["singleton_hash_fraction"]],
        )).update_layout(title=f"Singleton fractions within the {population}",
                         yaxis={"title": "Fraction", "tickformat": ".0%", "range": [0, 1]}),
        go.Figure(go.Bar(
            x=[str(row["n_distinct_kgrams"]) for row in old["collision_multiplicity"]],
            y=[row["n_hashes"] for row in old["collision_multiplicity"]],
        )).update_layout(title="Current hash: number of different normalized k-grams per value",
                         xaxis_title="Distinct k-grams per hash",
                         yaxis={"title": "Number of hashes", "type": "log"}),
    ]
    with (output_dir / "collision_study.html").open("w", encoding="utf-8") as output:
        output.write("<!doctype html><html><head><meta charset='utf-8'>"
                     f"<title>{scope} hash collision study</title>"
                     "<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}"
                     "td,th{padding:.5em;border:1px solid #ccc}table{border-collapse:collapse}"
                     "</style></head><body>")
        output.write(f"<h1>{scope} collision study</h1><p>Document frequencies and "
                     f"singleton fractions refer to the {population}.</p>")
        output.write(
            f"<p>{report['counters']['documents']:,} analyzed documents; "
            f"{report['counters']['validated_forward_documents']:,} recomputed old forward "
            "sets verified against the source index. Existing indexes were not modified.</p>"
            "<h2>What the experiments discover</h2>"
            "<p><b>Fixed selection:</b> keep the current selected positions and count exact "
            "normalized token tuples per old hash. Different tuples sharing a hash are "
            "collisions; repeated identical tuples are not. This isolates hash merging.</p>"
            f"<p><b>Full 64-bit pipeline:</b> hash every k-gram with {escape(wide_name)} "
            "and re-winnow. This measures the combined effects of a wider hash space and "
            "changed window minima, not collisions alone.</p>"
            "<h2>Comparison</h2><table><tr><th>Representation</th>"
            "<th>Distinct values</th><th>Singleton fraction</th></tr>"
        )
        for label, unique, fraction in (
            (labels[0], old["n_unique_hashes"], old["singleton_hash_fraction"]),
            (labels[1], exact["n_unique_fingerprints"], exact["singleton_fraction"]),
            (labels[2], wide["n_unique_hashes"], wide["singleton_hash_fraction"]),
        ):
            output.write(f"<tr><td>{escape(label)}</td><td>{unique:,}</td>"
                         f"<td>{fraction:.4%}</td></tr>")
        output.write(
            "</table><h2>Observed collisions in the current hashes</h2>"
            f"<p>{old['n_collided_hashes']:,} collided hashes "
            f"({old['collided_hash_fraction']:.4%} of observed hashes); "
            f"{old['n_distinct_kgrams_merged_by_hashing']:,} distinct k-gram identities merged. "
            f"{old['n_occurrences_in_collided_hashes']:,} selected occurrences "
            f"({old['occurrence_fraction_in_collided_hashes']:.4%}) lie in collided buckets.</p>"
            "<table><tr><th>Different k-grams per hash</th><th>Number of hashes</th></tr>"
        )
        for row in old["collision_multiplicity"]:
            output.write(f"<tr><td>{row['n_distinct_kgrams']:,}</td>"
                         f"<td>{row['n_hashes']:,}</td></tr>")
        output.write("</table>")
        examples = json.loads((output_dir / "collision_examples.json").read_text())
        output.write("<h2>Verified examples: normalized k-grams</h2>")
        for example in examples["old"]:
            output.write(f"<h3>Old hash {example['hash']:,}</h3><ul>")
            for variant in example["variants"]:
                phrase = escape(" ".join(variant["tokens"]))
                output.write(f"<li><code>{phrase}</code> &mdash; "
                             f"{variant['n_documents']:,} analyzed document(s), "
                             f"{variant['n_occurrences']:,} occurrence(s)</li>")
            output.write("</ul>")
        output.write(
            "<p>Counts use exact normalized k-grams. Zero observed wide-hash collisions "
            "is an observation for the analyzed data, not a guarantee about unseen data.</p>"
        )
        for number, figure in enumerate(figures):
            output.write(figure.to_html(full_html=False, include_plotlyjs=number == 0))
        output.write("</body></html>")


def run_study(_args):
    raise ValueError(
        "Legacy31 collision collection is retired; historical reports are frozen. "
        "Use --report-dir to render a completed report, or rebuild indexes with rolling64-v1."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="Directory containing an existing report.json and collision_examples.json.")
    args = parser.parse_args()
    report = json.loads((args.report_dir / "report.json").read_text())
    write_figures(report, args.report_dir)
    print(f"Rendered frozen report: {args.report_dir / 'collision_study.html'}")


if __name__ == "__main__":
    main()
