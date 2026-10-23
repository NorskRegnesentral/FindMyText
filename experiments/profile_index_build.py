"""Production rolling64 index builds with phases and separate CPU profiles.

The default experiment builds a 50,000-document Nynorsk index using rolling64-v1.
Timed builds are distinct from instrumented diagnostic builds. The historical
legacy/rolling64 comparison is frozen in HASHING_FINDINGS.md; existing corpus
indexes and historical results are never modified.
"""

import argparse
import cProfile
import csv
import functools
import hashlib
import html
import io
import itertools
import json
import multiprocessing as mp
import os
import pstats
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
import orjson
import plotly.graph_objects as go
import zstandard as zstd

from experiments.hash_collision_study import ROOT, sample_documents
from findmytext import index_builder, indexing
from findmytext.winnower import ROLLING64_HASH, Winnower

_ORIGINAL_BATCH = Winnower.get_winnowed_fingerprints_batch
_ORIGINAL_PRODUCE = index_builder._produce
_ORIGINAL_READER = index_builder._stream_file_to_queue
_ORIGINAL_ADD = indexing.MemoryBasedIndex.add_fingerprints
_ORIGINAL_EXPORT = indexing.MemoryBasedIndex.to_msgpack
_PROFILES = {}
_DIRECTORY = None
_PARENT_WALL = {"insertion_seconds": 0.0, "sort_export_seconds": 0.0}


class DetailedStats(pstats.Stats):
    # The runtime exposes these raw entries; the standard-library stub omits them.
    stats: dict[tuple[str, int, str], tuple[int, int, float, float, dict]]


def _child_profile(role, function, *args):
    # A fork inherits the parent's profiler; only the child's own profile is saved.
    for profiler in _PROFILES.values():
        profiler.disable()
    key = (os.getpid(), role)
    profiler = _PROFILES.setdefault(key, cProfile.Profile(timer=process_time))
    profiler.enable()
    try:
        return function(*args)
    finally:
        profiler.disable()
        _save_profile(profiler, role)


def _save_profile(profiler, role):
    if _DIRECTORY is None:
        raise RuntimeError("Profile output directory has not been configured")
    path = _DIRECTORY / f"{role}-{os.getpid()}.prof"
    temporary = path.with_suffix(".tmp")
    profiler.dump_stats(str(temporary))
    temporary.replace(path)


@functools.wraps(_ORIGINAL_BATCH)
def _profiled_batch(self, batch):
    return _child_profile("hash-worker", _ORIGINAL_BATCH, self, batch)


@functools.wraps(_ORIGINAL_PRODUCE)
def _profiled_produce(*args):
    return _child_profile("input-producer", _ORIGINAL_PRODUCE, *args)


@functools.wraps(_ORIGINAL_READER)
def _profiled_reader(*args):
    return _child_profile("merge-reader", _ORIGINAL_READER, *args)


@functools.wraps(_ORIGINAL_ADD)
def _timed_add(self, *args, **kwargs):
    started = perf_counter()
    try:
        return _ORIGINAL_ADD(self, *args, **kwargs)
    finally:
        _PARENT_WALL["insertion_seconds"] += perf_counter() - started


@functools.wraps(_ORIGINAL_EXPORT)
def _timed_export(self, *args, **kwargs):
    started = perf_counter()
    try:
        return _ORIGINAL_EXPORT(self, *args, **kwargs)
    finally:
        _PARENT_WALL["sort_export_seconds"] += perf_counter() - started


def _phase(role, function, diagnostic, *args, **kwargs):
    profiler = None
    if diagnostic:
        profiler = cProfile.Profile(timer=process_time)
        _PROFILES[(os.getpid(), role)] = profiler
        profiler.enable()
    started = perf_counter()
    try:
        result = function(*args, **kwargs)
    finally:
        elapsed = perf_counter() - started
        if profiler is not None:
            profiler.disable()
            _save_profile(profiler, role)
    return result, elapsed


def execute_build(config):
    global _DIRECTORY
    # The production extractor uses fork already. Explicitly use it for merge
    # readers too, so diagnostic hooks and warmed JIT code reach child processes.
    mp.set_start_method("fork", force=True)
    if config.get("hash_method", ROLLING64_HASH) != ROLLING64_HASH:
        raise ValueError("Legacy profiling is retired; only rolling64-v1 builds are supported. "
                         "See experiments/HASHING_FINDINGS.md for frozen comparisons.")
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    _DIRECTORY = directory / "profiles"
    diagnostic = config["diagnostic"]
    if diagnostic:
        _DIRECTORY.mkdir()
    _PROFILES.clear()
    for key in _PARENT_WALL:
        _PARENT_WALL[key] = 0.0
    source = Path(config["source"])
    method = ROLLING64_HASH
    configured = Winnower(length=4, window_size=6, hash_method=method)
    before = perf_counter()
    with source.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as text:
            for line in itertools.islice(text, 64):
                configured.get_winnowed_fingerprints(orjson.loads(line)["text"])
    warmup = perf_counter() - before
    with ExitStack() as stack:
        stack.enter_context(patch.object(indexing.MemoryBasedIndex, "add_fingerprints", _timed_add))
        stack.enter_context(patch.object(indexing.MemoryBasedIndex, "to_msgpack", _timed_export))
        if diagnostic:
            stack.enter_context(patch.object(Winnower, "get_winnowed_fingerprints_batch", _profiled_batch))
            stack.enter_context(patch.object(index_builder, "_produce", _profiled_produce))
            stack.enter_context(patch.object(index_builder, "_stream_file_to_queue", _profiled_reader))
        started = perf_counter()
        shards, extraction = _phase(
            "extract-parent", index_builder.index_file, diagnostic,
            str(source), str(directory / "shards"), length=4, window_size=6,
            nb_workers=config["nb_workers"], hash_method=method,
        )
        _, merge = _phase(
            "merge-parent", index_builder.merge_indexes, diagnostic,
            shards, str(directory / "index"), build_forward=False, save_every_n=0,
        )
        _, forward = _phase(
            "forward-parent", index_builder.build_forward_index, diagnostic,
            str(directory / "index"),
        )
        elapsed = perf_counter() - started
    index = indexing.DiskBasedIndex(str(directory / "index"))
    if len(index.to_external_doc_id) != config["documents"]:
        raise AssertionError("Completed index has an unexpected document count")
    if index.forward_offsets is None:
        raise AssertionError("Completed forward index is missing")
    counts = {
        "documents": len(index.to_external_doc_id),
        "unique_fingerprints": len(index.fingerprints),
        "postings": int(index.lengths.sum()),
        "unique_document_fingerprint_pairs": int(index.forward_offsets[-1]),
    }
    # Confirm persisted forward values match fresh hashes for representative records.
    internal_by_id = {index.to_external_doc_id[i]: i for i in range(len(index.to_external_doc_id))}
    validated = 0
    with source.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as text:
            for line in itertools.islice(text, 100):
                record = orjson.loads(line)
                fingerprints, _ = configured.get_winnowed_fingerprints(record["text"])
                if not np.array_equal(np.unique(fingerprints),
                                      index.get_document_fingerprints(internal_by_id[record["id"]])):
                    raise AssertionError("Persisted forward fingerprints differ from fresh hashing")
                validated += 1
    input_wait = extraction - sum(_PARENT_WALL.values())
    if input_wait < -0.01:
        raise AssertionError("Exclusive extraction accounting is inconsistent")
    result = {
        "hash_method": method, "diagnostic": diagnostic, "directory": str(directory),
        "nb_workers_argument": config["nb_workers"],
        "fingerprint_pool_workers": max(1, config["nb_workers"] - 1),
        "start_method": "fork", "jit_warmup_seconds_excluded": warmup,
        "phases": {
            "extraction": extraction, "merge_inverted": merge, "forward_index": forward,
        },
        "extraction_main_process_breakdown": {
            **_PARENT_WALL, "input_hash_ipc_wait_and_orchestration_seconds": max(0.0, input_wait),
        },
        "total_build_seconds": elapsed,
        "timing_note": "Build time includes the three production phases, not input sampling, "
                       "JIT warmup, post-build validation or CPU-profile analysis. "
                       "Subphase timers add two clock reads per document insertion. "
                       "Waiting/orchestration is not pure hashing time.",
        "counts": counts, "validated_forward_documents": validated,
        "intermediate_shards": len(shards),
        "index_bytes": sum(path.stat().st_size for path in (directory / "index").iterdir()
                           if path.is_file()),
    }
    (directory / "build_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def analyze_profiles(directory):
    files = sorted((directory / "profiles").glob("*.prof"))
    for role in ("extract-parent", "hash-worker", "input-producer", "merge-parent",
                 "merge-reader", "forward-parent"):
        if not any(path.name.startswith(role + "-") for path in files):
            raise AssertionError(f"Missing diagnostic profile for {role}")
    stats = DetailedStats()
    stats.add(*(str(path) for path in files))
    total = sum(value[2] for value in stats.stats.values())
    functions = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        functions.append({
            "file": filename, "line": line, "function": name,
            "calls": calls, "primitive_calls": primitive,
            "exclusive_cpu_seconds": own, "inclusive_cpu_seconds": cumulative,
            "exclusive_cpu_percent": 100 * own / total if total else 0.0,
        })
    functions.sort(key=lambda row: row["exclusive_cpu_seconds"], reverse=True)
    roles = {}
    for path in files:
        role = path.stem.rsplit("-", 1)[0]
        profile = DetailedStats(str(path))
        roles[role] = roles.get(role, 0.0) + sum(row[2] for row in profile.stats.values())
    result = {
        "summed_profiled_process_cpu_seconds": total,
        "process_roles_cpu_seconds": roles, "functions": functions[:100],
        "note": "Separate cProfile diagnostic build, using per-process process_time clocks. "
                "Exclusive times are additive; inclusive times overlap. CPU seconds across "
                "processes are not wall time, and instrumentation changes absolute timings. "
                "Use the non-cProfile builds for speedup comparisons.",
    }
    (directory / "cpu_hotspots.json").write_text(json.dumps(result, indent=2))
    with (directory / "cpu_hotspots.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(functions[0]))
        writer.writeheader()
        writer.writerows(functions)
    return result


def write_report(report, output_dir):
    rows = []
    builds = report.get("timing_summary", report["timed_builds"])
    for method, result in builds.items():
        total = result["total_build_seconds"]
        for phase, seconds in result["phases"].items():
            rows.append({"hash_method": method, "phase": phase, "seconds": seconds,
                         "percent_of_build_wall_time": seconds / total * 100})
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    with (output_dir / "wall_time_phases.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = go.Figure()
    for phase in ("extraction", "merge_inverted", "forward_index"):
        fig.add_bar(name=phase, x=list(builds),
                    y=[builds[method]["phases"][phase]
                       for method in builds])
    fig.update_layout(barmode="stack",
                      title=f"{report['documents']:,}-document production index build: warm-JIT wall time",
                      yaxis_title="Seconds")
    with (output_dir / "build_profile.html").open("w", encoding="utf-8") as output:
        output.write("<!doctype html><html><head><meta charset='utf-8'>"
                     "<title>Index build profiling</title>"
                     "<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}"
                     "td,th{border:1px solid #ccc;padding:.5em}table{border-collapse:collapse}"
                     "</style></head><body><h1>Production index-build profiling</h1>")
        output.write(f"<p>{report['documents']:,} identical sampled Nynorsk documents. "
                     "Source preparation, JIT warm-up and post-build validation are excluded. "
                     "Normal corpus indexes are unchanged.</p>")
        output.write(fig.to_html(full_html=False, include_plotlyjs=True))
        output.write("<h2>Build wall time</h2><table><tr><th>Method</th><th>Phase</th>"
                     "<th>Seconds</th><th>Percent of build</th></tr>")
        for row in rows:
            output.write(
                f"<tr><td>{html.escape(row['hash_method'])}</td>"
                f"<td>{html.escape(row['phase'])}</td><td>{row['seconds']:.2f}</td>"
                f"<td>{row['percent_of_build_wall_time']:.2f}%</td></tr>"
            )
        output.write("</table><h2>Extraction: main-process wall-time breakdown</h2>"
                     "<table><tr><th>Method</th><th>Activity</th><th>Seconds</th>"
                     "<th>Percent of whole build</th></tr>")
        for method, result in builds.items():
            for activity, seconds in result.get("extraction_main_process_breakdown", {}).items():
                output.write(
                    f"<tr><td>{html.escape(method)}</td><td>{html.escape(activity)}</td>"
                    f"<td>{seconds:.2f}</td>"
                    f"<td>{100 * seconds / result['total_build_seconds']:.2f}%</td></tr>"
                )
        output.write("</table>")
        output.write("<p>Timing builds are separate from cProfile diagnostic builds. "
                     "CPU hotspot percentages use summed profiled process CPU time, "
                     "not end-to-end wall time. Profiling adds overhead, especially "
                     "for frequent short calls.</p>")
        for method, profile in report.get("cpu_profiles", {}).items():
            output.write(
                f"<h2>{html.escape(method)}: diagnostic CPU hotspots</h2><table>"
                "<tr><th>Function</th><th>Exclusive CPU seconds</th>"
                "<th>Percent of profiled CPU</th></tr>"
            )
            for row in profile["functions"][:12]:
                name = f"{Path(row['file']).name}:{row['line']} {row['function']}"
                output.write(
                    f"<tr><td>{html.escape(name)}</td>"
                    f"<td>{row['exclusive_cpu_seconds']:.2f}</td>"
                    f"<td>{row['exclusive_cpu_percent']:.2f}%</td></tr>"
                )
            output.write("</table>")
        output.write("<details><summary>Complete measurements and limitations</summary><pre>")
        output.write(html.escape(json.dumps(report, indent=2)))
        output.write("</pre></details></body></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--documents", type=int, default=50_000)
    parser.add_argument("--nb-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "nno_Latn")
    parser.add_argument("--run-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_config:
        execute_build(json.loads(args.run_config.read_text()))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if min(args.documents, args.nb_workers) < 1:
        parser.error("Documents and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    documents, populations = sample_documents(
        sorted(args.corpus_dir.glob("*.jsonl.zst")), args.documents, args.seed, 100, 100_000,
    )
    source = args.output_dir / "sample.jsonl.zst"
    digest = hashlib.sha256()
    with source.open("wb") as raw, zstd.ZstdCompressor(level=3).stream_writer(raw) as output:
        for row in documents:
            encoded = orjson.dumps({"id": row["id"], "text": row["text"]}) + b"\n"
            digest.update(encoded)
            output.write(encoded)
    (args.output_dir / "sample_ids.json").write_text(json.dumps([row["id"] for row in documents]))
    preparation = perf_counter() - started
    del documents
    report = {
        "documents": args.documents, "sample_seed": args.seed, "sample_populations": populations,
        "normalized_input_sha256": digest.hexdigest(),
        "source_preparation_seconds_excluded": preparation,
        "timed_builds": {}, "diagnostic_builds": {}, "cpu_profiles": {},
        "limitations": [
            "One timed rolling64 build; cache and shared-server effects are not eliminated.",
            "Input contains sampled id/text records, not all original HPLT metadata fields.",
            "One intermediate shard at 50k documents; larger multi-shard merges may behave differently.",
            "JIT compilation is warmed and reported separately.",
            "CPU profiling is a separate instrumented run, not the speedup measurement.",
        ],
    }
    for diagnostic in (False, True):
        for method in (ROLLING64_HASH,):
            label = "diagnostic" if diagnostic else "timed"
            directory = args.output_dir / f"{label}_{method}"
            config = {"source": str(source.resolve()), "directory": str(directory.resolve()),
                      "hash_method": method, "diagnostic": diagnostic,
                      "documents": args.documents, "nb_workers": args.nb_workers}
            config_path = args.output_dir / f"{label}_{method}_config.json"
            config_path.write_text(json.dumps(config, indent=2))
            print(f"Starting {label} build: {method}", flush=True)
            with (args.output_dir / f"{label}_{method}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, "-m", "experiments.profile_index_build",
                     "--run-config", str(config_path)],
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )
            result = json.loads((directory / "build_result.json").read_text())
            report["diagnostic_builds" if diagnostic else "timed_builds"][method] = result
            if diagnostic:
                report["cpu_profiles"][method] = analyze_profiles(directory)
            if report["timed_builds"]:
                write_report(report, args.output_dir)
            print(f"Finished {label} {method}: {result['total_build_seconds']:.2f}s", flush=True)
    print(f"Saved index build profiling to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
