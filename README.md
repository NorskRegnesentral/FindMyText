<p align="center">
  <img src="assets/findmytext_logo.png" alt="FindMyText logo" width="320"/>
</p>

# FindMyText

**FindMyText** detects verbatim and near-verbatim text containment in large
corpora. It indexes winnowed fingerprints and clusters matching positions to
identify shared passages, tolerating small edits and formatting differences.

See [demo.ipynb](demo.ipynb) for a walkthrough.

## Installation

```bash
git clone https://github.com/NorskRegnesentral/FindMyText.git
cd FindMyText
pip install -e .
```

For the notebook and alignment visualizations: `pip install -e ".[demo]"`.
For tests: `pip install -e ".[test]"`, then `python -m pytest`.

## Quick start

### 1. Build an index

Input records are JSON objects with `"text"` and a unique `"id"`, one per line.
JSONL, GZip (`.jsonl.gz`) and Zstandard (`.jsonl.zst`) are supported.

```python
from findmytext import index_builder

index_builder.index_file("corpus.jsonl", "my_fingerprints", nb_workers=1)
index_builder.merge_indexes_from_dir("my_fingerprints", "my_index")
```

Or use the CLI:

```bash
python -m findmytext.index_builder index corpus.jsonl my_fingerprints --nb_workers 4
python -m findmytext.index_builder merge my_fingerprints my_index
```

Defaults are **4-token k-grams** and **window size 6**. Documents shorter than
100 characters are skipped; the maximum is 100,000 characters for JSONL/GZip
and 500,000 for Zstandard. See [utils.py](findmytext/utils.py) for ingestion
filters.

Use fresh build directories. `max_nb_documents_before_flush` controls shard
size (default 1,000,000 documents). Use `nb_workers=1` in notebooks; parallel
Python scripts need an `if __name__ == "__main__":` guard.

### 2. Detect containment

```python
from findmytext import detectors

detector = detectors.FingerprintChainDetector("my_index")
scores = detector.get_containment_scores(query_text)
print(scores)  # {document_id: containment_score}
```

The disk-based index is memory-mapped, and posting reads are sequential by
default. `DiskBasedIndex(..., posting_read_threads=8)` enables parallel reads
when useful; this is separate from index-building workers.

### 3. Inspect a match

```python
from findmytext import oracle

alignment = oracle.align(query_text, corpus_document_text)
alignment.show()  # highlighted alignment in a notebook
```

## Hashing and rebuilding indexes

**`polynomial64-fmix64` is the only supported hashing method.** It uses full
unsigned 64-bit fingerprints, combining compiled UTF-8 token hashing, rolling arithmetic
modulo `2**64`, and the `fmix64` finalizer. It is non-cryptographic; the larger
hash space reduces collisions but does not guarantee their absence.

**Breaking change: rebuild existing corpus indexes and intermediate shards from
source.** Missing method metadata, `legacy31`, and the previous experimental
`rolling64-v1` label are rejected with a rebuild error. There is no hashing-method
argument or CLI selector. Do not edit metadata to make an old index appear new.

## Performance

Compiled hashing, buffered compression and fewer unnecessary processes improve
the build pipeline. Sequential posting reads avoid thread-pool overhead for
many small reads. [PR #3](https://github.com/NorskRegnesentral/FindMyText/pull/3)
preserves the measurement methodology and links to archived scripts and raw
results.

### Build speedup compared with original main

Rerun medians for the same **5,000 Nynorsk documents** (19.3 MB JSONL),
k=4/window=6. Original main uses legacy hashes; the updated implementation uses
the 64-bit default. API timings use five warmed-JIT runs with three shards,
excluding interpreter/import/JIT warm-up. CLI timings use three runs of separate
index and merge processes, including startup/imports/JIT, with default flushing
(one shard). Normal filesystem/JIT caches are retained.

| Measurement | Original main (s) | Updated implementation (s) | Speedup |
|---|---:|---:|---:|
| Tokenization through winnowing | 1.87 | **0.69** | **2.7x** |
| API extraction + merge, `nb_workers=1` | 6.16 | **4.63** | **1.33x** |
| API extraction + merge, `nb_workers=4` | 4.82 | 4.81 | ~1.00x |
| CLI index + merge, `nb_workers=1` | 11.57 | **7.91** | **1.46x** |
| CLI index + merge, `nb_workers=4` | 9.66 | **8.51** | **1.14x** |

These are complete builds of the **sample, not the full corpus**. Hashing is
about **2.7x faster** and the single-worker API pipeline **1.33x faster**;
four-worker API times are essentially unchanged. CLI/API differences are not
startup alone because shard counts differ. Outputs were verified after all
32 builds; speedups are not universal.

### Reading an existing index

Median seconds for **200 queries** against the full Nynorsk index, five warmed
runs per configuration:

| Operation | Eight threads | Sequential default | Speedup |
|---|---:|---:|---:|
| Positional posting retrieval | 2.744 | **0.204** | **13.4x** |
| Retrieval plus candidate counting/ranking | 3.907 | **0.594** | **6.6x** |

All timed results matched their references. These shared-host, warm-cache
measurements are not cold-cache guarantees. Query timings exclude fingerprinting
and passage clustering, and are **not index-build speedups**.

### Additional metrics: build time, index size and query latency

Example measurements on the **same fresh 5,000-document Nynorsk index**,
k=4/window=6, one build worker and sequential posting reads, on a shared
Intel Xeon 6761P/Linux host with Python 3.14:

| Metric | Measurement |
|---|---:|
| Index build time, median of three CLI builds | **8.01 s** |
| Final index size, excluding intermediate shards | **18.36 MB** (17.51 MiB) |
| Query latency, median / mean / p95 | **13.35 / 43.51 / 295.43 ms** |

Build timing includes interpreter startup, imports and JIT with normal existing
caches. Query timing measures the complete `FingerprintChainDetector.get_containment_scores`
call on a loaded, warmed index: fingerprinting, retrieval, clustering and scoring.
It excludes detector initialization and query preparation.

Queries are 50 seeded positive 2,000-character fragments, measured over three
passes using detector defaults. All 50 source documents were retrieved, all
timed scores were stable, and no resource-safety exclusions were needed.
These are **small-workload examples, not full-corpus paper results**; report
the corpus size, settings, hardware, query workload and cache state with results.

## License

[MIT License](LICENSE).

## Reference

If you use FindMyText in your research, please cite:

```bibtex
@article{findmytext2026,
  authors = {Lars Henry Berge Olsen and Pierre Lison and Martin Jullum and Mark Anderson},
  title   = {FindMyText: Robust, Scalable Detection of Text Containment in Large Web-Crawled Corpora},
  year    = {2026}
}
```
