<p align="center">
  <img src="assets/findmytext_logo.png" alt="FindMyText logo" width="320"/>
</p>

# **FindMyText**

**FindMyText** is an open-source Python package for efficiently detecting whether a given text appears, in part or in full, within a large text corpus. It is particularly suited for verifying the presence of copyrighted or licensed material in large, web-crawled corpora. This can notably provide important insights into which texts have been used to pre-train LLMs. 

The tool builds on standard document fingerprinting techniques, and extends them with a novel mechanism that explicitly captures *sequences* (chains) of matching fingerprints. This makes it robust to near-verbatim copies — texts that share the same content but with minor differences due to OCR errors, formatting variants, text normalisation, or added boilerplate. Leveraging a distributed, disk-based indexing framework, FindMyText scales to large corpora that cannot be held in memory.

A full step-by-step walkthrough is available in [`demo.ipynb`](demo.ipynb).

---

## Installation

```bash
git clone https://github.com/NorskRegnesentral/FindMyText.git
cd FindMyText
pip install -e .
```

For the notebook and alignment visualizations, install `pip install -e ".[demo]"`.
For tests, install `pip install -e ".[test]"` and run `python -m pytest`.

---

## Quick start

### 1. Build the index

The first step is to index your corpus. Fingerprints are extracted in parallel and then merged into a single disk-based index:

```python
from findmytext import index_builder

# corpus.jsonl contains one object per line, with "text" and a unique "id".
# Put parallel Python builds in a script guarded by if __name__ == "__main__".
files = index_builder.index_file("corpus.jsonl", "my_fingerprints", nb_workers=1)
index_builder.merge_indexes_from_dir("my_fingerprints", "my_index")
```

The resulting index is stored on disk and memory-mapped at query time, so it scales to corpora that are too large to fit in RAM.
This publication path builds only the positional inverted index: it does not
compute document-document similarities or add a forward/overlap index.

The `index_builder` can also be used directly from the command line:
```bash
# Step 1: extract fingerprints from a corpus file into intermediate shards
python -m findmytext.index_builder index corpus.jsonl my_fingerprints --nb_workers 4

# Step 2: merge shards into a final disk-based index
python -m findmytext.index_builder merge my_fingerprints my_index
```

Both the API and CLI default to 4-token k-grams and a window size of 6.
Input can be `.jsonl`, `.jsonl.gz`, or `.jsonl.zst`; compressed readers need no
separate preprocessing step. Existing length/content filters are unchanged:
minimum 100 characters; maximum 100,000 for JSONL/GZip and 500,000 for Zstandard;
records containing the exact whitespace-separated tokens `sex` or `porn` are
skipped. Pass `max_nb_documents_before_flush` to bound the number of documents
accumulated per intermediate shard (default 1,000,000).

`nb_workers=1` fingerprints directly in the parent. Larger values use
`nb_workers - 1` fingerprint workers and keep a bounded number of pending
batches; the corpus generator remains in the parent, including with Python
3.14's `forkserver` default. Use `nb_workers=1` in a notebook or the CLI for
parallel builds. Merging uses a reader per shard, or reads directly when there
is just one shard. Each input shard directory should contain only the intended
build's `.mpk.gz` files; use a fresh destination for a new build.

### Hashing and existing indexes

The default fingerprint implementation is **`polynomial64-fmix64`**:
compiled UTF-8 token hashing using the inherited FNV-1a-style recurrence,
an odd-base polynomial rolling hash modulo `2**64`, then MurmurHash3's
`fmix64` output finalizer before winnowing. The inherited token-hash offset
basis is `1469598103934665603`, not the standard FNV-1a-64 offset basis.
This composition is non-cryptographic, not a standardized hash with a collision
guarantee. Rightmost-minimum tie handling and tokenization are unchanged.

Its default base is `0x9E3779B185EBCA87`. Fingerprints preserve all unsigned
64 bits through shard serialization, index storage, detector scores and the
web adapter. Metadata records the algorithm and parameters, and merging
rejects incompatible shard settings before writing the output.

For paper-era reproduction, explicitly choose **`legacy31`**. This restores the
original token hashing and signed rolling kernel modulo `2**31 - 1`, including
the default base **256**, from
[`8d032a8`](https://github.com/NorskRegnesentral/FindMyText/commit/8d032a8b30d515925e98f288d3dae4893ee1f41b).
It is a compatibility option, not the recommended method for new indexes:
its much smaller effective hash space retains the original collision behavior.

| Method | Default? | Default base | Intended use |
|---|---|---|---|
| `polynomial64-fmix64` | Yes | `0x9E3779B185EBCA87` | New builds with full-width hashes |
| `legacy31` | No; explicit opt-in | `256` | Original-main/paper hash reproduction |

```bash
python -m findmytext.index_builder index corpus.jsonl paper_fingerprints \
  --hash-method legacy31 --length 4 --window_size 6 --nb_workers 1
python -m findmytext.index_builder merge paper_fingerprints paper_index
```

The equivalent API option is
`index_builder.index_file("corpus.jsonl", "paper_fingerprints", hash_method="legacy31")`.
Standalone fingerprinting also accepts
`Winnower(length=4, window_size=6, hash_method="legacy31")`.
Select the k-gram/window settings, exact input records, filters, seeds and scoring
parameters of the experiment being reproduced; selecting a hash method alone
does not recreate an entire evaluation.

**Readers automatically use the method recorded in the index.** Detectors and
the web adapter propagate it to their runtime winnower; no separate query-side
choice is needed. Old-main interleaved indexes and MsgPack shards without a
`hash_method` field are recognized as `legacy31`, so they can be used without
rebuilding. New builds explicitly label their method. Unknown method labels
are errors, and merging different methods or incompatible parameters is rejected.

Switching an existing index from legacy hashes to the new method still
**requires rebuilding from source**. Hashing changes which positions winnowing
selects, so scores and matches can differ between methods. Changing a metadata
label does not convert fingerprints.

The research branch's name `rolling64-v1` denotes the identical 64-bit algorithm;
readers accept that metadata spelling for compatibility, while new 64-bit builds
write `polynomial64-fmix64`. This is a metadata alias, not a third hash method.
Only interleaved `(doc_id:uint32, position:uint32)` postings are supported.

The restored legacy mode was compared with the actual original-main code on
100 Nynorsk documents: **1,600 fingerprint/position cases** across k-gram
lengths 1/4/5/7, bases 256/257 and both punctuation settings matched. All six
positional-index payload files were byte-identical; both detectors in both
rarity modes returned identical scores, including when reading the original
unlabelled index. The metadata label is intentionally new. The bounded check,
not a full-paper rerun, is recorded in
[`experiments/legacy_hash_compatibility.json`](experiments/legacy_hash_compatibility.json).

### Build-time comparison

Small same-machine comparison against the original `main` commit above:
**5,000 identical Nynorsk documents**, k=4/window=6, three timing runs per
implementation/configuration with alternating old/new order, warmed JIT,
and 2,000 documents per flush (three shards). Times are medians, in seconds.
These build measurements compare old main's legacy hashes with the new 64-bit
default at `1bcd60c`, before the separate posting-read default change and before
the legacy option was restored. They are not timings of legacy mode in the
updated builder.
The host was a shared Intel Xeon 6761P machine running Linux/Python 3.14.0,
NumPy 2.5.3, Numba 0.68.0 and optional ISA-L GZip.

| Measurement | Original main | Publication branch | Interpretation |
|---|---:|---:|---|
| Tokenization through winnowing only | 1.87 | 0.69 | About **2.7x faster**; excludes index insertion/I/O |
| Full positional build, `nb_workers=1` | 6.08 | 4.66 | About **1.31x faster** (23% less wall time) |
| Full positional build, `nb_workers=4` | 4.79 | 5.00 | No parallel-build speedup here; new run was 4.3% slower |

The full-build timers include extraction, sorting, compressed intermediate
export, document-ID mapping and positional merging. They exclude input sampling,
JIT warm-up and post-build verification. Every document and stored occurrence
was verified against that implementation's source fingerprinting after each
build. The hash migration changes the output identities: old/new indexes are
not expected to be byte-identical.

The original builder actually created a producer and a hashing child even for
`nb_workers=1`; the new single-worker path avoids both. For `nb_workers=4`,
the old path used three hashing children plus a producer and the parent; the
new path uses three hashing children and the parent. Both were explicitly
run with Linux `fork` for a controlled warmed-JIT comparison, since unmodified
main cannot pickle its corpus generator with Python 3.14's default start
method. Production is also tested with `spawn` and `forkserver`.

The 64-bit intermediate files were 9.65 MB versus 6.68 MB for old main;
the final positional indexes were both about 18.36 MB. Faster hashing alone
does not imply a comparable end-to-end speedup: compression, Python merging,
IPC and cache/shared-host effects still matter. These are small, warmed runs,
not full-corpus throughput estimates or an isolated proof of any one
optimization's contribution. No forward-index work is included.

The implementation ports compiled token hashing/64-bit arithmetic from the
research branch, adds bounded 1 MiB buffering before GZip compression, and
avoids unnecessary single-worker/single-shard processes. Other fixes repair
stale README/notebook and web-adapter APIs, sub-1,000-document flush-name
collisions, unsigned detector columns, missing MsgPack/Zstandard dependencies,
and failed/truncated merge readers that previously could hang or stop early.

Raw timings, output counts, input checksum and environment:
[`experiments/index_build_benchmark.json`](experiments/index_build_benchmark.json).
To repeat with a local corpus and an unmodified old checkout:

```bash
python experiments/benchmark_index_build.py \
  --baseline /path/to/old-main-checkout --corpus /path/to/corpus.jsonl.zst \
  --output /path/to/new-results --documents 5000 --repeats 3 --workers 4 \
  --start-method fork
```

Use a new output directory; rerun with `--workers 1` for the single-worker
comparison. The source corpus is not distributed with the benchmark.

### Posting-read comparison (not index building)

Posting-list retrieval now defaults to **one thread**, with no thread pool.
This restores the research branch's sequential-read default; all existing
detectors and the web adapter inherit it. Explicit parallel reads remain
available for other storage/workload combinations:

```python
from findmytext.indexing import DiskBasedIndex

index = DiskBasedIndex("my_index")  # sequential posting reads
parallel_index = DiskBasedIndex("my_index", posting_read_threads=8)
```

This setting is separate from the builder's `nb_workers`. It controls reads of
an **already-built index**, not corpus fingerprinting or shard merging.

On the existing full Nynorsk positional index (1,028,632 documents;
96,496,752 distinct hashes), the same **200 queries** were measured with one
and eight read threads. Queries use unique stored-window fingerprints from
the first 200 eligible records of the frozen 50,000-document sample, in a
seeded common order. Each configuration/path was warmed first; five measured
passes alternated thread-configuration order. Reader creation, corpus loading,
fingerprinting, warm-up and output verification are excluded.

| Work for all 200 queries, median seconds | Former 8-thread behavior | New sequential default | Speedup |
|---|---:|---:|---:|
| Fingerprint lookup, positional reads and array decoding | 2.744 | 0.204 | **13.4x** |
| Retrieval plus candidate counting and top-50 ranking | 3.907 | 0.594 | **6.6x** |

These queries requested 67,085 posting lists in total (including the same
list requested by different queries), representing 31.51 MB of positional
payload per pass. Many are small reads: thread scheduling and synchronization
can cost more than the I/O they overlap. The second row includes document-ID
deduplication/counting/ranking but **not passage clustering or text
fingerprinting**, so it is not complete detector latency.

All **4,000 timed query results** matched the sequential reference by output
digest; focused tests also compare arrays and candidate counts directly.
Index metadata/file identities were unchanged, and the benchmark never writes
to the source index. The test corpus is warmed and the host is shared: these
numbers are not cold-cache throughput, a guarantee on other storage, or a
13x improvement in building the index. NumPy/OpenMP thread environment limits
were set to one for both configurations.

Raw per-pass timings, query-volume statistics, result digests and source
identity are in
[`experiments/posting_read_benchmark.json`](experiments/posting_read_benchmark.json).
To repeat with matching corpus text and an existing positional index:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/benchmark_posting_reads.py \
  /path/to/index /path/to/corpus.jsonl.zst \
  --output /path/to/new-read-results.json --documents 200 --repeats 5
```

The script reads only the positional index and source text; it does not build
or require a forward index, compute document similarities, or rebuild data.


### 2. Detect content

Once the index is built, detection is a single call:

```python
from findmytext import detectors

detector = detectors.FingerprintChainDetector("my_index")
scores = detector.get_containment_scores(query_text)

# `scores` is a dict mapping document IDs to containment scores
best_match_id = max(scores, key=scores.get)
print(f"Best match: {best_match_id}  (score: {scores[best_match_id]:.3f})")
```

The text in `query_text` does not need to be exactly identical to the one found in the corpus. **FindMyText** is designed to be robust to small differences between the documents. 

### 3. Verifying a match with local alignment

To inspect a detected match in detail, use the built-in local alignment:

```python
from findmytext import oracle

alignment = oracle.align(query_text, corpus_document_text)
alignment.show()
```

---

## How it works

1. **Fingerprinting**: each document is tokenised and converted to a set of $k$-gram hashes using the *winnowing* algorithm, which selects a representative subset of hashes from each sliding window.
2. **Inverted index**: fingerprints are stored in a disk-based inverted index mapping each hash to the list of `(doc_id, position)` pairs where it was observed.
3. **Chain detection**: at query time, the fingerprints of the query are looked up in the index. Rather than simply counting matches, FindMyText clusters matching fingerprints by their relative positions to identify *chains* — contiguous sequences of matches — and uses the total length of those chains as the containment score.

---

## License

**FindMyText** is released under an MIT License.

The MIT License is a short and simple permissive license allowing both commercial and non-commercial use of the software. The only requirement is to preserve the copyright and license notices (see file License). Licensed works, modifications, and larger works may be distributed under different terms and without source code.

---

## Reference

If you use **FindMyText** in your research, please cite our paper:

```
@article{findmytext2026,
  authors = {Lars Henry Berge Olsen and Pierre Lison and Martin Jullum and Mark Anderson},
  title   = {FindMyText:  Robust, Scalable Detection of Text Containment in Large Web-Crawled Corpora},
  year    = {2026}
}
```
