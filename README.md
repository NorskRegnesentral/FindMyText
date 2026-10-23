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
git clone https://github.com/your-org/FindMyText.git
cd FindMyText
pip install -e .
```

---

## Quick start

### 1. Build the index

The first step is to index your corpus. Fingerprints are extracted in parallel and then merged into a single disk-based index:

```python
from findmytext import index_builder

# Write corpus records as JSONL objects with "text" and "id" fields first.
index_builder.index_file("corpus.jsonl", "my_fingerprints", nb_workers=4)
index_builder.merge_indexes_from_dir("my_fingerprints", "my_index")
```

The resulting index is stored on disk and memory-mapped at query time. Besides
the inverted index (fingerprint → postings), the merge writes a forward index
(`forward_offsets.npy`, `forward_fingerprints.npy`: each document's sorted unique
fingerprints). For an index merged without it (`build_forward=False`), run
`index_builder.build_forward_index("my_index")`.

#### Rolling64 fingerprints and rebuilding old indexes

All builds and queries now use the versioned non-cryptographic `rolling64-v1`
method. Legacy31 hashing has been removed. Create a **new** index from source:

```python
index_builder.index_file(
    "corpus.jsonl", "my_fingerprints64", nb_workers=4,
)
index_builder.merge_indexes_from_dir("my_fingerprints64", "my_index64")
```

No hash selection flag is needed. For standalone fingerprinting, use
`Winnower(length=4, window_size=6)`.
Index metadata records the hash method and base. Readers and detectors use
these settings automatically. Missing/legacy/unknown hash versions are errors,
as are incompatible shard settings. Existing corpus indexes and old
intermediate shards must be rebuilt; they are not converted automatically.
Do not mix old/new hashes or change an existing index's metadata to "upgrade"
it: changing hashes requires rebuilding from source text.

Rolling64 uses compiled unreduced token hashing, odd-base unsigned arithmetic
modulo `2**64`, and a bijective fmix64 output finalizer before winnowing. Its
fingerprints and hash DataFrame columns preserve the full UInt64 range, including
values above `2**63`. It is not intended for cryptographic/adversarial integrity
checks, and collision/quality measurements remain corpus-specific.
The full findings, historical comparisons, and migration decision are in
[experiments/HASHING_FINDINGS.md](experiments/HASHING_FINDINGS.md).

#### Document similarities (separate step)

Document-document similarities are computed *after* the index is built, so
several methods can be run and compared on the same index. Each method writes
`similarities_<method>.npy` into the index directory: a structured NumPy array
with `doc_i < doc_j` (internal integer IDs) and `count` (number of unique shared
fingerprints), loaded with `DiskBasedIndex.load_similarities(method)`.

| Method | Description |
| --- | --- |
| `inverted` | Exhaustive. One sequential pass over the inverted index; every fingerprint adds one count to each pair of documents in its posting list. |
| `forward_topk` | For each document, its forward-index fingerprints are queried against the inverted index and its `top_k` neighbours sharing at least `min_shared_fingerprints` fingerprints are kept (union over documents; exact counts). |

```python
from findmytext import similarity

similarity.compute_similarities_inverted("my_index", min_shared_fingerprints=1)
similarity.compute_similarities_forward_topk(
    "my_index", top_k=50, min_shared_fingerprints=5
)
```

For large exhaustive builds, `max_pair_updates_in_memory` bounds the pair-count
updates buffered before a sorted chunk is flushed to disk, `max_chunks_per_merge`
bounds the chunks opened in one merge pass, and
`max_fingerprint_document_frequency` skips very common fingerprints (making
counts approximate). `posting_batch_entries` controls the posting entries read
per batch.

The `index_builder` can also be used directly from the command line:
```bash
# Step 1: extract fingerprints from a corpus file into intermediate shards
python -m findmytext.index_builder index corpus.jsonl my_fingerprints --nb_workers 4

# Step 2: merge shards into a final disk-based index (incl. forward index)
python -m findmytext.index_builder merge my_fingerprints my_index

# Step 3 (optional): compute document similarities with one or more methods
python -m findmytext.similarity my_index --method inverted
python -m findmytext.similarity my_index --method forward_topk --top-k 50 --min-shared-fingerprints 5
```

To compare the runtime of both methods and check that `forward_topk` equals the
exhaustive result truncated to each document's top-k neighbours:

```bash
python -m experiments.compare_similarity_methods --index-dir my_index --top-k 50 --min-shared-fingerprints 5
```


#### Inspect existing indexes

Print index statistics and save interactive figures for the four HPLT corpora
(Nynorsk, Bokmal, Icelandic, and Latvian) in a parent folder:

```bash
.venv/bin/python -m experiments.index_stats /path/to/corpora --output-dir /path/to/analysis
```

With no arguments, it analyzes the four indexes at
`/nr/project/stat/COPY.AI/data/Lars/hplt_<language>/findmytext_build/index`
(`nynorsk`, `bokmal`, `icelandic`, and `latvian`) and saves reports in
`/nr/project/stat/COPY.AI/data/Lars/index_analysis`:

```bash
.venv/bin/python -m experiments.index_stats
```

Explicit index arguments and `--output-dir` override these defaults.
The command discovers merged indexes in each immediate child folder, including
the `hplt_<language>/findmytext_build/index` layout. It also accepts a single
index/corpus folder, or explicitly named indexes:

```bash
.venv/bin/python -m experiments.index_stats \
  nynorsk=/path/to/hplt_nynorsk/findmytext_build/index \
  bokmal=/path/to/hplt_bokmal/findmytext_build/index \
  icelandic=/path/to/hplt_icelandic/findmytext_build/index \
  latvian=/path/to/hplt_latvian/findmytext_build/index \
  --output-dir /path/to/analysis
```

Install the visualization extra (`pip install -e '.[viz]'`) for Plotly.
The report includes document and distinct fingerprint counts, total positional
postings, distinct document/fingerprint pairs, fingerprints confined to one
document or shared by multiple documents, within-document repetitions, disk
sizes, and distribution minima, means, quantiles, and maxima. A singleton means
**one distinct document**, even if the fingerprint occurs repeatedly there.
Document counts refer to indexed documents, not discarded source records.

Results are printed and saved as `index_stats.json`, `index_stats.csv`, and one
offline-viewable `index_stats_<name>.html` per corpus, containing distributions
of fingerprints per document, documents per fingerprint, and posting-list lengths.
Plots use logarithmic axes for positive counts, with quantile and maximum
markers at their actual count values. A distribution containing zero uses a
linear x-axis so zeros remain visible.
Results for previously analyzed corpus names are retained in the same output
folder; rerunning a name replaces its results.

An additional `index_stats_comparison.html` shows distinct fingerprints on the
x-axis and singleton fingerprints on the y-axis, with one labeled point per
corpus. Color indicates singleton fraction; hover details include document
counts, posting counts, document/fingerprint pairs, index size, and index path.
A second plot in the same HTML report shows distinct fingerprints on the x-axis
and singleton fraction on the y-axis, displayed as a percentage from 0% to 100%.
Both plots include the same corpus metadata on hover.
These are comparisons of different corpora, not a growth curve for one corpus.

To fix previously generated histogram axes and produce this comparison without
rescanning any index:

```bash
.venv/bin/python -m experiments.index_stats --repair-plots
# Or: --repair-plots --output-dir /path/to/analysis
```

This reads the saved JSON and HTML reports, preserves the embedded histogram
data, and updates only each chart's layout. Original HTML is retained as
`index_stats_<name>.html.bak`; repeated repairs keep the original backup.

Fingerprint counts measure distinct **hash values**, not necessarily distinct
text k-grams. The retired legacy method used modulus `2**31 - 1` (about 2.147
billion values), despite storing fingerprints in `uint64` arrays. Hash collisions
can merge unrelated k-grams, increasing apparent sharing across documents.
Winnowing selects window minima, so selected hashes are not uniformly distributed.
A lower singleton fraction in a larger corpus can reflect collisions, genuine
shared text, or both; these aggregate reports cannot separate the causes.

##### Frozen collision experiments and current throughput benchmark

Legacy31-dependent experiment execution is retired. The completed Nynorsk
and Latvian comparisons are preserved and explained in
[experiments/HASHING_FINDINGS.md](experiments/HASHING_FINDINGS.md), including
their output locations. Existing reports can be rendered without loading old
indexes or implementing legacy hashing:

```bash
.venv/bin/python -m experiments.hash_collision_study \
  --report-dir /path/to/existing/completed_study
```

The original JSON, CSV, exact-variant Parquet data and occurrence files remain
unchanged by migration. Prefer Parquet for full-width numeric hashes; CSV
readers must parse `hash` and `signature64` as unsigned 64-bit integers, not
signed integers or floats. Generic regrouping helpers remain available for
saved occurrence data; new legacy corpus collection and verification are not.

The throughput benchmark now compares production rolling64 with an independent
BLAKE2b-64 reference, not with a retained legacy implementation:

```bash
.venv/bin/python -m experiments.benchmark_hash_width --documents 5000 --repeats 5 \
  --output /path/to/new/hash_width_speed.json
```

Timings include tokenization and UTF-8 preparation through winnowing, exclude
decompression/index I/O and JIT warm-up, and rotate method order across repeats.
The benchmark calls the same rolling implementation used by the index builder
and detectors.
This does not certify a collision rate for unseen data: collision quality and
changes in selected positions must be measured on representative corpora.
Merely widening the old modulus or storing old reduced hashes in uint64 does
not implement this design.

##### Profile complete index builds

Profile a rolling64 build on a seeded 50,000-document Nynorsk sample, including
extraction, inverted-index merging and forward-index creation. The historical
legacy/rolling64 comparison is frozen; legacy build configurations are rejected.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m experiments.profile_index_build \
  --documents 50000 --nb-workers 4 --output-dir /path/to/new/build_profile
```

This creates one phase-timed rolling64 index and one separate CPU-profiled
diagnostic index. The output directory must be new; source indexes are not
changed. `--nb-workers 4` gives three fingerprint-pool workers plus the producer,
alongside the main indexing process. The runner uses fresh subprocesses, warms
JIT compilation before timing, and explicitly uses Linux `fork`.

`report.json`, `wall_time_phases.csv` and `build_profile.html` contain wall-time
phases in seconds and percentages of total build time. Extraction also reports
main-process insertion, sorting/export, and the remaining input/hash/IPC
waiting and orchestration time; that last remainder is not pure hashing time.
Diagnostic directories contain per-process cProfile files and CPU hotspot
JSON/CSV tables. Exclusive CPU times are additive across processes; inclusive
times overlap, and summed CPU time is not wall time. Instrumentation changes
timings, so use the non-cProfile build for performance comparisons.

Input preparation is reported separately and excluded from build timings.
The controlled compressed input contains sampled IDs and text, not all original
HPLT metadata. A single timing run, cache effects, and the
single intermediate shard produced by this sample limit generalization to
whole-corpus multi-shard builds. Completed indexes are retained for inspection.

The index-statistics command (unlike the collision study) makes one sequential
pass over postings; no corpus
text or similarity computation is needed. Indexes without a forward index are
supported. `--batch-entries` (default: 1,000,000) controls the target number of
postings processed per batch, not a strict RAM limit: a single fingerprint's
posting list is never split, and index-wide arrays and frequency tables also
use memory.

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

### 4. Analyze passages shared by several documents

For reproducible experiments, `generate_controlled_corpus` creates overlapping
source intervals and independently sampled common hub fingerprints:

```python
from findmytext.synthetic import generate_controlled_corpus

corpus = generate_controlled_corpus(
    n_documents=50,
    n_hub_fingerprints=6,
    hub_inclusion_probability=[0.95, 0.9, 0.8, 0.7, 0.5, 0.3],
    source_intervals={
        "copy-a": (0, 900),
        "copy-b": (600, 1500),
        "copy-c": (1200, 2100),
    },
    source_token_count=2100,
    seed=7,
)
```

`corpus.documents` contains JSONL-style `id`/`text` records.
`corpus.ground_truth` contains source intervals, token and character spans,
pairwise source-overlap lengths, exact shared-winnowed-fingerprint counts, and
both intended and realized hub memberships. Hub k-grams use verified
document-specific carriers so every intended hub hash survives winnowing without
sharing its surrounding guard text.

`MultiDocumentFingerprintChainAnalyzer` winnows an explicit set of documents,
finds chains shared by all selected documents, and then classifies pairwise chains
as `novel`, `partial_overlap`, or `subsumed` by an all-document passage:

```python
from findmytext.multi_document import MultiDocumentFingerprintChainAnalyzer

analyzer = MultiDocumentFingerprintChainAnalyzer(
    coordinate_representation="centered_offsets",
    position_threshold=30,
    offset_threshold=30,
)
analysis = analyzer.analyze(
    {"doc-a": text_a, "doc-b": text_b, "doc-c": text_c},
    pairwise_mode="novel",
)
```

The analyzer supports any $N \geq 2$ for clustering. Its interactive Plotly view
is specifically three-dimensional:

```python
from findmytext.multi_document import plot_three_document_fingerprints

figure = plot_three_document_fingerprints(analysis)
figure.show()
```

#### Coordinate representations

Raw positions are always retained in the result and used for plotting. The
`coordinate_representation` option only controls the geometry used by linkage:

| Value | Coordinates | Advantages | Limitations |
| --- | --- | --- | --- |
| `raw` | $(p_1,\ldots,p_N)$ | Simplest representation; $O(N)$ construction; directly matches the plot | Ordinary distance mixes progress along the shared passage with deviation from its diagonal |
| `anchor_offsets` | $(p_1,p_2-p_1,\ldots,p_N-p_1)$ | $O(N)$; cheapest transformed form; extends the pairwise detector directly; offsets are easy to interpret | Numerically privileges the first document as an anchor; changing document order can change threshold-boundary decisions |
| `centered_offsets` | $(\bar p,p_1-\bar p,\ldots,p_N-\bar p)$ | $O(N)$; symmetric under document reordering; separates diagonal progress from disagreement between documents | Stores one redundant offset because centered offsets sum to zero; slightly more arithmetic than anchor offsets |

`centered_offsets` is the default when document order should not matter.
`anchor_offsets` is useful when the first document is intentionally a source or
reference. `raw` is appropriate when a single isotropic positional distance is
the desired model. Position and offset coordinates are divided by
`position_threshold` and `offset_threshold` before linkage, respectively.

Hashes with too many positional combinations can otherwise create a Cartesian
product across documents. `max_position_combinations_per_hash` bounds that work;
skipped hashes are reported in `analysis.skipped_hashes`.

To stress the complete merge path with many intermediate indexes, two-pair
similarity chunks, and fan-in-two compaction, run:

```bash
python -m experiments.verify_multifile_similarity
```

The experiment independently re-winnows every generated document and requires
the complete external-ID similarity map to equal `similarities_inverted.npy`, and
every forward-index entry to equal the document's winnowed fingerprint set.

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
