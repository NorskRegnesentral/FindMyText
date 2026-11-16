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
[docs/HASHING_FINDINGS.md](docs/HASHING_FINDINGS.md).

#### Canonical interleaved positional index

The full inverted index always stores interleaved `(doc_id:uint32,
position:uint32)` records: 8 bytes per occurrence. The fingerprint values are
stored separately as uint64. The full forward index is unchanged: sorted
unique uint64 fingerprints per document, with no positions.

Columnar storage and the `--postings-layout`/`postings_layout` build options
have been removed. Columnar metadata is rejected with an explicit rebuild
message; changing metadata alone cannot convert posting bytes. Existing
interleaved rolling64 indexes remain usable.

The `df >= 2` document-only overlap inverted index is built separately, as described
below. Full indexes retain singletons for external queries.
The trade-offs and migration details are recorded in
[docs/POSTING_LAYOUT_FINDINGS.md](docs/POSTING_LAYOUT_FINDINGS.md).

#### Rebuild Nynorsk: positional inverted, forward, and overlap inverted

The local Nynorsk rebuild command uses the existing corpus shards under
`/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/nno_Latn` without downloading:

```bash
.venv/bin/python hplt_nynorsk.py --build --nb-workers 8
```

This writes a **new** `hplt_nynorsk/findmytext_build/index` with rolling64-v1
interleaved positional postings and the full hash-sorted forward index, then
derives `hplt_nynorsk/findmytext_build/overlap_inverted`. It preserves the existing
`findmytext_build_31_bit` directory and does not compute similarities or open
plots. Settings remain `length=4`, `window_size=6`, and source-text lengths
100 through 100,000 characters (inclusive), with the existing ingestion filters.
The full Nynorsk rebuild and verification completed on 23 October 2026:

| Representation | Entries | Stored size |
| --- | --- | --- |
| Positional inverted | 96,496,752 fingerprints; 159,523,198 occurrences | 3.024 GiB |
| Forward | 153,553,490 unique document/fingerprint entries | 1.152 GiB |
| Overlap inverted | 18,718,065 fingerprints; 75,774,803 document-ID entries | 582.5 MiB |

The build contains **1,028,632 documents**. The overlap inverted index omits
77,778,687 singleton fingerprints (80.60%) and collapses repeated occurrences;
the positional and forward indexes retain all singletons. All memberships,
ordering, coverage, accounting and byte counts passed verification. Exact
overlap counts matched for 256 seeded pairs; 20 source-text samples across all
five shards matched their forward sets and all 5,884 selected occurrence
positions. The local report is
`/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/findmytext_build/verification.json`.
These checks do not constitute an all-pairs join or passage-match validation.

The equivalent generic HPLT command, with an optional custom build destination:

```bash
.venv/bin/python hplt_build.py nno_Latn hplt_nynorsk \
  --skip-download --overlap-inverted --nb-workers 8 \
  --build-dir /path/to/new-build
```

Both commands default to flushing extraction every 100,000 documents
(`--max-documents-before-flush`). A build manifest binds the destination to
corpus paths, sizes, modification times, hash settings, and length filters.
Completed shards have explicit completion manifests, including shards that
flush exactly at the document limit and therefore have no `final.mpk.gz`.
Incomplete extraction shards are rebuilt on retry; a merged index is published
only once its positional and forward files are complete. Completed indexes
and matching sidecars are reused. Untracked/legacy directories, changed source
files/settings, missing completed-shard files, and stale sidecars produce
explicit errors: use a new build destination rather than relabeling artifacts.
Do not run concurrent rebuilds against the same build destination.

To derive only the overlap inverted index from an already rebuilt positional index:

```bash
.venv/bin/python -m findmytext.overlap_inverted /path/to/index /path/to/overlap_inverted \
  --batch-entries 1000000
```

The overlap inverted index must be outside the positional index, and an existing destination is
never overwritten. It contains:

| File | Contents |
| --- | --- |
| `fingerprints.npy` | UInt64 hashes ordered by `(document frequency, hash)`, both ascending |
| `offsets.npy` | UInt64 document-ID entry offsets, length `n_fingerprints + 1` |
| `doc_ids.npy` | Concatenated ascending unique UInt32 source document IDs; each list has `df >= 2` |
| `eligible_lengths.npy` | UInt64 counts of retained unique fingerprints per source document |
| `meta.json` | Format/order contract, source binding, settings, statistics, timings, and storage accounting |

Use the dedicated rank-based reader, **not** `DiskBasedIndex` or numeric-hash
`searchsorted` on the rarity-ordered hashes:

```python
from findmytext.overlap_inverted import OverlapInvertedIndex

overlap = OverlapInvertedIndex("/path/to/overlap_inverted", "/path/to/index")
documents = overlap.get_postings(0)  # first rarity rank, not fingerprint value
```

The single-worker Numba scanner handles unsorted/repeated positional occurrences
and retains every fingerprint found in at least two distinct documents,
including common fingerprints. Its working arrays scale with the scan batch
target plus document count, even when one posting list exceeds the target.
Sorted runs are merged on disk with two-way Numba merges; there is no global
in-memory sorting permutation. Memory-mapped inputs, temporary runs, and
outputs also consume OS-managed resident pages, so `--batch-entries` is not a
hard process-RSS limit.

Build progress reports scanning/de-duplication, frequency ranking, posting-list
layout, and completion. The printed/saved report includes singleton removal,
collapsed repeated occurrences, eligible-length and retained-frequency
distributions (nearest-rank quantiles), actual file/total bytes, phase timings,
peak temporary file bytes (including staging), and process-lifetime peak RSS
on Linux (including imports/Numba compilation, not an isolated build peak).
Source binding uses a resolved source path, file sizes/modification times,
and a metadata SHA256; it is a filesystem freshness check, not a content
checksum of the entire index. Source files must remain immutable during and
after the build.

The public module, reader, builder and CLI use `overlap_inverted`,
`OverlapInvertedIndex`, `build_overlap_inverted_index` and `--overlap-inverted`.
These replace the initial `screening` names; no compatibility aliases or
automatic relabeling of previously built sidecars are provided.

To verify the persisted indexes and save a JSON report:

```bash
.venv/bin/python -m experiments.indexing.verify_overlap_inverted \
  /path/to/build/index /path/to/build/overlap_inverted \
  --corpus-dir /path/to/nno_Latn \
  --report /path/to/build/verification.json
```

The verifier checks every positional/forward membership, every overlap inverted
membership, full numeric-hash and rare-first ordering, unique ascending
document-only postings, eligible lengths, singleton coverage, accounting, and
actual storage bytes. It compares exact full-forward and overlap-index counts
for 256 seeded document pairs, including co-occurring pairs. With `--corpus-dir`,
it also reservoir-samples four eligible documents per source shard, re-extracts
their fingerprints, and verifies their full forward sets and every selected
occurrence position against the positional index. Source indexes are read-only;
temporary document frequencies are disk-backed and cleaned up. This is not
an all-pairs similarity run or passage-match acceptance test.

No overlap threshold K is needed for the index build step. A bounded-subset
overlap-join prototype is described below; full-corpus joining, candidate
graphs, and positional passage verification remain future work.

#### Single-worker overlap-join prototype (Nynorsk first)

```bash
.venv/bin/python -m experiments.overlap.benchmark_overlap_join \
  /path/to/build/index /path/to/build/overlap_inverted \
  /path/to/new-experiment-output \
  --n-documents 10000 --seed 41 --k 50 30 20 10 5 \
  --candidate-df-cap 1000 --repetitions 3
```

This command is a **subset benchmark**, not a full-corpus or production
time-budgeted join. It uses Numba single-worker batched candidate processing
and a merge verifier against complete, unique, hash-sorted forward sets.
Token positions are unnecessary for set-overlap counts: they stay in the
positional inverted index for later passage evidence. A forward offset locates
an array slice, not a source-text position.

Documents are sampled uniformly without replacement; original source IDs are
preserved in outputs. The independent reference groups full-forward hashes and
accumulates exact pair counts without relying on the derived overlap postings.
The benchmark compares rare-first fingerprint traversal with smaller-eligible-
set-first document probing. Both use the same subset and rarity order. No
frequent fingerprints are discarded in the uncapped exact experiment.
Eligible lengths are recomputed within the selected subset: fingerprints found
in no other selected document cannot contribute to any subset pair. Rarity and
the candidate cap still use full-corpus df, not this smaller subset frequency.

The additional `df <= 1000` experiment uses **full-corpus** document frequency
to restrict candidate discovery. Discovered pairs are still verified against
complete forward sets, including above-cap hashes. Consequently, accepted
counts remain exact but qualifying pairs sharing only above-cap hashes can be
missing. Reported recall is against the selected subset's uncapped reference;
it is not a full-corpus recall or a passage-match precision estimate.

Output modes have deliberately different schemas:

- `exact_counts`: `doc_i:uint32, doc_j:uint32, count:uint32`.
- `threshold_only`: `doc_i:uint32, doc_j:uint32`, with no claimed exact counts.

Within each candidate mode and K, both schedulers and output modes must emit
the same canonical pair set. Every uncapped result must match the independent
reference; every capped accepted count must also match that reference.
Per-run checks suppress duplicate verification of both accepted and rejected
pairs. Document ownership follows `(eligible set size, source document ID)`.

The command saves `document_ids.npy`, independent/reference and reusable count
arrays, per-configuration result arrays, and `report.json`. It refuses an
existing output directory and never writes into source index directories.
The report includes candidate visits, duplicate discoveries, length rejections,
verifier calls, hash comparisons, state/output bytes, medians and all timing
samples, missing reference pairs, and source/subset/settings identities.

Working-array memory now defaults to **32 GiB**, with an explicit available-RAM
check, or can be set with `--memory-gib`. Historical subset benchmarks below
used the earlier 8 GiB allowance; their recorded settings are unchanged. Allocations and bounded
output buffers are checked against that budget, failing explicitly rather than
evicting exact pair history. This is not a hard process-RSS limit: imported
libraries, compilation, and OS-managed mapped pages are reported separately.
The fingerprint scheduler uses a dense triangular **byte-per-pair** checked
table only for the bounded subset. The document scheduler uses per-probe
UInt32 epoch markers. Neither the dense independent-reference counters nor the
global checked table is proposed as a full-corpus backend. Subset outputs are
buffered within the budget; production streaming/checkpoint recovery is not
implemented.

The 25 October Nynorsk experiment (`seed=41`, 10,000 documents) passed all 40
configurations, each repeated three times. Prepared arrays occupied 13.85 MiB;
global checked flags occupied 47.68 MiB versus 40,000 bytes for document markers.
Run timings measure the prepared in-RAM subset, not full-corpus random I/O:

| K | Exact qualifying pairs | Pairs with candidate df cap 1000 | Subset recall |
| --- | ---: | ---: | ---: |
| 50 | 1,464 | 1,437 | 98.16% |
| 30 | 2,391 | 2,318 | 96.95% |
| 20 | 3,258 | 3,079 | 94.51% |
| 10 | 5,713 | 4,441 | 77.73% |
| 5 | 8,946 | 5,828 | 65.15% |

The independent reference found 232,687 pairs with any overlap, from 505,937
pair contributions. Computing every discovered pair's exact count once took
about 0.97 s; subsequent K filters took about 0.5 ms each (including equality
checks). Independently run exact-count joins took about 0.20-0.80 s per K.
Thus high K can save verifier work in a fixed-K run, while retaining exact
counts can avoid repeated work across a sweep. Neither observation proves
full-corpus performance.

Exact counts are K-independent, but their cache is valid only for the same
source, document subset, fingerprint semantics, and discovery coverage.
A mere rejection at K=50 cannot be reused as a rejection at K=20. Production
anytime operation should commit bounded batches, record coverage and cap,
and mark deadline-limited output non-exhaustive. Automatically reducing K or
adding a wall-clock deadline is intentionally deferred; the current benchmark
has **no time limit**.

The local measured report is
`/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/overlap_experiments/2026-10-25-seed41-10000-final/report.json`.
The full contract and remaining work are in
[docs/SIMILARITY_JOIN_PLAN.md](docs/SIMILARITY_JOIN_PLAN.md).

#### Streaming joins, checkpoints, and larger Nynorsk samples

The scalable path uses document-owned probing, writes bounded result parts to
disk, and uses independent **external pair-count aggregation**, rather than
allocating an all-pairs reference matrix:

```bash
.venv/bin/python -m experiments.overlap.scale_overlap_join \
  /path/to/build/index /path/to/build/overlap_inverted \
  /path/to/experiment-output \
  --sizes 25000 50000 100000 --k 50 30 20 10 5 \
  --candidate-df-cap 1000 --memory-gib 32
```

Samples are nested uniform subsets from one seeded draw. Before candidate
enumeration, each K run removes ineligible documents from posting lists and
skips their probes. Eligible lengths remain based on all subset-shared hashes;
the cap affects discovery only, and the verifier reads complete source forward
sets. This length bound is exact and does not assume Jaccard-style size ratios.

The main streamed runs access the **full source forward memmaps**, not just
copied in-RAM subset arrays. Reports include page faults, process I/O counters,
owned-buffer estimates, process-lifetime peak RSS, output bytes, and end-to-end
configuration times including setup and durable checkpoint/output writes.
Per-run join timing excludes independent-reference construction, validation,
and contract/checksum checks before the join starts; the suite report also
records overall session elapsed time. It is not an isolated verifier timing.
These are single observations on a shared host with naturally warm OS/NFS
caches; first signature compilation may be included. No caches are flushed
and no cold-disk throughput claim is made.

Each run has a locked `checkpoint.json` binding source settings and file
identity, subset-array identity, K, output mode, cap, ownership, and batch
geometry. Completed probe ranges and checksummed output parts are committed
atomically with file/directory synchronization. Resume validates settings and
committed parts, discards only uncommitted run-owned parts, and retries an
unfinished probe range. Epoch markers are transient—not checkpoints.
Failure-free execution verifies a discovered pair at most once; interrupted,
uncommitted work can be recomputed. Partial runs are not exhaustive results.
Rerun the same command to resume matching checkpoints.

The smallest stage additionally measures late versus early length filtering
and controlled fingerprint-driven versus document-driven runs on copied
subset arrays. The former global checked table is kept only as a small-sample
comparison. Main-stream versus control timings are not isolated kernel
comparisons because streamed runs include checkpoint/fsync costs.

For a known K sweep, exact accepted counts at its **lowest K** are sufficient;
there is no need to retain every below-threshold candidate count:

```python
from findmytext.overlap_stream import filter_count_cache

filter_count_cache(
    "/path/to/100000/k5_exact_counts_dfall",
    "/path/to/new-count-cache-output",
    (50, 30, 20, 10, 5),
)
```

This is valid only for the same source/subset/fingerprint semantics and
discovery coverage. A request below the cache's K floor fails explicitly.
A capped cache remains non-exhaustive. The 100k Nynorsk K=5 cache contained
814,981 exact pairs and occupied about 9.6 MiB including part headers, compared
with 265 MiB of count payload for all 23,179,805 positive-overlap pairs.
Reproducing all five threshold files took about 4.9 s including writes; their
pair sequences and counts matched fresh joins and the independent reference.

#### Bounded explicit-pair passage inspection

After a stage completes:

```bash
.venv/bin/python -m experiments.overlap.inspect_overlap_passages \
  /path/to/build/index /path/to/nno_Latn \
  /path/to/experiment-output/25000 \
  /path/to/new-inspection-output
```

The command selects a bounded purposive set of high-overlap, near-K and
cap-omitted pairs. `FingerprintChainDetector.get_pair_passage(i, j)` bypasses
candidate retrieval/top-k entirely, retrieves the pair's stored positional
occurrences, and retains its strongest surviving distinct-hash cluster.
Resource limits are explicit: at most 5,000 alignment points and 256 MiB of
posting payload per inspected pair; a limit produces a reported resource-limit
status, never a false “no passage”.

Source texts are loaded in one corpus pass. Their forward sets are recomputed,
and every retained aligned hash/position is checked against exact normalized
source k-grams. Evidence includes token spans, distinct support, bounded short
prefixes, and a bounded normalized sequence comparison. Rectangle thresholds
30/30, minimum distinct support 5, and a 20-token exact-run inspection label are
**exploratory**, not calibrated production acceptance criteria.

Synthetic controls test a genuinely aligned passage, scattered shared k-grams,
and contiguous template text. The last is intentionally important: a template
can be a real aligned passage while still being a semantic negative for the
intended reuse investigation. Passage evidence does not establish copying
direction, licensing status, or copyright infringement. This is not a
representative precision sample or graph-building stage.

The 25 October progressive Nynorsk run completed all 60 main configurations
across 25k, 50k and 100k documents. The 100k stage stayed at about 2.15 GiB
process-lifetime peak RSS, with a 185 MiB join-owned estimate and 400,000 bytes
of marker state. Its uncapped qualifying-pair counts were 119,988 / 205,713 /
286,277 / 503,803 / 814,981 for K=50 / 30 / 20 / 10 / 5.
The df-cap experiment retained 98.36% at K=50 but 64.82% at K=5; returned counts
were exact in both cases. Every exact/threshold pair set matched as required.

The real-data pause/resume demonstration matched the uninterrupted output.
Inspection of 26 selected pairs from the 25k stage found 22 surviving positional
clusters and four without a cluster, with all retained source evidence verified.
These samples do not establish passage precision. Detailed measurements,
limits and remaining decisions are in [docs/SIMILARITY_JOIN_PLAN.md](docs/SIMILARITY_JOIN_PLAN.md).
Local reports and output parts are under
`/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/overlap_experiments/2026-10-25-progressive/`.

#### Exact prefixes and a full-corpus overlap run

`run_document_stream(..., prefix_filter=True)` uses both documents' rarity
prefixes of length `L - K + 1`, for eligible length `L >= K`. A shared global
order guarantees that a pair with overlap at least K has a prefix intersection.
This is lossless in uncapped mode; the complete numeric-hash forward sets
still determine exact counts. Combining prefixes with a df cap is a separate
approximate candidate policy whose recall must be measured again.

This is **not** a fix for duplicate verification: both existing schedulers
already skip repeat discoveries of accepted and rejected pairs within a run.
Prefix filtering instead avoids checking many distinct pairs that cannot
reach K. Separate K runs and interrupted uncommitted probes can legitimately
repeat work; lowest-K exact-count reuse avoids the former for a known sweep.

The gated full-run command validates prefixes against a saved subset oracle,
prepares full-corpus views, validates representative owner probes against all
earlier-owned corpus partners, and only then runs the uncapped exact join:

```bash
.venv/bin/python -m experiments.overlap.run_full_overlap_join \
  /path/to/build/index /path/to/build/overlap_inverted \
  /path/to/progressive-experiment/100000 \
  /path/to/full-run-output --memory-gib 32
```

It runs K=5 once and derives K=50,30,20,10,5 afterward. The full view reuses
the source forward/overlap payloads and stores only K-independent document-rank
helpers. It does not rebuild, reorder or copy the normal index payloads.
Helpers and checkpoints bind source and array identities. Prefix settings
and explicit pilot probe ranges are part of the checkpoint contract; partial
owner ranges are never advertised as an exhaustive corpus join.

Full-output checks cover every output checksum, ID/count bounds, monotone
ownership and duplicate suppression. A seeded accepted-pair sample is
recomputed with the original numeric forward sets. Completeness rests on the
lossless prefix bound, subset oracle comparisons, full-view integrity and
full-partner pilot checks; an independent exhaustive full-corpus oracle is not
silently claimed.

New full runs coalesce accepted rows into bounded 1,000,000-row output buffers,
flushing before each committed probe range. This avoids emitting a tiny file
for every candidate-processing batch. Existing checkpoint chunk geometry is
preserved on resume. Derived K files are written and checked with bounded
buffered NumPy streams rather than keeping all output mappings resident.
Full-corpus source mappings are reused, and preparation releases its old
mappings before reopening the published view.

The completed Nynorsk run covered all **1,028,632 documents** and found
87,326,219 exact K>=5 candidate pairs. Filtering the same counts produced
12,425,346 / 21,685,303 / 30,313,569 / 53,966,431 pairs at K=50 / 30 / 20 / 10.
The join took about 73.4 minutes; process peak RSS was 4.81 GiB. Duplicate
discoveries were suppressed, and the source indexes were unchanged.

All output structure/checksums were checked, 10,000 accepted-pair counts were
independently recomputed with NumPy set intersection, and every derived
threshold row matched the exact floor-K output. The report is
`/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/overlap_experiments/2026-10-25-prefix-full/report.json`;
the additional proof is in `independent_verification.json` beside it.
These are overlap candidates, not all positionally verified passages.

#### Nynorsk groups, default passage review and document retrieval

Open the executed [Nynorsk review notebook](notebooks/nynorsk_group_review.ipynb) with the
workspace `.venv` kernel. It contains complete component summaries, bounded
candidate-versus-passage-support graphs, paired text evidence, raw metadata
and reusable batch retrieval examples.

Complete components were computed **before** selecting display examples:

| K | Candidate pairs | Documents with partners | Groups with 2+ documents | Largest group |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 12,425,346 | 161,988 | 39,299 | 27,691 |
| 30 | 21,685,303 | 246,275 | 55,289 | 48,112 |
| 20 | 30,313,569 | 316,766 | 66,445 | 76,596 |

Every component label and document degree matched an independent full-edge
SciPy sparse-graph calculation. Components are not cliques and do not imply
one passage shared by every member. The largest component contains about
96-98% of the candidate edges, depending on K; giant-component displays are
explicitly labelled subsets, not complete shared-passage groups.

The bounded review includes **149 unique pairs / 239 documents**: 104
random-priority pairs stratified by overlap and smaller fingerprint-set size,
plus strongest-overlap and bounded-group examples. It calls
`FingerprintChainDetector(index_dir).get_pair_passage(doc_i, doc_j)` without
changing constructor defaults: rectangle clustering, position/offset/distance
thresholds 30, minimum distinct support 5, no rarity weighting. It uses stored
4-gram/window-6 fingerprints, bypasses top-k retrieval, and reports only the
best surviving cluster.

148 pairs produced a cluster; one exceeded the 5,000-alignment-point safety
limit and remains **unresolved**, not negative. Source forward sets and every
returned aligned hash/position/exact normalized k-gram were verified.
This unweighted, supplemented sample is **not a precision estimate**.
Template/boilerplate alignment can also pass. Human relevance labels remain
blank in the review worksheet; no passage threshold calibration was performed.
Inspected examples include shared `Detaljer` recommendation snippets on
differently titled Docplayer pages and applicant-disclosure wording in job ads.
Ten selected pairs share a source URL, but none has identical raw text; such
crawl/version/extraction variants need a different interpretation from cross-site
copying.

Follow-up analysis of the complete K=50 giant group is reproducible with
[analyze_overlap_component.py](experiments/review/analyze_overlap_component.py) and
appears in notebook section 7. Of its 27,691 documents, 20,520 (74.1%) are from
`docplayer.me`; **99.61% of its edges are Docplayer-to-Docplayer**. Exact raw-text
layout counts find 10,640 of those records with at least three standalone
`Detaljer` lines, including 10,340 with at least 50. The sampled edges reveal
repeated recommendation/excerpt blocks alongside genuine document text.
Removing Docplayer vertices as a diagnostic leaves a largest component of 983.
Subtracting the 899 fingerprints occurring in more than 1,000 component
documents removes 69.01% of edges but still leaves a 27,654-document component.
These are diagnostic counterfactuals, not production filters or a full-corpus
df-cap experiment. No fingerprint is present in every giant-group member.
Investigate content regions before treating graph components as reuse groups.

The [three-host follow-up](reports/nynorsk/three_domain_overlap_review.md) separately examines
Helsetilsynet, Skogholt and Regjeringen with exact member/edge counts and
source-verified examples. It distinguishes inspection-report templates,
article text repeated across archive/category pages, and recurring official
document text or publication mirrors. The report links complete highlighted
text pairs for all three hosts. Its reproducible evidence collection is
[analyze_overlap_hosts.py](experiments/review/analyze_overlap_hosts.py); bounded
sample interpretations are not automatic corpus-wide relevance labels.

[export_pair_review.py](experiments/review/export_pair_review.py) exports two complete
stored text fields to Markdown, using the existing token-to-character offset
helper. Yellow highlights mark shared stored k-grams; bold yellow marks the
best default-detector cluster. It preserves original punctuation/whitespace,
checks source fingerprints and exact normalized shared k-grams, and includes
navigation anchors plus a separately labelled full-corpus-df<=1,000 count
diagnostic. Open the output in Markdown Preview; unhighlighted words may also
match because the display is limited to stored winnowed fingerprints.

The reusable APIs are [overlap_groups.py](findmytext/overlap_groups.py) and
[document_store.py](findmytext/document_store.py). The latter makes one initial,
resumable pass over raw `.jsonl.zst` shards, retains **all original JSON fields**
(including nested annotations), and publishes an integrity-checked SQLite
store. Later batch lookups use its document-ID index, not another corpus scan:

```python
from pathlib import Path
from findmytext.document_store import CorpusDocumentStore, build_document_store
from findmytext.overlap_groups import OverlapGroups

root = Path("/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk")
index = root / "findmytext_build/index"
store_file = build_document_store(
    root / "nno_Latn", root / "document_store/corpus.sqlite",
    index_dir=index,
)
with CorpusDocumentStore(store_file, index_dir=index) as store:
    records = store.get_many([0, 1], id_kind="internal")
    metadata = store.get_many([0, 1], id_kind="internal", metadata_only=True)
    originals = store.get_many([records[0]["id"]])  # original string corpus IDs

with OverlapGroups(root / "group_review/2026-10-26-defaults/k50") as groups:
    largest = groups.select(mode="largest", min_documents=3, limit=5)
    all_at_least_5 = groups.select(mode="all_above_min", min_documents=5)
    members = groups.members(int(largest[0]["group_id"])).copy()
```

`get_many` returns a dict keyed by requested IDs; duplicate requests are
deduplicated in request order. `iter_many` yields records with SQL queries
batched at 500 IDs. Missing/invalid IDs are explicit errors. Internal IDs
require the matching index snapshot; external string IDs also retrieve records
excluded by the fingerprint ingestion filters. Metadata-only calls omit
`text` and `xml` but retain other original fields, including `seg_langs`,
`doc_scores` and `web-register`. Keep readers open while using their mapped
arrays, or copy membership slices before closing them.

The Nynorsk store contains **1,034,804 raw records**, including every indexed
ID, and occupies about **3.28 GiB**. A local 239-document batch took about
1.57 seconds for complete JSON and 0.78 seconds for metadata only. Query-plan
inspection confirmed indexed lookup; these are filesystem/cache observations,
not cold-cache measurements or an SLA. The source and index are bound by
paths/settings/file sizes/modification times, not a cryptographic hash of every
raw corpus byte. Resume skips completed shards and replays only an unfinished
compressed shard up to its committed line.

Results, full component arrays, independent checks and `human_review.csv` are
under `hplt_nynorsk/group_review/2026-10-26-defaults/`.
To rerun the package into a **new** review destination:

```bash
.venv/bin/python -m experiments.review.review_overlap_groups \
  --index-dir /nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/findmytext_build/index \
  --join-dir /nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/overlap_experiments/2026-10-25-prefix-full \
  --store-file /nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/document_store/corpus.sqlite \
  --output-dir /path/to/new-review
```

All graphs remain undirected: raw `ts` is crawl/provenance time, not established
publication time or evidence of copying direction. A future **selected-pair
winnowing-window-1** experiment (all 4-grams) is noted in the notebook and plan
but has **not** been executed. Neither grouping thresholds nor this review
require rebuilding the existing fingerprint indexes.

#### Safeguarded overnight cross-corpus prefix experiments

[overnight_prefix_benchmarks.py](experiments/overlap/overnight_prefix_benchmarks.py)
uses existing Icelandic, Latvian and Bokmal indexes; it does not download,
rebuild or blindly launch whole-corpus joins. It waits for Nynorsk's final
independent checks, then runs one corpus stage at a time with a 32 GiB working
allowance and single-worker kernels.

Each corpus gets a 25k uniform screening subset, an independent external count
reference, all five K values, both output modes, and uncapped/df<=1000 prefix
discovery. Screening pair contributions and positive-pair counts determine
whether to try 100k, fall back to 50k, or retain only screening results.
Current expansion safeguards are 250 million projected pair contributions
and 100 million projected positive pairs, using pair-space scaling estimates.
These are workload safeguards, not runtime guarantees or changes to exact
overlap semantics. Resource checks require 64 GiB available host RAM and
100 GiB free disk before starting a stage. Stage failures are logged and do
not prevent independent later corpora from being attempted.

The run freezes source files and records its policy, stage logs, source
identities, selected expansion sizes and results. Prefix-only stages can also
be invoked with `scale_overlap_join --prefix-filter --no-controls
--no-fingerprint-control`; unfiltered control repeats are deliberately omitted
from this overnight study. Subset recall is always measured against that
subset's uncapped exact reference, with document frequencies from the complete
source corpus.

The 25 October queue is under
`/nr/project/stat/COPY.AI/data/Lars/findmytext_runs/2026-10-25-prefix-overnight/`.
Attach without starting another queue:

```bash
tmux -L findmytext-builds attach -t prefix-overlap-overnight
```

After these pilots, [queue_full_overlap.py](experiments/overlap/queue_full_overlap.py)
rechecks independent results and source identities, checks working-memory and
projected disk headroom, and runs eligible full corpora sequentially. Each full
job is followed by independent numeric and complete derived-row validation.
Jobs that do not fit are explicitly held, not launched or marked complete.

The current 32 GiB preflights admit Icelandic (about 8.6 GiB) and Latvian
(about 27.8 GiB). The current full Bokmal path requires about 99.4 GiB and is
held pending an out-of-core redesign or explicit allowance change. Full runs
can extend beyond a night and potentially take days; the queue persists in
tmux session `full-overlap-followups` on socket `findmytext-builds`.
Its status/logs are under
`/nr/project/stat/COPY.AI/data/Lars/findmytext_runs/2026-10-25-full-followups/`.

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
python -m experiments.similarity.compare_similarity_methods --index-dir my_index --top-k 50 --min-shared-fingerprints 5
```


#### Inspect existing indexes

Print index statistics and save interactive figures for the four HPLT corpora
(Nynorsk, Bokmal, Icelandic, and Latvian) in a parent folder:

```bash
.venv/bin/python -m experiments.indexing.index_stats /path/to/corpora --output-dir /path/to/analysis
```

With no arguments, it analyzes the four indexes at
`/nr/project/stat/COPY.AI/data/Lars/hplt_<language>/findmytext_build/index`
(`nynorsk`, `bokmal`, `icelandic`, and `latvian`) and saves reports in
`/nr/project/stat/COPY.AI/data/Lars/index_analysis`:

```bash
.venv/bin/python -m experiments.indexing.index_stats
```

Explicit index arguments and `--output-dir` override these defaults.
The command discovers merged indexes in each immediate child folder, including
the `hplt_<language>/findmytext_build/index` layout. It also accepts a single
index/corpus folder, or explicitly named indexes:

```bash
.venv/bin/python -m experiments.indexing.index_stats \
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
.venv/bin/python -m experiments.indexing.index_stats --repair-plots
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
[docs/HASHING_FINDINGS.md](docs/HASHING_FINDINGS.md), including
their output locations. Existing reports can be rendered without loading old
indexes or implementing legacy hashing:

```bash
.venv/bin/python -m experiments.indexing.hash_collision_study \
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
.venv/bin/python -m experiments.indexing.benchmark_hash_width --documents 5000 --repeats 5 \
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
  .venv/bin/python -m experiments.indexing.profile_index_build \
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
python -m experiments.similarity.verify_multifile_similarity
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
