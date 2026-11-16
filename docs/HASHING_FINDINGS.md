# Fingerprint hashing, collisions, storage, and build profiling

Findings from the October 2026 discussion and experiments. Measurements below
are historical results, not claims that the old experiments can be rerun with
the current production code.

## Decision and migration

- Production now uses **rolling64-v1 only**, for both building and querying.
- The legacy31 implementation has been removed, not retained as an option.
- Indexes/shards with missing or unsupported hash metadata are rejected with
  an explicit rebuild message. Changing metadata cannot convert hash values.
- The four original corpus indexes were not modified by the experiments.
  They need rebuilding from source before use with the rolling64-only code.
- Historical report JSON/CSV/HTML, occurrence data, examples, and profiles are
  preserved. Legacy-dependent collection/build comparisons are retired.
  Existing collision reports can still be rendered without loading indexes.
- A subsequent, separate decision standardized the positional index on
  interleaved and removed columnar support. The findings and migration are in
  [POSTING_LAYOUT_FINDINGS.md](POSTING_LAYOUT_FINDINGS.md).

Production code: [winnower](../findmytext/winnower.py),
[builder](../findmytext/index_builder.py), and
[index reader](../findmytext/indexing.py).

## Hash algorithm

`rolling64-v1` is this project's versioned combination, not a standard named
algorithm with a published guarantee for the whole composition:

1. Hash each token's UTF-8 bytes using the inherited FNV-1a-style recurrence,
   retaining all 64 bits. Its offset basis is `1469598103934665603`, not the
   standard FNV-1a-64 offset basis.
2. Combine token hashes with an odd-base polynomial rolling hash modulo
   `2**64`. The default base is `0x9E3779B185EBCA87`.
3. Apply MurmurHash3's `fmix64` finalizer to each k-gram value. Roll the unmixed
   state, not the finalizer's output.
4. Apply the existing rightmost-window-minimum winnowing rule.

Unsigned 64-bit wraparound is intentional modular arithmetic; full 128-bit
products are not needed. The finalizer is bijective and cannot remove
collisions already present in its input. Neither this construction nor
legacy31 is cryptographic.

Background:
[FNV](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function),
[polynomial rolling hashes](https://en.wikipedia.org/wiki/Rolling_hash#Polynomial_rolling_hash),
and [MurmurHash3 fmix64](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp).

## Where legacy31 came from

Remote main was inspected at commit
[`8d032a8`](https://github.com/NorskRegnesentral/FindMyText/commit/8d032a8b30d515925e98f288d3dae4893ee1f41b)
(28 July 2026). It already generated hashes modulo `2**31 - 1` while storing
fingerprints as uint64. The modulus/comment existed in the first repository
version, `dfae28e` (30 June 2026).

The documented intention was to bound arithmetic intermediates in signed
int64 and assume the small hash space was sufficient for multi-fingerprint
matching. It was not a disk-packing optimization. Main did not persist a
forward index. The feature branch introduced one in `7943133` (30 September
2026), with uint64 fingerprints and offsets from the beginning.

## Study methodology and definitions

Both studies used the same eligible documents and the original tokenization,
`k=4`, and window size 6.

- **Fixed selection:** recover exact normalized token tuples at the positions
  selected by legacy31. Group by `(old hash, exact tuple)`. Repeated identical
  tuples are reuse, not collisions. Independent BLAKE2b-64 signatures provide
  additional checks, but exact tuple bytes are the ground truth.
- **Full wide-hash pipeline:** hash all k-grams independently of legacy31,
  then re-winnow. This changes selected positions as well as hash values.
- **Singleton:** a fingerprint found in exactly one distinct indexed document,
  even if it occurs repeatedly within that document.
- **Collided hash bucket:** one hash value representing at least two different
  exact normalized k-grams.
- **Merged identities:** `distinct exact k-grams - distinct hash values`.
- **Purely artificial shared bucket:** its union of documents has size greater
  than one, but every constituent exact k-gram occurs in only one document.

These percentages describe hashes or occurrences, not document-pair
false-positive rates. Even a pair receiving a collision contribution can
share genuine text elsewhere. Normalization removes punctuation but preserves
case. Full-corpus figures refer to indexed documents after ingestion filters.

### Matched 50,000-document Nynorsk sample

Uniform seeded reservoir sampling across all five shards, seed `20261022`.
All sampled forward fingerprint sets were checked against the source index.

| Representation | Distinct values | Singleton fraction |
|---|---:|---:|
| Legacy31, original selection | 6,428,841 | 94.1294% |
| Exact k-grams, original selection | 6,451,729 | 94.4630% |
| BLAKE2b-64, re-winnowed | 6,452,626 | 94.4747% |
| Rolling64-v1, re-winnowed | 6,452,387 | 94.4725% |

The original selection had 22,830 collided buckets (0.3551%); 22,772 mapped to
two k-grams and 58 to three. Neither wide-hash selection had an observed
collision in the sample. Sample singleton fractions are not corpus-wide
singleton fractions, and sample multiplicities can miss additional variants.

### Full Nynorsk

1,028,632 documents; 560,956,406 k-gram occurrences before winnowing.
The full wide-hash run used **BLAKE2b-64**, not rolling64. Rolling64 was
evaluated on the matched Nynorsk sample above.

| Representation | Distinct values | Singletons | Singleton fraction |
|---|---:|---:|---:|
| Legacy31 | 91,635,849 | 70,133,133 | 76.5346% |
| Exact k-grams, original selection | 96,487,057 | 77,769,362 | 80.6008% |
| BLAKE2b-64, re-winnowed | 96,482,056 | 77,771,437 | 80.6072% |

- 4,659,842 old hashes collided: 5.0852% of observed hash values.
- 4,851,208 exact identities were merged.
- 9.9854% of selected occurrences lay in collided buckets; maximum multiplicity 6.
- 2,979,856 shared buckets had no actually shared constituent k-gram.
- Zero observed BLAKE2b-64 collisions in the analyzed selections.
- Holding selection fixed, collision correction raised the singleton fraction
  by 4.0662 percentage points.

### Full Latvian

11,298,371 documents; 5,701,124,632 k-gram occurrences before winnowing.
The full wide-hash run used the production **rolling64-v1** implementation.

| Representation | Distinct values | Singletons | Singleton fraction |
|---|---:|---:|---:|
| Legacy31 | 585,543,576 | 290,324,782 | 49.5821% |
| Exact k-grams, original selection | 897,568,432 | 705,627,237 | 78.6154% |
| Rolling64-v1, re-winnowed | 897,868,933 | 705,873,391 | 78.6165% |

- 217,681,166 old hashes collided: 37.1759% of observed values.
- 312,024,856 exact identities were merged.
- 967,619,675 selected occurrences lay in collided buckets: 59.7077% of
  1,620,595,129 selected occurrences. This does not mean each is a false match.
- Maximum observed multiplicity was 12.
- 122,211,189 buckets were shared only because of collisions: 41.3968% of
  the 295,218,794 old shared buckets, not of candidate pairs.
- Zero observed rolling64 collisions among 897,868,933 distinct selected
  k-grams. This is empirical evidence, not a guarantee for unseen inputs.
- Fixed-selection collision correction raised the singleton fraction by
  29.0333 percentage points; re-winnowing changed it by another 0.0011 points.
- Only about 28.48% of original selected positions remained selected, despite
  the nearly identical aggregate corrected singleton fractions.

After excluding singletons, the number of retained hash lists falls from
295,218,794 to 191,995,542: 103,223,252 fewer lists, a 34.97% reduction. Runtime
also depends on list lengths and candidate-pair work.

Example hash `15979536` represented 12 different k-grams in 12 documents and
13 occurrences; every individual k-gram belonged to only one document.
The examples JSON intentionally shows at most three variants per bucket.
The separate untruncated example contains all 12.

## Physical storage versus hash precision

| Representation | Fields | Bytes |
|---|---|---:|
| In-memory build occurrence | uint64 hash, uint32 document, uint32 position | 16 |
| Positional inverted posting | uint32 document, uint32 position | 8 |
| Fingerprint directory row, excluding headers | uint64 hash, uint64 byte offset, uint32 length | 20 |
| Forward entry | uint64 fingerprint, no position | 8 |

The original legacy fingerprints had only 31 effective bits, but their storage
fields were already 64-bit. Forward lists contain unique, numerically sorted
hashes per document; forward offsets are array-entry offsets, not positions.
Inverted lengths count occurrences, not unique documents.

Moving to rolling64 does not double these fields. More distinct fingerprints
can nevertheless enlarge the directory. Latvian's measured forward-entry count
changes only from 1,573,590,799 to 1,573,864,374: about 2.19 MB of additional
8-byte payload, not a doubled forward index.

Configurable 40/48-bit hashes and packed positions were discussed but are
**not implemented**. A 48-bit hash plus 16-bit position fits a 64-bit pair,
but still needs document-ID storage. An alternative is full-width hashes
with uint16 positions, reducing inverted postings from 8 to 6 bytes if
range checks and a new format are implemented. No such layout change was made.

## Longest indexed documents (historical positional indexes)

| Corpus | Maximum stored position | Normalized token maximum |
|---|---:|---:|
| Nynorsk | 22,962 | 22,970, exact from full manifest |
| Bokmal | 20,094 | 20,098-20,103, bounded |
| Icelandic | 29,153 | 29,157-29,162, bounded |
| Latvian | 17,762 | 17,766, exact from full manifest |

Positions are zero-based k-gram starts. With k=4 and window=6, a final selected
position p bounds token count by `p+4 <= n <= p+9`. Highest-position ranking
need not identify the exact token-length winner when lengths are close.
Distinct-forward-fingerprint count is not document word count.

## Hashing speed versus full build speed

### Warmed tokenization-to-winnowing benchmark

Same 5,000 Nynorsk documents, five repetitions; median seconds. Excludes
decompression, index I/O, and JIT warm-up.

| Historical method | Seconds |
|---|---:|
| Original legacy31, Python token-byte loop | 4.1366 |
| Bit-identical legacy31 with compiled token hashing | 1.4474 |
| Production rolling64-v1 | 1.4089 |
| Independent BLAKE2b-64 reference | 4.6643 |

Most pipeline improvement came from compiling token hashing. The optimized
legacy variant was benchmark-only and was never the production legacy default.
There was no shipped rolling32-v1 option. A separate synthetic kernel test
on one million prehashed tokens measured about 4.31 ms for the legacy rolling
kernel and 1.74 ms for rolling64, excluding all text processing and winnowing.

Both rolling kernels were Numba-compiled. Modulo-2**64 wraparound avoids the
explicit prime reduction. `cache=True` helps compilation reuse, not warmed-loop
throughput; `nogil=True` permits concurrent calls but does not parallelize a loop.

### Complete 50,000-document build

Same sampled input, k=4, window=6, `nb_workers=4` (three hashing workers plus
producer, alongside the main process), fresh subprocesses and warmed JIT.
Two timing builds per method, with method order reversed for the second pair.
Separate instrumented builds captured CPU profiles in all relevant processes.

| Exclusive wall-time activity | Legacy31 seconds (% total) | Rolling64 seconds (% total) |
|---|---:|---:|
| Other extraction / pipeline waiting / orchestration | 13.87 (15.85%) | 16.63 (17.49%) |
| Main-process occurrence insertion | 3.70 (4.23%) | 4.16 (4.37%) |
| Sorting and compressed intermediate export | 16.01 (18.30%) | 17.39 (18.28%) |
| Inverted-index merging | 43.25 (49.43%) | 46.78 (49.19%) |
| Forward-index construction | 10.66 (12.19%) | 10.15 (10.67%) |
| Total | 87.49 | 95.10 |

Rolling64 took 8.70% longer end to end in this workload. The waiting remainder
is not pure hashing time: worker work overlaps other processes.

| Output | Legacy31 | Rolling64 |
|---|---:|---:|
| Compressed intermediate MsgPack | 69.88 MB | 99.49 MB |
| Final index, including forward index | 252.27 MB | 252.71 MB |

MB is decimal. Intermediate size grew 42.37%; final size grew only 0.1731%
in this sample. MsgPack/process messages encode integer values differently,
whereas final arrays use fixed-width fields. Full-corpus directory growth
can be larger because more collision-free identities are retained.

CPU diagnostics highlighted GZip small writes, per-hash Python export/merge
loops, pickling/IPC, and small NumPy conversions. Their exclusive CPU times
are additive across processes, not wall time; profiler overhead affects them.
The one-shard, small-run-count, controlled id/text input and cache/server-load
conditions limit extrapolation to full-corpus multi-shard builds.

Prioritize batched binary serialization/transfer and vectorized merging before
further hash arithmetic tuning. Possible hashing improvements include batched
UTF-8 preparation, buffer reuse, and fused rolling/winnowing with identical
tie handling. Token caching requires a benchmark and bounded memory.

## Historical layout trade-off and interleaved-only decision

Both layouts used 8 bytes per occurrence:

- Interleaved: `(document, position), (document, position), ...`.
- Columnar: all document IDs, followed by all positions.

Columnar saved position I/O for document-only lookups. Interleaved obtains
document/position pairs with one contiguous read and is now the only format.
The proposed sidecar stores sorted unique document IDs for `df >= 2`, with
fingerprints ordered by `(df, hash)`; it is still a separate planned structure.

Once self-join candidate generation uses that sidecar, the main reason for
columnar postings in the similarity workflow is weaker. However, external
query candidate retrieval still uses the original index. A corpus-singleton
fingerprint can be useful evidence for an external query, so the singleton-free
sidecar cannot replace that lookup losslessly.

The subsequent decision accepts that possible external-query I/O trade-off
to simplify storage and standardizes on interleaved. Columnar readers/writers,
selection flags and the conversion benchmark are removed; declared columnar
indexes are rejected rather than misread. This is an architectural simplicity
decision, not a measured universal interleaved speed advantage. The sidecar
is still planned. See [POSTING_LAYOUT_FINDINGS.md](POSTING_LAYOUT_FINDINGS.md).

## Location of frozen measurements

Historical outputs remain outside git under:

```text
/nr/project/stat/COPY.AI/data/Lars/index_analysis/
  nynorsk_hash_study_20261022_50k/
  nynorsk_hash_study_20261022_full/
  nynorsk_rolling64_v1_20261022_50k/
  latvian_rolling64_v1_20261022_full/
  nynorsk_index_build_profile_20261023_50k/
  longest_indexed_documents_20261023.json
  longest_indexed_documents_20261023.csv
```

Their JSON/CSV data, exact-variant Parquet tables, example IDs/positions, and
HTML reports are the primary measured evidence. Migration does not relabel or
overwrite them. Future index profiling and new builds use rolling64 only.

The earlier histogram x-axis explosion (ticks near `10**297`) was a Plotly
annotation-coordinate bug, not a count error: logarithmic-axis annotations
need log10 coordinates, while the bars/shapes use raw counts. All twelve
existing histograms were repaired from embedded HTML data without rescanning
indexes, retaining `.html.bak` originals. The cross-corpus comparison report
contains both singleton-count and singleton-fraction plots with metadata on
hover. Those repairs did not change JSON/CSV statistics.
