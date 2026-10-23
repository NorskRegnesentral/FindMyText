# Document Similarity Join: Discussion Plan

**Status:** Draft with an agreed index/scoring contract. The derived join index
and optimized join described below are not implemented yet.

## Agreed build and post-processing contract

1. **Build the normal positional indexes with rolling64-v1.** The legacy hash
   implementation has been removed; existing legacy indexes/shards require
   rebuilding from source. Keep the original inverted
   index with document IDs and occurrence positions, and the existing forward
   index with each document's unique fingerprints sorted by numeric hash value.
   Both full indexes retain singletons; filtering applies only to the sidecar.
   "Full forward" means all unique selected fingerprint values, not every
   occurrence or every pre-winnowing k-gram.
2. **After the normal build, create a separate screening inverted index.**
   Retain a fingerprint if and only if it occurs in **at least two distinct
   documents**, `df(f) >= 2`. This means more than one document, not more than
   two: fingerprints occurring in exactly two documents must be retained.
3. **Store document IDs only in this derived index.** Each fingerprint points
   to an ascending list of unique document IDs. Repeated occurrences within one
   document contribute only one entry. Do not store token positions or
   occurrence offsets in these posting lists.
4. **Order the derived fingerprint records rarest first**, by the deterministic
   key `(df(f) ascending, fingerprint_value ascending)`. Rarity is measured
   only by document frequency. Equal-frequency fingerprints have equal rarity;
   numeric hash value is a deterministic tie-breaker, not another rarity score.
   This orders fingerprints, not document IDs within a posting list.
5. **Keep the original forward index hash-sorted and unchanged.** Its offsets
   locate each document's slice in the forward array; they are not positions
   in the source document. A rarity-ordered forward view, if needed for prefix
   filtering, is a separate derived representation.
6. **Use an absolute overlap-count threshold**, `|F(a) intersection F(b)| >= K`,
   where `F(d)` is the set of distinct fingerprint values for document `d` and
   `K` is a positive integer. Do not divide by union size or call this Jaccard.
7. **Support two explicit overlap-output modes:** `threshold_only`, which
   returns qualifying pairs without claiming final overlap counts, and
   `exact_counts`, which also returns exact counts for accepted pairs. With
   uncapped exact candidate generation, both modes must return the same
   exhaustive qualifying pair set. The default mode remains to be chosen;
   record the selected mode and any candidate-generation cap in metadata.
8. **Keep both execution orders open.** Benchmark global rare-first
   fingerprint-driven traversal against document-driven probing ordered from
   smaller to larger eligible fingerprint sets. Choose by end-to-end time,
   memory, and I/O; do not assume one is fastest. They require different
   checked-pair state, but share the sidecar and overlap verifier.
9. **Use Numba for the first compiled implementation**, including batched
   candidate processing and overlap verification.
10. **Start with Nynorsk and visible, incremental deliverables.** Establish a
    correct single-worker structure before exploring parallel speedups.
11. **Preserve the hashing findings separately.** The completed experiments
    are frozen in [experiments/HASHING_FINDINGS.md](./experiments/HASHING_FINDINGS.md).
    Production now uses rolling64 only; no old index was overwritten or relabeled.
12. **Use a candidate document graph for downstream inspection.** Documents
    are nodes and qualifying overlap pairs are undirected edges. Group them
    by connected components, select the largest eligible document group or
    all groups above a configurable minimum size, and verify their candidate
    edges using `FingerprintChainDetector`. For each pair retain only its
    highest-scoring surviving passage cluster if it meets the separate
    passage-score threshold, then display graph and shared-text evidence.

Omitting singletons from the derived index preserves every document-pair
overlap count: a singleton cannot belong to the intersection of two documents.
The original positional index remains available for final text verification.
The value of K and the final text-match acceptance rule remain to be chosen.

The execution and data-structure choices below are recommendations to prototype
and benchmark, not claims that the optimized join already exists.

## Goal

Find pairs of documents that share enough winnowed fingerprints to be worth
checking for an actual shared text segment, while avoiding the quadratic
candidate explosion caused by very common fingerprints. Fingerprint overlap is
a cheap screening signal; the final result should be supported by a coherent,
position-aligned text match, not merely by fingerprints scattered throughout
two documents.

The proposal in the shared sketch is a good starting point:

1. Build the normal inverted and forward indexes without changing their
   current behavior, then derive a position-free screening inverted index.
2. Rank fingerprints by document frequency, and avoid spending work on
   fingerprints that cannot help identify a pair.
3. Propose pairs from the derived inverted index, then count shared fingerprints
   using the original hash-sorted forward lists.
4. Keep qualifying pairs as possible matches and check whether their matching
   fingerprints form a coherent text segment.

This plan makes the correctness conditions explicit and stages the work so we
can compare each optimization with the current exact implementation.

## What exists today

- The merged index stores sorted fingerprint posting lists with document IDs
  and token positions. The postings can be interleaved or columnar.
- The forward index stores each document's **unique** fingerprints in numeric
  fingerprint order, **not source-document position order**. Its offsets are
  array slice boundaries, not token offsets. Posting lists may contain multiple
  occurrences of a fingerprint in a document; the current similarity computation
  de-duplicates a document within each fingerprint before counting shared fingerprints.
- `compute_similarities_inverted` enumerates all document pairs for each
  fingerprint and accumulates their exact shared-fingerprint counts. It can
  cap fingerprint document frequency, but doing so drops pair contributions
  and therefore makes results approximate.
- `compute_similarities_forward_topk` retrieves a bounded number of neighbors
  per document; it is useful for retrieval, but is not a complete all-pairs
  join.
- Posting positions are available for later verification. The current forward
  index does not contain positions.

#### Positional posting-layout decision remains open

The `columnar`/`interleaved` setting applies only to the full inverted index's
document/position posting payload. The full forward index always uses
document offsets plus concatenated hash-sorted unique uint64 fingerprints,
without positions. The document-only sidecar has no document/position pair
to interleave; its fingerprint/offset/document-ID arrays are independent of
this positional layout choice.

The intended lookup routes are:

| Operation | Fingerprint source | Lookup/verification structure |
|---|---|---|
| External text query | Fingerprint the query text | Full positional inverted index, including singletons |
| Existing `get_document_candidates(id)` | Full forward set for that indexed document | Full inverted index; this API is not silently redirected |
| New indexed-document screening join | Selected indexed-document fingerprints | Derived `df >= 2` document-only sidecar for candidate generation |
| Exact overlap verification of an indexed pair | Both full forward sets | Sorted set intersection |
| Positional passage verification | Selected pair and shared hashes | Full positional inverted index and source texts |

`get_document_candidates` expects an internal ID of an indexed document, not
arbitrary external text. External query helpers accept freshly computed
fingerprints. Keeping all fingerprints in the full indexes is distinct from
choosing their physical posting layout.

Keep interleaved (the default) and columnar support while implementing the
sidecar. The document-only `df >= 2` sidecar removes much of the motivation for
columnar reads during indexed-document self-joins. Positional verification
usually needs both IDs and positions, where interleaved provides one contiguous
read. Both layouts have the same uncompressed posting payload size.

However, an external query can legitimately match a fingerprint found in only
one indexed document. The singleton-free sidecar is not a lossless replacement
for that search, and original-index document-only candidate retrieval still
exists. Columnar may retain an I/O advantage there. Do not route all external
queries through the sidecar or remove a layout before measuring those paths.

Recommendation, not yet an implementation decision: use interleaved as the
canonical positional format if sidecar and external-query benchmarks show no
material benefit from keeping columnar. Do not store both physical copies just
to support an option.

Relevant code: [index_builder.py](./findmytext/index_builder.py),
[indexing.py](./findmytext/indexing.py), and
[similarity.py](./findmytext/similarity.py).

## Agreed overlap semantics

The initial experiment uses the following exact rule:

> A pair is a fingerprint-overlap candidate if its two documents share at
> least **K distinct fingerprints**.

This is an **absolute overlap-count threshold**, not a Jaccard threshold or
normalized overlap coefficient. Jaccard would instead be
`|F(a) intersection F(b)| / |F(a) union F(b)|`; no such normalization is used.
Counts must use unique fingerprint values per document, matching the current
`iter_unique_posting_batches` behavior. Repeated occurrences remain available
as positional evidence for the later text check.

Keep these thresholds separate:

- **K:** minimum shared unique fingerprints for screening.
- **Text-match rule:** minimum length/coverage/continuity of a verified common
  segment. This is not defined by K and needs its own explicit semantics.

The papers below provide relevant filter-verification and prefix/tree
techniques. Adapt their bounds to the agreed fixed overlap threshold K;
Jaccard-specific bounds cannot be copied unchanged. Set-overlap verification
also does not replace the later positional text verification.

## Candidate approach

### Stage 1 — build normally and establish an exact baseline

Normal corpus ingestion, fingerprint extraction, merging, and forward-index
construction are unchanged. All screening structures below are post-processing
outputs in a separate directory; they must not overwrite source index files.

Use the existing inverted-index pair counting as the reference implementation.
On a manageable corpus, record the exact set of pairs with overlap at least K,
their overlap counts, and run time/memory/intermediate-output size. Use
synthetic cases with known results as well as a real sample.

This gives us a correctness oracle for every optimized candidate generator.
Do not use `max_fingerprint_document_frequency` for this reference: dropping
high-frequency fingerprints can remove true qualifying pairs.

### Stage 2 — derive the position-free, rare-first inverted index

Compute document frequency `df(f)` as the number of distinct documents
containing fingerprint `f`, not its number of positional posting occurrences.
The existing `iter_unique_posting_batches` iterator already yields unique
`(fingerprint, doc_id)` pairs and is suitable as the source.

For each fingerprint:

- de-duplicate document IDs;
- discard the derived record only when `df(f) < 2`;
- retain the original fingerprint value and the ascending unique document IDs;
- discard positions only from this derived representation.

Sort the retained fingerprint records by `(df(f), fingerprint_value)`.
Both keys are ascending. This is the agreed rare-first order; do not reverse
it or drop common fingerprints. Construct the sidecar in bounded memory,
including its sorting and posting-list rearrangement steps.

A compact representation consists of:

- fingerprint values in rare-first order;
- posting-list offsets of length `n_retained_fingerprints + 1`;
- concatenated unique document IDs, using the same internal IDs as the source.

Here the posting-list offsets are storage pointers, **not document/token
positions**. Document frequency is the difference between adjacent posting
offsets. Store source-index identity/settings and the ordering/filter contract
in metadata so unrelated indexes cannot be mixed accidentally.

This is a new layout: its fingerprint array is not numerically sorted.
Do not pass it to the existing `DiskBasedIndex` reader or use numeric
`searchsorted` on the rare-first array. If hash lookup is needed, maintain a
separate hash-to-rank lookup without changing fingerprint identity.

#### Document order within posting lists

The initial storage contract remains ascending original document ID, preserving
source IDs and allowing simple range restriction for pair ownership. Sorting
fingerprint records by rarity and sorting documents within each record are
independent choices.

Benchmark a separate length-ordered variant before changing that contract.
Mann et al., Sections 2.2 and 3, store posting-list sets in increasing set size
and process probing sets in increasing size. Equal-sized input sets are ordered
lexicographically in their implementation.

For our overlap join, the relevant length is the number of distinct eligible
fingerprints, not source-text characters, token count, or positional posting
count. Define `E(d) = {f in F(d): df(f) >= 2}` and compute `|E(d)|` during the
sidecar build. Removing singletons preserves all cross-document intersections.
A length-order experiment should use `(len(E(d)), original_doc_id)` as a stable
join order and retain mappings to original IDs; do not renumber source indexes.

For fixed overlap K, the length filter only requires both sets to have at
least K elements. There is no finite upper bound on a partner's size: a short
set can be contained in a much larger one. Therefore do not import Jaccard
length-ratio pruning. Shorter-first order may still help scheduling, locality,
and candidate processing, but its end-to-end benefit must be measured.
Any length-ordered posting variant must declare its order in metadata; it cannot
use numeric-document-ID binary search or early termination as if IDs were sorted.

#### Build progress and final statistics

Print progress by build phase: scanning/de-duplication, frequency ranking,
posting-list layout/write, and completion. Print a final human-readable summary
and save the same counters/settings as JSON, including:

- source index, hash/settings identity, document count, and `df >= 2` cutoff;
- input distinct fingerprints, removed singletons and percentage, retained
  fingerprints and percentage, and any zero-document records detected;
- input positional postings, unique document/fingerprint pairs before
  filtering, and within-document repeated postings collapsed;
- unique singleton pairs removed and final derived document-ID posting count;
- documents with no retained fingerprints and the distribution of `|E(d)|`;
- document-frequency min/quantiles/max for retained fingerprints;
- actual bytes for fingerprints, offsets, document IDs, lookup structures, and
  total sidecar size, not just savings in the document-ID payload;
- elapsed time by phase, temporary disk use, and measured peak memory where
  available; fingerprint and document ordering conventions.

For valid source indexes, check these accounting identities:

```text
retained_fingerprints = input_fingerprints - singleton_fingerprints
retained_doc_id_postings = unique_doc_fingerprint_pairs - singleton_fingerprints
collapsed_repeats = input_positional_postings - unique_doc_fingerprint_pairs
```

Each retained list must have at least two distinct in-range document IDs.
Unexpected zero-document fingerprints or broken accounting are validation
errors, not silently discarded input.

#### Forward representations

The original forward index remains unique and **numeric-hash sorted** within
each document. It supports exact overlap verification by a two-pointer
intersection. Neither building this sidecar nor counting overlaps requires
reordering the original forward index.

If a later prefix-filter implementation uses rarity as its global element
order, build a separate rank-ordered forward view using the same
`(df, fingerprint_value)` ordering. Do not mix hash-ordered prefixes with
rarity-ordered prefix bounds or reinterpret rank as the fingerprint value.

### Stage 3 — generate and verify candidates without a giant pair list

Keep the existing fingerprint-driven pair-count/chunk-merge method as the
reference. Its aggregation combines repeated pair contributions; it is not
an index of pairs whose full overlap has already been verified.

**Scheduling option A: fingerprint-driven traversal**, matching the global
rare-first workflow under discussion:

1. Traverse derived fingerprint records in global `(df, hash)` order.
2. Retrieve that fingerprint's sorted unique document IDs.
3. Generate the unordered document pairs in bounded blocks, never a complete
   `triu_indices`/pair matrix for a very large list.
4. Look up each canonical pair in the run-wide checked-pair index. Skip pairs
   whose full overlap decision is already complete.
5. For each new pair, run the compiled merge verifier against the complete,
   unique, numeric-hash-sorted forward lists.
6. Record a completed rejection or acceptance; exact-count mode also records
   the final overlap for accepted pairs. Stream accepted results to bounded
   output chunks.
7. Continue through all fingerprint records for exhaustive mode, or stop at an
   explicitly configured document-frequency cap in a labeled non-exhaustive
   candidate mode.

The fingerprint being traversed discovers the pair; it is not the only
fingerprint counted during verification. A pair's full overlap decision is
final for the fixed run settings, even when more shared fingerprints are
encountered later in the outer traversal.

Rare-first ordering alone does not reduce the number of pair updates if every
list is still expanded into every document pair. Measure the benefit of
provably safe pruning separately from the storage/I/O benefit of the sidecar.
Candidate verification can use the unchanged hash-sorted forward sets.

#### Checked-pair state for the fingerprint-driven loop

This loop requires a global history across fingerprint records, containing
both accepted and rejected pairs. Per-document epoch markers cannot replace
that history without changing the loop order: the same pair can reappear
arbitrarily later under another fingerprint.

Use one canonical identity for an unordered pair:

```text
a = min(doc_i, doc_j)
b = max(doc_i, doc_j)
pair_key = (uint64(a) << 32) | uint64(b)
```

This encoding matches the current uint32 source-document-ID representation;
validate the range, widen before shifting, and omit self-pairs. Storage
options are a checked-key set plus separate accepted-result output, or a
key-to-state table distinguishing rejected, accepted-threshold-only, and
accepted-with-exact-count results. A threshold-only acceptance must not
masquerade as an exact count.

Cache validity is tied to the source-index snapshot, fingerprint semantics,
K, and output mode. In particular, a rejection at a previous K is not reusable
at a lower K without additional evidence. Record these settings in persistent
cache/checkpoint metadata.

For a small prototype, an exact in-RAM table is appropriate if it fits a
measured budget. Count distinct checked pairs, rejection/acceptance rates,
bytes per entry, and lookup/insert costs. Do not assume the accepted-pair
count predicts memory: rejected pairs also need history.

An unrestricted global cache is not assumed to fit Bokmal. For its
34,397,813 documents there are 591,604,752,392,578 possible unordered pairs:
even a dense triangular one-bit-per-pair table needs about 73.95 TB. An
eight-byte key for every possible pair would need about 4.73 PB before any
hash-table overhead. These are worst-case complete-pair estimates, not measured
candidate counts.

More concretely, the largest recorded Bokmal posting list has 796,600
documents. That list alone produces 317,285,381,700 pair attempts before
additional pruning, and representing each of those distinct pairs as an
eight-byte key would require about 2.54 TB before table overhead. A visited
cache avoids repeated verification but not the quadratic pair enumeration
and cache lookups.

Before a full run, choose a bounded global-state strategy:

- an exact disk-backed pair index with a bounded RAM front cache and batched
  reads/writes; benchmark it because random lookup/insert I/O may dominate;
- or externally sort/de-duplicate bounded chunks of generated pair keys,
  merge them, and verify each unique pair once. This avoids online visited
  lookups but changes verification scheduling and can require large temporary
  storage; global rare-first discovery no longer implies immediate rare-first
  verification.

The persistent backend and memory/disk budgets are not yet selected. If an
in-RAM prototype reaches its configured limit, spill to a configured exact
backend or stop with a clear error; do not silently forget checked keys.
A bounded LRU cache alone is not global de-duplication. A Bloom filter may
accelerate lookups but a positive result must be confirmed by exact state;
discarding on a Bloom-filter false positive could miss a qualifying pair.

For parallel verification, reserve/claim new keys atomically or route each key
to one deterministic owner before scheduling it. Distinguish in-flight work
from completed checks, and make abandoned in-flight work retryable after a
crash. Persist completion and accepted output consistently so a cache entry
does not silently suppress a result that was never committed.

##### Deterministic disk routing to evaluate

Knowing `N` and the canonical `(a, b)` permits direct routing; a new pair does
not need to search every disk file. Evaluate these layouts only if global
checked-pair state is needed by the chosen scheduler:

- **Range tiles:** choose a document block size `B` and route the pair to
  `(a // B, b // B)`. Only upper-triangular tiles are possible because `a < b`.
  Create nonempty tiles lazily. Within a tile, use sorted keys or row-indexed
  sorted `b` values, avoiding repetition of `a` where useful.
- **Direct triangular addressing:** the zero-based position of a pair is
  `p = a * (2*N - a - 1) // 2 + (b - a - 1)`. A page/subfile is determined
  by `p // pairs_per_shard`, and the local position by `p % pairs_per_shard`.
  This supports direct exact membership bits without a key lookup. Use wide
  arithmetic and test row boundaries and maximum supported IDs.

A fully allocated triangular bitmap still has the worst-case size described
above. Lazy sparse pages, row-compressed bitmaps, or compressed containers may
reduce physical storage for suitable checked-pair patterns; they are not a
guarantee that arbitrary pair history fits RAM or disk. Cap open files and
measure tile/page occupancy, skew, and metadata overhead.

Keep membership of completed checks separate from accepted counts if this
reduces storage: rejected pairs need a checked bit, not necessarily a stored
exact count. Preserve the threshold/mode/source validity contract and
in-flight/completed distinction.

Updates to sorted data must be buffered. Search pending in-memory keys and
every relevant immutable run in the routed shard; flush sorted unique batches
and compact runs in bounded memory. Routing to one shard does not mean one
physical read, and random lookups/writes may remain the bottleneck. Use
per-run min/max metadata, sparse search indexes, or confirmed Bloom-filter
front checks where measurements justify them.

Compare a simple prototype with a mature disk-backed index before building
a custom storage engine. Persist the chosen block/page geometry and source
document count so routing is deterministic on restart. Range routing can be
skewed; measure it against hash partitioning if necessary.

#### Scheduling option B: smaller-to-larger document probing

This is a different execution order, not a ruled-out or secondary product
choice. Mann et al. process probing sets in increasing set size. Use
`(len(E(d)), original_doc_id)` as a deterministic join rank for the
smaller-to-larger experiment, while preserving original document IDs.

For a document `a`, visit its eligible fingerprints and retrieve their derived
posting lists. Visit fingerprints in rare-first order within that document,
using a hash-to-rank lookup or equivalent derived view. The original forward
array remains hash-sorted. Include lookup storage and I/O in the benchmark;
the rare-first fingerprint array itself does not support numeric hash search.

Then:

1. Give each unordered pair exactly one owner. In the smaller-to-larger
   self-join, probe `a` against earlier-ranked documents `b`; omit self-pairs.
   The alternative numeric-ID baseline uses `b > a`. Choose one consistent
   rule for a run, not a mixture of both. Numeric-ID posting lists permit
   numeric range search only for the numeric-ID rule; a rank-based rule must
   filter ranks or use a separately rank-ordered posting representation.
2. Use a dense `uint32` marker array indexed by candidate document ID, plus a
   nonzero epoch for the current probe. A different probe uses a new epoch
   rather than clearing the entire array.
3. On first discovery of `b` in that probe, mark/schedule it once. Run the full
   threshold/count verifier once, immediately or through a bounded candidate
   block. Later fingerprint hits for the same `(a, b)` must not schedule it again,
   whether the completed verification accepts or rejects it.
4. Stream accepted pairs to bounded output chunks. Do not retain all candidates,
   all accepted pairs, or historical rejected pairs in RAM.

For Bokmal, one marker array needs 137,591,252 bytes, about 137.6 MB per worker,
plus the bounded candidate block, counters, and verifier state. Read-only
indexes can be shared/memory-mapped; each concurrently active worker needs its
own mutable markers. Cap worker count using the complete memory budget.
Handle epoch wraparound by resetting markers at a safe probe boundary.

This eliminates repeated full verification, not necessarily repeated visits
to a document ID in different posting lists. Safe prefix filtering is still
needed if posting-list traversal dominates.

The marker is safe for one-time full verification from the start of both
sets. Do not permanently reject a pair based only on partial evidence that
later fingerprints could change. If a future algorithm accumulates prefix
evidence, keep that candidate state distinct from a completed verification.

Normalize emitted pairs back to `(min(original_id), max(original_id))`.
Probe length/rank is not a replacement for fingerprint identity or source IDs.

RAM markers are not restart checkpoints. Persist completion at committed probe
partition boundaries, with corresponding output chunks, so resuming does not
silently duplicate or omit pairs. Unfinished partitions must have an explicit
restart/temporary-output policy.

#### Merge verification: adapt Algorithm 1

Algorithm 1 in Mann et al. takes sorted sets, a required overlap, and optional
already-accounted prefix overlap/positions. It returns whether the threshold
is met, stopping on early success or impossibility.

Use the same principle with `t = K`. Initially use the complete, unique,
hash-sorted forward lists with `i = j = matches = 0`. At any scan position:

```text
maximum_possible_overlap = matches + min(len(A) - i, len(B) - j)
```

- Reject when this upper bound is **less than K**, or either complete set has
  fewer than K elements. Equality with K is not grounds for rejection.
- In `threshold_only` mode, accept as soon as `matches >= K`. Report a boolean
  or explicit qualification/lower-bound evidence, not a fabricated exact count.
- In `exact_counts` mode, early rejection is allowed, but accepted pairs must
  obtain the final count, not necessarily visit every remaining array element.
- Advance both pointers on equal hash values and only the pointer with the
  smaller hash otherwise, counting each unique fingerprint once.

**Exact terminal conditions:** stop when either list is exhausted. Also, while
both remaining suffixes are nonempty, stop with the exact count accumulated so
far if `A[-1] < B[j]` or `B[-1] < A[i]`. Sorted order proves that the remaining
suffixes have no intersection. For example, if A's maximum is 15 and B's current
value is 18, none of B's remaining values can match A. This applies even if
several values remain unread in A. The comparisons must be strict: equality
can still contribute a match. Guard empty lists/pointer bounds before reading
current or final elements. Cache endpoint values and benchmark the placement
of these checks rather than assuming extra checks on every comparison are free.

Thus exact-count mode may terminate as soon as the remaining overlap is
provably zero; it must not terminate solely because K matches have been found.

The compared sets contain the same kind of values and use the same numeric
hash ordering, so equal fingerprints align during the merge. They need not
occupy equal array indices: `[3, 9, 15]` and `[1, 2, 9, 18]` share hash 9 at
different indices. "Positions" in Algorithm 1 means these independent merge
indices, not source-text positions.

A nonzero initial match count and nonzero starting pointers may be reused only
when the generator proves the exact prefix overlap and the suffix bounds in
the same element order. Rarity-prefix positions cannot be passed as numeric
hash-order positions. Start from zero until a tested optimization establishes
that invariant.

**Agreed first implementation:** use Numba in nopython mode
(`@njit(cache=True, nogil=True)`), which is already a project dependency.
Compile candidate/probe batches and the inner verification loop, not just a
tiny function called from Python for every pair. Use typed contiguous array
slices/CSR offsets, bounded buffers, and worker-local mutable state; avoid
per-pair Python objects and intersection-array allocations.

Warm up JIT compilation outside timings and profile comparisons, candidate
generation, memory-map page faults, and I/O separately. Numba releases the GIL
but does not itself make a shared marker array safe or run work in parallel.
Consider C/C++ only if profiling identifies a remaining kernel/binding issue
or a measured benefit from specialized compiled code. Do not assume changing
language fixes posting-list volume or random forward-index I/O.

#### Safe candidate filters

For an overlap threshold K, use only **provably safe** pruning. Examples:

- A document with fewer than K eligible distinct fingerprints cannot qualify.
- A pair can be discarded when its current overlap plus the maximum possible
  remaining overlap is less than K.
- A high-frequency fingerprint may be processed later or handled in a
  different execution path, but simply dropping it is not exact: a pair may
  reach K only because of common fingerprints.

#### Optional common-fingerprint candidate cap

Keep all `df >= 2` fingerprints in the sidecar. A separately configured
candidate-generation cap may stop the rare-first traversal when `df(f) > cap`.
Because records are sorted by ascending df, all remaining records then exceed
the cap. This is an explicit non-exhaustive/approximate candidate mode.

In that mode, a pair found through an allowed fingerprint is still verified
against its complete original forward lists, including fingerprints above
the cap. Therefore its accepted exact overlap count remains exact, but a
qualifying pair sharing only above-cap fingerprints can be missed entirely.
Threshold-only and exact-count verification must apply the same candidate
cap when comparing their pair sets.

This differs from the current inverted pair-counting implementation's cap,
which drops contributions from common fingerprints and can undercount pairs.
Name/document the new cap as a candidate-generation cap, and report its value,
skipped fingerprint/list volume, and non-exhaustive status. Never present it
as a lossless optimization of the uncapped join.

The product motivation is copyright-related shared-passage discovery, so a
common-fingerprint cap is a deliberate way to reduce boilerplate and
non-discriminative candidate work. It is not evidence that common material
cannot be protected: popular copied passages, syndicated articles, and widely
redistributed works may also have high frequency.

Use rare fingerprints as candidate anchors where effective, then include
common fingerprints and source positions in full verification. Compare
absolute df caps with relative prevalence `df / N` across the four differently
sized corpora. Evaluate caps against known shared passages and boilerplate
negatives, recording both runtime reduction and missed relevant matches.
An IDF-like ranking experiment would be a separate score, not a silent change
to the fixed overlap rule.

With the current limited hash space, collisions can also inflate a hash
bucket's df. A high-df hash is not always evidence of a genuinely common
phrase. Connect the cap study to the collision experiment below.

Report matched segments and evidence, not a legal finding of infringement:
fingerprint overlap alone does not establish copying, ownership, licensing,
or whether a particular reuse is unlawful.

#### Prefix filtering and output correctness

An optional rank-ordered forward view also allows a prefix-filter experiment.
For fixed overlap K and any eligible set `E` with `len(E) >= K`, the prefix
length is `len(E) - K + 1`. Under a shared total element order, two sets with
overlap at least K must have intersecting prefixes; see the paper's Section 2.1.
Test this bound and the ownership/indexing scheme before implementation.
Jaccard set-size and prefix bounds are not interchangeable with this rule.

In uncapped exact candidate mode, both output modes must match the baseline
qualifying pair set. The
`exact_counts` mode must also match its counts. A threshold-only result must
not be written into an exact-count output schema as if K were the final count.
Top-k retrieval can remain a separate approximate mode; it should not silently
become the implementation of a complete join.

### Stage 4 — candidate graph, document groups, and passage verification

#### Candidate graph and document-group selection

Build an undirected graph whose nodes are documents and whose edges are the
qualifying overlap pairs from Stage 3. Preserve internal/external document
identity and the join's K, source snapshot, output mode, and cap policy.

In exact-count mode the edge may be weighted by exact shared-fingerprint count.
Threshold-only mode initially supplies an unweighted qualification edge,
unless exact counts are subsequently computed. Do not invent a weight of K
and present it as an exact similarity count.

The agreed initial document groups are **connected components, not cliques**:
if A-B and B-C qualify but A-C does not, all three can belong to one group.
Membership does not assert that every pair overlaps or that all members share
the same passage. Verify existing candidate edges, not all combinations of
documents within a component.

Expose document-group selection independently of passage detection:

- `largest`: retain the eligible component with the most document nodes;
- `all_above_min`: retain every component with at least
  `min_document_cluster_size` nodes.

Make the minimum size, eligibility rule, and deterministic tie-break explicit;
size is measured in documents, not edges, fingerprint count, or passage score.
Report excluded groups/documents/edges as outside the verification scope,
not as rejected matches. Isolated source documents should be accounted for
but need not be displayed as candidate groups.

For large runs, compute components from streamed edge data with bounded
component-label/union-find state rather than loading an unbounded NetworkX
graph. Reuse existing graph/plot helpers for bounded selected groups. Do not
use a top-weighted-node visualization cap before component selection: it
changes graph connectivity and may change which component is largest.

#### Explicit-pair verification with FingerprintChainDetector

Load the selected documents through a source-text loader keyed by their
original IDs, with bounded caching. Retrieve positional evidence from the
original index, not the position-free screening sidecar.

Reuse/refactor [FingerprintChainDetector](./findmytext/detectors.py) rather
than writing an unrelated chain-clustering algorithm. Its current
`_score_doc` already takes the maximum surviving cluster score, but its public
query path retrieves its own top-k candidates and returns containment scores,
not passage-cluster evidence.

Provide an explicit-pair path for the graph-selected edges:

- accept the already selected document IDs and verify each edge exactly once;
- do not reapply top-k retrieval or silently omit a selected pair;
- cluster shared fingerprint occurrence pairs using the detector's positional
  method, tolerances, and configured support filters;
- select the **highest-scoring surviving passage cluster**;
- retain it only when its score meets `min_passage_score`, along with explicit
  minimum support/span requirements; report no surviving qualifying passage
  as a completed rejection;
- return the selected cluster's score, score definition, distinct fingerprint
  support, aligned token/character spans, positional evidence, and parameters.

The initial passage policy is best-cluster-only, not all surviving passage
clusters. This is distinct from the graph option to select one document
component or several components.

Keep three thresholds separate: overlap K, minimum document-component size,
and passage acceptance. Also distinguish detector raw/rarity-weighted cluster
score from normalized containment probability; choose and record the unit of
`min_passage_score` before implementation. Do not silently compare thresholds
in different score units.

Repeated occurrences may provide several alignment hypotheses, but one hash
must not inflate distinct-fingerprint support through many occurrence pairs.
Audit the existing detector's support/noise filtering when exposing passage
results and preserve existing public detector behavior for unrelated callers.
Source-text load failures and detector errors must be reported separately
from a genuine no-passage result.

#### Compare and display document-group evidence

Keep candidate and passage-verified graphs distinct. Mark edge status as
pending, passage-supported, rejected, or error, with the relevant overlap
and detector evidence. A passage-verified graph can drop rejected edges and
recompute components within the selected scope; a candidate component may
split after verification. Do not claim a globally largest verified component
if only a subset of candidate components was examined.

For each selected group, display:

- document nodes with IDs, titles/source links where available, lengths, and
  candidate/verification metadata;
- candidate edges distinguished from passage-supported edges, showing
  overlap count when known and the separate passage score;
- a selectable pair view with side-by-side text, the retained passage
  highlighted in both documents, spans, support, and score;
- a group summary of which documents share which aligned passages, allowing
  comparison of selected groups without assuming one passage is common to
  every member or inferring a copying direction from graph connectivity.

Reuse [visualization.py](./findmytext/visualization.py) graph and alignment
helpers where appropriate. Existing `locate_shared_passage` also depends on
top-k retrieval, so it must not be used unchanged as exhaustive verification
of supplied graph edges. Inspect [multi_document.py](./findmytext/multi_document.py)
for optional passage-support comparison among selected documents; do not
require an all-member common passage merely because a graph is connected.

Produce bounded interactive HTML group/graph views with drill-down passage
evidence. Pagination or sampled layouts are rendering controls only and must
not silently change the analyzed graph. Persist group membership, selection
settings, pair statuses, and passage evidence so inspection is reproducible.

## Iterative implementation: Nynorsk first

Do not implement the whole join, disk engine, parallel execution, and new
hashing pipeline in one step. Deliver and validate these increments:

1. **Sidecar build command:** derive the Nynorsk `df >= 2` position-free index,
   print/persist removal and storage statistics, and verify the source is
   unchanged. Include bounded memory and a clear completion manifest.
2. **Numba verifier:** implement both output modes and terminal conditions;
   validate against ordinary set intersection on small fixtures and randomized
   unique sorted sets. Measure compilation separately from warmed execution.
3. **Bounded scheduling comparison:** run both schedulers on the same fixed,
   reproducible Nynorsk document subset with the same K, cap policy, output
   mode, and document-frequency definition. Compare pair sets/counts with the
   reference, then measure total time, I/O, and state size.
4. **Full Nynorsk run:** use the measured scheduler/storage choice, explicit
   RAM and temporary-disk budgets, incremental output, and restartable
   checkpoints. First inspect bounded candidate graph groups and explicit
   pair passage evidence, then expand graph selection/display before scaling.
5. **Later scaling:** profile parallel execution, then evaluate Icelandic,
   Latvian, and Bokmal with the same correctness/quality contracts.

Define subset scope explicitly. A benchmark among selected document IDs must
restrict both schedulers to those IDs, not compare a partial fingerprint scan
with a complete document join. Record whether df comes from full Nynorsk or
is recomputed on the subset. For the first controlled scheduler comparison,
use full-Nynorsk df consistently and a fixed subset manifest.

Show periodic progress with completed work units, pair attempts, distinct
checked pairs, duplicate discoveries skipped, qualified pairs, verifier work,
cache/buffer sizes, elapsed time, and throughput. Document-driven progress
should include completed probe documents; fingerprint-driven progress should
include completed fingerprint records and pair work within the current record.
Rare-first fingerprint percentage alone is misleading because the common tail
may dominate runtime. Persist committed progress, not just console messages.

Before the first meaningful run, specify K, output mode, candidate cap or
explicit no-cap choice, RAM budget, temporary-disk budget, and output directory.
Keep known aligned-copy positives, scattered-overlap negatives, and common
boilerplate cases as quality fixtures. Detailed text-segment acceptance can be
refined after the index/verifier structure is working, but must be explicit
before interpreting results as relevant shared passages.

## Parallelization: explore after the single-worker structure

Do not initially mix concurrency bugs with index-layout/verifier development.
Keep natural partitions and worker-local state available, then profile:

- **Sidecar construction:** independent fingerprint batches can de-duplicate
  and count in parallel. Frequency ordering and output offsets require
  coordinated sort/prefix-sum phases before disjoint writes. Storage bandwidth
  may limit speedup.
- **Document-driven joins:** partition probe documents/ranks among workers,
  with one deterministic owner per unordered pair, read-only shared indexes,
  and private epoch markers/candidate buffers. This avoids a shared global
  checked-pair table but multiplies per-worker working memory.
- **Fingerprint-driven joins:** parallel discovery can rediscover the same
  pair concurrently. Route canonical keys to one owner or atomically claim
  them before verification. A coordinator can batch cache lookups and submit
  fresh pair batches to verifier workers; do not make a disk lookup from Python
  for every comparison in a compiled kernel.
- **Overlap verification and positional verification:** independent candidate
  batches are natural parallel work once pair ownership is resolved. Use
  Numba's GIL-free execution with private mutable state and bounded queues;
  shared memory maps do not make shared writes safe.

Benchmark worker counts with warmed kernels and realistic storage/cache
conditions. Track wall time, CPU use, I/O/page faults, memory bandwidth, queue
backpressure, and peak RAM; stop increasing workers if I/O or memory contention
reduces throughput. Preserve output equality and restart correctness. Disk
shards/pages must have explicit write ownership or synchronization, including
different membership bits that share a byte/page.

## Separate experiment: wider hashes and collision effects

**Implementation update:** `rolling64-v1` is the only production method.
Legacy hashing has been removed, and readers reject old/missing hash metadata
with a rebuild message. Existing legacy data files remain untouched; they
cannot be converted by changing metadata.

The completed full Latvian study observed zero rolling64 collisions among
897,868,933 distinct selected k-grams. The full Nynorsk wide-hash study used
BLAKE2b-64; rolling64 was checked on its matched 50,000-document sample.
The historical comparisons, storage findings and complete-build profiling
are frozen in [experiments/HASHING_FINDINGS.md](./experiments/HASHING_FINDINGS.md).
Legacy-dependent experiment runs are retired by choice. This migration does
not implement the similarity-join sidecar or remove either posting layout.

The retired legacy implementation reduced token and rolling hashes modulo
`2**31 - 1`, about 2.147 billion possibilities. It is not a full unsigned
32-bit hash; fingerprints are already stored in uint64 arrays. Wider storage
alone, or hashing an existing 31-bit value again, cannot recover distinctions
already lost to collisions.

The original experimental design was as follows (preserved for context, not
as instructions to rerun legacy collection with the current code):

1. **Isolate collisions on a fixed selection.** For reproducible samples from
   all four corpora, recover the original normalized k-grams at the positions
   selected by current winnowing. Compare each stored hash with an independent
   wide signature/original k-gram. Measure different k-grams sharing a stored
   hash, artificial document sharing, and bucket-df inflation.
2. **Evaluate the complete alternative pipeline.** Recompute token/k-gram
   hashes and winnowing from the same source documents with a validated
   64-bit design, keeping tokenization, k, window size, and document filters
   controlled. Do not simply enlarge the modulus: arithmetic/overflow and
   effective hash entropy must be reviewed and tested end to end.
3. **Compare all four corpora incrementally, starting with Nynorsk.** Measure
   unique hash values, singleton counts/fractions, df distribution,
   candidate/verification work, shared-passage quality, build/runtime costs,
   and storage. Use matched samples before committing to full rebuilds.

Rehashing changes window minima and thus which k-grams winnowing selects.
Changes in singleton fraction after a full rebuild are not solely a collision
measurement; keep the fixed-selection diagnostic separate from the end-to-end
comparison. Recovering original k-grams requires source text plus the original
tokenization/position convention; existing hash arrays alone are insufficient.

Record hash algorithm/version, effective width, seed/base/modulus as relevant,
and tokenization/winnowing settings in new index metadata. Reject incompatible
index comparisons or mixed old/new fingerprints. Test fingerprint/rank array
widths and sentinel values at the potentially larger distinct-value counts,
and do not modify the current hash implementation merely as part of the
screening-index work.

## Research ideas to evaluate

### Mann, Augsten, and Bouros (PVLDB 2016)

The paper evaluates exact set-similarity-join algorithms and emphasizes the
filter–verification framework. Its useful practical lessons here are:

- Prefix filtering is a central, well-established way to reduce candidate
  pairs for threshold set joins, provided the sets use a shared global element
  order and the prefix bound matches the chosen similarity threshold.
- Verification can be cheap when it is implemented as a tight merge-style
  scan; extra elaborate filters are not automatically faster. Measure the
  whole pipeline, not just how many pairs a filter removes.
- Keep the baseline simple and compare candidate-generation cost against
  verification cost before adding more sophisticated filters.
- Sections 2.2 and 3 explicitly use increasing set-size order for posting
  lists and probes; this motivates a benchmark, not Jaccard length-ratio
  pruning for our fixed-overlap join.
- Algorithm 1 is directly relevant to threshold verification: it terminates
  once enough overlap is found or the remaining scan cannot reach the
  threshold. It does not return the complete overlap count on early success.
- Their candidate collection is per probing set, with temporary state reset
  during verification; it is not a global history of all checked pairs.

For this project the measure is already chosen: **absolute overlap count**,
not Jaccard. Use the paper's filter-verification framework and derive/test
filter bounds for fixed K. This does not mean the repository already
implements AllPairs, PPJoin, or another evaluated algorithm. Set-intersection
verification and final positional text verification remain separate steps.

Reference: [An Empirical Evaluation of Set Similarity Join Techniques —
Mann, Augsten, and Bouros, PVLDB 2016](https://ssjoin.dbresearch.uni-salzburg.at/)
([PDF](https://ssjoin.dbresearch.uni-salzburg.at/downloads/projects/ssjoin/p636-mann.pdf)).

### Filter-and-Verification Trees (2025)

The R-S set-similarity-join paper proposes Filter-and-Verification Trees (FVT)
and a more compact Linear FVT (LFVT). The useful direction is to organize
shared prefixes/sets so filtering and overlap verification happen during tree
traversal, rather than first materializing a potentially huge candidate list.
It also motivates compact, sequential data layouts and load-aware partitioning
for parallel execution.

For this project, treat FVT/LFVT as a **later alternative to benchmark**, not
as a drop-in replacement:

- First establish that exact inverted/prefix filtering is insufficient.
- Prototype an in-memory FVT/LFVT on a bounded sample of ranked document
  fingerprint sets.
- Prove its pruning rules for our chosen threshold (especially if K is an
  overlap count rather than Jaccard).
- Compare tree build/memory cost and traversal time with the current
  disk-backed posting lists. Include skewed-frequency workloads; a compact
  tree may help repeated prefixes but can also cost too much to build.
- Consider load-aware work partitioning only if a single-machine join becomes
  the bottleneck. The paper's MapReduce architecture is not an immediate
  requirement for this repository.

Reference: [Efficient Candidate-Free R-S Set Similarity Joins with
Filter-and-Verification Trees on MapReduce](https://arxiv.org/abs/2506.03893)
([HTML](https://arxiv.org/html/2506.03893v3)).

## Milestones

### 0. Agree on the contract

- [x] Choose absolute overlap count, not Jaccard, for candidate screening.
- [ ] Choose the positive integer overlap threshold K.
- [ ] Specify text-segment acceptance: minimum span, coverage, positional
  tolerance, and treatment of repeated occurrences.
- [ ] Specify whether output is exhaustive, top-k, or both (as separate modes).
- [x] Preserve normal index construction and the original positional index.
- [x] Build a separate position-free inverted screening index with `df >= 2`,
  unique sorted document IDs, and fingerprint records ordered by `(df, hash)`.
- [x] Preserve numeric-hash ordering within the original forward index.
- [x] Support explicit `threshold_only` and `exact_counts` modes with the same
  exhaustive qualifying pair set in uncapped exact candidate mode.
- [x] Keep fingerprint-driven and smaller-to-larger document-driven scheduling
  open, sharing the verifier and choosing by measured end-to-end performance.
- [x] Use Numba for the first compiled overlap/candidate kernels.
- [x] Start with incremental Nynorsk deliverables and postpone parallel tuning
  until the single-worker structure is validated.
- [x] Plan a separate, controlled 64-bit collision/pipeline study for all four
  corpora without changing the current indexes.
- [x] Group the overlap graph by connected components, with largest-component
  and all-above-minimum-size document-group selection.
- [x] Verify selected candidate edges using FingerprintChainDetector and keep
  each pair's highest-scoring surviving passage cluster above its threshold.
- [ ] Choose the default overlap-output mode and separate output schemas.
- [ ] Choose a checked-pair backend, RAM/disk budgets, and whether to enable an
  explicitly non-exhaustive candidate-frequency cap.

### 1. Measure and lock down the baseline

- [ ] Add small correctness fixtures covering duplicate occurrences, singleton
  fingerprints, fingerprints in exactly two documents, empty documents, and
  pairs exactly at K.
- [ ] Capture exact pairs/counts from the current inverted method.
- [ ] Record per-fingerprint `df`, posting-entry volume, candidate-pair updates,
  peak temporary storage, and elapsed time on representative data.
- [ ] Record verification calls, repeated discoveries skipped, and forward
  elements inspected separately from pair-count contributions.

### 2. Build the screening sidecar and prototype exact candidate pruning

- [ ] Derive position-free unique document lists with `df >= 2`.
- [ ] Sort derived fingerprint records by `(df, hash)` and preserve source IDs.
- [ ] Verify the source indexes remain unchanged, singletons are absent only
  from the sidecar, and exact pair overlaps are preserved.
- [ ] Print and persist build-removal/de-duplication/storage accounting, phase
  progress, and eligible document lengths.
- [ ] Implement a first exact candidate strategy and compare output pair sets
  and counts against Milestone 1.
- [ ] Prototype global rare-first fingerprint traversal, bounded pair blocks,
  and an exact checked-pair table storing both accepted and rejected decisions.
- [ ] Measure cache cardinality/bytes/hit rate and pair-attempt volume before
  selecting a full-corpus global-state backend.
- [ ] Benchmark smaller-to-larger document-owned probing with per-worker epoch
  markers against fingerprint-driven execution, under identical semantics.
- [ ] If global disk state is needed, benchmark deterministic range-tile or
  triangular-page routing, compression, buffered updates, and exact lookups.
- [ ] Implement/test a Numba merge verifier for both output modes, including
  early success, impossible remaining overlap, exact-K boundaries, empty sets,
  asymmetric set sizes, disjoint suffix ranges, equal endpoint values, and
  exact stopping when a list is exhausted.
- [ ] Test repeated discoveries of accepted and rejected pairs across distant
  fingerprint records, canonical key encoding, concurrent claims/retries, and
  cache validity for source/K/mode changes.
- [ ] For the document-driven alternative, test epoch rollover and deterministic
  pair ownership.
- [ ] Test capped candidate generation separately: discovered-pair exact counts
  include above-cap fingerprints; above-cap-only qualifying pairs may be absent
  and the output must declare that limitation.
- [ ] Measure the agreed rare-first order; do not discard frequent fingerprints
  in exact mode.
- [ ] Benchmark original-ID versus eligible-set-length posting/probe order
  without applying Jaccard length bounds or changing source IDs.
- [ ] Measure candidate counts, verification work, I/O, peak memory, and total
  end-to-end time.

### 3. Add candidate graph selection and positional passage verification

- [ ] Derive connected document components from candidate edges and persist
  reproducible membership/selection metadata.
- [ ] Implement largest-eligible-component and all-above-minimum-size options,
  testing exact size boundaries, ties, empty graphs, and isolated documents.
- [ ] Test A-B/B-C connectivity without A-C; do not require clique membership
  or expand each component into an unrequested all-pairs verification task.
- [ ] Expose/reuse explicit-pair FingerprintChainDetector clustering without
  top-k candidate loss, returning best qualifying passage evidence.
- [ ] Define passage score units and minimum support/span independently of
  overlap K and graph-component size.
- [ ] Return aligned positions/spans, distinct fingerprint support, score,
  parameters, and pair status; test repeated occurrences and load errors.
- [ ] Test multiple surviving passage clusters, retaining only the
  highest-scoring qualifying one, and test no qualifying cluster.
- [ ] Test scattered shared fingerprints versus genuinely aligned segments.
- [ ] Build separate candidate and verified graph views, including components
  that split when unsupported edges are removed.
- [ ] Display bounded document graphs and side-by-side highlighted passages,
  plus group-level passage-support comparisons.
- [ ] Tune thresholds on labeled or manually reviewed examples.

### 4. Evaluate advanced join structures

- [ ] Implement a prefix-filter prototype if the ranked-list approach leaves
  substantial candidate work.
- [ ] If using rarity-ordered prefixes, derive a separate forward rank view
  without modifying the original hash-sorted forward index.
- [ ] Evaluate FVT/LFVT on a bounded, representative sample if candidate
  materialization or repeated-prefix processing remains a bottleneck.
- [ ] Consider layout/parallel partitioning improvements only after measuring
  build costs, skew, and end-to-end throughput.
- [ ] Explore the parallel work units above after the single-worker Nynorsk
  run, preserving pair ownership and output equality.
- [ ] Run matched-sample collision diagnostics and a separate wide-hash
  end-to-end experiment before considering full rebuilds.

## Evaluation and acceptance criteria

Correctness takes priority over speed:

- Both overlap-output modes in uncapped exact candidate mode must match the
  baseline qualifying pair set on fixtures and manageable corpora; exact-count
  mode must additionally match accepted-pair counts. Threshold-only mode must
  not advertise partial counts as exact.
- In a failure-free run, a pair discovered through several fingerprints must
  undergo at most one full overlap verification, including rejected candidates.
  Use global checked-pair state for fingerprint-driven execution and per-probe
  ownership/markers for the document-driven alternative. Test parallel claims
  and checkpoint/retry boundaries explicitly. Crash recovery may recompute
  uncommitted work, but must not duplicate or omit committed accepted results.
- Every filter/prune must have a stated bound and a test for its boundary
  cases.
- Any approximate mode (frequency caps, sampling, top-k) must be explicitly
  named and report that it may omit qualifying pairs.
- Position verification must reject synthetic pairs with sufficient scattered
  overlap but no coherent segment, while accepting known planted segments.
- Connected-component membership must not imply pairwise or all-member
  passage similarity. Graph selection must preserve the intended scope and
  supplied candidate edges must not disappear through detector top-k limits.
- Passage output must contain only the best surviving qualifying cluster per
  pair under the documented score, while keeping graph-size and passage
  thresholds separate. Visualization limits must not change analysis results.

For performance, report a table by dataset/configuration including:

| Metric | Why it matters |
|---|---|
| Exact candidate pairs and final accepted pairs | Quantifies each stage's reduction |
| Pair updates and high-frequency-list work | Reveals the source of join cost |
| Verification calls and position evidence examined | Separates filter cost from verification cost |
| Duplicate discoveries skipped and forward elements inspected | Tests pair de-duplication and early-stop effectiveness |
| Global checked-state bytes/cardinality, cache hit rate, and lookup/write time | Tests feasibility of fingerprint-driven pair history |
| Buffer memory, alternative per-worker markers, and committed progress | Bounds memory and makes restart semantics explicit |
| Wall time by stage | Avoids optimizing one stage at the expense of the whole pipeline |
| Peak memory and temporary bytes written | Ensures bounded-memory behavior scales |
| Recall/precision for any approximate mode | Makes quality/performance tradeoffs explicit |
| Physical shard/page bytes, occupancy, and lookup/write I/O | Tests whether deterministic disk routing actually helps |
| Progress/checkpoint counters and committed output | Makes incremental runs observable and restartable |
| Fixed-selection collision rate versus full rehash changes | Separates collisions from changed winnowing selection |
| Candidate/verified components, selected nodes/edges, and pair statuses | Separates graph selection from passage support |
| Source loading, detector time, and best-passage support/span | Measures evidence cost and quality |

Do not adopt a more complex filter/tree solely because it reduces candidate
count. It should improve total elapsed time and resource use without breaking
the exact-mode contract.

## Questions for our discussion

1. What positive integer K should the absolute-overlap screening use, and
   should the product expose candidates, verified text segments, or both?
2. What constitutes a coherent text match: a minimum token span, fingerprint
   count, coverage, or a combination? Should offset drift be allowed?
3. Should the initial exact join produce every qualifying pair, or is top-k
   per document sufficient for the intended workflow?
4. What build-time and storage budget is available for the agreed derived
   position-free screening index and any optional rarity-ordered forward view?
5. Which known aligned matches and boilerplate negatives should accompany the
   agreed Nynorsk-first benchmark?
6. Should the next implementation stop after exact ranked-prefix filtering, or
   should FVT/LFVT be an explicit early experiment?
7. Which overlap-output mode should be the default, and what CPU/RAM/storage
   budget should bound the first Nynorsk scheduler comparison?
8. Which global checked-pair backend should be used beyond the in-RAM
   prototype, and what optional candidate-frequency cap is acceptable?
9. Which source-text samples and validated wide-hash design should be used for
   the separate four-corpus collision experiment?
10. What minimum document-component size, passage score definition/threshold,
    and support/span requirements should be used in the first graph-to-text
    inspection run?
