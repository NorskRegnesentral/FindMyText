# How we build the indexes

Presentation notes for the document-similarity research branch, 17 November 2026.

## The one-minute explanation

> We turn each document into a compact selection of 64-bit fingerprints of
> four-token sequences. We first build a positional inverted index, which
> tells us where each fingerprint occurs. From it, we derive a forward index
> for fast document-to-document comparison and a smaller overlap inverted
> index for finding candidate document pairs. The full indexes keep all
> selected fingerprints; only the overlap index drops fingerprints that
> cannot be shared between documents. Positions and original source text
> remain available for investigating the resulting matches.

## 1. The pipeline at a glance

```text
HPLT compressed JSONL shards
  |
  +-- stream, apply existing ingestion filters, retain source document IDs
  |
  +-- tokenize -> hash every 4-token sequence -> winnow with window 6
  |
  +-- accumulate occurrences; periodically flush sorted intermediate files
  |
  +-- one k-way merge across all positional intermediate streams
        |
        +-- full positional inverted index: fingerprint -> (document, position)
              |
              +-- full forward index: document -> sorted unique fingerprints
              |
              +-- overlap inverted index: fingerprint -> unique document IDs
                    keep df >= 2; order fingerprints by (df, numeric hash)

Separate source-data pass:
  raw HPLT records -> SQLite document store -> original text + complete metadata
```

The SQLite store is for retrieving source records, not another fingerprint
representation, and is built separately from the three indexes.

## 2. Read and filter the corpus

The [HPLT build command](../hplt_build.py) processes the local compressed shards
one at a time; documents are streamed rather than loading the corpus into RAM.
Fingerprint extraction can use worker processes.

For the established Nynorsk build:

- Input: five shards under
  `/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/nno_Latn`.
- Keep texts of **100 through 100,000 characters, inclusive**. Longer texts
  are skipped, not truncated.
- Preserve the existing ingestion filters, including the inherited exact-token
  content filter in [utils.py](../findmytext/utils.py).
- Keep original external document IDs; create compact internal integer IDs
  during the merge.

This produced **1,028,632 indexed documents**. It is not every raw record:
the separate source store contains 1,034,804 records, including filtered ones.
There is no new validated main-content/boilerplate extraction step here.

## 3. Turn text into fingerprints

The [winnower](../findmytext/winnower.py) performs these steps:

1. Remove punctuation with the existing normalization and split on whitespace.
   Letter case is preserved; there is no stemming or translation.
2. Form overlapping **4-token sequences**, also called 4-grams.
3. Hash the tokens and compute a rolling polynomial hash for every 4-gram
   using unsigned 64-bit arithmetic, followed by the fmix64 finalizer.
4. Slide a window over **six consecutive 4-gram hashes**. Select its minimum,
   choosing the rightmost occurrence on a tie. Emit a selection only when
   its position changes.
5. Retain the selected hash and its **zero-based token start position**.

Winnowing reduces the number of stored occurrences while preserving local
matching evidence. A selected fingerprint is not a word, and the window size
is not a number of documents. "Full index" below means all **selected**
fingerprints, not all pre-winnowing 4-grams.

The research indexes record the method as `rolling64-v1`. The separate
publication branch calls the algorithm `polynomial64-fmix64`; readers enforce
their own metadata contracts, so these labels must not be interchanged to
make an existing index load. Legacy 31-bit indexes require rebuilding from
source. The hash is non-cryptographic and not guaranteed collision-free.

## 4. Flush intermediate positional indexes

The [index builder](../findmytext/index_builder.py) accumulates
`fingerprint -> (document ID, token position)` occurrences in memory.

- The HPLT command defaults to flushing every **100,000 indexed documents**.
- Each flush writes a numbered gzip-compressed MsgPack intermediate file,
  sorted by numeric fingerprint value, with settings and a document-ID table.
- Any remaining documents produce a final intermediate file.
- Each completed input shard gets a completion manifest listing its outputs.

These are temporary extraction partitions, not separate final indexes.
The flush threshold controls accumulated documents, not a hard RAM limit:
document lengths and occurrence counts also matter.

## 5. Merge the positional files

The positional merge is **one k-way heap merge**, not a hierarchy of pairwise
merges and not one complete file after another:

1. Check that all intermediate files use compatible fingerprint settings.
2. Collect external document IDs and assign compact internal IDs.
3. Reader processes decompress sorted input streams concurrently.
4. A heap selects the smallest next fingerprint across the streams.
5. Combine that fingerprint's occurrences, remap document IDs, and write its
   final posting list once.

The result has numerically sorted unique UInt64 fingerprint keys and
interleaved `(document ID: UInt32, token position: UInt32)` postings:
**8 bytes per occurrence**.

| Files in the final index directory | Purpose |
| --- | --- |
| `fingerprints.npy`, `offsets.npy`, `lengths.npy` | Fingerprint directory: keys, byte offsets and occurrence counts |
| `postings.dat` | Positional occurrence lists |
| `doc_name_offsets.npy`, `doc_name_bytes.npy` | Mapping back to original external document IDs |
| `meta.json` | Fingerprinting and posting-layout settings |

This index answers: **"Where does this fingerprint occur?"** It retains
fingerprints occurring in only one document and repeated positions within
a document, so it remains useful for external queries and passage alignment.

## 6. Derive the full forward index

The merge builds the forward representation by default, using two batched
passes over the positional postings: first count unique memberships per
document, then place them in memory-mapped output arrays.

- `forward_offsets.npy` locates each document's slice.
- `forward_fingerprints.npy` stores its **numerically sorted, distinct UInt64
  fingerprints**, including singletons.
- Repeated occurrences of the same hash in one document become one membership.
- There are **no token positions** in this representation.

This index answers: **"Which fingerprints does document A contain?"**
Intersecting two sorted slices gives their exact distinct-hash overlap.
Forward offsets locate array slices; they are not source-text positions.

## 7. Derive the overlap inverted index

The [overlap builder](../findmytext/overlap_inverted.py) reads the positional
index without modifying it:

1. Deduplicate document IDs within each fingerprint's posting list.
2. Compute document frequency `df`: the number of **distinct documents**.
3. Keep the fingerprint exactly when **df >= 2**, including df=2.
4. Store ascending unique document IDs only; discard occurrence positions
   from this derived representation.
5. Sort fingerprint records by **(df, numeric hash)**, rarest first.

Unlike the positional merge, this ranking uses bounded sorted disk runs
merged **pairwise over successive generations**. The builder then writes
the final rarity-ordered arrays and publishes the completed sidecar.

| Files in the separate overlap directory | Purpose |
| --- | --- |
| `fingerprints.npy` | UInt64 keys in rarity order, not numeric-hash order |
| `offsets.npy`, `doc_ids.npy` | Posting boundaries and ascending unique UInt32 document IDs |
| `eligible_lengths.npy` | Number of retained fingerprints in each source document |
| `meta.json` | Source binding, ordering, counts, timings and storage accounting |

This index answers: **"Which documents could share fingerprints?"**
It requires its dedicated rarity-rank reader, not numeric-hash binary search.

**Why removing singletons is safe:** a fingerprint present in only one
document cannot be in any two-document intersection. Repeated occurrences
also cannot increase a distinct-hash overlap count. Thus this reduction
preserves every indexed document-pair intersection, not necessarily every
possible common passage in the original un-winnowed text.

## 8. Resume safely and validate

The build manifest binds the destination to source paths, sizes, modification
times, fingerprint settings and length filters. It is not a full-byte
cryptographic digest of the corpus.

- Matching completed extraction shards can be reused; unfinished shards are
  rebuilt on retry.
- The final positional/forward directory is renamed into place only after
  both representations finish.
- The overlap sidecar is published separately and checked against its source.
- Changed sources/settings and incompatible or untracked destinations produce
  explicit errors. Use a new destination; do not relabel old metadata.
- Do not run two builds against the same destination.

The [verification command](../experiments/indexing/verify_overlap_inverted.py)
checked full Nynorsk memberships, ordering, coverage, accounting and byte
counts. Additional checks compared 256 seeded pair intersections and source
fingerprints/positions for 20 documents across all five shards. Index
verification is not an exhaustive all-pairs join or a relevance assessment.

## 9. What we actually built for Nynorsk

The research rebuild was completed and verified on **23 October 2026**:

| Representation | Entries | Size |
| --- | --- | ---: |
| Positional inverted | 96,496,752 hashes; 159,523,198 occurrences | 3.024 GiB |
| Full forward | 153,553,490 unique document/fingerprint memberships | 1.152 GiB |
| Overlap inverted | 18,718,065 hashes; 75,774,803 document-ID entries | 0.569 GiB |

The overlap representation removes **77,778,687 singleton hashes (80.60%)**
and eliminates **5,969,708 repeated within-document occurrences** relative
to the positional input. No entries are removed from the two full indexes.
Sizes describe the respective representation, not intermediates or the
separate approximately 3.28 GiB source-record store.

To invoke the existing local build workflow from the repository root:

```bash
.venv/bin/python hplt_build.py nno_Latn hplt_nynorsk \
  --skip-download --overlap-inverted --nb-workers 8
```

This uses the default source-bound `findmytext_build` destination and reuses a
matching completed build. For a fresh build, explicitly pass `--build-dir`
with a new destination; existing indexes need not be rebuilt for presentation.

## 10. What happens after indexing?

Index construction does **not** itself compute all document pairs:

1. Use the overlap index and lossless rarity prefixes to discover candidates.
2. Verify candidates by intersecting the full forward sets; accept when
   the number of distinct shared fingerprints is at least **K**.
3. Stream exact results to disk with checkpoints. Compute K=5 once and
   filter accepted counts for K=10/20/30/50.
4. Form connected components from qualifying edges.
5. Inspect selected pairs using the positional index and retrieve original
   text/metadata from the separate [document store](../findmytext/document_store.py).

Full exact joins are complete for Nynorsk, Icelandic and Latvian; Bokmal's
full join is held by the memory preflight although its indexes are built.
Nynorsk group/passage examples are in the executed
[review notebook](../notebooks/nynorsk_group_review.ipynb).

**Presentation caveat:** exact fingerprint overlap is candidate evidence,
not proof of meaningful reuse or copying direction. Connected components
are not cliques or groups sharing one common passage. Templates, archives
and recommendation excerpts can create genuine but misleading overlap.
Human relevance labels and calibrated content-aware acceptance remain next
steps; see the [current plan](SIMILARITY_JOIN_PLAN.md) and
[three-domain review](../reports/nynorsk/three_domain_overlap_review.md).
