# Positional posting layout: interleaved-only decision

## Decision

As of 23 October 2026, the full positional inverted index uses **interleaved
postings only**. Columnar reader/writer support, layout-selection parameters
and CLI flags, and the columnar conversion/benchmark script have been removed.

This decision simplifies the implementation around the intended separation
between the full positional index and a document-only screening sidecar.
It is **not** a claim that interleaved is universally faster, and no new
columnar-versus-interleaved throughput comparison was run for this decision.
The measured 50k build profiles in [HASHING_FINDINGS.md](HASHING_FINDINGS.md)
used interleaved; those numbers do not establish a layout speed ranking.

## What the two layouts represented

Both positional formats stored the same logical occurrence records and used
8 payload bytes per occurrence:

```text
Interleaved:
    [doc_0:uint32, position_0:uint32]
    [doc_1:uint32, position_1:uint32]
    ...

Columnar (retired):
    [all document IDs:uint32]
    [all positions:uint32]
```

Each fingerprint is stored separately in the fingerprint directory, not
repeated inside every on-disk posting record. The directory has uint64 hash
values, uint64 byte offsets and uint32 posting-list lengths.

The forward index was never controlled by the positional layout option. It
still contains document offsets and each document's numerically sorted unique
uint64 fingerprints, without occurrence positions.

## Why columnar was explored

Candidate retrieval often needs document IDs but not positions. With columnar
storage, such a lookup could read 4 bytes per occurrence instead of the full
8-byte interleaved record. That is an I/O-volume advantage, not a guaranteed
2x runtime improvement: caching, copying, batch sizes and device behavior
also matter.

When both fields are needed, interleaved retrieves each posting list with one
contiguous read. The retired columnar reader read the document and position
regions separately. Supporting both introduced branches in building,
random reads, sequential iteration and forward-index construction, plus
conversion code and metadata/API choices.

## Intended architecture

Build and retain:

1. **Full positional inverted index:** all selected fingerprints, including
   singletons, with all document/position occurrences.
2. **Full forward index:** all unique selected fingerprints per document,
   including singletons.

Then derive a separate **document-only screening inverted sidecar**:

- retain only fingerprints in at least two distinct documents, `df >= 2`;
- store each document ID once per fingerprint, with no positions;
- order fingerprint records by `(document frequency, hash)`, ascending.

The sidecar is still planned, not implemented by this layout migration.
Its document-ID payload is 4 bytes per unique document/fingerprint pair,
plus its own fingerprint/offset metadata. It is additional data, not a
replacement for the full positional index.

| Operation | Data source |
|---|---|
| External text query | Fingerprint the text, query the full inverted index |
| Existing `get_document_candidates(id)` | Full forward list, then full inverted index |
| New indexed-document join candidate generation | Document-only sidecar |
| Exact indexed-pair overlap | Sorted full forward sets |
| Passage verification | Full positional inverted index and source text |

Corpus-singleton fingerprints are irrelevant to overlap between two indexed
documents, but can identify a match for an external query. The filtered
sidecar must not silently replace full-index external-query lookup.

## Trade-offs accepted

- Interleaved is the single canonical positional format.
- Document-only lookup remains correct, but reads position bytes along with
  document IDs and then selects the ID field.
- The future sidecar avoids position reads and occurrence duplicates for
  self-join screening. This should be benchmarked when implemented.
- External-query document-only reads may lose the old columnar I/O advantage.
  That is an explicit trade-off for fewer formats and simpler maintenance.
- No hash-width or position-width change was made. Fingerprints remain uint64,
  and positional records remain uint32 document ID plus uint32 position.
- No claim is made about the future sidecar's runtime or overall speedup.

## Metadata, migration and compatibility

New merged indexes write `postings_layout: "interleaved"`.
Readers and forward-index construction reject an explicit columnar or unknown
layout before interpreting posting bytes. Missing layout metadata means the
historical interleaved default; explicit `rolling64-v1` hash metadata is still
required independently.

Old interleaved rolling64 indexes remain usable. Columnar indexes must be
rebuilt from source or compatible rolling64 intermediate shards. Old legacy31
indexes/shards must be rebuilt from source due to the separate hashing
migration. Merely editing metadata does not convert either hashes or posting
bytes. No existing external corpus index was modified or converted.

The `postings_layout` arguments to merge/batch APIs and the
`--postings-layout` CLI option no longer exist. The retired
`experiments/benchmark_posting_layout.py` can be inspected in git history;
it is not an available conversion or benchmarking command.

Implementation:
[indexing.py](../findmytext/indexing.py),
[index_builder.py](../findmytext/index_builder.py).
Tests cover ID-only and positional reads, candidate results, sequential
de-duplication, forward-index rebuilding, both similarity methods, high-bit
fingerprints, empty indexes, and rejection of unsupported layouts.
