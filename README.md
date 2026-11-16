<p align="center">
  <img src="assets/findmytext_logo.png" alt="FindMyText logo" width="320"/>
</p>

# **FindMyText**

**FindMyText** is an open-source Python package for efficiently detecting whether a given text appears, in part or in full, within a large text corpus. It is particularly suited for verifying the presence of copyrighted or licensed material in large, web-crawled corpora. This can notably provide important insights into which texts have been used to pre-train LLMs. 

The tool builds on standard document fingerprinting techniques, and extends them with a novel mechanism that explicitly captures *sequences* (chains) of matching fingerprints. This makes it robust to near-verbatim copies — texts that share the same content but with minor differences due to OCR errors, formatting variants, text normalisation, or added boilerplate. Leveraging a distributed, disk-based indexing framework, FindMyText scales to large corpora that cannot be held in memory.

A full step-by-step walkthrough is available in [`demo.ipynb`](demo.ipynb).

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

The first step is to index your corpus. Input records are JSON objects with `"text"` and a unique `"id"`, one per line.
JSONL, GZip (`.jsonl.gz`) and Zstandard (`.jsonl.zst`) are supported.
Fingerprints are extracted in parallel and then merged into a single disk-based index:

```python
from findmytext import index_builder

index_builder.index_file("corpus.jsonl", "my_fingerprints", nb_workers=1)
index_builder.merge_indexes_from_dir("my_fingerprints", "my_index")
```

The resulting index is stored on disk and memory-mapped at query time, so it scales to corpora that are too large to fit in RAM.

The `index_builder` can also be used directly from the command line:

```bash
# Step 1: extract fingerprints from a corpus file into intermediate shards
python -m findmytext.index_builder index corpus.jsonl my_fingerprints --nb_workers 1

# Step 2: merge shards into a final disk-based index
python -m findmytext.index_builder merge my_fingerprints my_index
```

Defaults are **4-token k-grams** and **window size 6**. Documents shorter than
100 characters are skipped; the maximum is 100,000 characters for JSONL/GZip
and 500,000 for Zstandard. `max_nb_documents_before_flush` controls shard
size (default 1,000,000 documents). See [utils.py](findmytext/utils.py) for ingestion
filters.

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
