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

For the notebook and alignment visualizations: `pip install -e ".[demo]"`.
For tests: `pip install -e ".[test]"`, then `python -m pytest`.

---

## Quick start

### 1. Build the index

The first step is to index your corpus:

```python
import findmytext

# `corpus.jsonl` contains one JSON object per line with "text" and a unique "id"
index = findmytext.build_index("corpus.jsonl", "my_index", nb_workers=4)
```

The resulting index is stored on disk in `my_index` and memory-mapped at query time, so it
scales to corpora that are too large to fit in RAM. The same build is available from the
command line:
```bash
python -m findmytext.index_builder index corpus.jsonl my_index --nb_workers 4
```

JSONL, GZip (`.jsonl.gz`) and Zstandard (`.jsonl.zst`) are supported.
Defaults are **4-token k-grams** and **window size 6**. 
See [utils.py](findmytext/utils.py) for ingestion filters.
`nb_workers` (default 1) is the number of processes used for fingerprinting and merging;
`max_memory_gb` (default 8) bounds the memory they use.

Under the hood, `build_index` writes fingerprint *shards* (one whenever the posting buffer
fills up) into `my_index/shards`, then merges them into the final index and deletes them.
For a corpus split over many files, build one index per file (e.g. one cluster job each)
and merge them afterwards; the output of a merge is itself a valid input for later merges:

```python
findmytext.build_index("part1.jsonl.zst", "partial/part1", nb_workers=4)
# ... one call per corpus file, then:
index = findmytext.merge_indexes("partial", "my_index", nb_workers=4)
```

`merge_indexes` also accepts an explicit list of index directories instead of a parent
directory, and returns the opened index like `build_index`. From the command line:
```bash
python -m findmytext.index_builder merge partial my_index --nb_workers 4
```

The merge splits the fingerprint space into ranges holding about the same number of
postings, processed in parallel by `nb_workers` processes; `max_memory_gb` bounds the
memory they use in total (about 80 bytes per posting of the ranges in flight). A worker
killed by the kernel for exceeding a job's memory limit makes the merge fail with an
explicit error; lower `nb_workers` or raise the limit and rerun the merge on the same
inputs. Document IDs are assumed to be unique across the merged inputs.


### 2. Detect content

Once the index is built, detection is a single call:

```python
detector = findmytext.FingerprintChainDetector(index)  # or the path: "my_index"
scores = detector.search(query_text)

# `scores` is a dict mapping document IDs to containment probabilities
best_match_id = max(scores, key=scores.get)
print(f"Best match: {best_match_id}  (score: {scores[best_match_id]:.3f})")
```

The text in `query_text` does not need to be exactly identical to the one found in the corpus. **FindMyText** is designed to be robust to small differences between the documents.

Scores are probabilities that the query shares a passage of at least `min_fragment_length`
characters (default 1000) with the document, calibrated for near-verbatim reuse: a heavily
edited passage must be correspondingly longer to score high. The steepness of the
score curve is set by the detector's `dispersion` argument (default 0.5). Only documents
scoring at least `min_probability` (default 0.05) are returned; the same threshold also
sets how many shared fingerprints a document needs before it is retrieved and scored at all.

Posting reads are sequential by default. To enable parallel reads, pass the number of
available CPUs to the detector, e.g. `FingerprintChainDetector(index_dir, nb_cpus=8)`
(or `posting_read_threads=8` when opening `indexing.DiskBasedIndex` directly). This
only helps on cold or network storage; on warm or local data, sequential reads are faster.

### 3. Verifying a match with local alignment

To inspect a detected match in detail, use the built-in local alignment:

```python
alignment = findmytext.align(query_text, corpus_document_text)
alignment.show()
```

---

## How it works

1. **Fingerprinting**: each document is tokenised and converted to a set of $k$-gram hashes using the *winnowing* algorithm, which selects a representative subset of hashes from each sliding window.
2. **Inverted index**: fingerprints are stored in a disk-based inverted index mapping each hash to the list of `(doc_id, position)` pairs where it was observed.
3. **Chain detection**: at query time, the fingerprints of the query are looked up in the index. Rather than simply counting matches, FindMyText clusters matching fingerprints by their relative positions to identify *chains* — contiguous sequences of matches — and uses the total length of those chains as the containment score.

---

## License

**FindMyText** is released under an [MIT License](LICENSE).

The MIT License is a short and simple permissive license allowing both commercial and non-commercial use of the software. The only requirement is to preserve the copyright and license notices (see file License). Licensed works, modifications, and larger works may be distributed under different terms and without source code.

---

## Reference

If you use **FindMyText** in your research, please cite our paper:

```bibtex
@article{findmytext2026,
  authors = {Lars Henry Berge Olsen and Pierre Lison and Martin Jullum and Mark Anderson},
  title   = {FindMyText: Robust, Scalable Detection of Text Containment in Large Web-Crawled Corpora},
  year    = {2026}
}
```
