<p align="center">
  <img src="https://raw.githubusercontent.com/NorskRegnesentral/FindMyText/main/assets/findmytext_logo.png" alt="FindMyText logo" width="320"/>
</p>

# **FindMyText**

**FindMyText** is an open-source Python package for efficiently detecting whether a given text appears, in part or in full, within a large text corpus. It is particularly suited for verifying the presence of copyrighted or licensed material in large, web-crawled corpora. This can notably provide important insights into which texts have been used to pre-train LLMs. 

The tool builds on standard document fingerprinting techniques, and extends them with an efficient mechanism to capture *sequences* (chains) of matching fingerprints. This makes it robust to near-verbatim copies — texts that share the same content but with minor differences due to OCR errors, formatting variants, text normalisation, or added boilerplate. Leveraging a distributed, disk-based indexing framework, FindMyText scales to large corpora that cannot be held in memory.

A full step-by-step walkthrough is available in [`demo.ipynb`](https://github.com/NorskRegnesentral/FindMyText/blob/main/demo.ipynb).  See our [paper](https://arxiv.org/abs/2607.10020) for more details.

---

## Installation

```bash
pip install findmytext
```

For the notebook and alignment visualizations: `pip install "findmytext[demo]"`.

To work on the code, install it from source instead:

```bash
git clone https://github.com/NorskRegnesentral/FindMyText.git
cd FindMyText
pip install -e ".[test]"
python -m pytest
```

---

## Quick start

### 1. Build the index

The first step is to index your corpus:

```python
import findmytext

# `corpus.jsonl` contains one JSON object per line with "text" and a unique "id"
index = findmytext.build_index("corpus.jsonl", "my_index", n_workers=4)
```

The resulting index is stored on disk in `my_index` and memory-mapped at query time, so it
scales to corpora that are too large to fit in RAM. The same build is available from the
command line:
```bash
python -m findmytext.index_builder index corpus.jsonl my_index --n_workers 4
```

JSONL, GZip (`.jsonl.gz`) and Zstandard (`.jsonl.zst`) are supported. Each corpus file must contain one JSON object per line with at least an ``id`` and a ``text`` field. Other fields are ignored.  Document IDs must be unique.

The indexing process reads the corpus, computes winnowed fingerprints for each document (using by default 4-token k-grams and window size 6) and uses those to populate the index. 
`n_workers` is the number of processes used for fingerprinting and merging (default: the
CPUs available to the process) while `max_memory_gb` bounds the memory they use (default: 80% of
the memory currently available). 

For a corpus split over many files, you can start by building one index per file (e.g. one cluster job each)
and merge them afterwards; the output of a merge is itself a valid input for later merges:

```python
findmytext.build_index("part1.jsonl.zst", "partial/part1", n_workers=4)
# ... one call per corpus file, then:
index = findmytext.merge_indexes("partial", "my_index", n_workers=4)
```
Document IDs are assumed to be unique across the merged inputs.


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
characters (default 1000) with the document. Only documents
scoring at least `min_probability` (default 0.05) are returned.

To enable parallel reads, pass a number of reader threads to the detector, e.g. 
`FingerprintChainDetector(index_dir, n_workers=8)`. 


### 3. Verifying a match with local alignment

To inspect a detected match in detail, use the built-in local alignment:

```python
alignment = findmytext.align(query_text, corpus_document_text)
alignment.show()
```

---

## How it works

**Fingerprinting.** Comparing a query against every document character by character
would be hopeless at corpus scale, so each text is first reduced to a compact set of
*fingerprints*. The text is tokenised into words, every run of *k* consecutive words (a
*k*-gram, *k* = 4 by default) is hashed, and the *winnowing* algorithm then keeps only a
small, deterministic subset of these hashes: within each sliding window of *w* consecutive
*k*-grams (*w* = 6 by default), the smallest hash is selected. Because the choice depends
only on the local content, two texts sharing a passage end up selecting the same
fingerprints for that passage, wherever it appears in the document and regardless of what
surrounds it. Winnowing also guarantees that at least one fingerprint is picked per window,
so no passage longer than a few words can slip through unfingerprinted, while storing
only a fraction of all $k$-grams.

**Inverted index.** The fingerprints of the whole corpus are stored in an *inverted
index*: for each fingerprint value, a sorted list of the documents and positions where it
occurs. The index lives on disk and is memory-mapped at query time, so only the parts
actually touched by a query are read, and corpora far larger than RAM can be searched
from a modest machine.

**Building at scale.** Indexing a large corpus is split into two independent steps so
that it can be spread over many machines. First, documents are fingerprinted in parallel
and written out as *shards*: self-contained partial indexes, each produced whenever the
in-memory buffer of fingerprints reaches its memory budget. Then the shards are *merged*
into a single index: the space of fingerprint values is cut into ranges holding roughly
the same number of entries, and each range is assembled by a separate worker from the
corresponding slice of every shard. The merge is itself restartable and composable, so one
index per corpus file can be built as separate cluster jobs and combined afterwards, and
an existing index can later be extended with new material.

**Searching.** At query time the query text is fingerprinted in the same way, its
fingerprints are looked up in the index, and the documents sharing enough of them are
retrieved as candidates. A plain count of shared fingerprints is a fragile signal: common
phrases produce scattered coincidental matches, while a genuine copy shows up as a *chain*
of matches whose positions in the query and in the candidate advance together. FindMyText
therefore clusters the shared fingerprints by their relative positions and scores each
candidate by its largest coherent cluster. That score is finally converted into a
calibrated probability that the two texts share a passage of at least a chosen length,
which is what `search` returns. Because winnowing picks fingerprints locally, a copy with
OCR errors, changed formatting or added boilerplate still shares most of its fingerprints
with the original, and the chain survives the noise.

**Verifying.** For the documents flagged as likely matches, a local sequence alignment
(`findmytext.align`) pinpoints the exact shared passage and shows the differences between
the two versions side by side.

---

## Evaluation details

The evaluation described in the FindMyText [paper](https://arxiv.org/abs/2306.17100) relies on an "adversarial" benchmarking procedure, in order to assess whether detectors can distinguish content reuse from mere similarity between documents. 

The benchmarking and evaluation scripts live in the `experiments/` folder of the
[repository](https://github.com/NorskRegnesentral/FindMyText) and are not part of the PyPI
package. From a source checkout, their dependencies are installed with
`pip install -e ".[experiments]"`.

Concretely, based on an corpus, we first generate a set of examples consisting of both positive and negative cases:
- *Positive examples* are perturbed chunks of corpus documents that still share a segment of Smith-Waterman score ≥ `pos_sw_threshold` with their source. Those pertubed chunks are concatenated with out-of-corpus texts. The length of the shared chunk is drawn between `pos_sw_threshold` and a share `max_fragment_share` (default 0.5) of the final text length; `max_fragment_share=None` gives the hardest positives, whose shared segment is barely longer than the threshold.
- *Negative examples* are corpus documents that have been permuted and paraphrased (LLM) until no segment scores ≥ `neg_sw_threshold`. 

A new benchmark can be created with the following command:

```python
import experiments.benchmark as b
b.build_benchmark("corpus_dir/", "outofcorpus_sample.jsonl", "eval_set.jsonl",
                  nb_examples=2000, pos_sw_threshold=1000, neg_sw_threshold=100,
                  max_fragment_share=0.5)
```

The steps (`sample_corpus`, `generate_negatives`, `generate_positives`,
`merge_examples`) can also be run separately, e.g. over several jobs appending to the
same file; see the module docstring.

Once the benchmark is created, it can be used to evaluate detectors with `experiments/eval.py`.
The script measures how well the detectors separate labelled positive
examples from negatives, using the ROC-AUC and precision at 90 % and 99 % recall metrics.

The evaluation can be run with the following command:

```bash
# Run all detectors against an index (requires scikit-learn)
python experiments/eval.py run --index my_index --data eval_set.jsonl
```

---

## License

**FindMyText** is released under an [MIT License](https://github.com/NorskRegnesentral/FindMyText/blob/main/LICENSE).

The MIT License is a short and simple permissive license allowing both commercial and non-commercial use of the software. The only requirement is to preserve the copyright and license notices (see file License). Licensed works, modifications, and larger works may be distributed under different terms and without source code.

---

## Reference

If you use **FindMyText** in your research, please cite our [paper](https://arxiv.org/abs/2607.10020):

```bibtex
@article{findmytext2026,
  authors = {Lars Henry Berge Olsen and Pierre Lison and Martin Jullum and Mark Anderson},
  title   = {FindMyText: Robust, Scalable Detection of Text Containment in Large Web-Crawled Corpora},
  year    = {2026}
}
```
