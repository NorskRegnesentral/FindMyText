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

### 4. Evaluating detectors

`experiments/benchmark.py` creates the labelled data: positives are perturbed chunks of
corpus documents (hyphenation, casing, boilerplate, unicode normalisation) that still
share a segment of Smith-Waterman score ≥ `pos_sw_threshold` with their source, padded
with out-of-corpus text to the median length of the negatives; negatives are corpus
documents permuted and paraphrased (LLM) until no segment scores ≥ `neg_sw_threshold`,
while keeping most of the original wording. One call builds the whole benchmark
(intermediate files in a work directory, resumable):

```python
import experiments.benchmark as b
b.build_benchmark("corpus_dir/", "outofcorpus_sample.jsonl", "eval_set.jsonl",
                  nb_examples=2000, pos_sw_threshold=1000, neg_sw_threshold=100)
```

The steps (`sample_corpus`, `generate_negatives`, `generate_positives`,
`merge_examples`) can also be run separately, e.g. over several jobs appending to the
same file; see the module docstring.

`experiments/eval.py` measures how well the detectors separate labelled positive
examples (texts containing a passage of an indexed document, with the document's id
in `source_doc_id`) from negatives. It reports ROC-AUC and the precision at 90 % and
99 % recall for the shared-fingerprint baseline and the chain detector (plain and
rarity-weighted); a positive only counts as detected through its source document.
Positives whose source document is not returned at all score 0 (the `missed` column);
once more than 1 % of them do, `P@R=99` collapses to the positive rate, and the
`diagnose` command explains which cut (`min_fingerprints`, `top_k`,
`min_probability`) dropped each source document and which values would recover them.

```bash
# Run all detectors against an index (requires scikit-learn)
python experiments/eval.py run --index my_index --data eval_set.jsonl
# Explain the missed positives
python experiments/eval.py diagnose --index my_index --data eval_set.jsonl
```

`experiments/baselines.py` provides a dense-retrieval baseline for comparison:
`DenseRetrievalDetector` embeds the query with a sentence-embedding model
(`Qwen/Qwen3-Embedding-4B` by default) and returns the closest documents of a database
of pre-computed document embeddings by exact cosine similarity (faiss). The database is
built from the Hugging Face dataset `maknee/wikipedia_qwen_4b` (1 M Wikipedia articles,
whose row ids match the `source_doc_id` of the Wikipedia evaluation data; ~10 GB):

```bash
# Requires faiss-cpu, sentence-transformers and huggingface_hub
python experiments/baselines.py build wiki_dense_db
```

```python
from baselines import DenseRetrievalDetector
from eval import Evaluator

Evaluator("my_index", detectors={"dense": DenseRetrievalDetector("wiki_dense_db")}).run("eval_set.jsonl")
```

---

## How it works

**Fingerprinting.** Comparing a query against every document character by character
would be hopeless at corpus scale, so each text is first reduced to a compact set of
*fingerprints*. The text is tokenised into words, every run of $k$ consecutive words (a
$k$-gram, $k=4$ by default) is hashed, and the *winnowing* algorithm then keeps only a
small, deterministic subset of these hashes: within each sliding window of $w$ consecutive
$k$-grams ($w=6$ by default), the smallest hash is selected. Because the choice depends
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
