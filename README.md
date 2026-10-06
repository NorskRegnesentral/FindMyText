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
