<p align="center">
  <img src="assets/findmytext_logo.png" alt="FindMyText logo" width="320"/>
</p>

# FindMyText

**FindMyText** is an open-source Python package for efficiently detecting whether a given text appears, in part or in full, within a large text corpus. It is particularly suited for verifying the presence of copyrighted or licensed material in large, web-crawled corpora. This can notably provide important insights into which texts have been used to pre-train LLMs. 

The tool builds on standard document fingerprinting techniques, and extends them with a novel mechanism that explicitly captures *sequences* (chains) of matching fingerprints. This makes it robust to near-verbatim copies — texts that share the same content but with minor differences due to OCR errors, formatting variants, text normalisation, or added boilerplate. Leveraging a distributed, disk-based indexing framework, FindMyText scales to large corpora that cannot be held in memory.

A full step-by-step walkthrough is available in [`demo.ipynb`](demo.ipynb).

**Online demo**: [here](https://findmytext.nr.no).

---

## Installation

```bash
git clone https://github.com/NorskRegnesentral/FindMyText.git
cd FindMyText
pip install -r requirements.txt
```

---

## Quick start

### 1. Build the index

The first step is to index your corpus. Fingerprints are extracted in parallel and then merged into a single disk-based index:

```python
import index_builder

# `corpus` is any iterable of dicts with "text" and "id" fields
files = index_builder.index_data_parallel(corpus, "temp_data/my_fingerprints", n_workers=4)
index_builder.merge_indexes_from_prefix("temp_data/my_fingerprints", "my_index")
```

See `index_builder` for more details on possible options to specify the input corpus. 

The resulting index is stored on disk and memory-mapped at query time, so it scales to corpora that are too large to fit in RAM.

### 2. Detect content

Once the index is built, detection is a single call:

```python
import detectors

detector = detectors.FingerprintChainDetector("my_index")

query_text = "Here is the text I would like to check"
scores = detector.get_containment_scores(query_text)

# `scores` is a dict mapping document IDs to containment scores
best_match_id = max(scores, key=scores.get)
print(f"Best match: {best_match_id}  (score: {scores[best_match_id]:.3f})")
```

The text in `query_text` does not need to be exactly identical to the one found in the corpus. **FindMyText** is designed to be robust to small differences between the documents. 

### 3. Verifying a match with local alignment

To inspect a detected match in detail, use the built-in local alignment:

```python
import oracle

alignment = oracle.align(query_text, corpus_document_text)
alignment.show()
```

---

## How it works

1. **Fingerprinting**: each document is tokenised and converted to a set of $k$-gram hashes using the *winnowing* algorithm, which selects a representative subset of hashes from each sliding window.
2. **Inverted index**: fingerprints are stored in a disk-based inverted index mapping each hash to the list of `(doc_id, position)` pairs where it was observed.
3. **Chain detection**: at query time, the fingerprints of the query are looked up in the index. Rather than simply counting matches, **FindMyText** clusters matching fingerprints by their relative positions to identify *chains* — contiguous sequences of matches — and uses the total length of those chains as the containment score.

---

## Reference

If you use FindMyText in your research, please cite our paper:

```
@article{findmytext2026,
  authors= {Lars Henry Berge Olsen and Pierre Lison and Martin Jullum and Mark Anderson},
  title   = {FindMyText: Robust, Scalable Detection of Text Containment in Large Web-Crawled Corpora},
  year    = {2026}
}
```
