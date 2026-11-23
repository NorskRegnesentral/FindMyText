"""Website-only detector adapter for the FindMyText demo app.

This module keeps the web app's count-based scoring and highlight helpers out of
the general detector modules. It intentionally exposes the small API expected by
``web/corpussearch/detection.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

import numpy as np

from findmytext import indexing, winnower
from findmytext.detectors import (
    Candidates,
    cluster_linkage,
    cluster_rectangle,
    largest_cluster_weight,
)

DEFAULT_CLUSTERING_PARAMS: Dict[str, object] = {
    "method": "rectangle",
    "position_threshold": 30,
    "offset_threshold": 30,
    "distance_threshold": 30,
    "min_cluster_size": 5,
}

_WORD_RE = re.compile(r"\w", re.UNICODE)


def tokenize_with_offsets(document: str, punctuation: bool = False) -> list[tuple[int, int]]:
    """Tokenize like ``Winnower.tokenize``, but return character offsets."""
    spans: list[tuple[int, int]] = []
    i, n = 0, len(document)
    while i < n:
        if document[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not document[i].isspace():
            i += 1
        end = i
        if punctuation:
            spans.append((start, end))
            continue
        wstart = wend = None
        for j in range(start, end):
            if _WORD_RE.match(document[j]):
                if wstart is None:
                    wstart = j
                wend = j + 1
        if wstart is not None and wend is not None:
            spans.append((wstart, wend))
    return spans


def _validate_score(score: str) -> None:
    if score not in ("count", "rarity"):
        raise ValueError(f"score must be 'count' or 'rarity', got {score!r}.")


def _fingerprint_rarity(
    index: indexing.DiskBasedIndex, hashes: np.ndarray, num_documents: int
) -> np.ndarray:
    """Rarity ``1 - df(h) / N`` of each fingerprint in ``hashes`` (sorted, possibly
    repeated), 0.0 for those absent from the index."""
    unique, inverse = np.unique(hashes, return_inverse=True)
    postings = index._get_postings(unique, only_doc_ids=True)
    rarity = np.zeros(len(unique), dtype=np.float64)
    if postings:
        found = np.fromiter(postings.keys(), dtype=np.uint64, count=len(postings))
        rarity[np.searchsorted(unique, found)] = [
            1 - len(np.unique(doc_ids)) / num_documents for doc_ids in postings.values()
        ]
    return rarity[inverse]


@dataclass(frozen=True, eq=False)
class _Prepared:
    """A query with its candidate hits, and the rarity of each query fingerprint."""

    candidates: Candidates
    rarity: np.ndarray  # aligned with candidates.hashes

    def weights(self, score: str) -> np.ndarray:
        return self.rarity if score == "rarity" else np.ones(len(self.candidates.hashes))


class TextContainmentDetector:
    """Count-scoring detector API used by the website."""

    def __init__(self, index_dir: str, top_k: int = 50, min_fingerprints: int = 5):
        self.index_dir = index_dir
        self.top_k = top_k
        self.min_fingerprints = min_fingerprints
        self.index = indexing.DiskBasedIndex(index_dir)
        self.num_documents = self.index.num_documents
        self.runtime_winnower = winnower.Winnower.from_metadata(
            {**self.index.get_metadata(), "window_size": 1}
        )

    def find_matches_jaccard(self, text: str, score: str = "count") -> Dict[str, float]:
        """Per candidate, the number (or rarity sum) of distinct shared fingerprints."""
        _validate_score(score)
        prepared = self._prepare(text)
        candidates, weights = prepared.candidates, prepared.weights(score)
        return {
            doc_id: float(weights[candidates.shared_fingerprints(block)].sum())
            for doc_id, block in candidates.hits
        }

    def find_matches_clustering(
        self,
        text: str,
        clustering_params: Dict[str, object] | None = None,
        score: str = "count",
    ) -> Dict[str, float]:
        """Per candidate, the number (or rarity sum) of distinct shared fingerprints
        in its largest cluster (0 when no cluster survives ``min_cluster_size``)."""
        _validate_score(score)
        params = {**DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}
        prepared = self._prepare(text)
        candidates, weights = prepared.candidates, prepared.weights(score)
        scores: Dict[str, float] = {}
        for doc_id, block in candidates.hits:
            fingerprint, pos_query, pos_doc = candidates.join(block)
            labels = self._cluster(pos_query, pos_doc - pos_query, params)
            scores[doc_id] = largest_cluster_weight(labels, fingerprint, weights)
        return scores

    def get_match_highlight_positions(
        self,
        text: str,
        doc_id: str,
        clustering_params: Dict[str, object] | None = None,
        method: str = "clustering",
    ) -> list[int]:
        groups = self.get_match_highlight_clusters(
            text, doc_id, clustering_params, method=method, which="largest"
        )
        return sorted({p for group in groups for p in group})

    def get_match_highlight_clusters(
        self,
        text: str,
        doc_id: str,
        clustering_params: Dict[str, object] | None = None,
        method: str = "clustering",
        which: str = "largest",
    ) -> list[list[int]]:
        """Query positions (k-gram indices) shared with ``doc_id``, grouped by cluster
        (largest first by distinct fingerprints) or as a single group for
        ``method="jaccard"``."""
        if method not in ("clustering", "jaccard"):
            raise ValueError(f"method must be 'clustering' or 'jaccard', got {method!r}.")
        if which not in ("largest", "all"):
            raise ValueError(f"which must be 'largest' or 'all', got {which!r}.")

        candidates = self._prepare(text).candidates
        if doc_id not in candidates.hits.doc_ids:
            return []
        block = candidates.hits.block(candidates.hits.doc_ids.index(doc_id))
        fingerprint, pos_query, pos_doc = candidates.join(block)

        if method == "jaccard":
            return [np.unique(pos_query).tolist()]

        params = {**DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}
        labels = self._cluster(pos_query, pos_doc - pos_query, params)
        keep = labels >= 0
        if not keep.any():
            return []
        labels, fingerprint, pos_query = labels[keep], fingerprint[keep], pos_query[keep]

        # Distinct fingerprints per cluster, to rank the clusters by size.
        _, first = np.unique(labels.astype(np.int64) * len(candidates.hashes) + fingerprint, return_index=True)
        n_unique = np.bincount(labels[first])
        cluster_ids = np.argsort(-n_unique, kind="stable")
        if which == "largest":
            cluster_ids = cluster_ids[:1]
        return [np.unique(pos_query[labels == c]).tolist() for c in cluster_ids.tolist()]

    def _prepare(self, text: str) -> _Prepared:
        hashes, positions = self.runtime_winnower.get_winnowed_fingerprints(text)
        hits = self.index.get_closest_hits(
            query=hashes, top_k=self.top_k, min_fingerprints=self.min_fingerprints, verbose=False
        )
        candidates = Candidates.from_query(hashes, positions, hits)
        rarity = _fingerprint_rarity(self.index, candidates.hashes, self.num_documents)
        return _Prepared(candidates, rarity)

    @staticmethod
    def _cluster(pos_query: np.ndarray, offset: np.ndarray, params: Dict[str, object]) -> np.ndarray:
        method = params["method"]
        if method == "rectangle":
            return cluster_rectangle(
                pos_query, offset,
                params["position_threshold"], params["offset_threshold"], params["min_cluster_size"],
            )
        if method in {"euclidean", "cityblock", "chebyshev"}:
            return cluster_linkage(
                pos_query, offset, method, params["distance_threshold"], params["min_cluster_size"]
            )
        raise ValueError(
            f"Unknown clustering method {method!r}; must be one of "
            "'rectangle', 'euclidean', 'cityblock', or 'chebyshev'."
        )
