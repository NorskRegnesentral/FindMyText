"""Detection engine: load corpus indexes on demand and run the two detection
algorithms (shared-fingerprint "Jaccard" baseline and the position-aware
"clustering" method) by delegating all scoring to the FindMyText core
(``detector.TextContainmentDetector``). This module owns only the web-layer
glue: an LRU of loaded detectors, streaming progress events, and turning the
core's highlight positions into character spans for the browser.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Iterator

# The FindMyText core modules (detector, indexing, winnower) live at the repo
# root, two levels up from this file (web/corpussearch/detection.py). Ensure the
# repo root is importable regardless of the working directory the app is
# launched from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from detector import TextContainmentDetector  # noqa: E402  (FindMyText core)

from .config import AppConfig, CorpusConfig, AlgoParams  # noqa: E402


def _clustering_params(params: AlgoParams) -> dict:
    """Map the web-layer AlgoParams onto the core clustering configuration."""
    return {
        "method": "rectangle",
        "position_threshold": params.cc_doc1_position_threshold,
        "offset_threshold": params.cc_offset_threshold,
        "min_cluster_size": params.cc_min_cluster_size,
    }


def _spans_from_positions(
    positions, token_spans: list[tuple[int, int]], length: int
) -> list[list[int]]:
    """Map query fingerprint token positions to merged character spans.

    A fingerprint at token position ``p`` covers the k-gram of ``length`` tokens
    starting there, i.e. tokens ``[p, p + length - 1]``. This is presentation
    logic (not scoring), so it stays in the web layer.
    """
    ntok = len(token_spans)
    raw: list[tuple[int, int]] = []
    for p in positions:
        if p < 0 or p >= ntok:
            continue
        end_tok = min(p + length - 1, ntok - 1)
        raw.append((token_spans[p][0], token_spans[end_tok][1]))
    raw.sort()
    merged: list[list[int]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


# ---------------------------------------------------------------------------
# Index manager: load a core detector per corpus on demand, keep an LRU
# of resident ones.
# ---------------------------------------------------------------------------
class IndexManager:
    def __init__(self, max_loaded: int = 1):
        self.max_loaded = max(1, max_loaded)
        self._cache: "OrderedDict[str, TextContainmentDetector]" = OrderedDict()
        self._lock = threading.Lock()

    def _load(self, corpus: CorpusConfig, params: AlgoParams) -> TextContainmentDetector:
        if not corpus.index_dir or not os.path.isdir(corpus.index_dir):
            raise FileNotFoundError(
                f"Index directory for corpus '{corpus.id}' not found: {corpus.index_dir}"
            )
        return TextContainmentDetector(
            corpus.index_dir,
            top_k=params.top_k,
            min_fingerprints=params.min_fingerprints,
        )

    def get(self, corpus: CorpusConfig, params: AlgoParams):
        """Return a loaded core detector for the corpus, loading (and evicting)
        as needed. Returns ``(detector, was_already_loaded)``.
        """
        with self._lock:
            if corpus.id in self._cache:
                self._cache.move_to_end(corpus.id)
                return self._cache[corpus.id], True
        # Load outside the lock (can be slow for large disk indexes).
        detector = self._load(corpus, params)
        with self._lock:
            self._cache[corpus.id] = detector
            self._cache.move_to_end(corpus.id)
            while len(self._cache) > self.max_loaded:
                self._cache.popitem(last=False)
        return detector, False

    def is_loaded(self, corpus_id: str) -> bool:
        with self._lock:
            return corpus_id in self._cache


# ---------------------------------------------------------------------------
# Main entry point: streaming detection
# ---------------------------------------------------------------------------
ALGORITHMS = {
    "jaccard": "Shared fingerprints (Jaccard baseline)",
    "connected_components": "Position-aware match (our method)",
}


def _ranking_from_scores(scores: dict) -> list[tuple[str, int]]:
    """Turn a ``{doc_id: score}`` mapping into a descending (doc_id, score) list."""
    ranking = [(doc_id, int(round(score))) for doc_id, score in scores.items()]
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


def run_detection(
    manager: IndexManager,
    cfg: AppConfig,
    corpus: CorpusConfig,
    text: str,
    algorithms: list[str],
) -> Iterator[dict]:
    """Generator yielding progress events and a final result event.

    Event shapes::

        {"type": "progress", "stage": str, "pct": int, "elapsed": float, "note": str?}
        {"type": "result", ...}
        {"type": "error", "message": str}
    """
    start = time.time()

    def progress(stage, pct, note=None):
        ev = {"type": "progress", "stage": stage, "pct": pct,
              "elapsed": round(time.time() - start, 1)}
        if note:
            ev["note"] = note
        return ev

    params = cfg.params
    cparams = _clustering_params(params)

    yield progress("starting", 5)

    # 1. Load the core detector (slow on first use for large disk corpora).
    already = manager.is_loaded(corpus.id)
    yield progress(
        "loading_index", 15,
        None if already else "First use of this corpus — loading the index, "
        "this can take a few seconds.",
    )
    try:
        detector, _ = manager.get(corpus, params)
    except Exception as exc:  # noqa: BLE001 - surface to the user
        yield {"type": "error", "message": f"Could not load corpus: {exc}"}
        return

    # The k-gram length is a property of the index the winnower was built with,
    # so read it from the loaded index rather than trusting the corpus config.
    length = detector.index.winnower.length

    # 2. Character offsets for the query tokens, so we can turn the core's
    #    query token positions into highlight spans. Also lets us bail out
    #    early on text that produces no tokens/fingerprints.
    yield progress("fingerprinting", 35)
    query_token_spans = detector.index.winnower.tokenize_with_offsets(text)
    if not query_token_spans:
        yield {"type": "result", **_empty_result(corpus, text, algorithms)}
        return

    # 3. Score candidates with each requested algorithm (all scoring lives in
    #    the core detector).
    yield progress("scoring", 70)
    results: dict[str, dict] = {}
    highlight = None

    if "jaccard" in algorithms:
        scores = detector.find_matches_jaccard(text, score="count")
        ranking = _ranking_from_scores(scores)
        results["jaccard"] = _format_ranking(corpus, ranking, params.normalizing_constant)
        if ranking and ranking[0][1] > 0 and highlight is None:
            top_doc = ranking[0][0]
            positions = detector.get_match_highlight_positions(
                text, top_doc, method="jaccard"
            )
            highlight = _make_highlight(
                "jaccard", top_doc, positions, query_token_spans, length, text
            )

    if "connected_components" in algorithms:
        scores = detector.find_matches_clustering(text, cparams, score="count")
        ranking = _ranking_from_scores(scores)
        results["connected_components"] = _format_ranking(
            corpus, ranking, params.normalizing_constant
        )
        # The clustering method is our headline method: prefer its highlight
        # when it is among the selected algorithms and it found a match.
        if ranking and ranking[0][1] > 0:
            top_doc = ranking[0][0]
            positions = detector.get_match_highlight_positions(
                text, top_doc, cparams, method="clustering"
            )
            highlight = _make_highlight(
                "connected_components", top_doc, positions,
                query_token_spans, length, text,
            )

    yield progress("done", 100)
    yield {
        "type": "result",
        "corpus": corpus.id,
        "corpus_label": corpus.label,
        "query_length": len(text),
        "query_fingerprints": len(query_token_spans),
        "algorithms": {k: ALGORITHMS[k] for k in algorithms if k in ALGORITHMS},
        "results": results,
        "highlight": highlight,
        "elapsed": round(time.time() - start, 1),
    }


def _format_ranking(corpus: CorpusConfig, ranking, C: float, limit: int = 5) -> dict:
    items = []
    for doc_id, score in ranking[:limit]:
        items.append({
            "doc_id": doc_id,
            "url": corpus.doc_url(doc_id),
            "score": int(score),
            "score_normalized": round(score / C, 2) if C else None,
        })
    top = items[0] if items and items[0]["score"] > 0 else None
    return {"top": top, "ranking": items}


def _make_highlight(algorithm, doc_id, qpositions, token_spans, length, text) -> dict:
    spans = _spans_from_positions(qpositions, token_spans, length)
    matched = sum(e - s for s, e in spans)
    return {
        "algorithm": algorithm,
        "doc_id": doc_id,
        "spans": spans,
        "matched_chars": matched,
        "coverage_pct": round(100 * matched / len(text), 1) if text else 0.0,
    }


def _empty_result(corpus: CorpusConfig, text: str, algorithms: list[str]) -> dict:
    return {
        "corpus": corpus.id,
        "corpus_label": corpus.label,
        "query_length": len(text),
        "query_fingerprints": 0,
        "algorithms": {k: ALGORITHMS[k] for k in algorithms if k in ALGORITHMS},
        "results": {k: {"top": None, "ranking": []} for k in algorithms},
        "highlight": None,
        "elapsed": 0.0,
    }
