"""Detection engine for the corpus inclusion checker.

This module owns only the Flask-facing glue: an LRU of loaded detectors,
streaming progress events, and turning source-level highlight positions into
character spans for the browser. Website-specific count scoring lives in
``findmytext.web.TextContainmentDetector``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Iterator

# The FindMyText source package lives at the repo root, two levels up from this
# file (web/corpussearch/detection.py). Ensure it is importable regardless of
# the working directory the app is launched from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from findmytext.web import TextContainmentDetector, tokenize_with_offsets  # noqa: E402

from .config import AppConfig, CorpusConfig, AlgoParams  # noqa: E402


# User-tunable bounds for the position-aware ("our method") hyperparameters.
# Keys match the core clustering-config keys used by ``_clustering_params``.
CC_PARAM_BOUNDS = {
    "position_threshold": (1, 200),
    "offset_threshold": (1, 200),
    "min_cluster_size": (1, 50),
}


def sanitize_cc_overrides(data: dict | None) -> dict:
    """Extract and clamp user-supplied clustering hyperparameters.

    Unknown/absent/invalid values are ignored, so the caller falls back to the
    configured defaults. Returned keys match the core clustering config.
    """
    out: dict = {}
    if not data:
        return out
    for key, (lo, hi) in CC_PARAM_BOUNDS.items():
        v = data.get(key)
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        out[key] = max(lo, min(hi, iv))
    return out


def _clustering_params(params: AlgoParams, overrides: dict | None = None) -> dict:
    """Map the web-layer AlgoParams onto the core clustering configuration.

    ``overrides`` (already sanitized) may replace ``position_threshold``,
    ``offset_threshold`` and/or ``min_cluster_size``.
    """
    cp = {
        "method": "rectangle",
        "position_threshold": params.cc_doc1_position_threshold,
        "offset_threshold": params.cc_offset_threshold,
        "min_cluster_size": params.cc_min_cluster_size,
    }
    if overrides:
        for key in CC_PARAM_BOUNDS:
            if overrides.get(key) is not None:
                cp[key] = overrides[key]
    return cp


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
    return _merge_spans(raw)


def _merge_spans(spans) -> list[list[int]]:
    """Merge overlapping/touching ``(start, end)`` character spans."""
    ordered = sorted([int(s), int(e)] for s, e in spans)
    merged: list[list[int]] = []
    for s, e in ordered:
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
    "jaccard": "Shared fingerprints (baseline)",
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
    cc_overrides: dict | None = None,
) -> Iterator[dict]:
    """Generator yielding progress events and a final scoring result event.

    Scoring only — no position/highlight work is done here. Highlighting is a
    separate, on-demand step (see :func:`run_highlight`).

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
    cparams = _clustering_params(params, cc_overrides)

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

    winnower = detector.index.runtime_winnower

    # 2. Tokenise the query with the plain core tokenizer (no position tracking)
    #    purely to report a fingerprint count and to bail out early on text that
    #    produces no tokens.
    yield progress("fingerprinting", 35)
    n_query_tokens = len(winnower.tokenize(text))
    if n_query_tokens == 0:
        yield {"type": "result", **_empty_result(corpus, text, algorithms)}
        return

    # 3. Score candidates with each requested algorithm (all scoring lives in
    #    the core detector).
    yield progress("scoring", 70)
    results: dict[str, dict] = {}

    if "jaccard" in algorithms:
        scores = detector.find_matches_jaccard(text, score="count")
        ranking = _ranking_from_scores(scores)
        results["jaccard"] = _format_ranking(corpus, ranking, params.normalizing_constant)

    if "connected_components" in algorithms:
        scores = detector.find_matches_clustering(text, cparams, score="count")
        ranking = _ranking_from_scores(scores)
        results["connected_components"] = _format_ranking(
            corpus, ranking, params.normalizing_constant
        )

    yield progress("done", 100)
    yield {
        "type": "result",
        "corpus": corpus.id,
        "corpus_label": corpus.label,
        "query_length": len(text),
        "query_fingerprints": n_query_tokens,
        "algorithms": {k: ALGORITHMS[k] for k in algorithms if k in ALGORITHMS},
        "results": results,
        "elapsed": round(time.time() - start, 1),
    }


# ---------------------------------------------------------------------------
# On-demand highlighting (phase 2): compute highlight spans for a single
# already-scored document, in one or more highlight modes that can be overlaid.
# ---------------------------------------------------------------------------
HIGHLIGHT_MODES = {
    "cc_largest": {
        "label": "Position-aware — largest aligned passage",
        "method": "clustering",
        "which": "largest",
    },
    "cc_all": {
        "label": "Position-aware — all aligned passages",
        "method": "clustering",
        "which": "all",
    },
    "jaccard": {
        "label": "Shared fingerprints (baseline) — all overlapping passages",
        "method": "jaccard",
        "which": "largest",
    },
}


def run_highlight(
    manager: IndexManager,
    cfg: AppConfig,
    corpus: CorpusConfig,
    text: str,
    doc_id: str,
    modes: list[str],
    cc_overrides: dict | None = None,
) -> dict:
    """Compute highlight spans for one matched document in the requested modes.

    Returns a dict with a ``layers`` list; each layer corresponds to one
    requested mode and carries its clusters (one entry per highlighted group,
    each with merged character ``spans``) plus coverage stats. Layers are meant
    to be overlaid in the browser.
    """
    params = cfg.params
    cparams = _clustering_params(params, cc_overrides)
    detector, _ = manager.get(corpus, params)
    winnower = detector.index.runtime_winnower
    length = winnower.length

    # Character offsets for the query tokens (the tool-specific tokenizer). This
    # is what lets us turn the core's matched query token positions into spans.
    token_spans = tokenize_with_offsets(text, punctuation=winnower.punctuation)

    layers = []
    for mode in modes:
        spec = HIGHLIGHT_MODES.get(mode)
        if spec is None:
            continue
        cp = cparams if spec["method"] == "clustering" else None
        groups = detector.get_match_highlight_clusters(
            text, doc_id, cp, method=spec["method"], which=spec["which"]
        )
        clusters = []
        all_spans: list[list[int]] = []
        for positions in groups:
            spans = _spans_from_positions(positions, token_spans, length)
            if spans:
                clusters.append({"spans": spans})
                all_spans.extend(spans)
        union = _merge_spans(all_spans)
        matched = sum(e - s for s, e in union)
        layers.append({
            "mode": mode,
            "label": spec["label"],
            "clusters": clusters,
            "matched_chars": matched,
            "coverage_pct": round(100 * matched / len(text), 1) if text else 0.0,
        })

    return {
        "doc_id": doc_id,
        "corpus": corpus.id,
        "query_length": len(text),
        "layers": layers,
    }


def _format_ranking(corpus: CorpusConfig, ranking, C: float, limit: int = 5) -> dict:
    items = []
    for doc_id, score in ranking[:limit]:
        meta = corpus.doc_meta(doc_id)
        items.append({
            "doc_id": doc_id,
            "title": meta["title"],
            "url": meta["url"],
            "archive_url": meta["archive_url"],
            "score": int(score),
            "score_normalized": round(score / C, 2) if C else None,
        })
    top = items[0] if items and items[0]["score"] > 0 else None
    return {"top": top, "ranking": items}


def _empty_result(corpus: CorpusConfig, text: str, algorithms: list[str]) -> dict:
    return {
        "corpus": corpus.id,
        "corpus_label": corpus.label,
        "query_length": len(text),
        "query_fingerprints": 0,
        "algorithms": {k: ALGORITHMS[k] for k in algorithms if k in ALGORITHMS},
        "results": {k: {"top": None, "ranking": []} for k in algorithms},
        "elapsed": 0.0,
    }

