"""Evaluation of text-containment detectors on labelled positive / negative examples.

The evaluation file is a JSONL file with one example per line:

- ``text``: the query text;
- ``type``: ``"positive"`` (the text contains a long passage shared with an indexed
  document) or ``"negative"`` (it does not);
- ``source_doc_id``: for positives, the id of the indexed document the shared passage
  comes from.

Each detector gives one score per example: for a positive, the containment probability
it assigns to the *source* document (0 if that document is not retrieved at all), and
for a negative the highest probability assigned to any document (0 if none is
retrieved).  A positive thus only counts as detected through the right document.  From
these scores we compute the ROC-AUC and the precision reachable at 90 % and 99 % recall
(``P@R=90``, ``P@R=99``).

Positives whose source document is not returned at all score exactly 0, like the
negatives without any result.  Once more than 1 % of the positives are in that
situation, 99 % recall is only reachable by accepting everything and ``P@R=99``
collapses to the positive rate: the table then reports the number of such ``missed``
positives and :meth:`Evaluator.diagnose_misses` (CLI: ``diagnose``) explains, for
each of them, which cut dropped the source document (too few shared fingerprints,
pushed out of ``top_k``, or probability below ``min_probability``) and which
parameter values would have recovered them.

Typical use::

    python experiments/eval.py run --index my_index --data eval_set.jsonl
    python experiments/eval.py diagnose --index my_index --data eval_set.jsonl

or, from Python::

    evaluator = Evaluator("my_index")
    results = evaluator.run("eval_set.jsonl")
    report = evaluator.diagnose_misses("eval_set.jsonl")

Any object exposing ``search(text) -> dict[str, float]`` or
``search(text, min_fragment_length) -> dict[str, float]`` can be added to the
evaluation with :meth:`Evaluator.add_detector`.  Detectors from
:mod:`findmytext.detectors` that are bound to the evaluator's index get a fast path:
one ``retrieve`` call per example feeds every detector's ``score_candidates``.
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
import time
from typing import Any, Iterable, Optional, Union

import numpy as np
import orjson
import polars as pl
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from findmytext import indexing  # noqa: E402
from findmytext import detectors as fmt_detectors  # noqa: E402

POSITIVE = "positive"
NEGATIVE = "negative"

DEFAULT_DETECTOR_NAMES = ("baseline", "chain", "baseline_rw", "chain_rw")


class Evaluator:
    """Evaluate containment detectors against one disk-based index.

    Four detectors are always built on the index: ``baseline``
    (:class:`~findmytext.detectors.NbSharedFingerprintsDetector`), ``chain``
    (:class:`~findmytext.detectors.FingerprintChainDetector`, rectangle method),
    and their rarity-weighted variants ``baseline_rw`` / ``chain_rw``.  Further
    detectors can be registered with :meth:`add_detector`.

    Parameters
    ----------
    index : str or indexing.DiskBasedIndex
        Directory of a disk-based winnowing index, or an already-opened index.
    min_probability : float
        ``min_probability`` of the default detectors (default: 1e-3).  Kept low so
        that negatives still receive informative, non-zero scores for ranking.
    top_k : int
        ``top_k`` of the default detectors (default: 50).
    n_workers : int
        Posting-read threads used when ``index`` is a path (default: 1).

    """

    def __init__(
        self,
        index: Union[str, indexing.DiskBasedIndex],
        min_probability: float = 1e-3,
        top_k: int = 50,
        n_workers: int = 1,
    ):
        if isinstance(index, indexing.DiskBasedIndex):
            self.index = index
        else:
            self.index = indexing.DiskBasedIndex(index, n_workers=n_workers)
        self.index_dir = self.index.index_dir
        self.detectors: dict[str, Any] = {}

        common = dict(top_k=top_k, min_probability=min_probability)
        self.add_detector(
            "baseline", fmt_detectors.NbSharedFingerprintsDetector(self.index, **common)
        )
        self.add_detector(
            "chain", fmt_detectors.FingerprintChainDetector(self.index, **common)
        )
        self.add_detector(
            "baseline_rw",
            fmt_detectors.NbSharedFingerprintsDetector(
                self.index, rarity_weighted=True, **common
            ),
        )
        self.add_detector(
            "chain_rw",
            fmt_detectors.FingerprintChainDetector(
                self.index, rarity_weighted=True, **common
            ),
        )

    def add_detector(self, name: str, detector: Any) -> None:
        """Register an extra detector to evaluate under ``name``.

        Any object with a ``search(text) -> dict[str, float]`` or
        ``search(text, min_fragment_length) -> dict[str, float]`` method is
        accepted; the fragment length is only passed when ``search`` can take it.
        Detectors from :mod:`findmytext.detectors` bound to this evaluator's index
        share one index retrieval per example with the default ones; any other
        detector is run through its ``search`` method.
        """
        if name in self.detectors:
            raise ValueError(f"a detector named {name!r} is already registered")
        if not callable(getattr(detector, "search", None)):
            raise TypeError(f"detector {name!r} has no callable 'search' method")
        self.detectors[name] = detector

    def _search(self, name: str, text: str, min_fragment_length: float) -> dict:
        """Call ``search`` of detector ``name``, passing ``min_fragment_length``
        only if its signature has that parameter (or a ``**kwargs`` catch-all)."""
        
        search = self.detectors[name].search
        params = inspect.signature(search).parameters
        if "min_fragment_length" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            return search(text, min_fragment_length=min_fragment_length)
        return search(text)

    @property
    def _shared(self) -> list[str]:
        """Names of the detectors that can share one retrieval per example."""
        return [
            name
            for name, d in self.detectors.items()
            if isinstance(d, fmt_detectors._BaseDetector) and d.index is self.index
        ]

    @property
    def _separate(self) -> list[str]:
        """Names of the detectors run through their own ``search`` method."""
        shared = set(self._shared)
        return [n for n in self.detectors if n not in shared]

    # ------------------------------------------------------------------ running

    def run(
        self,
        jsonl_path: str,
        min_fragment_length: float = 1000.0,
        output_scores: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Evaluate every detector on the examples of ``jsonl_path``.

        Parameters
        ----------
        jsonl_path : str
            JSONL file with fields ``text``, ``type`` (``"positive"`` /
            ``"negative"``) and, for positives, ``source_doc_id``.
        min_fragment_length : float
            Passed to the detectors (default 1 000 characters).
        output_scores : str, optional
            If given, one JSON line per example is written there with the example's
            ``type``, ``source_doc_id``, ``text_length`` and, per detector, the
            ``score`` used for the metrics plus the best-scoring document
            (``top_doc``, ``top_prob``).
        limit : int, optional
            Only evaluate the first ``limit`` examples.
        verbose : bool
            Show a progress bar and print the result table (default ``True``).

        Returns
        -------
        dict[str, dict[str, float]]
            ``{detector_name: {"roc_auc", "p_at_r90", "p_at_r99", "top1_acc",
            "n_pos_missed", "n_neg_empty", "max_recall", "mean_score_time",
            "mean_retrieval_time", "n_pos", "n_neg"}}``.
            ``n_pos_missed`` counts the positives whose source document was not
            returned at all (score 0; see :func:`compute_metrics`) and
            ``max_recall`` the recall reachable without accepting them all;
            :meth:`diagnose_misses` explains those misses.
            ``mean_retrieval_time`` is the per-example index query time shared by
            the detectors on the fast path (``NaN`` for the others, whose
            ``mean_score_time`` is then the full ``search`` time).

        """
        examples = list(_read_examples(jsonl_path, limit))
        if not examples:
            raise ValueError(f"no examples found in {jsonl_path}")
        labels = np.array([ex["type"] == POSITIVE for ex in examples], dtype=bool)
        n_pos, n_neg = int(labels.sum()), int((~labels).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"both classes are needed, found {n_pos} positives and {n_neg} negatives"
            )

        names = list(self.detectors)
        scores = {n: np.zeros(len(examples)) for n in names}
        top1 = {n: np.zeros(len(examples), dtype=bool) for n in names}
        score_times = {n: 0.0 for n in names}
        retrieval_time = 0.0

        out = open(output_scores, "wb") if output_scores else None
        try:
            iterator = tqdm(examples, desc="Evaluating", disable=not verbose)
            for i, ex in enumerate(iterator):
                results, elapsed, retrieval_elapsed = self._search_all(
                    ex["text"], min_fragment_length
                )
                retrieval_time += retrieval_elapsed
                record = {
                    "type": ex["type"],
                    "source_doc_id": ex.get("source_doc_id"),
                    "text_length": len(ex["text"]),
                    "detectors": {},
                }
                for name in names:
                    # Normalise keys/values so that external detectors returning
                    # non-string ids or numpy floats are handled (and serialisable).
                    probs = {str(k): float(v) for k, v in results[name].items()}
                    score_times[name] += elapsed[name]
                    top_doc, top_prob = _argmax(probs)
                    if ex["type"] == POSITIVE:
                        source_doc_id = str(ex["source_doc_id"])
                        score = probs.get(source_doc_id, 0.0)
                        top1[name][i] = top_doc == source_doc_id
                    else:
                        score = top_prob
                    scores[name][i] = score
                    record["detectors"][name] = {
                        "score": score, "top_doc": top_doc, "top_prob": top_prob,
                    }
                if out is not None:
                    out.write(orjson.dumps(record))
                    out.write(b"\n")
        finally:
            if out is not None:
                out.close()

        n = len(examples)
        shared_names = set(self._shared)
        metrics: dict[str, dict[str, float]] = {}
        for name in names:
            m = compute_metrics(labels, scores[name])
            m["top1_acc"] = float(top1[name][labels].mean())
            m["mean_score_time"] = score_times[name] / n
            m["mean_retrieval_time"] = (
                retrieval_time / n if name in shared_names else float("nan")
            )
            m["n_pos"], m["n_neg"] = n_pos, n_neg
            metrics[name] = m

        if verbose:
            print(format_results(metrics))
        return metrics

    def _search_all(
        self, text: str, min_fragment_length: float
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], float]:
        """Run every detector on ``text``.

        Returns the per-detector ``{doc_id: prob}`` results, the per-detector
        scoring (or full search) time, and the shared retrieval time.
        """
        results: dict[str, dict[str, float]] = {}
        elapsed: dict[str, float] = {}
        retrieval_elapsed = 0.0

        shared_names = self._shared
        if shared_names:
            shared = [self.detectors[n] for n in shared_names]
            t0 = time.perf_counter()
            # One retrieval wide enough for every detector: the extra candidates
            # a detector gets are scored and dropped by its own min_probability,
            # so results are unchanged.
            candidates = shared[0].retrieve(
                text,
                min_fragment_length,
                top_k=max(d.top_k for d in shared),
                min_fingerprints=min(
                    d._min_shared_fingerprints(min_fragment_length) for d in shared
                ),
            )
            retrieval_elapsed = time.perf_counter() - t0
            for name, detector in zip(shared_names, shared):
                t0 = time.perf_counter()
                results[name] = detector.score_candidates(candidates, min_fragment_length)
                elapsed[name] = time.perf_counter() - t0

        for name in self._separate:
            t0 = time.perf_counter()
            results[name] = self._search(name, text, min_fragment_length)
            elapsed[name] = time.perf_counter() - t0

        return results, elapsed, retrieval_elapsed

    # --------------------------------------------------------------- debugging

    def diagnose_misses(
        self,
        jsonl_path: str,
        min_fragment_length: float = 1000.0,
        detector_names: Optional[list[str]] = None,
        limit: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Explain why positives are *missed*, i.e. get a score of 0 in :meth:`run`.

        A positive is missed by a detector when its source document is absent from
        the detector's result, which happens at one of three hard cuts.  For every
        positive example, the source document is located in the index ranking of
        the query (all documents sharing at least one fingerprint, ordered as the
        index orders them) and, per detector, its raw score and probability are
        computed before any filtering.  The miss is then classified as:

        - ``no_shared_fingerprints``: the source shares no fingerprint with the
          query (document not indexed, ``source_doc_id`` mismatch, or every
          fingerprint of the passage destroyed by the edits);
        - ``below_min_fingerprints``: fewer shared fingerprints than the detector's
          retrieval threshold (``_min_shared_fingerprints``);
        - ``outside_top_k``: enough shared fingerprints, but more than ``top_k``
          documents share at least as many (near-duplicates, boilerplate);
        - ``below_min_probability``: retrieved, but the normalised raw score (the
          largest cluster for chain detectors) falls under ``min_probability``.

        Only detectors from :mod:`findmytext.detectors` bound to this evaluator's
        index can be diagnosed.  The "what-if" summary gives, per detector and
        independently for each parameter, the value that would recover every
        recoverable miss, and the one that would bring the miss rate within 1 %
        (so that ``P@R=99`` leaves the floor).  Note that ``min_fingerprints`` is
        derived from ``min_probability`` unless set explicitly, so lowering the
        latter also lowers the former.

        Parameters
        ----------
        jsonl_path : str
            Evaluation JSONL file (only the positives are used).
        min_fragment_length : float
            Passed to the detectors (default 1 000 characters).
        detector_names : list[str], optional
            Detectors to diagnose (default: all diagnosable ones).
        limit : int, optional
            Only consider the first ``limit`` examples of the file.
        output_path : str, optional
            If given, one JSON line per missed positive is written there.
        verbose : bool
            Show a progress bar and print the misses and the summary.

        Returns
        -------
        dict
            ``{"misses": [...], "summary": {detector: {...}}, "thresholds":
            {detector: {...}}}``.  Each miss has ``source_doc_id``,
            ``text_length``, ``n_query_fingerprints``, ``n_shared`` (unique
            fingerprints shared with the source), ``rank`` (``None`` if none is
            shared), ``n_candidates``, ``top_doc`` / ``top_count`` (the best
            ranked document) and, under ``detectors``, ``raw_score``,
            ``probability`` and ``reason`` for every detector that missed it.

        """
        names = list(detector_names) if detector_names is not None else self._shared
        diagnosable = set(self._shared)
        for name in names:
            if name not in self.detectors:
                raise ValueError(f"unknown detector {name!r}")
            if name not in diagnosable:
                raise ValueError(
                    f"{name!r} is not a findmytext detector bound to this index "
                    "and cannot be diagnosed"
                )
        if not names:
            raise ValueError("no diagnosable detector")
        dets = [self.detectors[n] for n in names]

        positives = [
            ex for ex in _read_examples(jsonl_path, limit) if ex["type"] == POSITIVE
        ]
        if not positives:
            raise ValueError(f"no positive examples found in {jsonl_path}")

        thresholds = {
            name: {
                "min_fingerprints": d._min_shared_fingerprints(min_fragment_length),
                "top_k": d.top_k,
                "min_probability": d.min_probability,
                "min_raw_score": float(d._min_raw_score(min_fragment_length)),
                "expected_fingerprints": float(
                    d._expected_fingerprints(min_fragment_length)
                ),
            }
            for name, d in zip(names, dets)
        }

        misses: list[dict] = []
        iterator = tqdm(positives, desc="Diagnosing", disable=not verbose)
        for ex in iterator:
            source_doc_id = str(ex["source_doc_id"])
            df_query = dets[0].get_fingerprints(ex["text"])
            located = self._locate_source(df_query["hash"].to_numpy(), source_doc_id)
            positions = located.pop("positions")
            source_only = (
                _source_candidates(df_query, source_doc_id, positions)
                if positions
                else None
            )
            record = {
                "source_doc_id": source_doc_id,
                "text_length": len(ex["text"]),
                "n_query_fingerprints": int(df_query["hash"].n_unique()),
                **located,
                "detectors": {},
            }
            for name, det in zip(names, dets):
                raw = prob = None
                if source_only is not None:
                    raw = float(det.raw_scores(source_only).get(source_doc_id, 0.0))
                    prob = float(
                        det._normalise_scores({source_doc_id: raw}, min_fragment_length)[
                            source_doc_id
                        ]
                    )
                reason = _classify_miss(
                    located["n_shared"], located["rank"], prob, thresholds[name]
                )
                if reason is not None:
                    record["detectors"][name] = {
                        "raw_score": raw, "probability": prob, "reason": reason,
                    }
            if record["detectors"]:
                misses.append(record)

        summary = {
            name: _summarise_misses(misses, name, thresholds[name], len(positives))
            for name in names
        }

        if output_path:
            with open(output_path, "wb") as out:
                for record in misses:
                    out.write(orjson.dumps(record))
                    out.write(b"\n")
        if verbose:
            print(format_misses(misses, names))
            print(format_miss_summary(summary))
        return {"misses": misses, "summary": summary, "thresholds": thresholds}

    def _locate_source(self, hashes: np.ndarray, source_doc_id: str) -> dict[str, Any]:
        """Rank ``source_doc_id`` among the documents sharing fingerprints with a query.

        Reads the posting lists once and reproduces the ordering used by the
        index's retrieval (shared-fingerprint count descending, ties by internal
        id).  Returns ``n_shared``, ``rank`` (``None`` when nothing is shared),
        ``n_candidates``, ``top_doc`` / ``top_count`` and ``positions``, the
        ``{hash: [positions]}`` mapping of the source document (``None`` when
        nothing is shared), as consumed by :func:`_source_candidates`.
        """
        empty = {
            "n_shared": 0, "rank": None, "n_candidates": 0,
            "top_doc": None, "top_count": 0, "positions": None,
        }
        postings = self.index._get_postings(np.asarray(hashes, dtype=np.uint64))
        if not postings:
            return empty
        all_doc_ids = np.concatenate(
            [np.unique(arr["doc_id"]) for arr in postings.values()]
        )
        unique_docs, counts = np.unique(all_doc_ids, return_counts=True)
        # np.unique sorts by internal id, and the index breaks count ties in that
        # order (stable heapq.nlargest), hence this two-key sort.
        order = np.lexsort((unique_docs, -counts.astype(np.int64)))

        to_external = self.index.to_external_doc_id
        rank = internal_id = None
        for r, i in enumerate(order, start=1):
            if to_external[int(unique_docs[i])] == source_doc_id:
                rank, internal_id = r, int(unique_docs[i])
                break
        top = int(order[0])
        located = {
            **empty,
            "n_candidates": int(len(unique_docs)),
            "top_doc": to_external[int(unique_docs[top])],
            "top_count": int(counts[top]),
        }
        if rank is None:
            return located

        positions = {}
        for fp, arr in postings.items():
            pos = arr["position"][arr["doc_id"] == internal_id]
            if len(pos):
                positions[int(fp)] = pos.tolist()
        located.update(
            n_shared=int(counts[order[rank - 1]]), rank=rank, positions=positions
        )
        return located


# ------------------------------------------------------------------- helpers


def _read_examples(jsonl_path: str, limit: Optional[int] = None) -> Iterable[dict]:
    """Yield validated examples from ``jsonl_path``."""
    with open(jsonl_path, "rb") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and line_no > limit:
                break
            if not line.strip():
                continue
            ex = orjson.loads(line)
            if "text" not in ex or "type" not in ex:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: every example needs 'text' and 'type'"
                )
            if ex["type"] not in (POSITIVE, NEGATIVE):
                raise ValueError(
                    f"{jsonl_path}:{line_no}: 'type' must be '{POSITIVE}' or "
                    f"'{NEGATIVE}', got {ex['type']!r}"
                )
            if ex["type"] == POSITIVE and ex.get("source_doc_id") is None:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: positive examples need a 'source_doc_id'"
                )
            yield ex


def _source_candidates(
    df_query: pl.DataFrame, source_doc_id: str, positions: dict[int, list[int]]
) -> fmt_detectors.Candidates:
    """Debugging aid: a :class:`~findmytext.detectors.Candidates` holding only the
    source document, built from its ``{hash: [positions]}`` mapping (see
    :meth:`Evaluator._locate_source`), so that detectors can score it even when the
    index's retrieval would have cut it off."""
    matches = fmt_detectors.convert_closest_matches_with_positions_to_df(
        {source_doc_id: positions}
    )
    return fmt_detectors.Candidates(df_query, matches)


def _argmax(probs: dict[str, float]) -> tuple[Optional[str], float]:
    """Highest-probability document of a detector result (``(None, 0.0)`` if empty)."""
    if not probs:
        return None, 0.0
    doc = max(probs, key=probs.get)
    return doc, float(probs[doc])


NO_SHARED_FINGERPRINTS = "no_shared_fingerprints"
BELOW_MIN_FINGERPRINTS = "below_min_fingerprints"
OUTSIDE_TOP_K = "outside_top_k"
BELOW_MIN_PROBABILITY = "below_min_probability"
MISS_REASONS = (
    NO_SHARED_FINGERPRINTS, BELOW_MIN_FINGERPRINTS, OUTSIDE_TOP_K, BELOW_MIN_PROBABILITY,
)


def _classify_miss(
    n_shared: int, rank: Optional[int], probability: Optional[float], thresholds: dict
) -> Optional[str]:
    """Why a source document is absent from a detector's result, in the order the
    cuts are applied by ``search`` (``None`` when it would be returned)."""
    if n_shared == 0 or rank is None:
        return NO_SHARED_FINGERPRINTS
    if n_shared < thresholds["min_fingerprints"]:
        return BELOW_MIN_FINGERPRINTS
    if rank > thresholds["top_k"]:
        return OUTSIDE_TOP_K
    if probability is None or probability < thresholds["min_probability"]:
        return BELOW_MIN_PROBABILITY
    return None


def _value_recovering(values: list[float], n_recover: int, easiest_first: bool) -> Optional[float]:
    """Threshold on one parameter that recovers the ``n_recover`` easiest misses.

    ``values`` are the misses' values of the quantity the parameter cuts on;
    ``easiest_first=True`` when a miss is easier to recover the smaller its value
    (ranks), ``False`` when the larger (shared counts, probabilities).
    """
    if n_recover <= 0 or not values:
        return None
    ordered = sorted(values, reverse=not easiest_first)
    return ordered[min(n_recover, len(ordered)) - 1]


def _summarise_misses(
    misses: list[dict], name: str, thresholds: dict, n_pos: int
) -> dict[str, Any]:
    """Per-detector counts of miss reasons and "what-if" parameter values.

    ``to_recover_all`` gives, independently for ``top_k``, ``min_fingerprints`` and
    ``min_probability``, the value recovering every miss that shares at least one
    fingerprint; ``to_reach_recall_99`` the value recovering enough of them for
    the detector to miss at most 1 % of the positives (``None`` if already so).
    """
    rows = [(m, m["detectors"][name]) for m in misses if name in m["detectors"]]
    reasons = {reason: 0 for reason in MISS_REASONS}
    for _, d in rows:
        reasons[d["reason"]] += 1
    recoverable = [(m, d) for m, d in rows if d["reason"] != NO_SHARED_FINGERPRINTS]
    ranks = [m["rank"] for m, _ in recoverable]
    n_shared = [m["n_shared"] for m, _ in recoverable]
    probs = [d["probability"] for _, d in recoverable if d["probability"] is not None]

    def what_if(n_recover: int) -> Optional[dict[str, Any]]:
        if n_recover <= 0:
            return None
        n = min(n_recover, len(recoverable))
        return {
            "n_to_recover": n_recover,
            "achievable": n_recover <= len(recoverable),
            "top_k": _value_recovering(ranks, n, easiest_first=True),
            "min_fingerprints": _value_recovering(n_shared, n, easiest_first=False),
            "min_probability": _value_recovering(probs, n, easiest_first=False),
        }

    allowed = int(np.floor(0.01 * n_pos))
    return {
        "n_pos": n_pos,
        "n_misses": len(rows),
        "n_recoverable": len(recoverable),
        "miss_rate": len(rows) / n_pos,
        "reasons": reasons,
        "thresholds": thresholds,
        "n_shared": _stats(n_shared),
        "rank": _stats(ranks),
        "probability": _stats(probs),
        "to_recover_all": what_if(len(recoverable)),
        "to_reach_recall_99": what_if(len(rows) - allowed) if len(rows) > allowed else None,
    }


def _stats(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()),
    }


def format_misses(misses: list[dict], names: list[str]) -> str:
    """Render the misses of :meth:`Evaluator.diagnose_misses` as text, one block per
    missed positive followed by the reason of each detector that missed it."""
    if not misses:
        return "No missed positive."
    lines = [f"{len(misses)} missed positives"]
    for m in misses:
        rank = m["rank"] if m["rank"] is not None else "-"
        lines.append(
            f"{m['source_doc_id'][:60]:<60} len={m['text_length']:<6} "
            f"fps={m['n_query_fingerprints']:<5} shared={m['n_shared']:<4} "
            f"rank={rank}/{m['n_candidates']} top={m['top_count']} ({m['top_doc']})"
        )
        for name in names:
            d = m["detectors"].get(name)
            if d is None:
                continue
            raw = f"{d['raw_score']:.1f}" if d["raw_score"] is not None else "-"
            prob = f"{d['probability']:.2e}" if d["probability"] is not None else "-"
            lines.append(f"    {name:<14}{d['reason']:<24} raw={raw:<7} p={prob}")
    return "\n".join(lines)


def format_miss_summary(summary: dict[str, dict[str, Any]]) -> str:
    """Render the per-detector summary of :meth:`Evaluator.diagnose_misses`."""
    header = (
        f"{'detector':<14}{'missed':>8}{'rate':>8}"
        + "".join(f"{reason:>24}" for reason in MISS_REASONS)
    )
    lines = [header, "-" * len(header)]
    for name, s in summary.items():
        lines.append(
            f"{name:<14}{s['n_misses']:>8d}{s['miss_rate']:>8.3f}"
            + "".join(f"{s['reasons'][reason]:>24d}" for reason in MISS_REASONS)
        )
    for name, s in summary.items():
        th = s["thresholds"]
        lines.append("")
        lines.append(
            f"{name}: min_fingerprints={th['min_fingerprints']} top_k={th['top_k']} "
            f"min_probability={th['min_probability']:g} "
            f"(N0={th['expected_fingerprints']:.1f}, raw score cut at "
            f"{th['min_raw_score']:.1f})"
        )
        for label, key in (
            ("shared fingerprints", "n_shared"), ("rank", "rank"), ("probability", "probability"),
        ):
            st = s[key]
            if st is not None:
                lines.append(
                    f"    missed {label}: min={st['min']:g} median={st['median']:g} "
                    f"max={st['max']:g}"
                )
        for label, key in (
            ("to recover all misses sharing fingerprints", "to_recover_all"),
            ("to miss at most 1% of positives (P@R=99 off the floor)", "to_reach_recall_99"),
        ):
            w = s[key]
            if w is None:
                lines.append(
                    f"    {label}: "
                    + ("nothing to do" if s["n_misses"] == 0 or key != "to_recover_all"
                       else "no miss shares a fingerprint")
                )
                continue
            parts = []
            if w["top_k"] is not None:
                parts.append(f"top_k >= {int(w['top_k'])}")
            if w["min_fingerprints"] is not None:
                parts.append(f"min_fingerprints <= {int(w['min_fingerprints'])}")
            if w["min_probability"] is not None:
                parts.append(f"min_probability <= {w['min_probability']:.2e}")
            text = " and ".join(parts) or "no recoverable miss"
            if not w["achievable"]:
                text += (
                    f" (not enough: {w['n_to_recover'] - s['n_recoverable']} of the "
                    f"{w['n_to_recover']} misses to recover share no fingerprint)"
                )
            lines.append(f"    {label}: {text}")
    return "\n".join(lines)


def precision_at_recall(
    labels: np.ndarray, scores: np.ndarray, min_recall: float
) -> float:
    """Best precision among the operating points reaching at least ``min_recall``.

    Computed on the precision-recall curve of ``scores`` (higher = more positive).
    A recall of ``min_recall`` is always reachable by accepting everything, so the
    value is well defined (and at worst the positive rate).
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(precision[recall >= min_recall].max())


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """ROC-AUC and precision at 90 % / 99 % recall for one detector.

    Also reports the *score floor*: ``n_pos_missed`` positives and ``n_neg_empty``
    negatives have a score of exactly 0, i.e. the detector returned nothing useful
    for them (source document absent, resp. no document at all).  Those examples
    are indistinguishable, so recall above ``max_recall = 1 - n_pos_missed / n_pos``
    is only reachable by accepting all of them at once, and ``P@R=xx`` collapses to
    the positive rate as soon as ``xx`` exceeds ``max_recall``.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_pos_missed = int((scores[labels] <= 0.0).sum())
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "p_at_r90": precision_at_recall(labels, scores, 0.90),
        "p_at_r99": precision_at_recall(labels, scores, 0.99),
        "n_pos_missed": n_pos_missed,
        "n_neg_empty": int((scores[~labels] <= 0.0).sum()),
        "max_recall": 1.0 - n_pos_missed / n_pos if n_pos else float("nan"),
    }


def floor_warnings(metrics: dict[str, dict[str, float]]) -> list[str]:
    """Explain the ``P@R`` values that sit on the score floor (see
    :func:`compute_metrics`): one message per affected detector, empty if none."""
    messages = []
    for name, m in metrics.items():
        floored = [
            f"P@R={int(r * 100)}"
            for key, r in (("p_at_r90", 0.90), ("p_at_r99", 0.99))
            if key in m and m.get("max_recall", 1.0) < r
        ]
        if floored:
            n_pos, n_neg = m["n_pos"], m["n_neg"]
            messages.append(
                f"{name}: {', '.join(floored)} is the positive rate "
                f"({n_pos}/{n_pos + n_neg}) because {m['n_pos_missed']} positives "
                f"({1 - m['max_recall']:.1%}) have score 0 (source document not "
                f"returned), tied with {m['n_neg_empty']} negatives without any "
                f"result; max recall above the floor is {m['max_recall']:.3f}. "
                "Use Evaluator.diagnose_misses() to see why they are missed."
            )
    return messages


def format_results(metrics: dict[str, dict[str, float]]) -> str:
    """Render the output of :meth:`Evaluator.run` as a text table."""
    first = next(iter(metrics.values()))
    header = (
        f"{'detector':<14}{'ROC-AUC':>9}{'P@R=90':>9}{'P@R=99':>9}{'top-1':>8}"
        f"{'missed':>8}{'score ms':>10}{'retr. ms':>10}"
    )
    lines = [
        f"{first['n_pos']} positives, {first['n_neg']} negatives",
        header,
        "-" * len(header),
    ]
    for name, m in metrics.items():
        retr = m["mean_retrieval_time"]
        retr_s = f"{retr * 1000:>10.1f}" if not np.isnan(retr) else f"{'-':>10}"
        lines.append(
            f"{name:<14}{m['roc_auc']:>9.4f}{m['p_at_r90']:>9.4f}{m['p_at_r99']:>9.4f}"
            f"{m['top1_acc']:>8.3f}{m['n_pos_missed']:>8d}"
            f"{m['mean_score_time'] * 1000:>10.1f}{retr_s}"
        )
    lines.extend(floor_warnings(metrics))
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate detectors on a labelled JSONL file")
    run.add_argument("--index", required=True, help="index directory")
    run.add_argument("--data", required=True, help="evaluation JSONL file")
    run.add_argument("--min-fragment-length", type=float, default=1000.0)
    run.add_argument("--min-probability", type=float, default=1e-3)
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--n-workers", type=int, default=1)
    run.add_argument("--output-scores", help="write per-example scores to this JSONL")
    run.add_argument("--limit", type=int, help="only evaluate the first N examples")

    diag = sub.add_parser(
        "diagnose", help="explain why positives are missed (score 0) by the detectors"
    )
    diag.add_argument("--index", required=True, help="index directory")
    diag.add_argument("--data", required=True, help="evaluation JSONL file")
    diag.add_argument("--min-fragment-length", type=float, default=1000.0)
    diag.add_argument("--min-probability", type=float, default=1e-3)
    diag.add_argument("--top-k", type=int, default=50)
    diag.add_argument("--n-workers", type=int, default=1)
    diag.add_argument("--output", help="write one JSON line per missed positive here")
    diag.add_argument("--limit", type=int, help="only consider the first N examples")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command in ("run", "diagnose"):
        evaluator = Evaluator(
            args.index,
            min_probability=args.min_probability,
            top_k=args.top_k,
            n_workers=args.n_workers,
        )
        if args.command == "run":
            evaluator.run(
                args.data,
                min_fragment_length=args.min_fragment_length,
                output_scores=args.output_scores,
                limit=args.limit,
            )
        else:
            evaluator.diagnose_misses(
                args.data,
                min_fragment_length=args.min_fragment_length,
                limit=args.limit,
                output_path=args.output,
            )


if __name__ == "__main__":
    main()
