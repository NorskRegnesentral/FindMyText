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
positives.

Typical use::

    python experiments/eval.py run --index my_index --data eval_set.jsonl

or, from Python::

    evaluator = Evaluator("my_index")
    results = evaluator.run("eval_set.jsonl")

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
import sys
import time
from typing import Any, Iterable, Optional, Union

import numpy as np
import orjson
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
    detectors can be registered with :meth:`add_detector`.  Without an index
    (``index=None``) no default detector is built, which allows evaluating external
    detectors (e.g. :class:`baselines.DenseRetrievalDetector`) on their own.

    Parameters
    ----------
    index : str or indexing.DiskBasedIndex, optional
        Directory of a disk-based winnowing index, or an already-opened index.
        ``None`` for an evaluator with only externally added detectors.
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
        index: Union[str, indexing.DiskBasedIndex, None],
        min_probability: float = 1e-3,
        top_k: int = 50,
        n_workers: int = 1,
    ):
        self.detectors: dict[str, Any] = {}
        if index is None:
            self.index = None
            self.index_dir = None
            return
        if isinstance(index, indexing.DiskBasedIndex):
            self.index = index
        else:
            self.index = indexing.DiskBasedIndex(index, n_workers=n_workers)
        self.index_dir = self.index.index_dir

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
        if self.index is None:
            return []
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
            ``max_recall`` the recall reachable without accepting them all.
            ``mean_retrieval_time`` is the per-example index query time shared by
            the detectors on the fast path (``NaN`` for the others, whose
            ``mean_score_time`` is then the full ``search`` time).

        """
        if not self.detectors:
            raise ValueError("no detector registered (see add_detector)")
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


def _argmax(probs: dict[str, float]) -> tuple[Optional[str], float]:
    """Highest-probability document of a detector result (``(None, 0.0)`` if empty)."""
    if not probs:
        return None, 0.0
    doc = max(probs, key=probs.get)
    return doc, float(probs[doc])


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
                f"result; max recall above the floor is {m['max_recall']:.3f}."
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

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        evaluator = Evaluator(
            args.index,
            min_probability=args.min_probability,
            top_k=args.top_k,
            n_workers=args.n_workers,
        )
        evaluator.run(
            args.data,
            min_fragment_length=args.min_fragment_length,
            output_scores=args.output_scores,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
