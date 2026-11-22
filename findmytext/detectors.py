"""Production interface for text-containment detection.

This module exposes two detector classes that wrap a disk-based winnowing index and
provide different scoring strategies for deciding whether a query document is
(partially) contained in any indexed document:

- :class:`NbSharedFingerprintsDetector`: scores by the number of unique winnowed fingerprints the
  query shares with a candidate document, optionally weighted by fingerprint rarity.
- :class:`FingerprintChainDetector`: scores by the sum of fingerprint weights
  inside the *largest* spatial cluster of shared fingerprints.  Clusters are found
  in the 2-D ``(position_in_query, position_offset)`` space using either the
  rectangle method (axis-aligned position/offset thresholds) or single-linkage
  hierarchical clustering (with Euclidean or Manhattan distance thresholds).

Both detectors expose a ``search`` method returning a ``dict[str, float]`` mapping
each candidate document id to a containment probability in [0, 1].  ``search`` is
the composition of two public steps: ``retrieve`` fetches the query fingerprints and
the matching documents from the index into a :class:`Candidates` object, and
``score_candidates`` turns it into probabilities (via the unnormalised
``raw_scores``: shared fingerprint count or largest-cluster weight).  Several
detectors bound to the same index can thus score one shared retrieval.

"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm

from . import indexing, winnower

# Average number of characters per word (including trailing space) for typical
# English text.  This is the size of one k-gram sliding step and is used to
# convert a fragment length in characters to an expected fingerprint count.
_AVG_CHARS_PER_WORD: float = 5.5


@dataclass(frozen=True)
class Candidates:
    """Result of :meth:`_BaseDetector.retrieve`: a query and its candidate matches.

    Attributes
    ----------
    query : pl.DataFrame
        Fingerprints of the query text, with columns ``hash`` (UInt64) and
        ``position`` (see :meth:`_BaseDetector.get_fingerprints`).
    matches : pl.DataFrame
        One row per (document, shared hash, position in that document), with
        columns ``doc_match_id``, ``doc_match_closeness_rank``, ``hash`` and
        ``position`` (see :func:`convert_closest_matches_with_positions_to_df`).
        Empty when no document shares enough fingerprints.

    """

    query: pl.DataFrame
    matches: pl.DataFrame


class _BaseDetector(ABC):
    """Shared initialisation and helpers for all fingerprint-based detectors.

    Parameters
    ----------
    index : str or indexing.DiskBasedIndex
        Path to the directory containing a disk-based winnowing index
        (``meta.json``, ``fingerprints.npy``, ``postings.dat``, ...), or an index
        already opened (e.g. the result of :func:`findmytext.build_index`).
    top_k : int
        Number of closest candidate documents to retrieve per query (default: 50).
    min_probability : float
        Documents whose containment probability falls below this value are not
        returned by :meth:`search` (default: 0.05). It also sets the minimum number
        of shared fingerprints a document needs to be retrieved from the index at
        all (see :meth:`_min_shared_fingerprints`), which skips the scoring of
        hopeless candidates.
    rarity_weighted : bool
        If ``True``, score by an IDF-weighted fingerprint sum rather than raw count
        (default: ``False``; see :meth:`_get_fingerprint_weights`).
    dispersion : float
        Ratio of the variance to the mean of the shared-fingerprint count, used to
        convert raw scores to probabilities (default: 0.5; see
        :meth:`_normalise_scores`).
    n_workers : int
        Number of threads reading posting lists in parallel when ``index`` is a
        path (default: 1). Parallel reads only pay off on cold or network storage;
        on warm or local data the sequential default is faster. Ignored, with a
        warning, when an already-opened index is passed.
    min_fingerprints : int, optional
        Explicit minimum number of shared fingerprints for a document to be
        retrieved from the index, overriding the value derived from
        ``min_probability`` (default: ``None``, i.e. derived).

    """

    def __init__(
        self,
        index: Union[str, indexing.DiskBasedIndex],
        top_k: int = 50,
        min_probability: float = 0.01,
        rarity_weighted: bool = False,
        dispersion: float = 0.5,
        n_workers: int = 1,
        min_fingerprints: Optional[int] = None,
    ):
        if dispersion <= 0:
            raise ValueError("dispersion must be positive")
        if not 0 < min_probability < 1:
            raise ValueError("min_probability must be strictly between 0 and 1")
        if min_fingerprints is not None and min_fingerprints < 1:
            raise ValueError("min_fingerprints must be at least 1")
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if isinstance(index, indexing.DiskBasedIndex):
            if n_workers != 1:
                warnings.warn(
                    "n_workers is ignored when an already-opened DiskBasedIndex is "
                    "passed; set n_workers when opening the index instead.",
                    stacklevel=2,
                )
            self.index = index
        else:
            self.index = indexing.DiskBasedIndex(index, n_workers=n_workers)
        self.n_workers = n_workers
        self.index_dir = self.index.index_dir
        self.top_k = top_k
        self.min_probability = min_probability
        self.min_fingerprints = min_fingerprints
        self.rarity_weighted = rarity_weighted
        self.dispersion = dispersion
        self.num_documents = self.index.num_documents
        # Queries use every k-gram (window_size=1) hashed like the indexed corpus.
        self.index_metadata = self.index.get_metadata()
        self.runtime_winnower = winnower.Winnower.from_metadata(
            {**self.index_metadata, "window_size": 1}
        )

        # Rarity weights are divided by their occurrence-weighted mean over the index
        # (the expected weight of a fingerprint met in running text), so that a
        # verbatim passage still scores N0 on average; the extra variance of summing
        # unequal weights is folded into the dispersion (see _normalise_scores).
        self._mean_weight = 1.0
        self._score_dispersion = dispersion
        if rarity_weighted:
            counts = _sample_posting_counts(self.index.lengths).astype(np.float64)
            weights = self._idf_weights(counts)
            self._mean_weight = float(np.average(weights, weights=counts))
            mean_sq_weight = float(np.average(weights**2, weights=counts))
            self._score_dispersion = dispersion + mean_sq_weight / self._mean_weight**2 - 1.0

    def search(self, text: str, min_fragment_length: float = 1000.0) -> dict[str, float]:
        """Find the indexed documents likely to contain a passage of ``text``.

        Equivalent to :meth:`score_candidates` applied to :meth:`retrieve`.

        Parameters
        ----------
        text : str
            The query document to search for.
        min_fragment_length : float
            Minimum length of a matching fragment in characters (default 1 000).
            Used to derive the expected fingerprint count :math:`N_0` (see
            :meth:`_normalise_scores`).

        Returns
        -------
        dict[str, float]
            Mapping from external document id to containment probability, for the
            documents whose probability is at least ``min_probability``.

        """
        candidates = self.retrieve(text, min_fragment_length)
        return self.score_candidates(candidates, min_fragment_length)

    def retrieve(
        self,
        text: str,
        min_fragment_length: float = 1000.0,
        *,
        top_k: Optional[int] = None,
        min_fingerprints: Optional[int] = None,
    ) -> Candidates:
        """Fingerprint ``text`` and fetch the candidate documents worth scoring.

        Parameters
        ----------
        text : str
            The query document.
        min_fragment_length : float
            Minimum length of a matching fragment in characters (default 1 000);
            sets the default retrieval threshold (see :meth:`_min_shared_fingerprints`).
        top_k : int, optional
            Number of closest documents to retrieve (default: the detector's
            ``top_k``).
        min_fingerprints : int, optional
            Minimum number of shared fingerprints for a document to be retrieved
            (default: derived from ``min_fragment_length``).

        A larger ``top_k`` or smaller ``min_fingerprints`` than the defaults only
        adds candidates, which :meth:`score_candidates` scores and drops below
        ``min_probability`` as usual; this lets several detectors bound to the same
        index share one retrieval.

        Returns
        -------
        Candidates
            The query fingerprints and the candidate-match table.

        """
        if len(text) < min_fragment_length:
            warnings.warn(
                f"Query ({len(text)} chars) is shorter than min_fragment_length "
                f"({min_fragment_length} chars)",
                UserWarning,
                stacklevel=3,
            )
        query = self.get_fingerprints(text)
        if min_fingerprints is None:
            min_fingerprints = self._min_shared_fingerprints(min_fragment_length)
        closest = self.index.get_closest_matches_with_positions(
            query=query["hash"],
            top_k=self.top_k if top_k is None else top_k,
            min_fingerprints=min_fingerprints,
            verbose=False,
        )
        return Candidates(query, convert_closest_matches_with_positions_to_df(closest))

    def score_candidates(
        self, candidates: Candidates, min_fragment_length: float = 1000.0
    ) -> dict[str, float]:
        """Score retrieved candidate documents against their query.

        The raw scores come from :meth:`raw_scores` and are converted with
        :meth:`_normalise_scores`; documents below ``min_probability`` are dropped.

        Parameters
        ----------
        candidates : Candidates
            Output of :meth:`retrieve` (possibly from another detector on the
            same index).
        min_fragment_length : float
            Minimum length of a matching fragment in characters (default 1 000).

        Returns
        -------
        dict[str, float]
            Mapping from external document id to containment probability, for the
            documents whose probability is at least ``min_probability``.

        """
        return self._to_probabilities(self.raw_scores(candidates), min_fragment_length)

    @abstractmethod
    def raw_scores(self, candidates: Candidates) -> dict[str, float]:
        """Unnormalised score of every candidate document against the query.

        This is the quantity :math:`n` of :meth:`_normalise_scores` (a count, or a
        sum of fingerprint weights with ``rarity_weighted=True``), before any
        conversion to a probability or ``min_probability`` filtering.  Useful for
        debugging: it tells how far a document is from the detection threshold.

        Parameters
        ----------
        candidates : Candidates
            Output of :meth:`retrieve`.

        Returns
        -------
        dict[str, float]
            Mapping from external document id to raw score, for every document of
            ``candidates.matches`` (empty if the table is empty).

        """
        raise NotImplementedError

    def get_fingerprints(self, text: str) -> pl.DataFrame:
        """Return the winnowed fingerprints of ``text`` as a Polars DataFrame.

        The returned DataFrame has columns ``hash`` and ``position``.
        """
        fingerprints, fingerprints_pos = (
            self.runtime_winnower.get_winnowed_fingerprints(text)
        )
        return pl.DataFrame({"hash": fingerprints, "position": fingerprints_pos}).with_columns(
            pl.col("hash").cast(pl.UInt64)
        )

    def _to_probabilities(
        self, raw_scores: dict[str, float], min_fragment_length: float
    ) -> dict[str, float]:
        """Convert raw scores to probabilities and drop those below ``min_probability``."""
        probabilities = self._normalise_scores(raw_scores, min_fragment_length)
        return {doc: p for doc, p in probabilities.items() if p >= self.min_probability}

    def _min_raw_score(self, min_fragment_length: float) -> float:
        """Raw score at which :meth:`_normalise_scores` reaches ``min_probability``.

        Inverts the normal model: documents with a smaller raw score end up below
        ``min_probability`` and are dropped by :meth:`score_candidates`.
        """
        N0 = self._expected_fingerprints(min_fragment_length)
        return N0 - 0.5 + norm.ppf(self.min_probability) * np.sqrt(self._score_dispersion * N0)

    def _min_shared_fingerprints(self, min_fragment_length: float) -> int:
        """Smallest shared-fingerprint count worth scoring for this fragment length.

        Inverts :meth:`_normalise_scores` (see :meth:`_min_raw_score`): a document
        with fewer shared fingerprints would end up below ``min_probability`` even
        if all of them formed one passage, so the index need not return it.  With
        rarity weighting the score is a weighted sum, so the count is scaled down
        by the largest normalised weight to remain a lower bound.
        ``min_fingerprints`` overrides the derived value.
        """
        if self.min_fingerprints is not None:
            return self.min_fingerprints
        min_score = self._min_raw_score(min_fragment_length)
        if self.rarity_weighted:
            max_weight = float(self._idf_weights(np.array(1.0))) / self._mean_weight
            min_score /= max(max_weight, 1.0)
        return max(1, int(np.ceil(min_score)))

    def _expected_fingerprints(self, min_fragment_length: float, edit_rate: float = 0.1) -> float:
        """Expected shared-fingerprint count :math:`N_0` of a verbatim passage of
        ``min_fragment_length`` characters (see :meth:`_normalise_scores`)."""
        k = self.index_metadata["length"]
        w = self.index_metadata["window_size"]
        n_kgrams = max(min_fragment_length / _AVG_CHARS_PER_WORD - k + 1, 1.0)
        corrected_ngrams = (1- edit_rate) ** k * n_kgrams
        return corrected_ngrams * 2 / (w + 1)

    def _idf_weights(self, posting_counts: np.ndarray) -> np.ndarray:
        """Unnormalised rarity weight ``log(1 + N / cf)`` of fingerprints with the
        given (positive) posting counts ``cf``, ``N`` being the number of documents."""
        return np.log1p(self.num_documents / posting_counts)

    def _get_fingerprint_weights(self, fingerprints) -> dict[int, float]:
        """Per-fingerprint weights for scoring: uniform (1.0) or IDF-like.

        Returns a ``{hash: weight}`` mapping over the fingerprints present in the
        index.  When ``rarity_weighted=False`` every fingerprint gets weight 1.0.
        When ``rarity_weighted=True`` the weight is ``log(1 + N / cf(h))``, with
        ``N`` the number of indexed documents and ``cf(h)`` the number of postings
        of ``h``, divided by the occurrence-weighted mean of that quantity over the
        index so that a fingerprint met in running text has weight 1.0 on average.
        """
        hashes = np.unique(np.asarray(fingerprints, dtype=np.uint64))
        if not self.rarity_weighted:
            return dict.fromkeys(hashes.tolist(), 1.0)
        counts = self.index.get_posting_counts(hashes)
        found = counts > 0
        weights = self._idf_weights(counts[found].astype(np.float64)) / self._mean_weight
        return dict(zip(hashes[found].tolist(), weights.tolist()))

    def _normalise_scores(
        self, raw_scores: dict[str, float], min_fragment_length: float
    ) -> dict[str, float]:
        r"""Convert raw fingerprint scores to containment probabilities.

        **Model.**  Let :math:`n` be the observed score (number of shared fingerprints,
        or sum of their weights) and :math:`L_0` = ``min_fragment_length`` in
        characters.  A passage of :math:`L_0` characters holds about
        :math:`L_0 / \bar{c} - k + 1` k-grams, with :math:`\bar{c} \approx 5.5`
        characters per word, and winnowing with window :math:`w` keeps on average a
        fraction :math:`2 / (w + 1)` of them.  The expected number of shared
        fingerprints for a verbatim passage of exactly :math:`L_0` characters is thus

        .. math::
            N_0 = \left(\frac{L_0}{\bar{c}} - k + 1\right) \frac{2}{w + 1}

        (:math:`N_0 \approx 51` for :math:`L_0 = 1000`, :math:`k = 4`, :math:`w = 6`).

        The count varies much less than a Poisson variable with the same mean, because
        winnowing selects fingerprints regularly: every window contains at least one and
        neighbouring k-grams are rarely both selected.  Simulations give a variance of
        0.2–0.6 times the mean depending on the edit rate; ``dispersion`` (default 0.5,
        slightly above the simulated values since real text varies more) sets this
        ratio.  With a normal approximation and continuity correction,

        .. math::
            P(\text{match} \mid n) = \Phi\!\left(
                \frac{n + 0.5 - N_0}{\sqrt{\text{dispersion} \cdot N_0}}\right)

        so that :math:`P \approx 0.5` at :math:`n = N_0` and the transition spans about
        :math:`\pm 2\sqrt{\text{dispersion} \cdot N_0}` (:math:`\pm 10` fingerprints
        in the example above).

        **Calibration.**  :math:`N_0` assumes a verbatim passage.  Edits remove shared
        fingerprints (a k-gram survives independent word edits with probability
        :math:`(1 - \epsilon)^k`), so an edited passage needs to be proportionally
        longer to reach the same probability: with 5% of words changed, about 1250
        characters instead of 1000.

        **Rarity weighting.**  With ``rarity_weighted=True`` the score is a sum of
        weights :math:`W` normalised to mean 1 over fingerprint occurrences, so its
        mean is still :math:`N_0`, but a random sum of unequal terms has variance
        :math:`\mathrm{Var}(n)\,E[W]^2 + E[n]\,\mathrm{Var}(W)`, i.e. the effective
        dispersion becomes :math:`\text{dispersion} + E[W^2] - 1`; both moments are
        estimated from the index's posting counts when the detector is created.

        Parameters
        ----------
        raw_scores : dict[str, float]
            Mapping from ``doc_match_id`` to the raw fingerprint score.
        min_fragment_length : float
            Matching-passage length threshold :math:`L_0` in characters.

        Returns
        -------
        dict[str, float]
            Mapping from ``doc_match_id`` to containment probability in :math:`[0, 1]`.

        """
        N0 = self._expected_fingerprints(min_fragment_length)
        scores = np.array(list(raw_scores.values()), dtype=float)
        probs = norm.cdf((scores + 0.5 - N0) / np.sqrt(self._score_dispersion * N0))
        return dict(zip(raw_scores.keys(), probs.tolist()))


def _sample_posting_counts(
    lengths: np.ndarray, n_chunks: int = 1000, chunk_size: int = 10_000
) -> np.ndarray:
    """Posting counts of up to n_chunks * chunk_size fingerprints, read as evenly
    spaced contiguous chunks of the (memory-mapped) lengths array.

    Fingerprints are stored in hash order, which is unrelated to their frequency,
    so the chunks form an unbiased sample at a tiny fraction of the I/O of a full
    pass over a multi-GB array.
    """
    n = len(lengths)
    if n <= n_chunks * chunk_size:
        return np.asarray(lengths)
    starts = np.linspace(0, n - chunk_size, n_chunks).astype(np.int64)
    return np.concatenate([lengths[s : s + chunk_size] for s in starts])


class NbSharedFingerprintsDetector(_BaseDetector):
    """Detect text containment by counting shared unique fingerprints.

    Scores each candidate document by the sum of fingerprint weights over its
    unique shared fingerprints — either the raw count (uniform weights, default)
    or a rarity-weighted sum (IDF-like weights, see
    :meth:`_BaseDetector._get_fingerprint_weights`).

    Parameters
    ----------
    index : str or indexing.DiskBasedIndex
        Path to the disk-based winnowing index directory, or an opened index.
    top_k : int
        Number of closest candidate documents to retrieve per query (default: 50).
    min_probability : float
        Documents below this containment probability are not returned, nor even
        scored (default: 0.05; see :class:`_BaseDetector`).
    rarity_weighted : bool
        If ``True``, score by rarity-weighted fingerprint sum rather than raw count
        (default: ``False``).
    n_workers : int
        Number of threads reading posting lists in parallel
        when ``index`` is a path (default: 1; see :class:`_BaseDetector`).
    min_fingerprints : int, optional
        Explicit minimum shared-fingerprint count for retrieval, overriding the
        value derived from ``min_probability`` (default: ``None``).

    """

    def raw_scores(self, candidates: Candidates) -> dict[str, float]:
        """Total weight of the unique fingerprints each candidate shares with the
        query (the plain count when ``rarity_weighted=False``).  See
        :meth:`_BaseDetector.raw_scores` for the parameters.
        """
        df_query, df_candidates = candidates.query, candidates.matches
        if df_candidates.height == 0:
            return {}

        # Compute the weight of each fingerprint in the query (uniform or rarity-weighted).
        fingerprint_weights = self._get_fingerprint_weights(df_query["hash"])
        weights_df = pl.DataFrame(
            list(fingerprint_weights.items()),
            schema={"hash": pl.UInt64, "weight": pl.Float64},
            orient="row",
        )

        # Compute the total weight of unique shared fingerprints for each candidate document.
        df_unique = (
            df_candidates.select(["doc_match_id", "hash"])
            .unique(maintain_order=True)
            .join(weights_df, on="hash", how="left")
            .with_columns(pl.col("weight").fill_null(0.0))
        )

        # Aggregate the weights by candidate document to get the raw score for each candidate.
        df_scores = df_unique.group_by("doc_match_id", maintain_order=True).agg(
            pl.col("weight").sum().alias("raw_score")
        )
        return dict(zip(df_scores["doc_match_id"], df_scores["raw_score"]))


class FingerprintChainDetector(_BaseDetector):
    """Detect text containment by clustering shared fingerprints in position space.

    Scores each candidate document by the sum of fingerprint weights inside the
    *largest* spatial cluster of shared fingerprints.
    Clusters are found in the 2-D ``(position_in_query, position_offset)`` space using
    either the rectangle method (axis-aligned thresholds) or single-linkage hierarchical
    clustering (Euclidean, Manhattan, or Chebyshev distance).

    Parameters
    ----------
    index : str or indexing.DiskBasedIndex
        Path to the disk-based winnowing index directory, or an opened index.
    top_k : int
        Number of closest candidate documents to retrieve per query (default: 50).
    min_probability : float
        Documents below this containment probability are not returned, nor even
        scored (default: 0.05; see :class:`_BaseDetector`).
    method : str
        Clustering method: ``"rectangle"`` (default), ``"euclidean"``,
        ``"cityblock"``, or ``"chebyshev"``.
    position_threshold : int
        Maximum difference in query position between neighbours (rectangle only,
        default: 30).
    offset_threshold : int
        Maximum difference in position offset between neighbours (rectangle only,
        default: 30).
    distance_threshold : int
        Maximum single-linkage distance (euclidean / cityblock / chebyshev only,
        default: 30).
    min_cluster_size : int
        Minimum number of fingerprints for a cluster to be kept; smaller clusters
        are labelled noise (default: 5).
    rarity_weighted : bool
        If ``True``, score by the largest rarity-weighted cluster sum rather than
        the largest cluster count (default: ``False``).
    n_workers : int
        Number of threads reading posting lists in parallel
        when ``index`` is a path (default: 1; see :class:`_BaseDetector`).
    min_fingerprints : int, optional
        Explicit minimum shared-fingerprint count for retrieval, overriding the
        value derived from ``min_probability`` (default: ``None``).

    """

    def __init__(
        self,
        index: Union[str, indexing.DiskBasedIndex],
        top_k: int = 50,
        min_probability: float = 0.01,
        rarity_weighted: bool = False,
        method: str = "rectangle",
        position_threshold: int = 30,
        offset_threshold: int = 30,
        distance_threshold: int = 30,
        min_cluster_size: int = 5,
        n_workers: int = 1,
        min_fingerprints: Optional[int] = None,
    ):
        super().__init__(
            index,
            top_k=top_k,
            min_probability=min_probability,
            rarity_weighted=rarity_weighted,
            n_workers=n_workers,
            min_fingerprints=min_fingerprints,
        )
        self.method = method
        self.position_threshold = position_threshold
        self.offset_threshold = offset_threshold
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    def raw_scores(self, candidates: Candidates) -> dict[str, float]:
        """Weight sum of the largest cluster of fingerprints each candidate shares
        with the query, clustering in the 2-D ``(position_query, offset)`` space
        (0.0 when no cluster survives ``min_cluster_size``).  See
        :meth:`_BaseDetector.raw_scores` for the parameters.
        """
        df_query, df_candidates = candidates.query, candidates.matches
        if df_candidates.height == 0:
            return {}

        fingerprint_weights = self._get_fingerprint_weights(df_query["hash"])

        raw_scores = {}
        for doc_id in df_candidates["doc_match_id"].unique(maintain_order=True):
            raw_scores[doc_id] = self._score_doc(
                df_query, df_candidates, fingerprint_weights, doc_id
            )
        return raw_scores

    def _score_doc(
        self,
        df_query: pl.DataFrame,
        df_candidates: pl.DataFrame,
        fingerprint_weights: dict[int, float],
        doc_id: str,
    ) -> float:
        """Cluster the shared fingerprints for one candidate doc.

        Returns the maximum per-cluster sum of fingerprint weights, or 0.0 when
        no cluster survives the ``min_cluster_size`` filter.
        """
        df_match = (
            df_candidates.filter(pl.col("doc_match_id") == doc_id)
            .select(["hash", "position"])
            .with_columns(
                pl.col("hash")
                .map_elements(
                    lambda h: fingerprint_weights.get(h, 0.0), return_dtype=pl.Float64
                )
                .alias("weight")
            )
        )

        # Shared fingerprints with their positions in both documents and the offset.
        df_shared = (
            df_query.join(df_match, on="hash", how="inner", suffix="_doc2")
            .select(
                [
                    pl.col("hash"),
                    pl.col("weight"),
                    pl.col("position").alias("position_doc1"),
                    pl.col("position_doc2"),
                ]
            )
            .with_columns(
                (pl.col("position_doc2") - pl.col("position_doc1")).alias(
                    "position_offset"
                )
            )
            .sort("position_doc1")
        )

        if df_shared.height == 0:
            return 0.0

        if self.method == "rectangle":
            df_clustered = self._get_df_hash_cluster_rectangle(df_shared)
        elif self.method in {"euclidean", "cityblock", "chebyshev"}:
            df_clustered = self._get_df_hash_cluster_linkage(df_shared)
        else:
            raise ValueError(
                f"Unknown clustering method '{self.method}'; must be one of "
                "'rectangle', 'euclidean', 'cityblock', or 'chebyshev'."
            )

        df_clusters = (
            df_clustered.filter(pl.col("cluster_id") != -1)
            .group_by("cluster_id", maintain_order=True)
            .agg(
                pl.struct(["hash", "weight"])
                .unique()
                .struct.field("weight")
                .sum()
                .alias("score")
            )
        )

        if df_clusters.height == 0:
            return 0.0

        return float(df_clusters["score"].max())

    def _get_df_hash_cluster_rectangle(
        self, df_shared_hashes: pl.DataFrame
    ) -> pl.DataFrame:
        """Cluster shared fingerprints using rectangle-neighbourhood connected components.

        Two points are *connected* iff both
        ``|Δposition_doc1| ≤ position_threshold`` and
        ``|Δposition_offset| ≤ offset_threshold``.
        Connected components form initial clusters; those with fewer than
        ``min_cluster_size`` unique members are relabelled −1 (noise).

        Parameters
        ----------
        df_shared_hashes : pl.DataFrame
            Shared-hash DataFrame with at least columns ``position_doc1`` and
            ``position_offset``.

        Returns
        -------
        pl.DataFrame
            Input DataFrame with an added ``cluster_id`` column (Int32; −1 = noise).

        """
        pos1 = df_shared_hashes["position_doc1"].to_numpy()
        offsets = df_shared_hashes["position_offset"].to_numpy()
        diff_pos1 = np.abs(pos1[:, None] - pos1[None, :])
        diff_offsets = np.abs(offsets[:, None] - offsets[None, :])

        _, raw_labels = connected_components(
            csr_matrix(
                (diff_pos1 <= self.position_threshold)
                & (diff_offsets <= self.offset_threshold)
            ),
            directed=False,
        )
        cluster_ids = _relabel_small_clusters(raw_labels, self.min_cluster_size)
        return df_shared_hashes.with_columns(pl.Series("cluster_id", cluster_ids))

    def _get_df_hash_cluster_linkage(
        self, df_shared_hashes: pl.DataFrame
    ) -> pl.DataFrame:
        """Cluster shared fingerprints via single-linkage hierarchical clustering.

        Clustering is performed on the 2-D feature space ``(position_doc1,
        position_offset)`` using ``scipy.cluster.hierarchy.linkage`` with
        ``method="single"`` and the given distance metric.

        .. note::
           Single linkage with the Chebyshev metric and ``distance_threshold`` *t*
           is mathematically equivalent to
           :meth:`_get_df_hash_cluster_rectangle` with
           ``position_threshold = offset_threshold = t``.

        Parameters
        ----------
        df_shared_hashes : pl.DataFrame
            Shared-hash DataFrame with at least columns ``position_doc1`` and
            ``position_offset``.

        Returns
        -------
        pl.DataFrame
            Input DataFrame with an added ``cluster_id`` column (Int32; −1 = noise).

        """
        n = len(df_shared_hashes)

        if n == 0:
            raw_labels = np.empty(0, dtype=np.int32)
        elif n == 1:
            raw_labels = np.array([1], dtype=np.int32)
        else:
            pos1 = df_shared_hashes["position_doc1"].to_numpy()
            offsets = df_shared_hashes["position_offset"].to_numpy()
            X = np.column_stack([pos1, offsets]).astype(np.float64)
            Z = linkage(X, method="single", metric=self.method, optimal_ordering=False)
            raw_labels = fcluster(Z, t=self.distance_threshold, criterion="distance")

        cluster_ids = _relabel_small_clusters(raw_labels, self.min_cluster_size)
        return df_shared_hashes.with_columns(pl.Series("cluster_id", cluster_ids))


def _relabel_small_clusters(
    raw_labels: np.ndarray, min_cluster_size: int
) -> np.ndarray:
    """Re-index valid clusters from 0 and label small ones as −1 (noise).

    Parameters
    ----------
    raw_labels : np.ndarray
        Cluster labels as returned by a clustering algorithm (1-D integer array).
    min_cluster_size : int
        Clusters with fewer members than this are relabelled −1.

    Returns
    -------
    np.ndarray
        Integer array of the same length; valid clusters numbered 0, 1, 2, …;
        noise points set to −1.

    """
    result = np.full(len(raw_labels), -1, dtype=np.int32)
    unique, counts = np.unique(raw_labels, return_counts=True)
    new_id = 0
    for label, count in zip(unique, counts):
        if count >= min_cluster_size:
            result[raw_labels == label] = new_id
            new_id += 1
    return result


def convert_closest_matches_with_positions_to_df(
    closest_with_positions: dict,
) -> pl.DataFrame:
    """Convert ``{doc_id: {hash: [positions]}}`` into a tidy Polars DataFrame.

    Parameters
    ----------
    closest_with_positions : dict
        Mapping returned by the index's ``get_closest_matches_with_positions``.

    Returns
    -------
    pl.DataFrame
        One row per (doc, hash, position) with columns ``doc_match_id``,
        ``doc_match_closeness_rank``, ``hash`` (UInt64), ``position`` (Int32).
        Returns an empty DataFrame with the correct schema when the input is
        empty.

    """
    if not closest_with_positions:
        return pl.DataFrame(
            {
                "doc_match_id": pl.Series([], dtype=pl.Utf8),
                "doc_match_closeness_rank": pl.Series([], dtype=pl.Int32),
                "hash": pl.Series([], dtype=pl.UInt64),
                "position": pl.Series([], dtype=pl.Int32),
            }
        )

    frames = []
    for rank, (doc_id, hash_positions) in enumerate(
        closest_with_positions.items(), start=1
    ):
        # Flatten {hash: [positions]} directly rather than via explode(): no empty
        # lists are involved, so the frame is simpler and Polars-version independent.
        hashes = [h for h, positions in hash_positions.items() for _ in positions]
        positions = [p for positions in hash_positions.values() for p in positions]
        frames.append(
            pl.DataFrame(
                {
                    "doc_match_id": pl.Series([doc_id] * len(hashes), dtype=pl.Utf8),
                    "doc_match_closeness_rank": pl.Series([rank] * len(hashes), dtype=pl.Int32),
                    "hash": pl.Series(hashes, dtype=pl.UInt64),
                    "position": pl.Series(positions, dtype=pl.Int32),
                }
            )
        )
    return pl.concat(frames)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for the text-containment detector."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="detector",
        description=(
            "Score a query document against a winnowing index and print "
            "containment probabilities for the top matching documents."
        ),
    )

    parser.add_argument(
        "index_dir",
        help="Path to the disk-based winnowing index directory.",
    )

    # ── Query text ──────────────────────────────────────────────────────── #
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "--text",
        metavar="TEXT",
        help="Query text passed as a command-line string.",
    )
    query_group.add_argument(
        "--text-file",
        metavar="FILE",
        help="Path to a file containing the query text; use '-' for stdin.",
    )

    # ── Detector selection ───────────────────────────────────────────────── #
    parser.add_argument(
        "--detector",
        choices=["nb_shared", "fingerprint_chain"],
        default="fingerprint_chain",
        help="Scoring strategy (default: fingerprint_chain).",
    )

    # ── Shared parameters ───────────────────────────────────────────────── #
    parser.add_argument(
        "--min-fragment-length",
        type=float,
        default=1000.0,
        metavar="CHARS",
        help="Minimum matching-fragment length in characters (default: 1000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        metavar="K",
        help="Number of candidate documents to retrieve from the index (default: 50).",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.05,
        metavar="P",
        help="Drop documents below this containment probability (default: 0.05).",
    )
    parser.add_argument(
        "--min-fingerprints",
        type=int,
        default=None,
        metavar="N",
        help="Explicit minimum shared fingerprints for retrieval (default: derived from --min-probability).",
    )
    parser.add_argument(
        "--rarity-weighted",
        action="store_true",
        help="Weight fingerprints by rarity (IDF-like) instead of counting uniformly.",
    )

    # ── FingerprintChainDetector parameters ─────────────────────────────── #
    chain = parser.add_argument_group("FingerprintChainDetector options")
    chain.add_argument(
        "--method",
        choices=["rectangle", "euclidean", "cityblock", "chebyshev"],
        default="rectangle",
        help="Clustering method (default: rectangle).",
    )
    chain.add_argument(
        "--position-threshold",
        type=int,
        default=30,
        metavar="N",
        help="Rectangle: max |Δposition_query| between neighbours (default: 30).",
    )
    chain.add_argument(
        "--offset-threshold",
        type=int,
        default=30,
        metavar="N",
        help="Rectangle: max |Δposition_offset| between neighbours (default: 30).",
    )
    chain.add_argument(
        "--distance-threshold",
        type=int,
        default=30,
        metavar="N",
        help="Linkage: max single-linkage distance for merging clusters (default: 30).",
    )
    chain.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        metavar="N",
        help="Minimum cluster size; smaller clusters are discarded (default: 5).",
    )

    # ── Resources ───────────────────────────────────────────────────────── #
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of threads reading posting lists in parallel "
            "(default: 1; only helps on cold or network storage)."
        ),
    )

    # ── Output ──────────────────────────────────────────────────────────── #
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of top results to display (default: 10; 0 = all).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    args = parser.parse_args()

    # ── Read query text ──────────────────────────────────────────────────── #
    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        src = (
            sys.stdin
            if args.text_file == "-"
            else open(args.text_file, encoding="utf-8")
        )
        text = src.read()
    else:
        text = sys.stdin.read()

    # ── Build detector ───────────────────────────────────────────────────── #
    if args.detector == "nb_shared":
        detector = NbSharedFingerprintsDetector(
            args.index_dir,
            top_k=args.top_k,
            min_probability=args.min_probability,
            min_fingerprints=args.min_fingerprints,
            rarity_weighted=args.rarity_weighted,
            n_workers=args.n_workers,
        )
    else:
        detector = FingerprintChainDetector(
            args.index_dir,
            top_k=args.top_k,
            min_probability=args.min_probability,
            min_fingerprints=args.min_fingerprints,
            rarity_weighted=args.rarity_weighted,
            n_workers=args.n_workers,
            method=args.method,
            position_threshold=args.position_threshold,
            offset_threshold=args.offset_threshold,
            distance_threshold=args.distance_threshold,
            min_cluster_size=args.min_cluster_size,
        )

    # ── Score ────────────────────────────────────────────────────────────── #
    scores = detector.search(text, min_fragment_length=args.min_fragment_length)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if args.top_n > 0:
        ranked = ranked[: args.top_n]

    # ── Output ───────────────────────────────────────────────────────────── #
    if args.format == "json":
        print(json.dumps(dict(ranked), indent=2))
    else:
        if not ranked:
            print("No matches found.")
        else:
            print(f"{'Rank':<6} {'Score':>8}  Document ID")
            print(f"{'----':<6} {'-------':>8}  -----------")
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                print(f"{rank:<6} {score:>8.4f}  {doc_id}")


if __name__ == "__main__":
    main()
