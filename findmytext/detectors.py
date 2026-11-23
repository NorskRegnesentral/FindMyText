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

Everything is plain NumPy: the index returns one flat ``(doc, hash, position)``
table (:class:`findmytext.indexing.Hits`), the query is kept sorted by hash, and
joining the two is a ``searchsorted`` of the hits into the query (every hit's hash
is a query hash by construction).

"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import norm

from . import indexing, winnower
from .indexing import Hits

# Average number of characters per word (including trailing space) for typical
# English text.  This is the size of one k-gram sliding step and is used to
# convert a fragment length in characters to an expected fingerprint count.
_AVG_CHARS_PER_WORD: float = 5.5

_LINKAGE_METRICS = {"euclidean", "cityblock", "chebyshev"}


@dataclass(frozen=True, eq=False)
class Candidates:
    """Result of :meth:`_BaseDetector.retrieve`: a query and its candidate matches.

    Build with :meth:`from_query`, which sorts the query by hash so that
    :meth:`join` can pair hits with query positions by binary search.

    Attributes
    ----------
    hashes : np.ndarray
        uint64 fingerprints of the query text, one per k-gram, sorted (with repeats
        when a k-gram occurs several times in the query).
    positions : np.ndarray
        int64 position (k-gram index) of each query fingerprint, aligned with ``hashes``.
    hits : Hits
        Every occurrence of a query fingerprint in each candidate document, grouped
        by document in closeness order (see :class:`findmytext.indexing.Hits`).
        Empty when no document shares enough fingerprints.

    """

    hashes: np.ndarray
    positions: np.ndarray
    hits: Hits = field(default_factory=Hits)

    @classmethod
    def from_query(
        cls, hashes: np.ndarray, positions: np.ndarray, hits: Optional[Hits] = None
    ) -> Candidates:
        """Build from the aligned ``(hashes, positions)`` of a query in text order."""
        hashes = np.asarray(hashes, dtype=np.uint64)
        order = np.argsort(hashes, kind="stable")
        return cls(
            hashes[order],
            np.asarray(positions, dtype=np.int64)[order],
            Hits() if hits is None else hits,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.hits) == 0

    def join(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pair every hit of ``block`` (a document's rows of ``hits.table``) with
        every query position of its hash, like an inner join on the hash.

        Returns aligned arrays ``(fingerprint, pos_query, pos_doc)`` with one entry
        per pair.  ``fingerprint`` is the index into ``hashes`` of the first query
        occurrence of the hash: an id of the fingerprint that also indexes any
        array aligned with ``hashes`` (e.g. weights).  Positions are int64.
        """
        lo = np.searchsorted(self.hashes, block["hash"], side="left")
        hi = np.searchsorted(self.hashes, block["hash"], side="right")
        reps = hi - lo  # query occurrences of each hit's hash
        hit = np.repeat(np.arange(len(block)), reps)
        # Row r of hit i (r = 0 .. reps[i] - 1) takes the r-th query occurrence.
        query = lo[hit] + np.arange(len(hit)) - np.repeat(np.cumsum(reps) - reps, reps)
        return lo[hit], self.positions[query], block["position"][hit].astype(np.int64)

    def shared_fingerprints(self, block: np.ndarray) -> np.ndarray:
        """Fingerprint ids (as in :meth:`join`) of the distinct hashes in ``block``."""
        return np.unique(np.searchsorted(self.hashes, block["hash"]))


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
        (default: ``False``; see :class:`RarityWeighting`).
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

        # Rarity weights are calibrated once from the index so that scores stay on
        # the scale of a plain count (see RarityWeighting); summing unequal weights
        # widens the score distribution, which the dispersion accounts for.
        self.weighting = RarityWeighting(self.index) if rarity_weighted else None
        self._score_dispersion = dispersion + (
            self.weighting.extra_dispersion if self.weighting else 0.0
        )

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
        query_hash, query_pos = self.get_fingerprints(text)
        if min_fingerprints is None:
            min_fingerprints = self._min_shared_fingerprints(min_fragment_length)
        hits = self.index.get_closest_hits(
            query=query_hash,
            top_k=self.top_k if top_k is None else top_k,
            min_fingerprints=min_fingerprints,
            verbose=False,
        )
        return Candidates.from_query(query_hash, query_pos, hits)

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
            ``candidates.hits`` (empty if there are no hits).

        """
        raise NotImplementedError

    def get_fingerprints(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the fingerprints of ``text`` as aligned ``(hashes, positions)``
        arrays (uint64 and int64), one entry per k-gram."""
        hashes, positions = self.runtime_winnower.get_winnowed_fingerprints(text)
        return np.asarray(hashes, dtype=np.uint64), np.asarray(positions, dtype=np.int64)

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
        if self.weighting:
            min_score /= max(self.weighting.max, 1.0)
        return max(1, int(np.ceil(min_score)))

    def _expected_fingerprints(self, min_fragment_length: float, edit_rate: float = 0.1) -> float:
        """Expected shared-fingerprint count :math:`N_0` of a passage of
        ``min_fragment_length`` characters with a word-edit rate of ``edit_rate``
        (see :meth:`_normalise_scores`)."""
        k = self.index_metadata["length"]
        w = self.index_metadata["window_size"]
        n_kgrams = max(min_fragment_length / _AVG_CHARS_PER_WORD - k + 1, 1.0)
        corrected_ngrams = (1- edit_rate) ** k * n_kgrams
        return corrected_ngrams * 2 / (w + 1)

    def _fingerprint_weights(self, hashes: np.ndarray) -> np.ndarray:
        """Scoring weight of each fingerprint in ``hashes``: 1.0 for every
        fingerprint, or its :class:`RarityWeighting` weight when ``rarity_weighted``."""
        if self.weighting is None:
            return np.ones(len(hashes), dtype=np.float64)
        return self.weighting(hashes)

    def _normalise_scores(
        self, raw_scores: dict[str, float], min_fragment_length: float
    ) -> dict[str, float]:
        r"""Convert raw fingerprint scores to containment probabilities.

        **Model.**  Let :math:`n` be the observed score (number of shared fingerprints,
        or sum of their weights) and :math:`L_0` = ``min_fragment_length`` in
        characters.  A passage of :math:`L_0` characters holds about
        :math:`L_0 / \bar{c} - k + 1` k-grams, with :math:`\bar{c} \approx 5.5`
        characters per word, and winnowing with window :math:`w` keeps on average a
        fraction :math:`2 / (w + 1)` of them.  A k-gram survives independent word
        edits at rate :math:`\epsilon` with probability :math:`(1 - \epsilon)^k`, so
        the expected number of shared fingerprints for a passage of :math:`L_0`
        characters with a tolerated edit rate :math:`\epsilon` is

        .. math::
            N_0 = (1 - \epsilon)^k \left(\frac{L_0}{\bar{c}} - k + 1\right) \frac{2}{w + 1}

        (:math:`N_0 \approx 33` for :math:`L_0 = 1000`, :math:`k = 4`, :math:`w = 6`,
        :math:`\epsilon = 0.1`; see :meth:`_expected_fingerprints`).

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
        :math:`\pm 2\sqrt{\text{dispersion} \cdot N_0}` (:math:`\pm 8` fingerprints
        in the example above).

        **Calibration.**  Passages edited more heavily than :math:`\epsilon` lose
        further fingerprints and must be proportionally longer than :math:`L_0` to
        reach the same probability.

        **Rarity weighting.**  With ``rarity_weighted=True`` the score is a sum of
        weights :math:`W` normalised to mean 1 over fingerprint occurrences, so its
        mean is still :math:`N_0`, but a random sum of unequal terms has variance
        :math:`\mathrm{Var}(n)\,E[W]^2 + E[n]\,\mathrm{Var}(W)`, i.e. the effective
        dispersion becomes :math:`\text{dispersion} + E[W^2] - 1`; both moments are
        estimated by :class:`RarityWeighting` when the detector is created.

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


class RarityWeighting:
    r"""IDF-like fingerprint weights, on the scale of a plain fingerprint count.

    A fingerprint with ``cf`` postings in an index of ``N`` documents gets the raw
    weight :math:`\log(1 + N / cf)` (:meth:`raw`).  Raw weights are divided by their
    mean over fingerprint *occurrences* — a fingerprint met in running text is drawn
    with probability proportional to ``cf`` — so that a verbatim passage of
    :math:`n` fingerprints still scores :math:`n` on average and the count-based
    probability model of :meth:`_BaseDetector._normalise_scores` applies unchanged.
    Summing unequal weights :math:`W` does widen the score distribution: its
    variance-to-mean ratio grows by :math:`E[W^2] / E[W]^2 - 1` (``extra_dispersion``).
    Both moments are estimated when the object is created, from a sample of the
    index's posting counts.

    Attributes
    ----------
    mean : float
        Mean raw weight over fingerprint occurrences (the normalisation constant).
    extra_dispersion : float
        :math:`E[W^2] / E[W]^2 - 1` of the scaled weights.
    max : float
        Largest scaled weight, that of a fingerprint with a single posting.

    """

    def __init__(self, index: indexing.DiskBasedIndex):
        self.index = index
        counts = self._sample_posting_counts(index.lengths).astype(np.float64)
        if counts.sum() == 0:  # index without fingerprints
            self.mean, self.extra_dispersion = 1.0, 0.0
        else:
            raw = self.raw(counts)
            self.mean = float(np.average(raw, weights=counts))
            self.extra_dispersion = float(np.average(raw**2, weights=counts)) / self.mean**2 - 1.0
        self.max = float(self.raw(np.float64(1.0))) / self.mean

    def raw(self, counts: np.ndarray) -> np.ndarray:
        """Unscaled weight ``log(1 + N / cf)`` of fingerprints with (positive) posting counts ``cf``."""
        return np.log1p(self.index.num_documents / counts)

    def __call__(self, hashes: np.ndarray) -> np.ndarray:
        """Scaled weight of each fingerprint in ``hashes`` (0.0 when absent from the index)."""
        counts = self.posting_counts(hashes)
        weights = np.zeros(len(hashes), dtype=np.float64)
        found = counts > 0
        weights[found] = self.raw(counts[found].astype(np.float64)) / self.mean
        return weights

    def posting_counts(self, hashes: np.ndarray) -> np.ndarray:
        """Number of postings of each fingerprint in the index (0 when absent).

        Only the fingerprint and length arrays are consulted, so this is much
        cheaper than reading the postings themselves.
        """
        index = self.index
        queries = np.asarray(hashes, dtype=np.uint64)
        counts = np.zeros(len(queries), dtype=np.int64)
        if not len(index.fingerprints):
            return counts
        idx = index._lookup(queries)
        idx_clipped = np.clip(idx, 0, len(index.fingerprints) - 1)
        found = (idx < len(index.fingerprints)) & (index.fingerprints[idx_clipped] == queries)
        counts[found] = index.lengths[idx_clipped[found]]
        return counts

    @staticmethod
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
    :class:`RarityWeighting`).

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
        if candidates.is_empty:
            return {}
        weights = self._fingerprint_weights(candidates.hashes)
        return {
            doc_id: float(weights[candidates.shared_fingerprints(block)].sum())
            for doc_id, block in candidates.hits
        }


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
        if candidates.is_empty:
            return {}
        weights = self._fingerprint_weights(candidates.hashes)
        scores = {}
        for doc_id, block in candidates.hits:
            fingerprint, pos_query, pos_doc = candidates.join(block)
            labels = self.cluster(pos_query, pos_doc - pos_query)
            scores[doc_id] = largest_cluster_weight(labels, fingerprint, weights)
        return scores

    def cluster(self, pos_query: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Cluster shared fingerprints given their query positions and offsets,
        with the detector's method and thresholds (see :func:`cluster_rectangle`
        and :func:`cluster_linkage`).  Returns int32 labels, −1 for noise."""
        if self.method == "rectangle":
            return cluster_rectangle(
                pos_query, offset,
                self.position_threshold, self.offset_threshold, self.min_cluster_size,
            )
        if self.method in _LINKAGE_METRICS:
            return cluster_linkage(
                pos_query, offset, self.method, self.distance_threshold, self.min_cluster_size
            )
        raise ValueError(
            f"Unknown clustering method '{self.method}'; must be one of "
            "'rectangle', 'euclidean', 'cityblock', or 'chebyshev'."
        )


def cluster_rectangle(
    pos_query: np.ndarray,
    offset: np.ndarray,
    position_threshold: int,
    offset_threshold: int,
    min_cluster_size: int,
) -> np.ndarray:
    """Cluster shared fingerprints using rectangle-neighbourhood connected components.

    Two points are *connected* iff both ``|Δpos_query| ≤ position_threshold`` and
    ``|Δoffset| ≤ offset_threshold``.  Connected components form initial clusters;
    those with fewer than ``min_cluster_size`` members are relabelled −1 (noise).

    The neighbour pairs come from a k-d tree query within the larger threshold
    (Chebyshev distance), refined per axis, so the cost grows with the number of
    neighbouring pairs rather than quadratically with the number of points.

    Parameters
    ----------
    pos_query, offset : np.ndarray
        Aligned coordinates of the shared fingerprints: position in the query and
        ``position_in_doc - position_in_query``.

    Returns
    -------
    np.ndarray
        int32 cluster label of each point, −1 for noise.

    """
    points = np.column_stack([pos_query, offset]).astype(np.float64)
    radius = max(position_threshold, offset_threshold) + 0.5
    pairs = cKDTree(points).query_pairs(radius, p=np.inf, output_type="ndarray")
    diff = np.abs(points[pairs[:, 0]] - points[pairs[:, 1]])
    pairs = pairs[(diff[:, 0] <= position_threshold) & (diff[:, 1] <= offset_threshold)]

    n = len(points)
    graph = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, raw_labels = connected_components(graph, directed=False)
    return _relabel_small_clusters(raw_labels, min_cluster_size)


def cluster_linkage(
    pos_query: np.ndarray,
    offset: np.ndarray,
    metric: str,
    distance_threshold: int,
    min_cluster_size: int,
) -> np.ndarray:
    """Cluster shared fingerprints via single-linkage hierarchical clustering.

    Clustering is performed on the 2-D feature space ``(pos_query, offset)`` using
    ``scipy.cluster.hierarchy.linkage`` with ``method="single"`` and the given
    distance ``metric`` (``"euclidean"``, ``"cityblock"`` or ``"chebyshev"``);
    clusters are cut at ``distance_threshold`` and those with fewer than
    ``min_cluster_size`` members relabelled −1 (noise).

    .. note::
       Single linkage with the Chebyshev metric and ``distance_threshold`` *t*
       is mathematically equivalent to :func:`cluster_rectangle` with
       ``position_threshold = offset_threshold = t``.

    Returns
    -------
    np.ndarray
        int32 cluster label of each point, −1 for noise.

    """
    n = len(pos_query)
    if n == 0:
        raw_labels = np.empty(0, dtype=np.int32)
    elif n == 1:
        raw_labels = np.array([1], dtype=np.int32)
    else:
        X = np.column_stack([pos_query, offset]).astype(np.float64)
        Z = linkage(X, method="single", metric=metric, optimal_ordering=False)
        raw_labels = fcluster(Z, t=distance_threshold, criterion="distance")
    return _relabel_small_clusters(raw_labels, min_cluster_size)


def largest_cluster_weight(
    labels: np.ndarray, fingerprint: np.ndarray, weights: np.ndarray
) -> float:
    """Largest per-cluster sum of fingerprint weights, each distinct fingerprint
    counted once per cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster label of each shared-fingerprint pair (−1 = noise, ignored).
    fingerprint : np.ndarray
        Id of the pair's fingerprint, indexing ``weights`` (see :meth:`Candidates.join`).
    weights : np.ndarray
        Weight of each fingerprint.

    Returns
    -------
    float
        0.0 when every point is noise.

    """
    keep = labels >= 0
    if not keep.any():
        return 0.0
    labels, fingerprint = labels[keep].astype(np.int64), fingerprint[keep].astype(np.int64)
    # One row per distinct (cluster, fingerprint) pair.
    _, first = np.unique(labels * len(weights) + fingerprint, return_index=True)
    sums = np.bincount(labels[first], weights=weights[fingerprint[first]])
    return float(sums.max())


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
    _, inverse, counts = np.unique(raw_labels, return_inverse=True, return_counts=True)
    valid = counts >= min_cluster_size
    new_ids = np.where(valid, np.cumsum(valid) - 1, -1).astype(np.int32)
    return new_ids[inverse]


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
