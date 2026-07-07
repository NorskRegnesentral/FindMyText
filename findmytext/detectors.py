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

Both detectors expose a ``get_containment_score`` method returning a
``dict[str, float]`` mapping each candidate document id to a containment
probability in [0, 1].

"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm

from . import indexing

# Average number of characters per word (including trailing space) for typical
# English text.  This is the size of one k-gram sliding step and is used to
# convert a fragment length in characters to an expected fingerprint count.
_AVG_CHARS_PER_WORD: float = 5.5


class _BaseDetector(ABC):
    """Shared initialisation and helpers for all fingerprint-based detectors.

    Parameters
    ----------
    index_dir : str
        Path to the directory containing a disk-based winnowing index
        (``meta.json``, ``fingerprints.npy``, ``postings.dat``, ...).
    top_k : int
        Number of closest candidate documents to retrieve per query (default: 50).
    min_fingerprints : int
        Minimum number of shared fingerprints for a candidate to be considered a
        match (default: 5).
    rarity_weighted : bool
        If ``True``, score by rarity-weighted fingerprint sum rather than raw count
        (default: ``False``).

    """

    def __init__(
        self,
        index_dir: str,
        top_k: int = 50,
        min_fingerprints: int = 5,
        rarity_weighted: bool = False,
    ):
        self.index_dir = index_dir
        self.top_k = top_k
        self.min_fingerprints = min_fingerprints
        self.rarity_weighted = rarity_weighted
        self.index = indexing.DiskBasedIndex(index_dir)
        self.num_documents = len(self.index.to_external_doc_id)

        # Mean fingerprint weight across the whole index, used to normalise rarity
        # weights so that scores remain on the same scale as the raw fingerprint count.
        # lengths[i] approximates df(h) (exact when each fingerprint appears at most
        # once per document, which is typical for rare word k-grams).
        self._mean_fingerprint_weight = (
            float(1.0 - self.index.lengths.mean() / self.num_documents)
            if rarity_weighted
            else 1.0
        )

    @abstractmethod
    def get_containment_scores(
        self, text: str, min_fragment_length: float = 1000.0
    ) -> dict[str, float]:
        """Return a containment-probability score for each candidate document.

        Parameters
        ----------
        text : str
            The query document to search for.
        min_fragment_length : float
            Minimum length of a matching fragment in characters (default 1 000).
            Used to derive the expected fingerprint count :math:`N_0` for the
            Poisson model.

        Returns
        -------
        dict[str, float]
            Mapping from external document id to containment probability in
            :math:`[0, 1]`.

        """
        raise NotImplementedError

    def get_fingerprints(self, text: str) -> pl.DataFrame:
        """Return the winnowed fingerprints of ``text`` as a Polars DataFrame.

        The returned DataFrame has columns ``hash`` and ``position``.
        """
        fingerprints, fingerprints_pos = (
            self.index.runtime_winnower.get_winnowed_fingerprints(text)
        )
        return pl.DataFrame({"hash": fingerprints, "position": fingerprints_pos})

    def _get_candidates(self, fingerprints) -> pl.DataFrame:
        """Return the candidate-match table for the given fingerprints.

        Accepts any array-like of hash values (numpy array, Polars Series, …).
        The returned DataFrame has columns ``doc_match_id``,
        ``doc_match_closeness_rank``, ``hash``, and ``position`` (the position
        of each shared hash inside the candidate document).
        """
        candidates = self.index.get_closest_matches_with_positions(
            query=fingerprints,
            top_k=self.top_k,
            min_fingerprints=self.min_fingerprints,
            verbose=False,
        )
        return convert_closest_matches_with_positions_to_df(candidates)

    def _get_fingerprint_weights(self, fingerprints) -> dict[int, float]:
        """Per-fingerprint weights for scoring: uniform (1.0) or rarity-based.

        Returns a ``{hash: weight}`` mapping.  When ``rarity_weighted=False``
        every fingerprint gets weight 1.0.  When ``rarity_weighted=True`` the
        weight is ``1 - df(h) / num_documents``, normalised so the index-wide
        mean equals 1.0.
        """
        if not self.rarity_weighted:
            hashes = np.unique(np.asarray(fingerprints, dtype=np.int64))
            return dict.fromkeys(hashes.tolist(), 1.0)
        postings = self.index._get_postings(fingerprints, only_doc_ids=True)
        return {
            int(h): float(1 - len(np.unique(doc_ids)) / self.num_documents)
            / self._mean_fingerprint_weight
            for h, doc_ids in postings.items()
        }

    def _normalise_scores(
        self, raw_scores: dict[str, float], min_fragment_length: float
    ) -> dict[str, float]:
        r"""Convert raw fingerprint scores to containment probabilities.

        **Model.**  Let :math:`N` be the observed score (sum of fingerprint weights)
        and :math:`L_0` = ``min_fragment_length`` in characters.  The expected
        fingerprint count for a passage of exactly :math:`L_0` characters is

        .. math::
            N_0 = \frac{L_0}{w \cdot \bar{c}}

        where :math:`w` is the winnowing window size and :math:`\bar{c} \approx 5.5`
        is the average characters per word (one k-gram sliding step — *not* the length
        of the k-gram string itself, which would be :math:`k \times 5.5 \approx 22`
        characters for k = 4).  With :math:`w = 6` and :math:`L_0 = 1000`:
        :math:`N_0 \approx 30`.

        Under a Poisson noise model, the posterior probability that the underlying
        passage length exceeds :math:`L_0` is approximated via the normal approximation
        to the Poisson/Gamma posterior:

        .. math::
            P(\text{match} \mid N = n) \approx
            \Phi\!\left(\frac{n - N_0}{\sqrt{N_0}}\right)

        This gives :math:`P = 0.5` when :math:`n = N_0`, with a sharp transition over
        :math:`\pm\sqrt{N_0} \approx \pm 5` fingerprints.

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
        N0 = min_fragment_length / (
            self.index.indexing_winnower.window_size * _AVG_CHARS_PER_WORD
        )
        probs = norm.cdf((np.array(list(raw_scores.values())) - N0) / N0**0.5)
        return dict(zip(raw_scores.keys(), probs.tolist()))


class NbSharedFingerprintsDetector(_BaseDetector):
    """Detect text containment by counting shared unique fingerprints.

    Scores each candidate document by the sum of fingerprint weights over its
    unique shared fingerprints — either the raw count (uniform weights, default)
    or a rarity-weighted sum.

    Parameters
    ----------
    index_dir : str
        Path to the disk-based winnowing index directory.
    top_k : int
        Number of closest candidate documents to retrieve per query (default: 50).
    min_fingerprints : int
        Minimum number of shared fingerprints for a candidate to be considered a
        match (default: 5).
    rarity_weighted : bool
        If ``True``, score by rarity-weighted fingerprint sum rather than raw count
        (default: ``False``).

    """

    def get_containment_scores(
        self, text: str, min_fragment_length: float = 1000.0
    ) -> dict[str, float]:
        """Return a containment-probability score for each candidate document.

        Scores each candidate by the total sum of fingerprint weights over
        unique shared fingerprints, then converts that sum to a probability
        using the Poisson model in :meth:`_normalise_scores`.

        Parameters
        ----------
        text : str
            The query document to search for.
        min_fragment_length : float
            Minimum length of a matching fragment in characters (default 1 000).

        Returns
        -------
        dict[str, float]
            Mapping from external document id to containment probability in
            :math:`[0, 1]`.

        """

        if len(text) < min_fragment_length:
            warnings.warn(
                f"Query ({len(text)} chars) is shorter than min_fragment_length "
                f"({min_fragment_length} chars)",
                UserWarning,
                stacklevel=2,
            )

        df_query = self.get_fingerprints(text)
        df_candidates = self._get_candidates(df_query["hash"])
        if df_candidates.height == 0:
            return {}

        # Compute the weight of each fingerprint in the query (uniform or rarity-weighted).
        fingerprint_weights = self._get_fingerprint_weights(df_query["hash"])
        weights_df = pl.DataFrame(
            list(fingerprint_weights.items()),
            schema={"hash": pl.Int64, "weight": pl.Float64},
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
        raw_scores = dict(zip(df_scores["doc_match_id"], df_scores["raw_score"]))

        # Convert the raw fingerprint scores to containment probabilities using the Poisson model.
        return self._normalise_scores(raw_scores, min_fragment_length)


class FingerprintChainDetector(_BaseDetector):
    """Detect text containment by clustering shared fingerprints in position space.

    Scores each candidate document by the sum of fingerprint weights inside the
    *largest* spatial cluster of shared fingerprints.
    Clusters are found in the 2-D ``(position_in_query, position_offset)`` space using
    either the rectangle method (axis-aligned thresholds) or single-linkage hierarchical
    clustering (Euclidean, Manhattan, or Chebyshev distance).

    Parameters
    ----------
    index_dir : str
        Path to the disk-based winnowing index directory.
    top_k : int
        Number of closest candidate documents to retrieve per query (default: 50).
    min_fingerprints : int
        Minimum number of shared fingerprints for a candidate to be considered a
        match (default: 5).
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

    """

    def __init__(
        self,
        index_dir: str,
        top_k: int = 50,
        min_fingerprints: int = 5,
        rarity_weighted: bool = False,
        method: str = "rectangle",
        position_threshold: int = 30,
        offset_threshold: int = 30,
        distance_threshold: int = 30,
        min_cluster_size: int = 5,
    ):
        super().__init__(index_dir, top_k, min_fingerprints, rarity_weighted)
        self.method = method
        self.position_threshold = position_threshold
        self.offset_threshold = offset_threshold
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    def get_containment_scores(
        self, text: str, min_fragment_length: float = 1000.0
    ) -> dict[str, float]:
        """Return a containment-probability score for each candidate document.

        Clusters shared fingerprints in the 2-D ``(position_query, offset)``
        space; the weight sum of the largest surviving cluster is converted to
        a probability using the same Poisson model as
        :class:`NbSharedFingerprintsDetector`.

        Parameters
        ----------
        text : str
            The query document to search for.
        min_fragment_length : float
            Minimum matching-passage length in characters (default 1 000).
            Defines the Poisson threshold :math:`N_0`.

        Returns
        -------
        dict[str, float]
            Mapping from external document id to containment probability in
            :math:`[0, 1]`.

        """

        if len(text) < min_fragment_length:
            warnings.warn(
                f"Query ({len(text)} chars) is shorter than min_fragment_length "
                f"({min_fragment_length} chars)",
                UserWarning,
                stacklevel=2,
            )

        df_query = self.get_fingerprints(text)
        df_candidates = self._get_candidates(df_query["hash"])
        if df_candidates.height == 0:
            return {}

        fingerprint_weights = self._get_fingerprint_weights(df_query["hash"])

        raw_scores = {}
        for doc_id in df_candidates["doc_match_id"].unique(maintain_order=True):
            raw_scores[doc_id] = self._score_doc(
                df_query, df_candidates, fingerprint_weights, doc_id
            )

        return self._normalise_scores(raw_scores, min_fragment_length)

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
        ``doc_match_closeness_rank``, ``hash`` (Int64), ``position`` (Int32).
        Returns an empty DataFrame with the correct schema when the input is
        empty.

    """
    if not closest_with_positions:
        return pl.DataFrame(
            {
                "doc_match_id": pl.Series([], dtype=pl.Utf8),
                "doc_match_closeness_rank": pl.Series([], dtype=pl.Int32),
                "hash": pl.Series([], dtype=pl.Int64),
                "position": pl.Series([], dtype=pl.Int32),
            }
        )

    frames = []
    for rank, (doc_id, hash_positions) in enumerate(
        closest_with_positions.items(), start=1
    ):
        frames.append(
            pl.DataFrame(
                {
                    "hash": list(hash_positions.keys()),
                    "position": list(hash_positions.values()),
                }
            )
            .explode("position")
            .with_columns(
                pl.lit(doc_id).alias("doc_match_id"),
                pl.lit(rank, dtype=pl.Int32).alias("doc_match_closeness_rank"),
                pl.col("hash").cast(pl.Int64),
                pl.col("position").cast(pl.Int32),
            )
            .select(["doc_match_id", "doc_match_closeness_rank", "hash", "position"])
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
        "--min-fingerprints",
        type=int,
        default=5,
        metavar="N",
        help="Minimum shared fingerprints for a candidate to be considered (default: 5).",
    )
    parser.add_argument(
        "--rarity-weighted",
        action="store_true",
        help="Weight fingerprints by rarity instead of counting uniformly.",
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
            min_fingerprints=args.min_fingerprints,
            rarity_weighted=args.rarity_weighted,
        )
    else:
        detector = FingerprintChainDetector(
            args.index_dir,
            top_k=args.top_k,
            min_fingerprints=args.min_fingerprints,
            rarity_weighted=args.rarity_weighted,
            method=args.method,
            position_threshold=args.position_threshold,
            offset_threshold=args.offset_threshold,
            distance_threshold=args.distance_threshold,
            min_cluster_size=args.min_cluster_size,
        )

    # ── Score ────────────────────────────────────────────────────────────── #
    scores = detector.get_containment_scores(
        text, min_fragment_length=args.min_fragment_length
    )

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
