"""
Production interface for text-containment detection.

This module exposes a single class, :class:`TextContainmentDetector`, that wraps a
disk-based winnowing index and provides two scoring strategies for deciding whether a
query document is (partially) contained in any indexed document:

- **Jaccard**: the number of unique winnowed fingerprints the
  query shares with a candidate document, optionally weighted by fingerprint rarity.
- **Clustering**: the number of unique shared fingerprints inside the *largest* spatial
  cluster of shared fingerprints, plus the largest rarity-weighted sum over clusters.
  Clusters are found in the 2-D ``(position_in_query, position_offset)`` space using either
  the rectangle method (axis-aligned position/offset thresholds) or single-linkage
  hierarchical clustering (with Euclidean or Manhattan distance thresholds).

The Jaccard score falls out of the clustering computation for free, so
:meth:`TextContainmentDetector.find_matches_clustering` always computes it as well.

Both ``find_matches_*`` methods return ``Dict[str, float]`` (external document id -> score).
The ``*_df`` wrapper methods return the full per-match Polars DataFrame, including the
count-based and rarity-based scores for both strategies.

TODO: Need to add a thresholding mechanism to decide whether a score is high enough to be considered a match.
"""

from __future__ import annotations

import indexing
import polars as pl

from typing import Dict
from detector_utils import (
    convert_closest_matches_with_positions_to_df,
    compute_fingerprint_rarity,
    get_df_hash_cluster,
)

# Default clustering configuration. Override individual keys via the ``clustering_params``
# argument of :meth:`TextContainmentDetector.find_matches_clustering`.
DEFAULT_CLUSTERING_PARAMS: Dict[str, object] = {
    "method": "rectangle",          # or "euclidean" / "cityblock" / "chebyshev"
    "position_threshold": 30,       # rectangle only
    "offset_threshold": 30,         # rectangle only
    "distance_threshold": 30,       # euclidean / cityblock / chebyshev only
    "min_cluster_size": 5,          # all methods
}


def _validate_score(score: str) -> None:
    """Validate the user-facing ``score`` argument of the dict-returning methods."""
    if score not in ("count", "rarity"):
        raise ValueError(f"score must be 'count' or 'rarity', got {score!r}.")


class TextContainmentDetector:
    """Detect whether a query document is contained in any indexed document.

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
    """

    def __init__(self, index_dir: str, top_k: int = 50, min_fingerprints: int = 5):
        self.index_dir = index_dir
        self.top_k = top_k
        self.min_fingerprints = min_fingerprints
        self.index = indexing.DiskBasedIndex(index_dir)
        self.num_documents = len(self.index.to_external_doc_id)

    # ------------------------------------------------------------------ #
    # Public API: dict-returning methods (wrappers)                      #
    # ------------------------------------------------------------------ #
    def find_matches_jaccard(self, text: str, score: str = "count") -> Dict[str, float]:
        """Score candidate documents by shared unique fingerprints (Jaccard).

        Parameters
        ----------
        text : str
            The query document.
        score : str
            Which score to return: ``"count"`` (number of unique shared fingerprints,
            default) or ``"rarity"`` (rarity-weighted sum of unique shared fingerprints).

        Returns
        -------
        Dict[str, float]
            Mapping from external document id to the selected score.
        """
        _validate_score(score)
        df = self.find_matches_jaccard_df(text)
        col = "num_shared_unique_hashes" if score == "count" else "rarity_sum_unique"
        return self._df_to_dict(df, col)

    def find_matches_clustering(
        self,
        text: str,
        clustering_params: Dict[str, object] | None = None,
        score: str = "count",
    ) -> Dict[str, float]:
        """Score candidate documents by the largest cluster of shared fingerprints.

        Parameters
        ----------
        text : str
            The query document.
        clustering_params : dict, optional
            Clustering configuration; missing keys fall back to
            :data:`DEFAULT_CLUSTERING_PARAMS`. Recognised keys:
            ``method``, ``position_threshold``, ``offset_threshold``,
            ``distance_threshold``, ``min_cluster_size``.
        score : str
            Which score to return: ``"count"`` (unique shared fingerprints in the
            cluster with the most unique shared fingerprints, default) or ``"rarity"``
            (largest unique rarity sum over all clusters).

        Returns
        -------
        Dict[str, float]
            Mapping from external document id to the selected score.
        """
        _validate_score(score)
        df = self.find_matches_clustering_df(text, clustering_params)
        col = (
            "num_shared_unique_hashes_in_largest_cluster"
            if score == "count"
            else "rarity_sum_in_top_rarity_cluster_unique"
        )
        return self._df_to_dict(df, col)

    # ------------------------------------------------------------------ #
    # Public API: DataFrame-returning                                    #
    # ------------------------------------------------------------------ #
    def find_matches_jaccard_df(self, text: str) -> pl.DataFrame:
        """Jaccard scores for all candidate matches as a Polars DataFrame.

        Columns: ``doc_match_id``, ``doc_match_closeness_rank``,
        ``num_shared_unique_hashes``, ``rarity_sum_unique``.
        """
        _, fingerprints_rarity, df_closest = self._prepare(text)
        return self._jaccard_from_closest(df_closest, fingerprints_rarity)

    def find_matches_clustering_df(
        self, text: str, clustering_params: Dict[str, object] | None = None
    ) -> pl.DataFrame:
        """Clustering (and Jaccard) scores for all candidate matches as a DataFrame.

        Columns: ``doc_match_id``, ``doc_match_closeness_rank``,
        ``num_shared_unique_hashes``, ``rarity_sum_unique`` (Jaccard side product),
        ``num_shared_unique_hashes_in_largest_cluster``,
        ``rarity_sum_in_top_rarity_cluster_unique`` (clustering).
        """
        params = {**DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}
        df_query, fingerprints_rarity, df_closest = self._prepare(text)

        # Jaccard scores come for free from the closest-match postings.
        df_jaccard = self._jaccard_from_closest(df_closest, fingerprints_rarity)

        if df_closest.height == 0:
            return df_jaccard.with_columns([
                pl.lit(0, dtype=pl.Int64).alias("num_shared_unique_hashes_in_largest_cluster"),
                pl.lit(0.0, dtype=pl.Float64).alias("rarity_sum_in_top_rarity_cluster_unique"),
            ])

        df_cluster = self._cluster_scores_per_match(
            df_query=df_query,
            df_closest=df_closest,
            fingerprints_rarity=fingerprints_rarity,
            params=params,
        )

        return (
            df_jaccard
            .join(df_cluster, on=["doc_match_id", "doc_match_closeness_rank"], how="left")
            .with_columns([
                pl.col("num_shared_unique_hashes_in_largest_cluster").fill_null(0),
                pl.col("rarity_sum_in_top_rarity_cluster_unique").fill_null(0.0),
            ])
            .sort("doc_match_closeness_rank")
        )

    def get_match_highlight_positions(
        self,
        text: str,
        doc_id: str,
        clustering_params: Dict[str, object] | None = None,
        method: str = "clustering",
    ) -> list[int]:
        """Return the query token positions to highlight for a matched document.

        Parameters
        ----------
        text : str
            The query document.
        doc_id : str
            External id of the matched document to highlight against.
        clustering_params : dict, optional
            Clustering configuration (only used when ``method="clustering"``);
            missing keys fall back to :data:`DEFAULT_CLUSTERING_PARAMS`.
        method : str
            ``"clustering"`` (default): the query positions belonging to the
            cluster with the most unique shared fingerprints — the same cluster
            that produces the count score of
            :meth:`find_matches_clustering`. ``"jaccard"``: every shared
            fingerprint's query position.

        Returns
        -------
        list[int]
            Sorted, de-duplicated query token positions. Empty if ``doc_id`` is
            not among the candidate matches or has no positions to highlight.
        """
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
        """Return query token positions to highlight, grouped into clusters.

        This is the grouped counterpart of :meth:`get_match_highlight_positions`.
        Each returned inner list is one group of query token positions that
        should be highlighted together (e.g. shown in the same colour).

        Parameters
        ----------
        text : str
            The query document.
        doc_id : str
            External id of the matched document to highlight against.
        clustering_params : dict, optional
            Clustering configuration (only used when ``method="clustering"``);
            missing keys fall back to :data:`DEFAULT_CLUSTERING_PARAMS`.
        method : str
            ``"clustering"`` (default) or ``"jaccard"``.
        which : str
            Only used when ``method="clustering"``. ``"largest"`` (default)
            returns a single group: the cluster with the most unique shared
            fingerprints (the same cluster that produces the count score of
            :meth:`find_matches_clustering`). ``"all"`` returns one group per
            non-noise cluster, ordered from most to fewest unique shared
            fingerprints.

        Returns
        -------
        list[list[int]]
            A list of groups; each group is a sorted, de-duplicated list of
            query token positions. Empty if ``doc_id`` is not among the
            candidate matches or has no positions to highlight. For
            ``method="jaccard"`` there is at most one group (every shared
            fingerprint's query position).
        """
        if method not in ("clustering", "jaccard"):
            raise ValueError(
                f"method must be 'clustering' or 'jaccard', got {method!r}."
            )
        if which not in ("largest", "all"):
            raise ValueError(f"which must be 'largest' or 'all', got {which!r}.")

        df_query, _, df_closest = self._prepare(text)
        if df_closest.height == 0:
            return []

        df_match = df_closest.filter(pl.col("doc_match_id") == doc_id)
        if df_match.height == 0:
            return []

        if method == "jaccard":
            shared_hashes = df_match.select("hash").unique()
            positions = (
                df_query
                .join(shared_hashes, on="hash", how="inner")
                .get_column("position")
                .to_list()
            )
            pos = sorted({int(p) for p in positions})
            return [pos] if pos else []

        # method == "clustering"
        params = {**DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}
        df_shared = (
            df_query
            .join(df_match.select(["hash", "position"]), on="hash", how="inner", suffix="_doc2")
            .select([
                pl.col("hash"),
                pl.col("position").alias("position_doc1"),
                pl.col("position_doc2"),
            ])
            .with_columns(
                (pl.col("position_doc2") - pl.col("position_doc1")).alias("position_offset")
            )
            .sort("position_doc1")
        )
        if df_shared.height == 0:
            return []

        df_clustered = get_df_hash_cluster(
            df_shared_hashes=df_shared,
            clustering_method=params["method"],
            position_threshold=params["position_threshold"],
            offset_threshold=params["offset_threshold"],
            distance_threshold=params["distance_threshold"],
            min_cluster_size=params["min_cluster_size"],
        )

        non_noise = df_clustered.filter(pl.col("cluster_id") != -1)
        if non_noise.height == 0:
            return []

        # Order clusters by unique shared fingerprints, most first. The largest
        # is the one that produces the count score of find_matches_clustering.
        sizes = (
            non_noise
            .group_by("cluster_id")
            .agg(pl.col("hash").n_unique().alias("n_unique"))
            .sort("n_unique", descending=True)
        )
        cluster_ids = sizes.get_column("cluster_id").to_list()
        if which == "largest":
            cluster_ids = cluster_ids[:1]

        groups: list[list[int]] = []
        for cid in cluster_ids:
            pos = sorted({
                int(p)
                for p in non_noise
                .filter(pl.col("cluster_id") == cid)
                .get_column("position_doc1")
                .to_list()
            })
            if pos:
                groups.append(pos)
        return groups

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _prepare(self, text: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Compute query fingerprints, their rarity, and the closest matches."""
        fingerprints, fingerprints_pos = self.index.winnower.get_winnowed_fingerprints(text)
        df_query = pl.DataFrame({"hash": fingerprints, "position": fingerprints_pos})

        fingerprints_rarity = compute_fingerprint_rarity(
            index=self.index,
            fingerprints=fingerprints,
            index_num_documents=self.num_documents,
            all_hashes=False,
        )

        closest = self.index.get_closest_matches_with_positions(
            query_text=text,
            top_k=self.top_k,
            min_fingerprints=self.min_fingerprints,
            verbose=False,
        )
        df_closest = convert_closest_matches_with_positions_to_df(closest)
        return df_query, fingerprints_rarity, df_closest

    @staticmethod
    def _jaccard_from_closest(
        df_closest: pl.DataFrame, fingerprints_rarity: pl.DataFrame
    ) -> pl.DataFrame:
        """Per-match count and rarity of unique shared fingerprints."""
        if df_closest.height == 0:
            return pl.DataFrame(schema={
                "doc_match_id": pl.Utf8,
                "doc_match_closeness_rank": pl.Int32,
                "num_shared_unique_hashes": pl.UInt32,
                "rarity_sum_unique": pl.Float64,
            })

        df_unique = (
            df_closest
            .select(["doc_match_id", "doc_match_closeness_rank", "hash"])
            .unique(maintain_order=True)
            .join(fingerprints_rarity, on="hash", how="left")
            .with_columns(pl.col("rarity").fill_null(0.0))
        )
        return (
            df_unique
            .group_by(["doc_match_id", "doc_match_closeness_rank"], maintain_order=True)
            .agg([
                pl.len().alias("num_shared_unique_hashes"),
                pl.col("rarity").sum().alias("rarity_sum_unique"),
            ])
            .sort("doc_match_closeness_rank")
        )

    @staticmethod
    def _cluster_scores_per_match(
        df_query: pl.DataFrame,
        df_closest: pl.DataFrame,
        fingerprints_rarity: pl.DataFrame,
        params: Dict[str, object],
    ) -> pl.DataFrame:
        """For each matched doc, cluster the shared fingerprints and return two scores:

        - ``num_shared_unique_hashes_in_largest_cluster``: unique shared fingerprints in
          the cluster that has the most unique shared fingerprints (count score).
        - ``rarity_sum_in_top_rarity_cluster_unique``: the largest unique rarity sum over
          all clusters (rarity score). This may come from a different cluster than the
          largest-by-count one.

        Noise (cluster id ``-1``) is excluded. Documents with no surviving cluster get 0.
        """
        df_closest_ids = (
            df_closest
            .select(["doc_match_id", "doc_match_closeness_rank"])
            .group_by("doc_match_id", maintain_order=True)
            .first()
        )

        rows = []
        for doc_id, rank in df_closest_ids.iter_rows():
            df_match = (
                df_closest
                .filter(pl.col("doc_match_id") == doc_id)
                .select(["hash", "position"])
                .join(fingerprints_rarity, on="hash", how="left")
                .with_columns(pl.col("rarity").fill_null(0.0))
            )

            # Shared fingerprints with their positions in both documents and the offset.
            df_shared = (
                df_query
                .join(df_match, on="hash", how="inner", suffix="_doc2")
                .select([
                    pl.col("hash"),
                    pl.col("rarity"),
                    pl.col("position").alias("position_doc1"),
                    pl.col("position_doc2"),
                ])
                .with_columns(
                    (pl.col("position_doc2") - pl.col("position_doc1")).alias("position_offset")
                )
                .sort("position_doc1")
            )

            n_largest_cluster, rarity_top_cluster = 0, 0.0
            if df_shared.height > 0:
                df_clustered = get_df_hash_cluster(
                    df_shared_hashes=df_shared,
                    clustering_method=params["method"],
                    position_threshold=params["position_threshold"],
                    offset_threshold=params["offset_threshold"],
                    distance_threshold=params["distance_threshold"],
                    min_cluster_size=params["min_cluster_size"],
                )

                df_clusters = (
                    df_clustered
                    .filter(pl.col("cluster_id") != -1)
                    .group_by("cluster_id", maintain_order=True)
                    .agg([
                        pl.col("hash").n_unique().alias("n_unique"),
                        pl.struct(["hash", "rarity"]).unique().struct.field("rarity").sum().alias("rarity_unique"),
                    ])
                )
                if df_clusters.height > 0:
                    # Count score: the cluster with the most unique shared fingerprints.
                    n_largest_cluster = int(df_clusters.select(pl.col("n_unique").max()).item())
                    # Rarity score: the cluster with the largest unique rarity sum
                    # (possibly a different cluster than the largest-by-count one).
                    rarity_top_cluster = float(df_clusters.select(pl.col("rarity_unique").max()).item())

            rows.append({
                "doc_match_id": doc_id,
                "doc_match_closeness_rank": rank,
                "num_shared_unique_hashes_in_largest_cluster": n_largest_cluster,
                "rarity_sum_in_top_rarity_cluster_unique": rarity_top_cluster,
            })

        return pl.DataFrame(
            rows,
            schema={
                "doc_match_id": pl.Utf8,
                "doc_match_closeness_rank": pl.Int32,
                "num_shared_unique_hashes_in_largest_cluster": pl.Int64,
                "rarity_sum_in_top_rarity_cluster_unique": pl.Float64,
            },
        )

    @staticmethod
    def _df_to_dict(df: pl.DataFrame, score_col: str) -> Dict[str, float]:
        """Convert a per-match DataFrame into a ``{doc_id: score}`` dictionary."""
        if df.height == 0:
            return {}
        return dict(zip(df["doc_match_id"].to_list(), df[score_col].to_list()))
