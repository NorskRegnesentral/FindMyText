"""Website-only detector adapter for the FindMyText demo app.

This module keeps the web app's count-based scoring and highlight helpers out of
the general detector modules. It intentionally exposes the small API expected by
``web/corpussearch/detection.py``.
"""

from __future__ import annotations

import re
from typing import Dict

import numpy as np
import polars as pl

from . import indexing
from .detectors import (
    FingerprintChainDetector,
    convert_closest_matches_with_positions_to_df,
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


def _compute_fingerprint_rarity(
    index: indexing.DiskBasedIndex,
    fingerprints: np.ndarray,
    index_num_documents: int,
) -> pl.DataFrame:
    postings = index._get_postings(fingerprints, only_doc_ids=True)
    hashes = np.fromiter(postings.keys(), dtype=np.int64, count=len(postings))
    rarities = np.fromiter(
        (1 - len(np.unique(doc_ids)) / index_num_documents for doc_ids in postings.values()),
        dtype=np.float64,
        count=len(postings),
    )
    return pl.DataFrame({"hash": hashes, "rarity": rarities}).sort(
        "rarity", descending=True
    )


class TextContainmentDetector:
    """Count-scoring detector API used by the website."""

    def __init__(self, index_dir: str, top_k: int = 50, min_fingerprints: int = 5):
        self.index_dir = index_dir
        self.top_k = top_k
        self.min_fingerprints = min_fingerprints
        self.index = indexing.DiskBasedIndex(index_dir)
        self.num_documents = len(self.index.to_external_doc_id)

    def find_matches_jaccard(self, text: str, score: str = "count") -> Dict[str, float]:
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
        _validate_score(score)
        df = self.find_matches_clustering_df(text, clustering_params)
        col = (
            "num_shared_unique_hashes_in_largest_cluster"
            if score == "count"
            else "rarity_sum_in_top_rarity_cluster_unique"
        )
        return self._df_to_dict(df, col)

    def find_matches_jaccard_df(self, text: str) -> pl.DataFrame:
        _, fingerprints_rarity, df_closest = self._prepare(text)
        return self._jaccard_from_closest(df_closest, fingerprints_rarity)

    def find_matches_clustering_df(
        self, text: str, clustering_params: Dict[str, object] | None = None
    ) -> pl.DataFrame:
        params = {**DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}
        df_query, fingerprints_rarity, df_closest = self._prepare(text)
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
        if method not in ("clustering", "jaccard"):
            raise ValueError(f"method must be 'clustering' or 'jaccard', got {method!r}.")
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

        params = {**DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}
        df_shared = self._shared_positions(df_query, df_match)
        if df_shared.height == 0:
            return []

        df_clustered = self._cluster_shared(df_shared, params)
        non_noise = df_clustered.filter(pl.col("cluster_id") != -1)
        if non_noise.height == 0:
            return []

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
        for cluster_id in cluster_ids:
            pos = sorted({
                int(p)
                for p in non_noise
                .filter(pl.col("cluster_id") == cluster_id)
                .get_column("position_doc1")
                .to_list()
            })
            if pos:
                groups.append(pos)
        return groups

    def _prepare(self, text: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        fingerprints, positions = self.index.runtime_winnower.get_winnowed_fingerprints(text)
        df_query = pl.DataFrame({"hash": fingerprints, "position": positions})
        fingerprints_rarity = _compute_fingerprint_rarity(
            index=self.index,
            fingerprints=fingerprints,
            index_num_documents=self.num_documents,
        )
        closest = self.index.get_closest_matches_with_positions(
            query=fingerprints,
            top_k=self.top_k,
            min_fingerprints=self.min_fingerprints,
            verbose=False,
        )
        return df_query, fingerprints_rarity, convert_closest_matches_with_positions_to_df(closest)

    @staticmethod
    def _jaccard_from_closest(
        df_closest: pl.DataFrame, fingerprints_rarity: pl.DataFrame
    ) -> pl.DataFrame:
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

    def _cluster_scores_per_match(
        self,
        df_query: pl.DataFrame,
        df_closest: pl.DataFrame,
        fingerprints_rarity: pl.DataFrame,
        params: Dict[str, object],
    ) -> pl.DataFrame:
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
            df_shared = self._shared_positions(df_query, df_match, include_rarity=True)

            n_largest_cluster, rarity_top_cluster = 0, 0.0
            if df_shared.height > 0:
                df_clustered = self._cluster_shared(df_shared, params)
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
                    n_largest_cluster = int(df_clusters.select(pl.col("n_unique").max()).item())
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
    def _shared_positions(
        df_query: pl.DataFrame, df_match: pl.DataFrame, include_rarity: bool = False
    ) -> pl.DataFrame:
        selected = [
            pl.col("hash"),
            pl.col("position").alias("position_doc1"),
            pl.col("position_doc2"),
        ]
        if include_rarity:
            selected.insert(1, pl.col("rarity"))
        return (
            df_query
            .join(df_match, on="hash", how="inner", suffix="_doc2")
            .select(selected)
            .with_columns(
                (pl.col("position_doc2") - pl.col("position_doc1")).alias("position_offset")
            )
            .sort("position_doc1")
        )

    @staticmethod
    def _cluster_shared(df_shared: pl.DataFrame, params: Dict[str, object]) -> pl.DataFrame:
        chain = FingerprintChainDetector.__new__(FingerprintChainDetector)
        chain.method = params["method"]
        chain.position_threshold = params["position_threshold"]
        chain.offset_threshold = params["offset_threshold"]
        chain.distance_threshold = params["distance_threshold"]
        chain.min_cluster_size = params["min_cluster_size"]
        if chain.method == "rectangle":
            return FingerprintChainDetector._get_df_hash_cluster_rectangle(chain, df_shared)
        if chain.method in {"euclidean", "cityblock", "chebyshev"}:
            return FingerprintChainDetector._get_df_hash_cluster_linkage(chain, df_shared)
        raise ValueError(
            f"Unknown clustering method {chain.method!r}; must be one of "
            "'rectangle', 'euclidean', 'cityblock', or 'chebyshev'."
        )

    @staticmethod
    def _df_to_dict(df: pl.DataFrame, score_col: str) -> Dict[str, float]:
        if df.height == 0:
            return {}
        return dict(zip(df["doc_match_id"].to_list(), df[score_col].to_list()))
