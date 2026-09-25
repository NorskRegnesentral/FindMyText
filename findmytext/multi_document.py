"""Fingerprint-chain analysis across an explicit set of documents."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage

from .winnower import Winnower

CoordinateRepresentation = Literal["raw", "anchor_offsets", "centered_offsets"]
PairwiseMode = Literal["all", "novel", "none"]


@dataclass(frozen=True)
class SharedPassage:
    """One positional fingerprint chain shared by a document subset."""

    passage_id: str
    document_ids: tuple[str, ...]
    spans: dict[str, tuple[int, int]]
    fingerprint_count: int
    hashes: tuple[int, ...]
    relationship: str
    classification: str
    subsumed_by: str | None = None


@dataclass(frozen=True)
class MultiDocumentAnalysis:
    """Passage summaries and raw points produced by an analysis run."""

    document_ids: tuple[str, ...]
    passages: tuple[SharedPassage, ...]
    points: pl.DataFrame
    coordinate_representation: CoordinateRepresentation
    skipped_hashes: tuple[int, ...]


def transform_positions(
    positions: np.ndarray,
    representation: CoordinateRepresentation,
    *,
    position_threshold: float,
    offset_threshold: float,
) -> np.ndarray:
    """Transform and threshold-scale raw ``(p1, ..., pN)`` positions."""
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("positions must be a 2-D array for at least two documents")
    if position_threshold <= 0 or offset_threshold <= 0:
        raise ValueError("position_threshold and offset_threshold must be positive")

    if representation == "raw":
        return values / position_threshold
    if representation == "anchor_offsets":
        anchor = values[:, :1]
        return np.concatenate(
            [anchor / position_threshold, (values[:, 1:] - anchor) / offset_threshold],
            axis=1,
        )
    if representation == "centered_offsets":
        means = values.mean(axis=1, keepdims=True)
        return np.concatenate(
            [means / position_threshold, (values - means) / offset_threshold], axis=1
        )
    raise ValueError(
        "coordinate_representation must be 'raw', 'anchor_offsets', or "
        "'centered_offsets'"
    )


class MultiDocumentFingerprintChainAnalyzer:
    """Find all-document and pairwise chains in two or more documents.

    Parameters are measured in fingerprint token positions. Coordinates are
    scaled by their corresponding threshold before single-linkage clustering,
    so ``distance_threshold=1`` gives the direct threshold interpretation.
    """

    def __init__(
        self,
        *,
        winnower: Winnower | None = None,
        coordinate_representation: CoordinateRepresentation = "centered_offsets",
        metric: Literal["euclidean", "cityblock", "chebyshev"] = "chebyshev",
        position_threshold: float = 30,
        offset_threshold: float = 30,
        distance_threshold: float = 1.0,
        min_cluster_size: int = 5,
        max_position_combinations_per_hash: int = 1_000,
        subsumption_threshold: float = 0.8,
    ):
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be positive")
        if max_position_combinations_per_hash <= 0:
            raise ValueError("max_position_combinations_per_hash must be positive")
        if not 0 <= subsumption_threshold <= 1:
            raise ValueError("subsumption_threshold must be between 0 and 1")
        self.winnower = winnower or Winnower(length=4, window_size=6)
        self.coordinate_representation: CoordinateRepresentation = (
            coordinate_representation
        )
        self.metric: Literal["euclidean", "cityblock", "chebyshev"] = metric
        self.position_threshold = position_threshold
        self.offset_threshold = offset_threshold
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.max_position_combinations_per_hash = max_position_combinations_per_hash
        self.subsumption_threshold = subsumption_threshold

    def analyze(
        self,
        documents: Mapping[str, str],
        *,
        document_ids: Sequence[str] | None = None,
        pairwise_mode: PairwiseMode = "novel",
    ) -> MultiDocumentAnalysis:
        """Analyze selected documents, then classify their pairwise passages."""
        selected = tuple(document_ids or documents.keys())
        if len(selected) < 2:
            raise ValueError("At least two documents are required")
        if len(set(selected)) != len(selected):
            raise ValueError("document_ids must be unique")
        missing = [doc_id for doc_id in selected if doc_id not in documents]
        if missing:
            raise KeyError(f"Missing document text for: {missing}")
        if pairwise_mode not in {"all", "novel", "none"}:
            raise ValueError("pairwise_mode must be 'all', 'novel', or 'none'")

        positions = self._fingerprint_positions(documents, selected)
        passages: list[SharedPassage] = []
        point_frames: list[pl.DataFrame] = []
        skipped: set[int] = set()

        all_passages, all_points, all_skipped = self._analyze_subset(
            selected, positions, relationship="all_documents", id_prefix="all"
        )
        passages.extend(all_passages)
        if not all_points.is_empty():
            point_frames.append(all_points)
        skipped.update(all_skipped)

        if pairwise_mode != "none" and len(selected) > 2:
            for pair_index, pair in enumerate(itertools.combinations(selected, 2)):
                pair_passages, pair_points, pair_skipped = self._analyze_subset(
                    pair,
                    positions,
                    relationship="pairwise",
                    id_prefix=f"pair-{pair_index}",
                )
                skipped.update(pair_skipped)
                classified = [
                    self._classify_pairwise(passage, all_passages)
                    for passage in pair_passages
                ]
                keep_ids = {
                    passage.passage_id
                    for passage in classified
                    if pairwise_mode == "all" or passage.classification != "subsumed"
                }
                passages.extend(
                    passage for passage in classified if passage.passage_id in keep_ids
                )
                if not pair_points.is_empty() and keep_ids:
                    point_frames.append(
                        pair_points.filter(pl.col("passage_id").is_in(keep_ids))
                    )

        points = (
            pl.concat(point_frames, how="diagonal_relaxed")
            if point_frames
            else self._empty_points(selected)
        )
        return MultiDocumentAnalysis(
            document_ids=selected,
            passages=tuple(passages),
            points=points,
            coordinate_representation=self.coordinate_representation,
            skipped_hashes=tuple(sorted(skipped)),
        )

    def _fingerprint_positions(
        self, documents: Mapping[str, str], document_ids: Sequence[str]
    ) -> dict[str, dict[int, list[int]]]:
        result: dict[str, dict[int, list[int]]] = {}
        for doc_id in document_ids:
            hashes, positions = self.winnower.get_winnowed_fingerprints(
                documents[doc_id]
            )
            by_hash: dict[int, list[int]] = {}
            for fingerprint, position in zip(hashes, positions):
                by_hash.setdefault(int(fingerprint), []).append(int(position))
            result[doc_id] = by_hash
        return result

    def _analyze_subset(
        self,
        document_ids: Sequence[str],
        positions: Mapping[str, Mapping[int, list[int]]],
        *,
        relationship: str,
        id_prefix: str,
    ) -> tuple[list[SharedPassage], pl.DataFrame, set[int]]:
        common_hashes = set(positions[document_ids[0]])
        for doc_id in document_ids[1:]:
            common_hashes.intersection_update(positions[doc_id])

        rows: list[tuple[int, ...]] = []
        skipped: set[int] = set()
        for fingerprint in sorted(common_hashes):
            position_lists = [positions[doc_id][fingerprint] for doc_id in document_ids]
            combinations_count = int(
                np.prod([len(values) for values in position_lists])
            )
            if combinations_count > self.max_position_combinations_per_hash:
                skipped.add(fingerprint)
                continue
            rows.extend(
                (fingerprint, *position_tuple)
                for position_tuple in itertools.product(*position_lists)
            )
        if not rows:
            return [], self._empty_points(document_ids), skipped

        raw_positions = np.asarray([row[1:] for row in rows], dtype=np.float64)
        transformed = transform_positions(
            raw_positions,
            self.coordinate_representation,
            position_threshold=self.position_threshold,
            offset_threshold=self.offset_threshold,
        )
        if len(rows) == 1:
            labels = np.array([1], dtype=np.int32)
        else:
            hierarchy = linkage(
                transformed,
                method="single",
                metric=self.metric,
                optimal_ordering=False,
            )
            labels = fcluster(
                hierarchy, t=self.distance_threshold, criterion="distance"
            )

        hashes = np.asarray([row[0] for row in rows], dtype=np.int64)
        valid_labels = [
            int(label)
            for label in np.unique(labels)
            if len(np.unique(hashes[labels == label])) >= self.min_cluster_size
        ]
        passages: list[SharedPassage] = []
        point_rows: list[dict[str, object]] = []
        for cluster_index, label in enumerate(valid_labels):
            mask = labels == label
            cluster_positions = raw_positions[mask].astype(int)
            cluster_hashes = tuple(sorted(set(map(int, hashes[mask]))))
            passage_id = f"{id_prefix}-{cluster_index}"
            spans = {
                doc_id: (
                    int(cluster_positions[:, dimension].min()),
                    int(cluster_positions[:, dimension].max()) + self.winnower.length,
                )
                for dimension, doc_id in enumerate(document_ids)
            }
            passages.append(
                SharedPassage(
                    passage_id=passage_id,
                    document_ids=tuple(document_ids),
                    spans=spans,
                    fingerprint_count=len(cluster_hashes),
                    hashes=cluster_hashes,
                    relationship=relationship,
                    classification="shared"
                    if relationship == "all_documents"
                    else "novel",
                )
            )
            for row_index in np.flatnonzero(mask):
                point_row: dict[str, object] = {
                    "hash": int(hashes[row_index]),
                    "passage_id": passage_id,
                    "relationship": relationship,
                    "document_ids": tuple(document_ids),
                }
                point_row.update(
                    {
                        f"position_{doc_id}": int(raw_positions[row_index, dimension])
                        for dimension, doc_id in enumerate(document_ids)
                    }
                )
                point_rows.append(point_row)
        points = (
            pl.DataFrame(point_rows) if point_rows else self._empty_points(document_ids)
        )
        return passages, points, skipped

    def _classify_pairwise(
        self, passage: SharedPassage, all_passages: Sequence[SharedPassage]
    ) -> SharedPassage:
        best_classification = "novel"
        best_parent: str | None = None
        for parent in all_passages:
            overlap_fractions = []
            for doc_id in passage.document_ids:
                child_start, child_end = passage.spans[doc_id]
                parent_start, parent_end = parent.spans[doc_id]
                overlap = max(
                    0, min(child_end, parent_end) - max(child_start, parent_start)
                )
                overlap_fractions.append(overlap / max(1, child_end - child_start))
            if min(overlap_fractions) >= self.subsumption_threshold:
                best_classification = "subsumed"
                best_parent = parent.passage_id
                break
            if min(overlap_fractions) > 0:
                best_classification = "partial_overlap"
                best_parent = parent.passage_id
        return SharedPassage(
            passage_id=passage.passage_id,
            document_ids=passage.document_ids,
            spans=passage.spans,
            fingerprint_count=passage.fingerprint_count,
            hashes=passage.hashes,
            relationship=passage.relationship,
            classification=best_classification,
            subsumed_by=best_parent,
        )

    @staticmethod
    def _empty_points(document_ids: Sequence[str]) -> pl.DataFrame:
        columns: dict[str, pl.Series] = {
            "hash": pl.Series([], dtype=pl.Int64),
            "passage_id": pl.Series([], dtype=pl.Utf8),
            "relationship": pl.Series([], dtype=pl.Utf8),
            "document_ids": pl.Series([], dtype=pl.List(pl.Utf8)),
        }
        columns.update(
            {
                f"position_{doc_id}": pl.Series([], dtype=pl.Int64)
                for doc_id in document_ids
            }
        )
        return pl.DataFrame(columns)


def plot_three_document_fingerprints(
    analysis: MultiDocumentAnalysis,
) -> Any:
    """Plot raw positions for a three-document analysis with Plotly.

    Pairwise points are projected onto the missing document's zero plane and
    use a different marker symbol from all-document points. Zero is therefore a
    display convention, not missing-data storage in the analysis result.
    """
    if len(analysis.document_ids) != 3:
        raise ValueError("The interactive 3D plot requires exactly three documents")
    try:
        import plotly.express as px
    except ImportError as error:
        raise ImportError(
            "plot_three_document_fingerprints requires plotly; install findmytext[viz]"
        ) from error

    first, second, third = analysis.document_ids
    coordinate_columns = [
        f"position_{first}",
        f"position_{second}",
        f"position_{third}",
    ]
    points = analysis.points
    for column in coordinate_columns:
        if column not in points.columns:
            points = points.with_columns(pl.lit(0).cast(pl.Int64).alias(column))
        else:
            points = points.with_columns(pl.col(column).fill_null(0))
    if points.is_empty():
        raise ValueError("The analysis contains no clustered fingerprint points")

    frame = points.select(
        coordinate_columns + ["hash", "passage_id", "relationship", "document_ids"]
    ).to_pandas()

    def _format_membership(value: object) -> str:
        if isinstance(value, (list, tuple, np.ndarray)):
            return ", ".join(str(doc_id) for doc_id in value)
        return str(value)

    frame["membership"] = frame["document_ids"].map(_format_membership)
    figure = px.scatter_3d(
        frame,
        x=coordinate_columns[0],
        y=coordinate_columns[1],
        z=coordinate_columns[2],
        color="passage_id",
        symbol="relationship",
        hover_data=["hash", "membership"],
    )
    figure.update_traces(marker={"size": 4, "opacity": 0.75})
    figure.update_layout(
        scene={
            "xaxis_title": first,
            "yaxis_title": second,
            "zaxis_title": third,
        },
        legend_title_text="Passage / membership",
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
    )
    return figure
