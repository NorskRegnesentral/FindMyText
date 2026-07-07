"""
Self-contained utility functions for :mod:`detector`.

- :func:`convert_closest_matches_with_positions_to_df` — turn the index's
  closest-match dictionary into a tidy Polars DataFrame.
- :func:`compute_fingerprint_rarity` — rarity score per fingerprint.
- :func:`get_df_hash_cluster` (plus the two backends
  :func:`get_df_hash_cluster_rectangle` and
  :func:`get_df_hash_cluster_linkage`) — cluster shared fingerprints in the
  2-D ``(position_doc1, position_offset)`` space.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import polars as pl

from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster

from findmytext import indexing


def convert_closest_matches_with_positions_to_df(closest_with_positions):
    """Convert the index's ``{doc_id: {hash: [positions]}}`` mapping into a tidy
    Polars DataFrame with one row per (matched document, hash, position).

    Returns a DataFrame with columns ``doc_match_id``, ``doc_match_closeness_rank``,
    ``hash`` and ``position``. When the input is empty, an empty DataFrame with the
    correct schema is returned.
    """
    dfs_closest_positions = []
    for i, (doc_id_key, closest_positions_dict) in enumerate(closest_with_positions.items()):
        df_closest_pos = pl.DataFrame({
            "hash": list(closest_positions_dict.keys()),
            "position": list(closest_positions_dict.values()),
        }).explode("position").with_columns([
            pl.lit(doc_id_key).alias("doc_match_id"),
            pl.lit(i+1).alias("doc_match_closeness_rank"),
            pl.col("hash").cast(pl.Int64).alias("hash"),
            pl.col("position").cast(pl.Int32).alias("position"),
        ]).select(["doc_match_id", "doc_match_closeness_rank", "hash", "position"]).sort(["doc_match_closeness_rank", "position"])
        dfs_closest_positions.append(df_closest_pos)
    return pl.concat(dfs_closest_positions) if dfs_closest_positions else pl.DataFrame({
        "doc_match_id": pl.Series([], dtype=pl.Utf8),
        "doc_match_closeness_rank": pl.Series([], dtype=pl.Int32),
        "hash": pl.Series([], dtype=pl.Int64),
        "position": pl.Series([], dtype=pl.Int32),
    })


################################################################
# Function to compute the rarity score for each fingerprint in #
# the index based on the number of documents that contain it   #
################################################################
def compute_fingerprint_rarity(
    index: "indexing.DiskBasedIndex",
    fingerprints: np.ndarray,
    index_num_documents: int | None = None,
    all_hashes: bool = False,
) -> pl.DataFrame:
    """
    Computes a rarity score for a given set of fingerprints based on the number of
    documents in the index that contain each fingerprint.

    Example:
    compute_fingerprint_rarity(
        index=index,
        fingerprints=np.array([123, 125, 456, 789]),
    )

    Parameters
    ----------
    index : indexing.DiskBasedIndex
        The disk-based index containing the documents and their winnowed fingerprints.
    fingerprints : np.ndarray
        An array of winnowed fingerprint integers for which to compute the rarity scores.
    index_num_documents : int, optional
        The total number of documents in the index. If not provided, it will be calculated
        as the length of index.to_external_doc_id.
    all_hashes : bool, optional
        If True, the returned DataFrame will contain all hashes in `fingerprints`, including
        those not found in the index (i.e. df(h) = 0). Such hashes receive a rarity of 1.0
        since they appear in no indexed document. Default is False (only hashes found in the
        index are returned).

    Returns
    -------
    pl.DataFrame
        A DataFrame with two columns: "hash" and "rarity". The "hash" column contains the fingerprint
        integers, and the "rarity" column contains the corresponding rarity scores, calculated as
        1 - df(h) / index_num_documents, where index_num_documents is the total number of documents in the index and df(h) is the number
        of documents containing fingerprint h.
        A larger rarity score indicates that the fingerprint is rarer across the documents in the index.
        Max value is 1 - 1/index_num_documents when the fingerprint is present in only one document, and
        min value is 0 when the fingerprint is present in all documents.
        When all_hashes=True, hashes absent from the index have df(h) = 0 and rarity = 1.0.
    """
    if index_num_documents is None:
        index_num_documents = len(index.to_external_doc_id)
    postings = index._get_postings(fingerprints, only_doc_ids=True)
    hashes = np.fromiter(postings.keys(), dtype=np.int64, count=len(postings))
    rarities = np.fromiter(
        (1 - len(np.unique(doc_ids)) / index_num_documents for doc_ids in postings.values()),
        dtype=np.float64, count=len(postings),
    )
    df = pl.DataFrame({"hash": hashes, "rarity": rarities})

    # If all_hashes is True, include hashes in `fingerprints` that were not found in the index (df(h) = 0) and assign them a rarity of 1.0
    if all_hashes:
        # Find hashes in `fingerprints` that were not returned by _get_postings (df(h) = 0)
        # and append them with rarity = 1.0.
        missing_hashes = np.setdiff1d(np.unique(fingerprints).astype(np.int64), hashes)
        if len(missing_hashes) > 0:
            df_missing = pl.DataFrame({
                "hash": missing_hashes,
                "rarity": np.ones(len(missing_hashes), dtype=np.float64),
            })
            df = pl.concat([df, df_missing])

    # Sort in decreasing order of rarity so that the rarest fingerprints are at the top of the DataFrame
    df = df.sort("rarity", descending=True)

    return df


###############################################################################################################################################
# Functions to create dataframes with the cluster assignments for the shared hashes based on different clustering methods and hyperparameters #
###############################################################################################################################################
def get_df_hash_cluster_rectangle(
        doc1_df: pl.DataFrame | None = None,
        doc2_df: pl.DataFrame | None = None,
        df_shared_hashes: pl.DataFrame | None = None,
        position_threshold: int | list[int] = 10,
        offset_threshold: int | list[int] = 10,
        min_cluster_size: int | list[int] = 5,
    ) -> pl.DataFrame:
    """
    This function takes in two documents as Polars DataFrames (two columns: "hash" and "position"), finds the shared hashes and their positions,
    and then clusters the shared hashes by grouping together those whose positions fall within a
    rectangle defined by ``position_threshold`` (max difference in position_doc1) and ``offset_threshold``
    (max difference in position_offset). The grouping is computed via connected components on the
    resulting adjacency graph. Or alternatively, it can take in a data frame of shared hashes with their
    positions and offsets and directly perform the clustering step.
    It then discards clusters that are smaller than a specified minimum cluster size and
    labels them as noise (-1). The resulting data frame includes the shared hashes, their positions in both documents,
    their offsets, and their assigned cluster IDs (with noise labeled as -1).

    Parameters:
    - doc1_df: The first document as a Polars DataFrame with columns "hash" and "position".
    - doc2_df: The second document as a Polars DataFrame with columns "hash" and "position".
    - df_shared_hashes: An optional Polars DataFrame containing the shared hashes and their positions and offsets, with columns "hash", "position_doc1", "position_doc2", and "position_offset". If this is provided, the function will skip the step of finding shared hashes and calculating their positions and offsets, and will directly use this data frame for clustering. If this is not provided, the function will compute it by performing an inner join on the "hash" column of doc1_df and doc2_df.
    - position_threshold: The maximum allowed difference in position_doc1 between neighbors to be considered part of the same cluster. Can be an int or a list of ints for hyperparameter tuning.
    - offset_threshold: The maximum allowed difference in position_offset between neighbors to be considered part of the same cluster. Can be an int or a list of ints for hyperparameter tuning.
    - min_cluster_size: The minimum number of shared hashes required for a cluster to be considered meaningful (clusters smaller than this will be labeled as -1 for noise). Can be an int or a list of ints for hyperparameter tuning.

    Returns:
    - A Polars DataFrame with columns for:
        - "hash": The shared hash value.
        - "position_doc1": The position of the hash in doc1.
        - "position_doc2": The position of the hash in doc2.
        - "position_offset": The offset between the positions in doc1 and doc2 (position_doc2 - position_doc1).
        - "cluster_id": The assigned cluster ID for each shared hash, with noise labeled as -1.
    """
    # Convert position_threshold, offset_threshold, and min_cluster_size to list of integers
    if isinstance(position_threshold, int):
        position_threshold = [position_threshold]
    if isinstance(offset_threshold, int):
        offset_threshold = [offset_threshold]
    if isinstance(min_cluster_size, int):
        min_cluster_size = [min_cluster_size]

    # If df_shared_hashes is not provided, find shared hashes and their
    # positions in the documents by performing an inner join on the "hash"
    # column of the two document data frames. Then calculate
    if df_shared_hashes is None:
        if doc1_df is None or doc2_df is None:
            raise ValueError("Either df_shared_hashes or both doc1_df and doc2_df must be provided.")
        # Find shared hashes and their positions in the documents
        df_shared_hashes = doc1_df.join(doc2_df, on="hash", how="inner", suffix="_doc2").select([
            pl.col("hash"),
            pl.col("position").alias("position_doc1"),
            pl.col("position_doc2").alias("position_doc2"),
        ]).with_columns([
            (pl.col("position_doc2") - pl.col("position_doc1")).alias("position_offset")
        ]).sort("position_doc1")

    # Build adjacency matrix based on position_doc1 and position_offset thresholds using numpy broadcasting,
    # then find connected components to get initial cluster labels.
    pos1=df_shared_hashes.get_column("position_doc1").to_numpy()
    offsets=df_shared_hashes.get_column("position_offset").to_numpy()
    diff_pos1 = np.abs(pos1[:, None] - pos1[None, :])
    diff_offsets = np.abs(offsets[:, None] - offsets[None, :])

    # Create a list to store the results for each combination of parameters, which we will concatenate into a data frame at the end
    df_list = []

    # Loop over all combinations of position_threshold, offset_threshold, and min_cluster_size to perform clustering and discard small clusters for each combination of parameters
    for position, offset, min_cluster in itertools.product(position_threshold, offset_threshold, min_cluster_size):
        # Create adjacency matrix based on the thresholds and find connected components to get initial cluster labels
        _, raw_labels = connected_components(csr_matrix((diff_pos1 <= position) & (diff_offsets <= offset)), directed=False)

        # Discard clusters that are too small by relabeling their members as -1.
        cluster_df = pl.DataFrame({"cluster_id": raw_labels})

        # Count cluster sizes
        counts = cluster_df.group_by("cluster_id", maintain_order=True).agg(pl.len().alias("count"))

        # Get valid clusters (size >= min_cluster)
        valid_clusters = counts.filter(pl.col("count") >= min_cluster).get_column("cluster_id").to_list()

        # Create a mapping from old cluster IDs to new ones
        relabel_df = counts.filter(pl.col("cluster_id").is_in(valid_clusters)).with_row_index("new_id").select([
            pl.col("cluster_id"),
            pl.col("new_id")
        ])

        # Map clusters: valid clusters get new IDs, others become -1
        new_cluster_df = cluster_df.with_row_index("idx").join(
            relabel_df,
            on="cluster_id",
            how="left"
        ).select([
            pl.col("idx"),
            pl.coalesce(pl.col("new_id"), pl.lit(-1)).alias("cluster_id")
        ]).get_column("cluster_id")

        # Add new column to df_shared_hashes
        df_shared_hashes_now = df_shared_hashes.with_columns([
            pl.lit("rectangle").alias("cluster_method"),
            pl.lit(position).alias("position_threshold"),
            pl.lit(offset).alias("offset_threshold"),
            pl.lit(min_cluster).alias("min_cluster_size"),
            pl.Series(values = new_cluster_df, name="cluster_id"),
        ])

        # Append the data frame for this combination of parameters to the list of data frames, which we will concatenate into a single data frame at the end
        df_list.append(df_shared_hashes_now)

    # Concatenate the data frames for all combinations of parameters into a single data frame
    df_shared_hashes = pl.concat(df_list)

    return df_shared_hashes


def get_df_hash_cluster_linkage(
        doc1_df: pl.DataFrame | None = None,
        doc2_df: pl.DataFrame | None = None,
        df_shared_hashes: pl.DataFrame | None = None,
        distance_threshold: int | list[int] = [10, 20, 30],
        min_cluster_size: int | list[int] = [5],
        cluster_method: str | list[str] = ["euclidean", "cityblock", "chebyshev"],
    ) -> pl.DataFrame:
    """
    Clusters shared winnowed fingerprints using single-linkage hierarchical clustering
    (scipy.cluster.hierarchy.linkage) on the 2-D feature space (position_doc1, position_offset).
    Small clusters (size < min_cluster_size) are relabelled as noise (-1), matching the
    convention used by get_df_hash_cluster_rectangle.

    Relationship to the rectangle version:
        Single linkage with the Chebyshev metric and distance_threshold t is mathematically
        equivalent to the rectangle method with position_threshold = offset_threshold = t,
        because Chebyshev distance = max(|Δpos1|, |Δoffset|) ≤ t iff both differences are ≤ t.
        Euclidean and cityblock (Manhattan) clustering provide alternative neighbourhood shapes.

    Parameters:
    - doc1_df: First document as a Polars DataFrame with columns "hash" and "position".
    - doc2_df: Second document as a Polars DataFrame with columns "hash" and "position".
    - df_shared_hashes: Optional pre-computed shared-hash DataFrame with columns
        "hash", "position_doc1", "position_doc2", "position_offset". If provided,
        the join step is skipped.
    - distance_threshold: Maximum linkage distance at which two clusters are merged
        (fcluster criterion="distance"). Can be an int or list of ints for grid search.
    - min_cluster_size: Minimum number of members for a cluster to be kept; smaller
        clusters are relabelled -1 (noise). Can be an int or list of ints for grid search.
    - cluster_method: Clustering method used. Supported values:
        "euclidean", "cityblock" (Manhattan), "chebyshev" (L∞ / Chessboard).
        Can be a str or list of strs for grid search.

    Returns:
    - A Polars DataFrame with the same columns as the rectangle version:
        - "hash", "position_doc1", "position_doc2", "position_offset"
        - "cluster_id": Assigned cluster ID (0-based), noise labelled -1.
        - "cluster_method", "distance_threshold", "min_cluster_size" are added when more than
          one parameter combination is evaluated; dropped otherwise (single-combo mode).
    """
    # Normalise scalar parameters to lists for uniform iteration
    if isinstance(distance_threshold, int):
        distance_threshold = [distance_threshold]
    if isinstance(min_cluster_size, int):
        min_cluster_size = [min_cluster_size]
    if isinstance(cluster_method, str):
        cluster_method = [cluster_method]

    # Build shared-hash DataFrame if not provided
    if df_shared_hashes is None:
        if doc1_df is None or doc2_df is None:
            raise ValueError("Either df_shared_hashes or both doc1_df and doc2_df must be provided.")
        df_shared_hashes = doc1_df.join(doc2_df, on="hash", how="inner", suffix="_doc2").select([
            pl.col("hash"),
            pl.col("position").alias("position_doc1"),
            pl.col("position_doc2").alias("position_doc2"),
        ]).with_columns([
            (pl.col("position_doc2") - pl.col("position_doc1")).alias("position_offset")
        ]).sort("position_doc1")

    n = len(df_shared_hashes)

    # Pre-extract the 2-D feature matrix (position_doc1, position_offset)
    if n >= 2:
        pos1 = df_shared_hashes.get_column("position_doc1").to_numpy()
        offsets = df_shared_hashes.get_column("position_offset").to_numpy()
        X = np.column_stack([pos1, offsets]).astype(np.float64)

    df_list = []

    for method, dist_thresh, min_cluster in itertools.product(cluster_method, distance_threshold, min_cluster_size):
        if n == 0:
            # Empty input — nothing to cluster
            raw_labels = np.empty(0, dtype=np.intp)
        elif n == 1:
            # Single point — trivially its own cluster of size 1
            raw_labels = np.array([1], dtype=np.intp)  # fcluster is 1-indexed
        else:
            Z = linkage(X, method="single", metric=method, optimal_ordering=False)
            raw_labels = fcluster(Z, t=dist_thresh, criterion="distance")  # 1-indexed

        # Relabel clusters smaller than min_cluster_size as noise (-1)
        cluster_df = pl.DataFrame({"cluster_id": raw_labels.astype(np.int32)})
        counts = cluster_df.group_by("cluster_id", maintain_order=True).agg(pl.len().alias("count"))
        valid_clusters = counts.filter(pl.col("count") >= min_cluster).get_column("cluster_id").to_list()
        relabel_df = (
            counts.filter(pl.col("cluster_id").is_in(valid_clusters))
            .with_row_index("new_id")
            .select([pl.col("cluster_id"), pl.col("new_id")])
        )
        new_cluster_ids = (
            cluster_df.with_row_index("idx")
            .join(relabel_df, on="cluster_id", how="left")
            .select([
                pl.col("idx"),
                pl.coalesce(pl.col("new_id"), pl.lit(-1)).alias("cluster_id"),
            ])
            .get_column("cluster_id")
        )

        df_shared_hashes_now = df_shared_hashes.with_columns([
            pl.lit(method).alias("cluster_method"),
            pl.lit(dist_thresh).alias("distance_threshold"),
            pl.lit(min_cluster).alias("min_cluster_size"),
            pl.Series(values=new_cluster_ids, name="cluster_id"),
        ])
        df_list.append(df_shared_hashes_now)

    df_result = pl.concat(df_list)

    return df_result


def get_df_hash_cluster(
        doc1_df: pl.DataFrame | None = None,
        doc2_df: pl.DataFrame | None = None,
        df_shared_hashes: pl.DataFrame | None = None,
        clustering_method: str | list[str] = ["rectangle", "euclidean", "cityblock", "chebyshev"],
        position_threshold: int | list[int] = [10, 20, 30],
        offset_threshold: int | list[int] = [10, 20, 30],
        distance_threshold: int | list[int] = [10, 20, 30],
        min_cluster_size: int | list[int] = [5],
    ) -> pl.DataFrame:
    """
    Unified wrapper around get_df_hash_cluster_rectangle and
    get_df_hash_cluster_linkage. Dispatches based on clustering_method and,
    when a list of methods is provided, runs each one and merges the results
    via diagonal_relaxed concatenation so that columns specific to one method
    are null for rows produced by the other.

    Parameters:
    - doc1_df: First document as a Polars DataFrame with columns "hash" and "position".
    - doc2_df: Second document as a Polars DataFrame with columns "hash" and "position".
    - df_shared_hashes: Optional pre-computed shared-hash DataFrame. If provided, the join
        step is skipped and doc1_df / doc2_df are ignored.
    - clustering_method: One of "rectangle", "chebyshev", "euclidean",
        "cityblock", or a list of these to run multiple methods simultaneously.
    - position_threshold: (rectangle only) Max position-in-doc1 difference.
    - offset_threshold: (rectangle only) Max position-offset difference.
    - distance_threshold: (euclidean / cityblock / chebyshev only) Max single-linkage distance threshold.
    - min_cluster_size: Minimum cluster size; smaller clusters are relabelled -1 (noise).

    Returns:
    - A Polars DataFrame with all shared-hash rows and their cluster assignments.
      Columns from both methods are present; inapplicable columns are null for each method.
    """
    methods = [clustering_method] if isinstance(clustering_method, str) else list(clustering_method)

    dfs = []
    for method in methods:
        if method == "rectangle":
            df = get_df_hash_cluster_rectangle(
                doc1_df=doc1_df,
                doc2_df=doc2_df,
                df_shared_hashes=df_shared_hashes,
                position_threshold=position_threshold,
                offset_threshold=offset_threshold,
                min_cluster_size=min_cluster_size,
            )

        else:  # any other value is a linkage distance metric
            df = get_df_hash_cluster_linkage(
                doc1_df=doc1_df,
                doc2_df=doc2_df,
                df_shared_hashes=df_shared_hashes,
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                cluster_method=method,
            )
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]

    # If multiple methods were run, concatenate the resulting data frames using diagonal_relaxed concatenation so that columns specific to one method are null for rows produced by the other method, and return the combined data frame
    df_return = pl.concat(dfs, how="diagonal_relaxed")

    # Make cluster_id the last column.
    cols = [col for col in df_return.columns if col not in ["cluster_id"]]
    df_return = df_return.select(cols + ["cluster_id"])

    return df_return


def load_examples_from_jsonl(
        filepath: str,
        verbose: bool = False,
        timer: bool = False,
    ) -> list[dict]:
    """
    This function loads examples from a JSONL file, where each line is a JSON object representing an example with keys "text1", "text2", and "pair_type".
    It filters out any incomplete records (lines that are empty or cannot be parsed as JSON) and returns a list of dictionaries for the valid examples.

    Parameters:
    - filepath: The path to the JSONL file containing the examples.
    - verbose: Whether to print information about the loaded examples.
    - timer: Whether to measure and print the time taken to load the examples.

    Returns:
    - A list of dictionaries, where each dictionary has keys "text1", "text2", and "pair_type" for the valid examples.
    """
    if timer:
        start_time = time.time()

    # Load and filter out any incomplete records (generator may still be running)
    examples = [
        json.loads(line)
        for line in open(filepath)
        if line.strip()
    ]
    # Filter out any examples that are None (incomplete records)
    examples = [e for e in examples if e is not None]

    # Print the time taken to load the examples if timer is True
    if timer:
        time_taken = time.time() - start_time
        print(f"Time taken to load examples: {time_taken:.2f} seconds")

    # Print summary statistics about the loaded examples if verbose
    if verbose:
        # Get the number of examples and the number of positive and negative
        pair_type_counts = Counter(e["pair_type"] for e in examples)
        print(f"Total examples: {len(examples)}, Pair type counts: {dict(pair_type_counts)}, Filepath: {filepath}")

    return examples
