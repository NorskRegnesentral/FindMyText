"""Visualization and analysis helpers for exploring a document similarity index.

These functions require the optional "networkx" and "matplotlib" packages
(install with ``pip install findmytext[viz]``); "polars" is already a core
dependency. The heavy ones are only imported when called so the core package
has no hard dependency on them.
"""

import itertools
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import polars as pl

from . import detectors, oracle, web, winnower

if TYPE_CHECKING:
    import matplotlib.figure
    import networkx as nx

    from .indexing import DiskBasedIndex


def build_similarity_graph(
    index: "DiskBasedIndex",
    min_similarity: int = 1,
    max_documents: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
    method: str = "inverted",
) -> "nx.Graph":
    """Build a weighted graph of documents connected by shared fingerprints.

    Nodes are external document IDs; each edge is weighted by the number of
    fingerprints shared between the two documents (``S(i, j)``).

    Normal mode (``doc_ids=None``): edges with fewer than ``min_similarity``
    shared fingerprints are dropped first. If more than ``max_documents`` nodes
    remain, only the ``max_documents`` with the highest *weighted degree* (i.e.
    the sum of the weights of their surviving edges) are kept, and edges outside
    that set are dropped along with them. Weighted degree was chosen over e.g.
    unweighted degree or centrality measures because it directly reflects total
    shared content per document, which is the most natural way to prioritise
    documents that overlap heavily with several others.

    Selection mode (``doc_ids`` given): ``min_similarity`` and ``max_documents``
    are bypassed entirely -- the graph contains exactly the requested documents
    (as nodes, even if isolated) and every edge between them, regardless of
    weight. Since a caller-selected set may include weakly related documents by
    mistake, a warning is issued for every pair whose shared-fingerprint count
    (0 if no edge exists) is below ``min_similarity``.

    Args:
        index: A DiskBasedIndex whose similarities were computed with
            ``findmytext.similarity.compute_similarities``.
        min_similarity: Minimum shared fingerprints to keep an edge (normal
            mode), or the warning threshold for weak pairs (selection mode).
        max_documents: If set (normal mode only), restrict the graph to at most
            this many documents.
        doc_ids: If set, switch to selection mode and use exactly this set of
            documents instead of `min_similarity`/`max_documents` filtering.
        method: Similarity method whose results to use.

    Returns:
        A networkx.Graph with a "weight" edge attribute holding the shared
        fingerprint count.

    Raises:
        FileNotFoundError: If similarities for ``method`` were not computed.
    """
    import networkx as nx

    similarities = index.load_similarities(method)

    if doc_ids is not None:
        doc_ids = list(dict.fromkeys(doc_ids))  # de-duplicate, preserve order
        doc_id_set = set(doc_ids)
        graph = nx.Graph()
        graph.add_nodes_from(doc_ids)
        for row in similarities:
            doc_a = index.to_external_doc_id[int(row["doc_i"])]
            doc_b = index.to_external_doc_id[int(row["doc_j"])]
            if doc_a in doc_id_set and doc_b in doc_id_set:
                graph.add_edge(doc_a, doc_b, weight=int(row["count"]))

        for doc_a, doc_b in itertools.combinations(doc_ids, 2):
            weight = graph.get_edge_data(doc_a, doc_b, default={}).get("weight", 0)
            if weight < min_similarity:
                warnings.warn(
                    f"Requested documents {doc_a!r} and {doc_b!r} share only "
                    f"{weight} fingerprint(s) (< min_similarity={min_similarity}); "
                    "double-check they belong together."
                )
        return graph

    graph = nx.Graph()
    for row in similarities:
        count = int(row["count"])
        if count < min_similarity:
            continue
        doc_a = index.to_external_doc_id[int(row["doc_i"])]
        doc_b = index.to_external_doc_id[int(row["doc_j"])]
        graph.add_edge(doc_a, doc_b, weight=count)

    if max_documents is not None and graph.number_of_nodes() > max_documents:
        weighted_degree = dict(graph.degree(weight="weight"))
        top_nodes = sorted(weighted_degree, key=weighted_degree.get, reverse=True)[
            :max_documents
        ]
        graph = graph.subgraph(top_nodes).copy()

    return graph


def plot_similarity_graph(
    graph: "nx.Graph",
    figsize=(10, 10),
    layout: str = "spring",
    seed: int = 0,
    min_node_size: float = 100,
    node_size_scale: float = 800,
    min_edge_width: float = 0.5,
    max_edge_width: float = 4.0,
    font_size: int = 8,
    cmap_name: str = "viridis",
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Draw a similarity graph produced by build_similarity_graph.

    Node size is proportional to weighted degree (total fingerprints shared with
    all neighbours); edge width and color are proportional to the pairwise
    shared fingerprint count.

    Args:
        graph: A graph as returned by build_similarity_graph.
        figsize: Matplotlib figure size.
        layout: One of "spring", "kamada_kawai", or "circular".
        seed: Random seed for the "spring" layout (for reproducible figures).
        min_node_size, node_size_scale: Control node marker size as
            ``min_node_size + node_size_scale * (weighted_degree / max_weighted_degree)``.
        min_edge_width, max_edge_width: Range of edge line widths, scaled by the
            edge's shared-fingerprint count relative to the largest one in the graph.
        font_size: Font size for document ID labels.
        cmap_name: Matplotlib colormap used for edge color (by shared fingerprint count).
        title: Optional plot title.

    Returns:
        The created matplotlib Figure.

    Raises:
        ValueError: If the graph has no nodes, or layout is not recognised.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    if graph.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes to plot (try lowering min_similarity)")

    fig, ax = plt.subplots(figsize=figsize)

    if layout == "spring":
        pos = nx.spring_layout(graph, seed=seed, weight="weight")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph, weight="weight")
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    weighted_degree = dict(graph.degree(weight="weight"))
    max_degree = max(weighted_degree.values(), default=1) or 1
    node_sizes = [
        min_node_size + node_size_scale * weighted_degree[n] / max_degree
        for n in graph.nodes()
    ]

    edge_weights = [d["weight"] for _, _, d in graph.edges(data=True)]
    max_weight = max(edge_weights, default=1) or 1
    min_weight = min(edge_weights, default=1)
    edge_widths = [
        min_edge_width + (max_edge_width - min_edge_width) * w / max_weight
        for w in edge_weights
    ]

    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=node_sizes, node_color="#4C72B0", alpha=0.85
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_weights,
        edge_cmap=plt.get_cmap(cmap_name),
        alpha=0.6,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=font_size)

    if edge_weights:
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=plt.get_cmap(cmap_name),
            norm=plt.Normalize(vmin=min_weight, vmax=max_weight),
        )
        scalar_mappable.set_array([])
        fig.colorbar(
            scalar_mappable,
            ax=ax,
            label="Shared fingerprints",
            fraction=0.046,
            pad=0.04,
        )

    ax.set_axis_off()
    if title:
        ax.set_title(title)

    return fig


def verify_similarity_counts(
    index: "DiskBasedIndex",
    id_to_text: Dict[str, str],
    check_n_documents: int,
    length: int = 4,
    window_size: int = 6,
    verbose: bool = True,
    seed: Optional[int] = None,
    method: str = "inverted",
) -> Dict[str, Any]:
    """Independently re-verify stored similarity counts for a random document sample.

    Randomly selects up to ``check_n_documents`` unique documents (every document
    if ``check_n_documents`` is at least the number indexed), recomputes their
    winnowed fingerprints from scratch, and compares every pairwise shared-
    fingerprint count against what is stored in the similarity index. Since
    documents with zero shared fingerprints are never stored, a missing stored
    pair is treated as a count of 0 rather than a mismatch. A warning is issued
    for every mismatch found.

    Args:
        index: A DiskBasedIndex with exhaustive similarities for ``method``.
        id_to_text: Mapping from external document ID to its original text.
        check_n_documents: Number of documents to randomly sample and cross-check.
        length: k-gram length used for fingerprinting (must match the index).
        window_size: Winnowing window size (must match the index).
        verbose: If True, print progress approximately every 10% of pairs checked.
        seed: Optional seed for reproducible document sampling.
        method: Similarity method whose results to verify.

    Returns:
        A dict with "n_documents_checked", "n_pairs_checked", "n_mismatches", and
        "mismatches" (a list of (doc_id_a, doc_id_b, manual_count, stored_count)).
    """
    import random

    all_doc_ids = [
        index.to_external_doc_id[i] for i in range(len(index.to_external_doc_id))
    ]
    n_to_check = min(check_n_documents, len(all_doc_ids))
    sampled_doc_ids = random.Random(seed).sample(all_doc_ids, n_to_check)

    total_pairs = n_to_check * (n_to_check - 1) // 2
    print(f"Checking {n_to_check} documents -> {total_pairs:,} pairs to verify.")
    if total_pairs == 0:
        return {
            "n_documents_checked": n_to_check,
            "n_pairs_checked": 0,
            "n_mismatches": 0,
            "mismatches": [],
        }

    stored_pairs = {
        (
            index.to_external_doc_id[int(row["doc_i"])],
            index.to_external_doc_id[int(row["doc_j"])],
        ): int(row["count"])
        for row in index.load_similarities(method)
    }
    fingerprinter = winnower.Winnower(length=length, window_size=window_size)
    fingerprint_sets = {
        doc_id: set(
            fingerprinter.get_winnowed_fingerprints(id_to_text[doc_id])[0].tolist()
        )
        for doc_id in sampled_doc_ids
    }

    mismatches = []
    checked = 0
    next_percent_threshold = 10
    for doc_id_a, doc_id_b in itertools.combinations(sampled_doc_ids, 2):
        manual_count = len(fingerprint_sets[doc_id_a] & fingerprint_sets[doc_id_b])
        stored_count = (
            stored_pairs.get((doc_id_a, doc_id_b))
            or stored_pairs.get((doc_id_b, doc_id_a))
            or 0
        )
        if manual_count != stored_count:
            warnings.warn(
                f"Similarity mismatch for ({doc_id_a}, {doc_id_b}): "
                f"manual={manual_count}, stored={stored_count}"
            )
            mismatches.append((doc_id_a, doc_id_b, manual_count, stored_count))

        checked += 1
        percent_done = checked * 100 // total_pairs
        if verbose and percent_done >= next_percent_threshold:
            print(f"  ... {percent_done}% ({checked:,}/{total_pairs:,} pairs)")
            next_percent_threshold += 10

    print(
        f"Done: checked {checked:,} pairs across {n_to_check} documents, "
        f"found {len(mismatches)} mismatch(es)."
    )

    return {
        "n_documents_checked": n_to_check,
        "n_pairs_checked": checked,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }


def rank_documents_by_node_strength(
    index: "DiskBasedIndex", min_similarity: int = 1, method: str = "inverted"
) -> "pl.DataFrame":
    """Rank documents by *node strength* (weighted degree).

    Node strength is the sum of shared-fingerprint counts a document has with
    every other document, restricted to pairs with at least ``min_similarity``
    shared fingerprints. Documents with no qualifying pairs are omitted.

    Args:
        index: A DiskBasedIndex with similarities computed for ``method``.
        min_similarity: Only count pairs with at least this many shared fingerprints.
        method: Similarity method whose results to use.

    Returns:
        A DataFrame with columns "doc_id" and "node_strength", sorted in
        descending order of node_strength.
    """
    strengths: Dict[str, int] = {}
    for row in index.load_similarities(method):
        count = int(row["count"])
        if count < min_similarity:
            continue
        doc_a = index.to_external_doc_id[int(row["doc_i"])]
        doc_b = index.to_external_doc_id[int(row["doc_j"])]
        strengths[doc_a] = strengths.get(doc_a, 0) + count
        strengths[doc_b] = strengths.get(doc_b, 0) + count

    sorted_items = sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)
    return pl.DataFrame(
        {
            "doc_id": [doc_id for doc_id, _ in sorted_items],
            "node_strength": [strength for _, strength in sorted_items],
        }
    )


def find_potential_source_clusters(
    index: "DiskBasedIndex",
    min_similarity: int = 50,
    min_cluster_size: int = 3,
    hub_dominance_ratio: float = 1.3,
    method: str = "inverted",
) -> List[Dict[str, Any]]:
    """Heuristically flag clusters of documents that may derive from a common source.

    Builds a similarity graph restricted to edges with at least ``min_similarity``
    shared fingerprints, then examines each connected component with at least
    ``min_cluster_size`` documents. Within a component, the document with the
    highest node strength (see ``rank_documents_by_node_strength``) is treated as
    the candidate "hub" (common source). The component is flagged if the hub's
    average edge weight to the other members is at least ``hub_dominance_ratio``
    times higher than the average edge weight among the remaining (non-hub)
    members -- the star-like pattern expected when several documents each copy
    from the hub, but only partially overlap with each other.

    This is a heuristic, not proof of copying direction or intent; use it to
    prioritise document sets for closer inspection (e.g. with
    ``analyze_shared_passages`` and ``show_pairwise_alignment``).

    Args:
        index: A DiskBasedIndex with similarities computed for ``method``.
        min_similarity: Minimum shared fingerprints for an edge to be considered.
        min_cluster_size: Minimum number of documents in a component to examine.
        hub_dominance_ratio: Minimum ratio of (hub's mean edge weight to members)
            over (mean edge weight among members) required to flag a component.
        method: Similarity method whose results to use.

    Returns:
        A list of dicts (sorted by descending dominance ratio), each with keys
        "hub", "members", "component_size", "hub_mean_weight",
        "internal_mean_weight", and "dominance_ratio".
    """
    import networkx as nx

    graph = build_similarity_graph(index, min_similarity=min_similarity, method=method)

    flagged = []
    for component in nx.connected_components(graph):
        if len(component) < min_cluster_size:
            continue
        sub = graph.subgraph(component)
        node_strength = dict(sub.degree(weight="weight"))
        hub = max(node_strength, key=node_strength.get)
        members = [n for n in component if n != hub]

        hub_weights = [sub[hub][m]["weight"] for m in members if sub.has_edge(hub, m)]
        internal_weights = [
            d["weight"] for u, v, d in sub.edges(data=True) if hub not in (u, v)
        ]

        hub_mean = sum(hub_weights) / len(hub_weights) if hub_weights else 0.0
        internal_mean = (
            sum(internal_weights) / len(internal_weights) if internal_weights else 0.0
        )
        ratio = (
            float("inf")
            if internal_mean == 0
            else (hub_mean / internal_mean if hub_mean else 0.0)
        )

        if ratio >= hub_dominance_ratio:
            flagged.append(
                {
                    "hub": hub,
                    "members": members,
                    "component_size": len(component),
                    "hub_mean_weight": hub_mean,
                    "internal_mean_weight": internal_mean,
                    "dominance_ratio": ratio,
                }
            )

    flagged.sort(key=lambda cluster: cluster["dominance_ratio"], reverse=True)
    return flagged


def locate_shared_passage(
    index: "DiskBasedIndex",
    hub_doc_id: str,
    other_doc_id: str,
    id_to_text: Dict[str, str],
    clustering_params: Optional[Dict[str, Any]] = None,
    top_k: int = 50,
    min_fingerprints: int = 5,
    context_chars: int = 200,
) -> Optional[Dict[str, Any]]:
    """Cheaply localise the shared passage between two documents before aligning.

    Running the Smith-Waterman aligner (``findmytext.oracle``) over two full
    documents is expensive -- exactly the problem
    ``detectors.FingerprintChainDetector`` / ``web.TextContainmentDetector``
    already solve for detection. This function reuses that same fingerprint-
    clustering machinery to find the *largest chain* of shared fingerprints
    between ``hub_doc_id`` and ``other_doc_id`` and returns a small,
    padded character-offset excerpt around it in *both* documents, so a caller
    can run the expensive aligner on that excerpt instead of the full text.

    Args:
        index: A DiskBasedIndex built from a corpus that includes both documents.
        hub_doc_id: The candidate common-source document (must be indexed).
        other_doc_id: The document to compare against the hub (must be indexed).
        id_to_text: Mapping from external document ID to its original text.
        clustering_params: Overrides for ``web.DEFAULT_CLUSTERING_PARAMS``
            (method, position_threshold, offset_threshold, distance_threshold,
            min_cluster_size).
        top_k: Number of closest candidates to retrieve per query fingerprint set.
        min_fingerprints: Minimum shared fingerprints for a candidate to be
            considered (passed to ``index.get_closest_matches_with_positions``).
        context_chars: Extra characters of surrounding context to include on
            each side of the located span, for readability.

    Returns:
        None if no qualifying fingerprint cluster is found (e.g. hub_doc_id
        does not appear among other_doc_id's top-k candidates, or no cluster
        survives ``min_cluster_size`` -- this can legitimately happen even for
        documents with many *coincidentally* shared fingerprints, e.g. very
        large documents that accumulate short, scattered common-phrase matches
        without ever sharing one contiguous passage). Otherwise a dict with:
        "n_shared_fingerprints_in_cluster", "hub_span"/"other_span" (exact
        token-derived character spans), "hub_excerpt_span"/"other_excerpt_span"
        (padded spans), and "hub_excerpt"/"other_excerpt" (the excerpt text).
    """
    params = {**web.DEFAULT_CLUSTERING_PARAMS, **(clustering_params or {})}

    # Use the SAME winnowing parameters as the index, not a denser window_size=1
    # (appropriate for _BaseDetector's short *external* queries): both documents
    # here are already-indexed, potentially very large documents, and a denser
    # unwinnowed fingerprint set would multiply purely coincidental 31-bit hash
    # collisions across a huge token space, drowning any real signal.
    query_winnower = winnower.Winnower(
        length=index.winnower.length,
        window_size=index.winnower.window_size,
        base=index.winnower.base,
        punctuation=index.winnower.punctuation,
    )
    other_text = id_to_text[other_doc_id]
    fingerprints, positions = query_winnower.get_winnowed_fingerprints(other_text)
    if len(fingerprints) == 0:
        return None

    closest = index.get_closest_matches_with_positions(
        query=fingerprints,
        top_k=top_k,
        min_fingerprints=min_fingerprints,
        verbose=False,
    )
    df_closest = detectors.convert_closest_matches_with_positions_to_df(closest)
    df_match = df_closest.filter(pl.col("doc_match_id") == hub_doc_id).select(
        ["hash", "position"]
    )
    if df_match.height == 0:
        return None

    df_query = pl.DataFrame({"hash": fingerprints, "position": positions})
    df_shared = web.TextContainmentDetector._shared_positions(df_query, df_match)
    if df_shared.height == 0:
        return None

    df_clustered = web.TextContainmentDetector._cluster_shared(df_shared, params)
    non_noise = df_clustered.filter(pl.col("cluster_id") != -1)
    if non_noise.height == 0:
        return None

    sizes = (
        non_noise.group_by("cluster_id")
        .agg(pl.col("hash").n_unique().alias("n_unique"))
        .sort("n_unique", descending=True)
    )
    largest_cluster_id = sizes[0, "cluster_id"]
    n_shared = int(sizes[0, "n_unique"])
    cluster_rows = non_noise.filter(pl.col("cluster_id") == largest_cluster_id)

    other_positions = cluster_rows["position_doc1"].to_list()
    hub_positions = cluster_rows["position_doc2"].to_list()

    hub_text = id_to_text[hub_doc_id]
    hub_tokens = web.tokenize_with_offsets(
        hub_text, punctuation=index.winnower.punctuation
    )
    other_tokens = web.tokenize_with_offsets(
        other_text, punctuation=index.winnower.punctuation
    )
    k = index.winnower.length

    def _token_span_to_char_span(tokens, min_pos, max_pos):
        start_idx = max(0, min_pos)
        end_idx = min(max_pos + k - 1, len(tokens) - 1)
        return tokens[start_idx][0], tokens[end_idx][1]

    hub_start, hub_end = _token_span_to_char_span(
        hub_tokens, min(hub_positions), max(hub_positions)
    )
    other_start, other_end = _token_span_to_char_span(
        other_tokens, min(other_positions), max(other_positions)
    )

    hub_excerpt_span = (
        max(0, hub_start - context_chars),
        min(len(hub_text), hub_end + context_chars),
    )
    other_excerpt_span = (
        max(0, other_start - context_chars),
        min(len(other_text), other_end + context_chars),
    )

    return {
        "hub_doc_id": hub_doc_id,
        "other_doc_id": other_doc_id,
        "n_shared_fingerprints_in_cluster": n_shared,
        "hub_span": (hub_start, hub_end),
        "other_span": (other_start, other_end),
        "hub_excerpt_span": hub_excerpt_span,
        "other_excerpt_span": other_excerpt_span,
        "hub_excerpt": hub_text[hub_excerpt_span[0] : hub_excerpt_span[1]],
        "other_excerpt": other_text[other_excerpt_span[0] : other_excerpt_span[1]],
    }


def analyze_shared_passages(
    index: "DiskBasedIndex",
    id_to_text: Dict[str, str],
    hub_doc_id: str,
    other_doc_ids: List[str],
    clustering_params: Optional[Dict[str, Any]] = None,
    min_region_length: int = 20,
    multiple_passes: bool = False,
) -> "pl.DataFrame":
    """Locate where in ``hub_doc_id`` each of ``other_doc_ids`` aligns.

    For each other document, ``locate_shared_passage`` first narrows the search
    to a small excerpt around the largest chain of shared fingerprints (see
    that function's docstring for why this matters); the Smith-Waterman local
    aligner (``findmytext.oracle``) is then run on just that excerpt, not the
    full documents. This checks whether several documents reused the *same*
    passage of the hub document, or different, non-overlapping passages --
    e.g. Doc1 may have copied an earlier chapter than Doc3, even though both
    derive from the same hub document.

    All local-alignment regions of at least ``min_region_length`` characters
    are combined into a single ``[hub_start, hub_end]`` span (the outer bounds
    of their union, in the hub document's full-text coordinates) along with
    their total matched character count.

    Args:
        index: A DiskBasedIndex built from a corpus that includes the hub and
            all other documents.
        id_to_text: Mapping from external document ID to its original text.
        hub_doc_id: The candidate common-source document.
        other_doc_ids: Documents to align against the hub.
        clustering_params: Overrides for ``web.DEFAULT_CLUSTERING_PARAMS``,
            forwarded to ``locate_shared_passage``.
        min_region_length: Minimum character length for an alignment region to count.
        multiple_passes: Whether to look for multiple, non-overlapping matching
            regions per pair (passed to ``oracle.align``).

    Returns:
        A DataFrame with columns: doc_id, n_regions, hub_start, hub_end,
        total_matched_chars. Documents with no qualifying fingerprint cluster
        or alignment region get hub_start/hub_end = None and
        n_regions/total_matched_chars = 0.
    """
    rows = []
    for other_id in other_doc_ids:
        location = locate_shared_passage(
            index, hub_doc_id, other_id, id_to_text, clustering_params=clustering_params
        )
        if location is None:
            rows.append(
                {
                    "doc_id": other_id,
                    "n_regions": 0,
                    "hub_start": None,
                    "hub_end": None,
                    "total_matched_chars": 0,
                }
            )
            continue

        alignment = oracle.align(
            location["hub_excerpt"],
            location["other_excerpt"],
            multiple_passes=multiple_passes,
        )
        excerpt_offset = location["hub_excerpt_span"][0]
        regions = [
            r
            for r in alignment.regions
            if (r["end1"] - r["start1"]) >= min_region_length
        ]
        if not regions:
            rows.append(
                {
                    "doc_id": other_id,
                    "n_regions": 0,
                    "hub_start": None,
                    "hub_end": None,
                    "total_matched_chars": 0,
                }
            )
            continue
        rows.append(
            {
                "doc_id": other_id,
                "n_regions": len(regions),
                "hub_start": excerpt_offset + min(r["start1"] for r in regions),
                "hub_end": excerpt_offset + max(r["end1"] for r in regions),
                "total_matched_chars": sum(r["end1"] - r["start1"] for r in regions),
            }
        )
    return pl.DataFrame(rows)


def plot_shared_passage_overlap(
    passage_summary: "pl.DataFrame", hub_doc_id: str, figsize=(9, 4)
) -> "matplotlib.figure.Figure":
    """Draw a timeline of matched passage locations within the hub document.

    Each bar spans ``[hub_start, hub_end]`` (see ``analyze_shared_passages``) for
    one other document. Overlapping bars indicate the documents likely reused
    the same passage of the hub; non-overlapping bars indicate different passages.

    Args:
        passage_summary: The DataFrame returned by ``analyze_shared_passages``.
        hub_doc_id: The hub document ID (used only for the x-axis label).
        figsize: Matplotlib figure size.

    Returns:
        The created matplotlib Figure.

    Raises:
        ValueError: If no document has a qualifying aligned passage.
    """
    import matplotlib.pyplot as plt

    plotted = passage_summary.drop_nulls(subset=["hub_start", "hub_end"])
    if plotted.is_empty():
        raise ValueError("No documents with a qualifying aligned passage to plot")

    fig, ax = plt.subplots(figsize=figsize)
    for row_index, row in enumerate(plotted.iter_rows(named=True)):
        ax.barh(
            row_index,
            row["hub_end"] - row["hub_start"],
            left=row["hub_start"],
            color="#4C72B0",
            alpha=0.8,
        )
    ax.set_yticks(range(len(plotted)))
    ax.set_yticklabels(plotted["doc_id"].to_list())
    ax.set_xlabel(f"Character position in hub document {hub_doc_id!r}")
    ax.set_title(
        "Matched passage location per document (overlapping bars \u21d2 same passage)"
    )
    ax.invert_yaxis()
    return fig


def show_pairwise_alignment(
    index: "DiskBasedIndex",
    id_to_text: Dict[str, str],
    doc_id_a: str,
    doc_id_b: str,
    clustering_params: Optional[Dict[str, Any]] = None,
    multiple_passes: bool = True,
) -> "oracle.AlignmentResult":
    """Compute and display (via ``AlignmentResult.show()``) the local alignment
    between two documents, for visually inspecting a specific shared passage.

    Uses ``locate_shared_passage`` to narrow to a small excerpt around the
    largest shared-fingerprint chain first, so the (expensive) Smith-Waterman
    aligner only runs on that excerpt rather than the full documents.

    Args:
        index: A DiskBasedIndex built from a corpus that includes both documents.
        id_to_text: Mapping from external document ID to its original text.
        doc_id_a: First document ID (treated as the hub/source).
        doc_id_b: Second document ID (treated as the derived/query document).
        clustering_params: Overrides for ``web.DEFAULT_CLUSTERING_PARAMS``,
            forwarded to ``locate_shared_passage``.
        multiple_passes: Whether to look for multiple matching regions.

    Returns:
        The AlignmentResult (already displayed as a side-by-side HTML diff).

    Raises:
        ValueError: If no qualifying fingerprint cluster is found between the
            two documents (see ``locate_shared_passage``).
    """
    location = locate_shared_passage(
        index, doc_id_a, doc_id_b, id_to_text, clustering_params=clustering_params
    )
    if location is None:
        raise ValueError(
            f"No shared fingerprint cluster found between {doc_id_a!r} and {doc_id_b!r}"
        )
    alignment = oracle.align(
        location["hub_excerpt"],
        location["other_excerpt"],
        multiple_passes=multiple_passes,
    )
    alignment.show()
    return alignment
