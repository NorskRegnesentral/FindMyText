"""Complete streaming components; bounded views are selected only afterwards."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from numba import njit

from .overlap_inverted import _close_on_exit
from .overlap_stream import atomic_json

_GROUP_DTYPE = np.dtype([
    ("group_id", "<u4"), ("documents", "<u4"),
    ("edges", "<u8"), ("max_degree", "<u8"),
])
_EDGE_DTYPE = np.dtype([("doc_i", "<u4"), ("doc_j", "<u4"), ("count", "<u4")])


@njit(cache=True)
def _root(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True)
def _union_rows(rows, parent, sizes, minimum, degree, k):
    for row in rows:
        first, second = int(row["doc_i"]), int(row["doc_j"])
        if first >= second or second >= len(parent) or row["count"] < k:
            raise ValueError("Expected canonical, in-range edges with count >= K")
        degree[first] += 1
        degree[second] += 1
        a, b = _root(parent, first), _root(parent, second)
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
            minimum[a] = min(minimum[a], minimum[b])


@njit(cache=True)
def _labels(parent, minimum):
    result = np.empty(len(parent), dtype=np.uint32)
    for node in range(len(parent)):
        result[node] = minimum[_root(parent, node)]
    return result


def build_overlap_groups(
    edge_file: str | Path, num_documents: int, k: int, output_dir: str | Path,
) -> dict:
    """Analyze all edges of one verified exact-count output, including isolates.

    Input pairs must be unique, as checked by the overlap pipeline. Group IDs
    are minimum original internal document IDs, independent of edge order.
    Memory is O(documents + chunk size), never O(all edges).
    """
    if not 0 <= num_documents < 2**32 or k < 1:
        raise ValueError("Use a UInt32 document count and positive K")
    source, output = Path(edge_file).resolve(), Path(output_dir).resolve()
    stat = source.stat()
    contract = {
        "format": "findmytext-overlap-groups-v1", "edge_file": str(source),
        "edge_size": stat.st_size, "edge_mtime_ns": stat.st_mtime_ns,
        "documents": num_documents, "k": k,
        "input_requirement": "unique canonical pairs from verified exact-count output",
    }
    if (output / "report.json").exists():
        report = json.loads((output / "report.json").read_text())
        if report["contract"] != contract or report["status"] != "complete":
            raise ValueError("Existing component analysis has different source/settings")
        return report
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    parent = np.arange(num_documents, dtype=np.uint32)
    minimum = parent.copy()
    sizes = np.ones(num_documents, dtype=np.uint32)
    degree = np.zeros(num_documents, dtype=np.uint64)
    with ExitStack() as lifetime:
        edges = np.load(source, mmap_mode="r", allow_pickle=False)
        _close_on_exit(lifetime, edges)
        if edges.ndim != 1 or edges.dtype != _EDGE_DTYPE:
            raise ValueError("Expected one-dimensional UInt32 doc_i/doc_j/count edges")
        edge_count = len(edges)
        for start in range(0, edge_count, 1_000_000):
            _union_rows(edges[start:start + 1_000_000], parent, sizes, minimum, degree, k)
    labels = _labels(parent, minimum)
    group_ids, counts = np.unique(labels, return_counts=True)
    group_edges = np.zeros(num_documents, dtype=np.uint64)
    max_degree = np.zeros(num_documents, dtype=np.uint64)
    np.add.at(group_edges, labels, degree)
    np.maximum.at(max_degree, labels, degree)
    table = np.empty(len(group_ids), dtype=_GROUP_DTYPE)
    table["group_id"], table["documents"] = group_ids, counts
    table["edges"] = group_edges[group_ids] // 2
    table["max_degree"] = max_degree[group_ids]
    members = np.argsort(labels, kind="stable").astype(np.uint32)
    offsets = np.zeros(len(group_ids) + 1, dtype=np.uint64)
    np.cumsum(counts, out=offsets[1:])
    if int(degree.sum()) != 2 * edge_count or int(table["edges"].sum()) != edge_count:
        raise ValueError("Component edge/degree accounting failed")
    nontrivial = table[table["documents"] >= 2]
    largest = nontrivial[np.lexsort((nontrivial["group_id"], -nontrivial["documents"].astype(np.int64)))[:20]]
    bins = [(2, 2), (3, 5), (6, 10), (11, 50), (51, 100), (101, 1000), (1001, None)]
    report = {
        "contract": contract, "status": "complete", "edges": edge_count,
        "documents_with_partners": int(np.count_nonzero(degree)),
        "isolates": int(np.count_nonzero(degree == 0)),
        "components_including_isolates": len(table), "nontrivial_components": len(nontrivial),
        "largest_group_documents": int(largest[0]["documents"]) if len(largest) else 0,
        "largest_groups": [
            {name: int(row[name]) for name in _GROUP_DTYPE.names} for row in largest
        ],
        "component_size_distribution": [
            {
                "minimum": lower, "maximum": upper,
                "groups": int(np.count_nonzero((counts >= lower) & ((counts <= upper) if upper else True))),
                "documents": int(counts[(counts >= lower) & ((counts <= upper) if upper else True)].sum()),
            } for lower, upper in bins
        ],
        "max_degree": int(degree.max()) if len(degree) else 0,
        "seconds": time.perf_counter() - started,
        "interpretation": "Connected components, not cliques or common-passage groups; edges remain undirected.",
    }
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("Edge source changed during component analysis")
    for name, array in (
        ("labels", labels), ("degrees", degree), ("groups", table),
        ("member_offsets", offsets), ("members", members),
    ):
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    atomic_json(output / "report.json", report)
    return report


class OverlapGroups:
    """Read complete groups without constructing a full NetworkX graph."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.report = json.loads((directory / "report.json").read_text())
        if self.report["status"] != "complete":
            raise ValueError("Component analysis is incomplete")
        with ExitStack() as lifetime:
            for name in ("labels", "degrees", "groups", "member_offsets", "members"):
                array = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                _close_on_exit(lifetime, array)
                setattr(self, "_" + name, array)
            self._lifetime = lifetime.pop_all()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._lifetime.close()

    @property
    def groups(self):
        return self._groups

    @property
    def labels(self):
        return self._labels

    @property
    def degrees(self):
        return self._degrees

    def select(self, *, mode="largest", min_documents=2, limit=None):
        if mode not in ("largest", "all_above_min") or min_documents < 2 or (limit is not None and limit < 1):
            raise ValueError("Use largest/all_above_min, minimum >=2 and a positive optional limit")
        rows = self._groups[self._groups["documents"] >= min_documents]
        if mode == "largest":
            rows = rows[np.lexsort((rows["group_id"], -rows["documents"].astype(np.int64)))]
        return rows if limit is None else rows[:limit]

    def members(self, group_id: int) -> np.ndarray:
        position = int(np.searchsorted(self._groups["group_id"], group_id))
        if position == len(self._groups) or self._groups[position]["group_id"] != group_id:
            raise KeyError(f"Unknown group ID: {group_id}")
        start, stop = self._member_offsets[position:position + 2]
        return self._members[int(start):int(stop)]
