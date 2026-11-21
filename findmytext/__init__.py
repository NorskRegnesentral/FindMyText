"""FindMyText: text-containment detection via winnowed fingerprinting.

    import findmytext

    index = findmytext.build_index("corpus.jsonl", "my_index")
    detector = findmytext.FingerprintChainDetector(index)
    scores = detector.search(query_text)
"""

from importlib import import_module

from .oracle import align

# Lazily resolved so that lightweight submodules (oracle, utils) can be imported
# in environments without the indexing/detection dependencies (polars, scipy, ...).
_LAZY_ATTRS = {
    "FingerprintChainDetector": ".detectors",
    "NbSharedFingerprintsDetector": ".detectors",
    "build_index": ".index_builder",
    "merge_indexes": ".index_builder",
    "DiskBasedIndex": ".indexing",
}

__all__ = [
    "align",
    "build_index",
    "merge_indexes",
    "DiskBasedIndex",
    "FingerprintChainDetector",
    "NbSharedFingerprintsDetector",
    "detectors", "index_builder", "indexing", "oracle", "utils", "winnower",
]


def __getattr__(name):
    if name in _LAZY_ATTRS:
        return getattr(import_module(_LAZY_ATTRS[name], __name__), name)
    if name in ("detectors", "index_builder", "indexing", "utils", "winnower"):
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
