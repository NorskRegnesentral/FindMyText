"""Side-car metadata resolution: turn a matched document id into a human title
and a working URL.

The heavy lifting is done offline by the ``web/tools/build_*.py`` scripts, which
write compact gzip-JSON maps next to each corpus index:

* ``<corpus>/titles.json.gz`` — ``{doc_id: title}`` (arXiv, Wikipedia).
* ``<corpus>/urls.json.gz``   — ``{doc_id: {"u": url, "ts": crawl_time}}`` (HPLT).

Maps are loaded lazily on first use and cached in memory (one small dict per
corpus). Everything degrades gracefully: if a map file is missing the resolver
simply returns ``None`` and the UI falls back to showing the raw document id.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import threading
from typing import Optional


class MetaResolver:
    """Thread-safe, lazily-loaded cache of per-corpus side-car maps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}

    def _load(self, path: str) -> dict:
        with self._lock:
            cached = self._cache.get(path)
        if cached is not None:
            return cached
        data: dict = {}
        try:
            if path and os.path.exists(path):
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    data = json.load(fh)
        except Exception:  # noqa: BLE001 - a broken map must not break search
            data = {}
        with self._lock:
            self._cache[path] = data
        return data

    def lookup(self, path: Optional[str], doc_id: str):
        if not path:
            return None
        return self._load(path).get(doc_id)


RESOLVER = MetaResolver()


def wayback_url(url: str, ts: str) -> str:
    """Build a Wayback Machine URL that redirects to the snapshot nearest ``ts``."""
    compact = re.sub(r"[^0-9]", "", ts or "")[:14]
    return f"https://web.archive.org/web/{compact}/{url}"
