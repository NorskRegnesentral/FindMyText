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
import sqlite3
import threading
from typing import Optional


class MetaResolver:
    """Thread-safe, lazily-loaded cache of per-corpus side-car maps.

    Small maps (arXiv/Wikipedia titles, HPLT sample urls) are gzip-JSON dicts
    loaded fully into memory. Very large maps (e.g. the ~50M-row HPLT url map)
    are stored as SQLite and queried on disk, so the web process never has to
    hold tens of millions of entries in RAM. A map path ending in ``.sqlite``
    (or ``.db``) is treated as SQLite; anything else as gzip-JSON.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._conns: dict[str, Optional[sqlite3.Connection]] = {}

    @staticmethod
    def _is_sqlite(path: str) -> bool:
        return path.endswith(".sqlite") or path.endswith(".db")

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

    def _conn(self, path: str) -> Optional[sqlite3.Connection]:
        with self._lock:
            if path in self._conns:
                return self._conns[path]
        conn: Optional[sqlite3.Connection] = None
        try:
            if path and os.path.exists(path):
                conn = sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True, check_same_thread=False
                )
        except Exception:  # noqa: BLE001 - a broken db must not break the app
            conn = None
        with self._lock:
            self._conns[path] = conn
        return conn

    def lookup(self, path: Optional[str], doc_id: str):
        if not path:
            return None
        if self._is_sqlite(path):
            conn = self._conn(path)
            if conn is None:
                return None
            try:
                with self._lock:
                    row = conn.execute(
                        "SELECT u, ts FROM urls WHERE id = ? LIMIT 1", (doc_id,)
                    ).fetchone()
            except Exception:  # noqa: BLE001
                return None
            if not row:
                return None
            return {"u": row[0], "ts": row[1]}
        return self._load(path).get(doc_id)

    def search_urls(self, path: Optional[str], query: str, limit: int = 20):
        """Full-text search a SQLite url map by domain, returning ``[(id, url, ts)]``.

        Requires a companion FTS5 table ``url_fts(host, url, id)`` (built by
        ``web/tools/build_hplt.py --fts``). The query is reduced to alphanumeric
        tokens; the last token is treated as a prefix so typing ``wikip`` already
        matches ``wikipedia``. ``ts`` (the crawl timestamp, for a Wayback link)
        is joined back from the ``urls`` table.
        """
        if not path or not self._is_sqlite(path):
            return []
        tokens = re.sub(r"[^0-9a-z]+", " ", (query or "").lower()).split()
        if not tokens:
            return []
        match = " ".join(tokens[:-1] + [tokens[-1] + "*"])
        conn = self._conn(path)
        if conn is None:
            return []
        try:
            with self._lock:
                rows = conn.execute(
                    "SELECT f.id, f.url, u.ts FROM url_fts f "
                    "JOIN urls u ON u.id = f.id "
                    "WHERE url_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, limit),
                ).fetchall()
        except Exception:  # noqa: BLE001 - missing/broken FTS must not 500
            return []
        return [(r[0], r[1], r[2]) for r in rows]



RESOLVER = MetaResolver()


class TitleSearcher:
    """Lazily-built, cached substring search over a corpus' title map.

    Reuses the title dict already loaded by ``RESOLVER`` (``{doc_id: title}``)
    and prepares a flat ``[(title_lower, doc_id)]`` list once per map, so users
    can search *exactly* the documents that are in the corpus. Prefix matches
    are ranked ahead of other substring matches; results are alphabetical.
    """

    def __init__(self, resolver: MetaResolver) -> None:
        self._resolver = resolver
        self._lock = threading.Lock()
        self._prepared: dict[str, list[tuple[str, str]]] = {}

    def _prepare(self, path: str) -> list[tuple[str, str]]:
        with self._lock:
            cached = self._prepared.get(path)
        if cached is not None:
            return cached
        data = self._resolver._load(path)  # {doc_id: title}
        prepared = sorted(
            ((str(title).lower(), doc_id) for doc_id, title in data.items() if title),
            key=lambda x: x[0],
        )
        with self._lock:
            self._prepared[path] = prepared
        return prepared

    def search(self, path: Optional[str], query: str, limit: int = 20) -> list[str]:
        """Return up to ``limit`` matching doc ids (prefix matches first)."""
        q = (query or "").strip().lower()
        if not path or len(q) < 2:
            return []
        prepared = self._prepare(path)
        starts: list[str] = []
        contains: list[str] = []
        for title_lower, doc_id in prepared:
            if title_lower.startswith(q):
                starts.append(doc_id)
                if len(starts) >= limit:
                    break
            elif q in title_lower and len(contains) < limit:
                contains.append(doc_id)
        return (starts + contains)[:limit]


SEARCHER = TitleSearcher(RESOLVER)


def wayback_url(url: str, ts: str) -> str:
    """Build a Wayback Machine URL that redirects to the snapshot nearest ``ts``."""
    compact = re.sub(r"[^0-9]", "", ts or "")[:14]
    return f"https://web.archive.org/web/{compact}/{url}"
