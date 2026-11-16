"""Indexed batch retrieval of complete HPLT JSON records and metadata."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import orjson
import zstandard as zstd
from filelock import FileLock

from .indexing import _DocIdMap
from .overlap_inverted import _close_on_exit, _source_identity

_FORMAT = "findmytext-corpus-documents-v1"
_CONTENT_FIELDS = ("text", "xml")


def _corpus_files(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("*.jsonl.zst"))
    if not paths:
        raise FileNotFoundError(f"No JSONL Zstandard shards in {directory}")
    return [
        {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


class CorpusDocumentStore:
    """Read-only, context-managed lookup; raw fields are never normalized away.

    metadata_only excludes text and XML, but preserves all other raw fields.
    Internal IDs require the same index snapshot bound when the store was built.
    Missing IDs are explicit errors, not silent partial results.
    """

    def __init__(self, path: str | Path, *, index_dir: str | Path | None = None):
        root = Path(path).resolve()
        with ExitStack() as lifetime:
            self.connection = sqlite3.connect(root.as_uri() + "?mode=ro", uri=True)
            lifetime.callback(self.connection.close)
            self.connection.execute("PRAGMA query_only=ON")
            row = self.connection.execute("SELECT value FROM info WHERE key='contract'").fetchone()
            if row is None:
                raise ValueError("Document store has no format/source contract")
            self.meta = json.loads(row[0])
            if self.meta["format"] != _FORMAT or not self.meta.get("complete"):
                raise ValueError("Document store is incomplete or has an unsupported format")
            self._ids = None
            if index_dir is not None:
                index = Path(index_dir).resolve()
                if self.meta["index_source"] != _source_identity(index):
                    raise ValueError("Document store and index snapshots do not match")
                offsets = np.load(index / "doc_name_offsets.npy", mmap_mode="r")
                names = np.load(index / "doc_name_bytes.npy", mmap_mode="r")
                _close_on_exit(lifetime, offsets)
                _close_on_exit(lifetime, names)
                self._ids = _DocIdMap(offsets, names)
            self._decompressor = zstd.ZstdDecompressor()
            self._lifetime = lifetime.pop_all()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._lifetime.close()

    def iter_many(
        self, document_ids: Iterable[str | int | np.integer], *,
        id_kind: str = "external", metadata_only: bool = False,
    ) -> Iterator[tuple[str | int, dict]]:
        if id_kind not in ("external", "internal"):
            raise ValueError("id_kind must be 'external' or 'internal'")
        normalized = []
        external_ids = []
        seen = set()
        for document_id in document_ids:
            if id_kind == "external":
                if not isinstance(document_id, str) or not document_id:
                    raise ValueError("External IDs must be nonempty strings; use id_kind='internal' for index IDs")
                key = document_id
            else:
                if self._ids is None:
                    raise ValueError("Open the store with matching index_dir to retrieve internal IDs")
                if (
                    not isinstance(document_id, (int, np.integer)) or isinstance(document_id, (bool, np.bool_))
                    or not 0 <= int(document_id) < len(self._ids)
                ):
                    raise ValueError(f"Internal document ID is out of range: {document_id!r}")
                key = int(document_id)
            if key not in seen:
                seen.add(key)
                normalized.append(key)
                external_ids.append(key if id_kind == "external" else self._ids[key])
        for start in range(0, len(normalized), 500):
            batch = external_ids[start:start + 500]
            columns = "id, metadata" if metadata_only else "id, metadata, content"
            placeholders = ",".join("?" for _ in batch)
            found = {
                row[0]: row[1:]
                for row in self.connection.execute(
                    f"SELECT {columns} FROM documents WHERE id IN ({placeholders})", batch
                )
            }
            missing = [key for key in batch if key not in found]
            if missing:
                raise KeyError(f"Document IDs not found: {missing[:20]!r} ({len(missing)} missing in this batch)")
            for offset, external in enumerate(batch):
                values = found[external]
                record = orjson.loads(self._decompressor.decompress(values[0]))
                if not metadata_only:
                    record.update(orjson.loads(self._decompressor.decompress(values[1])))
                yield normalized[start + offset], record

    def get_many(
        self, document_ids: Iterable[str | int | np.integer], *,
        id_kind: str = "external", metadata_only: bool = False,
    ) -> dict[str | int, dict]:
        return dict(self.iter_many(document_ids, id_kind=id_kind, metadata_only=metadata_only))


def build_document_store(
    corpus_dir: str | Path, output_file: str | Path, *,
    index_dir: str | Path | None = None, batch_records: int = 2000,
) -> Path:
    """One corpus pass, resumable at transaction boundaries; publish on completion.

    Resuming an interrupted compressed shard replays its prefix to the committed
    line, but completed shards are not scanned again. Full records, including
    ingestion-filtered records, are retained; duplicate raw IDs are errors.
    """
    if batch_records < 1:
        raise ValueError("batch_records must be positive")
    corpus = Path(corpus_dir).resolve()
    output = Path(output_file).resolve()
    files = _corpus_files(corpus)
    index_source = _source_identity(Path(index_dir).resolve()) if index_dir is not None else None
    if index_dir is not None:
        manifest = Path(index_dir).resolve().parent / "build.json"
        if manifest.exists() and json.loads(manifest.read_text())["corpus_files"] != files:
            raise ValueError("Raw corpus shards do not match the index build manifest")
        if Path(index_dir).resolve() in output.parents:
            raise ValueError("Document store must be outside the fingerprint index")
    contract = {
        "format": _FORMAT, "source_files": files, "index_source": index_source,
        "compression": "per-record zstandard, metadata/content separated",
        "metadata_excludes": list(_CONTENT_FIELDS), "preserves_all_raw_fields": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        if output.exists():
            with CorpusDocumentStore(output, index_dir=index_dir) as existing:
                for key, value in contract.items():
                    if existing.meta[key] != value:
                        raise ValueError("Existing document store has different source/settings")
            return output
        partial = output.with_suffix(output.suffix + ".building")
        fresh = not partial.exists()
        connection = sqlite3.connect(partial)
        started = time.perf_counter()
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA cache_size=-65536")
            if fresh:
                with connection:
                    connection.execute("CREATE TABLE info(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    connection.execute("CREATE TABLE shards(path TEXT PRIMARY KEY, line INTEGER NOT NULL, records INTEGER NOT NULL, complete INTEGER NOT NULL)")
                    connection.execute(
                        "CREATE TABLE documents(id TEXT NOT NULL UNIQUE, metadata BLOB NOT NULL, content BLOB NOT NULL, shard TEXT NOT NULL, line INTEGER NOT NULL)"
                    )
                    connection.execute("INSERT INTO info VALUES('contract',?)", (json.dumps({**contract, "complete": False}),))
            else:
                row = connection.execute("SELECT value FROM info WHERE key='contract'").fetchone()
                if row is None:
                    raise ValueError("Partial document store has no source/settings contract")
                saved = json.loads(row[0])
                if any(saved[key] != value for key, value in contract.items()):
                    raise ValueError("Partial document store source/settings changed")
            compressor = zstd.ZstdCompressor(level=3)
            for item in files:
                source = Path(item["path"])
                progress = connection.execute("SELECT line,records,complete FROM shards WHERE path=?", (str(source),)).fetchone()
                last_line, records, complete = progress if progress else (0, 0, 0)
                if complete:
                    print(f"[document store] reusing {source.name}: {records:,} records", flush=True)
                    continue
                batch = []
                line_number = 0

                def commit(line: int, finished: bool = False):
                    nonlocal records, batch
                    with connection:
                        try:
                            connection.executemany("INSERT INTO documents VALUES(?,?,?,?,?)", batch)
                        except sqlite3.IntegrityError as error:
                            raise ValueError(f"Duplicate/invalid raw ID near {source.name}:{line}; no record overwritten") from error
                        records += len(batch)
                        connection.execute(
                            "INSERT OR REPLACE INTO shards VALUES(?,?,?,?)",
                            (str(source), line, records, int(finished)),
                        )
                    batch = []

                with source.open("rb") as compressed, zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                    for line_number, line in enumerate(io.TextIOWrapper(reader, encoding="utf-8"), start=1):
                        if line_number <= last_line or not line.strip():
                            continue
                        try:
                            record = orjson.loads(line)
                        except orjson.JSONDecodeError as error:
                            raise ValueError(f"Invalid JSON in {source.name}:{line_number}") from error
                        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"] or not isinstance(record.get("text"), str):
                            raise ValueError(f"Expected raw string id/text in {source.name}:{line_number}")
                        external = record["id"]
                        content = {key: record.pop(key) for key in _CONTENT_FIELDS if key in record}
                        batch.append((
                            external, compressor.compress(orjson.dumps(record)),
                            compressor.compress(orjson.dumps(content)), str(source), line_number,
                        ))
                        if len(batch) >= batch_records:
                            commit(line_number)
                            if records % 100000 == 0:
                                print(f"[document store] {source.name}: {records:,} records", flush=True)
                commit(line_number, finished=True)
                print(f"[document store] completed {source.name}: {records:,} records", flush=True)
            if _corpus_files(corpus) != files or (
                index_dir is not None and _source_identity(Path(index_dir).resolve()) != index_source
            ):
                raise ValueError("Corpus/index changed during document-store construction")
            count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            checked = 0
            if index_dir is not None:
                with ExitStack() as lifetime:
                    offsets = np.load(Path(index_dir) / "doc_name_offsets.npy", mmap_mode="r")
                    names = np.load(Path(index_dir) / "doc_name_bytes.npy", mmap_mode="r")
                    _close_on_exit(lifetime, offsets)
                    _close_on_exit(lifetime, names)
                    mapping = _DocIdMap(offsets, names)
                    for first in range(0, len(mapping), 500):
                        ids = [mapping[i] for i in range(first, min(first + 500, len(mapping)))]
                        found = connection.execute(
                            f"SELECT count(*) FROM documents WHERE id IN ({','.join('?' for _ in ids)})", ids,
                        ).fetchone()[0]
                        if found != len(ids):
                            raise ValueError("An indexed document is absent from the raw document store")
                        checked += found
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite document-store integrity check failed")
            if _corpus_files(corpus) != files or (
                index_dir is not None and _source_identity(Path(index_dir).resolve()) != index_source
            ):
                raise ValueError("Corpus/index changed before document-store publication")
            with connection:
                connection.execute("UPDATE info SET value=? WHERE key='contract'", (json.dumps({
                    **contract, "complete": True, "records": count,
                    "indexed_document_ids_checked": checked, "build_seconds": time.perf_counter() - started,
                }),))
        finally:
            connection.close()
        partial.rename(output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        print(f"[document store] published {count:,} records: {output}", flush=True)
    return output
