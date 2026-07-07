"""Build the HPLT ``doc_id -> {url, ts}`` side-car map.

The web app turns a matched content-hash (the index' external doc id) into a
clickable source URL (plus a web.archive.org fallback, since HPLT is a
2015-2017 crawl and many originals are dead).

The index doc ids are exactly the ``id`` field of the HPLT records that passed
the indexing filter (see ``utils.json_line_reader``: text length in
[2000, 100000] and no ``sex``/``porn`` token). We reproduce that filter while
streaming the raw ``*.jsonl.zst`` shards and record ``id -> (url, ts)``.

Because the corpus has ~50M documents, the map is stored as **SQLite** (queried
on disk by the web app) rather than an in-memory gzip-JSON dict. Shards are
processed in parallel (one worker per shard) into per-shard SQLite files, then
merged and indexed.

Run (full build, ~hours):   ./.venv/bin/python web/tools/build_hplt.py
Quick sample DB for testing: ./.venv/bin/python web/tools/build_hplt.py --samples
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sqlite3
import sys
import time
from multiprocessing import Pool
from urllib.parse import urlsplit

import orjson
import zstandard as zstd

# --- Paths -----------------------------------------------------------------
INDEX_ROOT = os.environ.get("FINDMYTEXT_INDEX_ROOT", "/path/to/prebuilt-indexes")
SHARD_GLOB = os.environ.get(
    "FINDMYTEXT_HPLT_SHARD_GLOB",
    "/path/to/hplt/*.jsonl.zst",
)
SAMPLES_SRC = os.environ.get(
    "FINDMYTEXT_HPLT_SAMPLES_SRC",
    os.path.join(INDEX_ROOT, "hplt", "indexed_samples_hplt.jsonl"),
)
OUT_DIR = os.environ.get("FINDMYTEXT_HPLT_OUT_DIR", os.path.join(INDEX_ROOT, "hplt"))
FINAL_DB = os.path.join(OUT_DIR, "urls.sqlite")
SHARD_DIR = os.path.join(OUT_DIR, "_url_shards")

# --- Indexing filter (must match utils.json_line_reader) -------------------
MIN_LEN = 2000
MAX_LEN = 100000
BAD_WORDS = {"sex", "porn"}
BATCH = 100_000


def _new_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE urls (id TEXT, u TEXT, ts TEXT)")
    return conn


def build_shard(shard_path: str) -> tuple[str, int]:
    """Stream one zst shard into its own SQLite file. Skips if already done."""
    name = os.path.basename(shard_path).replace(".jsonl.zst", "")
    out = os.path.join(SHARD_DIR, f"{name}.sqlite")
    part = out + ".part"
    if os.path.exists(out):
        return (name, -1)  # already built

    os.makedirs(SHARD_DIR, exist_ok=True)
    if os.path.exists(part):
        os.remove(part)
    conn = _new_db(part)
    cur = conn.cursor()

    t0 = time.time()
    n = 0
    batch: list[tuple[str, str, str]] = []
    dctx = zstd.ZstdDecompressor()
    with open(shard_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_stream:
            if not line.strip():
                continue
            try:
                d = orjson.loads(line)
            except Exception:  # noqa: BLE001
                continue
            text = d.get("text") or ""
            tlen = len(text)
            if tlen < MIN_LEN or tlen > MAX_LEN:
                continue
            if any(w in BAD_WORDS for w in text.split()):
                continue
            doc_id = d.get("id")
            u = d.get("u")
            if not doc_id or not u:
                continue
            batch.append((doc_id, u, d.get("ts") or ""))
            if len(batch) >= BATCH:
                cur.executemany("INSERT INTO urls VALUES (?, ?, ?)", batch)
                n += len(batch)
                batch.clear()
                print(f"[{name}] {n:,} rows  ({n/(time.time()-t0):,.0f}/s)", flush=True)
    if batch:
        cur.executemany("INSERT INTO urls VALUES (?, ?, ?)", batch)
        n += len(batch)
    conn.commit()
    conn.close()
    os.replace(part, out)
    print(f"[{name}] DONE {n:,} rows in {time.time()-t0:,.0f}s -> {out}", flush=True)
    return (name, n)


def merge(shard_dbs: list[str]) -> None:
    """Merge per-shard SQLite files into FINAL_DB and build the id index."""
    tmp = FINAL_DB + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = _new_db(tmp)
    total = 0
    for db in shard_dbs:
        if not os.path.exists(db):
            print(f"WARNING missing shard db {db}", flush=True)
            continue
        conn.execute("ATTACH DATABASE ? AS s", (db,))
        conn.execute("INSERT INTO urls SELECT id, u, ts FROM s.urls")
        conn.commit()
        conn.execute("DETACH DATABASE s")
        total = conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
        print(f"merged {os.path.basename(db)} -> total {total:,}", flush=True)
    print(f"building index on {total:,} rows ...", flush=True)
    conn.execute("CREATE INDEX idx_urls_id ON urls (id)")
    conn.commit()
    conn.close()
    os.replace(tmp, FINAL_DB)
    print(f"wrote {FINAL_DB}: {total:,} rows", flush=True)


def build_full() -> None:
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        sys.exit(f"no shards match {SHARD_GLOB}")
    print(f"{len(shards)} shards; processing in parallel ...", flush=True)
    os.makedirs(SHARD_DIR, exist_ok=True)
    with Pool(processes=len(shards)) as pool:
        results = pool.map(build_shard, shards)
    for name, n in results:
        print(f"  {name}: {'(cached)' if n < 0 else f'{n:,} rows'}", flush=True)
    shard_dbs = [
        os.path.join(SHARD_DIR, os.path.basename(s).replace(".jsonl.zst", ".sqlite"))
        for s in shards
    ]
    merge(shard_dbs)


def build_samples() -> None:
    """Small SQLite from the indexed-samples file, for quick end-to-end testing."""
    tmp = FINAL_DB + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = _new_db(tmp)
    cur = conn.cursor()
    n = 0
    batch: list[tuple[str, str, str]] = []
    with open(SAMPLES_SRC, "r", encoding="utf-8") as fh:
        for line in fh:
            d = orjson.loads(line)
            doc_id, u = d.get("id"), d.get("u")
            if not doc_id or not u:
                continue
            batch.append((doc_id, u, d.get("ts") or ""))
            if len(batch) >= BATCH:
                cur.executemany("INSERT INTO urls VALUES (?, ?, ?)", batch)
                n += len(batch)
                batch.clear()
        if batch:
            cur.executemany("INSERT INTO urls VALUES (?, ?, ?)", batch)
            n += len(batch)
    conn.execute("CREATE INDEX idx_urls_id ON urls (id)")
    conn.commit()
    conn.close()
    os.replace(tmp, FINAL_DB)
    print(f"wrote {FINAL_DB}: {n:,} rows (from samples)")


def _host(url: str) -> str:
    """Bare hostname for search (drops scheme, port and a leading ``www.``)."""
    try:
        host = urlsplit(url).hostname or ""
    except Exception:  # noqa: BLE001
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def build_fts(db_path: str = FINAL_DB) -> None:
    """Add an FTS5 ``url_fts(host, url, id)`` table to an existing url map.

    Enables the in-app domain search for HPLT. ``host`` is the only indexed
    column; ``url``/``id`` are stored (UNINDEXED) for retrieval. Rebuilt from
    scratch each run, paging over ``urls`` by rowid so a single connection both
    reads and writes safely.
    """
    if not os.path.exists(db_path):
        sys.exit(f"no url map at {db_path}; build it first")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-1000000")  # ~1GB page cache
    conn.execute("DROP TABLE IF EXISTS url_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE url_fts USING fts5("
        "host, url UNINDEXED, id UNINDEXED, tokenize='unicode61')"
    )
    total = conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
    print(f"indexing {total:,} rows into url_fts ...", flush=True)

    ins = conn.cursor()
    read = conn.cursor()
    t0 = time.time()
    n = 0
    last_rowid = 0
    while True:
        rows = read.execute(
            "SELECT rowid, id, u FROM urls WHERE rowid > ? "
            "ORDER BY rowid LIMIT ?",
            (last_rowid, BATCH),
        ).fetchall()
        if not rows:
            break
        last_rowid = rows[-1][0]
        ins.executemany(
            "INSERT INTO url_fts (host, url, id) VALUES (?, ?, ?)",
            [(_host(u), u, doc_id) for _rid, doc_id, u in rows],
        )
        n += len(rows)
        if n % (BATCH * 10) == 0:
            print(f"  {n:,}/{total:,}  ({n/(time.time()-t0):,.0f}/s)", flush=True)
    conn.commit()
    print("optimizing FTS index ...", flush=True)
    conn.execute("INSERT INTO url_fts(url_fts) VALUES('optimize')")
    conn.commit()
    conn.close()
    print(f"done: {n:,} rows indexed in {time.time()-t0:,.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true",
                    help="build a small DB from indexed_samples for testing")
    ap.add_argument("--fts", action="store_true",
                    help="build the FTS5 domain-search index on an existing map")
    args = ap.parse_args()
    if args.fts:
        build_fts()
    elif args.samples:
        build_samples()
    else:
        build_full()
