"""Build the Wikipedia doc-id -> title/url side-car maps by exact text matching.

The Wikipedia index stores *internal* document ids (0-based row indices into the
source corpus ``data/wiki/wikipedia.jsonl``); they are **not** Wikipedia curids,
so titles cannot be looked up directly. However, the source corpus text is a
byte-for-byte prefix of the ``wikimedia/wikipedia`` 20231101.en dump (our corpus
is that dump, truncated to ~10k chars per article). We can therefore recover the
title/url for every indexed document by an **exact normalised-prefix text join**:

    key = sha1( collapse_whitespace(text)[:PREFIX] )

Both sides (our corpus and the dump) hash to the same key for the same article,
so we get near-100%% coverage with no Wikipedia API calls and no fuzzy matching.

Pipeline:
  1. Download the 41 dump parquet shards (~11.6 GB) to DUMP_DIR (resumable: a
     shard already present with the right size is skipped).
  2. Read the index doc ids and the source corpus; build ``key -> [doc_id,...]``
     for the indexed documents only.
  3. Stream the dump shards; for each row whose key is one we need, record the
     title/url (ambiguous keys hit by >1 distinct title are skipped).
  4. Write ``titles.json.gz`` ({doc_id: title}) and ``urls.json.gz``
     ({doc_id: {"u": url}}) next to the index.

Run (long, unattended):
    nohup ./.venv/bin/python web/tools/build_wiki.py > tmp/_wiki_map.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pyarrow.parquet as pq

# --- paths -----------------------------------------------------------------
SRC_JSONL = (
    "/nr/samba/user/jullum/Local_work/Projects/COPY.AI/"
    "data/wiki/wikipedia.jsonl/wikipedia.jsonl"
)
INDEX_DIR = "/home/jullum/copyai_local/wiki/index(4,6)_wikipedia"
DUMP_DIR = "/home/jullum/copyai_local/wiki/_wikidump_20231101_en"
TITLES_OUT = "/home/jullum/copyai_local/wiki/titles.json.gz"
URLS_OUT = "/home/jullum/copyai_local/wiki/urls.json.gz"

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"
PREFIX = 2000  # chars of normalised text used as the join key (well under ~10k)
UA = {"User-Agent": "FindMyTextDemo/1.0 (research; contact jullum@nr.no)"}

_WS = re.compile(r"\s+")


def norm_key(text: str) -> str:
    """Whitespace-collapsed prefix hashed to a stable join key."""
    prefix = _WS.sub(" ", text).strip()[:PREFIX]
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


def wiki_url(title: str) -> str:
    """Canonical article URL with underscores and readable punctuation."""
    slug = urllib.parse.quote(title.replace(" ", "_"), safe="_(),.:'!*-/&")
    return f"https://en.wikipedia.org/wiki/{slug}"


# --- step 1: download shards (resumable) -----------------------------------
def parquet_files() -> list[dict]:
    u = "https://datasets-server.huggingface.co/parquet?" + urllib.parse.urlencode(
        {"dataset": DATASET}
    )
    d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))
    files = [
        f
        for f in d["parquet_files"]
        if f["config"] == CONFIG and f["split"] == "train"
    ]
    files.sort(key=lambda f: f["url"])
    return files


def download_shard(url: str, dest: str, expected: int, retries: int = 5) -> None:
    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"  skip (present): {os.path.basename(dest)}")
        return
    part = dest + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(part, dest)
            got = os.path.getsize(dest)
            if expected and got != expected:
                raise IOError(f"size mismatch {got} != {expected}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  {os.path.basename(dest)} attempt {attempt+1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def ensure_dump() -> list[str]:
    os.makedirs(DUMP_DIR, exist_ok=True)
    files = parquet_files()
    print(f"dump shards: {len(files)} "
          f"({sum(f['size'] for f in files)/1e9:.2f} GB)")
    paths = []
    for i, f in enumerate(files):
        dest = os.path.join(DUMP_DIR, f"{i:04d}.parquet")
        download_shard(f["url"], dest, f.get("size", 0))
        paths.append(dest)
    return paths


# --- step 2: our side (indexed corpus) -------------------------------------
def load_index_ids() -> set[str]:
    off = np.load(os.path.join(INDEX_DIR, "doc_name_offsets.npy"), mmap_mode="r")
    b = np.load(os.path.join(INDEX_DIR, "doc_name_bytes.npy"), mmap_mode="r")
    return {
        bytes(b[off[i]:off[i + 1]]).decode("utf-8", "replace")
        for i in range(len(off) - 1)
    }


def build_our_keys(ids: set[str]) -> dict[str, list[str]]:
    """Map join-key -> [doc_id, ...] for the indexed documents."""
    key_to_ids: dict[str, list[str]] = {}
    n = 0
    t0 = time.time()
    with open(SRC_JSONL, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            doc_id = str(r["id"])
            if doc_id not in ids:
                continue
            key_to_ids.setdefault(norm_key(r["text"]), []).append(doc_id)
            n += 1
            if n % 50000 == 0:
                print(f"  our keys: {n:,} indexed docs read "
                      f"({time.time()-t0:.0f}s)")
    print(f"our keys: {n:,} indexed docs, {len(key_to_ids):,} distinct keys")
    return key_to_ids


# --- step 3: match against the dump ----------------------------------------
def match_dump(paths: list[str], key_to_ids: dict[str, list[str]]):
    titles: dict[str, str] = {}
    urls: dict[str, dict] = {}
    key_title: dict[str, str] = {}  # key -> title, for collision detection
    ambiguous: set[str] = set()
    t0 = time.time()
    for si, path in enumerate(paths):
        table = pq.read_table(path, columns=["title", "text"])
        col_title = table.column("title").to_pylist()
        col_text = table.column("text").to_pylist()
        for title, text in zip(col_title, col_text):
            if not text:
                continue
            key = norm_key(text)
            doc_ids = key_to_ids.get(key)
            if not doc_ids:
                continue
            if key in ambiguous:
                continue
            prev = key_title.get(key)
            if prev is not None and prev != title:
                # two different dump articles share this key -> unsafe
                ambiguous.add(key)
                for did in doc_ids:
                    titles.pop(did, None)
                    urls.pop(did, None)
                continue
            key_title[key] = title
            url = wiki_url(title)
            for did in doc_ids:
                titles[did] = title
                urls[did] = {"u": url}
        print(f"shard {si:02d}/{len(paths)}: matched docs={len(titles):,} "
              f"ambiguous_keys={len(ambiguous)} elapsed={time.time()-t0:.0f}s")
    return titles, urls, ambiguous


def main() -> None:
    print("=== step 1: download dump ===", flush=True)
    paths = ensure_dump()

    print("=== step 2: build our keys ===", flush=True)
    ids = load_index_ids()
    print(f"index ids: {len(ids):,}")
    key_to_ids = build_our_keys(ids)

    print("=== step 3: match against dump ===", flush=True)
    titles, urls, ambiguous = match_dump(paths, key_to_ids)

    with gzip.open(TITLES_OUT, "wt", encoding="utf-8") as out:
        json.dump(titles, out, ensure_ascii=False)
    with gzip.open(URLS_OUT, "wt", encoding="utf-8") as out:
        json.dump(urls, out, ensure_ascii=False)

    total = len(ids)
    print("=== done ===")
    print(f"matched {len(titles):,} / {total:,} indexed docs "
          f"({len(titles)/total*100:.1f}%)")
    print(f"unmatched: {total-len(titles):,} | ambiguous keys skipped: {len(ambiguous)}")
    print(f"wrote {TITLES_OUT} and {URLS_OUT}")


if __name__ == "__main__":
    main()
