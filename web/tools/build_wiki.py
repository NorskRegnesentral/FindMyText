"""Build the Wikipedia curid -> title side-car map and grab lead paragraphs.

Downloads the English wikimedia/wikipedia parquet shards one at a time (each is
deleted right after processing, so peak disk use is ~one shard), records the
title for every article whose curid is present in the local index, and writes a
compact gzip-JSON map used by the web app to show article titles next to
matches.

Also captures the lead paragraph for a hand-picked list of broad, varied
articles so they can be curated into the app config as samples.

Run:  ./.venv/bin/python web/tools/build_wiki.py
"""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq

INDEX_DIR = "/home/jullum/copyai_local/wiki/index(4,6)_wikipedia"
OUT = "/home/jullum/copyai_local/wiki/titles.json.gz"
CAND_OUT = os.path.join(os.path.dirname(__file__), "_wiki_candidates.json")
BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
N_SHARDS = 41

# Broad, fun, varied articles to feature as samples (matched case-insensitively
# on exact title, kept only if present in the index).
WANTED_TITLES = {
    t.lower(): t for t in [
        # science & nature
        "Coffee", "Black hole", "Octopus", "Volcano", "Photosynthesis",
        "Honey bee", "Dinosaur", "Coral reef", "Earthquake", "DNA",
        "Antarctica", "Lightning", "Rainbow", "Tsunami", "Glacier",
        "Galaxy", "Comet", "Photosynthesis", "Penguin", "Dolphin",
        "Elephant", "Tiger", "Shark", "Butterfly", "Mushroom",
        # arts & culture
        "Jazz", "Origami", "Guitar", "Ballet", "Photography",
        "Impressionism", "Opera", "Calligraphy", "Graffiti", "Tattoo",
        # history & people
        "Leonardo da Vinci", "Marie Curie", "Cleopatra", "Vikings",
        "Roman Empire", "Great Wall of China", "Pyramid", "Renaissance",
        # geography
        "Mount Everest", "Sahara", "Amazon rainforest", "Great Barrier Reef",
        "Grand Canyon", "Iceland", "Venice", "Kyoto",
        # food & drink
        "Pizza", "Chocolate", "Sushi", "Tea", "Cheese", "Bread",
        "Curry", "Ice cream",
        # games & sport
        "Chess", "Volleyball", "Surfing", "Marathon", "Cycling",
        "Basketball", "Skateboarding",
        # misc / fun
        "Lego", "Clock", "Umbrella", "Lighthouse", "Hot air balloon",
        "Windmill", "Compass", "Kite",
    ]
}


def load_index_ids() -> set[str]:
    off = np.load(os.path.join(INDEX_DIR, "doc_name_offsets.npy"), mmap_mode="r")
    b = np.load(os.path.join(INDEX_DIR, "doc_name_bytes.npy"), mmap_mode="r")
    return {
        bytes(b[off[i]:off[i + 1]]).decode("utf-8", "replace")
        for i in range(len(off) - 1)
    }


def lead_paragraph(text: str, title: str = "", limit: int = 1400) -> str | None:
    """Return a clean lead paragraph, or None for disambiguation-style pages."""
    for para in text.split("\n\n"):
        p = re.sub(r"\s+", " ", para).strip()
        # Drop leading pronunciation/parenthetical artifacts like ") is ...".
        p = re.sub(r"^[)\s,;:]+", "", p)
        p = re.sub(r"^\([^)]*\)\s*", "", p)
        if len(p) < 200:
            continue
        # Skip disambiguation pages: many "(YEAR" refs and no defining verb.
        if len(re.findall(r"\(\d{4}", p)) >= 2 and not re.search(
            r"\b(is|was|are|were|refers)\b", p[:120]
        ):
            return None
        if len(p) > limit:
            cut = p[:limit]
            end = cut.rfind(". ")
            p = cut[: end + 1] if end > limit // 2 else cut
        return p.strip()
    return None


def download(idx: int, dest: str, retries: int = 4) -> None:
    url = f"{BASE}/train-{idx:05d}-of-{N_SHARDS:05d}.parquet"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FindMyTextDemo/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  shard {idx} attempt {attempt+1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download shard {idx}")


def main() -> None:
    ids = load_index_ids()
    print(f"index ids: {len(ids):,}")

    titles: dict[str, str] = {}
    candidates: dict[str, dict] = {}
    t0 = time.time()

    for shard in range(N_SHARDS):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp = tf.name
        try:
            download(shard, tmp)
            table = pq.read_table(tmp, columns=["id", "url", "title", "text"])
            col_id = table.column("id").to_pylist()
            col_title = table.column("title").to_pylist()
            col_url = table.column("url").to_pylist()
            col_text = table.column("text").to_pylist()
            for cid, title, url, text in zip(col_id, col_title, col_url, col_text):
                cid = str(cid)
                if cid in ids:
                    titles[cid] = title
                    key = (title or "").lower()
                    if key in WANTED_TITLES and key not in candidates:
                        lead = lead_paragraph(text or "", title or "")
                        if lead:
                            candidates[key] = {
                                "curid": cid, "title": title, "url": url,
                                "lead": lead,
                            }
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        print(f"shard {shard:02d}/{N_SHARDS}: titles={len(titles):,} "
              f"samples={len(candidates)} elapsed={time.time()-t0:.0f}s")

    with gzip.open(OUT, "wt", encoding="utf-8") as out:
        json.dump(titles, out, ensure_ascii=False)
    print(f"wrote {OUT}: {len(titles):,} titles")

    with open(CAND_OUT, "w", encoding="utf-8") as out:
        json.dump(list(candidates.values()), out, ensure_ascii=False, indent=2)
    print(f"wrote {CAND_OUT}: {len(candidates)} sample candidates")


if __name__ == "__main__":
    main()
