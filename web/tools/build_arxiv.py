"""Build the arXiv doc_id -> title side-car map and propose topical samples.

Streams the common-pile/arxiv_papers gzipped JSONL shards over HTTP (no full
local copy kept), extracts the title (first ``# `` markdown heading) for every
paper whose id is present in the local index, and writes a compact gzip-JSON
map used by the web app to show a human title next to each match.

Also collects candidate samples per topic (with their abstracts) so a human can
curate them into the app config.

Run:  ./.venv/bin/python web/tools/build_arxiv.py
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import time
import urllib.request

import numpy as np

INDEX_ROOT = os.environ.get("FINDMYTEXT_INDEX_ROOT", "/path/to/prebuilt-indexes")
INDEX_DIR = os.environ.get(
    "FINDMYTEXT_ARXIV_INDEX_DIR",
    os.path.join(INDEX_ROOT, "arxiv", "index(4,6)"),
)
OUT = os.environ.get(
    "FINDMYTEXT_ARXIV_TITLES_OUT",
    os.path.join(INDEX_ROOT, "arxiv", "titles.json.gz"),
)
CAND_OUT = os.path.join(os.path.dirname(__file__), "_arxiv_candidates.json")
BASE = "https://huggingface.co/datasets/common-pile/arxiv_papers/resolve/main"
N_SHARDS = 22

# arXiv ids we always want captured as candidates (requested + to check).
FORCE_IDS = {"2004.14723"}

TOPIC_KEYWORDS = {
    "LLM": ["large language model", "language model", "instruction tuning",
            "in-context learning", "chain-of-thought", "transformer"],
    "XAI": ["explainable", "interpretab", "explanation", "saliency",
            "feature importance", "shapley", "counterfactual explanation"],
    "math": ["theorem", "manifold", "cohomology", "algebraic geometry",
             "conjecture", "homomorphism", "prime numbers", "lie algebra"],
    "physics": ["quantum", "black hole", "gauge theory", "cosmolog",
                "gravitational wave", "relativity", "dark matter", "particle"],
}


def load_index_ids() -> set[str]:
    off = np.load(os.path.join(INDEX_DIR, "doc_name_offsets.npy"), mmap_mode="r")
    b = np.load(os.path.join(INDEX_DIR, "doc_name_bytes.npy"), mmap_mode="r")
    return {
        bytes(b[off[i]:off[i + 1]]).decode("utf-8", "replace")
        for i in range(len(off) - 1)
    }


def extract_title(text: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(.+)", line.strip())
        if m:
            # Titles are frequently split across lines with a markdown hard
            # break; the block ends at the first blank line (before the author
            # list) or a following heading.
            parts = [m.group(1)]
            for cont in lines[i + 1 :]:
                s = cont.strip()
                if not s or s.startswith("#"):
                    break
                parts.append(s)
            return re.sub(r"\s+", " ", " ".join(parts)).strip()
    return None


def extract_abstract(text: str, limit: int = 1600) -> str | None:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^#+\s*abstract\b", line.strip(), re.I):
            start = i + 1
            break
    if start is None:
        return None
    out: list[str] = []
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("#"):
            break
        if s:
            out.append(s)
        elif out:
            break
    abstract = re.sub(r"\s+", " ", " ".join(out)).strip()
    return abstract[:limit] if abstract else None


def open_shard(idx: int, retries: int = 4):
    url = f"{BASE}/{idx:05d}_arxiv-papers.jsonl.gz"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "FindMyTextDemo/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=60)
            return gzip.GzipFile(fileobj=io.BufferedReader(resp))
        except Exception as exc:  # noqa: BLE001
            print(f"  shard {idx} attempt {attempt+1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not open shard {idx}")


def main() -> None:
    ids = load_index_ids()
    print(f"index ids: {len(ids):,}")

    titles: dict[str, str] = {}
    candidates: dict[str, list[dict]] = {t: [] for t in TOPIC_KEYWORDS}
    forced: list[dict] = []
    t0 = time.time()

    for shard in range(N_SHARDS):
        fh = open_shard(shard)
        seen = 0
        for raw in fh:
            seen += 1
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            aid = rec.get("id")
            if not aid or aid not in ids:
                continue
            text = rec.get("text") or ""
            title = extract_title(text)
            if title:
                titles[aid] = title
            # candidate collection
            low = (title or "").lower()
            if aid in FORCE_IDS:
                forced.append({"id": aid, "title": title,
                               "abstract": extract_abstract(text)})
            for topic, kws in TOPIC_KEYWORDS.items():
                if len(candidates[topic]) >= 12:
                    continue
                if any(k in low for k in kws):
                    ab = extract_abstract(text)
                    if ab and len(ab) > 300:
                        candidates[topic].append(
                            {"id": aid, "title": title, "abstract": ab})
        print(f"shard {shard:02d}: rows={seen:,} titles={len(titles):,} "
              f"elapsed={time.time()-t0:.0f}s")

    with gzip.open(OUT, "wt", encoding="utf-8") as out:
        json.dump(titles, out, ensure_ascii=False)
    print(f"wrote {OUT}: {len(titles):,} titles")

    with open(CAND_OUT, "w", encoding="utf-8") as out:
        json.dump({"forced": forced, "topics": candidates}, out,
                  ensure_ascii=False, indent=2)
    print(f"wrote {CAND_OUT}")


if __name__ == "__main__":
    main()
