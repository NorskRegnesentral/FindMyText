"""Build the HPLT doc_id -> {url, ts} side-car map and propose live samples.

Reads the local indexed-samples file (the only place HPLT urls live) and writes
a compact gzip-JSON map used by the web app to turn a matched content-hash into
a clickable source URL (plus a web.archive.org fallback, since it is a
2015-2017 crawl and many originals are dead).

Also verifies a handful of varied English candidates and prints the live ones so
they can be curated into the app config as samples.

Run:  ./.venv/bin/python web/tools/build_hplt.py
"""

from __future__ import annotations

import gzip
import json
import os
import re
import urllib.parse
import urllib.request

SRC = "/home/jullum/copyai_local/hplt/indexed_samples_hplt.jsonl"
OUT = "/home/jullum/copyai_local/hplt/urls.json.gz"


def wayback(url: str, ts: str) -> str:
    compact = re.sub(r"[^0-9]", "", ts or "")[:14]
    return f"https://web.archive.org/web/{compact}/{url}"


def url_alive(url: str, timeout: float = 6.0) -> bool:
    try:
        req = urllib.request.Request(
            url, method="GET",
            headers={"User-Agent": "Mozilla/5.0 (FindMyTextDemo link check)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def wayback_snapshot(url: str, ts: str, timeout: float = 12.0) -> str | None:
    """Return a Wayback URL if the page is archived, else None (via CDX API)."""
    compact = re.sub(r"[^0-9]", "", ts or "")[:14]
    cdx = "https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(
        {"url": url, "output": "json", "limit": "1", "filter": "statuscode:200"}
    )
    try:
        req = urllib.request.Request(
            cdx, headers={"User-Agent": "Mozilla/5.0 (FindMyTextDemo link check)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rows = json.loads(resp.read().decode())
        if len(rows) > 1:  # row[0] is the header
            return f"https://web.archive.org/web/{compact or rows[1][1]}/{url}"
    except Exception:
        return None
    return None


def excerpt(text: str, target: int = 1400) -> str:
    """A clean, paste-friendly excerpt: whole sentences up to ~target chars."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= target:
        return text
    cut = text[:target]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1].strip() if end > target // 2 else cut.strip()


def main() -> None:
    url_map: dict[str, dict] = {}
    candidates: list[dict] = []

    with open(SRC, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            doc_id = rec.get("id")
            u = rec.get("u")
            ts = rec.get("ts", "")
            if not doc_id or not u:
                continue
            url_map[doc_id] = {"u": u, "ts": ts}

            # collect diverse english candidates for manual curation
            text = rec.get("text") or ""
            langs = rec.get("lang") or []
            probs = rec.get("prob") or []
            eng = bool(langs) and langs[0] == "eng_Latn" and (not probs or probs[0] > 0.95)
            if eng and len(text) >= 1200 and len(candidates) < 400:
                candidates.append({"id": doc_id, "u": u, "ts": ts, "text": text})

    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(url_map, fh, ensure_ascii=False)
    print(f"wrote {OUT}: {len(url_map):,} url entries")

    # prefer candidates from varied domains that have a Wayback snapshot
    seen_domains: set[str] = set()
    live: list[dict] = []
    for c in candidates:
        dom = re.sub(r"^https?://(www\.)?", "", c["u"]).split("/")[0]
        if dom in seen_domains:
            continue
        snap = wayback_snapshot(c["u"], c["ts"])
        if snap:
            seen_domains.add(dom)
            c["archive_url"] = snap
            live.append(c)
        if len(live) >= 12:
            break

    print(f"\n=== {len(live)} candidates WITH Wayback snapshot ===")
    for c in live:
        print(f"\nid={c['id']}\nurl={c['u']}\nts={c['ts']}\narchive={c['archive_url']}\ntext={excerpt(c['text'])}")

    dump = [
        {"id": c["id"], "u": c["u"], "ts": c["ts"],
         "archive_url": c["archive_url"], "text": excerpt(c["text"])}
        for c in live
    ]
    cand_path = os.path.join(os.path.dirname(__file__), "_hplt_candidates.json")
    with open(cand_path, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {cand_path}: {len(dump)} candidates")


if __name__ == "__main__":
    main()
