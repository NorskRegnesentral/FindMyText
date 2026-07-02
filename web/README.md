# findmytext.nr.no — corpus inclusion web demo

A small, modern Flask website that lets anyone check whether a piece of text (or
a substantial part of it) appears in one of our document corpora, using the
winnowing-fingerprint detection methods from this repository.

It is intentionally lightweight: vanilla HTML/CSS/JS on the front end (no build
step, no framework) and a thin Flask layer that **reuses the project's own
detection code** rather than re-implementing anything.


---

## What it does

1. **Pick a corpus** — corpus indexes live on the server and are loaded on demand.
2. **Paste/type text** — choose from a dropdown of curated example texts (each
   with a link to its original source) or paste your own, with a character counter.
3. **Pick algorithm(s)** and run:
   - **Position-aware match (our method)** — the clustering score (largest
     geometrically-coherent cluster of shared fingerprints), computed by the
     FindMyText core via
     [`detector.TextContainmentDetector.find_matches_clustering`](../detector.py).
   - **Shared fingerprints (Jaccard baseline)** — number of shared winnowed
     fingerprints, via
     [`detector.TextContainmentDetector.find_matches_jaccard`](../detector.py).
4. **See results** — a summary score + best match per algorithm, a short ranking,
   and your text with the **matching passages highlighted**. Where possible each
   match shows the document's **title as a clickable link** (resolved from
   optional side-car maps; see below).

While a check runs, the server streams progress events (newline-delimited JSON)
so the page shows a **live progress bar, stage label and elapsed time**.

---

## Project layout

```
web/
├── wsgi.py                 # entry point (dev server + gunicorn/mod_wsgi target)
├── requirements.txt        # web-only deps (Flask, gunicorn) + core requirements
├── config.example.json     # copy + edit for a real deployment
├── run_local.sh            # convenience launcher using the repo venv
└── corpussearch/
    ├── __init__.py         # create_app(): Flask app + routes
    ├── config.py           # defaults + JSON override loading
    ├── detection.py        # on-demand detector loading, streaming, highlight spans
    ├── titles.py            # lazy resolver for title/url side-car maps + Wayback links
    ├── samples_data.json    # curated per-corpus example texts (+ source links)
    ├── protection.py        # pluggable abuse protection (all OFF by default)
    ├── templates/index.html
    └── static/{css,js}/
```

Optional helper scripts under [`web/tools/`](tools/) build the side-car
title/url maps for a corpus (`build_arxiv.py`, `build_hplt.py`, `build_wiki.py`).

All scoring and tokenisation live in the FindMyText core
([`detector.py`](../detector.py), [`winnower.py`](../winnower.py),
[`indexing.py`](../indexing.py)); the web layer only does routing, progress
streaming, corpus/LRU config and turning the core's highlight positions into
character spans.

---

## Run locally

The detection code reuses the repo's modules, so use the repo's virtualenv.

```bash
# from the repo root, one-time: install the web deps (this also pulls in the
# core requirements: numpy, numba, polars, scipy, ...)
.venv/bin/python -m pip install -r web/requirements.txt

# start the dev server
cd web
./run_local.sh            # or: ../.venv/bin/python wsgi.py
# open http://127.0.0.1:5000
```

The three corpora — **arXiv**, **English Wikipedia** and the **HPLT** web crawl —
are offered automatically whenever their prebuilt indexes are found under the
index root (default `/home/jullum/copyai_local`). At least one index must be
present for the site to have anything to search. Override the location with:

```bash
export FINDMYTEXT_INDEX_ROOT=/path/to/indexes
# expects: $ROOT/arxiv/index(4,6), $ROOT/wiki/index(4,6)_wikipedia, $ROOT/hplt/index(4,6)
```

Each corpus ships with two one-click sample texts: one that is known to match
the corpus and one unrelated passage that is known **not** to match.

---

## Configure corpora & links

For a real deployment, copy `config.example.json`, edit it, and point the app at
it via an environment variable — no code changes needed:

```bash
export FINDMYTEXT_CONFIG=/srv/findmytext/config.json
```

Each corpus entry needs a prebuilt `DiskBasedIndex` directory (the folder that
contains `meta.json`, `fingerprints.npy`, `postings.dat`, …). Build one with
[`index_builder.py`](../index_builder.py). Set `doc_url_template`
to turn a matched document id into a clickable link (e.g.
`https://arxiv.org/abs/{doc_id}` for arXiv), or leave it empty for ids that have
no public URL. Give each corpus its own `samples` list; each sample may carry a
`url` (original source) and an optional `archive_url` (e.g. a Wayback snapshot),
both shown as links under the example picker.

For richer results you can attach optional **side-car maps** (built by the
scripts in [`web/tools/`](tools/)):

- `title_map_path` — resolves a matched doc id to a human-readable **title**
  shown as the link text.
- `url_map_path` — resolves a doc id whose id is not itself a URL (e.g. HPLT
  content hashes) to a source url, with an automatic Wayback fallback.
- `dataset_url` — a link to browse/search the underlying dataset yourself.

Note that some indexes use **internal document ids** rather than a public
identifier (the Wikipedia index ids, for example, are not Wikipedia page ids),
so leave `doc_url_template` empty for those and rely on the maps above.

Also set `github_url` and `paper_url` so the page links to the code and paper.

### Memory note (important on a 16 GB host)

Each loaded disk index uses **~1.7–2 GB RAM** and the first query after loading
reads ~2 GB from disk (≈7 s on local NVMe, much slower over a network FS). The
app therefore loads indexes **on demand** and keeps only `max_loaded_indexes`
(default **1**) resident, evicting the least-recently-used one. Keep indexes on
**local SSD/NVMe**, not a network share, for acceptable first-query latency.

---

## Abuse protection (opt-in)

Nothing is enabled by default. Turn on any combination in the config
`"protection"` block — see [`corpussearch/protection.py`](corpussearch/protection.py):

| Mechanism | How to enable | Notes |
|---|---|---|
| **Shared password** | `"password": "secret"` | A password box appears on the page; the value is checked server-side. Simplest option. |
| **Per-IP rate limit** | `"rate_limits": ["20 per hour", "5 per minute"]` | Requires `pip install Flask-Limiter`. Good default protection. |
| **CAPTCHA** | `"captcha_provider": "hcaptcha"` (or `"recaptcha"`) + `captcha_secret` + `captcha_sitekey` | Widget is injected automatically and verified server-side. |

The single `guard_request()` function is the place to add other ideas later: an
IP allow-list, a per-day quota in a small SQLite/Redis counter, a concurrency
cap/queue, or putting the whole site behind NR SSO / reverse-proxy basic-auth.

**Recommendation for launch:** a per-IP rate limit (Flask-Limiter) is usually
enough and invisible to honest users; add a shared password if you want to keep
the demo semi-private during testing. CAPTCHA only if you see abuse.

---

## Deploy on the server (Apache + gunicorn)

Apache + `mod_wsgi` works, but the simplest and most robust modern setup is
**gunicorn behind Apache as a reverse proxy** — it keeps the heavy, long-lived
Python process (with the big index in RAM) separate from Apache and survives
Apache restarts.

1. Run the app with gunicorn (a couple of workers; long timeout because a cold
   first query can take several seconds, and the progress endpoint streams):

   ```bash
   cd /srv/findmytext/web
   FINDMYTEXT_CONFIG=/srv/findmytext/config.json \
   /srv/findmytext/.venv/bin/gunicorn --workers 2 --threads 4 \
       --timeout 180 --bind 127.0.0.1:8001 wsgi:app
   ```

   Run it as a `systemd` service so it starts on boot and restarts on failure.

2. Apache (`mod_proxy` + `mod_proxy_http`, and `mod_ssl` for HTTPS) as a
   reverse proxy for `findmytext.nr.no`:

   ```apache
   <VirtualHost *:443>
       ServerName findmytext.nr.no
       # ... SSL certificate directives ...
       ProxyPreserveHost On
       ProxyPass        / http://127.0.0.1:8001/
       ProxyPassReverse / http://127.0.0.1:8001/
   </VirtualHost>
   ```

   Enable the modules (`a2enmod proxy proxy_http ssl`) and reload Apache.

> **Why not `mod_wsgi`?** It embeds Python inside Apache, which makes the large
> in-RAM index awkward to manage (per-worker memory, reloads tied to Apache) and
> can buffer the streaming progress responses. If you prefer `mod_wsgi` anyway,
> point its `WSGIScriptAlias` at `web/wsgi.py` (it exposes `app`) and set
> `WSGIDaemonProcess` with 1 process so the index is loaded once.

### Worker/concurrency note

A running detection holds its worker for the duration of the (streamed) request.
With `--workers 2 --threads 4` a couple of users can run checks concurrently;
combine this with the rate limit above so a single visitor can't tie up the box.

---

## How the visualization works

The matched-passage highlighting maps shared-fingerprint **token positions** back
to character offsets in your original text using the winnower's exact
tokenisation (see `tokenize_with_offsets` in [`winnower.py`](../winnower.py)) and
the query positions returned by
`TextContainmentDetector.get_match_highlight_positions`. For "our method" the
highlighted positions are those in the largest cluster; for Jaccard they are all
shared fingerprints of the top match. No corpus document text is needed for this,
which keeps the server lightweight.
