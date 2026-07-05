# findmytext.nr.no — corpus inclusion web demo

A small Flask website that lets anyone check whether a piece of text (or a
substantial part of it) appears in one of our document corpora, using the
winnowing-fingerprint detection methods from this repository.

It is intentionally lightweight: vanilla HTML/CSS/JS on the front end (no build
step) and a thin Flask layer that **reuses the project's own detection code**.

---

## What it does

1. **Pick a corpus** — indexes live on the server and are loaded on demand.
2. **Paste/type text** — choose a curated example (each linked to its source) or
   paste your own.
3. **Pick algorithm(s)** and run:
   - **Position-aware match (our method)** — the clustering score (largest
     geometrically-coherent cluster of shared fingerprints), via
     [`detector.TextContainmentDetector.find_matches_clustering`](../detector.py).
   - **Shared fingerprints (baseline)** — number of shared winnowed fingerprints,
     via [`detector.TextContainmentDetector.find_matches_jaccard`](../detector.py).
4. **See results** — a score + best match per algorithm, a short ranking, and your
   text with the **matching passages highlighted**. Where possible each match
   shows the document's **title as a clickable link**.

While a check runs, the server streams progress events (newline-delimited JSON),
so the page shows a live progress bar, stage label and elapsed time.

---

## Project layout

```
web/
├── wsgi.py                 # entry point (dev server + gunicorn target)
├── requirements.txt        # web-only deps (Flask, gunicorn) + core requirements
├── run_local.sh            # convenience launcher using the repo venv
└── corpussearch/
    ├── __init__.py         # create_app(): Flask app + routes
    ├── config.json         # THE config: paths, params, protection, corpora
    ├── samples_data.json   # curated per-corpus example texts (+ source links)
    ├── config.py           # loads the two JSON files into typed dataclasses
    ├── detection.py        # on-demand detector loading, streaming, highlight spans
    ├── titles.py           # lazy resolver for title/url side-car maps + Wayback links
    ├── protection.py       # pluggable abuse protection (all OFF by default)
    ├── templates/index.html
    └── static/{css,js}/
```

Optional helper scripts under [`web/tools/`](tools/) build the side-car title/url
maps for a corpus (`build_arxiv.py`, `build_hplt.py`, `build_wiki.py`).

All scoring and tokenisation live in the FindMyText core; the web layer only does
routing, progress streaming, corpus/LRU config and turning the core's highlight
positions into character spans.

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
are offered automatically whenever their prebuilt indexes are found. Their
location is set by `index_root` in
[`corpussearch/config.json`](corpussearch/config.json); each corpus's `index_dir`
is resolved relative to it. At least one index must be present.

---

## Configure corpora & links

Everything is driven by two files in `corpussearch/`:

- [`config.json`](corpussearch/config.json) — paths (`index_root`, per-corpus
  `index_dir`, `title_map`, `url_map`), the detection `params`, abuse
  `protection`, and per-corpus metadata (`label`, `description`,
  `doc_url_template`, `dataset_url`). Also set `github_url` and `paper_url`.
- [`samples_data.json`](corpussearch/samples_data.json) — example texts keyed by
  corpus id, plus a shared `no_match` example. Each sample may carry a `url` and
  optional `archive_url` (e.g. a Wayback snapshot).

To relocate the config for a deployment, point the app at your own copy:

```bash
export FINDMYTEXT_CONFIG=/srv/findmytext/config.json
```

Each corpus needs a prebuilt `DiskBasedIndex` directory (`meta.json`,
`fingerprints.npy`, `postings.dat`, …), built with
[`index_builder.py`](../index_builder.py). Set `doc_url_template` to turn a
matched doc id into a link (e.g. `https://arxiv.org/abs/{doc_id}`), or leave it
empty for ids with no public URL (e.g. the Wikipedia index uses internal ids).

For richer results attach optional **side-car maps** (built by the scripts in
[`web/tools/`](tools/); paths relative to `index_root`):

- `title_map` — doc id → human-readable **title** (shown as the link text).
- `url_map` — doc id → source url, with an automatic Wayback fallback.
- `dataset_url` — a link to browse the underlying dataset yourself.

### Memory note

Each loaded disk index uses **~1.7–2 GB RAM** and the first query after loading
reads ~2 GB from disk (≈7 s on local NVMe, much slower over a network FS). The
app loads indexes **on demand** and keeps only `max_loaded_indexes` (default
**1**) resident, evicting the least-recently-used one. Keep indexes on **local
SSD/NVMe** for acceptable first-query latency.

---

## Abuse protection (opt-in)

Nothing is enabled by default. Turn on any combination in the config
`"protection"` block — see [`corpussearch/protection.py`](corpussearch/protection.py):

| Mechanism | How to enable | Notes |
|---|---|---|
| **Shared password** | `"password": "secret"` | A password box appears on the page; the value is checked server-side. Simplest option. |
| **Per-IP rate limit** | `"rate_limits": ["20 per hour", "5 per minute"]` | Requires `pip install Flask-Limiter`. Good default protection. |
| **CAPTCHA** | `"captcha_provider": "hcaptcha"` (or `"recaptcha"`) + `captcha_secret` + `captcha_sitekey` | Widget injected and verified server-side. |

**Recommendation for launch:** a per-IP rate limit is usually enough and
invisible to honest users; add a shared password to keep the demo semi-private.

---

## Deploy on the server (gunicorn + Apache)

Run the app under **gunicorn behind Apache as a reverse proxy**: this keeps the
heavy, long-lived Python process (with the big index in RAM) separate from Apache
and lets it survive Apache restarts.

1. Run gunicorn (a couple of workers; long timeout because a cold first query can
   take several seconds, and the progress endpoint streams). Run it as a
   `systemd` service so it starts on boot and restarts on failure:

   ```bash
   cd /srv/findmytext/web
   FINDMYTEXT_CONFIG=/srv/findmytext/config.json \
   /srv/findmytext/.venv/bin/gunicorn --workers 2 --threads 4 \
       --timeout 180 --bind 127.0.0.1:8001 wsgi:app
   ```

2. Apache (`mod_proxy` + `mod_proxy_http`, `mod_ssl` for HTTPS) as a reverse
   proxy for `findmytext.nr.no`:

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

A running detection holds its worker for the duration of the streamed request.
With `--workers 2 --threads 4` a couple of users can run checks concurrently;
combine this with the rate limit above so one visitor can't tie up the box.

---

## How the visualization works

The matched-passage highlighting maps shared-fingerprint **token positions** back
to character offsets in your text using the winnower's exact tokenisation
(`tokenize_with_offsets` in [`winnower.py`](../winnower.py)) and the query
positions from `TextContainmentDetector.get_match_highlight_positions`. For "our
method" the highlighted positions are those in the largest cluster; for the
baseline they are all shared fingerprints of the top match. No corpus document
text is needed, which keeps the server lightweight.
