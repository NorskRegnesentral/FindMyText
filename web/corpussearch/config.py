"""Application configuration for the findmytext.nr.no demo.

Two files drive everything, both loaded on startup:

* ``config.json``       — paths, method parameters, abuse protection and the
  per-corpus metadata (labels, links, side-car maps). This is the single place
  to edit for a deployment. Override its location with the ``FINDMYTEXT_CONFIG``
  environment variable.
* ``samples_data.json`` — the curated example texts shown in the UI, keyed by
  corpus id (plus a shared ``no_match`` example). Override with the
  ``samples_file`` key in ``config.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .titles import RESOLVER, wayback_url

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_HERE, "config.json")


@dataclass
class SampleText:
    label: str
    text: str
    # Optional source link so the user can confirm where the sample came from.
    url: Optional[str] = None
    # Optional archived snapshot (e.g. Wayback Machine) shown alongside ``url``
    # when the live page may have changed or gone offline.
    archive_url: Optional[str] = None


@dataclass
class CorpusConfig:
    """A single selectable corpus, backed by a prebuilt disk index."""

    id: str
    label: str
    description: str = ""
    # Path to a prebuilt DiskBasedIndex directory (meta.json, fingerprints.npy,
    # postings.dat, ...). The winnowing parameters are read from the index.
    index_dir: Optional[str] = None
    # Template turning an external doc id into a clickable URL. Use "{doc_id}".
    # If empty, the matched document id is shown as plain text.
    doc_url_template: str = ""
    # Optional side-car maps (built by web/tools/build_*.py) resolving a matched
    # doc id into a human title / a source url. Absolute paths; empty = none.
    title_map_path: str = ""
    url_map_path: str = ""
    # Optional link to browse/search the underlying dataset yourself.
    dataset_url: str = ""
    # Per-corpus example texts (one known to match, one known not to).
    samples: list[SampleText] = field(default_factory=list)
    enabled: bool = True

    def doc_url(self, doc_id: str) -> Optional[str]:
        if not self.doc_url_template:
            return None
        return self.doc_url_template.format(doc_id=doc_id)

    def doc_meta(self, doc_id: str) -> dict:
        """Resolve a matched doc id into ``{title, url, archive_url}``.

        Everything is best-effort: missing maps simply yield ``None`` fields and
        the UI falls back to showing the raw id.
        """
        title = RESOLVER.lookup(self.title_map_path, doc_id)
        url = self.doc_url(doc_id)
        archive_url = None
        if not url and self.url_map_path:
            entry = RESOLVER.lookup(self.url_map_path, doc_id)
            if entry and entry.get("u"):
                url = entry["u"]
                if entry.get("ts"):
                    archive_url = wayback_url(url, entry["ts"])
        return {"title": title, "url": url, "archive_url": archive_url}


@dataclass
class ProtectionConfig:
    """Abuse-protection switches. All OFF by default; flip them on per deployment.

    See ``protection.py`` for what each one does and how to extend it.
    """

    # Simple shared password (sent in the request). Empty string = disabled.
    password: str = ""
    # Per-IP rate limiting. Needs Flask-Limiter installed. Empty list = disabled.
    # Example: ["20 per hour", "5 per minute"].
    rate_limits: list[str] = field(default_factory=list)
    # hCaptcha / reCAPTCHA secret. Empty = disabled (no captcha verification).
    captcha_provider: str = ""  # "hcaptcha" | "recaptcha" | ""
    captcha_secret: str = ""
    captcha_sitekey: str = ""  # public site key, injected into the page


@dataclass
class AlgoParams:
    """Shared retrieval + scoring parameters (kept identical to the research code)."""

    top_k: int = 10
    min_fingerprints: int = 5
    # Position-aware ("our method") clustering parameters.
    cc_doc1_position_threshold: int = 10
    cc_offset_threshold: int = 10
    cc_min_cluster_size: int = 5
    # Normalising constant C (avg winnowed fingerprints per sentence).
    normalizing_constant: float = 8.0


@dataclass
class AppConfig:
    corpora: list[CorpusConfig]
    protection: ProtectionConfig
    params: AlgoParams
    samples: list[SampleText]
    github_url: str = "https://github.com/NorskRegnesentral/FindMyText"
    paper_url: str = ""
    # How many corpus indexes may stay resident in RAM at once (LRU eviction).
    max_loaded_indexes: int = 1
    # Max length of pasted text accepted (characters).
    max_text_chars: int = 50_000

    def corpus(self, corpus_id: str) -> Optional[CorpusConfig]:
        for c in self.corpora:
            if c.id == corpus_id and c.enabled:
                return c
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _resolve_path(path: str, root: str) -> str:
    """Resolve a possibly-relative path against ``root`` (empty stays empty)."""
    if not path:
        return ""
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


def _load_samples(path: str) -> tuple[dict[str, list[SampleText]], Optional[SampleText]]:
    """Load curated examples from ``samples_data.json``.

    Returns ``(samples_by_corpus_id, shared_no_match_sample)``. The examples are
    kept as data (rather than in code) to avoid escaping issues with LaTeX-heavy
    abstracts and to make them trivial to regenerate.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    no_match_raw = raw.pop("no_match", None)
    by_id = {cid: [SampleText(**s) for s in items] for cid, items in raw.items()}
    no_match = SampleText(**no_match_raw) if no_match_raw else None
    return by_id, no_match


def _build_corpus(
    spec: dict[str, Any],
    index_root: str,
    samples_by_id: dict[str, list[SampleText]],
    no_match: Optional[SampleText],
) -> CorpusConfig:
    cid = spec["id"]
    index_dir = _resolve_path(spec.get("index_dir", ""), index_root)
    samples = list(samples_by_id.get(cid, []))
    if no_match is not None:
        samples.append(no_match)
    return CorpusConfig(
        id=cid,
        label=spec.get("label", cid),
        description=spec.get("description", ""),
        index_dir=index_dir,
        doc_url_template=spec.get("doc_url_template", ""),
        title_map_path=_resolve_path(spec.get("title_map", ""), index_root),
        url_map_path=_resolve_path(spec.get("url_map", ""), index_root),
        dataset_url=spec.get("dataset_url", ""),
        samples=samples,
        # Only offer a corpus whose index is actually present on this machine.
        enabled=bool(index_dir) and os.path.isdir(index_dir),
    )


def load_config() -> AppConfig:
    """Build the application config from ``config.json`` + ``samples_data.json``."""
    config_path = os.environ.get("FINDMYTEXT_CONFIG", DEFAULT_CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    samples_file = raw.get("samples_file", "samples_data.json")
    samples_path = (
        samples_file if os.path.isabs(samples_file)
        else os.path.join(config_dir, samples_file)
    )
    samples_by_id, no_match = _load_samples(samples_path)

    # ``index_root`` may be overridden by the environment so a deployment can
    # relocate the indexes without editing the tracked config.json (e.g. the
    # systemd unit sets FINDMYTEXT_INDEX_ROOT).
    index_root = os.environ.get("FINDMYTEXT_INDEX_ROOT") or raw.get("index_root", "")
    corpora = [
        _build_corpus(spec, index_root, samples_by_id, no_match)
        for spec in raw.get("corpora", [])
    ]

    cfg = AppConfig(
        corpora=corpora,
        protection=ProtectionConfig(**raw.get("protection", {})),
        params=AlgoParams(**raw.get("params", {})),
        samples=[],
    )
    for key in ("github_url", "paper_url", "max_loaded_indexes", "max_text_chars"):
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg


def config_public_dict(cfg: AppConfig) -> dict[str, Any]:
    """Subset of the config that is safe to expose to the browser."""
    return {
        "corpora": [
            {
                "id": c.id,
                "label": c.label,
                "description": c.description,
                "dataset_url": c.dataset_url,
                "samples": [asdict(s) for s in c.samples],
            }
            for c in cfg.corpora
            if c.enabled
        ],
        "samples": [asdict(s) for s in cfg.samples],
        "github_url": cfg.github_url,
        "paper_url": cfg.paper_url,
        "max_text_chars": cfg.max_text_chars,
        "captcha": {
            "provider": cfg.protection.captcha_provider,
            "sitekey": cfg.protection.captcha_sitekey,
        },
        "password_required": bool(cfg.protection.password),
    }
