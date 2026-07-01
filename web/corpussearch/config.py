"""Application configuration for the findmytext.nr.no demo.

Configuration is plain data with sensible defaults baked in. To customise a
deployment without touching code, point the ``FINDMYTEXT_CONFIG`` environment
variable at a JSON file; its keys are shallow-merged over the defaults (see
``config.example.json``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Base directory holding the prebuilt corpus indexes. Override per machine with
# the FINDMYTEXT_INDEX_ROOT environment variable.
INDEX_ROOT = os.environ.get("FINDMYTEXT_INDEX_ROOT", "/home/jullum/copyai_local")


@dataclass
class SampleText:
    label: str
    text: str


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
    # Per-corpus example texts (one known to match, one known not to).
    samples: list[SampleText] = field(default_factory=list)
    enabled: bool = True

    def doc_url(self, doc_id: str) -> Optional[str]:
        if not self.doc_url_template:
            return None
        return self.doc_url_template.format(doc_id=doc_id)


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
# Defaults
# ---------------------------------------------------------------------------
# A passage that is not present in any of the corpora; reused as the
# "should not match" sample for every real corpus.
_NO_MATCH_TEXT = (
    "Last Tuesday, Mirabel Quincy-Tannenbaum repotted her seventeen miniature "
    "kumquat trees while humming an invented lullaby about a forgetful walrus "
    "named Pemberton who collected odd-numbered tram tickets beside the violet "
    "lighthouse at the edge of Grumbleton-on-Sea."
)


def _disk_corpus(
    corpus_id: str,
    label: str,
    description: str,
    rel_path: str,
    doc_url_template: str,
    match_label: str,
    match_text: str,
) -> Optional[CorpusConfig]:
    """Build a disk-backed corpus if its index directory exists, else None."""
    index_dir = os.path.join(INDEX_ROOT, rel_path)
    if not os.path.isdir(index_dir):
        return None
    return CorpusConfig(
        id=corpus_id,
        label=label,
        description=description,
        index_dir=index_dir,
        doc_url_template=doc_url_template,
        samples=[
            SampleText(label=match_label, text=match_text),
            SampleText(label="Unrelated text (should not match)", text=_NO_MATCH_TEXT),
        ],
    )


def _default_config() -> AppConfig:
    corpora: list[CorpusConfig] = []

    # The three large prebuilt corpora, added only when their index is present.
    real = [
        _disk_corpus(
            "arxiv",
            "arXiv papers",
            "~2.4 million scientific papers from arXiv. Try pasting an abstract.",
            "arxiv/index(4,6)",
            "https://arxiv.org/abs/{doc_id}",
            "arXiv abstract (matches the corpus)",
            (
                "We define a new notion of cuspidality for representations of over a "
                "finite quotient of the ring of integers of a non-Archimedean local "
                "field using geometric and infinitesimal induction functors, which "
                "involve automorphism groups of torsion modules. When is a prime, we "
                "show that this notion of cuspidality is equivalent to strong "
                "cuspidality, which arises in the construction of supercuspidal "
                "representations of finite reductive groups. We show that strongly "
                "cuspidal representations share many features of cuspidal "
                "representations of finite general linear groups."
            ),
        ),
        _disk_corpus(
            "wiki",
            "English Wikipedia",
            "English Wikipedia articles. Try pasting a sentence from an article.",
            "wiki/index(4,6)_wikipedia",
            "https://en.wikipedia.org/?curid={doc_id}",
            "Wikipedia passage (matches the corpus)",
            (
                "Jeanne Pruett (born Norma Jean Bowman; January 30, 1937) is an "
                "American country music singer and songwriter. She also has credits "
                "as a published author. Pruett had several major hits as a music "
                "artist, but became best-known for 1973's \"Satin Sheets\". The song "
                "topped the country music charts and helped her secure a membership "
                "in the Grand Ole Opry cast. Pruett was raised near Pell City, "
                "Alabama, and grew up with a large family. She performed with her "
                "family from an early age and learned several musical instruments. "
                "She married guitarist Jack Pruett before turning 20 years old. Soon "
                "after, the couple moved to Nashville, Tennessee, where he was hired "
                "to play guitar for country artist Marty Robbins."
            ),
        ),
        _disk_corpus(
            "hplt",
            "HPLT web crawl",
            "A large multilingual web-crawl corpus (HPLT). Try pasting text from a "
            "web page.",
            "hplt/index(4,6)",
            "",  # HPLT document ids are content hashes with no public URL.
            "Web-page text (matches the corpus)",
            (
                "The terms and conditions specifying the actions permitted by the "
                "provider to users of our Services, including SMS and other services "
                "that may be introduced in the future. The Company reserves the right "
                "to make changes in policy at any time. All subscribers of our "
                "services, directly or indirectly, are required to participate in the "
                "only Acceptable Usage Policy as amended from time to time. The "
                "community service should only be used for legal purposes. "
                "Transmission, distribution or storage of material in violation of "
                "the law or regulation is prohibited."
            ),
        ),
    ]
    corpora.extend(c for c in real if c is not None)

    return AppConfig(
        corpora=corpora,
        protection=ProtectionConfig(),
        params=AlgoParams(),
        samples=[],
    )


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------
def _corpus_from_dict(d: dict[str, Any]) -> CorpusConfig:
    d = dict(d)
    samples = d.pop("samples", None)
    corpus = CorpusConfig(**d)
    if samples:
        corpus.samples = [SampleText(**s) for s in samples]
    return corpus


def _merge(cfg: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    if "corpora" in overrides:
        cfg.corpora = [_corpus_from_dict(c) for c in overrides["corpora"]]
    if "protection" in overrides:
        cfg.protection = ProtectionConfig(**overrides["protection"])
    if "params" in overrides:
        cfg.params = AlgoParams(**overrides["params"])
    if "samples" in overrides:
        cfg.samples = [SampleText(**s) for s in overrides["samples"]]
    for key in ("github_url", "paper_url", "max_loaded_indexes", "max_text_chars"):
        if key in overrides:
            setattr(cfg, key, overrides[key])
    return cfg


def load_config() -> AppConfig:
    cfg = _default_config()
    path = os.environ.get("FINDMYTEXT_CONFIG")
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        cfg = _merge(cfg, overrides)
    return cfg


def config_public_dict(cfg: AppConfig) -> dict[str, Any]:
    """Subset of the config that is safe to expose to the browser."""
    return {
        "corpora": [
            {
                "id": c.id,
                "label": c.label,
                "description": c.description,
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
