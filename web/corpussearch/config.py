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

from .titles import RESOLVER, wayback_url

# Base directory holding the prebuilt corpus indexes. Override per machine with
# the FINDMYTEXT_INDEX_ROOT environment variable.
INDEX_ROOT = os.environ.get("FINDMYTEXT_INDEX_ROOT", "/home/jullum/copyai_local")


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

_NO_MATCH_SAMPLE = SampleText(
    label="Unrelated text (should not match)", text=_NO_MATCH_TEXT
)

# Dataset viewers so users can browse/search the corpora themselves.
_ARXIV_DATASET_URL = "https://huggingface.co/datasets/common-pile/arxiv_papers/viewer/default/"


def _load_curated() -> dict[str, list[SampleText]]:
    """Load curated per-corpus examples from ``samples_data.json`` (if present).

    The file is generated by ``web/tools/build_*.py``; keeping the examples as
    data (rather than in code) avoids escaping issues with LaTeX-heavy arXiv
    abstracts and makes them trivial to regenerate.
    """
    path = os.path.join(os.path.dirname(__file__), "samples_data.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (FileNotFoundError, ValueError):
        return {}
    return {k: [SampleText(**s) for s in v] for k, v in raw.items()}


_CURATED = _load_curated()


def _disk_corpus(
    corpus_id: str,
    label: str,
    description: str,
    rel_path: str,
    doc_url_template: str,
    samples: list[SampleText],
    *,
    title_map_name: str = "",
    url_map_name: str = "",
    dataset_url: str = "",
) -> Optional[CorpusConfig]:
    """Build a disk-backed corpus if its index directory exists, else None."""
    index_dir = os.path.join(INDEX_ROOT, rel_path)
    if not os.path.isdir(index_dir):
        return None
    corpus_dir = os.path.dirname(index_dir)
    return CorpusConfig(
        id=corpus_id,
        label=label,
        description=description,
        index_dir=index_dir,
        doc_url_template=doc_url_template,
        title_map_path=os.path.join(corpus_dir, title_map_name) if title_map_name else "",
        url_map_path=os.path.join(corpus_dir, url_map_name) if url_map_name else "",
        dataset_url=dataset_url,
        samples=[*samples, _NO_MATCH_SAMPLE],
    )


# --- Curated example texts -------------------------------------------------
# arXiv examples (expanded from the built title map — abstracts of papers that
# are present in the open-licensed subset behind the index).
_ARXIV_SAMPLES: list[SampleText] = [
    SampleText(
        label="arXiv abstract (matches the corpus)",
        text=(
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
]

# Wikipedia examples (expanded from the built title map).
_WIKI_SAMPLES: list[SampleText] = [
    SampleText(
        label="Wikipedia passage (matches the corpus)",
        text=(
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
]

# HPLT examples: varied English web pages present in the crawl, each with a
# confirmed web.archive.org snapshot (the originals are mostly gone).
_HPLT_SAMPLES: list[SampleText] = [
    SampleText(
        label="Health journalism — Morecambe Bay report",
        url="http://www.skepticalob.com/2015/03/maternity-horror-at-morecambe-bay-is-the-inevitable-result-of-the-radicalization-of-midwifery.html",
        archive_url="https://web.archive.org/web/20170506172930/http://www.skepticalob.com/2015/03/maternity-horror-at-morecambe-bay-is-the-inevitable-result-of-the-radicalization-of-midwifery.html",
        text=(
            "Ideas have consequences. Bad ideas have deadly consequences. Today\u2019s "
            "report on the deaths more than a dozen babies and mothers at a UK "
            "hospital is a catalog of horrors. According to The Guardian: Frontline "
            "staff were responsible for \u201cinappropriate and unsafe care\u201d and the "
            "response to potentially fatal incidents by the trust hierarchy was "
            "\u201cgrossly deficient, with repeated failure to investigate properly and "
            "learn lessons\u201d. Kirkup [the author of the report] said this \u201clethal "
            "mix\u201d of factors had led to 20 instances of significant or major "
            "failures of care at Furness general hospital, associated with three "
            "maternal deaths and the deaths of 16 babies at or shortly after birth."
        ),
    ),
    SampleText(
        label="Bike maintenance — ZIPP 217 hub",
        url="http://zipp.com/support/maintenance/model217.php",
        archive_url="https://web.archive.org/web/20170504220326/http://zipp.com/support/maintenance/model217.php",
        text=(
            "ZIPP 217 Hub Maintenance - Hey, this is easy! Normal maintenance of "
            "the ZIPP 217 cassette hub requires regular lubrication of the cassette "
            "bushings with ZIPP LUBE. One tube of ZIPP Lube is supplied with each "
            "new hub with additional tubes available directly from ZIPP Speed "
            "Weaponry or your ZIPP dealer. CAUTION: USE ONLY THE RECOMMENDED "
            "LUBRICANTS. OTHER BRANDS AND TYPES WILL CAUSE STICKING OF THE CASSETTE "
            "MECHANISM AND MAY RESULT IN NON-WARRANTY DAMAGE TO THE CASSETTE "
            "MECHANISM. Lubricate the hub every 300-500 miles. The cassette body is "
            "floating on a thin film of lubricant that must be replenished."
        ),
    ),
    SampleText(
        label="Mining & finance — gold exploration",
        url="https://web.archive.org/web/20170504190047/https://ceo.ca/@Goldfinger/millrock-resources-tsx-v-mro-leveraged-upside-to-the-future-of-gold-mining",
        text=(
            "Gold producers are going out of business, continuously depleting their "
            "reserves to sustain production levels. Made worse, in the past five "
            "year gold mining bear market, there has been a dearth of exploration "
            "activity and new discoveries globally. According to a recent Goldcorp "
            "presentation gold reserves among the major producers have declined "
            "~15% during the last three years. Moreover, production is set to "
            "decline further over the coming years as the majors exhaust mines "
            "which were discovered decades ago. Mining gold and finding gold are "
            "two very different businesses. Majors tend to be good at pulling gold "
            "out of the ground, an engineering and operational activity."
        ),
    ),
    SampleText(
        label="Law blog — pedestrian safety",
        url="https://web.archive.org/web/20170503203918/http://www.bardsleyandgray.com/blog/page/2/",
        text=(
            "New Bedford MA during the fall season is always a beautiful sight and "
            "a welcoming change from the oppressive summer heat. Unfortunately, the "
            "fall season brings about another major change: an increase in "
            "pedestrian and motor vehicle accidents. At the Law Office of Bardsley "
            "and Gray, our personal injury attorneys want New Bedford residents to "
            "know that both drivers and pedestrians should be more aware of their "
            "surroundings when traveling. Drivers are required by state law to stop "
            "for all pedestrians who have entered crosswalks. Pedestrians need to "
            "obey traffic signals at all intersections; they exist in order to "
            "provide safe passage from one side of the street to the other."
        ),
    ),
    SampleText(
        label="Travel diary — safari in Kenya",
        url="https://web.archive.org/web/20170416222030/https://thedreamingtraveller.com/2015/09/06/kenya/",
        text=(
            "It was amazing to look out the window flying from Uganda to Kenya as "
            "the difference in scenery changed so dramatically in just a 1 hour "
            "flight. It went from flat red dirt, to lushes greenery, hills, mass "
            "populated areas where all you can see for miles is housing and then "
            "back to vast nothingness. We flew into Nairobi, as it was our starting "
            "point for our overland trip with Acacia Africa. Nairobi is hustling "
            "and bustling with people everywhere, a real high-rise city. Our tour "
            "guide Pete is from Kenya and was very excited to get to our first "
            "destination, the Maasai Mara National Reserve."
        ),
    ),
    SampleText(
        label="Library tech — an unconference write-up",
        url="https://web.archive.org/web/20170505032634/https://www.preater.com/2014/08/",
        text=(
            "Earlier in August I had the pleasure of helping organize and run a "
            "tech-focused library unconference, Pi and Mash, at Senate House "
            "Library at the University of London. The other organizers were Simon "
            "Barron of University of London, and Ka-Ming Pang of St Georges, "
            "University of London. They were both brilliant to work with and "
            "brought enormous energy, fresh perspectives, and thoughtfulness and "
            "professionalism to organizing the day. This event was a long time in "
            "gestation, from the initial agreement back in January that we\u2019d work "
            "together to the day itself in early August."
        ),
    ),
]


def _default_config() -> AppConfig:
    corpora: list[CorpusConfig] = []

    # The three large prebuilt corpora, added only when their index is present.
    real = [
        _disk_corpus(
            "arxiv",
            "arXiv papers",
            "Scientific papers from arXiv (open-licensed subset). Try pasting an "
            "abstract, or load one of the examples.",
            "arxiv/index(4,6)",
            "https://arxiv.org/abs/{doc_id}",
            _CURATED.get("arxiv", _ARXIV_SAMPLES),
            title_map_name="titles.json.gz",
            dataset_url=_ARXIV_DATASET_URL,
        ),
        _disk_corpus(
            "wiki",
            "English Wikipedia",
            "English Wikipedia articles. Try pasting a sentence from an article, "
            "or load one of the examples.",
            "wiki/index(4,6)_wikipedia",
            "",  # index doc ids are internal, not Wikipedia curids — no per-match link.
            _CURATED.get("wiki", _WIKI_SAMPLES),
        ),
        _disk_corpus(
            "hplt",
            "HPLT web crawl",
            "A large multilingual web-crawl corpus (HPLT). Examples link to the "
            "original page and a web.archive.org snapshot, as many originals have "
            "since changed or gone offline.",
            "hplt/index(4,6)",
            "",  # HPLT ids are content hashes; urls come from the side-car map.
            _CURATED.get("hplt", _HPLT_SAMPLES),
            url_map_name="urls.json.gz",
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
