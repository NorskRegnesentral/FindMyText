"""Empirical harness to craft "decoy" demo examples.

A decoy = a genuine (short) excerpt from a known corpus document, padded with
generic boilerplate (generic ML phrasing for arXiv; common English sentences for
wiki/hplt). The goal: the shared-fingerprint *baseline* is distracted by the
scattered boilerplate overlap (it either ranks a WRONG document first or reports
many near-tied false positives), while the position-aware *clustering* method
still locks onto the true source via the one coherent excerpt.

Run:
    ../.venv/bin/python decoy_experiment.py arxiv
    ../.venv/bin/python decoy_experiment.py wiki
    ../.venv/bin/python decoy_experiment.py hplt
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from detector import TextContainmentDetector  # noqa: E402

INDEX_ROOT = "/home/jullum/copyai_local"
INDEX_DIRS = {
    "arxiv": "arxiv/index(4,6)",
    "wiki": "wiki/index(4,6)_wikipedia",
    "hplt": "hplt/index(4,6)",
}
CPARAMS = {
    "method": "rectangle",
    "position_threshold": 10,
    "offset_threshold": 10,
    "min_cluster_size": 5,
}
TOP_K = 10
MIN_FP = 5
NORM = 8.0

# --------------------------------------------------------------------------- #
# True-source excerpts (verbatim from the indexed docs) + generic filler.
# --------------------------------------------------------------------------- #

# arXiv: true source = the interpretable-by-design classifier paper (1710.10301).
# Keep three genuine sentences, followed by generic ML paper boilerplate.
ARXIV_TRUE_ID = "1710.10301"
ARXIV_EXCERPT = (
    "This work presents a new classifier that is specifically designed to be "
    "fully interpretable. This technique determines the probability of a class "
    "outcome, based directly on probability assignments measured from the "
    "training data. The accuracy of the predicted probability can be improved "
    "by measuring more probability estimates from the training data to create a "
    "series expansion that refines the predicted probability."
)
ARXIV_FILLER = (
    "In this paper we propose a novel approach to address this challenging "
    "problem. Extensive experiments on several benchmark datasets demonstrate "
    "the effectiveness of the proposed method. To the best of our knowledge, "
    "this is the first work to tackle this task in such a setting. Our model "
    "achieves state-of-the-art results and outperforms existing baselines by a "
    "significant margin. We conduct an ablation study to analyze the "
    "contribution of each component of our framework. The remainder of this "
    "paper is organized as follows. Section 2 reviews the related work, Section "
    "3 describes the proposed method in detail, and Section 4 presents the "
    "experimental results. Finally, we conclude the paper and discuss "
    "directions for future work. The results show that our method generalizes "
    "well to unseen data and is robust to noise. We hope that our findings will "
    "inspire further research in this direction."
)

# Wikipedia: true source = Nova Scotia. Genuine two sentences + common English.
WIKI_TRUE_ID = "Nova Scotia"
WIKI_EXCERPT = (
    "The mainland portion of the Nova Scotia peninsula is attached to North "
    "America through the Isthmus of Chignecto. Various offshore islands, the "
    "largest of which is Cape Breton Island, form the bulk of the eastern part "
    "of the province."
)
WIKI_FILLER = (
    "It is one of the most populous cities in the United States. The city is "
    "located in the northern part of the state and is the largest city in the "
    "region. During the second half of the twentieth century it grew rapidly "
    "and became an important centre of industry and trade. It is home to a wide "
    "range of cultural institutions and is best known for its historic "
    "architecture. According to the most recent census the population has "
    "continued to grow over the past decade. As well as the main university "
    "there are a number of colleges in the area. On the other hand the "
    "surrounding countryside remains largely rural. For the first time in many "
    "years the local economy has begun to recover. The area was originally "
    "settled in the early nineteenth century by immigrants from Europe. It is "
    "widely regarded as one of the most important places in the country."
)

# HPLT: true source = the Kenya safari travel diary (distinctive personal blog,
# not mirrored). Genuine excerpt + common English web boilerplate as filler.
HPLT_TRUE_HINT = "a6ed1259742a298e5e29c8fe237c10d4"
HPLT_EXCERPT = (
    "It was amazing to look out the window flying from Uganda to Kenya as the "
    "difference in scenery changed so dramatically in just a 1 hour flight. We "
    "flew into Nairobi, as it was our starting point for our overland trip with "
    "Acacia Africa. Our tour guide Pete is from Kenya and was very excited to "
    "get to our first destination, the Maasai Mara National Reserve."
)
HPLT_FILLER = (
    "Thank you for taking the time to read this. If you have any questions "
    "please do not hesitate to contact us. We are here to help you every step "
    "of the way. Please make sure to follow the instructions carefully. This "
    "will only take a few minutes of your time. We hope you find this "
    "information useful. Feel free to share it with your friends and family. "
    "Do not forget to check back for more updates soon. We look forward to "
    "hearing from you. Have a great day and thank you again. For best results "
    "follow each of the steps in order. If in doubt please ask for help before "
    "you begin."
)

DECOYS = {
    "arxiv": (ARXIV_EXCERPT, ARXIV_FILLER, ARXIV_TRUE_ID),
    "wiki": (WIKI_EXCERPT, WIKI_FILLER, WIKI_TRUE_ID),
    "hplt": (HPLT_EXCERPT, HPLT_FILLER, HPLT_TRUE_HINT),
}


def top_n(scores: dict, n: int = 6):
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(d, int(round(s))) for d, s in items[:n]]


def report(corpus: str, excerpt_override: str | None = None):
    det = TextContainmentDetector(
        os.path.join(INDEX_ROOT, INDEX_DIRS[corpus]),
        top_k=TOP_K,
        min_fingerprints=MIN_FP,
    )
    excerpt, filler, true_id = DECOYS[corpus]
    if excerpt_override:
        excerpt = excerpt_override

    layouts = {
        "excerpt_only": excerpt,
        "filler_then_excerpt": filler + " " + excerpt,
        "excerpt_then_filler": excerpt + " " + filler,
        "filler_excerpt_filler": filler + " " + excerpt + " " + filler,
    }

    for name, text in layouts.items():
        jac = det.find_matches_jaccard(text, score="count")
        clu = det.find_matches_clustering(text, CPARAMS, score="count")
        print(f"\n=== {corpus} :: {name} (true source ~ {true_id!r}) ===")
        print(f"  chars={len(text)}")
        print("  BASELINE (jaccard count) top:")
        for d, s in top_n(jac):
            mark = " <== TRUE" if true_id.lower() in d.lower() else ""
            print(f"    {s:4d}  ({s/NORM:.2f}x sent)  {d}{mark}")
        print("  OURS (clustering count) top:")
        for d, s in top_n(clu):
            mark = " <== TRUE" if true_id.lower() in d.lower() else ""
            print(f"    {s:4d}  ({s/NORM:.2f}x sent)  {d}{mark}")


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "arxiv"
    override = sys.argv[2] if len(sys.argv) > 2 else None
    report(corpus, override)
