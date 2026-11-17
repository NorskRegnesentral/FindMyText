# Why Helsetilsynet, Skogholt and Regjeringen appear in the giant group

**Nynorsk, K = 50. Investigation completed 26 October 2026.**

## Short answer

The three hosts have **different mechanisms**, not one common data-cleaning failure:

| Host | Documents in the giant group | Main explanation supported by inspected examples |
|---|---:|---|
| `www.helsetilsynet.no` | 585 | Standard inspection-report methodology, legal explanations and other recurring report sections. These are meaningful report templates, not primarily previews of unrelated pages. |
| `www.skogholt.org` | 467 | The same articles occur on article pages, monthly archives, tags, categories and author listings. Some substantial text also occurs on other publishing sites. |
| `www.regjeringen.no` | 375 | Repeated policy/budget explanations across official documents and years, legal quotations, and documents also published on other official sites. |

**A shared passage is real text overlap, but not necessarily evidence that two independently authored documents copied each other.** Templates, archive representations, document versions, quotations and publication mirrors need different interpretations.

### Open a complete highlighted pair for each host

These files contain both complete original corpus text fields, not just the excerpts below:

1. [Helsetilsynet: two inspection reports](host_overlap_examples/helsetilsynet_template.md).
2. [Skogholt: monthly archive versus individual article](host_overlap_examples/skogholt_archive_and_article.md).
3. [Regjeringen: recurring explanations in annual reports](host_overlap_examples/regjeringen_annual_report.md).

Use **Markdown Preview** (`Ctrl+Shift+V` in VS Code). Yellow marks the source ranges supporting shared stored fingerprints; bold yellow identifies the detector's best cluster. Links at the top jump to that cluster in either text. Original punctuation and whitespace are preserved.

## Scope and how to read the numbers

- The document counts refer to members of the **27,691-document K=50 giant component**, not every corpus document from each host.
- K=50 means **at least 50 distinct shared stored fingerprints**.
- Graph measurements below cover **all K=50 edges involving these members**. A partner of one of these members must be in the same connected component.
- All 1,427 target-host records were retrieved for raw-text layout/repeated-line inspection. Source hashes and positions were checked for the bounded passage review.
- There are **124 unique reviewed pairs**: 39 for Helsetilsynet, 44 for Skogholt and 41 for Regjeringen. Selection includes seeded random within-host different-URL pairs, available same-URL pairs, near-threshold cases and cross-host pairs, plus strongest-overlap supplements.
- All 124 produced a cluster with the unchanged `FingerprintChainDetector` defaults. **That is not precision:** standard templates and archive duplicates can legitimately produce clusters.
- “Best-cluster support” is distinct fingerprint support in one positional cluster. It is not the whole-record overlap, nor a count of words.
- Nine illustrations below are analyst-selected from that evidence, not a prevalence sample. Quotations are short excerpts; exact-run lengths refer to **normalized tokens inside the best-cluster envelopes**, not necessarily byte-identical text.
- URLs identify the historical captured records. A live URL may have changed or disappeared. Crawl dates are not publication dates, and no copying direction is inferred.

### Exact connectivity

| Measurement | Helsetilsynet | Skogholt | Regjeringen |
|---|---:|---:|---:|
| Member documents | 585 | 467 | 375 |
| Distinct captured URL strings | 585 | 420 | 374 |
| Edges between members of this same host | 19,227 | 994 | 403 |
| Of those, edges between captures with exactly the same URL | 0 | 25 | 1 |
| Edges to other hosts | 124 | 51 | 942 |
| Edges directly to Docplayer | 118 | 8 | 498 |
| Member documents directly touching Docplayer | 115 | 4 | 198 |
| Largest component using only this host's edges | 582 | 463 | 92 |

Cross-host edges can be counted in both relevant host rows, so do not sum the rows as disjoint edge sets. Same-URL counts do not include every possible alias, HTTP/HTTPS variant or equivalent chapter URL.

The graph does not imply that all 585, 467 or 375 documents mutually overlap. In particular, the Regjeringen members form **150 within-host components, including 116 isolated vertices**, when other hosts are removed.

---

## 1. Helsetilsynet: recurring inspection-report templates

### Finding

The inspected passages are mostly **methodology, inspection authority, legal standards, reporting procedure and standard explanations** within reports about different services and municipalities. This is not the same pattern as Docplayer recommendation lists.

There is strong evidence beyond a single example:

- “Tilsynet er derfor avslutta” has a selected fingerprint present in **441 of the 585 members**.
- The full line beginning “Ved dette tilsynet blei det konstatert avvik frå lov eller forskrift ...” occurs in **243 member records**.
- “Systemrevisjonen omfatta følgjande aktivitetar:” occurs in **244**.
- Repeated wording about what a system audit assesses also recurs across many inspected pairs.

These are statements about the captured text. They do not mean every report is interchangeable: case-specific findings can differ while methodology and legal frameworks repeat.

### H1. Different services, same audit-methodology wording

- **A, ID 250817:** [Meland kommune: services for young people / individual plans, 2013](https://www.helsetilsynet.no/tilsyn/tilsynsrapporter/hordaland/2013/meland-kommune-tjenester-unge-individuell-plan-2013/).
- **B, ID 726179:** [Haraldsplass Diakonale Sykehus: coordination when discharging patients, 2015](https://www.helsetilsynet.no/tilsyn/tilsynsrapporter/hordaland/2015/-haraldsplass-diakonale-sykehus-samhandling-om-utskriving-av-pasientar-fra-spesialisthelsetenesta-til-kommunen-2015/).
- **50 shared fingerprints overall; best-cluster support 16.**
- [Read both complete highlighted texts](host_overlap_examples/helsetilsynet_template.md).

Shared wording includes:

> tiltaka blir følgde opp i praksis og om nødvendig korrigerte

> tiltaka er gode nok for å sikre at lovgivinga blir følgd

The best cluster is around tokens **238–343** in A and **232–331** in B, in the descriptions of how the audit is conducted. There is a **45-token exact normalized run** within those envelopes.

**Interpretation:** different inspections use the same methodology template. This is not evidence that one inspection copied the other's findings.

### H2. Different municipalities, extensive shared legal framework

- **A, ID 987323:** [Molde: NAV's responsibility for children's needs when families seek financial support, 2023](https://www.helsetilsynet.no/tilsyn/tilsynsrapporter/more-romsdal/2023/molde-kommune-nav-kontorets-ansvar-for-aa-ivareta-barnas-behov-naar-familien-soeker-om-oekonomisk-stoenad-2023/).
- **B, ID 988473:** [Giske: NAV's responsibility for children's needs when families seek financial support, 2023](https://www.helsetilsynet.no/tilsyn/tilsynsrapporter/more-romsdal/2023/giske-kommune-nav-kontorets-ansvar-for-aa-ta-vare-pa-barns-behov-naar-familien-soeker-om-oekonomisk-stoenad-2023/).
- **969 shared fingerprints overall; best-cluster support 409.**

One shared legal explanation begins:

> Dersom Nav-kontoret avslår søknaden etter sosialtenestelova § 18, skal det vurderast om stønad i særlege tilfelle kan innvilgast etter sosialtenestelova § 19.

The strongest cluster covers long report sections with variable wording as well as matching text; inside it there is a **450-token exact normalized run**.

**Interpretation:** this is substantial reuse of a common legal/reporting framework, not just a generic website menu. Even a much higher K can retain such pairs.

### H3. How the report family connects to Docplayer

- **A, ID 584068:** [Hareid: services for people with intellectual disabilities, 2016](https://www.helsetilsynet.no/tilsyn/tilsynsrapporter/more-romsdal/2016/hareid-kommune-kommunale-helse-og-omsorgstenester-til-personar-med-utviklingshemming-2016/).
- **B, ID 943011:** [Captured Docplayer page named after a Skjåk out-of-hours medical-service inspection report, 2008](http://docplayer.me/3907018-Rapport-fra-tilsyn-med-legevakttenesta-og-deltaking-i-interkommunal-legevakt-skjak-kommune-2008.html).
- **60 shared fingerprints overall; best-cluster support 39.**

The shared section includes:

> Formålet med systemrevisjonen er å vurdere om verksemda tek hand om ulike krav i lovgivinga gjennom internkontrollen sin.

There is a **152-token exact normalized run** inside the best cluster.

**Interpretation:** standard inspection wording connects different report subjects across hosts. Such bridges can connect a large report family to the Docplayer-dominated component without all those reports being preview-contaminated.

### Frequency diagnostic and implication

Subtracting the **24 hashes present in at least 293 of these 585 members**, then applying K>=50 again, reduces within-host edges from **19,227 to 5,327**. The largest within-host component still contains **540 documents**.

This is a diagnostic, not a validated template-removal rule. Some of the common wording is meaningful report content. A global df threshold can also miss host-specific templates: the fingerprint for “Tilsynet er derfor avslutta” occurs in **512 documents across the full corpus**, below a 1,000-document cutoff.

**Recommended handling:** distinguish standard methods/legal frameworks from case-specific findings. Do not automatically delete all shared report text as extraction noise.

---

## 2. Skogholt: articles repeated across archive and listing pages

### Finding

Here the main mechanism is **multiple webpage representations of article text**: dated article pages, monthly archives, tags, categories and author listings. In inspected examples, listings contain substantial article text, sometimes nearly an entire article—not merely link titles or a short sidebar.

A URL-pattern diagnostic across all 467 members gives:

| URL role | Documents |
|---|---:|
| Archive/listing patterns (`/tema/`, `/merke/`, `/author/`, dated archives, etc.) | 332 |
| Dated article patterns | 121 |
| Other/unknown | 14 |

**962 of 994 within-host edges (96.8%) touch an archive/listing-pattern URL.** This is a URL diagnostic, not automatic proof of each edge's content mechanism, but it agrees with the inspected passages.

There is no single hash shared by half these members. The most frequent selected 4-gram is present in only **61 of 467**, unlike the much more widespread Helsetilsynet report wording.

### S1. Monthly archive versus the individual article

- **A, ID 968240:** [August 2023 archive](https://www.skogholt.org/2023/08/).
- **B, ID 1019443:** [“Ja, det går for tregt å ta vare på naturen”](https://www.skogholt.org/2023/08/ja-det-gar-for-tregt-a-ta-vare-pa-naturen/).
- **104 shared fingerprints overall; best-cluster support 104.**
- [Read both complete highlighted texts](host_overlap_examples/skogholt_archive_and_article.md).

The matching article text includes:

> er viktig for alle oss som bur i Tromsø. Korleis ulike element passar saman, spelar på lag for å gjere sentrum, bydeleane og bygdene våre til gode plassar å bu

There is a **370-token exact normalized run**, appearing at tokens **363–733** of the archive and **4–374** of the 376-token article.

**Interpretation:** nearly the whole article is present in the archive capture. This is not evidence of two independently authored articles copying each other.

### S2. Two category pages contain the same long article text

- **A, ID 475198:** [Politics category, page 2](https://www.skogholt.org/tema/politikk/page/2/).
- **B, ID 983954:** [Troms category](https://www.skogholt.org/tema/troms/).
- **276 shared fingerprints overall; best-cluster support 276.**

Both contain:

> i ein periode sagt at eg var ferdig med koronaen, men det viste seg at koronaen ikkje var ferdig med meg eller alle andre.

The best-cluster envelopes contain a **976-token exact normalized run**.

**Interpretation:** the same article appears under multiple categories. Different category pages can contain different overlapping sets of articles; this naturally chains many otherwise unrelated topics together.

### S3. There is also real cross-host republication/shared text

- **A, ID 464272:** [Environment category, AMP variant](https://www.skogholt.org/tema/miljo/?amp).
- **B, ID 553114:** [Nordnorsk Debatt: “Vi må stoppe Arctic Center”](https://nordnorskdebatt.no/article/vi-ma-stoppe-arctic-center).
- **197 shared fingerprints overall; best-cluster support 197.**

Shared argumentation includes:

> Ei utbygging og drenering av myra på Finnheia vil føre til store klimagassutslepp.

There is a **646-token exact normalized run** inside the best cluster.

**Interpretation:** this is substantial shared article content, consistent with republication or another common source. The archive representation and the cross-host text reuse are separate issues. The evidence does not establish copying direction, permission or authorship.

### Implication

Only **4** Skogholt members directly touch Docplayer, through **8 edges**. Nevertheless, its largest within-host cluster contains **463 documents**: membership in a large connected component does not require every document to have its own direct Docplayer link.

**Recommended handling:** distinguish article identity from archive/category/tag representations. For an article-level reuse graph, canonicalize or separately label those representations, retain capture/version provenance, and investigate cross-host article reuse independently. This is not a case where one global common-fingerprint cutoff removes the archive-chain mechanism.

---

## 3. Regjeringen: recurring official text, quotations and publication mirrors

### Finding

The inspected matches are **inside substantive government-document text**, including recurring annual explanations, statutory wording and policy statements. They do not resemble one universal recommendation widget.

There is also an important structural difference: these 375 members have only **403 within-host edges**, but **942 external edges**. They are not one densely connected Regjeringen-only family.

The largest external partner hosts are:

| Partner host | Edges |
|---|---:|
| `docplayer.me` | 498 |
| `www.statsbudsjettet.no` | 207 |
| `lovdata.no` | 39 |
| `www.stortinget.no` | 36 |
| `www.rettsdata.no` | 27 |
| `www.fn.no` | 19 |

These partner counts alone do not identify the mechanism; the examples below do.

### R1. Similar explanations in different annual government reports

- **A, ID 566497:** [Meld. St. 3 (2010–2011), chapter 1](https://www.regjeringen.no/no/dokumenter/meld-st-3-20102011/id641561/?ch=1).
- **B, ID 776811:** [Meld. St. 3 (2009–2010), chapter 1](https://www.regjeringen.no/no/dokumenter/Meld-St-3-2009--2010/id602886/?ch=1).
- **362 shared fingerprints overall; best-cluster support 67.**
- [Read both complete highlighted texts](host_overlap_examples/regjeringen_annual_report.md).

Shared explanatory text includes:

> Boks 1.1 Samanhengen mellom det strukturelle og det oljekorrigerte budsjettunderskotet

> Retningslinene for budsjettpolitikken tek sikte på ein gradvis og langsiktig forsvarleg auke i bruken av petroleumsinntekter i norsk økonomi

There is a **49-token exact normalized run** within the best cluster, alongside other shared and changed wording.

**Interpretation:** recurring explanations across annual reports. This is genuine document-body reuse, not an unrelated-page preview.

### R2. A long section published on both Regjeringen and Statsbudsjettet

- **A, ID 985844:** [Prop. 1 S (2019–2020), government publication](https://www.regjeringen.no/no/dokumenter/prop.-1-s-20192020/id2671714/?ch=1).
- **B, ID 986396:** [Svalbard budget 2020: corresponding section on Statsbudsjettet](https://www.statsbudsjettet.no/Statsbudsjettet-2020/Dokumenter1/Svalbard/Svalbardbudsjettet/Prop-1-S-/Del-1-Innleiing-/4-Forslag-til-loyvingar-til-svalbardformal-fra-andre-departement-over-statsbudsjettet-jf-vedlegg-1-og-2-/).
- **1,868 shared fingerprints overall; best-cluster support 1,369.**

The matched section starts with headings about the Ministry of Children and Families and child welfare, followed by:

> Forskrift av 1. september 1995 nr. 772 om lov om barneverntenesta på Svalbard regulerer fordelinga av ansvar og dekning av utgifter

The best cluster contains a **4,891-token exact normalized run**.

**Interpretation:** strong evidence of the same publication section appearing on two official publishing hosts. These should be considered publication representations/mirrors rather than automatically interpreted as independent copying events.

### R3. Legal wording in an official amendment and a handbook

- **A, ID 672:** [Captured Docplayer handbook about cooperation between child welfare and schools](http://docplayer.me/2688848-Handbok-for-samarbeid-mellom-barnevern-og-skule-opplaering-for-barn-og-unge-i-barneverninstitusjonar-i-hordaland.html).
- **B, ID 602429:** [Official decision amending the Education Act](https://www.regjeringen.no/no/dokumenter/vedtak-til-lov-om-endringar-i-opplarings/id106732/).
- **69 shared fingerprints overall; best-cluster support 30.**

Shared wording includes:

> tilsvarande opplæring, har etter søknad, rett til tre års heiltids vidaregåande opplæring.

There is a **51-token exact normalized run** inside the best cluster.

**Interpretation:** legal wording reproduced in another document provides a cross-host bridge. No recommendation-list explanation is needed for this particular match.

### Implication

No hash is shared by half these 375 members. Different document families reuse different passages. A few large publication mirrors, recurring annual sections and quotations can connect many such families to other hosts.

**Recommended handling:** distinguish document editions/chapters, same-publication mirrors, quotations and substantive reuse. Do not remove useful government-document content merely because it recurs.

---

## What this means for cleaning and scoring

There is **not one fix for all four hosts**:

| Mechanism | Appropriate response |
|---|---|
| Unrelated recommendation previews mixed into a document record | Identify content regions; exclude secondary previews from primary-document comparisons, or flag uncertain extraction. |
| Standard report methodology/legal framework | Label or downweight template/framework matches relative to case-specific text; retain the original meaningful content. |
| One article appearing on archive/tag/category pages | Model canonical article identity and archive representations separately. |
| Same official publication on multiple hosts/formats | Group publication representations and keep provenance. |
| Recurring annual policy text or legal quotations | Retain and characterize the reuse; do not equate it automatically with an improper copying event. |

The original Docplayer issue therefore should not be generalized to “all sites with many overlaps have bad extraction.”

For the three fully exported example pairs, a **full-corpus df>1,000 count filter** gives:

| Pair | Raw overlap | Filtered overlap |
|---|---:|---:|
| Helsetilsynet H1 | 50 | 49 |
| Skogholt S1 | 104 | 103 |
| Regjeringen R1 | 362 | 361 |

That near-threshold Helsetilsynet edge would disappear, but the filter has barely changed the evidence in any of the three examples. It does not distinguish templates, archive representations and publication reuse.

## Validation, limits and reproducibility

- All inspected raw texts reproduce their original forward fingerprint sets.
- Every returned aligned fingerprint position was checked against the exact normalized source k-grams.
- Exact normalized-run measurements were separately checked for the nine illustrations.
- Host-frequency subtraction was independently checked against NumPy intersections for **100 sampled edges per host**.
- The three highlighted full-text annexes preserve the complete source text; removing the HTML highlighting recovers the original characters exactly.
- Detector clustering defaults, stored 4-grams and window 6 were retained. There was no external-query top-k retrieval and no window-1 experiment.
- The study does not inspect all passages of all edges, recover all original HTML, establish copying direction, or measure corpus-wide precision.
- No original index, source record or production graph was changed.

Implementation: [analyze_overlap_hosts.py](../../experiments/review/analyze_overlap_hosts.py). Full-text export: [export_pair_review.py](../../experiments/review/export_pair_review.py).

Detailed local evidence is stored at:

`/nr/project/stat/COPY.AI/data/Lars/hplt_nynorsk/group_review/2026-10-26-defaults/three-hosts/report.json`

The evidence file contains exact counts, member IDs, sampling provenance, all reviewed pair contexts, and the nine source-verified illustrations.
