"use strict";

const CONFIG = JSON.parse(document.getElementById("public-config").textContent);

const ALGO_LABELS = {
    connected_components: "Position-aware match (our method)",
    jaccard: "Shared fingerprints (baseline)",
};
const STAGE_LABELS = {
    starting: "Starting…",
    loading_index: "Loading corpus index…",
    fingerprinting: "Fingerprinting your text…",
    searching: "Searching the corpus…",
    scoring: "Scoring matches…",
    done: "Done",
};

let lastText = "";
let lastCorpus = "";
let highlightDoc = null;
let highlightDocLabel = null;

// Distinct colours for the per-cluster "all aligned passages" mode.
const CLUSTER_COLORS = [
    "#e6550d", "#3182bd", "#756bb1", "#31a354", "#d62728",
    "#17becf", "#bcbd22", "#8c564b", "#e377c2", "#7f7f7f",
];

document.addEventListener("DOMContentLoaded", () => {
    buildLinks();
    buildCorpora();
    setupPassword();
    setupCaptcha();
    setupCharCount();
    document.getElementById("detect-form").addEventListener("submit", onSubmit);
    document.getElementById("hl-btn").addEventListener("click", onHighlight);
});

function hostFromUrl(url) {
    try {
        return new URL(url).hostname.replace(/^www\./, "");
    } catch (e) {
        return null;
    }
}

function docLabel(item) {
    if (item && item.title) return item.title;
    // Web-crawl corpora (HPLT) have no title; the domain is far more telling
    // than the opaque content hash.
    if (item && item.url) {
        const host = hostFromUrl(item.url);
        if (host) return host;
    }
    const id = item ? item.doc_id : null;
    if (id == null) return "";
    const s = String(id);
    // Numeric internal ids -> a document reference.
    if (/^\d+$/.test(s)) return `document ${s}`;
    // Opaque hashes tell the user nothing; show a short prefix instead.
    return s.length > 12 ? `${s.slice(0, 8)}…` : s;
}

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "html") node.innerHTML = v;
        else node.setAttribute(k, v);
    }
    for (const c of children) node.append(c);
    return node;
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

function buildLinks() {
    const make = (href, label) => {
        if (!href) return null;
        const a = el("a", { href, target: "_blank", rel: "noopener" });
        a.textContent = label;
        return a;
    };
    const top = document.getElementById("top-links");
    const gh = make(CONFIG.github_url, "Source code (GitHub)");
    const paper = make(CONFIG.paper_url, "Read the paper");
    if (gh) top.append(gh);
    if (paper) top.append(paper);

    const foot = document.getElementById("foot-links");
    foot.textContent = "A research demo from Norsk Regnesentral (NR). ";
    const gh2 = make(CONFIG.github_url, "GitHub");
    if (gh2) foot.append(gh2);
}

function buildCorpora() {
    const select = document.getElementById("corpus");
    const desc = document.getElementById("corpus-desc");
    CONFIG.corpora.forEach((c) => {
        const opt = el("option", { value: c.id });
        opt.textContent = c.label;
        select.append(opt);
    });
    const update = () => {
        const c = CONFIG.corpora.find((x) => x.id === select.value);
        desc.textContent = c ? c.description : "";
        renderDatasetLink(c);
        renderCorpusSearch(c);
        buildSamples(c);
    };
    select.addEventListener("change", update);
    update();
}

function renderDatasetLink(corpus) {
    const line = document.getElementById("dataset-line");
    line.innerHTML = "";
    if (!corpus || !corpus.dataset_url) return;
    line.append(document.createTextNode("Want to search the corpus yourself? "));
    const a = el("a", { href: corpus.dataset_url, target: "_blank", rel: "noopener" });
    a.textContent = "Browse the dataset";
    line.append(a);
    line.append(document.createTextNode(" ↗"));
    if (corpus.dataset_note) {
        line.append(el("span", { class: "dataset-note" }, corpus.dataset_note));
    }
}

let corpusSearchTimer = null;
let corpusSearchSeq = 0;

function renderCorpusSearch(corpus) {
    const box = document.getElementById("corpus-search");
    const input = document.getElementById("corpus-search-input");
    const results = document.getElementById("corpus-search-results");
    if (corpusSearchTimer) {
        clearTimeout(corpusSearchTimer);
        corpusSearchTimer = null;
    }
    input.value = "";
    results.innerHTML = "";
    if (!corpus || !corpus.searchable) {
        box.classList.add("hidden");
        return;
    }
    box.classList.remove("hidden");

    const isUrl = corpus.search_kind === "url";
    input.placeholder = isUrl
        ? "Search by domain or URL (e.g. bbc.co.uk)…"
        : "Search which articles are in this corpus…";
    const emptyMsg = isUrl
        ? "No pages in this corpus match that domain."
        : "No articles in this corpus match that.";

    const run = () => {
        const q = input.value.trim();
        if (q.length < 2) {
            results.innerHTML = "";
            return;
        }
        const seq = ++corpusSearchSeq;
        fetch(`/api/corpus/${encodeURIComponent(corpus.id)}/titles?q=${encodeURIComponent(q)}`)
            .then((r) => r.json())
            .then((data) => {
                if (seq !== corpusSearchSeq) return; // stale response
                results.innerHTML = "";
                const items = (data && data.results) || [];
                if (!items.length) {
                    results.append(el("li", { class: "corpus-search-empty" }, emptyMsg));
                    return;
                }
                items.forEach((item) => {
                    const li = el("li");
                    if (item.url) {
                        const a = el("a", { href: item.url, target: "_blank", rel: "noopener" });
                        a.textContent = item.title || item.url;
                        li.append(a);
                        if (item.archive_url) {
                            li.append(document.createTextNode(" · "));
                            const arch = el("a", {
                                href: item.archive_url,
                                target: "_blank",
                                rel: "noopener",
                                class: "archive-link",
                            });
                            arch.textContent = "archived";
                            li.append(arch);
                        }
                    } else {
                        li.textContent = item.title || "";
                    }
                    results.append(li);
                });
            })
            .catch(() => {
                if (seq !== corpusSearchSeq) return;
                results.innerHTML = "";
            });
    };

    input.oninput = () => {
        if (corpusSearchTimer) clearTimeout(corpusSearchTimer);
        corpusSearchTimer = setTimeout(run, 250);
    };
}

function buildSamples(corpus) {
    const wrap = document.getElementById("samples");
    wrap.innerHTML = "";
    const samples = (corpus && corpus.samples && corpus.samples.length)
        ? corpus.samples
        : (CONFIG.samples || []);
    if (!samples.length) return;

    const select = el("select", { id: "sample-select", class: "sample-select" });
    select.append(el("option", { value: "" }, "Load an example…"));
    samples.forEach((s, i) => {
        const opt = el("option", { value: String(i) });
        opt.textContent = s.label;
        select.append(opt);
    });
    const source = el("a", {
        id: "sample-source", class: "sample-source hidden",
        target: "_blank", rel: "noopener",
    });
    source.textContent = "view source ↗";
    const archive = el("a", {
        id: "sample-archive", class: "sample-source hidden",
        target: "_blank", rel: "noopener",
    });
    archive.textContent = "archived ↗";

    select.addEventListener("change", () => {
        if (select.value === "") {
            source.classList.add("hidden");
            archive.classList.add("hidden");
            return;
        }
        const s = samples[Number(select.value)];
        const ta = document.getElementById("text");
        ta.value = s.text;
        ta.dispatchEvent(new Event("input"));
        ta.focus();
        if (s.url) {
            source.href = s.url;
            source.classList.remove("hidden");
        } else {
            source.classList.add("hidden");
        }
        if (s.archive_url) {
            archive.href = s.archive_url;
            archive.classList.remove("hidden");
        } else {
            archive.classList.add("hidden");
        }
    });
    wrap.append(select, source, archive);
}

function setupPassword() {
    if (CONFIG.password_required) {
        document.getElementById("password-field").classList.remove("hidden");
    }
}

function setupCaptcha() {
    const cap = CONFIG.captcha || {};
    if (!cap.provider || !cap.sitekey) return;
    const container = document.getElementById("captcha-container");
    const src = cap.provider === "recaptcha"
        ? "https://www.google.com/recaptcha/api.js"
        : "https://js.hcaptcha.com/1/api.js";
    const cls = cap.provider === "recaptcha" ? "g-recaptcha" : "h-captcha";
    container.append(el("div", { class: cls, "data-sitekey": cap.sitekey }));
    const script = el("script", { src, async: "", defer: "" });
    document.body.append(script);
}

function getCaptchaToken() {
    const cap = CONFIG.captcha || {};
    if (!cap.provider) return null;
    try {
        if (cap.provider === "recaptcha" && window.grecaptcha) {
            return window.grecaptcha.getResponse() || null;
        }
        if (cap.provider === "hcaptcha" && window.hcaptcha) {
            return window.hcaptcha.getResponse() || null;
        }
    } catch (e) { /* ignore */ }
    const field = document.querySelector("[name='h-captcha-response'], [name='g-recaptcha-response']");
    return field ? field.value || null : null;
}

function setupCharCount() {
    const ta = document.getElementById("text");
    const count = document.getElementById("char-count");
    ta.addEventListener("input", () => {
        count.textContent = ta.value.length.toLocaleString();
    });
}

async function onSubmit(ev) {
    ev.preventDefault();
    const errorEl = document.getElementById("form-error");
    errorEl.textContent = "";

    const text = document.getElementById("text").value.trim();
    const corpus = document.getElementById("corpus").value;
    const algorithms = Array.from(
        document.querySelectorAll("input[name='algorithm']:checked")
    ).map((c) => c.value);

    if (!text) { errorEl.textContent = "Please enter some text to check."; return; }
    if (!algorithms.length) { errorEl.textContent = "Select at least one algorithm."; return; }
    if (CONFIG.max_text_chars && text.length > CONFIG.max_text_chars) {
        errorEl.textContent = `Text too long (max ${CONFIG.max_text_chars.toLocaleString()} characters).`;
        return;
    }

    lastText = text;
    lastCorpus = corpus;
    const payload = {
        text, corpus, algorithms,
        password: document.getElementById("password")?.value || null,
        captcha_token: getCaptchaToken(),
    };

    const runBtn = document.getElementById("run-btn");
    runBtn.disabled = true;
    showProgress(true);
    document.getElementById("results-card").classList.add("hidden");

    try {
        const resp = await fetch("/api/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.error || `Request failed (${resp.status}).`);
        }
        await readStream(resp);
    } catch (e) {
        errorEl.textContent = e.message || "Something went wrong.";
        showProgress(false);
    } finally {
        runBtn.disabled = false;
    }
}

async function readStream(resp) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, nl).trim();
            buffer = buffer.slice(nl + 1);
            if (line) handleEvent(JSON.parse(line));
        }
    }
}

function handleEvent(ev) {
    if (ev.type === "progress") {
        updateProgress(ev);
    } else if (ev.type === "result") {
        showProgress(false);
        renderResult(ev);
    } else if (ev.type === "error") {
        showProgress(false);
        document.getElementById("form-error").textContent = ev.message;
    }
}

function showProgress(on) {
    const card = document.getElementById("progress-card");
    card.classList.toggle("hidden", !on);
    if (on) {
        document.getElementById("progress-fill").style.width = "5%";
        document.getElementById("progress-note").textContent = "";
    }
}

function updateProgress(ev) {
    document.getElementById("progress-fill").style.width = `${ev.pct}%`;
    document.getElementById("progress-stage").textContent =
        STAGE_LABELS[ev.stage] || "Working…";
    document.getElementById("progress-elapsed").textContent =
        ev.elapsed != null ? `${ev.elapsed.toFixed(1)}s` : "";
    if (ev.note) document.getElementById("progress-note").textContent = ev.note;
}

function renderResult(res) {
    const card = document.getElementById("results-card");
    const summary = document.getElementById("summary");
    summary.innerHTML = "";
    const grid = el("div", { class: "summary-grid" });

    for (const [key, data] of Object.entries(res.results)) {
        const box = el("div", { class: "result-box" });
        box.append(el("h4", {}, ALGO_LABELS[key] || key));
        const top = data.top;
        if (top) {
            const score = el("div", { class: "score" });
            score.append(document.createTextNode(String(top.score)));
            if (top.score_normalized != null) {
                score.append(el("small", {}, ` (${top.score_normalized}× a sentence)`));
            }
            box.append(score);
            const doc = el("p", { class: "match-doc" });
            doc.append(document.createTextNode("Best match: "));
            const label = docLabel(top);
            if (top.url) {
                const a = el("a", { href: top.url, target: "_blank", rel: "noopener" });
                a.textContent = label;
                doc.append(a);
            } else {
                doc.append(document.createTextNode(label));
            }
            if (top.archive_url) {
                doc.append(document.createTextNode(" · "));
                const arc = el("a", {
                    href: top.archive_url, target: "_blank", rel: "noopener",
                    class: "archive-link",
                });
                arc.textContent = "archived";
                doc.append(arc);
            }
            box.append(doc);
        } else {
            box.classList.add("nomatch");
            box.append(el("div", { class: "score" }, "No match"));
            box.append(el("p", { class: "match-doc muted" }, "No document passed the threshold."));
        }

        if (data.ranking && data.ranking.length > 1) {
            const ul = el("ul", { class: "ranking" });
            // Disambiguate identical display labels (e.g. several pages from the
            // same domain) by appending a running number.
            const labels = data.ranking.map((r) => docLabel(r));
            const totals = {};
            labels.forEach((l) => { totals[l] = (totals[l] || 0) + 1; });
            const seen = {};
            data.ranking.forEach((r, i) => {
                let rkLabel = labels[i];
                if (totals[rkLabel] > 1) {
                    seen[rkLabel] = (seen[rkLabel] || 0) + 1;
                    rkLabel = `${rkLabel} (${seen[rkLabel]})`;
                }
                const li = el("li", {});
                const left = el("span", { class: "rk-left" });
                if (r.url) {
                    const a = el("a", {
                        class: "rk-doc", href: r.url,
                        target: "_blank", rel: "noopener",
                    });
                    a.textContent = rkLabel;
                    left.append(a);
                } else {
                    left.append(el("span", { class: "rk-doc" }, rkLabel));
                }
                if (r.archive_url) {
                    left.append(document.createTextNode(" · "));
                    const arc = el("a", {
                        href: r.archive_url, target: "_blank", rel: "noopener",
                        class: "archive-link",
                    });
                    arc.textContent = "archived";
                    left.append(arc);
                }
                li.append(left);
                li.append(el("span", {}, String(r.score)));
                ul.append(li);
            });
            box.append(ul);
        }
        grid.append(box);
    }
    summary.append(grid);

    const meta = el("p", { class: "hint" });
    meta.textContent = `Searched ${escapeText(res.corpus_label)} · `
        + `${res.query_fingerprints.toLocaleString()} fingerprints from your text · `
        + `${res.elapsed.toFixed(1)}s`;
    summary.append(meta);

    renderHighlightControls(res);
    card.classList.remove("hidden");
    card.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeText(s) { return s == null ? "" : String(s); }

// Prepare the on-demand highlight controls once results are in. The user can
// pick which matched document to compare against; the default is the
// position-aware method's top doc if available, otherwise the baseline's.
function renderHighlightControls(res) {
    const wrap = document.getElementById("highlight-wrap");
    const target = document.getElementById("highlight");
    const meta = document.getElementById("highlight-meta");
    const err = document.getElementById("hl-error");
    const select = document.getElementById("highlight-doc-select");
    target.innerHTML = "";
    meta.textContent = "";
    err.textContent = "";
    select.innerHTML = "";

    // Collect every candidate document from all algorithm rankings, keeping the
    // best score each doc reached across methods (so a doc the baseline flagged
    // but our method scored 0 is still available to compare against).
    const byId = new Map();
    for (const key of Object.keys(res.results)) {
        const ranking = (res.results[key] && res.results[key].ranking) || [];
        ranking.forEach((r) => {
            if (!r || r.doc_id == null) return;
            const id = String(r.doc_id);
            const prev = byId.get(id);
            if (!prev) byId.set(id, { item: r, best: r.score || 0 });
            else prev.best = Math.max(prev.best, r.score || 0);
        });
    }
    const candidates = [...byId.values()].filter((c) => c.best > 0);
    candidates.sort((a, b) => b.best - a.best);

    if (!candidates.length) {
        highlightDoc = null;
        highlightDocLabel = null;
        wrap.classList.add("hidden");
        return;
    }

    // Populate the picker, disambiguating identical display labels.
    const labels = candidates.map((c) => docLabel(c.item));
    const totals = {};
    labels.forEach((l) => { totals[l] = (totals[l] || 0) + 1; });
    const seen = {};
    candidates.forEach((c, i) => {
        let lbl = labels[i];
        if (totals[lbl] > 1) {
            seen[lbl] = (seen[lbl] || 0) + 1;
            lbl = `${lbl} (${seen[lbl]})`;
        }
        select.append(el("option", { value: String(c.item.doc_id) },
            `${lbl} — score ${c.best}`));
    });

    // Default to the position-aware top match, else the baseline top match.
    const cc = res.results.connected_components && res.results.connected_components.top;
    const jac = res.results.jaccard && res.results.jaccard.top;
    const defaultId = String(
        (cc && cc.doc_id) || (jac && jac.doc_id) || candidates[0].item.doc_id
    );
    select.value = defaultId;

    const applySelection = (clear) => {
        const id = select.value;
        const c = candidates.find((x) => String(x.item.doc_id) === id) || candidates[0];
        highlightDoc = c.item.doc_id;
        highlightDocLabel = c.item.title || docLabel(c.item) || String(c.item.doc_id);
        if (clear) {
            target.innerHTML = "";
            meta.textContent = "";
            err.textContent = "";
        }
    };
    select.onchange = () => applySelection(true);
    applySelection(false);

    wrap.classList.remove("hidden");
}

async function onHighlight() {
    const err = document.getElementById("hl-error");
    err.textContent = "";
    const modes = Array.from(
        document.querySelectorAll("input[name='hl-mode']:checked")
    ).map((c) => c.value);
    if (!modes.length) { err.textContent = "Select at least one highlight style."; return; }
    if (!highlightDoc) { err.textContent = "No match to highlight."; return; }

    const btn = document.getElementById("hl-btn");
    btn.disabled = true;
    btn.textContent = "Highlighting…";
    try {
        const resp = await fetch("/api/highlight", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: lastText,
                corpus: lastCorpus,
                doc_id: highlightDoc,
                modes,
                password: document.getElementById("password")?.value || null,
                captcha_token: getCaptchaToken(),
            }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || `Request failed (${resp.status}).`);
        renderHighlightLayers(data);
    } catch (e) {
        err.textContent = e.message || "Something went wrong.";
    } finally {
        btn.disabled = false;
        btn.textContent = "Show highlighting";
    }
}

function markRange(arr, clusters) {
    for (const c of clusters) {
        for (const span of c.spans) {
            const s = span[0], e = span[1];
            for (let i = s; i < e && i < arr.length; i++) arr[i] = 1;
        }
    }
}

// Overlay one or more highlight layers on the query text. Visual channels are
// kept distinct so layers can be shown together:
//   - background: largest aligned passage (strong) takes priority over Jaccard.
//   - underline:  per-cluster coloured bar for the "all aligned passages" mode.
function renderHighlightLayers(data) {
    const target = document.getElementById("highlight");
    const meta = document.getElementById("highlight-meta");
    const N = lastText.length;
    const jac = new Uint8Array(N);
    const ccL = new Uint8Array(N);
    const ccAll = new Int16Array(N);
    ccAll.fill(-1);

    for (const layer of data.layers) {
        if (layer.mode === "jaccard") {
            markRange(jac, layer.clusters);
        } else if (layer.mode === "cc_largest") {
            markRange(ccL, layer.clusters);
        } else if (layer.mode === "cc_all") {
            layer.clusters.forEach((c, idx) => {
                for (const span of c.spans) {
                    const s = span[0], e = span[1];
                    for (let i = s; i < e && i < N; i++) if (ccAll[i] < 0) ccAll[i] = idx;
                }
            });
        }
    }

    let html = "";
    let i = 0;
    while (i < N) {
        const kJ = jac[i], kL = ccL[i], kA = ccAll[i];
        let j = i + 1;
        while (j < N && jac[j] === kJ && ccL[j] === kL && ccAll[j] === kA) j++;
        const chunk = escapeHtml(lastText.slice(i, j));
        if (!kJ && !kL && kA < 0) {
            html += chunk;
        } else {
            const classes = [];
            if (kL) classes.push("hl-bg-cc-largest");
            else if (kJ) classes.push("hl-bg-jaccard");
            let style = "";
            if (kA >= 0) {
                classes.push("hl-underline");
                const color = CLUSTER_COLORS[kA % CLUSTER_COLORS.length];
                style = ` style="border-bottom-color:${color}"`;
            }
            html += `<mark class="${classes.join(" ")}"${style}>${chunk}</mark>`;
        }
        i = j;
    }
    target.innerHTML = html || escapeHtml(lastText);

    const parts = data.layers.map((l) => {
        const n = l.clusters.length;
        const extra = l.mode === "cc_all" && n
            ? ` (${n} cluster${n === 1 ? "" : "s"})`
            : "";
        return `${l.label}: ${l.coverage_pct}%${extra}`;
    });
    meta.textContent = parts.length
        ? `Overlap with “${highlightDocLabel || data.doc_id}” — ${parts.join(" · ")}`
        : `No overlapping passages found in “${highlightDocLabel || data.doc_id}” for the selected styles.`;
}
