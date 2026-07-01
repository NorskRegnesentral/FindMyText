"use strict";

const CONFIG = JSON.parse(document.getElementById("public-config").textContent);

const ALGO_LABELS = {
    connected_components: "Position-aware match (our method)",
    jaccard: "Shared fingerprints (Jaccard)",
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

document.addEventListener("DOMContentLoaded", () => {
    buildLinks();
    buildCorpora();
    setupPassword();
    setupCaptcha();
    setupCharCount();
    document.getElementById("detect-form").addEventListener("submit", onSubmit);
});

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
        buildSamples(c);
    };
    select.addEventListener("change", update);
    update();
}

function buildSamples(corpus) {
    const wrap = document.getElementById("samples");
    wrap.innerHTML = "";
    const samples = (corpus && corpus.samples && corpus.samples.length)
        ? corpus.samples
        : (CONFIG.samples || []);
    samples.forEach((s) => {
        const b = el("button", { type: "button" });
        b.textContent = s.label;
        b.addEventListener("click", () => {
            const ta = document.getElementById("text");
            ta.value = s.text;
            ta.dispatchEvent(new Event("input"));
            ta.focus();
        });
        wrap.append(b);
    });
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
            if (top.url) {
                const a = el("a", { href: top.url, target: "_blank", rel: "noopener" });
                a.textContent = top.doc_id;
                doc.append(a);
            } else {
                doc.append(document.createTextNode(top.doc_id));
            }
            box.append(doc);
        } else {
            box.classList.add("nomatch");
            box.append(el("div", { class: "score" }, "No match"));
            box.append(el("p", { class: "match-doc muted" }, "No document passed the threshold."));
        }

        if (data.ranking && data.ranking.length > 1) {
            const ul = el("ul", { class: "ranking" });
            data.ranking.forEach((r) => {
                const li = el("li", {});
                li.append(el("span", { class: "rk-doc" }, r.doc_id));
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

    renderHighlight(res.highlight);
    card.classList.remove("hidden");
    card.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeText(s) { return s == null ? "" : String(s); }

function renderHighlight(h) {
    const wrap = document.getElementById("highlight-wrap");
    if (!h || !h.spans || !h.spans.length) {
        wrap.classList.add("hidden");
        return;
    }
    const metaEl = document.getElementById("highlight-meta");
    metaEl.textContent =
        `${h.coverage_pct}% of your text overlaps with “${h.doc_id}” `
        + `(${ALGO_LABELS[h.algorithm] || h.algorithm}).`;

    const target = document.getElementById("highlight");
    let html = "";
    let cursor = 0;
    for (const [s, e] of h.spans) {
        html += escapeHtml(lastText.slice(cursor, s));
        html += "<mark>" + escapeHtml(lastText.slice(s, e)) + "</mark>";
        cursor = e;
    }
    html += escapeHtml(lastText.slice(cursor));
    target.innerHTML = html;
    wrap.classList.remove("hidden");
}
