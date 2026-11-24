"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
    constructor() {
        this.children = [];
        this.value = "";
        this.textContent = "";
        this.innerHTML = "";
        this.classList = { add() {}, remove() {}, toggle() {} };
    }

    setAttribute(key, value) {
        if (key === "value") this.value = value;
    }

    append(...children) {
        this.children.push(...children);
    }

    get text() {
        return this.textContent + this.children
            .map((child) => typeof child === "string" ? child : child.text).join("");
    }
}

const ids = [
    "highlight-wrap", "highlight", "highlight-meta", "hl-error",
    "highlight-doc-select", "hl-btn",
];
const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
elements["public-config"] = { textContent: JSON.stringify({ corpora: [] }) };
const requests = [];
let resolveFetch;
const context = {
    document: {
        getElementById: (id) => elements[id],
        createElement: () => new Element(),
        addEventListener() {},
        querySelectorAll: () => [{ value: "jaccard" }],
    },
    fetch: (url, options) => {
        requests.push({ url, body: JSON.parse(options.body) });
        return new Promise((resolve) => { resolveFetch = resolve; });
    },
    console,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(
    path.join(__dirname, "../corpussearch/static/js/app.js"), "utf8"
), context);

const first = { doc_id: "a", title: "A", score: 0.0012 };
const baselineOnly = { doc_id: "b", title: "B", score: 0.6 };
const chainOnly = { doc_id: "c", title: "C", score: 0 };
context.result = {
    results: {
        connected_components: { top: first, ranking: [first, chainOnly] },
        jaccard: { top: first, ranking: [{ ...first, score: 0.8 }, baselineOnly] },
    },
};
vm.runInContext("renderHighlightControls(result)", context);
const options = elements["highlight-doc-select"].children.map((option) => option.text);
assert(options.some((text) => text.includes("Position-aware: 0.12%") &&
    text.includes("Baseline: 80.0%")));
assert(options.some((text) => text.includes("Position-aware: not in results") &&
    text.includes("Baseline: 60.0%")));
assert(options.some((text) => text.includes("Position-aware: <0.01%") &&
    text.includes("Baseline: not in results")));

vm.runInContext('lastText = "😀hello 🦊foo"; highlightDocLabel = "A"', context);
context.layers = {
    doc_id: "a",
    layers: [{
        mode: "jaccard", label: "Shared", coverage_pct: 55,
        clusters: [{ spans: [[1, 6], [8, 11]] }],
    }],
};
vm.runInContext("renderHighlightLayers(layers)", context);
assert.equal(elements.highlight.innerHTML,
    '😀<mark class="hl-bg-jaccard">hello</mark> 🦊<mark class="hl-bg-jaccard">foo</mark>');
assert.equal(elements["highlight-meta"].textContent, "Highlights for “A” — Shared");

context.layers.layers = [
    { mode: "cc_all", label: "All chains", coverage_pct: 75,
        clusters: [{ spans: [[1, 3]] }, { spans: [[3, 6]] }] },
    { mode: "jaccard", label: "Shared", coverage_pct: 0, clusters: [] },
];
vm.runInContext("renderHighlightLayers(layers)", context);
assert.equal(elements["highlight-meta"].textContent,
    "Highlights for “A” — All chains: 2 clusters · Shared: no matches");
context.layers.layers = [{ mode: "cc_all", label: "All chains", clusters: [] }];
vm.runInContext("renderHighlightLayers(layers)", context);
assert.equal(elements["highlight-meta"].textContent, "Highlights for “A” — All chains: 0 clusters");

vm.runInContext(`
    lastCorpus = "wiki";
    highlightDoc = "a";
    lastMethodParams = {position_threshold: 30};
    getMethodParams = () => ({position_threshold: 200});
    renderHighlightLayers = () => { globalThis.rendered = true; };
`, context);

async function checkHighlightRequest() {
    const pending = vm.runInContext("onHighlight()", context);
    assert.equal(requests[0].body.method_params.position_threshold, 30);
    vm.runInContext('highlightDoc = "b"', context);
    resolveFetch({ ok: true, json: async () => ({ layers: [] }) });
    await pending;
    assert.equal(context.rendered, undefined);
}

checkHighlightRequest().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
