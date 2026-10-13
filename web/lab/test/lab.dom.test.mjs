// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/test/lab.dom.test.mjs"
// ============================================================================
// End-to-end test of the lab page: the real index.html in a headless DOM (jsdom), the real
// WebAssembly engine, and the real page code, driven by clicks and edits. Canvases are
// replaced by recording contexts so drawing can be checked without a browser. Run with:
//
//   MIDMAN_WASM=path/to/midman_api.wasm node --test web/lab/test/lab.dom.test.mjs
//
// jsdom is a test-only dependency: `npm install --no-save jsdom`. Set JSDOM_PATH to use a
// copy elsewhere, and REQUIRE_DOM=1 to fail instead of skip when it is missing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Engine } from "../engine.js";
import { init } from "../lab.js";
import { count, recorder } from "./support.mjs";

let JSDOM;
try {
  ({ JSDOM } = await import(process.env.JSDOM_PATH ?? "jsdom"));
} catch (e) {
  if (process.env.REQUIRE_DOM === "1") throw e;
}
const skip = JSDOM ? false : "jsdom is not installed (npm install --no-save jsdom)";

const wasmPath = process.env.MIDMAN_WASM ?? "dist/lab/midman_api.wasm";
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const bytes = readFileSync(wasmPath);

/** A fresh page, engine and wiring. Scheduled callbacks run immediately unless `queue` is given. */
async function page({ queue } = {}) {
  const dom = new JSDOM(html, { pretendToBeVisual: true });
  const { document, window } = dom.window;
  const contexts = new WeakMap();
  window.HTMLCanvasElement.prototype.getContext = function getContext() {
    if (!contexts.has(this)) contexts.set(this, recorder());
    return contexts.get(this);
  };
  const engine = await Engine.load(bytes);
  const schedule = queue ? (fn) => queue.push(fn) : (fn) => fn();
  const api = await init(document, engine, { schedule });
  const $ = (id) => document.getElementById(id);
  const ctxOf = (id) => contexts.get($(id));
  const fire = (el, type) => el.dispatchEvent(new window.Event(type, { bubbles: true }));
  const set = (id, value) => {
    $(id).value = String(value);
    fire($(id), "input");
  };
  const click = (id) => $(id).dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  const text = (id) => $(id).textContent;
  return { dom, document, window, engine, api, $, ctxOf, fire, set, click, text };
}

const TINY = [12, 16, 2, 2, 1, 32, 16, 10000, 1e-5, 1];

/** Opens the Train tab with the small test model and the default 150-step task. */
async function trainPage(options) {
  const p = await page(options);
  p.api.writeConfig(TINY);
  p.fire(p.$("cfg-0"), "input");
  p.click("calc-use");
  return p;
}

test("the page starts with the engine's own numbers", { skip }, async () => {
  const p = await page();
  assert.match(p.text("status"), /ABI 1/);
  assert.equal(p.$("preset").options.length, 7, "six presets and custom");
  assert.equal(p.text("calc-total"), "117,056", "the smoke preset's exact count");
  assert.equal(p.text("calc-head"), "16");
  assert.equal(p.text("calc-fits"), "yes");
  assert.equal(p.$("calc-invalid").hidden, true);
  assert.ok(count(p.ctxOf("calc-bars"), "fillRect") > 5, "the breakdown bars were drawn");
  assert.equal(p.$("error").hidden, true);
});

test("choosing a preset fills the form and the counts follow", { skip }, async () => {
  const p = await page();
  p.$("preset").value = "1";
  p.fire(p.$("preset"), "change");
  assert.equal(p.text("calc-total"), "1,315,968");
  assert.equal(p.$("cfg-1").value, "128");
  p.$("preset").value = "5";
  p.fire(p.$("preset"), "change");
  assert.match(p.text("calc-fits"), /^no/, "the large preset is beyond the lab's limit");
  assert.ok(p.$("calc-fits").className.includes("bad"));
});

test("editing the architecture switches to custom, and an invalid one is explained", { skip }, async () => {
  const p = await page();
  p.set("cfg-3", 3); // 3 heads do not divide the width
  assert.equal(p.$("preset").value, "custom");
  assert.equal(p.$("calc-invalid").hidden, false);
  assert.match(p.text("calc-invalid"), /\S/);
  assert.equal(p.text("calc-total"), "–");
  p.set("cfg-3", 4);
  assert.equal(p.$("calc-invalid").hidden, true);
  assert.notEqual(p.text("calc-total"), "–");
  // A value that is not a whole count is a malformed argument, shown instead of thrown.
  p.set("cfg-0", 12.5);
  assert.equal(p.$("calc-invalid").hidden, false);
  assert.match(p.text("calc-invalid"), /vocab_size/);
});

test("'use in Train' needs a valid model, then opens the Train tab", { skip }, async () => {
  const p = await page();
  p.set("cfg-3", 3);
  p.click("calc-use");
  assert.equal(p.$("error").hidden, false);
  assert.equal(p.$("tab-calc").getAttribute("aria-selected"), "true");
  p.set("cfg-3", 4);
  p.click("calc-use");
  assert.equal(p.$("error").hidden, true);
  assert.equal(p.$("tab-train").getAttribute("aria-selected"), "true");
  assert.equal(p.$("panel-train").hidden, false);
  assert.equal(p.$("panel-calc").hidden, true);
});

test("creating a model shows the task and enables training", { skip }, async () => {
  const p = await trainPage();
  assert.equal(p.$("train-step").disabled, true);
  p.click("train-create");
  assert.equal(p.$("error").hidden, true, p.text("error"));
  assert.equal(p.$("train-step").disabled, false);
  assert.equal(p.$("train-run").disabled, false);
  assert.equal(p.$("train-pause").disabled, true);
  assert.equal(p.text("r-step"), "0");
  assert.match(p.text("r-acc"), /^\d+\.\d%$/);
  const rows = p.document.querySelectorAll("#task-table tbody tr");
  assert.equal(rows.length, 2 * 3, "input, target and predicted for each of 2 sequences");
  assert.equal(p.document.querySelectorAll("#task-table thead th").length, 12 + 1);
  assert.equal(p.$("attn-seq").options.length, 2);
  assert.equal(p.$("attn-layer").options.length, 2);
});

test("stepping advances the readouts and draws the curves", { skip }, async () => {
  const p = await trainPage();
  p.click("train-create");
  p.click("train-step");
  assert.equal(p.text("r-step"), "10");
  assert.notEqual(p.text("r-loss"), "–");
  assert.notEqual(p.text("r-grad"), "–");
  assert.notEqual(p.text("r-ms"), "–");
  assert.ok(count(p.ctxOf("chart-loss"), "stroke") > 3, "the loss curve was stroked");
  assert.match(p.$("chart-loss").getAttribute("aria-label"), /Loss went from/);
  assert.ok(count(p.ctxOf("chart-grads"), "fillRect") > 18, "a bar per parameter");
  p.click("train-step");
  assert.equal(p.text("r-step"), "20");
});

test("running to the end memorizes the task and says so", { skip }, async () => {
  const p = await trainPage();
  p.click("train-create");
  p.click("train-run");
  assert.equal(p.text("r-step"), "150");
  assert.equal(p.text("r-acc"), "100.0%");
  const predicted = [...p.document.querySelectorAll("#task-table tbody tr")].filter((r) =>
    r.firstChild.textContent.endsWith("predicted"),
  );
  assert.equal(predicted.length, 2);
  for (const row of predicted) {
    const cells = [...row.querySelectorAll("td")].slice(1);
    assert.equal(cells.length, 12);
    assert.ok(cells.every((c) => c.className === "right"), "every prediction is right");
  }
  assert.equal(p.$("train-step").disabled, false, "buttons come back when the run is over");
  assert.equal(p.$("train-pause").disabled, true);
});

test("a run can be paused between chunks", { skip }, async () => {
  const queue = [];
  const p = await trainPage({ queue });
  p.click("train-create");
  p.click("train-run");
  assert.equal(p.$("train-pause").disabled, false);
  assert.equal(p.$("train-create").disabled, true, "no new model mid-run");
  queue.shift()(); // one chunk
  assert.equal(p.text("r-step"), "10");
  p.click("train-pause");
  assert.equal(p.api.state.running, false);
  while (queue.length) queue.shift()(); // anything still scheduled must do nothing
  assert.equal(p.text("r-step"), "10");
  assert.equal(p.$("train-step").disabled, false);
});

test("creating a model again frees the old one", { skip }, async () => {
  const p = await trainPage();
  p.click("train-create");
  const first = p.api.state.lab.handle;
  p.click("train-step");
  p.click("train-create");
  assert.notEqual(p.api.state.lab, null);
  assert.equal(p.text("r-step"), "0", "history starts over");
  assert.ok(p.engine.x.midman_lab_stats(first) < 0 || p.api.state.lab.handle === first, "the old handle is gone");
});

test("a bad request is shown as a message and nothing breaks", { skip }, async () => {
  const p = await trainPage();
  p.set("seq", 17); // longer than max_seq_len 16
  p.click("train-create");
  assert.equal(p.$("error").hidden, false);
  assert.match(p.text("error"), /max_seq_len/);
  assert.equal(p.$("train-step").disabled, true);
  p.set("seq", 12);
  p.click("train-create");
  assert.equal(p.$("error").hidden, true, "the next good action clears the message");
  assert.equal(p.$("train-step").disabled, false);
});

test("training that blows up stops, says why, and leaves the page usable", { skip }, async () => {
  const p = await trainPage();
  p.set("max-lr", 1e30);
  p.set("min-lr", 1e30);
  p.set("warmup", 0);
  p.click("train-create");
  p.click("train-run");
  assert.equal(p.$("error").hidden, false);
  assert.match(p.text("error"), /Training stopped/);
  assert.equal(p.api.state.running, false);
  assert.equal(p.$("train-pause").disabled, true);
  assert.equal(p.$("train-step").disabled, false);
});

test("the attention tab shows one map per head and lets you pick one", { skip }, async () => {
  const p = await trainPage();
  p.click("train-create");
  p.click("train-step");
  p.click("tab-attn");
  assert.equal(p.$("panel-attn").hidden, false);
  const heads = p.document.querySelectorAll("#attn-heads button");
  assert.equal(heads.length, 2, "this model has two heads");
  assert.equal(heads[0].getAttribute("aria-pressed"), "true");
  // 12 x 12 cells, lower triangle only: 78 filled, plus one background fill.
  assert.equal(count(p.ctxOf("attn-main"), "fillRect") - 1, 78);
  heads[1].dispatchEvent(new p.window.MouseEvent("click", { bubbles: true }));
  const after = p.document.querySelectorAll("#attn-heads button");
  assert.equal(after[1].getAttribute("aria-pressed"), "true");
  assert.equal(after[0].getAttribute("aria-pressed"), "false");
  p.$("attn-layer").value = "1";
  p.fire(p.$("attn-layer"), "change");
  assert.equal(p.$("error").hidden, true, p.text("error"));
});

test("the attention tab says nothing before there is a model, and does not crash", { skip }, async () => {
  const p = await page();
  p.click("tab-attn");
  assert.equal(p.document.querySelectorAll("#attn-heads button").length, 0);
  assert.equal(p.$("error").hidden, true);
});

test("the RoPE tab draws the real table and rejects a bad head size", { skip }, async () => {
  const p = await page();
  p.click("tab-rope");
  // 64 positions x 8 channel pairs, plus the background fill.
  assert.equal(count(p.ctxOf("rope-map"), "fillRect") - 1, 64 * 8);
  assert.ok(count(p.ctxOf("rope-lines"), "stroke") > 4);
  p.set("rope-dim", 7);
  assert.equal(p.$("error").hidden, false);
  const before = count(p.ctxOf("rope-map"), "fillRect");
  p.set("rope-dim", 32);
  assert.equal(p.$("error").hidden, true);
  // The same context is reused, so count only what this redraw added: 64 x 16 cells + a background.
  assert.equal(count(p.ctxOf("rope-map"), "fillRect") - before, 64 * 16 + 1);
});

test("the embeddings tab plots every token once a model exists", { skip }, async () => {
  const p = await trainPage();
  p.click("tab-emb");
  assert.equal(count(p.ctxOf("emb-plot"), "arc"), 0, "nothing to plot yet");
  p.click("tab-train");
  p.click("train-create");
  p.click("tab-emb");
  assert.equal(count(p.ctxOf("emb-plot"), "arc"), 12, "one dot per token of the 12-token vocabulary");
  const labels = p.ctxOf("emb-plot").calls.filter((c) => c[0] === "fillText").map((c) => c[1]);
  assert.ok(labels.some((l) => l.includes("Token embeddings")));
});

test("a missing element is reported by name instead of a vague failure", { skip }, async () => {
  const dom = new JSDOM(html.replace('id="status"', 'id="oops"'), { pretendToBeVisual: true });
  dom.window.HTMLCanvasElement.prototype.getContext = () => recorder();
  const engine = await Engine.load(bytes);
  await assert.rejects(() => init(dom.window.document, engine), /missing #status/);
});
