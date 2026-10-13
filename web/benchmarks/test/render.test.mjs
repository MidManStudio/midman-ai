// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "benchmarks/test/render.test.mjs"
// ============================================================================
// Renders real bench-results.json output into the real page in a headless DOM. Run with:
//   node --test web/benchmarks/test/render.test.mjs   (jsdom: npm install --no-save jsdom)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { badge, formatTime, render, renderMissing } from "../render.js";

let JSDOM;
try {
  ({ JSDOM } = await import(process.env.JSDOM_PATH ?? "jsdom"));
} catch (e) {
  if (process.env.REQUIRE_DOM === "1") throw e;
}
const skip = JSDOM ? false : "jsdom is not installed";
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const data = JSON.parse(readFileSync(new URL("./fixtures/bench-results.json", import.meta.url), "utf8"));
const doc = () => new JSDOM(html).window.document;

test("times get a sensible unit", () => {
  assert.equal(formatTime(12.5), "12.50 ns");
  assert.equal(formatTime(4403), "4.403 µs");
  assert.equal(formatTime(2_306_000), "2.306 ms");
  assert.equal(formatTime(3.2e9), "3.200 s");
});

test("badges use the thresholds bench_summary.py uses", () => {
  assert.deepEqual(badge(0.5), { text: "0.50x (faster)", level: "ok" });
  assert.equal(badge(1.0).level, "ok");
  assert.equal(badge(1.05).level, "ok");
  assert.equal(badge(1.3).level, "warn");
  assert.equal(badge(3).level, "bad");
  assert.match(badge(8.3).text, /8\.3x slower/);
});

test("every group and benchmark of a real run is shown", { skip }, () => {
  const d = doc();
  render(d, data);
  const sections = d.querySelectorAll("#groups section.group");
  assert.equal(sections.length, data.groups.length);
  const rows = [...d.querySelectorAll("#groups tr")].filter((r) => r.querySelector("td"));
  assert.equal(rows.length, data.groups.reduce((n, g) => n + g.rows.length, 0));
  assert.match(d.getElementById("summary").textContent, /51 benchmarks in 26 groups/);
  assert.match(d.getElementById("summary").textContent, /passed/);
});

test("a baseline group is badged and a unit group is a plain multiple", { skip }, () => {
  const d = doc();
  render(d, data);
  const byName = (name) => [...d.querySelectorAll("section.group")].find((s) => s.querySelector("h2").textContent === name);
  const rms = byName("rms_norm_512x64");
  assert.ok(rms.querySelector("td.ratio.bad"), "the slow RMSNorm is flagged");
  assert.match(rms.textContent, /slower|x/);
  const step = byName("train_step_smoke_b4_t32");
  assert.ok(!step.querySelector("td.ratio.ok, td.ratio.warn, td.ratio.bad"), "a unit group never gets a badge");
  assert.match(step.textContent, /\(unit\)/);
  assert.match(step.textContent, /x the unit variant/);
  const matmul = byName("matmul_square_192");
  assert.ok(matmul.querySelector("td.ratio.ok"));
  assert.match(matmul.textContent, /\(faster\)/);
});

test("the baseline row is marked, and every bar has a width", { skip }, () => {
  const d = doc();
  render(d, data);
  const baselines = d.querySelectorAll("span.bar.baseline");
  assert.equal(baselines.length, data.groups.filter((g) => g.kind).length);
  for (const bar of d.querySelectorAll("span.bar")) assert.match(bar.style.width, /^[\d.]+%$/);
});

test("rendering again replaces the old content", { skip }, () => {
  const d = doc();
  render(d, data);
  render(d, data);
  assert.equal(d.querySelectorAll("#groups section.group").length, data.groups.length);
});

test("diagnostics and the note are shown when present", { skip }, () => {
  const d = doc();
  const withExtras = structuredClone(data);
  withExtras.diagnostics = [{ kind: "unable to complete samples", count: 1 }];
  withExtras.meta.note = "after the fix";
  render(d, withExtras);
  assert.match(d.getElementById("groups").textContent, /unable to complete samples: 1/);
  assert.match(d.getElementById("summary").textContent, /after the fix/);
});

test("the missing-data message is shown instead of an empty page", { skip }, () => {
  const d = doc();
  renderMissing(d, "No benchmark results are published yet.");
  assert.match(d.getElementById("summary").textContent, /No benchmark results/);
});
