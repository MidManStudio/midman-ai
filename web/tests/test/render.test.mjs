// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "tests/test/render.test.mjs"
// ============================================================================
// Renders real ci-results.json output (and a smoke-test result) into the real page in a headless
// DOM. Run with: node --test web/tests/test/render.test.mjs   (jsdom: npm install --no-save jsdom)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { renderCi, renderMissing, renderSmoke } from "../render.js";

let JSDOM;
try {
  ({ JSDOM } = await import(process.env.JSDOM_PATH ?? "jsdom"));
} catch (e) {
  if (process.env.REQUIRE_DOM === "1") throw e;
}
const skip = JSDOM ? false : "jsdom is not installed";
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const ci = JSON.parse(readFileSync(new URL("./fixtures/ci-results.json", import.meta.url), "utf8"));
const doc = () => new JSDOM(html).window.document;

const SMOKE = {
  meta: { build: "2", runner: "ubuntu-latest (Linux X64)", rust_version: "rustc 1.99.0", commit: "74bfdf1609abcdef", note: "" },
  outcome: "success",
  runs: [{ first_loss: 5.5674, last_loss: 0.0005, accuracy: 1.0 }],
  tests: { status: "ok", passed: 1, failed: 0, ignored: 0 },
};

test("a real CI result shows every check and the test totals", { skip }, () => {
  const d = doc();
  renderCi(d, ci);
  const text = d.getElementById("ci").textContent;
  for (const check of ["fmt", "clippy", "build", "test"]) assert.match(text, new RegExp(check));
  assert.match(text, new RegExp(`${ci.test.passed} tests passed, 0 failed`));
  assert.match(text, /Clippy: 0 errors, 0 warnings/);
  assert.equal(d.querySelectorAll("#ci .good").length, 4);
});

test("every suite is listed, failures first", { skip }, () => {
  const d = doc();
  const bad = structuredClone(ci);
  bad.test.suites[3].failed = 2;
  bad.test.suites[3].failed_tests = ["a::b", "c::d"];
  renderCi(d, bad);
  const rows = [...d.querySelectorAll("#suites tr")].filter((r) => r.querySelector("td"));
  assert.equal(rows.length, ci.test.suites.length + 2, "a row per suite plus a row per failed test");
  assert.equal(d.getElementById("suites").hidden, false);
  assert.match(rows[0].textContent, new RegExp(bad.test.suites[3].name), "the failing suite comes first");
  assert.match(d.getElementById("suites").textContent, /failed: a::b/);
  assert.ok(d.querySelector("#suites td.bad"));
});

test("a failed check is marked as such", { skip }, () => {
  const d = doc();
  const bad = structuredClone(ci);
  bad.outcomes.clippy = "failure";
  renderCi(d, bad);
  assert.match(d.getElementById("ci").textContent, /FAIL/);
  assert.equal(d.querySelectorAll("#ci .bad").length, 1);
});

test("the smoke test shows its losses and accuracy", { skip }, () => {
  const d = doc();
  renderSmoke(d, SMOKE);
  const box = d.getElementById("smoke");
  assert.equal(box.hidden, false);
  assert.match(box.textContent, /5\.5674/);
  assert.match(box.textContent, /0\.0005/);
  assert.match(box.textContent, /100\.0%/);
  assert.match(box.textContent, /74bfdf1609/);
  assert.ok(box.querySelector("td.good"));
  const weak = structuredClone(SMOKE);
  weak.runs[0].accuracy = 0.5;
  renderSmoke(d, weak);
  assert.ok(box.querySelector("td.bad"), "low accuracy is flagged");
});

test("the missing-data message is shown instead of an empty page", { skip }, () => {
  const d = doc();
  renderMissing(d, "No CI results are published yet.");
  assert.match(d.getElementById("ci").textContent, /No CI results/);
  assert.equal(d.getElementById("suites").hidden, true);
  assert.equal(d.getElementById("smoke").hidden, true);
});
