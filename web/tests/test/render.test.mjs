// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "tests/test/render.test.mjs"
// ============================================================================
// Renders real ci-results.json output (and the two smoke-test results) into the real page in a
// headless DOM. Run with: node --test web/tests/test/render.test.mjs   (jsdom: npm install --no-save jsdom)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { renderCi, renderData, renderMissing, renderSmoke } from "../render.js";

let JSDOM;
try {
  ({ JSDOM } = await import(process.env.JSDOM_PATH ?? "jsdom"));
} catch (e) {
  if (process.env.REQUIRE_DOM === "1") throw e;
}
const skip = JSDOM ? false : "jsdom is not installed";
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const ci = JSON.parse(readFileSync(new URL("./fixtures/ci-results.json", import.meta.url), "utf8"));
const dataSmoke = JSON.parse(readFileSync(new URL("./fixtures/data-smoke-results.json", import.meta.url), "utf8"));
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

test("a real data smoke result shows the pipeline numbers, the digests and the gitleaks comparison", { skip }, () => {
  const d = doc();
  assert.equal(d.getElementById("data").hidden, true, "hidden until there is a result");
  renderData(d, dataSmoke);
  const box = d.getElementById("data");
  assert.equal(box.hidden, false);
  const text = box.textContent;
  assert.match(text, /Data pipeline smoke test, build #7/);
  assert.match(text, /pass on ubuntu-latest \(Linux X64\)/);
  assert.match(text, /256 tracked text files/);
  assert.match(text, /1,024 ids, 760 merges/);
  assert.match(text, /713,006 \(2\.25 bytes per token\)/);
  assert.match(text, /4, 5,568 sequences of 128 tokens/);
  assert.match(text, /Scrubber alone72\.1 MB\/s/);
  assert.match(text, /All 4 shards \(SHA-256\)/);
  assert.match(text, /f4fdebabe120/, "the digest of the first batches, cut to 12 digits");
  assert.match(text, /found 288 of 288 samples of 24 kinds, 0 findings outside them/);
  assert.match(text, /Gitleaks found fewer samples of: /);
  for (const miss of ["url-password (0 of 12)", "aws-secret (11 of 12)", "generic-quoted (11 of 12)"]) assert.ok(text.includes(miss), miss);
  assert.equal(box.querySelectorAll(".good").length, 2, "the status line and the comparison line");
  assert.equal(box.querySelectorAll(".bad").length, 0);
  assert.equal(box.querySelectorAll("tr").length, 1 + 12, "a header and twelve rows");
});

test("a failed data smoke run says FAIL and marks what went wrong", { skip }, () => {
  const d = doc();
  const bad = structuredClone(dataSmoke);
  bad.ok = false;
  bad.meta.note = "after the scrubber change";
  bad.oracle.ok = false;
  bad.oracle.ours_outside = 3;
  renderData(d, bad);
  const box = d.getElementById("data");
  assert.match(box.textContent, /FAIL on ubuntu-latest/);
  assert.match(box.textContent, /Note: after the scrubber change/);
  assert.match(box.textContent, /3 findings outside them/);
  assert.equal(box.querySelectorAll(".bad").length, 2);
});

test("without the gitleaks comparison there is no comparison line", { skip }, () => {
  const d = doc();
  const noOracle = structuredClone(dataSmoke);
  noOracle.oracle = null;
  renderData(d, noOracle);
  assert.doesNotMatch(d.getElementById("data").textContent, /gitleaks/);
  assert.equal(d.getElementById("data").querySelectorAll(".good").length, 1);
});

test("an incomplete data smoke result still renders and says so", { skip }, () => {
  const d = doc();
  const partial = { meta: { build: "3", runner: "macos-latest (macOS ARM64)", rust_version: "", commit: "abc" }, ok: false, train: null, shards: null, oracle: null, digests: {} };
  renderData(d, partial);
  const text = d.getElementById("data").textContent;
  assert.match(text, /build #3/);
  assert.match(text, /FAIL on macos-latest/);
  assert.match(text, /The run is incomplete/);
  assert.equal(d.querySelectorAll("#data table").length, 0);
});

test("the page has a hidden container for the data smoke test and says what it checks", { skip }, () => {
  const d = doc();
  assert.equal(d.getElementById("data").hidden, true);
  assert.match(d.querySelector(".lede").textContent, /data smoke test runs the data path/);
});

test("the missing-data message is shown instead of an empty page", { skip }, () => {
  const d = doc();
  renderMissing(d, "No CI results are published yet.");
  assert.match(d.getElementById("ci").textContent, /No CI results/);
  assert.equal(d.getElementById("suites").hidden, true);
  assert.equal(d.getElementById("smoke").hidden, true);
});
