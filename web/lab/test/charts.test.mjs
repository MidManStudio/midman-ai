// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/test/charts.test.mjs"
// ============================================================================
// Tests of charts.js: the numeric helpers directly, and the drawing functions against
// a recording canvas context. Run with: node --test web/lab/test/charts.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { barChart, formatNumber, heatmap, lineChart, niceTicks, pca2, scatter, viridis } from "../charts.js";
import { count, recorder } from "./support.mjs";

const SIZE = { width: 400, height: 240 };

test("ticks are round numbers that cover the range", () => {
  assert.deepEqual(niceTicks(0, 10, 6), [0, 2, 4, 6, 8, 10]);
  assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  const t = niceTicks(0.0013, 2.71, 5);
  assert.ok(t[0] >= 0.0013 && t.at(-1) <= 2.71 + 1e-9 && t.length >= 3);
  assert.deepEqual(niceTicks(5, 5), [5]);
  assert.deepEqual(niceTicks(NaN, 1), []);
});

test("the color map runs from dark purple to yellow and clamps", () => {
  assert.deepEqual(viridis(0), [68, 1, 84]);
  assert.deepEqual(viridis(1), [253, 231, 37]);
  assert.deepEqual(viridis(-3), viridis(0));
  assert.deepEqual(viridis(7), viridis(1));
  assert.deepEqual(viridis(NaN), viridis(0));
  const mid = viridis(0.5);
  assert.ok(mid[1] > 100 && mid[1] < 160, `the middle is teal-green: ${mid}`);
});

test("numbers are formatted compactly", () => {
  assert.equal(formatNumber(0), "0");
  assert.equal(formatNumber(2.4935), "2.49");
  assert.equal(formatNumber(0.0006), "6.0e-4");
  assert.equal(formatNumber(123456), "1.2e+5");
  assert.equal(formatNumber(NaN), "NaN");
});

test("PCA finds the direction a cloud is stretched along", () => {
  // Points on the line y = 2x in 3D (z = 0) plus tiny noise: one dominant direction.
  const rows = 20;
  const data = new Float32Array(rows * 3);
  for (let r = 0; r < rows; r++) {
    const t = r - 10;
    data.set([t, 2 * t + (r % 2) * 0.01, 0], r * 3);
  }
  const { points, explained } = pca2(data, rows, 3);
  assert.equal(points.length, rows);
  assert.ok(explained[0] > 0.999, `the first component explains almost everything: ${explained}`);
  assert.ok(explained[1] < 0.001);
  // The first coordinate is monotonic in t (up to sign).
  const xs = points.map((p) => p[0]);
  const increasing = xs.every((v, i) => i === 0 || v > xs[i - 1]);
  const decreasing = xs.every((v, i) => i === 0 || v < xs[i - 1]);
  assert.ok(increasing || decreasing);
});

test("PCA orders components by variance and keeps them uncorrelated", () => {
  const rows = 60;
  const data = new Float32Array(rows * 4);
  for (let r = 0; r < rows; r++) {
    data.set([5 * Math.sin(r), 2 * Math.cos(r * 1.3), 0.3 * Math.sin(r * 2.1), 0.1 * Math.cos(r * 0.7)], r * 4);
  }
  const { points, explained } = pca2(data, rows, 4);
  assert.ok(explained[0] >= explained[1] && explained[0] + explained[1] <= 1 + 1e-9);
  const mean = (i) => points.reduce((s, p) => s + p[i], 0) / rows;
  const cov = points.reduce((s, p) => s + (p[0] - mean(0)) * (p[1] - mean(1)), 0) / (rows - 1);
  const v0 = points.reduce((s, p) => s + (p[0] - mean(0)) ** 2, 0) / (rows - 1);
  assert.ok(Math.abs(cov) < 1e-6 * v0, `components are uncorrelated: ${cov}`);
});

test("PCA copes with degenerate input", () => {
  assert.deepEqual(pca2(new Float32Array(3), 1, 3).points, [[0, 0]]);
  const same = new Float32Array(10 * 2).fill(3);
  const r = pca2(same, 10, 2);
  assert.ok(r.points.every((p) => p.every(Number.isFinite)));
  assert.deepEqual(r.explained, [0, 0]);
});

test("a line chart draws one stroke per series and one point per value", () => {
  const ctx = recorder();
  const info = lineChart(ctx, SIZE, [{ name: "loss", values: [3, 2, 1.5, 1, 0.5] }], { title: "Loss" });
  assert.equal(info.points, 5);
  assert.ok(info.min < 0.5 && info.max > 3);
  assert.equal(count(ctx, "moveTo") - niceTicksCount(ctx), 1, "one line starts the series");
  assert.equal(count(ctx, "lineTo") - niceTicksCount(ctx), 4, "and four segments follow");
});

function niceTicksCount(ctx) {
  // gridlines are drawn as moveTo+lineTo pairs before the series
  return ctx.calls.filter((c) => c[0] === "stroke").length - 1;
}

test("a log axis skips values that have no logarithm and says so when nothing is left", () => {
  const ctx = recorder();
  const info = lineChart(ctx, SIZE, [{ name: "x", values: [0, -1, 10, 100, NaN] }], { logY: true });
  assert.ok(info.min <= 10 && info.max >= 100);
  const empty = recorder();
  assert.equal(lineChart(empty, SIZE, [{ name: "x", values: [0, -2] }], { logY: true }), null);
  assert.ok(empty.calls.some((c) => c[0] === "fillText" && c[1] === "no data yet"));
  const none = recorder();
  assert.equal(lineChart(none, SIZE, [], {}), null);
});

test("a flat series still gets a visible axis", () => {
  const ctx = recorder();
  const info = lineChart(ctx, SIZE, [{ name: "c", values: [1, 1, 1] }]);
  assert.ok(info.max > info.min);
});

test("a line chart breaks the line at gaps instead of drawing through them", () => {
  const ctx = recorder();
  lineChart(ctx, SIZE, [{ name: "g", values: [1, 2, NaN, 3, 4] }]);
  const strokesAfterGrid = ctx.calls.filter((c) => c[0] === "moveTo").length;
  const gridLines = ctx.calls.filter((c) => c[0] === "stroke").length - 1;
  assert.equal(strokesAfterGrid - gridLines, 2, "two separate runs");
});

test("a heat map fills one cell per value, or half with the upper triangle masked", () => {
  const n = 6;
  const matrix = new Float32Array(n * n).map((_, i) => (i % n) / n);
  const full = recorder();
  const info = heatmap(full, SIZE, matrix, n, n, {});
  assert.equal(count(full, "fillRect") - 1, n * n, "one background fill plus the cells");
  assert.ok(info.cell >= 2);
  const masked = recorder();
  heatmap(masked, SIZE, matrix, n, n, { maskAbove: true });
  assert.equal(count(masked, "fillRect") - 1, (n * (n + 1)) / 2);
});

test("a heat map labels its axes and handles an all-zero matrix", () => {
  const ctx = recorder();
  const labels = ["a", "b", "c"];
  heatmap(ctx, SIZE, new Float32Array(9), 3, 3, { rowLabels: labels, colLabels: labels });
  const drawn = ctx.calls.filter((c) => c[0] === "fillText").map((c) => c[1]);
  for (const l of labels) assert.ok(drawn.includes(l), `${l} is labelled`);
});

test("a bar chart draws one bar and two labels per value", () => {
  const ctx = recorder();
  const info = barChart(ctx, SIZE, ["a", "b", "c"], [1, 2, 4], { title: "t" });
  assert.equal(info.max, 4);
  assert.equal(count(ctx, "fillRect") - 1, 3);
  const widths = ctx.calls.filter((c) => c[0] === "fillRect").slice(1).map((c) => c[3]);
  assert.ok(widths[0] < widths[1] && widths[1] < widths[2], `bars scale with values: ${widths}`);
  assert.equal(barChart(recorder(), SIZE, [], []).max, 0);
});

test("a scatter plot draws and labels every point, even when they coincide", () => {
  const ctx = recorder();
  scatter(ctx, SIZE, [[0, 0], [1, 1], [2, 0]], ["x", "y", "z"], {});
  assert.equal(count(ctx, "arc"), 3);
  const same = recorder();
  scatter(same, SIZE, [[1, 1], [1, 1]], ["p", "q"], {});
  assert.equal(count(same, "arc"), 2);
  assert.equal(scatter(recorder(), SIZE, [], [], {}), null);
});
