// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/charts.js"
// ============================================================================
// Canvas charts for the lab. Every drawing function takes a 2D context and a size
// and draws onto it, so the same code runs against a real canvas and against the
// recording context the tests use. The numeric helpers (ticks, color map, PCA) are
// pure functions.

export const THEME = {
  background: "#0e1116",
  foreground: "#d7dde5",
  muted: "#7d8794",
  grid: "#232a33",
  series: ["#5ec2ff", "#ffb454", "#8ddb8c", "#ff7a90", "#c79bff"],
  font: "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
};

// ----------------------------------------------------------------- numerics

/** Round tick positions covering [min, max], about `count` of them. */
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const rough = (max - min) / Math.max(count - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / magnitude;
  const step = (fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10) * magnitude;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
    ticks.push(Number(t.toPrecision(12)));
  }
  return ticks;
}

// Nine samples of the viridis color map, interpolated linearly.
const VIRIDIS = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [253, 231, 37],
];

/** An `[r, g, b]` color for `t` in [0, 1] (clamped). */
export function viridis(t) {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (VIRIDIS.length - 1);
  const i = Math.min(Math.floor(x), VIRIDIS.length - 2);
  const f = x - i;
  return VIRIDIS[i].map((a, k) => Math.round(a + (VIRIDIS[i + 1][k] - a) * f));
}

/** Formats a number compactly for an axis or a readout. */
export function formatNumber(v) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
  return Number(v.toPrecision(3)).toString();
}

/**
 * Projects the rows of a `rows x cols` matrix onto its top two principal
 * components, by power iteration on the covariance. Deterministic.
 * Returns `{ points: [[x, y], ...], explained: [share1, share2] }`.
 */
export function pca2(data, rows, cols) {
  if (rows < 2 || cols < 1) return { points: Array.from({ length: rows }, () => [0, 0]), explained: [0, 0] };
  const mean = new Float64Array(cols);
  // Sum first and divide once: the mean of identical values is then exact, so a cloud with
  // no spread has exactly zero variance instead of rounding noise.
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) mean[c] += data[r * cols + c];
  for (let c = 0; c < cols; c++) mean[c] /= rows;
  const x = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) x[r * cols + c] = data[r * cols + c] - mean[c];

  const cov = new Float64Array(cols * cols);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < cols; i++) {
      const xi = x[r * cols + i];
      for (let j = i; j < cols; j++) cov[i * cols + j] += (xi * x[r * cols + j]) / (rows - 1);
    }
  }
  for (let i = 0; i < cols; i++) for (let j = 0; j < i; j++) cov[i * cols + j] = cov[j * cols + i];

  let total = 0;
  for (let i = 0; i < cols; i++) total += cov[i * cols + i];

  const components = [];
  const variances = [];
  for (let k = 0; k < 2; k++) {
    let v = new Float64Array(cols).map((_, i) => Math.sin(i * 1.7 + 1 + k));
    let eigenvalue = 0;
    for (let iteration = 0; iteration < 200; iteration++) {
      const next = new Float64Array(cols);
      for (let i = 0; i < cols; i++) for (let j = 0; j < cols; j++) next[i] += cov[i * cols + j] * v[j];
      for (const u of components) {
        let dot = 0;
        for (let i = 0; i < cols; i++) dot += next[i] * u[i];
        for (let i = 0; i < cols; i++) next[i] -= dot * u[i];
      }
      let norm = 0;
      for (let i = 0; i < cols; i++) norm += next[i] * next[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-300) break;
      for (let i = 0; i < cols; i++) next[i] /= norm;
      eigenvalue = norm;
      const change = next.reduce((s, a, i) => s + Math.abs(a - v[i]), 0);
      v = next;
      if (change < 1e-12) break;
    }
    components.push(v);
    variances.push(eigenvalue);
  }
  const points = [];
  for (let r = 0; r < rows; r++) {
    const p = components.map((u) => {
      let s = 0;
      for (let c = 0; c < cols; c++) s += x[r * cols + c] * u[c];
      return s;
    });
    points.push(p);
  }
  return { points, explained: variances.map((v) => (total > 0 ? v / total : 0)) };
}

// ----------------------------------------------------------------- drawing

function clear(ctx, { width, height }, theme) {
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);
  ctx.font = theme.font;
}

function text(ctx, theme, string, x, y, { align = "left", color = theme.muted, baseline = "alphabetic" } = {}) {
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(string, x, y);
}

/**
 * Line chart of one or more series sharing an x axis (the index of each value).
 * `series` is `[{ name, values, color? }]`. With `logY`, non-positive values are
 * skipped. Returns the plotted bounds, which is handy for tests.
 */
export function lineChart(ctx, size, series, options = {}, theme = THEME) {
  const { logY = false, title = "", yLabel = "" } = options;
  const { width, height } = size;
  clear(ctx, size, theme);
  const pad = { left: 56, right: 14, top: title ? 26 : 12, bottom: 22 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  if (title) text(ctx, theme, title, pad.left, 16, { color: theme.foreground });

  const transform = logY ? (v) => Math.log10(v) : (v) => v;
  const usable = (v) => Number.isFinite(v) && (!logY || v > 0);
  let lo = Infinity;
  let hi = -Infinity;
  let n = 0;
  for (const s of series) {
    n = Math.max(n, s.values.length);
    for (const v of s.values) {
      if (usable(v)) {
        lo = Math.min(lo, transform(v));
        hi = Math.max(hi, transform(v));
      }
    }
  }
  if (!Number.isFinite(lo) || n === 0) {
    text(ctx, theme, "no data yet", pad.left + w / 2, pad.top + h / 2, { align: "center" });
    return null;
  }
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const span = hi - lo;
  lo -= span * 0.05;
  hi += span * 0.05;
  const x = (i) => pad.left + (n <= 1 ? w / 2 : (i / (n - 1)) * w);
  const y = (v) => pad.top + h - ((transform(v) - lo) / (hi - lo)) * h;

  const ticks = niceTicks(lo, hi, 5);
  ctx.lineWidth = 1;
  for (const t of ticks) {
    const py = pad.top + h - ((t - lo) / (hi - lo)) * h;
    ctx.strokeStyle = theme.grid;
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(pad.left + w, py);
    ctx.stroke();
    text(ctx, theme, formatNumber(logY ? 10 ** t : t), pad.left - 6, py, { align: "right", baseline: "middle" });
  }
  text(ctx, theme, "0", pad.left, height - 6);
  text(ctx, theme, String(Math.max(n - 1, 0)), pad.left + w, height - 6, { align: "right" });
  if (yLabel) text(ctx, theme, yLabel, 4, 16);

  series.forEach((s, k) => {
    ctx.strokeStyle = s.color ?? theme.series[k % theme.series.length];
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    let pen = false;
    s.values.forEach((v, i) => {
      if (!usable(v)) {
        pen = false;
        return;
      }
      if (pen) ctx.lineTo(x(i), y(v));
      else ctx.moveTo(x(i), y(v));
      pen = true;
    });
    ctx.stroke();
  });
  if (series.length > 1) {
    series.forEach((s, k) => {
      text(ctx, theme, s.name, pad.left + 8 + k * 90, pad.top + 12, {
        color: s.color ?? theme.series[k % theme.series.length],
      });
    });
  }
  return { min: logY ? 10 ** lo : lo, max: logY ? 10 ** hi : hi, points: n };
}

/**
 * Heat map of a `rows x cols` matrix (row-major). Values are colored 0 to
 * `options.max` (default the largest value). `rowLabels` and `colLabels` name the
 * axes when there is room. With `maskAbove`, cells above the diagonal are left empty.
 */
export function heatmap(ctx, size, matrix, rows, cols, options = {}, theme = THEME) {
  const { rowLabels = null, colLabels = null, maskAbove = false, title = "" } = options;
  const { width, height } = size;
  clear(ctx, size, theme);
  const pad = { left: rowLabels ? 34 : 8, top: (title ? 22 : 8) + (colLabels ? 16 : 0), right: 8, bottom: 8 };
  const cell = Math.max(2, Math.floor(Math.min((width - pad.left - pad.right) / cols, (height - pad.top - pad.bottom) / rows)));
  if (title) text(ctx, theme, title, 8, 14, { color: theme.foreground });
  let max = options.max;
  if (max === undefined) {
    max = 0;
    for (const v of matrix) if (Number.isFinite(v) && v > max) max = v;
  }
  const scale = max > 0 ? 1 / max : 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (maskAbove && c > r) continue;
      const [red, green, blue] = viridis(matrix[r * cols + c] * scale);
      ctx.fillStyle = `rgb(${red},${green},${blue})`;
      ctx.fillRect(pad.left + c * cell, pad.top + r * cell, cell, cell);
    }
  }
  const labelEvery = cell >= 14 ? 1 : Math.ceil(14 / cell);
  if (rowLabels) {
    for (let r = 0; r < rows; r += labelEvery) {
      text(ctx, theme, String(rowLabels[r]), pad.left - 4, pad.top + r * cell + cell / 2, { align: "right", baseline: "middle" });
    }
  }
  if (colLabels) {
    for (let c = 0; c < cols; c += labelEvery) {
      text(ctx, theme, String(colLabels[c]), pad.left + c * cell + cell / 2, pad.top - 4, { align: "center" });
    }
  }
  return { cell, max };
}

/** Horizontal bars: one per label, scaled to the largest value. */
export function barChart(ctx, size, labels, values, options = {}, theme = THEME) {
  const { title = "", color = theme.series[0] } = options;
  const { width, height } = size;
  clear(ctx, size, theme);
  const pad = { left: 190, right: 56, top: title ? 24 : 8, bottom: 8 };
  if (title) text(ctx, theme, title, 8, 15, { color: theme.foreground });
  const max = Math.max(0, ...values.filter(Number.isFinite));
  const row = (height - pad.top - pad.bottom) / Math.max(values.length, 1);
  const barHeight = Math.max(2, Math.min(row - 3, 18));
  values.forEach((v, i) => {
    const top = pad.top + i * row + (row - barHeight) / 2;
    const length = max > 0 && Number.isFinite(v) ? (v / max) * (width - pad.left - pad.right) : 0;
    ctx.fillStyle = color;
    ctx.fillRect(pad.left, top, Math.max(length, 0), barHeight);
    text(ctx, theme, labels[i], pad.left - 6, top + barHeight / 2, { align: "right", baseline: "middle" });
    text(ctx, theme, formatNumber(v), pad.left + length + 5, top + barHeight / 2, { baseline: "middle" });
  });
  return { max };
}

/** Scatter plot of 2D points, each labelled. */
export function scatter(ctx, size, points, labels, options = {}, theme = THEME) {
  const { title = "" } = options;
  const { width, height } = size;
  clear(ctx, size, theme);
  const pad = { left: 12, right: 12, top: title ? 26 : 12, bottom: 12 };
  if (title) text(ctx, theme, title, 12, 16, { color: theme.foreground });
  if (points.length === 0) return null;
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const [px, py] of points) {
    x0 = Math.min(x0, px);
    x1 = Math.max(x1, px);
    y0 = Math.min(y0, py);
    y1 = Math.max(y1, py);
  }
  const sx = x1 > x0 ? (width - pad.left - pad.right) / (x1 - x0) : 0;
  const sy = y1 > y0 ? (height - pad.top - pad.bottom) / (y1 - y0) : 0;
  points.forEach(([px, py], i) => {
    const cx = pad.left + (sx ? (px - x0) * sx : (width - pad.left - pad.right) / 2);
    const cy = height - pad.bottom - (sy ? (py - y0) * sy : (height - pad.top - pad.bottom) / 2);
    ctx.fillStyle = theme.series[i % theme.series.length];
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fill();
    text(ctx, theme, String(labels[i]), cx + 6, cy - 4, { color: theme.foreground });
  });
  return { x: [x0, x1], y: [y0, y1] };
}
