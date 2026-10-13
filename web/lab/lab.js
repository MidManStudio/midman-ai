// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/lab.js"
// ============================================================================
// The lab page. `init(document, engine)` wires every control to the engine and the
// charts, and returns its internals so a test can drive the same code a browser runs.
import { barChart, formatNumber, heatmap, lineChart, pca2, scatter } from "./charts.js";

const CONFIG_INPUTS = 10;

function formatCount(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} K`;
  return String(n);
}

export async function init(doc, engine, env = {}) {
  const win = env.win ?? doc.defaultView ?? globalThis;
  const now = env.now ?? (() => win.performance.now());
  const schedule = env.schedule ?? ((fn) => setTimeout(fn, 0));
  const info = engine.describe();

  const $ = (id) => {
    const el = doc.getElementById(id);
    if (!el) throw new Error(`the lab page is missing #${id}`);
    return el;
  };

  const state = {
    lab: null,
    trainConfig: null,
    history: { loss: [], gradNorm: [], lr: [] },
    task: null,
    running: false,
    msPerStep: null,
    head: 0,
    tab: "calc",
  };

  // ------------------------------------------------------------ plumbing

  function showError(message) {
    const banner = $("error");
    banner.hidden = false;
    banner.textContent = message;
  }

  function clearError() {
    const banner = $("error");
    banner.hidden = true;
    banner.textContent = "";
  }

  /** Wraps an event handler so an engine error becomes a visible message. */
  const guard = (fn) => (...args) => {
    try {
      clearError();
      return fn(...args);
    } catch (e) {
      showError(e.message);
      return undefined;
    }
  };

  /**
   * Sizes a canvas to its on-screen width at the device pixel ratio and returns its
   * context and logical size. The ratio between the width and height attributes is
   * kept, so the layout does not jump.
   */
  function canvasFor(id, height) {
    const canvas = $(id);
    canvas.dataset.w ??= canvas.getAttribute("width");
    canvas.dataset.h ??= canvas.getAttribute("height");
    const baseW = Number(canvas.dataset.w);
    const baseH = Number(canvas.dataset.h);
    const width = Math.max(280, canvas.clientWidth || baseW);
    const logicalHeight = height ?? Math.round((width * baseH) / baseW);
    const ratio = win.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(logicalHeight * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform?.(ratio, 0, 0, ratio, 0, 0);
    return { ctx, size: { width, height: logicalHeight }, canvas };
  }

  const num = (id, label) => {
    const value = Number($(id).value);
    if (!Number.isFinite(value)) throw new Error(`${label} is not a number`);
    return value;
  };

  // ---------------------------------------------------------------- tabs

  const TABS = ["calc", "train", "attn", "rope", "emb"];

  function switchTab(name) {
    state.tab = name;
    for (const t of TABS) {
      $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
      $(`panel-${t}`).hidden = t !== name;
    }
    refreshTab();
  }

  function refreshTab() {
    if (state.tab === "attn") refreshAttention();
    else if (state.tab === "rope") refreshRope();
    else if (state.tab === "emb") refreshEmbeddings();
    else if (state.tab === "train") refreshTrain();
  }

  // ---------------------------------------------------------- calculator

  function readConfig() {
    return Array.from({ length: CONFIG_INPUTS }, (_, i) =>
      i === CONFIG_INPUTS - 1 ? ($(`cfg-${i}`).checked ? 1 : 0) : Number($(`cfg-${i}`).value),
    );
  }

  function writeConfig(values) {
    values.forEach((v, i) => {
      if (i === CONFIG_INPUTS - 1) $(`cfg-${i}`).checked = v !== 0;
      else $(`cfg-${i}`).value = String(v);
    });
  }

  function fillPresets() {
    const select = $("preset");
    select.textContent = "";
    info.presets.forEach((p, i) => {
      const option = doc.createElement("option");
      option.value = String(i);
      option.textContent = `${p.name} (${formatCount(p.params)})`;
      select.append(option);
    });
    const custom = doc.createElement("option");
    custom.value = "custom";
    custom.textContent = "custom";
    select.append(custom);
  }

  function recompute() {
    const invalid = $("calc-invalid");
    const blank = () => {
      for (const id of ["calc-total", "calc-head", "calc-kv", "calc-fits"]) $(id).textContent = "–";
      const { ctx, size } = canvasFor("calc-bars");
      barChart(ctx, size, [], [], {});
    };
    let report;
    try {
      report = engine.configReport(readConfig());
    } catch (e) {
      invalid.hidden = false;
      invalid.textContent = e.message;
      blank();
      return null;
    }
    if (!report.valid) {
      invalid.hidden = false;
      invalid.textContent = report.error;
      blank();
      return report;
    }
    invalid.hidden = true;
    const total = report.params.total;
    $("calc-total").textContent = total.toLocaleString("en-US");
    $("calc-head").textContent = String(report.head_dim);
    $("calc-kv").textContent = String(report.kv_dim);
    const fits = total <= info.limits.max_params;
    $("calc-fits").textContent = fits ? "yes" : `no (limit ${formatCount(info.limits.max_params)})`;
    $("calc-fits").className = `value ${fits ? "good" : "bad"}`;
    const parts = ["embedding", "attention", "mlp", "norms", "lm_head"];
    const { ctx, size } = canvasFor("calc-bars");
    barChart(
      ctx,
      size,
      parts.map((p) => `${p} (${((report.params[p] / total) * 100).toFixed(1)}%)`),
      parts.map((p) => report.params[p]),
      { title: "Parameters by category" },
    );
    return report;
  }

  function onPresetChange() {
    const value = $("preset").value;
    if (value === "custom") return;
    writeConfig(engine.presetConfig(Number(value)));
    recompute();
  }

  function onConfigEdit() {
    $("preset").value = "custom";
    recompute();
  }

  function useInTrain() {
    const report = recompute();
    if (!report || !report.valid) throw new Error("fix the model configuration first");
    state.trainConfig = readConfig();
    switchTab("train");
  }

  // --------------------------------------------------------------- train

  function setTrainButtons() {
    const have = state.lab !== null;
    $("train-step").disabled = !have || state.running;
    $("train-run").disabled = !have || state.running;
    $("train-pause").disabled = !state.running;
    $("train-create").disabled = state.running;
  }

  function destroyLab() {
    if (state.lab) {
      state.lab.destroy();
      state.lab = null;
    }
  }

  function createLab() {
    destroyLab();
    state.history = { loss: [], gradNorm: [], lr: [] };
    state.msPerStep = null;
    const config = state.trainConfig ?? readConfig();
    const hyper = [
      num("max-lr", "peak learning rate"),
      num("min-lr", "final learning rate"),
      num("warmup", "warmup steps"),
      num("total", "total steps"),
      num("decay", "weight decay"),
      num("clip", "clip norm"),
      0.9,
      0.99,
    ];
    state.lab = engine.createLab({
      config,
      hyper,
      seed: BigInt(Math.trunc(num("seed", "seed"))),
      batch: num("batch", "sequences"),
      seq: num("seq", "tokens each"),
    });
    state.task = state.lab.eval();
    state.head = 0;
    fillAttentionSelects(config);
    setTrainButtons();
    refreshTrain();
  }

  function advance(steps) {
    if (!state.lab) return;
    const t0 = now();
    const result = state.lab.step(steps);
    const done = result.loss.length;
    if (done > 0) state.msPerStep = (now() - t0) / done;
    for (let i = 0; i < done; i++) {
      state.history.loss.push(result.loss[i]);
      state.history.gradNorm.push(result.gradNorm[i]);
      state.history.lr.push(result.lr[i]);
    }
    if (result.stopped) {
      state.running = false;
      showError(`Training stopped: ${result.stopped}`);
    }
    setTrainButtons();
    refreshTrain();
    if (state.tab !== "train") refreshTab();
  }

  function stepOnce() {
    advance(Math.max(1, Math.trunc(num("chunk", "steps per click"))));
  }

  function runToEnd() {
    if (!state.lab) return;
    state.running = true;
    setTrainButtons();
    const loop = () => {
      if (!state.running || !state.lab) return;
      const total = Math.trunc(num("total", "total steps"));
      const left = total - state.history.loss.length;
      if (left <= 0) {
        state.running = false;
        setTrainButtons();
        return;
      }
      advance(Math.min(left, Math.max(1, Math.trunc(num("chunk", "steps per click")))));
      if (state.running) schedule(guard(loop));
    };
    schedule(guard(loop));
  }

  function pause() {
    state.running = false;
    setTrainButtons();
  }

  function setReadout(id, text) {
    $(id).textContent = text;
  }

  function refreshTrain() {
    const h = state.history;
    const n = h.loss.length;
    setReadout("r-step", state.lab ? String(n) : "–");
    setReadout("r-loss", n ? formatNumber(h.loss[n - 1]) : "–");
    setReadout("r-grad", n ? formatNumber(h.gradNorm[n - 1]) : "–");
    setReadout("r-lr", n ? formatNumber(h.lr[n - 1]) : "–");
    setReadout("r-ms", state.msPerStep === null ? "–" : state.msPerStep.toFixed(1));

    let accuracy = "–";
    if (state.lab) {
      state.task = state.lab.eval();
      accuracy = `${(state.task.accuracy * 100).toFixed(1)}%`;
    }
    setReadout("r-acc", accuracy);

    let ctx;
    let size;
    ({ ctx, size } = canvasFor("chart-loss"));
    lineChart(
      ctx,
      size,
      [
        { name: "loss", values: Array.from(h.loss) },
        { name: "gradient norm", values: Array.from(h.gradNorm) },
      ],
      { logY: true, title: "Loss and gradient norm (log scale)" },
    );
    $("chart-loss").setAttribute(
      "aria-label",
      n
        ? `Loss went from ${formatNumber(h.loss[0])} to ${formatNumber(h.loss[n - 1])} over ${n} steps`
        : "No training steps yet",
    );
    ({ ctx, size } = canvasFor("chart-lr"));
    lineChart(ctx, size, [{ name: "learning rate", values: Array.from(h.lr) }], {
      title: "Learning rate",
    });
    drawGradients();
    renderTask();
  }

  function drawGradients() {
    const stats = state.lab ? state.lab.stats() : { params: [] };
    const rows = stats.params.length;
    const { ctx, size } = canvasFor("chart-grads", Math.max(120, rows * 22 + 34));
    barChart(
      ctx,
      size,
      stats.params.map((p) => p.name.replace(/\.weight$/, "")),
      stats.params.map((p) => p.grad_norm ?? 0),
      { title: state.history.loss.length ? "Gradient norm before clipping, last step" : "Run a step to see gradients" },
    );
  }

  function renderTask() {
    const table = $("task-table");
    table.replaceChildren();
    if (!state.task) return;
    const { inputs, targets, predictions } = state.task;
    const head = doc.createElement("thead");
    const headRow = doc.createElement("tr");
    headRow.append(doc.createElement("th"));
    (inputs[0] ?? []).forEach((_, i) => {
      const th = doc.createElement("th");
      th.textContent = String(i);
      headRow.append(th);
    });
    head.append(headRow);
    const body = doc.createElement("tbody");
    inputs.forEach((_, s) => {
      for (const [name, ids] of [["input", inputs[s]], ["target", targets[s]], ["predicted", predictions[s]]]) {
        const tr = doc.createElement("tr");
        const label = doc.createElement("td");
        label.className = "row-name";
        label.textContent = `${s}: ${name}`;
        tr.append(label);
        ids.forEach((id, i) => {
          const td = doc.createElement("td");
          td.textContent = String(id);
          if (name === "predicted") td.className = id === targets[s][i] ? "right" : "wrong";
          tr.append(td);
        });
        body.append(tr);
      }
    });
    table.append(head, body);
  }

  // ----------------------------------------------------------- attention

  function fillAttentionSelects(config) {
    const seqSelect = $("attn-seq");
    const layerSelect = $("attn-layer");
    seqSelect.textContent = "";
    layerSelect.textContent = "";
    const batch = state.task.batch;
    for (let i = 0; i < batch; i++) seqSelect.append(new win.Option(`sequence ${i}`, String(i)));
    for (let i = 0; i < config[2]; i++) layerSelect.append(new win.Option(`layer ${i}`, String(i)));
  }

  function refreshAttention() {
    const host = $("attn-heads");
    host.replaceChildren();
    if (!state.lab) {
      const { ctx, size } = canvasFor("attn-main");
      heatmap(ctx, size, new Float32Array(0), 0, 0, {});
      return;
    }
    const sequence = Number($("attn-seq").value || 0);
    const layer = Number($("attn-layer").value || 0);
    const maps = state.lab.attention(sequence);
    const t = maps.seq;
    const labels = state.task.inputs[sequence];
    const mapOf = (l, h) => {
      const start = ((l * maps.heads + h) * t) * t;
      return maps.data.subarray(start, start + t * t);
    };
    state.head = Math.min(state.head, maps.heads - 1);
    const { ctx, size } = canvasFor("attn-main");
    heatmap(ctx, size, mapOf(layer, state.head), t, t, {
      rowLabels: labels,
      colLabels: labels,
      maskAbove: true,
      max: 1,
      title: `layer ${layer}, head ${state.head}: row = reading position, column = position read`,
    });
    for (let h = 0; h < maps.heads; h++) {
      const button = doc.createElement("button");
      button.setAttribute("aria-label", `layer ${layer} head ${h}`);
      button.setAttribute("aria-pressed", String(h === state.head));
      const canvas = doc.createElement("canvas");
      canvas.id = `attn-head-${h}`;
      canvas.setAttribute("width", "150");
      canvas.setAttribute("height", "150");
      button.append(canvas);
      button.addEventListener("click", () => {
        state.head = h;
        refreshAttention();
      });
      host.append(button);
      const small = canvasFor(canvas.id);
      heatmap(small.ctx, small.size, mapOf(layer, h), t, t, { maskAbove: true, max: 1 });
    }
  }

  // ---------------------------------------------------------------- rope

  function refreshRope() {
    const dim = Math.trunc(num("rope-dim", "head size"));
    const len = Math.trunc(num("rope-len", "positions"));
    const theta = num("rope-theta", "theta");
    const table = engine.ropeTable(dim, len, theta);
    const grid = Float32Array.from(table.cos, (c) => (c + 1) / 2);
    let drawn = canvasFor("rope-map");
    heatmap(drawn.ctx, drawn.size, grid, table.positions, table.half, {
      max: 1,
      title: "cos(angle): yellow = 1, purple = -1",
    });
    const pairs = [...new Set([0, 1, table.half >> 1, table.half - 1])].filter((p) => p >= 0 && p < table.half);
    const series = pairs.map((p) => ({
      name: `pair ${p}`,
      values: Array.from({ length: table.positions }, (_, pos) => table.cos[pos * table.half + p]),
    }));
    drawn = canvasFor("rope-lines");
    lineChart(drawn.ctx, drawn.size, series, { title: "cos(angle) by position" });
  }

  // ----------------------------------------------------------- embeddings

  function refreshEmbeddings() {
    const drawn = canvasFor("emb-plot");
    if (!state.lab) {
      scatter(drawn.ctx, drawn.size, [], [], { title: "Create a model on the Train tab first" });
      return;
    }
    const e = state.lab.embedding();
    const { points, explained } = pca2(e.data, e.rows, e.cols);
    scatter(
      drawn.ctx,
      drawn.size,
      points,
      points.map((_, i) => i),
      {
        title: `Token embeddings, top two directions (${(explained[0] * 100).toFixed(0)}% and ${(explained[1] * 100).toFixed(0)}% of the variance), step ${state.history.loss.length}`,
      },
    );
  }

  // ---------------------------------------------------------------- wiring

  fillPresets();
  writeConfig(engine.presetConfig(0));
  $("preset").value = "0";
  $("status").textContent =
    `Engine ready: ABI ${engine.abiVersion}, ${engine.build} build, ${(engine.memoryBytes / 1048576).toFixed(1)} MiB of memory.`;

  $("preset").addEventListener("change", guard(onPresetChange));
  for (let i = 0; i < CONFIG_INPUTS; i++) $(`cfg-${i}`).addEventListener("input", guard(onConfigEdit));
  $("calc-use").addEventListener("click", guard(useInTrain));
  $("train-create").addEventListener("click", guard(createLab));
  $("train-step").addEventListener("click", guard(stepOnce));
  $("train-run").addEventListener("click", guard(runToEnd));
  $("train-pause").addEventListener("click", guard(pause));
  $("attn-refresh").addEventListener("click", guard(refreshAttention));
  $("attn-seq").addEventListener("change", guard(refreshAttention));
  $("attn-layer").addEventListener("change", guard(refreshAttention));
  $("emb-refresh").addEventListener("click", guard(refreshEmbeddings));
  for (const id of ["rope-dim", "rope-len", "rope-theta"]) $(id).addEventListener("input", guard(refreshRope));
  for (const t of TABS) $(`tab-${t}`).addEventListener("click", guard(() => switchTab(t)));

  recompute();
  setTrainButtons();

  return { state, switchTab, recompute, createLab, advance, runToEnd, pause, refreshAttention, refreshRope, refreshEmbeddings, readConfig, writeConfig };
}
