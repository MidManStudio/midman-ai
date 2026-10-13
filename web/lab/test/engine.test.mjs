// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/test/engine.test.mjs"
// ============================================================================
// Tests of engine.js against the real WebAssembly build of midman-api. Run with:
//
//   MIDMAN_WASM=path/to/midman_api.wasm node --test web/lab/test/engine.test.mjs
//
// MIDMAN_WASM defaults to dist/lab/midman_api.wasm, where the site build puts it. Set
// MIDMAN_WASM_SIMD as well to also run the SIMD build through loadBest.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Engine, EngineError, simdSupported } from "../engine.js";

const wasmPath = process.env.MIDMAN_WASM ?? "dist/lab/midman_api.wasm";
const bytes = readFileSync(wasmPath);
const engine = await Engine.load(bytes);

const TINY = [12, 16, 2, 2, 1, 32, 16, 10000, 1e-5, 1];
const HYPER = [0.03, 0.003, 15, 150, 0.01, 1.0, 0.9, 0.99];

// The first losses and gradient norms of this exact run (seed 2024, batch 2, seq 12) on
// the native build, so the browser build and the native build are checked as one system.
const NATIVE_LOSSES = [2.493498, 2.4627542, 2.4066973, 2.3412845, 2.278558];
const NATIVE_GRAD_NORMS = [1.0999628, 1.0802915, 0.96541786, 0.8055438, 0.756838];

function close(actual, expected, relative, what) {
  const error = Math.abs(actual - expected) / Math.max(Math.abs(expected), 1e-12);
  assert.ok(error <= relative, `${what}: ${actual} vs ${expected} (relative error ${error})`);
}

test("the module needs nothing from JavaScript and speaks the expected ABI", () => {
  const module = new WebAssembly.Module(bytes);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  assert.equal(engine.abiVersion, 1);
  assert.ok(engine.memoryBytes >= 65536);
});

test("describe lists the presets with their exact parameter counts", () => {
  const info = engine.describe();
  assert.equal(info.abi, 1);
  assert.deepEqual(
    info.presets.map((p) => p.name),
    ["smoke", "tiny", "small", "base", "medium", "large"],
  );
  assert.equal(info.presets[0].params, 117056);
  assert.equal(info.presets[1].params, 1315968);
  assert.equal(info.config_fields.length, 10);
  assert.equal(info.hyper_fields.length, 8);
  assert.equal(info.limits.max_params, 8000000);
  assert.equal(engine.describe(), info, "the description is cached");
});

test("every preset's config reports the parameter count describe() gave for it", () => {
  const info = engine.describe();
  for (const [index, preset] of info.presets.entries()) {
    const config = engine.presetConfig(index);
    assert.deepEqual(config, preset.config);
    const report = engine.configReport(config);
    assert.equal(report.valid, true, preset.name);
    assert.equal(report.params.total, preset.params, preset.name);
    const parts = Object.entries(report.params).filter(([k]) => k !== "total");
    assert.equal(
      parts.reduce((sum, [, v]) => sum + v, 0),
      report.params.total,
      `${preset.name}: the categories add up to the total`,
    );
  }
});

test("an invalid config is reported, a malformed one throws", () => {
  const bad = [...TINY];
  bad[3] = 3; // 3 heads do not divide width 16
  const report = engine.configReport(bad);
  assert.equal(report.valid, false);
  assert.match(report.error, /\S/);

  const fractional = [...TINY];
  fractional[0] = 12.5;
  assert.throws(
    () => engine.configReport(fractional),
    (e) => e instanceof EngineError && e.code === -1 && /vocab_size/.test(e.message),
  );
  assert.throws(() => engine.presetConfig(99), (e) => e instanceof EngineError && e.code === -1);
});

test("the rope table is real rotation data", () => {
  const t = engine.ropeTable(8, 6, 10000);
  assert.equal(t.positions, 6);
  assert.equal(t.half, 4);
  assert.equal(t.cos.length, 24);
  assert.equal(t.sin.length, 24);
  assert.deepEqual(Array.from(t.cos.slice(0, 4)), [1, 1, 1, 1]);
  assert.deepEqual(Array.from(t.sin.slice(0, 4)), [0, 0, 0, 0]);
  for (let i = 0; i < t.cos.length; i++) {
    close(t.cos[i] ** 2 + t.sin[i] ** 2, 1, 1e-5, `rotation ${i}`);
  }
  assert.throws(() => engine.ropeTable(7, 6, 10000), EngineError);
});

test("the browser build trains the way the native build does", () => {
  const lab = engine.createLab({ config: TINY, hyper: HYPER, seed: 2024, batch: 2, seq: 12 });
  const first = lab.step(5);
  assert.equal(first.stopped, null);
  assert.equal(first.loss.length, 5);
  for (let i = 0; i < 5; i++) {
    close(first.loss[i], NATIVE_LOSSES[i], 1e-4, `loss ${i}`);
    close(first.gradNorm[i], NATIVE_GRAD_NORMS[i], 1e-4, `gradient norm ${i}`);
  }
  const rest = lab.step(145);
  assert.equal(rest.loss.length, 145);
  assert.ok(rest.loss[144] < 0.1 * first.loss[0], `loss only fell to ${rest.loss[144]}`);
  close(rest.loss[144], 0.00060959655, 0.05, "the final loss");
  const evaluation = lab.eval();
  assert.equal(evaluation.step, 150);
  assert.ok(evaluation.accuracy > 0.95, `accuracy ${evaluation.accuracy}`);
  assert.deepEqual(evaluation.inputs[0], [9, 6, 2, 3, 3, 7, 8, 7, 0, 5, 3, 5]);
  assert.deepEqual(evaluation.predictions, evaluation.targets, "it has memorized the task");
  lab.destroy();
});

test("the learning rate follows the schedule", () => {
  const lab = engine.createLab({ config: TINY, hyper: HYPER, seed: 1, batch: 2, seq: 12 });
  const { lr } = lab.step(20);
  assert.equal(lr.length, 20);
  close(lr[0], 0.03 / 15, 1e-5, "the first step");
  close(lr[14], 0.03, 1e-5, "the end of warmup");
  assert.ok(lr[5] > lr[0] && lr[19] < lr[14]);
  lab.destroy();
});

test("stats carry every parameter, and gradient norms appear after a step", () => {
  const lab = engine.createLab({ config: TINY, hyper: HYPER, seed: 3, batch: 2, seq: 12 });
  const before = lab.stats();
  assert.equal(before.step, 0);
  assert.equal(before.last_loss, null);
  assert.equal(before.params.length, 1 + 2 * 9 + 1);
  assert.equal(before.params[0].name, "embed_tokens.weight");
  assert.deepEqual(before.params[0].shape, [12, 16]);
  assert.ok(before.params.every((p) => p.grad_norm === null && p.rms > 0));
  assert.equal(before.num_parameters, engine.configReport(TINY).params.total);
  lab.step(1);
  const after = lab.stats();
  assert.equal(after.step, 1);
  assert.ok(after.last_loss > 0);
  assert.ok(after.params.every((p) => typeof p.grad_norm === "number" && p.grad_norm >= 0));
  lab.destroy();
});

test("attention maps are causal probability distributions", () => {
  const lab = engine.createLab({ config: TINY, hyper: HYPER, seed: 4, batch: 2, seq: 12 });
  lab.step(30);
  const maps = lab.attention(1);
  assert.deepEqual([maps.layers, maps.heads, maps.seq], [2, 2, 12]);
  assert.equal(maps.data.length, 2 * 2 * 12 * 12);
  const size = 12 * 12;
  for (let m = 0; m < 4; m++) {
    for (let row = 0; row < 12; row++) {
      let sum = 0;
      for (let col = 0; col < 12; col++) {
        const p = maps.data[m * size + row * 12 + col];
        sum += p;
        if (col > row) assert.equal(p, 0, `map ${m} row ${row} reads the future`);
      }
      close(sum, 1, 1e-4, `map ${m} row ${row} sums to 1`);
    }
  }
  assert.throws(() => lab.attention(2), EngineError);
  lab.destroy();
});

test("the embedding table has one row per token", () => {
  const lab = engine.createLab({ config: TINY, hyper: HYPER, seed: 5, batch: 2, seq: 12 });
  const e = lab.embedding();
  assert.deepEqual([e.rows, e.cols, e.data.length], [12, 16, 192]);
  assert.ok(e.data.every(Number.isFinite));
  lab.destroy();
});

test("results stay readable after the module grows its memory", () => {
  // The Tiny preset needs tens of megabytes, so creating it grows memory and
  // detaches any buffer view taken before. Results must be copied out safely.
  const before = engine.memoryBytes;
  const tiny = engine.presetConfig(1);
  const lab = engine.createLab({
    config: tiny,
    hyper: [0.01, 0.001, 1, 10, 0.01, 1, 0.9, 0.99],
    seed: 6,
    batch: 1,
    seq: 16,
  });
  assert.ok(engine.memoryBytes > before, "memory should have grown");
  const steps = lab.step(2);
  assert.equal(steps.loss.length, 2);
  assert.ok(steps.loss.every(Number.isFinite));
  const embedding = lab.embedding();
  assert.equal(embedding.data.length, 4096 * 128);
  assert.ok(embedding.data.every(Number.isFinite));
  const maps = lab.attention(0);
  assert.equal(maps.data.length, 4 * 4 * 16 * 16);
  lab.destroy();
});

test("two labs from one seed train identically, and different seeds do not", () => {
  const run = (seed) => {
    const lab = engine.createLab({ config: TINY, hyper: HYPER, seed, batch: 2, seq: 12 });
    const { loss } = lab.step(10);
    lab.destroy();
    return Array.from(loss);
  };
  assert.deepEqual(run(7), run(7));
  assert.notDeepEqual(run(7), run(8));
});

test("labs are independent and a freed handle is rejected", () => {
  const a = engine.createLab({ config: TINY, hyper: HYPER, seed: 1, batch: 2, seq: 12 });
  const b = engine.createLab({ config: TINY, hyper: HYPER, seed: 2, batch: 2, seq: 12 });
  a.step(3);
  assert.equal(b.stats().step, 0, "stepping one lab must not move the other");
  a.destroy();
  assert.throws(
    () => a.step(1),
    (e) => e instanceof EngineError && e.code === -1 && /handle/.test(e.message),
  );
  assert.throws(() => a.destroy(), EngineError);
  assert.equal(b.stats().step, 0, "the other lab survives");
  b.destroy();
});

test("a bad lab is refused with the reason", () => {
  const huge = [100000, 128, 2, 4, 4, 256, 16, 10000, 1e-5, 1];
  assert.throws(
    () => engine.createLab({ config: huge, hyper: HYPER, seed: 1, batch: 2, seq: 12 }),
    (e) => e instanceof EngineError && /up to/.test(e.message),
  );
  assert.throws(() => engine.createLab({ config: TINY, hyper: HYPER, seed: 1, batch: 0, seq: 12 }), EngineError);
  assert.throws(() => engine.createLab({ config: TINY, hyper: HYPER, seed: -1, batch: 2, seq: 12 }), EngineError);
});

test("training that blows up reports the steps it completed and why it stopped", () => {
  const lab = engine.createLab({
    config: TINY,
    hyper: [1e30, 1e30, 0, 150, 0.01, 1.0, 0.9, 0.99],
    seed: 2024,
    batch: 2,
    seq: 12,
  });
  const result = lab.step(50);
  assert.ok(result.stopped && /finite/.test(result.stopped), String(result.stopped));
  assert.ok(result.loss.length > 0 && result.loss.length < 50);
  lab.destroy();
});

test("a 64-bit seed is split correctly", () => {
  const big = 2n ** 40n + 5n;
  const run = (seed) => {
    const lab = engine.createLab({ config: TINY, hyper: HYPER, seed, batch: 2, seq: 12 });
    const { loss } = lab.step(3);
    lab.destroy();
    return Array.from(loss);
  };
  assert.deepEqual(run(big), run(big));
  assert.notDeepEqual(run(big), run(5n), "the high half of the seed matters");
});

test("loadBest picks the SIMD build when the runtime can run it, and both agree", async () => {
  const plain = bytes;
  const simdPath = process.env.MIDMAN_WASM_SIMD;
  if (!simdPath) return; // only checked when the SIMD build is supplied
  const simd = readFileSync(simdPath);
  const run = async (sources) => {
    const e = await Engine.loadBest(sources);
    const lab = e.createLab({ config: TINY, hyper: HYPER, seed: 2024, batch: 2, seq: 12 });
    const { loss } = lab.step(5);
    lab.destroy();
    return Array.from(loss);
  };
  assert.equal(typeof simdSupported(), "boolean");
  // Whichever build loadBest chooses, it must give the numbers the plain build gives.
  assert.deepEqual(await run({ simd, plain }), await run({ plain }));
  assert.deepEqual(await run({ simd, plain }), await run({ simd: undefined, plain }));
});
