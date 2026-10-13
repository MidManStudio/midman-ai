// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/engine.js"
// ============================================================================
// The JavaScript side of the midman-api C ABI. No dependencies, and the same file
// runs in a browser and in Node, so the real WebAssembly build can be tested
// without a browser.
//
// The ABI rules it follows (see crates/midman-api/src/ffi.rs):
//   - input arrays are written into memory from midman_alloc and freed after the call;
//   - results are read from library-owned buffers and copied out immediately,
//     because the next call replaces them and memory growth detaches old views;
//   - a negative status is an error with its message in the error buffer.

const ABI_VERSION = 1;
const STATUS_STOPPED = 1;

export class EngineError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "EngineError";
    this.code = code;
  }
}

function splitSeed(seed) {
  const value = typeof seed === "bigint" ? seed : BigInt(Math.trunc(seed));
  if (value < 0n || value > 0xffffffffffffffffn) {
    throw new EngineError(`seed must be from 0 to 2^64 - 1, got ${seed}`, -1);
  }
  return [Number(value & 0xffffffffn), Number(value >> 32n)];
}

async function toModule(source) {
  if (source instanceof WebAssembly.Module) return source;
  if (typeof source === "string" || source instanceof URL) {
    const response = await fetch(source);
    if (!response.ok) throw new EngineError(`could not fetch ${source}: HTTP ${response.status}`, -1);
    return toModule(response);
  }
  if (typeof Response !== "undefined" && source instanceof Response) {
    // Streaming compilation needs the application/wasm type; fall back when a
    // server sends something else.
    try {
      return await WebAssembly.compileStreaming(source.clone());
    } catch {
      return WebAssembly.compile(await source.arrayBuffer());
    }
  }
  return WebAssembly.compile(source);
}

// A minimal module that only validates where the 128-bit SIMD proposal is supported.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15,
  253, 98, 11,
]);

/** Whether this runtime can run the WebAssembly SIMD build. */
export function simdSupported() {
  return WebAssembly.validate(SIMD_PROBE);
}

export class Engine {
  /**
   * Loads the SIMD build when the runtime supports it and the plain build
   * otherwise. Each is a source accepted by `load`. The two build the same code
   * and give the same numbers; SIMD is about 1.6 times faster.
   */
  static async loadBest({ simd, plain }) {
    const useSimd = simdSupported() && simd !== undefined;
    const engine = await Engine.load(useSimd ? simd : plain);
    engine.build = useSimd ? "simd" : "plain";
    return engine;
  }

  /** Loads the module from a URL, a Response, bytes, or a compiled WebAssembly.Module. */
  static async load(source) {
    const module = await toModule(source);
    const instance = await WebAssembly.instantiate(module, {});
    const engine = new Engine(instance.exports);
    if (engine.abiVersion !== ABI_VERSION) {
      throw new EngineError(
        `this page speaks ABI ${ABI_VERSION} but the module is ABI ${engine.abiVersion}`,
        -1,
      );
    }
    return engine;
  }

  constructor(exports) {
    this.x = exports;
    this.decoder = new TextDecoder();
    this.info = null;
    /** Which build is loaded: "simd", "plain", or "custom" when loaded directly. */
    this.build = "custom";
  }

  get abiVersion() {
    return this.x.midman_abi_version();
  }

  /** Bytes of WebAssembly memory the module currently holds. */
  get memoryBytes() {
    return this.x.memory.buffer.byteLength;
  }

  // ------------------------------------------------------------- buffers

  readText() {
    const len = this.x.midman_text_len();
    if (len === 0) return "";
    return this.decoder.decode(new Uint8Array(this.x.memory.buffer, this.x.midman_text_ptr(), len));
  }

  readError() {
    const len = this.x.midman_error_len();
    if (len === 0) return "";
    return this.decoder.decode(new Uint8Array(this.x.memory.buffer, this.x.midman_error_ptr(), len));
  }

  readFloats() {
    const len = this.x.midman_floats_len();
    if (len === 0) return new Float32Array(0);
    // slice() copies, so the result survives the next call and memory growth.
    return new Float32Array(this.x.memory.buffer, this.x.midman_floats_ptr(), len).slice();
  }

  /** Throws for a negative status; returns the status otherwise. */
  check(status) {
    if (status < 0) throw new EngineError(this.readError() || `status ${status}`, status);
    return status;
  }

  /** Writes `values` into library memory as f64, calls `use(pointer)`, then frees it. */
  withF64(values, use) {
    const bytes = values.length * 8;
    const ptr = this.x.midman_alloc(bytes);
    if (ptr === 0) throw new EngineError("out of memory", -1);
    try {
      // The view is created after the allocation, which may have grown memory.
      new Float64Array(this.x.memory.buffer, ptr, values.length).set(values);
      return use(ptr);
    } finally {
      this.x.midman_free(ptr, bytes);
    }
  }

  // ------------------------------------------------------- stateless calls

  /** The ABI description: field orders, limits and every preset. Cached. */
  describe() {
    if (!this.info) {
      this.check(this.x.midman_describe());
      this.info = JSON.parse(this.readText());
    }
    return this.info;
  }

  /** Validates a config array and counts its parameters; see ffi.rs for the order. */
  configReport(config) {
    this.check(this.withF64(config, (ptr) => this.x.midman_config_report(ptr)));
    return JSON.parse(this.readText());
  }

  /** The config array of preset `index`. */
  presetConfig(index) {
    const out = new Float64Array(10);
    const status = this.withF64(out, (ptr) => {
      const s = this.x.midman_preset_config(index, ptr);
      if (s >= 0) out.set(new Float64Array(this.x.memory.buffer, ptr, 10));
      return s;
    });
    this.check(status);
    return Array.from(out);
  }

  /** The rotary rotation tables: `{ positions, half, cos, sin }`. */
  ropeTable(headDim, maxSeqLen, theta) {
    this.check(this.x.midman_rope_table(headDim, maxSeqLen, theta));
    const meta = JSON.parse(this.readText());
    const both = this.readFloats();
    const n = meta.positions * meta.half;
    return { positions: meta.positions, half: meta.half, cos: both.slice(0, n), sin: both.slice(n) };
  }

  // ---------------------------------------------------------------- labs

  /**
   * Creates a lab. `config` and `hyper` are arrays in the ABI's field order
   * (see describe()); `seed` is a number or BigInt.
   */
  createLab({ config, hyper, seed = 1, batch, seq }) {
    const [lo, hi] = splitSeed(seed);
    const handle = this.withF64(config, (cptr) =>
      this.withF64(hyper, (hptr) => this.x.midman_lab_create(cptr, hptr, lo, hi, batch, seq)),
    );
    if (handle === 0) throw new EngineError(this.readError() || "could not create the lab", -1);
    return new Lab(this, handle, { config, hyper, seed, batch, seq });
  }
}

export class Lab {
  constructor(engine, handle, settings) {
    this.engine = engine;
    this.handle = handle;
    this.settings = settings;
  }

  /**
   * Runs `steps` training steps. Returns columnar results for the steps that
   * completed, and `stopped`, the reason training ended early or null.
   */
  step(steps) {
    const e = this.engine;
    const status = e.check(e.x.midman_lab_step(this.handle, steps));
    const flat = e.readFloats();
    const n = flat.length / 3;
    const loss = new Float32Array(n);
    const gradNorm = new Float32Array(n);
    const lr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      loss[i] = flat[3 * i];
      gradNorm[i] = flat[3 * i + 1];
      lr[i] = flat[3 * i + 2];
    }
    return { loss, gradNorm, lr, stopped: status === STATUS_STOPPED ? e.readError() : null };
  }

  /** Loss, accuracy and the task's inputs, targets and current predictions. */
  eval() {
    this.engine.check(this.engine.x.midman_lab_eval(this.handle));
    return JSON.parse(this.engine.readText());
  }

  /** Per-parameter name, shape, rms and last gradient norm. */
  stats() {
    this.engine.check(this.engine.x.midman_lab_stats(this.handle));
    return JSON.parse(this.engine.readText());
  }

  /** Attention probabilities of task sequence `sequence`: `{ layers, heads, seq, data }`. */
  attention(sequence = 0) {
    const e = this.engine;
    e.check(e.x.midman_lab_attention(this.handle, sequence));
    const meta = JSON.parse(e.readText());
    return { ...meta, data: e.readFloats() };
  }

  /** The token embedding table: `{ rows, cols, data }`. */
  embedding() {
    const e = this.engine;
    e.check(e.x.midman_lab_embedding(this.handle));
    const meta = JSON.parse(e.readText());
    return { ...meta, data: e.readFloats() };
  }

  /** Frees the lab. Using it afterwards throws. */
  destroy() {
    this.engine.check(this.engine.x.midman_lab_destroy(this.handle));
  }
}
