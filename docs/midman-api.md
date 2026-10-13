# midman-api

## Overview
The interface for applications that embed MidMan. `Lab` trains a model on a fixed task from the outside and exposes the loss, the gradient norms, the attention maps and the embedding table. `config_report` and `rope_table` answer questions that need no model. A C ABI (`src/ffi.rs`) wraps all of it with opaque handles, and that ABI is exactly what the browser build exports, so the web lab runs the real Rust code.

The crate builds as `rlib` and `cdylib`. It is the one crate that opts out of the workspace's `unsafe_code = "deny"`, at its crate root with a comment saying why, as mid-engine's FFI crates do. Every `unsafe` block has a `// SAFETY:` comment and every `unsafe fn` documents its contract. It has no external dependencies.

Dependencies: `midman-foundation`, `midman-model-config`, `midman-tensor`, `midman-nn`, `midman-model` and `midman-optim`.

Test status when this was written: 24 unit tests, 16 tests of the C ABI, 1 doctest. Checked with `fmt`, `clippy -D warnings` and `rustdoc -D warnings` on Rust 1.75 and 1.91. The same ABI was also tested as a real WebAssembly binary in Node (17 tests in `web/lab/test/engine.test.mjs`), including agreement with the native build.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules, re-exports the public items, and carries the unsafe opt-out and its explanation. The crate docs include a runnable example.

**Tests:** The doctest.

### `json.rs`
**What it does:** `JsonWriter`, a small std-only JSON writer that tracks commas and colons for the caller. Numbers JSON cannot hold (NaN, infinities) are written as `null`, so the output always parses.

**Decisions:**
- Nothing in this crate reads JSON, so there is no parser and no dependency.

**Tests:** Unit tests for nesting, escaping, non-finite numbers, missing numbers and a top-level scalar. The browser tests parse every result with `JSON.parse`.

### `lab.rs`
**What it does:** `Lab`, a model, an optimizer, a warmup-and-cosine schedule and a fixed task, stepped from outside. The task is memorization: `batch` random sequences of `seq + 1` tokens drawn from the seed, inputs the first `seq` and targets the last `seq`, the same task the training smoke test uses. `step` returns one record (loss, global gradient norm before clipping, learning rate) per completed step. `eval` gives the loss, the accuracy and the top prediction at every position. `attention` gives every layer's and head's probabilities for one sequence, `embedding` the token table, and `param_stats` each parameter's shape, root mean square and last gradient norm.

**Decisions:**
- A lab refuses models over 8 million parameters and tasks over 4096 tokens. It holds the weights, two Adam moments and the gradients (about four copies), and it runs in a browser tab on a phone.
- Training that produces a non-finite loss or gradient stops, reports the steps that did complete, and says why, instead of returning an error that would throw those steps away.
- One seed fixes the weights and the task, so a run is reproducible bit for bit.

**Tests:** Unit tests that the task has the documented shape, that training learns it (loss falls below a tenth, accuracy above 95%), that records follow the schedule, that stepping in pieces equals stepping at once, bit-for-bit reproducibility, parameter statistics before and after a step, the attention layout and causality, the embedding shape, a run that blows up, and bad configurations including the size limit.

### `report.rs`
**What it does:** `config_report` validates a `ModelConfig` and counts its parameters by category without building anything, and `rope_table` returns the rotary cosines and sines.

**Tests:** Unit tests for a preset's exact count, an invalid configuration, and the rope table's first row and size.

### `ffi.rs`
**What it does:** The C ABI. Everything is `extern "C"` with fixed-width arguments. The rules it follows:
- **Handles, not pointers:** a `Lab` lives in a table inside the library and the caller holds a small integer, so a stale or invented handle is an error status, not undefined behaviour.
- **The caller owns its buffers, the library owns its results:** inputs are written into memory from `midman_alloc` and freed with `midman_free`; results are left in library-owned buffers (`midman_text_*`, `midman_floats_*`, `midman_error_*`) that stay valid until the next call.
- **Status codes, never unwinding:** `0` is success, `1` means a step run stopped early, negative is failure with the message in the error buffer. A panic is caught and reported as `-99`; a build with `panic = "abort"`, as WebAssembly has, traps instead.
- **One thread:** state is thread-local, so a handle is only valid on the thread that created it.

**Decisions:**
- Arrays of numbers cross as `f64`, in a fixed field order that `midman_describe` lists. A seed is two `u32` halves, which avoids 64-bit integers at the JavaScript boundary.
- Results are split into text (JSON) and numbers (`f32`, 4-byte aligned) so a caller can view the numbers without copying.
- The ABI is written by hand with no `wasm-bindgen`, so the crate has no dependencies and no tool whose version must match.

**Tests:** Unit tests that a panic becomes a status and a message and the library keeps working, the error-code mapping, and the number conversions. `tests/c_abi.rs` covers the rest.

### `tests/c_abi.rs`
**What it does:** Tests of the C ABI written the way a C caller uses it: through the exported functions, with raw pointers, reading results back through the exported buffers. They cover allocation and alignment, the description, preset round trips, an invalid config being an answer rather than a failure, malformed arguments, the full lab lifecycle, independent labs and slot reuse, invented handles, bad creation, early stopping, results being replaced by every call, determinism through the ABI, and the rope table.

## CI and Workflows
`ci.yml` covers this crate with the rest. `deploy-site.yml` builds it for WebAssembly (a SIMD and a plain build), runs the Node and headless-DOM tests against the result, and publishes it. See `docs/ci-and-workflows.md` and `docs/midman-web.md`.

## Fixes and Problems

### `lab.rs`
- **`Option::is_none_or` needs Rust 1.82.** The first version used it for the token limit and would not have built on 1.75. It is a `matches!` now.

### `tests/c_abi.rs`
- **Clippy `manual_is_multiple_of` on `len % 3 == 0`.** Only newer clippy flags it. It is written `(len / 3) * 3 == len`, as elsewhere in the workspace.
- **A clumsy placeholder assertion** was replaced with a plain check that an odd head size gives a negative status and a message.
