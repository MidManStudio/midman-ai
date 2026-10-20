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

### `midman_api.h`
**What it does:** The C header for the ABI in `ffi.rs`, in `crates/midman-api/ffi-smoke-test/`. It declares every exported function, the status codes, the array lengths and two enums that name the positions in the config and hyperparameter arrays, and it states the rules of the boundary (handles, who owns which buffer, how long results stay valid, one thread).

**Decisions:**
- Hand-written, as the rest of the ABI is. There is no header generator in the build, no `cbindgen`, and so no tool whose version has to match. The cost is that a signature changed in `ffi.rs` must be changed here in the same commit. `test.c` and `cxx_check.cpp` call every function through this header, so a prototype that no longer matches the library fails a test.
- The enums `midman_config_field` and `midman_hyper_field` give the array positions names, and `test.c` compares the library's `midman_describe` field lists with them, so the header and the library cannot disagree on the order without a failing check. Static assertions tie each enum's size to the array length.
- Static assertions (`_Static_assert` in C, `static_assert` in C++) require a 32-bit `float` and a 64-bit `double`, because the arrays and results are passed that way.
- The result rule is written conservatively: read what you need, and copy it, before the next call into the library that is not one of the accessors. `ffi.rs` says the same in its own words. The header does not promise that the accessors leave results alone, only that this library currently does.
- A result pointer may be non-NULL when its length is 0 (an empty `Vec` has a dangling pointer). The header says not to dereference it.
- Status codes and constants are `#define`s so the header works in C and C++ and in `static_assert` expressions.
- The header has an `extern "C"` guard and uses only `<stddef.h>` and `<stdint.h>`.

**Tests:** `test.c` (C11) and `cxx_check.cpp` (C++17) include it and are compiled with `-Wall -Wextra -Wpedantic -Werror`.

### `test.c`
**What it does:** A C11 program that calls the whole ABI through `midman_api.h` and links against the compiled library, as a shared library or as a static one. It has 90 checks, one `ok:` or `FAIL:` line each, and ends with `=== N check(s) failed ===`. It covers what `tests/c_abi.rs` covers, written the way a C caller writes it, and three things only a C caller can get wrong: arrays built in `midman_alloc` memory and freed after the call, a misaligned pointer, and results that have to be copied out before the next call.

**Decisions:**
- It is a separate program and not a Rust test, so it is compiled by a C compiler, links through the real C ABI, and can be run under AddressSanitizer, UndefinedBehaviorSanitizer and valgrind. A Rust test that calls the same `extern "C"` functions from the same binary proves less.
- Every lab a test creates is destroyed before the test returns, so the slot numbers asserted (the first lab is 1, a freed slot is reused) hold, and a leak checker sees a clean exit. Valgrind reports no blocks in use at exit.
- Results are copied out of the library's buffers into the program's own after each call. A failed check copies once more and prints the last text and error, so a log explains its own failure.
- A result too big for the program's buffers is a reported failure and not a silent cut.
- Standard output is set to line buffering. With a block buffer, a crash lost every line printed so far (found by crashing the library on purpose and looking at the log: 0 `ok:` lines survived), and the log would have said nothing about how far the program got. With line buffering the same crash leaves 25.
- The closing line is how the summary tells a finished run from a crash. A run with no closing line is reported as one failed check, "program ran to completion".

**Tests:** The program is the test. It was checked by injecting eight faults into `ffi.rs` one at a time (a step that stopped early reported as success, the alignment check removed, a freed slot never reused, the ABI version changed, two config fields swapped, results not cleared between calls, a destroyed handle left live, the null check removed) and each one made at least one check fail or the program abort. Six give a `FAIL:` line, and the two that remove a safety check crash the library with a failed precondition, which the summary reports as a failed run. It passes with the shared and static libraries, in debug and release builds, under AddressSanitizer and UndefinedBehaviorSanitizer, and under valgrind.

### `cxx_check.cpp`
**What it does:** A C++17 program that includes `midman_api.h` and calls the library: the version, `midman_describe`, a preset, a config report, and a lab from creation to destruction. Eight checks, with the same output convention as `test.c`.

**Decisions:**
- It exists only to prove the header parses as C++ with its `extern "C"` guard and that the symbols link without mangling. The behaviour is checked once, in `test.c`.
- It is compiled with `g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror`.

**Tests:** The program is the test.

## CI and Workflows
`ci.yml` covers this crate with the rest. `midman-api-ffi.yml` tests the C ABI from C and C++: the Rust tests, `test.c` against the shared and the static library, `cxx_check.cpp`, `test.c` under AddressSanitizer and UndefinedBehaviorSanitizer, and `test.c` under valgrind, on Ubuntu by default and on macOS and Linux ARM with `platforms: all`, with `profile: debug` or `release`. `deploy-site.yml` builds it for WebAssembly (a SIMD and a plain build), runs the Node and headless-DOM tests against the result, and publishes it. The script behind the FFI workflow's summary is `scripts/ffi_summary.py`. See `docs/ci-and-workflows.md` and `docs/midman-web.md`.

## Fixes and Problems

### `lab.rs`
- **Clippy `chunks_exact_to_as_chunks` on a constant chunk size in the attention-map test.** The size is now a local variable, which the lint does not flag. See `docs/workspace-cargo.md`, section "Toolchain".
- **`Option::is_none_or` needs Rust 1.82.** The first version used it for the token limit and would not have built on 1.75. It is a `matches!` now.

### `test.c`
- **A crash lost all of the program's output.** Standard output to a file is block-buffered, so when a mutated library aborted the program, none of the `ok:` lines printed before the abort reached the log. Both programs now set line buffering at the start of `main`.
- **A failed check printed stale text.** The library call sits inside the check, and the results were copied out only after it, so the `text:` and `error:` lines under a failure showed the previous call's results. A failed check now copies the buffers before it prints them.

### `tests/c_abi.rs`
- **Clippy `chunks_exact_to_as_chunks` on a constant chunk size in the lab test.** The size is now a local variable, which the lint does not flag. See `docs/workspace-cargo.md`, section "Toolchain".
- **Clippy `manual_is_multiple_of` on `len % 3 == 0`.** Only newer clippy flags it. It is written `(len / 3) * 3 == len`, as elsewhere in the workspace.
- **A clumsy placeholder assertion** was replaced with a plain check that an odd head size gives a negative status and a message.
