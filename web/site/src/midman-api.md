# midman-api

**What it does:** the interface for applications that embed MidMan. `Lab` trains a model on a fixed task from the outside and exposes the loss, gradient norms, attention maps and embeddings. A C ABI (`extern "C"`, opaque handles, status codes, no unwinding) wraps it, and that is exactly what the browser build exports.

**Example:** `midman_lab_create`, `midman_lab_step`, `midman_lab_attention`, then read results through `midman_floats_ptr`.

**Status:** done and tested natively and as real WebAssembly. The browser build is 256 KB with no imports and matches the native build to about one float rounding step.
