# midman-tensor

## Overview
A from-scratch tensor library with reverse-mode automatic differentiation, written for MidMan and used by nothing else. It has no dependencies except `midman-foundation`. It runs on the CPU only.

A `Tensor` is an immutable, reference-counted block of `f32` values with a shape, stored dense and row-major. Cloning a tensor copies a pointer. Every operation returns a new tensor. When any input of an operation tracks gradients, the result records its inputs and a backward closure, and `Tensor::backward` walks that graph in reverse to compute the gradient of a scalar loss with respect to every gradient-requiring leaf. A `Parameter` is a named leaf the optimizer can update in place.

Why it exists: the project decision is to build the tensor, autograd and kernel layers in-house instead of building on a framework. A reference implementation on the CPU comes first because it can be tested completely, and every later backend has to match it.

**What the model needs, and what is here.** The gradient checks cover each operation a Llama-style decoder block uses: embedding lookup, RMSNorm built from primitives, rotary embeddings, causal attention with grouped-query sharing, the SiLU-gated MLP, residual connections, a tied output head and cross-entropy.

**Operations a future backend must provide.** The surface is small on purpose:

| Group | Operations |
| --- | --- |
| Matrix products | three loop kernels: `A*B`, `A*B^T`, `A^T*B`, applied per batch |
| Elementwise | `add`, `sub`, `mul`, `div` with broadcasting; `scale`, `add_scalar`, `neg`, `exp`, `ln`, `sqrt`, `powf`, `tanh`, `sigmoid`, `relu`, `silu`, `gelu` |
| Reductions | `sum_all`, `mean_all`, `sum_axis`, `mean_axis`, `softmax`, `log_softmax`, `cross_entropy` |
| Data movement | `reshape`, `permute`, `transpose`, `narrow`, `concat`, `repeat_interleave`, `embedding` (gather, with scatter-add backward) |

Every backward pass is written in terms of the same kinds of kernels (for example, the backward of `matmul` is two more matrix products), so a backend that implements the forward set gets autograd for free.

**Limits.**
- CPU and `f32` only. There is no device abstraction yet and no GPU design. When a second backend arrives, the tensor storage has to be generalized, and every operation will need a kernel for it.
- The kernels are plain single-threaded loops, written so the compiler can vectorize them: no intrinsics and no threads. They are the reference first and fast second. `docs/midman-bench.md` has the measured speed, including a comparison with plain C.
- Zero-sized axes are rejected. No operation needs them yet, and rejecting them keeps every kernel free of empty-slice cases.
- There are no in-place operations. `reshape` shares data with its input; `permute`, `narrow` and the rest copy.
- There is no no-grad mode. Use `Tensor::detach` or a plain `Tensor::from_vec` to run without building a graph.
- `Tensor::backward` consumes the tensor and cannot be called twice on one graph.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports `Tensor`, `Grads` and `Parameter`. The crate docs include a runnable example that differentiates a sum of squares.

### `shape.rs`
**What it does:** Shape arithmetic shared by every operation: element counts, row-major strides, NumPy-style broadcasting (`broadcast_shapes`), negative-axis handling (`normalize_axis`), the zero-axis check, the `(outer, len, inner)` split used to walk any axis, row-by-row iteration over strided offsets (`for_each_row`, `for_each_row2`), `sum_to_shape` (the adjoint of broadcasting) and `permute_data`.

**Design notes:**
- Broadcasting is implemented once, with stride 0 on stretched axes, and every elementwise operation and its backward pass reuse it.
- `sum_to_shape` is what turns a gradient at the broadcast output shape back into a gradient at each operand's own shape.
- Broadcast and permuted layouts are walked one row at a time, a row being a run along the last axis. The index moves once per row and not once per element, and the caller's loop over a row is a plain loop over slices that the compiler can vectorize. The earlier walk moved the index once per element and made a broadcast add about 12 times slower (365 us against 29 us for `[256, 512] + [512]`).
- The per-element walkers `for_each_offset` and `for_each_offset2` remain in test builds only. They are the reference that the row walkers, and the operations built on them, are checked against bit for bit.
- `sum_to_shape` adds each output element's contributions in the same order as before (row-major order of the gradient), so no gradient changed by a single bit.
- The functions that take or return an axis accept negative values counted from the end, so model code can write `softmax(-1)`.

### `tensor.rs`
**What it does:** The `Tensor` type, its constructors (`from_vec`, `scalar`, `full`, `zeros`, `ones`, `uniform`, `randn`), accessors (`shape`, `rank`, `numel`, `id`, `data`, `item`), gradient tracking (`leaf_requiring_grad`, `detach`, `tracks_grad`) and the `Grads` map that `backward` returns.

**Design notes:**
- A tensor is `Arc<Inner>`. `Inner` holds the shape, the data (itself an `Arc<Vec<f32>>`, so views can share it), a unique id, the leaf flag and an optional graph node.
- Gradients are keyed by tensor id, not by pointer, so a parameter that is cloned into many places still has one gradient.
- `Grads` holds only leaf gradients. Intermediate gradients are dropped as soon as they have been used.
- `Inner` has a hand-written iterative `Drop` (see Fixes and Problems).

### `autograd.rs`
**What it does:** `make_op`, which every operation uses to create its result, and `Tensor::backward`.

**Design notes:**
- The graph is built eagerly. If no input tracks gradients, `make_op` drops the backward closure and returns a plain constant, so inference builds no graph.
- `backward` walks the graph with an iterative depth-first search, so graph depth is not limited by the call stack, then processes nodes in reverse order. Gradients that reach a tensor from several uses are summed.
- Each backward closure receives a flag per parent saying whether that parent needs a gradient, so constants such as masks and index tensors cost nothing.
- The engine is tested with hand-written closures, separately from the real operations, including a tensor used twice, a diamond-shaped graph and a 20,000-node chain.

### `param.rs`
**What it does:** `Parameter`, a named trainable leaf. `tensor()` hands out a gradient-tracking handle for a forward pass, `update()` changes the values in place, and `set_data()` replaces them (for loading a checkpoint).

**Design notes:**
- `update` edits in place when nothing else holds the parameter, which is the case once the step's graph has been dropped. If a live graph still references the old values, they are copied first so that graph keeps the values it computed with. The parameter's id survives the copy.
- `Tensor::backward` consuming its tensor is what normally lets the graph drop before the optimizer step.

### `ops_elementwise.rs`
**What it does:** `add`, `sub`, `mul`, `div` with broadcasting, `scale`, `add_scalar`, `neg`, and the unary functions `exp`, `ln`, `sqrt`, `powf`, `tanh`, `sigmoid`, `relu`, `silu` and `gelu` (tanh approximation).

**Design notes:**
- `add` and `sub` sum the output gradient straight back to each operand shape. `mul` and `div` evaluate each operand's partial derivative per element pair, then sum to the operand shape.
- Unary operations keep their output so derivatives such as `sigmoid` and `tanh` reuse it.
- `sigmoid` is written so large negative inputs do not overflow.
- The GELU constant is checked against its definition by a test.
- A row of a broadcast result has one of three forms on the last axis: two slices, a slice and one value, or one value and a slice. Each is a plain loop over zipped slices. Any other stride pattern takes a general strided loop. The same loops written over iterator adapters (`copied`, `repeat`, nested `zip`) were vectorized only under fat LTO and ran at a tenth of the speed with 16 codegen units, so the helpers take slices and scalars.
- `binary_op` and `binary_grads` are generic over their arithmetic, so each operation gets its own copy of the loops with the arithmetic inlined, and no function pointer is called per element.

### `ops_reduce.rs`
**What it does:** `sum_all`, `mean_all`, `sum_axis` and `mean_axis` (with `keepdim`), `softmax`, `log_softmax` and `cross_entropy`.

**Design notes:**
- Softmax and log-softmax subtract the row maximum first, so large logits do not overflow. A test uses logits of 1000.
- `cross_entropy` works from the logits directly with the log-sum-exp form and takes the mean over rows. Its gradient is `(softmax - one_hot) / N`.
- `sum_all` accumulates in `f64`, which keeps long sums accurate.
- Softmax over any axis works through the shared `(outer, len, inner)` index split, so attention does not need transposes.
- When the reduced axis is the last one, which is the common case, `sum_axis` and `softmax` and their backward passes work on contiguous slices. Every sum is added in the same order as in the strided loop, so the results are bit-for-bit the same. Tests compare against the strided loop on every axis. `log_softmax` still uses the strided loops; it has not been measured.
- The softmax maximum is a `fold`, not a `for` loop with a running maximum. The compiler vectorizes the first form and not the second, and the difference was 75 of the 175 microseconds of a `[256, 128]` softmax.

### `ops_matmul.rs`
**What it does:** The three matrix-product loop kernels and `Tensor::matmul`.

**Supported forms:** `[.., m, k] x [k, n]` (a linear layer: one matrix applied at every leading position) and `[b.., m, k] x [b.., k, n]` (batched matrices, for example attention scores).

**Design notes:**
- Forward uses `A*B`. The backward pass uses `A*B^T` for the left operand's gradient and `A^T*B` for the right operand's gradient.
- `A*B^T` copies `B` into a transposed temporary and runs the `A*B` kernel. The alternative, one dot product per output element, is a strictly ordered float sum that the compiler may not vectorize. It ran at 3.6 to 5.2 GFLOP/s against 16 to 20 for the other two kernels, and an AVX2 and FMA build did not change that. Each output element still adds its products in the same order, so the gradients are bit-for-bit those of the dot-product form: six training steps were compared on every loss, every gradient and every updated parameter. The kernel needs its output to start at zero, as it does at every call site, and a debug assertion checks that.
- For the linear-layer form, the right operand's gradient is a single product over all flattened leading positions, which sums the batch contributions correctly.
- The kernels are the reference for any future backend.

### `ops_shape.rs`
**What it does:** `reshape` (shares data), `permute`, `transpose`, `narrow`, `concat`, `repeat_interleave` and `embedding`.

**Design notes:**
- `permute`'s backward applies the inverse permutation. `narrow` and `concat` are each other's backward.
- `repeat_interleave` repeats each entry in place (`[a, b]` with 2 becomes `[a, a, b, b]`). Grouped-query attention uses it so that consecutive query heads share one key/value head.
- `embedding` takes the indices as a separate slice and shape, not as a tensor, because token ids are integers and carry no gradient. Its backward scatter-adds into the table, so repeated ids accumulate. Out-of-range ids are an error.

### `check.rs`
**What it does:** `grad_check`, which compares autograd's gradients with central finite differences, and `GradCheckReport` with `assert_within` for tests.

**How it works:** The function under test may return any shape. It is reduced to a scalar by a fixed random weighting so that every output element matters. Each input element is perturbed by plus and minus `eps` (default `1e-2`) and the numeric slope is compared with the analytic gradient, using an error scaled by `max(1, |analytic|, |numeric|)`.

**Design notes:**
- Tolerances are loose for `f32` (`2e-2` scaled error in the tests) and the inputs are chosen to avoid kinks and domain edges. A real gradient bug produces errors of order 1.
- The checker itself is tested: it passes correct gradients and fails a deliberately wrong one.
- Later crates (layers, the model) use the same function.

### `tests/grad_ops.rs`
**What it does:** 22 gradient checks. One group covers each operation across its supported shapes and broadcast patterns. The rest cover compositions: a two-layer perceptron, RMSNorm from primitives, causal attention, grouped-query attention, rotary embeddings, a tied-embedding language-model head, and a gated MLP block with a residual.

**Design notes:**
- Behavior checks sit beside the gradient checks: RMSNorm output has unit RMS, causal attention ignores future positions, rotary embeddings preserve vector length.
- The suite was mutation-tested. Eight plausible backward-pass bugs were injected one at a time (a flipped softmax sign, scatter-add replaced by assignment, unbroadcasting that overwrites instead of summing, a missing `1/N` in cross-entropy, a dropped factor in the `powf` and GELU derivatives, a wrong operand in the matmul gradient, and a skipped inverse permutation). Each made at least one test fail.

## CI and Workflows
The workspace workflows in `.github/workflows` cover this crate through the package dropdown. See `docs/ci-and-workflows.md`.

## Fixes and Problems

### `shape.rs`
- **The per-element index walk was the main cost of every broadcast.** Timing the stages of RMSNorm one at a time showed each broadcast multiply costing 77 us for 32,768 values, where the same work as a row loop costs 5 to 6 us. A broadcast add of `[256, 512] + [512]` took 365 us against 16 us for a plain row loop; through the row walker it takes 29 us, level with a plain C loop at 26 us. The same walk also served the backward pass of every broadcast, `sum_to_shape` and `permute_data`, and all four now go row by row. Tests compare each of them with the old per-element walk bit for bit, over shapes that cover every mix of kept and stretched last axes, and a mutation run (a missing carry reset, a dropped accumulation, a wrong stride) was caught each time.

### `ops_elementwise.rs`
- **The first version of the row helpers was fast only under fat LTO.** They took iterators (`copied`, `repeat`, nested `zip`). A benchmark run with LTO off showed no speed-up at all, and a harness reproduced it: 374 us against 32 us for slices, at 16 codegen units. Marking the helpers `#[inline]` did not help; a callgrind profile showed scalar code at about 28 instructions per element. They take slices and scalars now, and are fast under both profiles.
- **A function pointer was called per element.** `binary_op` took its arithmetic as `fn` pointers. It is generic now.

### `ops_reduce.rs`
- **`mean_axis` over a short last axis was slow.** It cost 66 us for 32,768 values (a `[512, 64]` input) because the general loop went through one-element chunks. The contiguous path takes 8 us, level with a plain loop, with the same bits.
- **The softmax maximum loop cost 75 microseconds.** A `for` loop with a running maximum is not vectorized by the compiler, and the equivalent `fold` is. Found by counting instructions per source line with callgrind, which showed about 7 per element in `f32::max`. A softmax over `[256, 128]` went from 182 to 103 us in the harness.

### `ops_matmul.rs`
- **`A*B^T` ran at a quarter of the speed of the other kernels.** See the design notes. By itself this change took a Smoke training step from 15.0 to 9.9 ms in a harness, and the matrix kernels with their backward closures were about 71 percent of the instructions of a training step in a callgrind profile, so the slowest of the three kernels sat on the hottest path. A debug-only assertion that the output starts at zero guards the one assumption the transposed form adds.

### `tensor.rs`
- **Open: large intermediates cost page faults.** Tensors of 128 KB or more that are allocated and freed in a sequence make glibc return memory to the operating system, and the next call takes page faults to get it back. RMSNorm over `[512, 64]` took 93 us with about 943,000 page faults over a harness run, and 29.6 us with 323 when glibc was told not to trim. The Smoke training step, whose tensors are all under 128 KB, changed by 3 percent with the same setting. Not fixed. The options are a reusable buffer pool in this crate, or setting glibc's thresholds in the binaries; the first changes the data model and the second is Linux-specific. Measured on glibc only.
- **Stack overflow when dropping a deep graph.** With the default recursive drop, releasing a 20,000-node chain of operations overflowed the stack and aborted the process (the `deep_chains_do_not_overflow_the_stack` test reproduces it). `Inner` now has an iterative `Drop` that unlinks parents onto a work list. `backward` already used an iterative search, so only the drop needed this.
