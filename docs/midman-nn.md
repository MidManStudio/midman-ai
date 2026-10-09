# midman-nn

## Overview
The layers a decoder-only transformer is built from: a linear layer, an embedding table, RMSNorm, rotary position embeddings, causal grouped-query attention, a SwiGLU MLP, and a pre-norm decoder block. Every layer owns its weights as `Parameter`s from `midman-tensor` and builds its forward pass from tensor operations, so `Tensor::backward` differentiates it with no extra code.

The crate does not know about `ModelConfig`. Every layer takes its sizes explicitly, and `midman-model` assembles them from a config. The math of each layer is also a public function over plain tensors (`rms_norm`, `apply_rope`, `causal_attention`, `attention_forward`, `swiglu`). The layer structs call those functions, so the gradient checks exercise the code the model actually runs, and they can differentiate with respect to the weights.

Dependencies: `midman-foundation` and `midman-tensor`.

Test status when this was written: 50 unit tests, 8 gradient-check tests and 1 doctest. Checked with `fmt`, `clippy -D warnings` and `rustdoc -D warnings` on Rust 1.75 and on Rust 1.91, and the unit and gradient tests were mutation-tested (see "Fixes and Problems").

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports the public items. The crate docs include a runnable example.

**Tests:** The doctest.

### `module.rs`
**What it does:** The `Module` trait (`parameters` and `parameters_mut`) and `count_parameters`.

**Decisions:**
- The order of the parameter list is fixed: declaration order, with a layer's children in the order they run. Optimizers and checkpoints depend on it, and tests in `midman-model` pin the full order.
- The trait covers parameter access only. Forward signatures differ per layer (token ids, a rotary table, a plain tensor), so a shared `forward` would only hide that.

**Tests:** Covered through the layer tests, which check that the shared and mutable lists agree, and the parameter-order tests in `block.rs` and in `midman-model`.

### `init.rs`
**What it does:** `Init`, the two standard deviations used to draw weights: `std` for ordinary weights and `out_std` for the projections that write into the residual stream (the attention output and the MLP down projection). `Init::scaled(std, n_layers)` sets `out_std = std / sqrt(2 * n_layers)`; `Init::uniform(std)` uses one value for both.

**Decisions:**
- The residual-stream scaling is the usual GPT-2 style choice that keeps the variance of the stream from growing with depth. The values (0.02 in the model) are defaults to start from and are not tuned.

**Tests:** Unit tests for the scaling with depth, the uniform constructor, and rejection of bad values.

### `linear.rs`
**What it does:** `Linear`, `y = x * W` with `W` of shape `[in_features, out_features]` and no bias. The weight is named `"{name}.weight"`.

**Decisions:**
- The layout is `[in, out]`, the transpose of PyTorch's `nn.Linear`. This only matters if a checkpoint is ever exchanged with another tool.
- `x` needs at least two axes, because it goes through the shared-right-operand form of `matmul`.

**Tests:** Unit tests for the product at every position, leading axes, naming and sizing, seeded reproducibility, the measured spread of the weights, and bad arguments.

### `embedding.rs`
**What it does:** `Embedding`, a table of shape `[num_embeddings, dim]` with a lookup that takes row-major ids and their shape. A model with tied embeddings also uses `weight()` as its output matrix.

**Tests:** Unit tests for row lookup in order, naming and sizing, and out-of-range ids.

### `norm.rs`
**What it does:** `rms_norm(x, weight, eps)` and the `RmsNorm` layer, `x * (mean(x^2) + eps)^(-1/2) * weight` with the mean over the last axis. The layer's weight starts at 1.

**Decisions:**
- Built from `mul`, `mean_axis`, `add_scalar` and `powf`, so it differentiates with respect to both arguments.
- `weight` must be a vector as long as the last axis. Without that check a weight of shape `[1]` would broadcast silently.

**Benchmarks:** `rms_norm_512x64` in `nn_layers.rs`. In one noisy local run the layer took 305 us against 22 us for a plain loop. See "Fixes and Problems" below and `docs/midman-bench.md`.

**Tests:** Unit tests for unit root mean square per row, a known row, the effect of `eps` on tiny inputs, invariance to scaling the input, the per-channel weight, initial value and naming, and bad arguments. Gradient checks in `tests/grad_layers.rs`.

### `rope.rs`
**What it does:** `apply_rope` and `RotaryTables`. Position `p` and frequency index `i` use the angle `p * theta^(-i / half)` with `half = head_dim / 2`. `RotaryTables::apply` takes `[batch, heads, time, head_dim]` and treats the time axis as positions `0..time`.

**Decisions:**
- Layout is "rotate half": the pairs are `(x[i], x[i + half])`, not interleaved neighbours. Any later import of weights trained with the other layout would need a permutation.
- The tables are computed in `f64` and stored as `f32`.
- There is no start-position argument. Nothing needs one until `midman-inference` has a KV cache.

**Benchmarks:** `rope_t128` in `nn_layers.rs`: 116 us in one noisy local run.

**Tests:** Unit tests for position zero, the table formula, preserved vector length, that dot products depend only on the distance between positions, shorter sequences using the first positions, and bad arguments. Gradient checks in `tests/grad_layers.rs`.

### `attention.rs`
**What it does:** `causal_mask`, `causal_attention` (scaled dot-product with grouped-query sharing), `AttentionWeights`, `attention_forward` (project, split heads, rotate, attend, merge, project) and the `Attention` layer.

**Decisions:**
- Grouped-query sharing copies each key/value head to its group with `repeat_interleave`, so consecutive query heads share one key/value head. That costs memory proportional to the number of query heads and is fine for a CPU reference; a faster kernel can avoid the copy later.
- The mask is built on every call, `O(T^2)` floats, with `-1e9` above the diagonal. The softmax subtracts the row maximum, so masked entries get weight exactly 0.
- The divisibility checks are written `(a / b) * b == a` instead of `a % b == 0`, because newer clippy flags `%` on unsigned integers (`manual_is_multiple_of`) and Rust 1.75 has no `is_multiple_of`.
- Gradient checks cannot tell a correct gradient of the wrong function from the right one, so the score scaling has its own known-answer test.

**Benchmarks:** `attention_forward_t*` and `attention_backward_t64` in `nn_layers.rs`. In one noisy local run grouped-query and multi-query attention were 4 to 18% faster than full multi-head despite the copy, and `backward` cost 2.44 times a forward pass.

**Tests:** Unit tests for the mask, a single position attending to itself, a known answer that pins down the score scaling, no leakage from the future, grouped attention equalling attention over copied heads, multi-query attention, huge scores staying finite, independence of batch entries, shape errors, and the layer's shape, causality, parameter naming and order, and bad arguments. Gradient checks in `tests/grad_layers.rs` cover equal and grouped head counts and the whole layer with respect to all four weights.

### `mlp.rs`
**What it does:** `swiglu(x, w_gate, w_up, w_down)` and the `SwiGluMlp` layer, `(silu(x * w_gate) * (x * w_up)) * w_down`.

**Benchmarks:** `swiglu_t64` in `nn_layers.rs`: 556 us in one noisy local run.

**Tests:** Unit tests for a known value (with an input where `silu` and `sigmoid` differ), a zero input, the shapes the layer accepts, parameter naming and sizes, and that the down projection uses the residual standard deviation. Gradient checks with respect to the input and all three weights in `tests/grad_layers.rs`.

### `block.rs`
**What it does:** `BlockSpec` (sizes, epsilon and `Init`) and `DecoderBlock`, `h = x + attention(norm(x))` then `h + mlp(norm(h))`. Parameters are named `"{name}.attn_norm"`, `"{name}.attn.{q,k,v,o}_proj"`, `"{name}.mlp_norm"` and `"{name}.mlp.{gate,up,down}_proj"`, each with a `.weight` suffix.

**Benchmarks:** `decoder_block_t64` in `nn_layers.rs`: backward cost 3.27 times a forward pass in one noisy local run.

**Tests:** Unit tests for parameter order, the parameter count against a hand calculation, shape, that zeroed output projections make the block an identity, that both branches add to the stream they read, and a misfit spec.

### `tests/grad_layers.rs`
**What it does:** Finite-difference gradient checks of the layer functions with respect to their inputs and weights: RMSNorm, RoPE (input, and the tables too), causal attention with equal and with grouped head counts, SwiGLU with all three weights, and a whole attention layer with respect to the input and all four projections. One more test checks that `attention_forward` equals the pieces it is built from.

## CI and Workflows
Covered by the workspace workflows in `docs/ci-and-workflows.md`. This crate has no workflow of its own. Its benchmarks live in `docs/midman-bench.md`.

## Fixes and Problems

### `attention.rs`
- **A mutation run found that dropping the `1/sqrt(d)` scale went unnoticed.** Eighteen plausible bugs were injected across this crate, `midman-model` and `midman-optim`, one at a time, and the tests were expected to fail for each. Five survived the first version of the tests. A gradient check cannot catch a missing scale, because the gradient stays consistent with the wrong function. A known-answer test now pins the scaling.
- **A batch-independence check and a huge-score check were added** after the mutation run, because the layer's batch handling and softmax stability had no direct test.

### `norm.rs`
- **A mutation run found that ignoring `eps` went unnoticed.** `eps` is negligible on ordinary inputs, so nothing noticed it was missing. A test with a tiny input now fails without it.
- **Open, not diagnosed: the layer is much slower than a plain loop.** `rms_norm_512x64` took 305 us against 22 us (13.9 times) in one local run. The layer is built from six tensor operations, two of them broadcasts, and records a graph. How the gap divides between those has not been measured.

### `mlp.rs`
- **The first known-value test could not tell `silu` from `sigmoid`.** It used input 1, where `silu(1)` equals `sigmoid(1)`, so a mutant that swapped them survived. The test now uses input 2.

### `block.rs`
- **A mutation run found that taking the second residual from `x` instead of `h` went unnoticed.** The identity test zeroes both output projections, which also makes `h` equal `x`. A test where both branches contribute now compares against the explicit composition.

### `init.rs`
- **Nothing measured the residual-stream scaling.** The mutation run showed that replacing `Init::scaled` with `Init::uniform` in `midman-model` passed every test. The test that catches it is in `midman-model`'s `tests/wiring.rs`.
