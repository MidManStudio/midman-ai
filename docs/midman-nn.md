# midman-nn

## Overview
The layers a decoder-only transformer is built from: a linear layer, an embedding table, RMSNorm, rotary position embeddings, causal grouped-query attention, a SwiGLU MLP, and a pre-norm decoder block. Every layer owns its weights as `Parameter`s from `midman-tensor` and builds its forward pass from tensor operations, so `Tensor::backward` differentiates it with no extra code.

The crate does not know about `ModelConfig`. Every layer takes its sizes explicitly, and `midman-model` assembles them from a config. The math of each layer is also a public function over plain tensors (`rms_norm`, `apply_rope`, `causal_attention`, `attention_forward`, `swiglu`). The layer structs call those functions, so the gradient checks exercise the code the model actually runs, and they can differentiate with respect to the weights.

Dependencies: `midman-foundation` and `midman-tensor`.

Test status when this was written: 47 unit tests, 8 gradient-check tests, 1 doctest. Checked on Rust 1.75 with `fmt`, `clippy -D warnings` and `rustdoc -D warnings`. Newer clippy versions were not available when the code was written.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports the public items. The crate docs include a runnable example.

### `module.rs`
**What it does:** The `Module` trait (`parameters` and `parameters_mut`) and `count_parameters`.

**Design notes:**
- The order of the parameter list is fixed: declaration order, with a layer's children in the order they run. Optimizers and checkpoints depend on it, and a test in `midman-model` pins the full order.
- The trait covers parameter access only. Forward signatures differ per layer (token ids, a rotary table, a plain tensor), so a shared `forward` would only hide that.

### `init.rs`
**What it does:** `Init`, the two standard deviations used to draw weights: `std` for ordinary weights and `out_std` for the projections that write into the residual stream (the attention output and the MLP down projection). `Init::scaled(std, n_layers)` sets `out_std = std / sqrt(2 * n_layers)`; `Init::uniform(std)` uses one value for both.

**Design notes:**
- The residual-stream scaling is the usual GPT-2 style choice that keeps the variance of the stream from growing with depth. The values (0.02 in the model) are defaults to start from and are not tuned.

### `linear.rs`
**What it does:** `Linear`, `y = x * W` with `W` of shape `[in_features, out_features]` and no bias. The weight is named `"{name}.weight"`.

**Design notes:**
- The layout is `[in, out]`, the transpose of PyTorch's `nn.Linear`. This only matters if a checkpoint is ever exchanged with another tool.
- `x` needs at least two axes, because it goes through the shared-right-operand form of `matmul`.

### `embedding.rs`
**What it does:** `Embedding`, a table of shape `[num_embeddings, dim]` with a lookup that takes row-major ids and their shape. A model with tied embeddings also uses `weight()` as its output matrix.

### `norm.rs`
**What it does:** `rms_norm(x, weight, eps)` and the `RmsNorm` layer, `x * (mean(x^2) + eps)^(-1/2) * weight` with the mean over the last axis. The layer's weight starts at 1.

**Design notes:**
- Built from `mul`, `mean_axis`, `add_scalar` and `powf`, so it differentiates with respect to both arguments.
- `weight` must be a vector as long as the last axis. Without that check a weight of shape `[1]` would broadcast silently.

### `rope.rs`
**What it does:** `apply_rope` and `RotaryTables`. Position `p` and frequency index `i` use the angle `p * theta^(-i / half)` with `half = head_dim / 2`. `RotaryTables::apply` takes `[batch, heads, time, head_dim]` and treats the time axis as positions `0..time`.

**Design notes:**
- Layout is "rotate half": the pairs are `(x[i], x[i + half])`, not interleaved neighbours. Any later import of weights trained with the other layout would need a permutation.
- The tables are computed in `f64` and stored as `f32`.
- There is no start-position argument. Nothing needs one until `midman-inference` has a KV cache.

### `attention.rs`
**What it does:** `causal_mask`, `causal_attention` (scaled dot-product with grouped-query sharing), `AttentionWeights`, `attention_forward` (project, split heads, rotate, attend, merge, project) and the `Attention` layer.

**Design notes:**
- Grouped-query sharing copies each key/value head to its group with `repeat_interleave`, so consecutive query heads share one key/value head. That costs memory proportional to the number of query heads and is fine for a CPU reference; a faster kernel can avoid the copy later.
- The mask is built on every call, `O(T^2)` floats, with `-1e9` above the diagonal. The softmax subtracts the row maximum, so masked entries get weight exactly 0.
- The divisibility checks are written `(a / b) * b == a` instead of `a % b == 0`, because newer clippy flags `%` on unsigned integers (`manual_is_multiple_of`) and the toolchain used to write the code has no `is_multiple_of`.
- Gradient checks cannot tell a correct gradient of the wrong function from the right one, so the score scaling has its own known-answer test.

### `mlp.rs`
**What it does:** `swiglu(x, w_gate, w_up, w_down)` and the `SwiGluMlp` layer, `(silu(x * w_gate) * (x * w_up)) * w_down`.

### `block.rs`
**What it does:** `BlockSpec` (sizes, epsilon and `Init`) and `DecoderBlock`, `h = x + attention(norm(x))` then `h + mlp(norm(h))`. Parameters are named `"{name}.attn_norm"`, `"{name}.attn.{q,k,v,o}_proj"`, `"{name}.mlp_norm"` and `"{name}.mlp.{gate,up,down}_proj"`, each with a `.weight` suffix.

### `tests/grad_layers.rs`
**What it does:** Finite-difference gradient checks of the layer functions with respect to their inputs and weights: RMSNorm, RoPE (input, and the tables too), causal attention with equal and with grouped head counts, SwiGLU with all three weights, and a whole attention layer with respect to the input and all four projections. One more test checks that `attention_forward` equals the pieces it is built from.

## CI and Workflows
Covered by the workspace workflows in `docs/ci-and-workflows.md`. This crate has no workflow of its own.

## Fixes and Problems
- **A mutation run found five gaps in the first version of the tests.** Eighteen plausible bugs were injected into this crate, `midman-model` and `midman-optim`, one at a time, and the test suite was expected to fail for each. Five survived: dropping the `1/sqrt(d)` attention scale, ignoring `eps` in RMSNorm, using sigmoid instead of silu in the MLP, taking the second residual from `x` instead of `h`, and removing the residual-stream init scaling. Gradient checks cannot catch the first (the gradient is consistent with the wrong function). The `eps` case needs a tiny input to matter. The MLP known-value test had used input 1, where `silu(1)` and `sigmoid(1)` are equal, so it could not tell them apart. The block's identity test zeroed both branches, which also makes `h == x`. Each now has a test that fails on its mutant, and the full run catches all 18.
- **Clippy `excessive_precision` on test constants.** Reference values copied from a float64 calculation had more digits than `f32` holds. They are rounded to 7 decimals and compared with a tolerance of 1e-6.
