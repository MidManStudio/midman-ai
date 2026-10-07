# midman-optim

## Overview
The three pieces of a training step that sit between the gradients and the weights: AdamW, a warmup-and-cosine learning-rate schedule, and global gradient-norm clipping.

Gradients travel between these pieces as plain `Vec<Vec<f32>>`, one vector per parameter in the same order as the parameter list. `gather_gradients` moves them out of the autograd result, `clip_global_norm` rescales them in place, and `AdamW::step` applies them. Nothing here needs a change in `midman-tensor`.

Dependencies: `midman-foundation` and `midman-tensor`.

Test status when this was written: 18 unit tests and 1 doctest. Checked on Rust 1.75 with `fmt`, `clippy -D warnings` and `rustdoc -D warnings`. Newer clippy versions were not available when the code was written.

## Modules

### `lib.rs`
**What it does:** Crate root. Re-exports `AdamW`, `WarmupCosine`, `gather_gradients`, `global_norm` and `clip_global_norm`. The crate docs include a runnable example of one training loop.

### `adamw.rs`
**What it does:** `AdamW`, with decoupled weight decay in the PyTorch formulation:
- `p *= 1 - lr * weight_decay` (only if `p` has rank 2 or more)
- `m = beta1 * m + (1 - beta1) * g`
- `v = beta2 * v + (1 - beta2) * g * g`
- `p -= lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)`

**Design notes:**
- Weight decay applies to matrices and embeddings but not to vectors (the norm weights). That is the common convention, chosen here as a default and not tuned. Setting `weight_decay` to 0 disables it.
- Moments are stored per parameter name in a `BTreeMap`, so the order is deterministic and the state can be saved later without the optimizer holding references to parameters. Parameter names must therefore be unique, and `step` rejects duplicates.
- The learning rate is an argument of `step`. The schedule lives outside the optimizer.
- `step` validates everything first (list lengths, gradient sizes, non-finite gradients, duplicate names, size changes, `lr`) and only then changes anything, so a rejected step leaves the parameters, moments and step count untouched.
- Checked against an independent float64 Python implementation of the same update over three steps, with one matrix and one vector parameter, and against a quadratic that converges through real gradients from `backward`.

### `clip.rs`
**What it does:** `gather_gradients`, `global_norm` and `clip_global_norm`.

**Design notes:**
- `gather_gradients` returns an error naming the first parameter with no gradient. Every parameter of a transformer is used on every step, so a missing gradient means a disconnected forward pass, and training silently skipping it would hide the bug.
- The norm is accumulated in `f64`. `clip_global_norm` returns the norm before clipping, scales by `max_norm / (norm + 1e-6)` only when the norm exceeds the limit, and returns an error without modifying anything if the norm is not finite.

### `schedule.rs`
**What it does:** `WarmupCosine`. For the zero-based step index `s`: `max_lr * (s + 1) / warmup` during warmup, `max_lr` at `s = warmup`, then half a cosine down to `min_lr` at `s = total`, then constant. Constructed with `new(max_lr, min_lr, warmup_steps, total_steps)`, which checks `0 <= min_lr <= max_lr` and `warmup_steps < total_steps`.

**Design notes:**
- The first step already has a nonzero rate (`s + 1` rather than `s`).
- Values at nine steps match a separate float64 calculation.

## CI and Workflows
Covered by the workspace workflows in `docs/ci-and-workflows.md`. This crate has no workflow of its own.

## Fixes and Problems
- **Clippy `excessive_precision` on test constants.** The AdamW and schedule reference values came from a float64 calculation and had more digits than `f32` holds. They are rounded to 7 decimals (and the trailing-zero literal to `0.001`), compared with a tolerance of 1e-6 and 1e-8 respectively.
- **Mutation run.** Seven injected bugs in this crate (no bias correction, decay on vectors, beta2 using beta1, eps outside the scaling, inverted clip factor, warmup off by one, un-halved cosine) were all caught by the tests as first written. See `docs/midman-nn.md` for the five gaps the same run found elsewhere.
