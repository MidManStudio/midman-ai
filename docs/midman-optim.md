# midman-optim

## Overview
The three pieces of a training step that sit between the gradients and the weights: AdamW, a warmup-and-cosine learning-rate schedule, and global gradient-norm clipping.

Gradients travel between these pieces as plain `Vec<Vec<f32>>`, one vector per parameter in the same order as the parameter list. `gather_gradients` moves them out of the autograd result, `clip_global_norm` rescales them in place, and `AdamW::step` applies them. Nothing here needs a change in `midman-tensor`.

Dependencies: `midman-foundation` and `midman-tensor`.

Test status when this was written: 19 unit tests and 1 doctest. Checked with `fmt`, `clippy -D warnings` and `rustdoc -D warnings` on Rust 1.75 and on Rust 1.91, and mutation-tested (see "Fixes and Problems").

## Modules

### `lib.rs`
**What it does:** Crate root. Re-exports `AdamW`, `WarmupCosine`, `gather_gradients`, `global_norm` and `clip_global_norm`. The crate docs include a runnable example of one training loop.

**Tests:** The doctest.

### `adamw.rs`
**What it does:** `AdamW`, with decoupled weight decay in the PyTorch formulation:
- `p *= 1 - lr * weight_decay` (only if `p` has rank 2 or more)
- `m = beta1 * m + (1 - beta1) * g`
- `v = beta2 * v + (1 - beta2) * g * g`
- `p -= lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)`

**Decisions:**
- Weight decay applies to matrices and embeddings but not to vectors (the norm weights). That is the common convention, chosen here as a default and not tuned. Setting `weight_decay` to 0 disables it.
- Moments are stored per parameter name in a `BTreeMap`, so the order is deterministic and the state can be saved later without the optimizer holding references to parameters. Parameter names must therefore be unique, and `step` rejects duplicates.
- The learning rate is an argument of `step`. The schedule lives outside the optimizer.
- `step` validates everything first (list lengths, gradient sizes, non-finite gradients, duplicate names, size changes, `lr`) and only then changes anything, so a rejected step leaves the parameters, moments and step count untouched.

**Benchmarks:** `adamw_step_65536` and `adamw_step_1048576` in `optim_step.rs`. In one noisy local run the step took 1.44 to 1.45 times as long as a plain-loop implementation of the same rule. See "Fixes and Problems" below and `docs/midman-bench.md`.

**Tests:** Unit tests for three steps against an independent float64 Python implementation of the same update (one matrix parameter and one vector parameter), that only matrices are decayed, a zero learning rate, that a rejected step changes nothing, duplicate names, a parameter that changes size, bad hyperparameters, that scaling every gradient does not change the trajectory (Adam's scale invariance, with a negligible epsilon), and convergence on a quadratic through real gradients from `backward`.

### `clip.rs`
**What it does:** `gather_gradients`, `global_norm` and `clip_global_norm`.

**Decisions:**
- `gather_gradients` returns an error naming the first parameter with no gradient. Every parameter of a transformer is used on every step, so a missing gradient means a disconnected forward pass, and training silently skipping it would hide the bug.
- The norm is accumulated in `f64`. `clip_global_norm` returns the norm before clipping, scales by `max_norm / (norm + 1e-6)` only when the norm exceeds the limit, and returns an error without modifying anything if the norm is not finite.

**Benchmarks:** `clip_1m` in `optim_step.rs`: global norm 747 us and clip with rescaling 1.16 ms over one million values, in one noisy local run.

**Tests:** Unit tests for the norm across gradients, rescaling to the limit while keeping directions, gradients within the limit left untouched, a non-finite norm changing nothing, gathering in parameter order, and the parameter name in the missing-gradient error.

### `schedule.rs`
**What it does:** `WarmupCosine`. For the zero-based step index `s`: `max_lr * (s + 1) / warmup` during warmup, `max_lr` at `s = warmup`, then half a cosine down to `min_lr` at `s = total`, then constant. Constructed with `new(max_lr, min_lr, warmup_steps, total_steps)`, which checks `0 <= min_lr <= max_lr` and `warmup_steps < total_steps`.

**Decisions:**
- The first step already has a nonzero rate (`s + 1` rather than `s`).

**Tests:** Unit tests for values at nine steps against a separate float64 calculation, that warmup rises and the decay never rises, a schedule with no warmup, and bad arguments.

## CI and Workflows
Covered by the workspace workflows in `docs/ci-and-workflows.md`. This crate has no workflow of its own. Its benchmarks live in `docs/midman-bench.md`.

## Fixes and Problems

### `adamw.rs`
- **Clippy `excessive_precision` on test constants.** The reference values came from a float64 calculation and had more digits than `f32` holds. They are rounded to 7 decimals and compared with a tolerance of 1e-6.
- **Open, not diagnosed: the step is 1.45 times slower than a plain loop.** Unlike the baseline, `step` makes an extra pass over every gradient to reject non-finite values, and checks names and sizes. Whether that accounts for the gap has not been measured.
- **Mutation run.** Four injected bugs in this file (no bias correction, decay on vectors, beta2 using beta1, eps outside the scaling) were all caught by the tests as first written.

### `clip.rs`
- **Mutation run.** An inverted clip factor was caught by the tests as first written.

### `schedule.rs`
- **Clippy `excessive_precision` on test constants.** Trailing-zero literals such as `0.000_250_000` are written in scientific notation instead (`2.5e-4`), which no clippy version has an opinion about.
- **Mutation run.** A warmup off by one and an un-halved cosine were both caught by the tests as first written.
