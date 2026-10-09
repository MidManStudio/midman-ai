# midman-bench

## Overview
Criterion benchmarks for the midman-ai crates, in a package of their own under `benches/`. Four bench targets hold 51 benchmarks in 26 groups: tensor kernels and the cost of autograd, the `midman-nn` layers, whole-model forward and training steps, and the optimizer.

The package is separate because criterion pulls in `clap_builder`, which needs a Cargo that understands edition 2024 (about 1.85 or newer). Cargo resolves a package's whole manifest, dev-dependencies included, before it builds any target from it, so a criterion dev-dependency inside a library crate would raise the toolchain requirement for that crate's tests as well. Kept here, the requirement stops at this package. See "Toolchain walls" in `docs/workspace-cargo.md`.

Conventions every bench file follows:
- A benchmark id is `group/variant`, and a group holds the variants that are compared with each other.
- A variant named `baseline-*` is a reference implementation of the same operation (plain loops over raw slices). `scripts/bench_summary.py` puts a ratio against it and a badge next to every other variant in the group. Each baseline is checked against the code under test in the setup, before anything is timed, and the benchmark panics if they disagree, so two different computations are never compared.
- A variant named `unit-*` is only a cost denominator, such as the forward pass. The summary shows a plain multiple for the other variants and no badge, because a backward pass costing about three forward passes is expected, not a regression.
- Throughput counts elements per second: floating-point operations for matrix products, tokens for training steps, parameters for model construction, array elements otherwise.

Dependencies: `midman-foundation`, `midman-model-config`, `midman-tensor`, `midman-nn`, `midman-model`, `midman-optim`, and `criterion` (a workspace dependency, with `html_reports`) as a dev-dependency.

The numbers below come from one local run on a shared single-CPU sandbox: release build without LTO, criterion timings shortened to a 1 second warm-up, 2 seconds of measurement and 10 samples. They check that the benchmarks and the parser work and show rough proportions. They are not reference figures. Reference figures come from `benchmarks.yml`, which uses the real bench profile.

## Modules

### `tensor_ops.rs`
**What it does:** Benchmarks the `midman-tensor` kernels: square matrix products at three sizes, the linear-layer and attention-score product shapes of the Smoke preset, softmax, SiLU, a broadcast add, a two-layer MLP with and without autograd, and filling a matrix with normal random values.

**Decisions:**
- The matrix-product baseline is the textbook i-j-k loop. The library kernel uses i-p-j order, so the comparison also shows what loop order buys.
- The MLP group has three variants of one computation: `unit-forward-untracked` (no graph is built), `forward-tracked` (a graph is built and dropped) and `forward-backward`. The first two show the cost of recording the graph, the third the cost of `backward`.
- Sizes follow the Smoke preset (width 64, 4 heads, feed-forward width 176).

**Benchmarks:** (one noisy local run, see above)
- Matrix product, 32 / 96 / 192 square: 4.9 us / 91.5 us / 875 us against 24.8 us / 637 us / 5.42 ms for the baseline, which is 5 to 7 times faster, at 13 to 19 GFLOP/s against 2.6 to 2.8.
- Linear layer shapes: 164 us (gate) and 266 us (head), about 16 to 18 GFLOP/s. Batched attention scores: 20.5 us at T=32, 235 us at T=128.
- Softmax over 256 rows of 128: 217 us against 132 us for the baseline (1.64 times slower).
- SiLU over 65,536 values: 198 us against 187 us (1.06 times).
- Broadcast add of a 512-vector to a 256x512 matrix: 305 us against 128 us (2.39 times slower).
- Two-layer MLP: 650 us untracked, 677 us tracked (1.04 times), 2.10 ms with `backward` (3.23 times).
- Filling a 1024x1024 matrix with normals: 20.4 ms, about 51 million values per second.

**Tests:** The baselines are checked against the tensor operations in the setup (matrix product 1e-4, softmax 1e-5, SiLU 1e-5, broadcast add 1e-6, relative to magnitude). Correctness of the operations themselves is covered by the `midman-tensor` tests.

### `nn_layers.rs`
**What it does:** Benchmarks the `midman-nn` layers at the Smoke preset width: attention forward with 4, 2 and 1 key/value heads at three sequence lengths, attention with and without `backward`, RMSNorm, rotary embeddings, the SwiGLU MLP, and a whole decoder block with and without `backward`.

**Decisions:**
- The attention group varies only the key/value head count. Full multi-head is the baseline, so the ratio shows what grouped-query and multi-query sharing change in practice, including the cost of copying each key/value head to its group.
- The RMSNorm baseline is a plain loop over rows.

**Benchmarks:** (one noisy local run)
- Attention forward, multi-head / grouped-query (2 kv heads) / multi-query, at T=16, 64, 128: 159 us, 1.00 ms and 3.99 ms for multi-head; grouped-query is 0.83, 0.89 and 0.96 times that, multi-query 0.82, 0.84 and 0.93 times.
- Attention with `backward` at T=64: 2.12 ms against 870 us forward (2.44 times). Decoder block at T=64: 7.23 ms against 2.21 ms (3.27 times).
- RMSNorm over 512 rows of 64: 305 us against 22 us for the baseline (13.9 times slower).
- Rotary embeddings at T=128: 116 us. SwiGLU at T=64: 556 us.

**Tests:** The RMSNorm baseline is checked against the layer in the setup (1e-4). Layer correctness is covered by the `midman-nn` tests.

### `model_step.rs`
**What it does:** Benchmarks whole-model work: a forward pass, forward plus backward, and a full training step (forward, backward, gradient clipping and AdamW) for the Smoke preset (4 sequences of 32 tokens) and the Tiny preset (1 sequence of 32 tokens), and the time to construct each model.

**Decisions:**
- The step groups use tokens as their throughput, so the raw log reports tokens per second.
- The Tiny group is slow, so it uses 10 samples and 10 seconds of measurement. Criterion would otherwise warn that it could not finish the default sample count in the default time.
- The full-step variant keeps training the same model, so the loss changes between iterations. The cost of an iteration does not.

**Benchmarks:** (one noisy local run)
- Smoke, 128 tokens: forward 3.39 ms (about 37.8 thousand tokens per second), forward plus backward 15.25 ms (4.50 times, 8.4 thousand tokens per second), full step 15.01 ms. At this size the optimizer and clipping are within the noise of the backward pass.
- Tiny, 32 tokens: forward 8.97 ms, forward plus backward 51.4 ms (5.73 times), full step 55.4 ms, about 580 tokens per second.
- Construction: Smoke 2.30 ms, Tiny 25.4 ms, about 51 million parameters per second.

**Tests:** No assertions beyond the unwraps. Model correctness is covered by the `midman-model` tests.

### `optim_step.rs`
**What it does:** Benchmarks `midman-optim`: an AdamW step over one matrix parameter of 65,536 and of 1,048,576 values, the global gradient norm, and clipping with rescaling over one million values.

**Decisions:**
- The AdamW baseline is a plain loop of the same update rule. Before timing, one step of each is run on the same data and the results are compared (1e-5), so the baseline really is the same algorithm.
- The clipping benchmark uses a limit small enough that every call rescales, which is the slower path. Gradients are cloned in the untimed setup of each batch.

**Benchmarks:** (one noisy local run)
- AdamW: 78 us against 54 us for the baseline at 65,536 values, and 1.26 ms against 870 us at 1,048,576 values (1.44 and 1.45 times).
- Global norm over one million values: 747 us. Clip and rescale: 1.16 ms.

**Tests:** The AdamW baseline is checked against the optimizer in the setup. Optimizer correctness is covered by the `midman-optim` tests.

## CI and Workflows
`benchmarks.yml` runs this package by default. `scripts/bench_summary.py` turns the criterion log into the job summary and `bench-results.json`, and `scripts/test_bench_summary.py` tests it against real captured output. See `docs/ci-and-workflows.md`.

## Fixes and Problems

### `Cargo.toml`
- **`missing_docs` fired on the macro-generated function.** `criterion_group!` expands to an undocumented `pub fn`, so the workspace's `missing_docs = "warn"` failed `clippy -D warnings`. Each bench file now opens with `#![allow(missing_docs)]` and a comment saying why.
- **The first lockfile update bumped `Cargo.lock` to version 4.** Cargo 1.91 rewrote the version 3 lockfile as version 4 when it added criterion's packages, and Cargo 1.75 cannot read version 4, which would have extended the toolchain requirement to every crate. The lockfile is version 3 again, and Cargo 1.91 keeps it that way on later commands. After a `cargo update` with a new Cargo, check that the line still says `version = 3`.

### `tensor_ops.rs`
- **Open, not diagnosed: two kernels are slower than a plain loop.** Softmax is 1.64 times and the broadcast add is 2.39 times slower than their baselines. RMSNorm in `nn_layers.rs` uses two broadcasts, so the broadcast path is a suspect for it, but that is a guess and nothing here has profiled any of them.

### `nn_layers.rs`
- **Open, not diagnosed: RMSNorm is 13.9 times slower than a plain loop.** The layer is built from six tensor operations, two of them broadcasts, and records a graph. How the 13.9 divides between those has not been measured.

### `model_step.rs`
- **First version had no `unit-*` convention.** The forward pass was named `baseline-forward`, so the summary showed `4.06x` with a failure badge for backward costing four forward passes, which is expected. Forward-only variants are now `unit-forward` and get a plain multiple.

### `optim_step.rs`
- **Open, not diagnosed: AdamW is 1.45 times slower than the plain-loop baseline.** Unlike the baseline, `step` validates every gradient for non-finite values and every parameter for size and name before it changes anything, which costs an extra pass over the gradients. Whether that explains the gap has not been measured.
