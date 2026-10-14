# midman-bench

## Overview
Criterion benchmarks for the midman-ai crates, in a package of their own under `benches/`. Four bench targets hold 63 benchmarks in 26 groups: tensor kernels and the cost of autograd, the `midman-nn` layers, whole-model forward and training steps, and the optimizer. The package also compiles a small set of C kernels (`c/baseline_kernels.c`), so the Rust kernels can be compared with a plain C implementation of the same operation.

The package is separate because criterion pulls in `clap_builder`, which needs a Cargo that understands edition 2024 (about 1.85 or newer). Cargo resolves a package's whole manifest, dev-dependencies included, before it builds any target from it, so a criterion dev-dependency inside a library crate would raise the toolchain requirement for that crate's tests as well. Kept here, the requirement stops at this package. See "Toolchain walls" in `docs/workspace-cargo.md`.

Conventions every bench file follows:
- A benchmark id is `group/variant`, and a group holds the variants that are compared with each other.
- A variant named `baseline-*` is a reference implementation of the same operation (plain loops over raw slices). `scripts/bench_summary.py` puts a ratio against it and a badge next to every other variant in the group. Each baseline is checked against the code under test in the setup, before anything is timed, and the benchmark panics if they disagree, so two different computations are never compared.
- A variant named `baseline-c-*` is the same operation in plain C, called through the safe wrappers in `src/lib.rs`. It is registered after the Rust baseline, so the Rust baseline stays the group's denominator and the C variant gets its own ratio against it. In a group with C variants the summary also adds a `vs fastest C` column that compares every variant with the fastest C one. The C kernels use the same algorithm and the same order of floating-point operations as the Rust baselines, with fused multiply-add and fast-math off, so a difference between them comes from the compiler and the memory access pattern and not from a different computation. The C compiler is printed once at the start of each bench run and shown in the summary header.
- A variant named `unit-*` is only a cost denominator, such as the forward pass. The summary shows a plain multiple for the other variants and no badge, because a backward pass costing about three forward passes is expected, not a regression.
- Throughput counts elements per second: floating-point operations for matrix products, tokens for training steps, parameters for model construction, array elements otherwise.

Dependencies: `midman-foundation`, `midman-model-config`, `midman-tensor`, `midman-nn`, `midman-model`, `midman-optim`, `criterion` (a workspace dependency, with `html_reports`) as a dev-dependency, and `cc` as a build dependency that compiles the C kernels. The package has a library target (`src/lib.rs`) holding the C wrappers and their tests, and a `build.rs`. It needs a C compiler on the machine: `gcc` on the Ubuntu runners, `clang` on macOS.

The numbers below come from one local run on a shared single-CPU sandbox: release build without LTO, criterion timings shortened to a 1 second warm-up, 2 seconds of measurement and 10 samples. They check that the benchmarks and the parser work and show rough proportions. They are not reference figures. Reference figures come from `benchmarks.yml`, which uses the real bench profile.

The C comparison numbers below come from a second local run on the same sandbox, with gcc 13.3: the bench profile (fat LTO) for `tensor_ops` and `nn_layers`, a release build without LTO for `optim_step`, and criterion shortened to 0.3 seconds of warm-up, 0.5 seconds of measurement and 10 samples. Same caveats: one noisy single-CPU machine, so read the ratios and not the microseconds.

## Modules

### `tensor_ops.rs`
**What it does:** Benchmarks the `midman-tensor` kernels: square matrix products at three sizes, the linear-layer and attention-score product shapes of the Smoke preset, softmax, SiLU, a broadcast add, a two-layer MLP with and without autograd, and filling a matrix with normal random values.

**Decisions:**
- The matrix-product baseline is the textbook i-j-k loop. The library kernel uses i-p-j order, so the comparison also shows what loop order buys.
- The MLP group has three variants of one computation: `unit-forward-untracked` (no graph is built), `forward-tracked` (a graph is built and dropped) and `forward-backward`. The first two show the cost of recording the graph, the third the cost of `backward`.
- Sizes follow the Smoke preset (width 64, 4 heads, feed-forward width 176).
- The matrix product, softmax, SiLU and broadcast add each have `baseline-c-*` variants. The product has two: `baseline-c-ijk` (the same loop order as the Rust baseline) and `baseline-c-ikj` (inner loop along a row, which a compiler can vectorize). Both sum every element in the same order as the Rust baseline, so they give bit-identical results to each other. The fastest C variant is the reference in the `vs fastest C` column, so for the product it is the ikj version, the best plain C a person would write.

**Benchmarks:** (one noisy local run, see above)
- Matrix product, 32 / 96 / 192 square: 4.9 us / 91.5 us / 875 us against 24.8 us / 637 us / 5.42 ms for the baseline, which is 5 to 7 times faster, at 13 to 19 GFLOP/s against 2.6 to 2.8.
- Linear layer shapes: 164 us (gate) and 266 us (head), about 16 to 18 GFLOP/s. Batched attention scores: 20.5 us at T=32, 235 us at T=128.
- Softmax over 256 rows of 128: 217 us against 132 us for the baseline (1.64 times slower).
- SiLU over 65,536 values: 198 us against 187 us (1.06 times).
- Broadcast add of a 512-vector to a 256x512 matrix: 305 us against 128 us (2.39 times slower).
- Two-layer MLP: 650 us untracked, 677 us tracked (1.04 times), 2.10 ms with `backward` (3.23 times).
- Filling a 1024x1024 matrix with normals: 20.4 ms, about 51 million values per second.
- Against plain C (second local run): `midman-tensor` is 1.13, 1.16 and 1.12 times the C ikj product at 32, 96 and 192 (4.5 / 97.8 / 774 us against 4.0 / 84.0 / 690 us). C ijk is 0.71 to 0.91 times the Rust i-j-k baseline, so the C compiler beats the Rust compiler by 10 to 30 percent on the same loop.
- Softmax over 256 rows of 128: C 165 us, Rust baseline 117 us, `midman-tensor` 202 us (1.23 times C). SiLU over 65,536 values: C 247 us, Rust baseline 164 us, `midman-tensor` 181 us (0.74 times C). Both C versions are slower than the Rust baselines, which is the opposite of the product; why has not been checked.
- Broadcast add: C 25.7 us, Rust baseline 112 us (4.35 times C), `midman-tensor` 317 us (12.3 times C).

**Tests:** The baselines, Rust and C, are checked against the tensor operations in the setup (matrix product 1e-4, softmax 1e-5, SiLU 1e-5, broadcast add 1e-6, relative to magnitude). Correctness of the operations themselves is covered by the `midman-tensor` tests, and the C kernels have their own tests in `lib.rs`.

### `nn_layers.rs`
**What it does:** Benchmarks the `midman-nn` layers at the Smoke preset width: attention forward with 4, 2 and 1 key/value heads at three sequence lengths, attention with and without `backward`, RMSNorm, rotary embeddings, the SwiGLU MLP, and a whole decoder block with and without `backward`.

**Decisions:**
- The attention group varies only the key/value head count. Full multi-head is the baseline, so the ratio shows what grouped-query and multi-query sharing change in practice, including the cost of copying each key/value head to its group.
- The RMSNorm baselines are a plain Rust loop over rows and the same loop in C.

**Benchmarks:** (one noisy local run)
- Attention forward, multi-head / grouped-query (2 kv heads) / multi-query, at T=16, 64, 128: 159 us, 1.00 ms and 3.99 ms for multi-head; grouped-query is 0.83, 0.89 and 0.96 times that, multi-query 0.82, 0.84 and 0.93 times.
- Attention with `backward` at T=64: 2.12 ms against 870 us forward (2.44 times). Decoder block at T=64: 7.23 ms against 2.21 ms (3.27 times).
- RMSNorm over 512 rows of 64: 305 us against 22 us for the baseline (13.9 times slower).
- Rotary embeddings at T=128: 116 us. SwiGLU at T=64: 556 us.
- RMSNorm against plain C (second local run): C 23.5 us, Rust baseline 23.8 us (parity), `midman-nn` 216 us (9.2 times C). The two baselines agree, so the gap is not an artifact of how the baseline was written.

**Tests:** Both RMSNorm baselines are checked against the layer in the setup (1e-4). Layer correctness is covered by the `midman-nn` tests.

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
- The AdamW baselines are a plain Rust loop and a plain C loop of the same update rule, with the same order of floating-point operations. Before timing, one step of each is run on the same data and the results are compared (1e-5), so the baselines really are the same algorithm.
- The clipping benchmark uses a limit small enough that every call rescales, which is the slower path. Gradients are cloned in the untimed setup of each batch.

**Benchmarks:** (one noisy local run)
- AdamW: 78 us against 54 us for the baseline at 65,536 values, and 1.26 ms against 870 us at 1,048,576 values (1.44 and 1.45 times).
- Global norm over one million values: 747 us. Clip and rescale: 1.16 ms.
- AdamW against plain C (second local run): C 63.9 us and 930 us, Rust baseline 54.3 us and 854 us, `midman-optim` 80.9 us and 1.295 ms. That is 1.27 and 1.39 times the C loop, and 1.49 and 1.52 times the Rust loop. The C loop is the slower of the two loops here.

**Tests:** Both AdamW baselines are checked against the optimizer in the setup. Optimizer correctness is covered by the `midman-optim` tests.

### `lib.rs`
**What it does:** The library target of the benchmark package. It declares the C kernels with `extern "C"` and wraps each one in a safe function that takes slices and returns a `Vec<f32>` (`matmul_ijk`, `matmul_ikj`, `softmax_rows`, `silu`, `broadcast_add`, `rms_norm`), plus `CAdamW` for the in-place optimizer step and `compiler()` for the C compiler's version string.

**Decisions:**
- All of the package's `unsafe` is here, behind `#![allow(unsafe_code)]` at the crate root with a comment, and every block has a `// SAFETY:` comment. The bench files themselves contain no `unsafe`, so the workspace's `unsafe_code = "deny"` still applies to them.
- Each wrapper checks its buffer lengths with a real `assert!` before it calls C. A wrong length would be an out-of-bounds read in C, so this is a safety check and not a debug aid, and the cost is a few comparisons per call.
- The wrappers allocate their output the way the Rust baselines do (a fresh `Vec` per call), so allocation cost is on both sides of the comparison. `CAdamW::update` works in place, like the Rust AdamW loop.
- The tests live next to the wrappers in this file, so `cargo test -p midman-bench` checks the C kernels without running any benchmark.

**Tests:** 17 unit tests. Each C kernel is compared with the `midman-tensor`, `midman-nn` or `midman-optim` implementation it stands in for (matrix product 1e-4, softmax 1e-5, SiLU 1e-5, RMSNorm 1e-4, AdamW over five steps 1e-5), against known answers (a 2x2 product, SiLU of 0 and 1, an RMSNorm row of `[3, 4]` with and without `eps`), and for edge cases: size zero, no rows, a ragged last row, and softmax over values near 1000, which overflows unless the row maximum is subtracted first. The two matrix loop orders are asserted bit-identical, and wrong buffer lengths are asserted to panic before reaching C.

### `build.rs`
**What it does:** Compiles `c/baseline_kernels.c` into a static library with the `cc` crate and links it into the package.

**Decisions:**
- The standard is C11 and `-Wpedantic` is on, so the kernels stay portable C. Warnings are not errors here, because a new compiler version that adds a warning should not stop a benchmark build.
- `-ffp-contract=off` keeps the C compiler from fusing multiply-adds. Clang on Apple silicon does that by default, and the Rust baselines never do, so without the flag the same loop would compute a different result on a Mac.
- `-fno-math-errno` lets the compiler use the hardware square root and vectorize around it. Rust's `sqrt` has no errno side effect, and without the flag the C version would be slower for a reason that has nothing to do with the algorithm.
- There is no `-march=native` and no `-ffast-math`. A committed build must run on the machine that is not the one that built it, and fast-math would change the arithmetic.
- `cc` reads `CC` and `CFLAGS` from the environment, so a different compiler can be tried without a code change.

### `baseline_kernels.h`
**What it does:** The prototypes of the C kernels, for C and C++ callers. Hand-written.

**Decisions:**
- No header generator is used. The Rust declarations in `lib.rs` are written by hand to match, and the tests call every function, so a prototype that drifts shows up as a failing test.
- The prototypes carry no `restrict`. It is part of the definitions in the `.c` file, where it affects the code, and a top-level `restrict` on a parameter does not change a function's type.
- Lengths are element counts as `size_t`, which is `usize` on the Rust side.

### `baseline_kernels.c`
**What it does:** Plain C versions of the operations the benchmarks time: a square matrix product in i-j-k and i-k-j order, row softmax, SiLU, a broadcast add, RMSNorm and one AdamW step.

**Decisions:**
- Each function does what the Rust baseline does in the same order. In particular every sum is sequential, which the compiler may not reorder without fast-math, so the C and Rust baselines are the same computation and can be compared.
- The pointer arguments are `restrict`, which matches what Rust's slices already promise the compiler (a mutable slice does not alias anything). Without it the C code would be compared at a disadvantage.
- `mmb_matmul_ikj` is the version a competent C programmer would write for speed. It is there so the `vs fastest C` column compares `midman-tensor` with something worth beating, not with the slow textbook loop.
- The functions do not allocate. The wrappers in `lib.rs` own all memory.
- `mmb_compiler` returns `__VERSION__`, so a benchmark log says which compiler produced the C numbers. The number depends on it: gcc and clang vectorize these loops differently.

**Benchmarks:** see `tensor_ops.rs`, `nn_layers.rs` and `optim_step.rs`.

**Tests:** `lib.rs` calls every function in this file. A bit-exact test pins the ikj order against the ijk order, and a large-value test pins the softmax's use of the row maximum.

## CI and Workflows
`benchmarks.yml` runs this package by default. `scripts/bench_summary.py` turns the criterion log into the job summary and `bench-results.json`, and `scripts/test_bench_summary.py` tests it against real captured output. The summary shows the C compiler in its header and adds a `vs fastest C` column to every group that has a `baseline-c-*` variant. The JSON gains `meta.c_compiler`, a `c_reference` per group and a `ratio_c` per row; the site's benchmarks page ignores them for now. See `docs/ci-and-workflows.md`.

`ci.yml` runs the unit tests of this package with the rest of the workspace, on `ubuntu-latest`, which compiles the C kernels with gcc. `benchmarks.yml` with `platforms: all` compiles them with clang on `macos-latest` and with gcc on `ubuntu-24.04-arm` as well. The macOS and ARM builds have not been run yet.

## Fixes and Problems

### `Cargo.toml`
- **`missing_docs` fired on the macro-generated function.** `criterion_group!` expands to an undocumented `pub fn`, so the workspace's `missing_docs = "warn"` failed `clippy -D warnings`. Each bench file now opens with `#![allow(missing_docs)]` and a comment saying why.
- **The first lockfile update bumped `Cargo.lock` to version 4.** Cargo 1.91 rewrote the version 3 lockfile as version 4 when it added criterion's packages, and Cargo 1.75 cannot read version 4, which would have extended the toolchain requirement to every crate. The lockfile is version 3 again, and Cargo 1.91 keeps it that way on later commands. After a `cargo update` with a new Cargo, check that the line still says `version = 3`.
- **Adding `cc` did it again.** The first build after adding the build dependency rewrote the lockfile as version 4 once more. The line was set back to 3 by hand, `cargo metadata --locked` and `cargo test -p midman-foundation --locked` were checked on Rust 1.75 afterwards, and the three new packages (`cc`, `shlex`, `find-msvc-tools`) are in the lockfile.

### `tensor_ops.rs`
- **Open, not diagnosed: two kernels are slower than a plain loop.** Softmax is 1.64 times and the broadcast add is 2.39 times slower than their baselines. RMSNorm in `nn_layers.rs` uses two broadcasts, so the broadcast path is a suspect for it, but that is a guess and nothing here has profiled any of them.
- **The C loop shows the Rust broadcast baseline was not a floor.** The plain Rust baseline for the broadcast add (a `flat_map` over `zip`) is itself 4.35 times slower than the same loop in C, so the 2.39 above understated the gap: `midman-tensor`'s broadcast add is 12.3 times the C loop. The Rust baseline is kept as it is, because it is the one the history of this benchmark is measured against.
- **Open, not diagnosed: the C softmax and SiLU are slower than the Rust baselines.** C takes 165 us against 117 us for softmax and 247 us against 164 us for SiLU, while the C matrix product and broadcast add are faster than their Rust baselines. Both slow cases call `expf` for every value. Whether the two compilers handle that call differently has not been checked.

### `nn_layers.rs`
- **Open, not diagnosed: RMSNorm is 13.9 times slower than a plain loop.** The layer is built from six tensor operations, two of them broadcasts, and records a graph. How the 13.9 divides between those has not been measured. The plain C loop runs in the same time as the plain Rust loop (23.5 us and 23.8 us), so the target for a fix is the same either way.

### `model_step.rs`
- **First version had no `unit-*` convention.** The forward pass was named `baseline-forward`, so the summary showed `4.06x` with a failure badge for backward costing four forward passes, which is expected. Forward-only variants are now `unit-forward` and get a plain multiple.

### `optim_step.rs`
- **Open, not diagnosed: AdamW is 1.45 times slower than the plain-loop baseline.** Unlike the baseline, `step` validates every gradient for non-finite values and every parameter for size and name before it changes anything, which costs an extra pass over the gradients. Whether that explains the gap has not been measured.
