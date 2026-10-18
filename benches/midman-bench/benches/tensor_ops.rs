// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "tensor_ops.rs"
// ============================================================================
//! Benchmarks for the `midman-tensor` kernels and for the cost of autograd.
//!
//! Every benchmark id is `group/variant`. A variant named `baseline-*` is a
//! reference implementation of the same operation and is badged by
//! `scripts/bench_summary.py`; a variant named `unit-*` is only a cost
//! denominator (such as the forward pass) and gets a plain multiple. Each
//! baseline is checked against the code under test before anything is timed.
//!
//! `baseline-naive-*` is plain Rust. `baseline-c-*` is the same operation in plain
//! C (`c/baseline_kernels.c`); the summary also shows every variant's ratio against
//! the fastest C one. The Rust baseline is registered first, so it stays the
//! group's denominator.
//!
//! This package builds `midman-tensor` with its `avx` feature, so on a CPU with AVX the
//! `midman-tensor` variants of the matrix-product groups run the AVX kernels. Each of
//! them has a `midman-tensor-scalar` twin that forces the scalar kernels, so one run
//! shows what AVX changes. Before timing, the setup checks that both give the same
//! bits. Where AVX is not in use (another architecture, or a CPU without it) the twins
//! are left out.

// `criterion_group!` expands to an undocumented `pub fn`, which `missing_docs` flags.
#![allow(missing_docs)]

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use midman_foundation::Rng;
use midman_tensor::kernels::{self, Backend};
use midman_tensor::{Parameter, Tensor};
use std::hint::black_box;

fn uniform(shape: &[usize], seed: u64) -> Tensor {
    Tensor::uniform(shape, -1.0, 1.0, &mut Rng::seed_from_u64(seed)).unwrap()
}

/// True when the matrix-product kernels are running on AVX, so a scalar twin of a
/// benchmark is worth timing.
fn avx_active() -> bool {
    kernels::backend() == Backend::Avx
}

/// Checks that the AVX and scalar kernels give the same bits for `product`, which is
/// what lets the twin benchmarks compare speed and nothing else.
fn assert_backends_agree(product: impl Fn() -> Tensor) {
    if !avx_active() {
        return;
    }
    let on = product();
    kernels::set_force_scalar(true);
    let off = product();
    kernels::set_force_scalar(false);
    let bits = |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&on), bits(&off), "the AVX and scalar kernels disagree");
}

/// Registers `id` as a copy of `routine` that runs with the scalar kernels forced, when
/// AVX is in use. The switch is restored before the next benchmark starts.
fn scalar_twin<R>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    mut routine: impl FnMut() -> R,
) {
    if !avx_active() {
        return;
    }
    group.bench_function(id, |bench| {
        kernels::set_force_scalar(true);
        bench.iter(&mut routine);
        kernels::set_force_scalar(false);
    });
}

fn assert_close(reference: &[f32], got: &[f32], tol: f32) {
    assert_eq!(reference.len(), got.len(), "the baseline and the tensor op disagree on length");
    for (r, g) in reference.iter().zip(got) {
        assert!(
            (r - g).abs() <= tol * (1.0 + r.abs().max(g.abs())),
            "the baseline and the tensor op disagree: {r} vs {g}"
        );
    }
}

/// Square matrix product with the textbook i-j-k loop order.
fn naive_matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..n {
                acc += a[i * n + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// One row at a time: subtract the row maximum, exponentiate, normalize.
fn naive_softmax_rows(data: &[f32], cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for row in data.chunks_exact(cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|e| e / sum));
    }
    out
}

fn naive_silu(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

fn naive_broadcast_add(matrix: &[f32], row: &[f32]) -> Vec<f32> {
    matrix.chunks_exact(row.len()).flat_map(|m| m.iter().zip(row).map(|(a, b)| a + b)).collect()
}

fn naive_mean_rows(data: &[f32], cols: usize) -> Vec<f32> {
    data.chunks_exact(cols).map(|row| row.iter().sum::<f32>() / cols as f32).collect()
}

fn naive_scale_rows(data: &[f32], per_row: &[f32]) -> Vec<f32> {
    let cols = data.len() / per_row.len();
    data.chunks_exact(cols)
        .zip(per_row)
        .flat_map(|(row, &s)| row.iter().map(move |v| v * s))
        .collect()
}

/// The elements throughput of a matrix product is its floating-point operation count.
fn matmul_square(c: &mut Criterion) {
    for n in [32usize, 96, 192] {
        let (a, b) = (uniform(&[n, n], 1), uniform(&[n, n], 2));
        let (a_data, b_data) = (a.to_vec(), b.to_vec());
        let reference = naive_matmul(&a_data, &b_data, n);
        assert_close(&reference, a.matmul(&b).unwrap().data(), 1e-4);
        assert_close(&reference, &midman_bench::matmul_ijk(&a_data, &b_data, n), 1e-4);
        assert_close(&reference, &midman_bench::matmul_ikj(&a_data, &b_data, n), 1e-4);
        assert_backends_agree(|| a.matmul(&b).unwrap());

        let mut group = c.benchmark_group(format!("matmul_square_{n}"));
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        group.bench_function("baseline-naive-ijk", |bench| {
            bench.iter(|| naive_matmul(black_box(&a_data), black_box(&b_data), n))
        });
        group.bench_function("baseline-c-ijk", |bench| {
            bench.iter(|| midman_bench::matmul_ijk(black_box(&a_data), black_box(&b_data), n))
        });
        group.bench_function("baseline-c-ikj", |bench| {
            bench.iter(|| midman_bench::matmul_ikj(black_box(&a_data), black_box(&b_data), n))
        });
        group.bench_function("midman-tensor", |bench| {
            bench.iter(|| a.matmul(black_box(&b)).unwrap())
        });
        scalar_twin(&mut group, "midman-tensor-scalar", || a.matmul(black_box(&b)).unwrap());
        group.finish();
    }
}

/// The shapes a Smoke-preset block multiplies by a shared weight matrix.
fn matmul_linear(c: &mut Criterion) {
    let cases = [("smoke_gate", 64usize, 176usize), ("smoke_head", 64, 256)];
    for (name, d_in, d_out) in cases {
        let x = uniform(&[4, 32, d_in], 3);
        let w = uniform(&[d_in, d_out], 4);
        assert_backends_agree(|| x.matmul(&w).unwrap());
        let mut group = c.benchmark_group(format!("matmul_linear_{name}"));
        group.throughput(Throughput::Elements(2 * (4 * 32 * d_in * d_out) as u64));
        group.bench_function("midman-tensor", |bench| {
            bench.iter(|| x.matmul(black_box(&w)).unwrap())
        });
        scalar_twin(&mut group, "midman-tensor-scalar", || x.matmul(black_box(&w)).unwrap());
        group.finish();
    }
}

/// Attention-score shaped batched products: `[2, 4, T, 16] x [2, 4, 16, T]`.
fn matmul_batched_scores(c: &mut Criterion) {
    for t in [32usize, 128] {
        let q = uniform(&[2, 4, t, 16], 5);
        let kt = uniform(&[2, 4, 16, t], 6);
        assert_backends_agree(|| q.matmul(&kt).unwrap());
        let mut group = c.benchmark_group(format!("matmul_batched_scores_t{t}"));
        group.throughput(Throughput::Elements(2 * (2 * 4 * t * t * 16) as u64));
        group.bench_function("midman-tensor", |bench| {
            bench.iter(|| q.matmul(black_box(&kt)).unwrap())
        });
        scalar_twin(&mut group, "midman-tensor-scalar", || q.matmul(black_box(&kt)).unwrap());
        group.finish();
    }
}

fn softmax_rows(c: &mut Criterion) {
    let (rows, cols) = (256usize, 128usize);
    let x = uniform(&[rows, cols], 7);
    let x_data = x.to_vec();
    let reference = naive_softmax_rows(&x_data, cols);
    assert_close(&reference, x.softmax(-1).unwrap().data(), 1e-5);
    assert_close(&reference, &midman_bench::softmax_rows(&x_data, cols), 1e-5);

    let mut group = c.benchmark_group("softmax_256x128");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("baseline-naive-rows", |bench| {
        bench.iter(|| naive_softmax_rows(black_box(&x_data), cols))
    });
    group.bench_function("baseline-c-rows", |bench| {
        bench.iter(|| midman_bench::softmax_rows(black_box(&x_data), cols))
    });
    group.bench_function("midman-tensor", |bench| bench.iter(|| x.softmax(-1).unwrap()));
    group.finish();
}

fn silu_elementwise(c: &mut Criterion) {
    let x = uniform(&[256, 256], 8);
    let x_data = x.to_vec();
    let reference = naive_silu(&x_data);
    assert_close(&reference, x.silu().data(), 1e-5);
    assert_close(&reference, &midman_bench::silu(&x_data), 1e-5);

    let mut group = c.benchmark_group("silu_64k");
    group.throughput(Throughput::Elements(x_data.len() as u64));
    group.bench_function("baseline-naive-map", |bench| {
        bench.iter(|| naive_silu(black_box(&x_data)))
    });
    group.bench_function("baseline-c-loop", |bench| {
        bench.iter(|| midman_bench::silu(black_box(&x_data)))
    });
    group.bench_function("midman-tensor", |bench| bench.iter(|| black_box(&x).silu()));
    group.finish();
}

fn broadcast_add(c: &mut Criterion) {
    let (rows, cols) = (256usize, 512usize);
    let matrix = uniform(&[rows, cols], 9);
    let row = uniform(&[cols], 10);
    let (matrix_data, row_data) = (matrix.to_vec(), row.to_vec());
    let reference = naive_broadcast_add(&matrix_data, &row_data);
    assert_close(&reference, matrix.add(&row).unwrap().data(), 1e-6);
    assert_close(&reference, &midman_bench::broadcast_add(&matrix_data, &row_data), 1e-6);

    let mut group = c.benchmark_group("broadcast_add_256x512");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("baseline-naive-zip", |bench| {
        bench.iter(|| naive_broadcast_add(black_box(&matrix_data), black_box(&row_data)))
    });
    group.bench_function("baseline-c-loop", |bench| {
        bench.iter(|| midman_bench::broadcast_add(black_box(&matrix_data), black_box(&row_data)))
    });
    group.bench_function("midman-tensor", |bench| {
        bench.iter(|| matrix.add(black_box(&row)).unwrap())
    });
    group.finish();
}

/// The backward pass of a linear layer is two more matrix products, one of them against
/// the transposed weights. `unit-forward` is the product alone; the ratio of
/// `forward-backward` to it is what the backward pass costs in units of the forward pass.
fn matmul_backward(c: &mut Criterion) {
    let cases = [("smoke_gate", 64usize, 176usize), ("smoke_head", 64, 256)];
    for (name, d_in, d_out) in cases {
        let x = Parameter::new("x", uniform(&[4, 32, d_in], 3));
        let w = Parameter::new("w", uniform(&[d_in, d_out], 4));
        let mut group = c.benchmark_group(format!("matmul_backward_{name}"));
        group.bench_function("unit-forward", |bench| {
            bench.iter(|| x.tensor().matmul(black_box(&w.tensor())).unwrap())
        });
        group.bench_function("forward-backward", |bench| {
            bench.iter(|| {
                x.tensor().matmul(black_box(&w.tensor())).unwrap().sum_all().backward().unwrap()
            })
        });
        group.finish();
    }
}

fn mean_last_axis(c: &mut Criterion) {
    let (rows, cols) = (512usize, 64usize);
    let x = uniform(&[rows, cols], 14);
    let x_data = x.to_vec();
    assert_close(&naive_mean_rows(&x_data, cols), x.mean_axis(-1, true).unwrap().data(), 1e-6);
    let mut group = c.benchmark_group("mean_last_axis_512x64");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("baseline-naive-rows", |bench| {
        bench.iter(|| naive_mean_rows(black_box(&x_data), cols))
    });
    group.bench_function("midman-tensor", |bench| {
        bench.iter(|| black_box(&x).mean_axis(-1, true).unwrap())
    });
    group.finish();
}

fn broadcast_mul_column(c: &mut Criterion) {
    let (rows, cols) = (512usize, 64usize);
    let x = uniform(&[rows, cols], 15);
    let column = uniform(&[rows, 1], 16);
    let (x_data, column_data) = (x.to_vec(), column.to_vec());
    assert_close(&naive_scale_rows(&x_data, &column_data), x.mul(&column).unwrap().data(), 1e-6);
    let mut group = c.benchmark_group("broadcast_mul_column_512x64");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("baseline-naive-rows", |bench| {
        bench.iter(|| naive_scale_rows(black_box(&x_data), black_box(&column_data)))
    });
    group.bench_function("midman-tensor", |bench| {
        bench.iter(|| black_box(&x).mul(black_box(&column)).unwrap())
    });
    group.finish();
}

/// Prints the C compiler once, so the job summary can say what built the C baselines.
fn announce_c_compiler(_: &mut Criterion) {
    eprintln!("C baselines compiled with: {}", midman_bench::compiler());
}

fn mlp_loss(x: &Tensor, w1: &Tensor, w2: &Tensor) -> Tensor {
    x.matmul(w1).unwrap().silu().matmul(w2).unwrap().sum_all()
}

/// What autograd costs on a two-layer MLP: the same forward pass without a
/// graph, with a graph that is never used, and with `backward`.
fn mlp_backward(c: &mut Criterion) {
    let (rows, d) = (128usize, 64usize);
    let x = uniform(&[rows, d], 11);
    let (w1, w2) = (uniform(&[d, 4 * d], 12), uniform(&[4 * d, d], 13));
    let (p1, p2) = (Parameter::new("w1", w1.clone()), Parameter::new("w2", w2.clone()));

    let mut group = c.benchmark_group("mlp_128x64x256");
    group.bench_function("unit-forward-untracked", |bench| bench.iter(|| mlp_loss(&x, &w1, &w2)));
    group.bench_function("forward-tracked", |bench| {
        bench.iter(|| mlp_loss(&x, &p1.tensor(), &p2.tensor()))
    });
    group.bench_function("forward-backward", |bench| {
        bench.iter(|| mlp_loss(&x, &p1.tensor(), &p2.tensor()).backward().unwrap())
    });
    group.finish();
}

/// Cost of filling a weight matrix, which dominates building a model.
fn randn_fill(c: &mut Criterion) {
    let mut rng = Rng::seed_from_u64(14);
    let mut group = c.benchmark_group("randn_1m");
    group.throughput(Throughput::Elements(1024 * 1024));
    group.bench_function("midman-tensor", |bench| {
        bench.iter(|| Tensor::randn(&[1024, 1024], 0.02, &mut rng).unwrap())
    });
    group.finish();
}

criterion_group!(
    benches,
    announce_c_compiler,
    matmul_square,
    matmul_linear,
    matmul_batched_scores,
    softmax_rows,
    silu_elementwise,
    broadcast_add,
    broadcast_mul_column,
    mean_last_axis,
    matmul_backward,
    mlp_backward,
    randn_fill
);
criterion_main!(benches);
