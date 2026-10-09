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

// `criterion_group!` expands to an undocumented `pub fn`, which `missing_docs` flags.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use midman_foundation::Rng;
use midman_tensor::{Parameter, Tensor};
use std::hint::black_box;

fn uniform(shape: &[usize], seed: u64) -> Tensor {
    Tensor::uniform(shape, -1.0, 1.0, &mut Rng::seed_from_u64(seed)).unwrap()
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

/// The elements throughput of a matrix product is its floating-point operation count.
fn matmul_square(c: &mut Criterion) {
    for n in [32usize, 96, 192] {
        let (a, b) = (uniform(&[n, n], 1), uniform(&[n, n], 2));
        let (a_data, b_data) = (a.to_vec(), b.to_vec());
        assert_close(&naive_matmul(&a_data, &b_data, n), a.matmul(&b).unwrap().data(), 1e-4);

        let mut group = c.benchmark_group(format!("matmul_square_{n}"));
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        group.bench_function("baseline-naive-ijk", |bench| {
            bench.iter(|| naive_matmul(black_box(&a_data), black_box(&b_data), n))
        });
        group.bench_function("midman-tensor", |bench| {
            bench.iter(|| a.matmul(black_box(&b)).unwrap())
        });
        group.finish();
    }
}

/// The shapes a Smoke-preset block multiplies by a shared weight matrix.
fn matmul_linear(c: &mut Criterion) {
    let cases = [("smoke_gate", 64usize, 176usize), ("smoke_head", 64, 256)];
    for (name, d_in, d_out) in cases {
        let x = uniform(&[4, 32, d_in], 3);
        let w = uniform(&[d_in, d_out], 4);
        let mut group = c.benchmark_group(format!("matmul_linear_{name}"));
        group.throughput(Throughput::Elements(2 * (4 * 32 * d_in * d_out) as u64));
        group.bench_function("midman-tensor", |bench| {
            bench.iter(|| x.matmul(black_box(&w)).unwrap())
        });
        group.finish();
    }
}

/// Attention-score shaped batched products: `[2, 4, T, 16] x [2, 4, 16, T]`.
fn matmul_batched_scores(c: &mut Criterion) {
    for t in [32usize, 128] {
        let q = uniform(&[2, 4, t, 16], 5);
        let kt = uniform(&[2, 4, 16, t], 6);
        let mut group = c.benchmark_group(format!("matmul_batched_scores_t{t}"));
        group.throughput(Throughput::Elements(2 * (2 * 4 * t * t * 16) as u64));
        group.bench_function("midman-tensor", |bench| {
            bench.iter(|| q.matmul(black_box(&kt)).unwrap())
        });
        group.finish();
    }
}

fn softmax_rows(c: &mut Criterion) {
    let (rows, cols) = (256usize, 128usize);
    let x = uniform(&[rows, cols], 7);
    let x_data = x.to_vec();
    assert_close(&naive_softmax_rows(&x_data, cols), x.softmax(-1).unwrap().data(), 1e-5);

    let mut group = c.benchmark_group("softmax_256x128");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("baseline-naive-rows", |bench| {
        bench.iter(|| naive_softmax_rows(black_box(&x_data), cols))
    });
    group.bench_function("midman-tensor", |bench| bench.iter(|| x.softmax(-1).unwrap()));
    group.finish();
}

fn silu_elementwise(c: &mut Criterion) {
    let x = uniform(&[256, 256], 8);
    let x_data = x.to_vec();
    assert_close(&naive_silu(&x_data), x.silu().data(), 1e-5);

    let mut group = c.benchmark_group("silu_64k");
    group.throughput(Throughput::Elements(x_data.len() as u64));
    group.bench_function("baseline-naive-map", |bench| {
        bench.iter(|| naive_silu(black_box(&x_data)))
    });
    group.bench_function("midman-tensor", |bench| bench.iter(|| black_box(&x).silu()));
    group.finish();
}

fn broadcast_add(c: &mut Criterion) {
    let (rows, cols) = (256usize, 512usize);
    let matrix = uniform(&[rows, cols], 9);
    let row = uniform(&[cols], 10);
    let (matrix_data, row_data) = (matrix.to_vec(), row.to_vec());
    assert_close(
        &naive_broadcast_add(&matrix_data, &row_data),
        matrix.add(&row).unwrap().data(),
        1e-6,
    );

    let mut group = c.benchmark_group("broadcast_add_256x512");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("baseline-naive-zip", |bench| {
        bench.iter(|| naive_broadcast_add(black_box(&matrix_data), black_box(&row_data)))
    });
    group.bench_function("midman-tensor", |bench| {
        bench.iter(|| matrix.add(black_box(&row)).unwrap())
    });
    group.finish();
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
    matmul_square,
    matmul_linear,
    matmul_batched_scores,
    softmax_rows,
    silu_elementwise,
    broadcast_add,
    mlp_backward,
    randn_fill
);
criterion_main!(benches);
