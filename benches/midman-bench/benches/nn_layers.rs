// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "nn_layers.rs"
// ============================================================================
//! Benchmarks for the `midman-nn` layers, at the width and head count of the
//! Smoke preset (`d_model` 64, 4 heads of size 16, `d_ff` 176).
//!
//! Every benchmark id is `group/variant`; a variant named `baseline-*` is a
//! reference implementation (badged by `scripts/bench_summary.py`) and one named
//! `unit-*` is a cost denominator (a plain multiple).

// `criterion_group!` expands to an undocumented `pub fn`, which `missing_docs` flags.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};
use midman_foundation::Rng;
use midman_nn::{Attention, BlockSpec, DecoderBlock, Init, RmsNorm, RotaryTables, SwiGluMlp};
use midman_tensor::Tensor;
use std::hint::black_box;

const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / N_HEADS;
const D_FF: usize = 176;
const MAX_T: usize = 128;

fn uniform(shape: &[usize], seed: u64) -> Tensor {
    Tensor::uniform(shape, -1.0, 1.0, &mut Rng::seed_from_u64(seed)).unwrap()
}

fn rope() -> RotaryTables {
    RotaryTables::new(HEAD_DIM, MAX_T, 10_000.0).unwrap()
}

fn attention(n_kv_heads: usize) -> Attention {
    Attention::new(
        "attn",
        D_MODEL,
        N_HEADS,
        n_kv_heads,
        Init::uniform(0.5),
        &mut Rng::seed_from_u64(1),
    )
    .unwrap()
}

/// RMSNorm over the last axis, written out with plain loops.
fn naive_rms_norm(data: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    let mut out = Vec::with_capacity(data.len());
    for row in data.chunks_exact(dim) {
        let mean_square = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean_square + eps).sqrt();
        out.extend(row.iter().zip(weight).map(|(x, w)| x * inv * w));
    }
    out
}

/// Attention forward pass with the key/value head count varied: full
/// multi-head (the baseline), grouped-query, and multi-query.
fn attention_forward(c: &mut Criterion) {
    let rope = rope();
    for t in [16usize, 64, 128] {
        let x = uniform(&[2, t, D_MODEL], 2);
        let mut group = c.benchmark_group(format!("attention_forward_t{t}"));
        for (name, n_kv) in [("baseline-mha", 4usize), ("gqa-2", 2), ("mqa", 1)] {
            let layer = attention(n_kv);
            group.bench_function(name, |bench| {
                bench.iter(|| layer.forward(black_box(&x), &rope).unwrap())
            });
        }
        group.finish();
    }
}

/// The same attention layer with and without `backward`, grouped-query with 2 kv heads.
fn attention_backward(c: &mut Criterion) {
    let rope = rope();
    let layer = attention(2);
    let x = uniform(&[2, 64, D_MODEL], 3);
    let mut group = c.benchmark_group("attention_backward_t64");
    group.bench_function("unit-forward", |bench| {
        bench.iter(|| layer.forward(black_box(&x), &rope).unwrap())
    });
    group.bench_function("forward-backward", |bench| {
        bench.iter(|| layer.forward(black_box(&x), &rope).unwrap().sum_all().backward().unwrap())
    });
    group.finish();
}

fn rms_norm(c: &mut Criterion) {
    let layer = RmsNorm::new("norm", D_MODEL, 1e-5).unwrap();
    let x = uniform(&[512, D_MODEL], 4);
    let x_data = x.to_vec();
    let weight = vec![1.0f32; D_MODEL];
    let want = naive_rms_norm(&x_data, &weight, 1e-5);
    for (a, b) in want.iter().zip(layer.forward(&x).unwrap().data()) {
        assert!((a - b).abs() < 1e-4, "the baseline and the layer disagree: {a} vs {b}");
    }

    let mut group = c.benchmark_group("rms_norm_512x64");
    group.bench_function("baseline-naive-rows", |bench| {
        bench.iter(|| naive_rms_norm(black_box(&x_data), &weight, 1e-5))
    });
    group.bench_function("midman-nn", |bench| bench.iter(|| layer.forward(black_box(&x)).unwrap()));
    group.finish();
}

fn rope_apply(c: &mut Criterion) {
    let rope = rope();
    let q = uniform(&[2, N_HEADS, MAX_T, HEAD_DIM], 5);
    let mut group = c.benchmark_group("rope_t128");
    group.bench_function("midman-nn", |bench| bench.iter(|| rope.apply(black_box(&q)).unwrap()));
    group.finish();
}

fn swiglu_forward(c: &mut Criterion) {
    let mlp = SwiGluMlp::new("mlp", D_MODEL, D_FF, Init::uniform(0.3), &mut Rng::seed_from_u64(6))
        .unwrap();
    let x = uniform(&[2, 64, D_MODEL], 7);
    let mut group = c.benchmark_group("swiglu_t64");
    group.bench_function("midman-nn", |bench| bench.iter(|| mlp.forward(black_box(&x)).unwrap()));
    group.finish();
}

fn decoder_block(c: &mut Criterion) {
    let spec = BlockSpec {
        d_model: D_MODEL,
        n_heads: N_HEADS,
        n_kv_heads: N_HEADS,
        d_ff: D_FF,
        norm_eps: 1e-5,
        init: Init::uniform(0.3),
    };
    let block = DecoderBlock::new("block", &spec, &mut Rng::seed_from_u64(8)).unwrap();
    let rope = rope();
    let x = uniform(&[2, 64, D_MODEL], 9);
    let mut group = c.benchmark_group("decoder_block_t64");
    group.bench_function("unit-forward", |bench| {
        bench.iter(|| block.forward(black_box(&x), &rope).unwrap())
    });
    group.bench_function("forward-backward", |bench| {
        bench.iter(|| block.forward(black_box(&x), &rope).unwrap().sum_all().backward().unwrap())
    });
    group.finish();
}

criterion_group!(
    benches,
    attention_forward,
    attention_backward,
    rms_norm,
    rope_apply,
    swiglu_forward,
    decoder_block
);
criterion_main!(benches);
