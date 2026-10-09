// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "optim_step.rs"
// ============================================================================
//! Benchmarks for `midman-optim`: an AdamW step, gradient clipping, and the
//! global gradient norm, over one matrix parameter of about 65 thousand and
//! about one million values.
//!
//! The AdamW baseline is a plain loop over the same update rule, checked
//! against the real optimizer before anything is timed. Every benchmark id is
//! `group/variant`; a variant named `baseline-*` is a reference implementation
//! (badged by `scripts/bench_summary.py`).

// `criterion_group!` expands to an undocumented `pub fn`, which `missing_docs` flags.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use midman_foundation::Rng;
use midman_optim::{clip_global_norm, global_norm, AdamW};
use midman_tensor::{Parameter, Tensor};
use std::hint::black_box;

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const WEIGHT_DECAY: f32 = 0.1;
const LR: f32 = 1e-3;

/// One AdamW step as a plain loop. `step` counts from 1.
struct PlainAdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: i32,
}

impl PlainAdamW {
    fn new(len: usize) -> PlainAdamW {
        PlainAdamW { m: vec![0.0; len], v: vec![0.0; len], step: 0 }
    }

    fn update(&mut self, weights: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let bias1 = 1.0 - BETA1.powi(self.step);
        let bias2_sqrt = (1.0 - BETA2.powi(self.step)).sqrt();
        let decay = 1.0 - LR * WEIGHT_DECAY;
        for i in 0..weights.len() {
            weights[i] *= decay;
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grads[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grads[i] * grads[i];
            weights[i] -= LR / bias1 * self.m[i] / (self.v[i].sqrt() / bias2_sqrt + EPS);
        }
    }
}

fn matrix(len: usize, seed: u64) -> Tensor {
    Tensor::uniform(&[len / 256, 256], -1.0, 1.0, &mut Rng::seed_from_u64(seed)).unwrap()
}

fn gradient(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::seed_from_u64(seed);
    (0..len).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

fn adamw_step(c: &mut Criterion) {
    for len in [65_536usize, 1_048_576] {
        let weights = matrix(len, 1);
        let grads = vec![gradient(len, 2)];

        // The baseline must compute the same update as the optimizer.
        let mut check_param = Parameter::new("w", weights.clone());
        AdamW::new(BETA1, BETA2, EPS, WEIGHT_DECAY)
            .unwrap()
            .step(&mut [&mut check_param], &grads, LR)
            .unwrap();
        let mut check_plain = weights.to_vec();
        PlainAdamW::new(len).update(&mut check_plain, &grads[0]);
        for (a, b) in check_plain.iter().zip(check_param.data()) {
            assert!((a - b).abs() < 1e-5, "the baseline and AdamW disagree: {a} vs {b}");
        }

        let mut param = Parameter::new("w", weights.clone());
        let mut optimizer = AdamW::new(BETA1, BETA2, EPS, WEIGHT_DECAY).unwrap();
        let mut plain_weights = weights.to_vec();
        let mut plain = PlainAdamW::new(len);

        let mut group = c.benchmark_group(format!("adamw_step_{len}"));
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function("baseline-plain-loop", |bench| {
            bench.iter(|| plain.update(black_box(&mut plain_weights), black_box(&grads[0])))
        });
        group.bench_function("midman-optim", |bench| {
            bench.iter(|| optimizer.step(&mut [&mut param], black_box(&grads), LR).unwrap())
        });
        group.finish();
    }
}

fn clipping(c: &mut Criterion) {
    let len = 1_048_576usize;
    let grads = vec![gradient(len, 3)];
    let mut group = c.benchmark_group("clip_1m");
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("global-norm", |bench| bench.iter(|| global_norm(black_box(&grads))));
    // A tiny limit makes every call rescale, the slower path.
    group.bench_function("clip-and-rescale", |bench| {
        bench.iter_batched(
            || grads.clone(),
            |mut g| clip_global_norm(&mut g, 1e-3).unwrap(),
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, adamw_step, clipping);
criterion_main!(benches);
