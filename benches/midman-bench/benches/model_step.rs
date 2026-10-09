// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "model_step.rs"
// ============================================================================
//! Benchmarks for whole-model work: building a model, a forward pass, and a full
//! training step (forward, backward, gradient clipping, AdamW).
//!
//! The elements throughput of the step benchmarks is tokens, so the raw log
//! reports tokens per second. A variant named `baseline-*` is a reference
//! implementation (badged by `scripts/bench_summary.py`) and one named `unit-*` is a
//! cost denominator (a plain multiple).

// `criterion_group!` expands to an undocumented `pub fn`, which `missing_docs` flags.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::{ModelConfig, Preset};
use midman_nn::Module;
use midman_optim::{clip_global_norm, gather_gradients, AdamW};
use std::time::Duration;

fn train_step(
    model: &mut MidManModel,
    optimizer: &mut AdamW,
    inputs: &[usize],
    targets: &[usize],
    batch: usize,
    seq: usize,
) {
    let loss = model.loss(inputs, targets, batch, seq).unwrap();
    let mut grads = loss.backward().unwrap();
    let mut params = model.parameters_mut();
    let mut gradients = gather_gradients(&params, &mut grads).unwrap();
    clip_global_norm(&mut gradients, 1.0).unwrap();
    optimizer.step(&mut params, &gradients, 1e-3).unwrap();
}

fn random_tokens(config: &ModelConfig, count: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::seed_from_u64(seed);
    (0..count).map(|_| rng.below(config.vocab_size)).collect()
}

fn step_group(c: &mut Criterion, name: &str, preset: Preset, batch: usize, seq: usize, slow: bool) {
    let config = preset.default_config();
    let mut model = MidManModel::new(&config, &mut Rng::seed_from_u64(1)).unwrap();
    let mut optimizer = AdamW::new(0.9, 0.999, 1e-8, 0.1).unwrap();
    let inputs = random_tokens(&config, batch * seq, 2);
    let targets = random_tokens(&config, batch * seq, 3);

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements((batch * seq) as u64));
    if slow {
        group.sample_size(10).measurement_time(Duration::from_secs(10));
    }
    group.bench_function("unit-forward", |bench| {
        bench.iter(|| model.forward(&inputs, batch, seq).unwrap())
    });
    group.bench_function("forward-backward", |bench| {
        bench.iter(|| model.loss(&inputs, &targets, batch, seq).unwrap().backward().unwrap())
    });
    group.bench_function("full-step", |bench| {
        bench.iter(|| train_step(&mut model, &mut optimizer, &inputs, &targets, batch, seq))
    });
    group.finish();
}

fn train_step_smoke(c: &mut Criterion) {
    step_group(c, "train_step_smoke_b4_t32", Preset::Smoke, 4, 32, false);
}

fn train_step_tiny(c: &mut Criterion) {
    step_group(c, "train_step_tiny_b1_t32", Preset::Tiny, 1, 32, true);
}

/// Construction time, which is dominated by drawing the random weights.
fn model_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_build");
    group.sample_size(10);
    for (name, preset) in [("smoke", Preset::Smoke), ("tiny", Preset::Tiny)] {
        let config = preset.default_config();
        group.throughput(Throughput::Elements(config.param_count().unwrap()));
        group.bench_function(name, |bench| {
            bench.iter(|| MidManModel::new(&config, &mut Rng::seed_from_u64(1)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, train_step_smoke, train_step_tiny, model_build);
criterion_main!(benches);
