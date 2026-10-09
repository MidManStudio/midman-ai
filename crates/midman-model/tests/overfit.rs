// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model.md, section "tests/overfit.rs"
// ============================================================================
//! Training smoke tests: a small model must memorize a few random token
//! sequences. Memorizing needs the embeddings, attention (a repeated token has
//! to be told apart by what came before it), the MLPs, the loss, backward, the
//! gradient plumbing and AdamW to all work together.

use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::{ModelConfig, Preset};
use midman_nn::Module;
use midman_optim::{clip_global_norm, gather_gradients, AdamW, WarmupCosine};

struct Outcome {
    first_loss: f32,
    last_loss: f32,
    accuracy: f32,
}

/// Trains `config` on `batch` random sequences of `seq + 1` tokens (inputs are
/// the first `seq`, targets the last `seq`) for `steps` steps.
fn memorize(config: &ModelConfig, batch: usize, seq: usize, steps: u64, max_lr: f32) -> Outcome {
    let mut rng = Rng::seed_from_u64(2024);
    let mut model = MidManModel::new(config, &mut rng).unwrap();

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..batch {
        let tokens: Vec<usize> = (0..=seq).map(|_| rng.below(config.vocab_size)).collect();
        inputs.extend_from_slice(&tokens[..seq]);
        targets.extend_from_slice(&tokens[1..]);
    }

    let schedule = WarmupCosine::new(max_lr, max_lr / 10.0, steps / 10, steps).unwrap();
    let mut optimizer = AdamW::new(0.9, 0.99, 1e-8, 0.01).unwrap();
    let (mut first_loss, mut last_loss) = (0.0, 0.0);
    for step in 0..steps {
        let loss = model.loss(&inputs, &targets, batch, seq).unwrap();
        last_loss = loss.item().unwrap();
        if step == 0 {
            first_loss = last_loss;
        }
        let mut grads = loss.backward().unwrap();
        let mut params = model.parameters_mut();
        let mut gradients = gather_gradients(&params, &mut grads).unwrap();
        clip_global_norm(&mut gradients, 1.0).unwrap();
        optimizer.step(&mut params, &gradients, schedule.lr(step)).unwrap();
    }

    let logits = model.forward(&inputs, batch, seq).unwrap();
    let vocab = config.vocab_size;
    let correct = logits
        .data()
        .chunks_exact(vocab)
        .zip(&targets)
        .filter(|(row, &want)| {
            let best = row.iter().enumerate().fold(0, |b, (i, &v)| if v > row[b] { i } else { b });
            best == want
        })
        .count();
    Outcome { first_loss, last_loss, accuracy: correct as f32 / targets.len() as f32 }
}

fn tiny_config(tie_embeddings: bool) -> ModelConfig {
    ModelConfig {
        vocab_size: 12,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        d_ff: 32,
        max_seq_len: 16,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        tie_embeddings,
    }
}

#[test]
fn a_tiny_model_memorizes_random_sequences() {
    let out = memorize(&tiny_config(true), 2, 12, 150, 3e-2);
    println!("loss {:.4} -> {:.4}, accuracy {:.3}", out.first_loss, out.last_loss, out.accuracy);
    let uniform = 12f32.ln();
    assert!((out.first_loss - uniform).abs() < 0.3, "it should start near ln(V) = {uniform}");
    assert!(out.last_loss < 0.1 * out.first_loss, "loss only fell to {}", out.last_loss);
    assert!(out.accuracy > 0.95, "accuracy {}", out.accuracy);
}

#[test]
fn a_tiny_model_with_its_own_output_head_memorizes_too() {
    let out = memorize(&tiny_config(false), 2, 12, 150, 3e-2);
    println!(
        "untied: loss {:.4} -> {:.4}, accuracy {:.3}",
        out.first_loss, out.last_loss, out.accuracy
    );
    assert!(out.last_loss < 0.1 * out.first_loss, "loss only fell to {}", out.last_loss);
    assert!(out.accuracy > 0.95, "accuracy {}", out.accuracy);
}

#[test]
fn training_is_reproducible_bit_for_bit() {
    // The same seed must give the same loss at every step, down to the last bit.
    let curve = || -> Vec<u32> {
        let config = tiny_config(true);
        let mut rng = Rng::seed_from_u64(77);
        let mut model = MidManModel::new(&config, &mut rng).unwrap();
        let inputs: Vec<usize> = (0..12).map(|_| rng.below(config.vocab_size)).collect();
        let targets: Vec<usize> = (0..12).map(|_| rng.below(config.vocab_size)).collect();
        let mut optimizer = AdamW::new(0.9, 0.99, 1e-8, 0.01).unwrap();
        let mut bits = Vec::new();
        for _ in 0..25 {
            let loss = model.loss(&inputs, &targets, 1, 12).unwrap();
            bits.push(loss.item().unwrap().to_bits());
            let mut grads = loss.backward().unwrap();
            let mut params = model.parameters_mut();
            let mut gradients = gather_gradients(&params, &mut grads).unwrap();
            clip_global_norm(&mut gradients, 1.0).unwrap();
            optimizer.step(&mut params, &gradients, 1e-2).unwrap();
        }
        bits
    };
    let (first, second) = (curve(), curve());
    assert_eq!(first, second);
    assert_ne!(
        first[0],
        *first.last().unwrap(),
        "the loss never moved, so the comparison proves nothing"
    );
}

/// The 117K-parameter smoke preset on a longer corpus. Slow in a debug build, so
/// it is skipped by default; the `train-smoke` workflow runs it in release with
/// `--ignored`.
#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn the_smoke_preset_memorizes_random_sequences() {
    let config = Preset::Smoke.default_config();
    let out = memorize(&config, 4, 32, 300, 1e-2);
    println!("loss {:.4} -> {:.4}, accuracy {:.3}", out.first_loss, out.last_loss, out.accuracy);
    assert!(out.last_loss < 0.1 * out.first_loss, "loss only fell to {}", out.last_loss);
    assert!(out.accuracy > 0.95, "accuracy {}", out.accuracy);
}
