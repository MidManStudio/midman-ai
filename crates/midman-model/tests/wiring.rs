// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model.md, section "tests/wiring.rs"
// ============================================================================
//! Whole-model checks: parameter accounting against the config, causality,
//! determinism, and a finite-difference gradient check of every parameter.

use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::{ModelConfig, Preset};
use midman_nn::Module;
use std::collections::BTreeSet;

/// A small model that exercises everything: grouped-query attention (4 query
/// heads over 2 key/value heads), several layers, and a configurable head.
fn small(tie: bool) -> ModelConfig {
    ModelConfig {
        vocab_size: 13,
        d_model: 16,
        n_layers: 3,
        n_heads: 4,
        n_kv_heads: 2,
        d_ff: 24,
        max_seq_len: 8,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        tie_embeddings: tie,
    }
}

fn names(model: &MidManModel) -> Vec<String> {
    model.parameters().iter().map(|p| p.name().to_string()).collect()
}

/// Total elements of the parameters whose name satisfies `pick`.
fn numel_where(model: &MidManModel, pick: impl Fn(&str) -> bool) -> u64 {
    model.parameters().iter().filter(|p| pick(p.name())).map(|p| p.numel() as u64).sum()
}

fn check_accounting(config: &ModelConfig) {
    let model = MidManModel::new(config, &mut Rng::seed_from_u64(0)).unwrap();
    let want = config.param_breakdown().unwrap();
    assert_eq!(model.num_parameters() as u64, want.total());
    assert_eq!(numel_where(&model, |n| n.starts_with("embed_tokens")), want.embedding);
    assert_eq!(numel_where(&model, |n| n.contains(".attn.")), want.attention);
    assert_eq!(numel_where(&model, |n| n.contains(".mlp.")), want.mlp);
    assert_eq!(numel_where(&model, |n| n.ends_with("_norm.weight")), want.norms);
    assert_eq!(numel_where(&model, |n| n.starts_with("lm_head")), want.lm_head);
}

#[test]
fn the_smoke_and_tiny_presets_have_exactly_the_counted_parameters() {
    check_accounting(&Preset::Smoke.default_config());
    check_accounting(&Preset::Tiny.default_config());
}

#[test]
fn a_grouped_query_model_matches_in_both_head_modes() {
    check_accounting(&small(true));
    check_accounting(&small(false));
}

#[test]
fn parameter_names_are_unique_and_in_running_order() {
    let model = MidManModel::new(&small(false), &mut Rng::seed_from_u64(0)).unwrap();
    let all = names(&model);
    assert_eq!(all.iter().collect::<BTreeSet<_>>().len(), all.len());
    assert_eq!(all.first().unwrap(), "embed_tokens.weight");
    assert_eq!(all[1], "blocks.0.attn_norm.weight");
    assert_eq!(all[all.len() - 3], "blocks.2.mlp.down_proj.weight");
    assert_eq!(all[all.len() - 2], "final_norm.weight");
    assert_eq!(all.last().unwrap(), "lm_head.weight");
    // 1 embedding + 9 per block * 3 + final norm + head
    assert_eq!(all.len(), 1 + 27 + 1 + 1);
}

#[test]
fn shared_and_mutable_parameter_lists_agree() {
    let mut model = MidManModel::new(&small(true), &mut Rng::seed_from_u64(0)).unwrap();
    let shared = names(&model);
    let mutable: Vec<String> =
        model.parameters_mut().iter().map(|p| p.name().to_string()).collect();
    assert_eq!(shared, mutable);
}

#[test]
fn weights_are_drawn_with_the_documented_standard_deviations() {
    let config = small(false);
    let model = MidManModel::with_init_std(&config, 0.5, &mut Rng::seed_from_u64(3)).unwrap();
    let rms = |p: &&midman_tensor::Parameter| {
        let d = p.data();
        (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt()
    };
    let residual_std = 0.5 / (2.0 * config.n_layers as f32).sqrt();
    let params = model.parameters();
    let mut checked = 0;
    for p in &params {
        let name = p.name();
        let want = if name.ends_with("_norm.weight") {
            1.0
        } else if name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight") {
            residual_std
        } else {
            0.5
        };
        let got = rms(p);
        // Some tensors have only ~100 entries, so allow a few standard errors of slack.
        assert!((got - want).abs() < 0.3 * want, "{name}: spread {got}, expected about {want}");
        checked += 1;
    }
    assert_eq!(checked, params.len());
}

#[test]
fn one_seed_fixes_the_whole_model() {
    let build = |seed| MidManModel::new(&small(true), &mut Rng::seed_from_u64(seed)).unwrap();
    let (a, b, c) = (build(5), build(5), build(6));
    let ids = [1usize, 4, 2, 7, 3, 9];
    assert_eq!(a.forward(&ids, 1, 6).unwrap().data(), b.forward(&ids, 1, 6).unwrap().data());
    assert_ne!(a.forward(&ids, 1, 6).unwrap().data(), c.forward(&ids, 1, 6).unwrap().data());
}

#[test]
fn changing_a_later_token_never_changes_earlier_logits() {
    let model = MidManModel::with_init_std(&small(true), 0.3, &mut Rng::seed_from_u64(1)).unwrap();
    let vocab = 13;
    let base_ids = [3usize, 1, 4, 1, 5, 9];
    let base = model.forward(&base_ids, 1, 6).unwrap();
    for position in 1..6 {
        let mut ids = base_ids;
        ids[position] = (ids[position] + 1) % vocab;
        let changed = model.forward(&ids, 1, 6).unwrap();
        for (a, b) in
            base.data()[..position * vocab].iter().zip(&changed.data()[..position * vocab])
        {
            assert!((a - b).abs() < 1e-5, "position {position}: {a} vs {b}");
        }
        assert_ne!(
            base.data()[position * vocab..(position + 1) * vocab],
            changed.data()[position * vocab..(position + 1) * vocab]
        );
    }
}

#[test]
fn sequences_in_a_batch_do_not_affect_each_other() {
    let model = MidManModel::with_init_std(&small(true), 0.3, &mut Rng::seed_from_u64(2)).unwrap();
    let first = [1usize, 2, 3, 4];
    let alone = model.forward(&first, 1, 4).unwrap();
    let together = model.forward(&[1, 2, 3, 4, 9, 8, 7, 6], 2, 4).unwrap();
    for (a, b) in alone.data().iter().zip(&together.data()[..alone.numel()]) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// Every parameter's analytic gradient against central finite differences.
///
/// The weights are drawn with a large standard deviation so the gradients are
/// well above `f32` noise; with the default 0.02 they would be too small for a
/// finite difference to resolve. A few entries of every parameter are sampled.
fn gradient_agreement(tie: bool) -> (f64, f64, Vec<(String, f64, f64)>) {
    let config = small(tie);
    let mut model = MidManModel::with_init_std(&config, 0.3, &mut Rng::seed_from_u64(11)).unwrap();
    let (batch, seq) = (2, 5);
    let inputs = [1usize, 5, 2, 8, 3, 12, 0, 7, 7, 4];
    let targets = [5usize, 2, 8, 3, 6, 0, 7, 7, 4, 10];

    let mut grads = model.loss(&inputs, &targets, batch, seq).unwrap().backward().unwrap();
    let analytic: Vec<Vec<f32>> = model
        .parameters()
        .iter()
        .map(|p| grads.take(&p.tensor()).unwrap_or_else(|| panic!("no gradient for {}", p.name())))
        .collect();

    let eps = 1e-2f32;
    let mut pick = Rng::seed_from_u64(99);
    let (mut diff_sq, mut ref_sq) = (0.0f64, 0.0f64);
    let mut per_param = Vec::new();
    for (index, gradient) in analytic.iter().enumerate() {
        let name = model.parameters()[index].name().to_string();
        let (mut p_diff, mut p_ref) = (0.0f64, 0.0f64);
        for _ in 0..6 {
            let at = pick.below(gradient.len());
            let mut loss_at = |delta: f32| -> f32 {
                model.parameters_mut()[index].update(|d| d[at] += delta);
                let loss = model.loss(&inputs, &targets, batch, seq).unwrap().item().unwrap();
                model.parameters_mut()[index].update(|d| d[at] -= delta);
                loss
            };
            let numeric = f64::from(loss_at(eps) - loss_at(-eps)) / (2.0 * f64::from(eps));
            let exact = f64::from(gradient[at]);
            p_diff += (numeric - exact) * (numeric - exact);
            p_ref += exact * exact;
        }
        diff_sq += p_diff;
        ref_sq += p_ref;
        per_param.push((name, (p_diff / p_ref.max(1e-12)).sqrt(), p_ref.sqrt()));
    }
    ((diff_sq / ref_sq).sqrt(), ref_sq.sqrt(), per_param)
}

#[test]
fn analytic_gradients_match_finite_differences_for_every_parameter() {
    for tie in [true, false] {
        let (relative_error, gradient_size, per_param) = gradient_agreement(tie);
        assert!(gradient_size > 1e-2, "the sampled gradients are too small to mean anything");
        assert!(relative_error < 5e-3, "tie={tie}: overall relative error {relative_error}");
        // Per parameter, so a bug confined to one small tensor cannot hide in the total.
        for (name, error, size) in &per_param {
            assert!(
                *size > 1e-3,
                "tie={tie}: the sampled gradient of {name} is {size}, too small to check"
            );
            assert!(*error < 3e-2, "tie={tie}: {name} has relative error {error}");
        }
    }
}
