// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "tests/grad_layers.rs"
// ============================================================================
//! Finite-difference gradient checks for the layer functions in `midman-nn`,
//! with respect to their inputs and their weights.

use midman_foundation::Rng;
use midman_nn::{
    apply_rope, attention_forward, causal_attention, rms_norm, swiglu, AttentionWeights,
    RotaryTables,
};
use midman_tensor::check::{grad_check, DEFAULT_EPS};
use midman_tensor::Tensor;

const TOL: f32 = 2e-2;

fn rand_t(shape: &[usize], seed: u64, lo: f32, hi: f32) -> Tensor {
    Tensor::uniform(shape, lo, hi, &mut Rng::seed_from_u64(seed)).unwrap()
}

fn check<F>(f: F, inputs: &[Tensor])
where
    F: Fn(&[Tensor]) -> midman_foundation::Result<Tensor>,
{
    grad_check(f, inputs, DEFAULT_EPS).unwrap().assert_within(TOL);
}

#[test]
fn rms_norm_with_respect_to_input_and_weight() {
    check(
        |v| rms_norm(&v[0], &v[1], 1e-5),
        &[rand_t(&[2, 3, 6], 1, -1.0, 1.0), rand_t(&[6], 2, 0.5, 1.5)],
    );
}

#[test]
fn rotary_embedding_with_respect_to_the_input() {
    let rope = RotaryTables::new(4, 8, 10_000.0).unwrap();
    check(move |v| rope.apply(&v[0]), &[rand_t(&[2, 2, 5, 4], 3, -1.0, 1.0)]);
}

#[test]
fn rotary_functional_form_with_respect_to_input_and_tables() {
    // The tables are constants in a model, but the function must be correct for
    // any table values, so check it with respect to them as well.
    check(
        |v| apply_rope(&v[0], &v[1], &v[2]),
        &[
            rand_t(&[1, 2, 3, 4], 4, -1.0, 1.0),
            rand_t(&[3, 2], 5, -1.0, 1.0),
            rand_t(&[3, 2], 6, -1.0, 1.0),
        ],
    );
}

#[test]
fn causal_attention_with_equal_head_counts() {
    check(
        |v| causal_attention(&v[0], &v[1], &v[2]),
        &[
            rand_t(&[2, 2, 4, 3], 7, -1.0, 1.0),
            rand_t(&[2, 2, 4, 3], 8, -1.0, 1.0),
            rand_t(&[2, 2, 4, 3], 9, -1.0, 1.0),
        ],
    );
}

#[test]
fn causal_attention_with_grouped_key_value_heads() {
    // 4 query heads over 2 key/value heads, and 3 over 1.
    check(
        |v| causal_attention(&v[0], &v[1], &v[2]),
        &[
            rand_t(&[1, 4, 3, 2], 10, -1.0, 1.0),
            rand_t(&[1, 2, 3, 2], 11, -1.0, 1.0),
            rand_t(&[1, 2, 3, 2], 12, -1.0, 1.0),
        ],
    );
    check(
        |v| causal_attention(&v[0], &v[1], &v[2]),
        &[
            rand_t(&[2, 3, 3, 2], 13, -1.0, 1.0),
            rand_t(&[2, 1, 3, 2], 14, -1.0, 1.0),
            rand_t(&[2, 1, 3, 2], 15, -1.0, 1.0),
        ],
    );
}

#[test]
fn swiglu_with_respect_to_input_and_all_three_weights() {
    check(
        |v| swiglu(&v[0], &v[1], &v[2], &v[3]),
        &[
            rand_t(&[2, 3, 4], 16, -1.0, 1.0),
            rand_t(&[4, 6], 17, -0.8, 0.8),
            rand_t(&[4, 6], 18, -0.8, 0.8),
            rand_t(&[6, 4], 19, -0.8, 0.8),
        ],
    );
}

#[test]
fn a_whole_attention_layer_with_respect_to_input_and_all_four_weights() {
    // d_model 8, 4 query heads over 2 key/value heads, head size 2.
    let rope = RotaryTables::new(2, 8, 10_000.0).unwrap();
    check(
        move |v| {
            let weights = AttentionWeights {
                q: v[1].clone(),
                k: v[2].clone(),
                v: v[3].clone(),
                o: v[4].clone(),
            };
            attention_forward(&v[0], &weights, 4, 2, &rope)
        },
        &[
            rand_t(&[2, 3, 8], 20, -1.0, 1.0),
            rand_t(&[8, 8], 21, -0.6, 0.6),
            rand_t(&[8, 4], 22, -0.6, 0.6),
            rand_t(&[8, 4], 23, -0.6, 0.6),
            rand_t(&[8, 8], 24, -0.6, 0.6),
        ],
    );
}

#[test]
fn the_attention_function_agrees_with_the_pieces_it_is_built_from() {
    // One head, so no splitting: project, rotate, attend, project.
    let rope = RotaryTables::new(4, 8, 10_000.0).unwrap();
    let x = rand_t(&[1, 5, 4], 25, -1.0, 1.0);
    let (wq, wk, wv, wo) = (
        rand_t(&[4, 4], 26, -0.7, 0.7),
        rand_t(&[4, 4], 27, -0.7, 0.7),
        rand_t(&[4, 4], 28, -0.7, 0.7),
        rand_t(&[4, 4], 29, -0.7, 0.7),
    );
    let weights = AttentionWeights { q: wq.clone(), k: wk.clone(), v: wv.clone(), o: wo.clone() };
    let whole = attention_forward(&x, &weights, 1, 1, &rope).unwrap();

    let head = |w: &Tensor| x.matmul(w).unwrap().reshape(&[1, 1, 5, 4]).unwrap();
    let q = rope.apply(&head(&wq)).unwrap();
    let k = rope.apply(&head(&wk)).unwrap();
    let context = causal_attention(&q, &k, &head(&wv)).unwrap();
    let by_hand = context.reshape(&[1, 5, 4]).unwrap().matmul(&wo).unwrap();
    assert_eq!(whole.data(), by_hand.data());
}
