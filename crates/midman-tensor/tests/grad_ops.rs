// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "tests/grad_ops.rs"
// ============================================================================
//! Finite-difference gradient checks for every differentiable operation, and
//! for the compositions a transformer is built from.

use midman_foundation::Rng;
use midman_tensor::check::{grad_check, DEFAULT_EPS};
use midman_tensor::Tensor;

const TOL: f32 = 2e-2;

/// A reproducible tensor with entries in `[lo, hi)`.
fn rand_t(shape: &[usize], seed: u64, lo: f32, hi: f32) -> Tensor {
    Tensor::uniform(shape, lo, hi, &mut Rng::seed_from_u64(seed)).unwrap()
}

/// Entries with magnitude in `[0.3, 1.3]` and random sign: away from the kinks
/// of piecewise-linear functions.
fn rand_away_from_zero(shape: &[usize], seed: u64) -> Tensor {
    let mut rng = Rng::seed_from_u64(seed);
    let n: usize = shape.iter().product();
    let data = (0..n)
        .map(|_| {
            let magnitude = rng.uniform(0.3, 1.3);
            if rng.next_u64() & 1 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect();
    Tensor::from_vec(data, shape).unwrap()
}

fn check<F>(f: F, inputs: &[Tensor])
where
    F: Fn(&[Tensor]) -> midman_foundation::Result<Tensor>,
{
    grad_check(f, inputs, DEFAULT_EPS).unwrap().assert_within(TOL);
}

// ---------------------------------------------------------------- arithmetic

#[test]
fn add_with_every_broadcast_pattern() {
    check(|v| v[0].add(&v[1]), &[rand_t(&[3, 4], 1, -1.0, 1.0), rand_t(&[3, 4], 2, -1.0, 1.0)]);
    check(|v| v[0].add(&v[1]), &[rand_t(&[3, 4], 3, -1.0, 1.0), rand_t(&[4], 4, -1.0, 1.0)]);
    check(|v| v[0].add(&v[1]), &[rand_t(&[3, 1], 5, -1.0, 1.0), rand_t(&[1, 4], 6, -1.0, 1.0)]);
    check(|v| v[0].add(&v[1]), &[rand_t(&[2, 3], 7, -1.0, 1.0), rand_t(&[], 8, -1.0, 1.0)]);
    check(|v| v[0].add(&v[1]), &[rand_t(&[2, 3, 4], 9, -1.0, 1.0), rand_t(&[3, 1], 10, -1.0, 1.0)]);
}

#[test]
fn sub_mul_div_with_broadcasting() {
    check(|v| v[0].sub(&v[1]), &[rand_t(&[3, 4], 11, -1.0, 1.0), rand_t(&[4], 12, -1.0, 1.0)]);
    check(|v| v[0].sub(&v[1]), &[rand_t(&[3, 1], 13, -1.0, 1.0), rand_t(&[1, 4], 14, -1.0, 1.0)]);
    check(|v| v[0].mul(&v[1]), &[rand_t(&[3, 4], 15, -1.0, 1.0), rand_t(&[4], 16, -1.0, 1.0)]);
    check(
        |v| v[0].mul(&v[1]),
        &[rand_t(&[3, 1], 17, -1.0, 1.0), rand_t(&[2, 1, 4], 18, -1.0, 1.0)],
    );
    check(|v| v[0].div(&v[1]), &[rand_t(&[3, 4], 19, -1.0, 1.0), rand_t(&[4], 20, 0.5, 1.5)]);
    check(|v| v[0].div(&v[1]), &[rand_t(&[2, 1], 21, -1.0, 1.0), rand_t(&[2, 3], 22, 0.5, 1.5)]);
}

#[test]
fn scalar_helpers() {
    check(|v| Ok(v[0].scale(-2.5)), &[rand_t(&[2, 3], 23, -1.0, 1.0)]);
    check(|v| Ok(v[0].add_scalar(0.75)), &[rand_t(&[2, 3], 24, -1.0, 1.0)]);
    check(|v| Ok(v[0].neg()), &[rand_t(&[2, 3], 25, -1.0, 1.0)]);
}

// ---------------------------------------------------------------- activations

#[test]
fn smooth_unary_functions() {
    check(|v| Ok(v[0].exp()), &[rand_t(&[2, 3], 30, -1.0, 1.0)]);
    check(|v| Ok(v[0].ln()), &[rand_t(&[2, 3], 31, 0.5, 2.0)]);
    check(|v| Ok(v[0].sqrt()), &[rand_t(&[2, 3], 32, 0.5, 2.0)]);
    check(|v| Ok(v[0].powf(-0.5)), &[rand_t(&[2, 3], 33, 0.5, 2.0)]);
    check(|v| Ok(v[0].powf(3.0)), &[rand_t(&[2, 3], 34, 0.5, 1.5)]);
    check(|v| Ok(v[0].tanh()), &[rand_t(&[2, 3], 35, -2.0, 2.0)]);
    check(|v| Ok(v[0].sigmoid()), &[rand_t(&[2, 3], 36, -3.0, 3.0)]);
    check(|v| Ok(v[0].silu()), &[rand_t(&[2, 3], 37, -3.0, 3.0)]);
    check(|v| Ok(v[0].gelu()), &[rand_t(&[2, 3], 38, -3.0, 3.0)]);
}

#[test]
fn relu_away_from_its_kink() {
    check(|v| Ok(v[0].relu()), &[rand_away_from_zero(&[3, 4], 39)]);
}

// ---------------------------------------------------------------- matmul

#[test]
fn matmul_in_all_supported_forms() {
    check(
        |v| v[0].matmul(&v[1]),
        &[rand_t(&[3, 4], 40, -1.0, 1.0), rand_t(&[4, 2], 41, -1.0, 1.0)],
    );
    check(
        |v| v[0].matmul(&v[1]),
        &[rand_t(&[2, 3, 4], 42, -1.0, 1.0), rand_t(&[4, 5], 43, -1.0, 1.0)],
    );
    check(
        |v| v[0].matmul(&v[1]),
        &[rand_t(&[2, 3, 4], 44, -1.0, 1.0), rand_t(&[2, 4, 2], 45, -1.0, 1.0)],
    );
    check(
        |v| v[0].matmul(&v[1]),
        &[rand_t(&[2, 2, 3, 4], 46, -1.0, 1.0), rand_t(&[2, 2, 4, 3], 47, -1.0, 1.0)],
    );
    // a one-row and a one-column operand
    check(
        |v| v[0].matmul(&v[1]),
        &[rand_t(&[1, 5], 48, -1.0, 1.0), rand_t(&[5, 1], 49, -1.0, 1.0)],
    );
}

// ---------------------------------------------------------------- shape ops

#[test]
fn reshape_transpose_permute() {
    check(|v| v[0].reshape(&[3, 2]), &[rand_t(&[2, 3], 50, -1.0, 1.0)]);
    check(|v| v[0].transpose(0, 1), &[rand_t(&[2, 3], 51, -1.0, 1.0)]);
    check(|v| v[0].transpose(-1, -2), &[rand_t(&[2, 3, 4], 52, -1.0, 1.0)]);
    check(|v| v[0].permute(&[2, 0, 1]), &[rand_t(&[2, 3, 4], 53, -1.0, 1.0)]);
    check(|v| v[0].permute(&[0, 2, 1, 3]), &[rand_t(&[2, 3, 4, 2], 54, -1.0, 1.0)]);
}

#[test]
fn narrow_concat_repeat() {
    check(|v| v[0].narrow(1, 1, 2), &[rand_t(&[2, 4], 55, -1.0, 1.0)]);
    check(|v| v[0].narrow(0, 1, 2), &[rand_t(&[3, 2, 2], 56, -1.0, 1.0)]);
    check(
        |v| Tensor::concat(&[v[0].clone(), v[1].clone(), v[2].clone()], 1),
        &[
            rand_t(&[2, 1, 3], 57, -1.0, 1.0),
            rand_t(&[2, 2, 3], 58, -1.0, 1.0),
            rand_t(&[2, 3, 3], 59, -1.0, 1.0),
        ],
    );
    check(
        |v| Tensor::concat(&[v[0].clone(), v[1].clone()], -1),
        &[rand_t(&[2, 3], 60, -1.0, 1.0), rand_t(&[2, 2], 61, -1.0, 1.0)],
    );
    check(|v| v[0].repeat_interleave(1, 3), &[rand_t(&[2, 2, 3], 62, -1.0, 1.0)]);
    check(|v| v[0].repeat_interleave(0, 2), &[rand_t(&[2, 3], 63, -1.0, 1.0)]);
}

#[test]
fn embedding_lookup_with_repeated_ids() {
    let ids = [3usize, 0, 3, 1, 3, 2];
    check(move |v| v[0].embedding(&ids, &[2, 3]), &[rand_t(&[4, 5], 64, -1.0, 1.0)]);
}

// ---------------------------------------------------------------- reductions

#[test]
fn sums_and_means() {
    check(|v| Ok(v[0].sum_all()), &[rand_t(&[2, 3], 70, -1.0, 1.0)]);
    check(|v| Ok(v[0].mean_all()), &[rand_t(&[2, 3], 71, -1.0, 1.0)]);
    check(|v| v[0].sum_axis(0, false), &[rand_t(&[3, 4], 72, -1.0, 1.0)]);
    check(|v| v[0].sum_axis(1, true), &[rand_t(&[3, 4], 73, -1.0, 1.0)]);
    check(|v| v[0].sum_axis(-1, false), &[rand_t(&[2, 3, 4], 74, -1.0, 1.0)]);
    check(|v| v[0].sum_axis(1, false), &[rand_t(&[2, 3, 4], 75, -1.0, 1.0)]);
    check(|v| v[0].mean_axis(-1, true), &[rand_t(&[2, 3, 4], 76, -1.0, 1.0)]);
}

#[test]
fn softmax_and_log_softmax_on_several_axes() {
    check(|v| v[0].softmax(-1), &[rand_t(&[3, 5], 80, -2.0, 2.0)]);
    check(|v| v[0].softmax(0), &[rand_t(&[3, 5], 81, -2.0, 2.0)]);
    check(|v| v[0].softmax(1), &[rand_t(&[2, 4, 3], 82, -2.0, 2.0)]);
    check(|v| v[0].log_softmax(-1), &[rand_t(&[3, 5], 83, -2.0, 2.0)]);
    check(|v| v[0].log_softmax(1), &[rand_t(&[2, 4, 3], 84, -2.0, 2.0)]);
}

#[test]
fn cross_entropy_loss() {
    let targets = [2usize, 0, 4, 4];
    check(move |v| v[0].cross_entropy(&targets), &[rand_t(&[4, 5], 85, -2.0, 2.0)]);
}

// ---------------------------------------------------------------- compositions

#[test]
fn two_layer_perceptron_with_bias_and_mse() {
    let target = rand_t(&[4, 2], 90, -1.0, 1.0);
    check(
        move |v| {
            let hidden = v[0].matmul(&v[1])?.add(&v[2])?.gelu();
            let out = hidden.matmul(&v[3])?.add(&v[4])?;
            let diff = out.sub(&target)?;
            Ok(diff.mul(&diff)?.mean_all())
        },
        &[
            rand_t(&[4, 3], 91, -1.0, 1.0),
            rand_t(&[3, 5], 92, -1.0, 1.0),
            rand_t(&[5], 93, -0.5, 0.5),
            rand_t(&[5, 2], 94, -1.0, 1.0),
            rand_t(&[2], 95, -0.5, 0.5),
        ],
    );
}

/// RMSNorm built from primitives: `x * rsqrt(mean(x^2) + eps) * weight`.
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> midman_foundation::Result<Tensor> {
    let mean_square = x.mul(x)?.mean_axis(-1, true)?;
    let inv_rms = mean_square.add_scalar(eps).powf(-0.5);
    x.mul(&inv_rms)?.mul(weight)
}

#[test]
fn rms_norm_from_primitives() {
    check(
        |v| rms_norm(&v[0], &v[1], 1e-5),
        &[rand_t(&[2, 3, 6], 100, -1.0, 1.0), rand_t(&[6], 101, 0.5, 1.5)],
    );
}

#[test]
fn rms_norm_output_has_unit_rms_before_the_weight() {
    let x = rand_t(&[4, 8], 102, -3.0, 3.0);
    let y = rms_norm(&x, &Tensor::ones(&[8]).unwrap(), 1e-6).unwrap();
    for row in y.data().chunks_exact(8) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / 8.0;
        assert!((ms - 1.0).abs() < 1e-3, "mean square {ms}");
    }
}

/// Additive causal mask: 0 on and below the diagonal, a large negative above it.
fn causal_mask(t: usize) -> Tensor {
    let data = (0..t * t).map(|i| if i % t > i / t { -1.0e9 } else { 0.0 }).collect();
    Tensor::from_vec(data, &[t, t]).unwrap()
}

/// `softmax(q k^T / sqrt(d) + mask) v` for `[B, H, T, D]` inputs.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> midman_foundation::Result<Tensor> {
    let d = q.shape()[3] as f32;
    let t = q.shape()[2];
    let scores = q.matmul(&k.transpose(-1, -2)?)?.scale(1.0 / d.sqrt()).add(&causal_mask(t))?;
    scores.softmax(-1)?.matmul(v)
}

#[test]
fn causal_attention() {
    check(
        |x| attention(&x[0], &x[1], &x[2]),
        &[
            rand_t(&[2, 2, 4, 3], 110, -1.0, 1.0),
            rand_t(&[2, 2, 4, 3], 111, -1.0, 1.0),
            rand_t(&[2, 2, 4, 3], 112, -1.0, 1.0),
        ],
    );
}

#[test]
fn causal_attention_ignores_the_future() {
    // Changing a later key/value must not change an earlier output row.
    let q = rand_t(&[1, 1, 4, 3], 113, -1.0, 1.0);
    let k = rand_t(&[1, 1, 4, 3], 114, -1.0, 1.0);
    let v = rand_t(&[1, 1, 4, 3], 115, -1.0, 1.0);
    let base = attention(&q, &k, &v).unwrap();
    let mut k2 = k.to_vec();
    let mut v2 = v.to_vec();
    for i in 9..12 {
        k2[i] += 5.0; // the last position's key
        v2[i] -= 5.0; // and value
    }
    let changed = attention(
        &q,
        &Tensor::from_vec(k2, k.shape()).unwrap(),
        &Tensor::from_vec(v2, v.shape()).unwrap(),
    )
    .unwrap();
    assert_eq!(base.data()[..9], changed.data()[..9], "rows before the change must be identical");
    assert_ne!(base.data()[9..], changed.data()[9..]);
}

#[test]
fn grouped_query_attention_shares_kv_heads() {
    // 4 query heads over 2 kv heads.
    check(
        |x| {
            let k = x[1].repeat_interleave(1, 2)?;
            let v = x[2].repeat_interleave(1, 2)?;
            attention(&x[0], &k, &v)
        },
        &[
            rand_t(&[1, 4, 3, 2], 116, -1.0, 1.0),
            rand_t(&[1, 2, 3, 2], 117, -1.0, 1.0),
            rand_t(&[1, 2, 3, 2], 118, -1.0, 1.0),
        ],
    );
}

/// Rotary embedding on `[B, H, T, D]` using `narrow`, `concat` and broadcasting.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> midman_foundation::Result<Tensor> {
    let half = x.shape()[3] / 2;
    let x1 = x.narrow(-1, 0, half)?;
    let x2 = x.narrow(-1, half, half)?;
    let rotated_first = x1.mul(cos)?.sub(&x2.mul(sin)?)?;
    let rotated_second = x2.mul(cos)?.add(&x1.mul(sin)?)?;
    Tensor::concat(&[rotated_first, rotated_second], -1)
}

fn rope_tables(t: usize, half: usize) -> (Tensor, Tensor) {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for pos in 0..t {
        for i in 0..half {
            let angle = pos as f32 / 10_000f32.powf(i as f32 / half as f32);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (Tensor::from_vec(cos, &[t, half]).unwrap(), Tensor::from_vec(sin, &[t, half]).unwrap())
}

#[test]
fn rotary_embedding() {
    let (cos, sin) = rope_tables(4, 2);
    check(move |v| rope(&v[0], &cos, &sin), &[rand_t(&[2, 2, 4, 4], 120, -1.0, 1.0)]);
}

#[test]
fn rotary_embedding_preserves_vector_length() {
    let (cos, sin) = rope_tables(5, 3);
    let x = rand_t(&[1, 1, 5, 6], 121, -1.0, 1.0);
    let y = rope(&x, &cos, &sin).unwrap();
    for (a, b) in x.data().chunks_exact(6).zip(y.data().chunks_exact(6)) {
        let na: f32 = a.iter().map(|v| v * v).sum();
        let nb: f32 = b.iter().map(|v| v * v).sum();
        assert!((na - nb).abs() < 1e-4, "{na} vs {nb}");
    }
}

#[test]
fn tied_embedding_language_model_head() {
    // The same table feeds the embedding lookup and the output projection, so its
    // gradient is the sum of both uses.
    let ids = [1usize, 3, 0, 2, 2, 1];
    let targets = [3usize, 0, 2, 2, 1, 0];
    check(
        move |v| {
            let h = v[0].embedding(&ids, &[6])?;
            let mixed = h.matmul(&v[1])?.tanh();
            let logits = mixed.matmul(&v[0].transpose(0, 1)?)?;
            logits.cross_entropy(&targets)
        },
        &[rand_t(&[4, 5], 130, -1.0, 1.0), rand_t(&[5, 5], 131, -0.5, 0.5)],
    );
}

#[test]
fn a_small_gated_mlp_block_with_residual() {
    // x + down(silu(gate(x)) * up(x)) after RMSNorm: the shape of a transformer MLP block.
    check(
        |v| {
            let normed = rms_norm(&v[0], &v[1], 1e-5)?;
            let gated = normed.matmul(&v[2])?.silu().mul(&normed.matmul(&v[3])?)?;
            v[0].add(&gated.matmul(&v[4])?)
        },
        &[
            rand_t(&[2, 3, 4], 140, -1.0, 1.0),
            rand_t(&[4], 141, 0.5, 1.5),
            rand_t(&[4, 6], 142, -0.8, 0.8),
            rand_t(&[4, 6], 143, -0.8, 0.8),
            rand_t(&[6, 4], 144, -0.8, 0.8),
        ],
    );
}
