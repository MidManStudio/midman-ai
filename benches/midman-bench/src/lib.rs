// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "lib.rs"
// ============================================================================
//! Safe wrappers around the C baseline kernels in `c/baseline_kernels.c`.
//!
//! The benchmarks call these to time plain C next to the Rust kernels. Every
//! wrapper checks its buffer lengths before it crosses into C, so the benches
//! and tests themselves contain no `unsafe`.

// The kernels are C, so calling them is `unsafe`. That is all the `unsafe` in this
// package, and every block below carries its safety argument.
#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, CStr};

extern "C" {
    fn mmb_compiler() -> *const c_char;
    fn mmb_matmul_ijk(a: *const f32, b: *const f32, c: *mut f32, n: usize);
    fn mmb_matmul_ikj(a: *const f32, b: *const f32, c: *mut f32, n: usize);
    fn mmb_softmax_rows(x: *const f32, out: *mut f32, rows: usize, cols: usize);
    fn mmb_silu(x: *const f32, out: *mut f32, len: usize);
    fn mmb_broadcast_add(
        matrix: *const f32,
        row: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
    );
    fn mmb_rms_norm(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        rows: usize,
        dim: usize,
        eps: f32,
    );
    fn mmb_adamw_update(
        weights: *mut f32,
        grads: *const f32,
        m: *mut f32,
        v: *mut f32,
        len: usize,
        step: c_int,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    );
}

/// The C compiler that built the kernels, as it reports itself.
pub fn compiler() -> &'static str {
    // SAFETY: `mmb_compiler` takes no arguments and returns a pointer to a static,
    // NUL-terminated string.
    unsafe { CStr::from_ptr(mmb_compiler()) }.to_str().unwrap_or("unknown C compiler")
}

fn product(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("the matrix size overflows usize")
}

/// Square matrix product, textbook i-j-k loop order. `a` and `b` are `n x n`, row-major.
///
/// # Panics
///
/// If either input does not hold `n * n` values.
pub fn matmul_ijk(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let len = product(n, n);
    assert_eq!(a.len(), len, "a must hold n * n values");
    assert_eq!(b.len(), len, "b must hold n * n values");
    let mut c = vec![0.0f32; len];
    // SAFETY: both inputs and the freshly allocated output hold `n * n` values, and
    // the three buffers are separate allocations.
    unsafe { mmb_matmul_ijk(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) };
    c
}

/// Square matrix product with the i-k-j loop order. Bit-identical to [`matmul_ijk`].
///
/// # Panics
///
/// If either input does not hold `n * n` values.
pub fn matmul_ikj(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let len = product(n, n);
    assert_eq!(a.len(), len, "a must hold n * n values");
    assert_eq!(b.len(), len, "b must hold n * n values");
    let mut c = vec![0.0f32; len];
    // SAFETY: as in `matmul_ijk`.
    unsafe { mmb_matmul_ikj(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) };
    c
}

/// Softmax over each row of `data`, which holds whole rows of `cols` values.
///
/// # Panics
///
/// If `cols` is 0 while `data` is not empty, or `data.len()` is not a multiple of `cols`.
pub fn softmax_rows(data: &[f32], cols: usize) -> Vec<f32> {
    let rows = whole_rows(data.len(), cols);
    let mut out = vec![0.0f32; data.len()];
    // SAFETY: the input and the output both hold `rows * cols` values (`rows` is
    // `data.len() / cols`), in separate allocations.
    unsafe { mmb_softmax_rows(data.as_ptr(), out.as_mut_ptr(), rows, cols) };
    out
}

/// `x / (1 + exp(-x))` for every value.
pub fn silu(data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    // SAFETY: input and output both hold `data.len()` values, in separate allocations.
    unsafe { mmb_silu(data.as_ptr(), out.as_mut_ptr(), data.len()) };
    out
}

/// Adds `row` to every row of `matrix`, which holds whole rows of `row.len()` values.
///
/// # Panics
///
/// If `row` is empty while `matrix` is not, or `matrix.len()` is not a multiple of `row.len()`.
pub fn broadcast_add(matrix: &[f32], row: &[f32]) -> Vec<f32> {
    let rows = whole_rows(matrix.len(), row.len());
    let mut out = vec![0.0f32; matrix.len()];
    // SAFETY: `matrix` and `out` hold `rows * row.len()` values and `row` holds
    // `row.len()`, all separate allocations.
    unsafe { mmb_broadcast_add(matrix.as_ptr(), row.as_ptr(), out.as_mut_ptr(), rows, row.len()) };
    out
}

/// RMSNorm over each row of `data`; `weight.len()` is the row length.
///
/// # Panics
///
/// If `weight` is empty while `data` is not, or `data.len()` is not a multiple of `weight.len()`.
pub fn rms_norm(data: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let rows = whole_rows(data.len(), weight.len());
    let mut out = vec![0.0f32; data.len()];
    // SAFETY: `data` and `out` hold `rows * weight.len()` values and `weight` holds
    // `weight.len()`, all separate allocations.
    unsafe {
        mmb_rms_norm(data.as_ptr(), weight.as_ptr(), out.as_mut_ptr(), rows, weight.len(), eps)
    };
    out
}

/// The row count of `len` values laid out in rows of `cols`, checking they divide evenly.
fn whole_rows(len: usize, cols: usize) -> usize {
    if cols == 0 {
        assert_eq!(len, 0, "a row length of 0 needs empty data");
        return 0;
    }
    assert_eq!(len % cols, 0, "the data is not a whole number of rows of {cols}");
    len / cols
}

/// AdamW over one flat parameter, with its two moment buffers and step counter.
/// The update rule and the order of the floating-point operations are those of the
/// `PlainAdamW` loop in `benches/optim_step.rs`.
#[derive(Debug, Clone)]
pub struct CAdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    step: c_int,
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Added to the denominator.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
}

impl CAdamW {
    /// An optimizer for `len` values with zeroed moments and no steps taken.
    pub fn new(len: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        CAdamW {
            m: vec![0.0; len],
            v: vec![0.0; len],
            step: 0,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }

    /// Steps taken so far.
    pub fn steps(&self) -> i32 {
        self.step
    }

    /// One update of `weights` from `grads`, in place.
    ///
    /// # Panics
    ///
    /// If `weights` and `grads` do not both have the length this optimizer was built for.
    pub fn update(&mut self, weights: &mut [f32], grads: &[f32]) {
        assert_eq!(weights.len(), self.m.len(), "weights have the wrong length");
        assert_eq!(grads.len(), self.m.len(), "grads have the wrong length");
        self.step += 1;
        // SAFETY: `weights`, `grads` and both moment buffers hold the same number of
        // values (checked above), in separate allocations (the caller's `&mut` weights
        // cannot alias the shared grads or this optimizer's own buffers).
        unsafe {
            mmb_adamw_update(
                weights.as_mut_ptr(),
                grads.as_ptr(),
                self.m.as_mut_ptr(),
                self.v.as_mut_ptr(),
                weights.len(),
                self.step,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_foundation::Rng;
    use midman_nn::RmsNorm;
    use midman_optim::AdamW;
    use midman_tensor::{Parameter, Tensor};

    fn uniform(shape: &[usize], seed: u64) -> Tensor {
        Tensor::uniform(shape, -1.0, 1.0, &mut Rng::seed_from_u64(seed)).unwrap()
    }

    fn assert_close(want: &[f32], got: &[f32], tol: f32) {
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(got) {
            assert!(
                (w - g).abs() <= tol * (1.0 + w.abs().max(g.abs())),
                "{w} vs {g} (tolerance {tol})"
            );
        }
    }

    #[test]
    fn the_compiler_reports_a_version() {
        assert!(!compiler().is_empty());
    }

    #[test]
    fn matmul_matches_the_tensor_kernel_for_both_loop_orders() {
        for n in [1usize, 2, 7, 32, 33] {
            let (a, b) = (uniform(&[n, n], 1), uniform(&[n, n], 2));
            let want = a.matmul(&b).unwrap();
            assert_close(want.data(), &matmul_ijk(&a.to_vec(), &b.to_vec(), n), 1e-4);
            assert_close(want.data(), &matmul_ikj(&a.to_vec(), &b.to_vec(), n), 1e-4);
        }
    }

    #[test]
    fn the_two_loop_orders_are_bit_identical() {
        let n = 33;
        let (a, b) = (uniform(&[n, n], 3).to_vec(), uniform(&[n, n], 4).to_vec());
        let bits = |v: Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(matmul_ijk(&a, &b, n)), bits(matmul_ikj(&a, &b, n)));
    }

    #[test]
    fn matmul_of_a_known_pair() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let (a, b) = ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(matmul_ijk(&a, &b, 2), [19.0, 22.0, 43.0, 50.0]);
        assert_eq!(matmul_ikj(&a, &b, 2), [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_of_size_zero_is_empty() {
        assert!(matmul_ijk(&[], &[], 0).is_empty());
        assert!(matmul_ikj(&[], &[], 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "a must hold n * n values")]
    fn matmul_rejects_a_short_input_before_calling_c() {
        matmul_ijk(&[1.0; 3], &[1.0; 4], 2);
    }

    #[test]
    fn softmax_matches_the_tensor_kernel_and_rows_sum_to_one() {
        let cols = 13;
        let x = uniform(&[9, cols], 5);
        let got = softmax_rows(&x.to_vec(), cols);
        assert_close(x.softmax(-1).unwrap().data(), &got, 1e-5);
        for row in got.chunks_exact(cols) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_survives_large_values_because_it_subtracts_the_row_maximum() {
        let got = softmax_rows(&[1000.0, 1001.0, 1002.0], 3);
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        assert!(got[2] > got[1] && got[1] > got[0]);
        assert!((got.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_of_no_rows_or_no_columns_is_empty() {
        assert!(softmax_rows(&[], 4).is_empty());
        assert!(softmax_rows(&[], 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "whole number of rows")]
    fn softmax_rejects_a_ragged_last_row() {
        softmax_rows(&[1.0; 5], 2);
    }

    #[test]
    fn silu_matches_the_tensor_kernel_and_known_values() {
        let x = uniform(&[300], 6);
        assert_close(x.silu().data(), &silu(&x.to_vec()), 1e-5);
        let got = silu(&[0.0, 1.0]);
        assert_eq!(got[0], 0.0);
        assert!((got[1] - 0.731_058_6).abs() < 1e-6, "{}", got[1]);
    }

    #[test]
    fn broadcast_add_is_exact() {
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let got = broadcast_add(&matrix, &[100.0, 200.0, 300.0]);
        let want: Vec<f32> = (0..12).map(|i| i as f32 + [100.0, 200.0, 300.0][i % 3]).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn rms_norm_matches_the_layer() {
        let layer = RmsNorm::new("norm", 16, 1e-5).unwrap();
        let x = uniform(&[10, 16], 7);
        let got = rms_norm(&x.to_vec(), &[1.0; 16], 1e-5);
        assert_close(layer.forward(&x).unwrap().data(), &got, 1e-4);
    }

    #[test]
    fn rms_norm_applies_the_weight_and_eps() {
        // One row of [3, 4]: mean square 12.5, so each value is divided by sqrt(12.5 + eps).
        let got = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0);
        let inv = 1.0 / 12.5f32.sqrt();
        assert!((got[0] - 3.0 * inv).abs() < 1e-6 && (got[1] - 8.0 * inv).abs() < 1e-6);
        let with_eps = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 12.5);
        assert!((with_eps[0] - 3.0 / 25.0f32.sqrt()).abs() < 1e-6, "{}", with_eps[0]);
    }

    #[test]
    fn adamw_matches_the_optimizer_over_several_steps() {
        let (lr, b1, b2, eps, wd) = (1e-3f32, 0.9f32, 0.999f32, 1e-8f32, 0.1f32);
        let len = 512;
        let start = uniform(&[2, len / 2], 8);
        let mut param = Parameter::new("w", start.clone());
        let mut reference = AdamW::new(b1, b2, eps, wd).unwrap();
        let mut c = CAdamW::new(len, lr, b1, b2, eps, wd);
        let mut c_weights = start.to_vec();
        for seed in 0..5u64 {
            let grad = uniform(&[2, len / 2], 100 + seed).to_vec();
            reference.step(&mut [&mut param], std::slice::from_ref(&grad), lr).unwrap();
            c.update(&mut c_weights, &grad);
        }
        assert_eq!(c.steps(), 5);
        assert_close(param.data(), &c_weights, 1e-5);
    }

    #[test]
    fn adamw_moves_weights_against_the_gradient() {
        let mut c = CAdamW::new(2, 0.1, 0.9, 0.999, 1e-8, 0.0);
        let mut w = [1.0f32, 1.0];
        c.update(&mut w, &[1.0, -1.0]);
        assert!(w[0] < 1.0 && w[1] > 1.0, "{w:?}");
    }

    #[test]
    #[should_panic(expected = "grads have the wrong length")]
    fn adamw_rejects_a_gradient_of_the_wrong_length() {
        CAdamW::new(4, 0.1, 0.9, 0.999, 1e-8, 0.0).update(&mut [0.0; 4], &[0.0; 3]);
    }
}
