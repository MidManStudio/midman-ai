// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "kernels.rs"
// ============================================================================
//! The matrix-product kernels and the choice between their implementations.
//!
//! The `scalar` module holds the plain loops over row-major slices. They are the reference
//! implementation every other backend must match bit for bit, and they are always
//! compiled and always tested. With the `avx` feature on x86_64, the dispatching
//! functions in this module try the AVX kernels first and fall back to the scalar
//! ones when the CPU lacks AVX, so one binary runs everywhere.

use std::sync::atomic::{AtomicBool, Ordering};

/// When set, the dispatching kernels use the scalar implementation even where AVX
/// is available. It exists so a benchmark can time both in one process.
static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);

/// The implementation the dispatching kernels use right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Plain loops over slices, written so the compiler can vectorize them.
    Scalar,
    /// 256-bit AVX kernels, with the same results bit for bit.
    Avx,
}

/// Returns the backend that matrix products use at this moment.
///
/// It is [`Backend::Avx`] only when the crate was built with the `avx` feature for
/// x86_64, the CPU supports AVX, and [`set_force_scalar`] has not switched it off.
pub fn backend() -> Backend {
    #[cfg(all(feature = "avx", target_arch = "x86_64"))]
    if !FORCE_SCALAR.load(Ordering::Relaxed) && crate::kernels_avx::available() {
        return Backend::Avx;
    }
    Backend::Scalar
}

/// Forces the scalar kernels (`true`) or restores the automatic choice (`false`).
///
/// The switch is process-wide. Results are bit-identical either way, so flipping it
/// while other threads multiply matrices changes how fast they run and nothing else.
pub fn set_force_scalar(force: bool) {
    FORCE_SCALAR.store(force, Ordering::Relaxed);
}

/// `C[m,n] += A[m,k] * B[k,n]`, with `m` implied by the slice lengths.
pub(crate) fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    #[cfg(all(feature = "avx", target_arch = "x86_64"))]
    if !FORCE_SCALAR.load(Ordering::Relaxed) && crate::kernels_avx::gemm_nn(a, b, c, k, n) {
        return;
    }
    scalar::gemm_nn(a, b, c, k, n);
}

/// `C[m,n] += A[m,k] * B[n,k]^T`, with `m` implied by the slice lengths. `C` must
/// start at zero, as it does at every call site.
///
/// `B` is copied once into a `[k, n]` layout and the product is done by
/// [`gemm_nn`], whose inner loop runs along a row and vectorizes. The obvious form,
/// one dot product per element of `C`, is a strictly ordered float sum that the
/// compiler may not vectorize, and it ran at a quarter of the speed. Each `C[i][j]`
/// still adds `A[i][p] * B[j][p]` for `p` in ascending order, starting from zero, so
/// the result is bit-for-bit the dot-product form.
pub(crate) fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    debug_assert!(c.iter().all(|&v| v == 0.0), "gemm_nt needs a zeroed C");
    let mut b_t = vec![0.0f32; k * n];
    for (j, b_row) in b.chunks_exact(k).enumerate() {
        for (p, &v) in b_row.iter().enumerate() {
            b_t[p * n + j] = v;
        }
    }
    gemm_nn(a, &b_t, c, k, n);
}

/// `C[m,n] += A[k,m]^T * B[k,n]`, with `k` implied by the slice lengths.
pub(crate) fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
    #[cfg(all(feature = "avx", target_arch = "x86_64"))]
    if !FORCE_SCALAR.load(Ordering::Relaxed) && crate::kernels_avx::gemm_tn(a, b, c, m, n) {
        return;
    }
    scalar::gemm_tn(a, b, c, m, n);
}

/// The reference kernels: plain loops, no intrinsics, on every target.
pub(crate) mod scalar {
    /// `C[m,n] += A[m,k] * B[k,n]`, with `m` implied by the slice lengths.
    pub(crate) fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
        for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
            for (&a_ip, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
                for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                    *c_ij += a_ip * b_pj;
                }
            }
        }
    }

    /// `C[m,n] += A[k,m]^T * B[k,n]`, with `k` implied by the slice lengths.
    pub(crate) fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
        for (a_row, b_row) in a.chunks_exact(m).zip(b.chunks_exact(n)) {
            for (&a_pi, c_row) in a_row.iter().zip(c.chunks_exact_mut(n)) {
                for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                    *c_ij += a_pi * b_pj;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_foundation::Rng;

    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::seed_from_u64(seed);
        (0..len)
            .map(|_| {
                let scale = [1e-3f32, 1.0, 1e3][rng.below(3)];
                rng.normal() * scale
            })
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn the_dispatching_kernels_give_the_scalar_answer_bit_for_bit() {
        // Whichever backend is active, the result must be the scalar one.
        for (m, k, n) in [(1, 1, 1), (4, 16, 16), (5, 9, 13), (33, 17, 41)] {
            let a = noise(m * k, 31);
            let b = noise(k * n, 32);
            let (mut want, mut got) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            scalar::gemm_nn(&a, &b, &mut want, k, n);
            gemm_nn(&a, &b, &mut got, k, n);
            assert_eq!(bits(&want), bits(&got), "nn {m}x{k}x{n}");

            let at = noise(k * m, 33);
            let (mut want, mut got) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            scalar::gemm_tn(&at, &b, &mut want, m, n);
            gemm_tn(&at, &b, &mut got, m, n);
            assert_eq!(bits(&want), bits(&got), "tn {m}x{k}x{n}");
        }
    }

    #[test]
    fn the_force_scalar_switch_selects_the_scalar_backend() {
        // This is the only test that flips the switch, so it cannot race another
        // assertion about the backend.
        let automatic = backend();
        set_force_scalar(true);
        assert_eq!(backend(), Backend::Scalar);
        set_force_scalar(false);
        assert_eq!(backend(), automatic);
        #[cfg(not(all(feature = "avx", target_arch = "x86_64")))]
        assert_eq!(automatic, Backend::Scalar);
    }
}
