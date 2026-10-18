// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "kernels_avx.rs"
// ============================================================================
//! AVX matrix-product kernels, compiled only with the `avx` feature on x86_64.
//!
//! Every function here computes exactly what the scalar loops in
//! [`crate::kernels::scalar`] compute, bit for bit. Each element of `C` starts from
//! its current value and adds `A[i][p] * B[p][j]` for `p` in ascending order, with the
//! product rounded and then the sum rounded. That is why the kernels use a separate
//! multiply and add and never a fused multiply-add: fusing rounds once instead of
//! twice and would change the last bits of every result.
//!
//! The speed comes from keeping a tile of `C` in registers across the whole inner
//! loop instead of reading and writing `C` for every `p`. A full tile is four rows by
//! sixteen columns, eight accumulators of eight lanes each. The columns left over when
//! the width is not a multiple of eight are handled with masked loads and stores, and
//! the rows left over when the height is not a multiple of four use smaller tiles.
//!
//! Benchmark record (rule 7 of `platform-optimization.md`): the numbers, the date and
//! the review date are in `docs/midman-tensor.md`, section "kernels_avx.rs". Review
//! date: 2027-01-17.

use core::arch::x86_64::{
    __m256i, _mm256_add_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_ps,
    _mm256_maskstore_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

/// Rows of `C` held in registers by a full tile.
const MR: usize = 4;

/// Lane masks for the column tail: a window of eight entries starting at `8 - rem`
/// has its first `rem` lanes set.
static LANE_MASK: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

/// True when this CPU, and the operating system, support AVX.
pub(crate) fn available() -> bool {
    std::is_x86_feature_detected!("avx")
}

/// `C[m,n] += A[m,k] * B[k,n]` for row-major slices, with `m` implied by the length of
/// `C`. Returns `false`, having written nothing, when the CPU lacks AVX or the
/// slice lengths do not describe those shapes exactly; the caller then uses the
/// scalar kernel, which has its own behavior for such inputs.
pub(crate) fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) -> bool {
    if k == 0 || n == 0 {
        return false;
    }
    let m = c.len() / n;
    if m * n != c.len() || m.checked_mul(k) != Some(a.len()) || k.checked_mul(n) != Some(b.len()) {
        return false;
    }
    run(a, (k, 1), b, c, (m, k, n))
}

/// `C[m,n] += A[k,m]^T * B[k,n]` for row-major slices, with `k` implied by the length
/// of `A`. Returns `false`, having written nothing, under the same conditions as
/// [`gemm_nn`].
pub(crate) fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) -> bool {
    if m == 0 || n == 0 {
        return false;
    }
    let k = a.len() / m;
    if k * m != a.len() || k.checked_mul(n) != Some(b.len()) || m.checked_mul(n) != Some(c.len()) {
        return false;
    }
    run(a, (1, m), b, c, (m, k, n))
}

/// Checks the CPU and enters the kernel. `a_strides` is `(row stride, column stride)`
/// of `A` as a matrix, so `A(i, p)` is `a[i * rs + p * cs]`. The callers have already
/// checked that `a`, `b` and `c` hold exactly `m * k`, `k * n` and `m * n` elements
/// and that the strides address every element of `A` inside `a`.
fn run(
    a: &[f32],
    a_strides: (usize, usize),
    b: &[f32],
    c: &mut [f32],
    dims: (usize, usize, usize),
) -> bool {
    if !available() {
        return false;
    }
    let (m, k, n) = dims;
    let (rs, cs) = a_strides;
    // SAFETY: AVX was detected just above. The callers established that `a` holds
    // every `a[i * rs + p * cs]` for `i < m` and `p < k` (for `gemm_nn` the strides are
    // `(k, 1)` and `a.len() == m * k`; for `gemm_tn` they are `(1, m)` and
    // `a.len() == k * m`), that `b.len() == k * n` and that `c.len() == m * n`. `c` is a
    // unique borrow, so it does not overlap `a` or `b`.
    unsafe { gemm(a.as_ptr(), rs, cs, b.as_ptr(), c.as_mut_ptr(), m, k, n) };
    true
}

/// Walks `C` four rows at a time, then finishes the remaining one to three rows.
///
/// # Safety
///
/// AVX must be available. `a` must be valid for reads of `a[i * rs + p * cs]` for all
/// `i < m`, `p < k`; `b` for `k * n` reads; `c` for `m * n` reads and writes, not
/// overlapping `a` or `b`.
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm(
    a: *const f32,
    rs: usize,
    cs: usize,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let mut i = 0;
    while i + MR <= m {
        row_block::<MR>(a.add(i * rs), rs, cs, b, c.add(i * n), k, n);
        i += MR;
    }
    let (a, c) = (a.add(i * rs), c.add(i * n));
    match m - i {
        3 => row_block::<3>(a, rs, cs, b, c, k, n),
        2 => row_block::<2>(a, rs, cs, b, c, k, n),
        1 => row_block::<1>(a, rs, cs, b, c, k, n),
        _ => {}
    }
}

/// Computes `R` rows of `C` across the whole width: sixteen columns at a time, then
/// eight, then the masked tail.
///
/// # Safety
///
/// AVX must be available. `a` points at `A(i0, 0)` and `c` at `C(i0, 0)`, with `R`
/// rows of each valid as described for [`gemm`].
#[target_feature(enable = "avx")]
#[inline]
unsafe fn row_block<const R: usize>(
    a: *const f32,
    rs: usize,
    cs: usize,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
) {
    let mut j = 0;
    while j + 16 <= n {
        tile::<R, 2>(a, rs, cs, b.add(j), n, c.add(j), k);
        j += 16;
    }
    if j + 8 <= n {
        tile::<R, 1>(a, rs, cs, b.add(j), n, c.add(j), k);
        j += 8;
    }
    if j < n {
        tile_tail::<R>(a, rs, cs, b.add(j), n, c.add(j), k, n - j);
    }
}

/// `C[0..R, 0..8V] += A[0..R, 0..k] * B[0..k, 0..8V]`, accumulating in registers.
///
/// The loops are plain index loops over constant bounds, not iterator adapters. Written
/// with `zip`, the same code was fast under fat LTO but, with LTO off and 16 codegen
/// units, `Zip::new` was not inlined, the accumulators were forced onto the stack and
/// the kernel ran at 8 GFLOP/s instead of 46 to 59. The loops unroll completely, so the
/// accumulators stay in registers in both profiles.
///
/// # Safety
///
/// AVX must be available. `c` points at the top-left of the tile and every row
/// `r < R` must have `8 * V` readable and writable elements at `c + r * n`; `b` must
/// have `8 * V` readable elements at `b + p * n` for every `p < k`; `a` must be valid
/// for `a[r * rs + p * cs]`.
#[target_feature(enable = "avx")]
#[inline]
#[allow(clippy::needless_range_loop)] // index loops on purpose: see the note above
unsafe fn tile<const R: usize, const V: usize>(
    a: *const f32,
    rs: usize,
    cs: usize,
    b: *const f32,
    n: usize,
    c: *mut f32,
    k: usize,
) {
    let mut acc = [[_mm256_setzero_ps(); V]; R];
    for r in 0..R {
        for v in 0..V {
            acc[r][v] = _mm256_loadu_ps(c.add(r * n + 8 * v));
        }
    }
    for p in 0..k {
        let b_row = b.add(p * n);
        let mut b_vec = [_mm256_setzero_ps(); V];
        for v in 0..V {
            b_vec[v] = _mm256_loadu_ps(b_row.add(8 * v));
        }
        for r in 0..R {
            let a_rp = _mm256_set1_ps(*a.add(r * rs + p * cs));
            for v in 0..V {
                // A separate multiply and add, never a fused one: see the module docs.
                acc[r][v] = _mm256_add_ps(acc[r][v], _mm256_mul_ps(a_rp, b_vec[v]));
            }
        }
    }
    for r in 0..R {
        for v in 0..V {
            _mm256_storeu_ps(c.add(r * n + 8 * v), acc[r][v]);
        }
    }
}

/// The last `rem` columns (1 to 7) of `R` rows, using masked loads and stores so that
/// nothing outside the matrices is read or written.
///
/// # Safety
///
/// AVX must be available and `1 <= rem <= 7`. Each of the `R` rows of `c` has `rem`
/// readable and writable elements at `c + r * n`, and `b` has `rem` readable elements
/// at `b + p * n` for every `p < k`; `a` is valid as for [`tile`].
#[target_feature(enable = "avx")]
#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)] // see `tile`
unsafe fn tile_tail<const R: usize>(
    a: *const f32,
    rs: usize,
    cs: usize,
    b: *const f32,
    n: usize,
    c: *mut f32,
    k: usize,
    rem: usize,
) {
    debug_assert!((1..8).contains(&rem));
    let mask = _mm256_loadu_si256(LANE_MASK.as_ptr().add(8 - rem).cast::<__m256i>());
    let mut acc = [_mm256_setzero_ps(); R];
    for r in 0..R {
        acc[r] = _mm256_maskload_ps(c.add(r * n), mask);
    }
    for p in 0..k {
        let b_vec = _mm256_maskload_ps(b.add(p * n), mask);
        for r in 0..R {
            let a_rp = _mm256_set1_ps(*a.add(r * rs + p * cs));
            acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(a_rp, b_vec));
        }
    }
    for r in 0..R {
        _mm256_maskstore_ps(c.add(r * n), mask, acc[r]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::scalar;
    use midman_foundation::Rng;

    /// Values of mixed magnitude, so a change in the order of a float sum shows up in the
    /// low bits instead of cancelling out.
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

    fn skip_without_avx() -> bool {
        let skip = !available();
        if skip {
            eprintln!("this CPU has no AVX: skipping");
        }
        skip
    }

    #[test]
    fn nn_matches_the_scalar_kernel_bit_for_bit_on_every_remainder() {
        if skip_without_avx() {
            return;
        }
        // Heights 1..=9 cross the four-row tile and every leftover; widths 1..=41 cross
        // the sixteen-column tile, the eight-column tile and every masked tail.
        for m in 1..=9usize {
            for n in 1..=41usize {
                for k in [1usize, 2, 7, 33] {
                    let a = noise(m * k, (m * 1000 + k) as u64);
                    let b = noise(k * n, (n * 1000 + k) as u64 + 7);
                    // Start from a nonzero C: the kernels add to what is there.
                    let start = noise(m * n, (m * n) as u64 + 13);
                    let (mut want, mut got) = (start.clone(), start);
                    scalar::gemm_nn(&a, &b, &mut want, k, n);
                    assert!(gemm_nn(&a, &b, &mut got, k, n), "m={m} n={n} k={k}");
                    assert_eq!(bits(&want), bits(&got), "m={m} n={n} k={k}");
                }
            }
        }
    }

    #[test]
    fn tn_matches_the_scalar_kernel_bit_for_bit_on_every_remainder() {
        if skip_without_avx() {
            return;
        }
        for m in 1..=9usize {
            for n in 1..=41usize {
                for k in [0usize, 1, 2, 7, 33] {
                    let a = noise(k * m, (m * 1000 + k) as u64 + 3);
                    let b = noise(k * n, (n * 1000 + k) as u64 + 5);
                    let start = noise(m * n, (m * n) as u64 + 17);
                    let (mut want, mut got) = (start.clone(), start);
                    scalar::gemm_tn(&a, &b, &mut want, m, n);
                    assert!(gemm_tn(&a, &b, &mut got, m, n), "m={m} n={n} k={k}");
                    assert_eq!(bits(&want), bits(&got), "m={m} n={n} k={k}");
                }
            }
        }
    }

    #[test]
    fn the_shapes_a_model_uses_match_bit_for_bit() {
        if skip_without_avx() {
            return;
        }
        // Linear layers of the Smoke and Tiny presets, attention scores, and a head
        // whose width is not a multiple of eight.
        for (m, k, n) in
            [(128, 64, 176), (128, 176, 64), (32, 128, 352), (32, 16, 128), (7, 64, 259)]
        {
            let a = noise(m * k, 1);
            let b = noise(k * n, 2);
            let (mut want, mut got) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            scalar::gemm_nn(&a, &b, &mut want, k, n);
            assert!(gemm_nn(&a, &b, &mut got, k, n));
            assert_eq!(bits(&want), bits(&got), "nn {m}x{k}x{n}");

            let at = noise(k * m, 3);
            let (mut want, mut got) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            scalar::gemm_tn(&at, &b, &mut want, m, n);
            assert!(gemm_tn(&at, &b, &mut got, m, n));
            assert_eq!(bits(&want), bits(&got), "tn {m}x{k}x{n}");
        }
    }

    #[test]
    fn special_values_follow_the_scalar_rules() {
        if skip_without_avx() {
            return;
        }
        // Signed zeros, infinities and NaN go through the same multiplies and adds.
        let a = [0.0f32, -0.0, f32::INFINITY, 1.0, f32::NAN, -2.0];
        let b = [-0.0f32, 0.0, 3.0, f32::NEG_INFINITY, 1.0, f32::MIN_POSITIVE];
        let (m, k, n) = (2, 3, 2);
        let (mut want, mut got) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        scalar::gemm_nn(&a, &b, &mut want, k, n);
        assert!(gemm_nn(&a, &b, &mut got, k, n));
        for (w, g) in want.iter().zip(&got) {
            assert!(w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()), "{w} vs {g}");
        }
    }

    #[test]
    fn masked_lanes_never_touch_memory_outside_the_matrices() {
        if skip_without_avx() {
            return;
        }
        // C sits between guard values that must survive. A width of 13 uses the
        // sixteen-column path's neighbors: one eight-column tile and a five-lane tail.
        let (m, k, n) = (5usize, 9usize, 13usize);
        let a = noise(m * k, 21);
        let b = noise(k * n, 22);
        let guard = f32::from_bits(0x7fc0_dead);
        let mut padded = vec![guard; 4 + m * n + 4];
        padded[4..4 + m * n].fill(0.0);
        let mut want = vec![0.0f32; m * n];
        scalar::gemm_nn(&a, &b, &mut want, k, n);
        assert!(gemm_nn(&a, &b, &mut padded[4..4 + m * n], k, n));
        assert_eq!(bits(&want), bits(&padded[4..4 + m * n]));
        assert!(padded[..4]
            .iter()
            .chain(&padded[4 + m * n..])
            .all(|v| v.to_bits() == guard.to_bits()));
    }

    #[test]
    fn inconsistent_slice_lengths_are_declined_without_writing() {
        if skip_without_avx() {
            return;
        }
        let a = [1.0f32; 6];
        let b = [1.0f32; 6];
        let mut c = [5.0f32; 4];
        // `A` is 2 x 3 and `B` is 3 x 2, so k = 4 does not describe them.
        assert!(!gemm_nn(&a, &b, &mut c, 4, 2));
        // A `C` whose length is not a whole number of rows.
        assert!(!gemm_nn(&a, &b, &mut c[..3], 3, 2));
        // `A` and `C` fit k = 3, n = 2, but `B` is one element short.
        assert!(!gemm_nn(&a, &b[..5], &mut c, 3, 2));
        // The same for the transposed form: `A` is 3 x 2, `B` is 3 x 2, `C` is 2 x 2.
        assert!(!gemm_tn(&a, &b[..5], &mut c, 2, 2));
        assert!(!gemm_tn(&a, &b, &mut c[..3], 2, 2));
        // Zero-sized dimensions are left to the scalar kernel.
        assert!(!gemm_nn(&a, &b, &mut c, 0, 2));
        assert!(!gemm_tn(&a, &b, &mut c, 0, 2));
        assert_eq!(c, [5.0; 4]);
    }
}
