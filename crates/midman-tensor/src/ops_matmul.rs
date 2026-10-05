// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "ops_matmul.rs"
// ============================================================================
//! Matrix multiplication: the three kernels and the differentiable operation.
//!
//! The kernels are plain loops over row-major slices. They are the reference
//! implementation every future backend must match, not an optimized one.

use std::sync::Arc;

use midman_foundation::{Error, Result};

use crate::autograd::make_op;
use crate::shape::numel;
use crate::tensor::Tensor;

/// `C[m,n] += A[m,k] * B[k,n]`, with `m` implied by the slice lengths.
fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (&a_ip, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_ip * b_pj;
            }
        }
    }
}

/// `C[m,n] += A[m,k] * B[n,k]^T`, with `m` implied by the slice lengths.
fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (c_ij, b_row) in c_row.iter_mut().zip(b.chunks_exact(k)) {
            *c_ij += a_row.iter().zip(b_row).map(|(&x, &y)| x * y).sum::<f32>();
        }
    }
}

/// `C[m,n] += A[k,m]^T * B[k,n]`, with `k` implied by the slice lengths.
fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
    for (a_row, b_row) in a.chunks_exact(m).zip(b.chunks_exact(n)) {
        for (&a_pi, c_row) in a_row.iter().zip(c.chunks_exact_mut(n)) {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_pi * b_pj;
            }
        }
    }
}

impl Tensor {
    /// Matrix product over the last two axes.
    ///
    /// Two forms are supported:
    ///
    /// * `[.., m, k] x [k, n]`: the right operand is a single matrix applied to
    ///   every leading position (a linear layer).
    /// * `[b.., m, k] x [b.., k, n]`: both operands have the same rank and the
    ///   same leading axes (batched matrices, for example attention scores).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if an operand has fewer than two axes, the inner
    /// dimensions differ, or the leading axes do not match.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let (ra, rb) = (self.rank(), rhs.rank());
        if ra < 2 || rb < 2 {
            return Err(Error::shape(
                "matmul",
                format!(
                    "both operands need at least 2 axes, got {:?} and {:?}",
                    self.shape(),
                    rhs.shape()
                ),
            ));
        }
        let (m, k) = (self.shape()[ra - 2], self.shape()[ra - 1]);
        let (k2, n) = (rhs.shape()[rb - 2], rhs.shape()[rb - 1]);
        if k != k2 {
            return Err(Error::shape(
                "matmul",
                format!("inner dimensions differ: {:?} x {:?}", self.shape(), rhs.shape()),
            ));
        }

        if rb == 2 {
            let rows = self.numel() / k;
            let mut out = vec![0.0f32; rows * n];
            gemm_nn(self.data(), rhs.data(), &mut out, k, n);
            let mut out_shape = self.shape()[..ra - 1].to_vec();
            out_shape.push(n);
            let (a, b) = (self.clone(), rhs.clone());
            return Ok(make_op(out_shape, Arc::new(out), &[self, rhs], move |g, need| {
                let da = need[0].then(|| {
                    let mut d = vec![0.0f32; rows * k];
                    gemm_nt(g, b.data(), &mut d, n, k);
                    d
                });
                let db = need[1].then(|| {
                    let mut d = vec![0.0f32; k * n];
                    gemm_tn(a.data(), g, &mut d, k, n);
                    d
                });
                vec![da, db]
            }));
        }

        if ra != rb || self.shape()[..ra - 2] != rhs.shape()[..rb - 2] {
            return Err(Error::shape(
                "matmul",
                format!(
                    "batched operands need the same leading axes, got {:?} and {:?}",
                    self.shape(),
                    rhs.shape()
                ),
            ));
        }
        let batch = numel(&self.shape()[..ra - 2]);
        let mut out = vec![0.0f32; batch * m * n];
        for ((a_b, b_b), c_b) in self
            .data()
            .chunks_exact(m * k)
            .zip(rhs.data().chunks_exact(k * n))
            .zip(out.chunks_exact_mut(m * n))
        {
            gemm_nn(a_b, b_b, c_b, k, n);
        }
        let mut out_shape = self.shape()[..ra - 2].to_vec();
        out_shape.extend([m, n]);
        let (a, b) = (self.clone(), rhs.clone());
        Ok(make_op(out_shape, Arc::new(out), &[self, rhs], move |g, need| {
            let da = need[0].then(|| {
                let mut d = vec![0.0f32; batch * m * k];
                for ((g_b, b_b), d_b) in g
                    .chunks_exact(m * n)
                    .zip(b.data().chunks_exact(k * n))
                    .zip(d.chunks_exact_mut(m * k))
                {
                    gemm_nt(g_b, b_b, d_b, n, k);
                }
                d
            });
            let db = need[1].then(|| {
                let mut d = vec![0.0f32; batch * k * n];
                for ((a_b, g_b), d_b) in a
                    .data()
                    .chunks_exact(m * k)
                    .zip(g.chunks_exact(m * n))
                    .zip(d.chunks_exact_mut(k * n))
                {
                    gemm_tn(a_b, g_b, d_b, k, n);
                }
                d
            });
            vec![da, db]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape).unwrap()
    }

    #[test]
    fn kernels_agree_with_a_hand_calculation() {
        // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2)
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [7., 8., 9., 10., 11., 12.];
        let expected = [58., 64., 139., 154.];
        let mut c = [0.0; 4];
        gemm_nn(&a, &b, &mut c, 3, 2);
        assert_eq!(c, expected);
        // nt: B given transposed (2x3)
        let bt = [7., 9., 11., 8., 10., 12.];
        let mut c = [0.0; 4];
        gemm_nt(&a, &bt, &mut c, 3, 2);
        assert_eq!(c, expected);
        // tn: A given transposed (3x2)
        let at = [1., 4., 2., 5., 3., 6.];
        let mut c = [0.0; 4];
        gemm_tn(&at, &b, &mut c, 2, 2);
        assert_eq!(c, expected);
    }

    #[test]
    fn two_dimensional_product() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = t(&[7., 8., 9., 10., 11., 12.], &[3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn a_shared_right_operand_applies_to_every_batch_position() {
        let a = t(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 2, 3]);
        let w = t(&[1., 0., 0., 1., 1., 1.], &[3, 2]);
        let c = a.matmul(&w).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        // row [1,2,3] -> [1+3, 2+3] = [4, 5]; row [10,11,12] -> [22, 23]
        assert_eq!(c.data()[..2], [4., 5.]);
        assert_eq!(c.data()[6..], [22., 23.]);
    }

    #[test]
    fn batched_product_multiplies_matching_matrices() {
        let a = t(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);
        let eye_and_swap = t(&[1., 0., 0., 1., 0., 1., 1., 0.], &[2, 2, 2]);
        let c = a.matmul(&eye_and_swap).unwrap();
        assert_eq!(c.to_vec(), vec![1., 2., 3., 4., 6., 5., 8., 7.]);
    }

    #[test]
    fn four_axis_attention_shaped_product() {
        // [B=1, H=2, T=2, D=3] x [B=1, H=2, D=3, T=2]
        let q = t(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[1, 2, 2, 3]);
        let kt = t(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[1, 2, 3, 2]);
        let s = q.matmul(&kt).unwrap();
        assert_eq!(s.shape(), &[1, 2, 2, 2]);
        // head 0: [[0,1,2],[3,4,5]] x [[0,1],[2,3],[4,5]] = [[10,13],[28,40]]
        assert_eq!(s.data()[..4], [10., 13., 28., 40.]);
    }

    #[test]
    fn shape_errors_are_reported() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        assert!(matches!(a.matmul(&t(&[1., 2., 3., 4.], &[2, 2])), Err(Error::Shape { .. })));
        assert!(a.matmul(&t(&[1., 2., 3.], &[3])).is_err());
        let batched_a = t(&[0.0; 12], &[2, 2, 3]);
        let batched_b = t(&[0.0; 18], &[3, 3, 2]);
        assert!(batched_a.matmul(&batched_b).is_err());
    }

    #[test]
    fn gradients_of_a_known_product() {
        // loss = sum(A @ B); dA = ones @ B^T, dB = A^T @ ones
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]).leaf_requiring_grad();
        let b = t(&[7., 8., 9., 10., 11., 12.], &[3, 2]).leaf_requiring_grad();
        let grads = a.matmul(&b).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[15., 19., 23., 15., 19., 23.]);
        assert_eq!(grads.get(&b).unwrap(), &[5., 5., 7., 7., 9., 9.]);
    }
}
