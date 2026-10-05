// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "ops_reduce.rs"
// ============================================================================
//! Reductions along an axis, softmax, and the cross-entropy loss.

use std::sync::Arc;

use midman_foundation::{Error, Result};

use crate::autograd::make_op;
use crate::shape::{axis_split, normalize_axis};
use crate::tensor::Tensor;

impl Tensor {
    /// Sum of all elements, as a rank-0 tensor. Accumulates in `f64`.
    pub fn sum_all(&self) -> Tensor {
        let total = self.data().iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
        let n = self.numel();
        make_op(Vec::new(), Arc::new(vec![total]), &[self], move |g, _| vec![Some(vec![g[0]; n])])
    }

    /// Mean of all elements, as a rank-0 tensor.
    pub fn mean_all(&self) -> Tensor {
        self.sum_all().scale(1.0 / self.numel() as f32)
    }

    /// Sum along one axis. With `keepdim` the axis stays with size 1,
    /// otherwise it is removed. Negative axes count from the end.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the axis is out of range.
    pub fn sum_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.rank())?;
        let (outer, len, inner) = axis_split(self.shape(), axis);
        let mut out = vec![0.0f32; outer * inner];
        for (block, out_row) in
            self.data().chunks_exact(len * inner).zip(out.chunks_exact_mut(inner))
        {
            for row in block.chunks_exact(inner) {
                for (o, &v) in out_row.iter_mut().zip(row) {
                    *o += v;
                }
            }
        }
        let mut out_shape = self.shape().to_vec();
        if keepdim {
            out_shape[axis] = 1;
        } else {
            out_shape.remove(axis);
        }
        Ok(make_op(out_shape, Arc::new(out), &[self], move |g, _| {
            let mut dx = vec![0.0f32; outer * len * inner];
            for (block, g_row) in dx.chunks_exact_mut(len * inner).zip(g.chunks_exact(inner)) {
                for row in block.chunks_exact_mut(inner) {
                    row.copy_from_slice(g_row);
                }
            }
            vec![Some(dx)]
        }))
    }

    /// Mean along one axis. See [`Tensor::sum_axis`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the axis is out of range.
    pub fn mean_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let len = self.shape()[normalize_axis(axis, self.rank())?];
        Ok(self.sum_axis(axis, keepdim)?.scale(1.0 / len as f32))
    }

    /// Softmax along one axis, computed in the numerically stable way.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the axis is out of range.
    pub fn softmax(&self, axis: isize) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.rank())?;
        let (outer, len, inner) = axis_split(self.shape(), axis);
        let x = self.data();
        let mut y = vec![0.0f32; x.len()];
        for o in 0..outer {
            for i in 0..inner {
                let base = o * len * inner + i;
                let mut max = f32::NEG_INFINITY;
                for k in 0..len {
                    max = max.max(x[base + k * inner]);
                }
                let mut sum = 0.0f32;
                for k in 0..len {
                    let e = (x[base + k * inner] - max).exp();
                    y[base + k * inner] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for k in 0..len {
                    y[base + k * inner] *= inv;
                }
            }
        }
        let y = Arc::new(y);
        let y2 = Arc::clone(&y);
        Ok(make_op(self.shape().to_vec(), y, &[self], move |g, _| {
            // dx = y * (g - sum(g * y)) along the axis
            let mut dx = vec![0.0f32; g.len()];
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * len * inner + i;
                    let mut dot = 0.0f32;
                    for k in 0..len {
                        dot += g[base + k * inner] * y2[base + k * inner];
                    }
                    for k in 0..len {
                        let idx = base + k * inner;
                        dx[idx] = y2[idx] * (g[idx] - dot);
                    }
                }
            }
            vec![Some(dx)]
        }))
    }

    /// Log-softmax along one axis, computed in the numerically stable way.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the axis is out of range.
    pub fn log_softmax(&self, axis: isize) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.rank())?;
        let (outer, len, inner) = axis_split(self.shape(), axis);
        let x = self.data();
        let mut y = vec![0.0f32; x.len()];
        for o in 0..outer {
            for i in 0..inner {
                let base = o * len * inner + i;
                let mut max = f32::NEG_INFINITY;
                for k in 0..len {
                    max = max.max(x[base + k * inner]);
                }
                let mut sum = 0.0f32;
                for k in 0..len {
                    sum += (x[base + k * inner] - max).exp();
                }
                let lse = max + sum.ln();
                for k in 0..len {
                    y[base + k * inner] = x[base + k * inner] - lse;
                }
            }
        }
        let y = Arc::new(y);
        let y2 = Arc::clone(&y);
        Ok(make_op(self.shape().to_vec(), y, &[self], move |g, _| {
            // dx = g - exp(y) * sum(g) along the axis
            let mut dx = vec![0.0f32; g.len()];
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * len * inner + i;
                    let mut total = 0.0f32;
                    for k in 0..len {
                        total += g[base + k * inner];
                    }
                    for k in 0..len {
                        let idx = base + k * inner;
                        dx[idx] = g[idx] - y2[idx].exp() * total;
                    }
                }
            }
            vec![Some(dx)]
        }))
    }

    /// Mean cross-entropy between `[N, V]` logits and `N` target class indices.
    ///
    /// Computed directly from the logits with the log-sum-exp trick, so large
    /// logits do not overflow.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the logits are not rank 2 or there is not one
    /// target per row, and [`Error::IndexOutOfRange`] if a target is not below `V`.
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(Error::shape(
                "cross_entropy",
                format!("logits must have shape [N, V], got {:?}", self.shape()),
            ));
        }
        let (n, v) = (self.shape()[0], self.shape()[1]);
        if targets.len() != n {
            return Err(Error::shape(
                "cross_entropy",
                format!("{} logit rows but {} targets", n, targets.len()),
            ));
        }
        if let Some(&bad) = targets.iter().find(|&&t| t >= v) {
            return Err(Error::index("target class", bad, v));
        }
        let mut total = 0.0f64;
        for (row, &t) in self.data().chunks_exact(v).zip(targets) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&z| (z - max).exp()).sum();
            total += f64::from(max + sum.ln() - row[t]);
        }
        let loss = (total / n as f64) as f32;
        let (logits, targets) = (self.clone(), targets.to_vec());
        Ok(make_op(Vec::new(), Arc::new(vec![loss]), &[self], move |g, _| {
            let scale = g[0] / n as f32;
            let mut dx = vec![0.0f32; n * v];
            for ((row, drow), &t) in
                logits.data().chunks_exact(v).zip(dx.chunks_exact_mut(v)).zip(&targets)
            {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|&z| (z - max).exp()).sum();
                for (d, &z) in drow.iter_mut().zip(row) {
                    *d = (z - max).exp() / sum * scale;
                }
                drow[t] -= scale;
            }
            vec![Some(dx)]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape).unwrap()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn whole_tensor_reductions() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(a.sum_all().item().unwrap(), 10.0);
        assert_eq!(a.mean_all().item().unwrap(), 2.5);
        assert_eq!(a.sum_all().rank(), 0);
    }

    #[test]
    fn axis_reductions_and_keepdim() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(a.sum_axis(0, false).unwrap().to_vec(), vec![5.0, 7.0, 9.0]);
        assert_eq!(a.sum_axis(1, false).unwrap().to_vec(), vec![6.0, 15.0]);
        assert_eq!(a.sum_axis(-1, true).unwrap().shape(), &[2, 1]);
        assert_eq!(a.sum_axis(0, true).unwrap().shape(), &[1, 3]);
        assert_eq!(a.mean_axis(1, false).unwrap().to_vec(), vec![2.0, 5.0]);
        assert!(a.sum_axis(2, false).is_err());
        let cube = t(&(0..24).map(|v| v as f32).collect::<Vec<_>>(), &[2, 3, 4]);
        // middle axis: out[o][i] = sum_k cube[o][k][i]
        let mid = cube.sum_axis(1, false).unwrap();
        assert_eq!(mid.shape(), &[2, 4]);
        assert_eq!(mid.to_vec()[0], 0.0 + 4.0 + 8.0);
        assert_eq!(mid.to_vec()[7], 15.0 + 19.0 + 23.0);
    }

    #[test]
    fn softmax_rows_sum_to_one_and_match_known_values() {
        let x = t(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]);
        let y = x.softmax(-1).unwrap();
        close(&y.data()[..3], &[0.090_030_57, 0.244_728_47, 0.665_240_96]);
        close(&y.data()[3..], &[1.0 / 3.0; 3]);
        let sums = y.sum_axis(1, false).unwrap();
        close(&sums.to_vec(), &[1.0, 1.0]);
    }

    #[test]
    fn softmax_over_a_non_last_axis() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = x.softmax(0).unwrap();
        // columns are normalized: (1,3) and (2,4)
        let e = (-2.0f32).exp();
        close(
            y.data(),
            &[
                1.0 / (1.0 + 1.0 / e),
                1.0 / (1.0 + 1.0 / e),
                1.0 - 1.0 / (1.0 + 1.0 / e),
                1.0 - 1.0 / (1.0 + 1.0 / e),
            ],
        );
        close(&y.sum_axis(0, false).unwrap().to_vec(), &[1.0, 1.0]);
    }

    #[test]
    fn softmax_is_stable_for_huge_logits() {
        let y = t(&[1000.0, 1000.0, -1000.0], &[3]).softmax(0).unwrap();
        assert!(y.is_finite());
        close(y.data(), &[0.5, 0.5, 0.0]);
    }

    #[test]
    fn log_softmax_is_the_log_of_softmax() {
        let x = t(&[0.5, -1.0, 2.0, 3.0, 3.0, 3.0], &[2, 3]);
        let a = x.log_softmax(1).unwrap().to_vec();
        let b: Vec<f32> = x.softmax(1).unwrap().data().iter().map(|v| v.ln()).collect();
        close(&a, &b);
        assert!(t(&[1000.0, 0.0], &[2]).log_softmax(0).unwrap().is_finite());
    }

    #[test]
    fn cross_entropy_known_values() {
        // Uniform logits over 2 classes: loss is ln 2 whatever the target is.
        let uniform = t(&[0.0, 0.0], &[1, 2]);
        close(&[uniform.cross_entropy(&[0]).unwrap().item().unwrap()], &[std::f32::consts::LN_2]);
        // A confident correct prediction has a near-zero loss.
        let sure = t(&[20.0, 0.0, 0.0], &[1, 3]);
        assert!(sure.cross_entropy(&[0]).unwrap().item().unwrap() < 1e-6);
        // Mean over rows: ln 2 and 0 average to ln 2 / 2.
        let two = t(&[0.0, 0.0, 50.0, -50.0], &[2, 2]);
        close(
            &[two.cross_entropy(&[1, 0]).unwrap().item().unwrap()],
            &[std::f32::consts::LN_2 / 2.0],
        );
    }

    #[test]
    fn cross_entropy_gradient_is_softmax_minus_onehot() {
        let logits = t(&[0.0, 0.0], &[1, 2]).leaf_requiring_grad();
        let grads = logits.cross_entropy(&[0]).unwrap().backward().unwrap();
        close(grads.get(&logits).unwrap(), &[-0.5, 0.5]);
    }

    #[test]
    fn cross_entropy_validates_its_inputs() {
        let logits = t(&[0.0; 6], &[2, 3]);
        assert!(matches!(logits.cross_entropy(&[0]), Err(Error::Shape { .. })));
        assert!(matches!(logits.cross_entropy(&[0, 3]), Err(Error::IndexOutOfRange { .. })));
        assert!(matches!(t(&[0.0; 3], &[3]).cross_entropy(&[0]), Err(Error::Shape { .. })));
    }
}
