// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "ops_elementwise.rs"
// ============================================================================
//! Elementwise arithmetic and activation functions.
//!
//! The binary operations broadcast their operands with NumPy rules, and their
//! backward passes sum gradients back down to each operand's own shape.

use std::sync::Arc;

use midman_foundation::{Error, Result};

use crate::autograd::make_op;
use crate::shape::{broadcast_shapes, broadcast_strides, for_each_row2, numel, sum_to_shape};
use crate::tensor::Tensor;

/// Constant `sqrt(2 / pi)` used by the tanh approximation of GELU.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// Cubic coefficient of the tanh approximation of GELU.
const GELU_COEFF: f32 = 0.044_715;

fn broadcast_result_shape(op: &str, a: &Tensor, b: &Tensor) -> Result<Vec<usize>> {
    broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
        Error::shape(op, format!("cannot broadcast shapes {:?} and {:?}", a.shape(), b.shape()))
    })
}

// A row of a broadcast result has one of three shapes on the last axis: both operands
// advance by one element (two slices), or one of them stays put (a slice and one value).
// The helpers below take exactly those forms. They are plain loops over zipped slices, which
// the compiler vectorizes in every build profile. The same loops written over iterator
// adapters (`copied`, `repeat`, nested `zip`) were vectorized only under fat LTO, and ran
// at a tenth of the speed without it.

/// `out[j] = f(x[j], y[j])`.
#[inline]
fn row_vv(out: &mut [f32], x: &[f32], y: &[f32], f: &impl Fn(f32, f32) -> f32) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        *o = f(a, b);
    }
}

/// `out[j] = f(x[j], y)`.
#[inline]
fn row_vs(out: &mut [f32], x: &[f32], y: f32, f: &impl Fn(f32, f32) -> f32) {
    for (o, &a) in out.iter_mut().zip(x) {
        *o = f(a, y);
    }
}

/// `out[j] = f(x, y[j])`.
#[inline]
fn row_sv(out: &mut [f32], x: f32, y: &[f32], f: &impl Fn(f32, f32) -> f32) {
    for (o, &b) in out.iter_mut().zip(y) {
        *o = f(x, b);
    }
}

/// Applies `f` to every pair of broadcast elements, in row-major output order.
///
/// The output is produced a row (a run along the last axis) at a time, with one of the
/// row helpers above when each operand either advances by one element or stays put on
/// that axis. Any other stride pattern takes a general strided loop.
fn zip_broadcast(
    a: &Tensor,
    b: &Tensor,
    out_shape: &[usize],
    f: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let (ad, bd) = (a.data(), b.data());
    if a.shape() == b.shape() {
        return ad.iter().zip(bd).map(|(&x, &y)| f(x, y)).collect();
    }
    let sa = broadcast_strides(a.shape(), out_shape);
    let sb = broadcast_strides(b.shape(), out_shape);
    let mut out = vec![0.0f32; numel(out_shape)];
    let inner = out_shape.last().copied().unwrap_or(1);
    let (la, lb) = (sa.last().copied().unwrap_or(0), sb.last().copied().unwrap_or(0));
    for_each_row2(out_shape, &sa, &sb, |i, oa, ob| {
        let row = &mut out[i..i + inner];
        match (la, lb) {
            (1, 1) => row_vv(row, &ad[oa..oa + inner], &bd[ob..ob + inner], &f),
            (1, 0) => row_vs(row, &ad[oa..oa + inner], bd[ob], &f),
            (0, 1) => row_sv(row, ad[oa], &bd[ob..ob + inner], &f),
            _ => {
                for (j, o) in row.iter_mut().enumerate() {
                    *o = f(ad[oa + j * la], bd[ob + j * lb]);
                }
            }
        }
    });
    out
}

/// `d[j] = g[j] * f(x[j], y[j])`.
#[inline]
fn grad_vv(d: &mut [f32], g: &[f32], x: &[f32], y: &[f32], f: &impl Fn(f32, f32) -> f32) {
    for (((d, &gi), &a), &b) in d.iter_mut().zip(g).zip(x).zip(y) {
        *d = gi * f(a, b);
    }
}

/// `d[j] = g[j] * f(x[j], y)`.
#[inline]
fn grad_vs(d: &mut [f32], g: &[f32], x: &[f32], y: f32, f: &impl Fn(f32, f32) -> f32) {
    for ((d, &gi), &a) in d.iter_mut().zip(g).zip(x) {
        *d = gi * f(a, y);
    }
}

/// `d[j] = g[j] * f(x, y[j])`.
#[inline]
fn grad_sv(d: &mut [f32], g: &[f32], x: f32, y: &[f32], f: &impl Fn(f32, f32) -> f32) {
    for ((d, &gi), &b) in d.iter_mut().zip(g).zip(y) {
        *d = gi * f(x, b);
    }
}

/// Gradients of an elementwise binary operation: `fa` and `fb` are the partial
/// derivatives with respect to each operand, evaluated per element pair. The
/// results are summed back to the operand shapes.
fn binary_grads(
    a: &Tensor,
    b: &Tensor,
    out_shape: &[usize],
    g: &[f32],
    need: &[bool],
    fa: impl Fn(f32, f32) -> f32 + Copy,
    fb: impl Fn(f32, f32) -> f32 + Copy,
) -> Vec<Option<Vec<f32>>> {
    let (ad, bd) = (a.data(), b.data());
    if a.shape() == b.shape() {
        let da = need[0].then(|| {
            g.iter().zip(ad.iter().zip(bd)).map(|(&gi, (&x, &y))| gi * fa(x, y)).collect()
        });
        let db = need[1].then(|| {
            g.iter().zip(ad.iter().zip(bd)).map(|(&gi, (&x, &y))| gi * fb(x, y)).collect()
        });
        return vec![da, db];
    }
    let sa = broadcast_strides(a.shape(), out_shape);
    let sb = broadcast_strides(b.shape(), out_shape);
    let mut da = need[0].then(|| vec![0.0f32; g.len()]);
    let mut db = need[1].then(|| vec![0.0f32; g.len()]);
    let inner = out_shape.last().copied().unwrap_or(1);
    let (la, lb) = (sa.last().copied().unwrap_or(0), sb.last().copied().unwrap_or(0));
    for_each_row2(out_shape, &sa, &sb, |i, oa, ob| {
        let mut da_row = da.as_mut().map(|d| &mut d[i..i + inner]);
        let mut db_row = db.as_mut().map(|d| &mut d[i..i + inner]);
        let g_row = &g[i..i + inner];
        match (la, lb) {
            (1, 1) => {
                let (x, y) = (&ad[oa..oa + inner], &bd[ob..ob + inner]);
                if let Some(d) = da_row {
                    grad_vv(d, g_row, x, y, &fa);
                }
                if let Some(d) = db_row {
                    grad_vv(d, g_row, x, y, &fb);
                }
            }
            (1, 0) => {
                let (x, y) = (&ad[oa..oa + inner], bd[ob]);
                if let Some(d) = da_row {
                    grad_vs(d, g_row, x, y, &fa);
                }
                if let Some(d) = db_row {
                    grad_vs(d, g_row, x, y, &fb);
                }
            }
            (0, 1) => {
                let (x, y) = (ad[oa], &bd[ob..ob + inner]);
                if let Some(d) = da_row {
                    grad_sv(d, g_row, x, y, &fa);
                }
                if let Some(d) = db_row {
                    grad_sv(d, g_row, x, y, &fb);
                }
            }
            _ => {
                for (j, &gi) in g_row.iter().enumerate() {
                    let (x, y) = (ad[oa + j * la], bd[ob + j * lb]);
                    if let Some(d) = da_row.as_deref_mut() {
                        d[j] = gi * fa(x, y);
                    }
                    if let Some(d) = db_row.as_deref_mut() {
                        d[j] = gi * fb(x, y);
                    }
                }
            }
        }
    });
    vec![
        da.map(|v| sum_to_shape(&v, out_shape, a.shape())),
        db.map(|v| sum_to_shape(&v, out_shape, b.shape())),
    ]
}

/// The three functions are generic and not function pointers, so each operation gets
/// its own copy of the loops with the arithmetic inlined, which the compiler can vectorize.
fn binary_op<FA, FB>(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32,
    fa: FA,
    fb: FB,
) -> Result<Tensor>
where
    FA: Fn(f32, f32) -> f32 + Copy + Send + Sync + 'static,
    FB: Fn(f32, f32) -> f32 + Copy + Send + Sync + 'static,
{
    let out_shape = broadcast_result_shape(op, a, b)?;
    let out = zip_broadcast(a, b, &out_shape, f);
    let (a2, b2, shape2) = (a.clone(), b.clone(), out_shape.clone());
    Ok(make_op(out_shape, Arc::new(out), &[a, b], move |g, need| {
        binary_grads(&a2, &b2, &shape2, g, need, fa, fb)
    }))
}

fn unary_op<F, D>(x: &Tensor, f: F, df: D) -> Tensor
where
    F: Fn(f32) -> f32,
    D: Fn(f32, f32) -> f32 + Send + Sync + 'static,
{
    let y: Arc<Vec<f32>> = Arc::new(x.data().iter().map(|&v| f(v)).collect());
    let (x2, y2) = (x.clone(), Arc::clone(&y));
    make_op(x.shape().to_vec(), y, &[x], move |g, _| {
        let dx = g
            .iter()
            .zip(x2.data())
            .zip(y2.iter())
            .map(|((&gi, &xi), &yi)| gi * df(xi, yi))
            .collect();
        vec![Some(dx)]
    })
}

fn sigmoid_value(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Tensor {
    /// Elementwise sum with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shapes cannot be broadcast together.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_result_shape("add", self, rhs)?;
        let out = zip_broadcast(self, rhs, &out_shape, |x, y| x + y);
        let (sa, sb, so) = (self.shape().to_vec(), rhs.shape().to_vec(), out_shape.clone());
        Ok(make_op(out_shape, Arc::new(out), &[self, rhs], move |g, need| {
            vec![
                need[0].then(|| sum_to_shape(g, &so, &sa)),
                need[1].then(|| sum_to_shape(g, &so, &sb)),
            ]
        }))
    }

    /// Elementwise difference with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shapes cannot be broadcast together.
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_result_shape("sub", self, rhs)?;
        let out = zip_broadcast(self, rhs, &out_shape, |x, y| x - y);
        let (sa, sb, so) = (self.shape().to_vec(), rhs.shape().to_vec(), out_shape.clone());
        Ok(make_op(out_shape, Arc::new(out), &[self, rhs], move |g, need| {
            vec![
                need[0].then(|| sum_to_shape(g, &so, &sa)),
                need[1].then(|| {
                    let mut v = sum_to_shape(g, &so, &sb);
                    v.iter_mut().for_each(|x| *x = -*x);
                    v
                }),
            ]
        }))
    }

    /// Elementwise product with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shapes cannot be broadcast together.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        binary_op("mul", self, rhs, |x, y| x * y, |_, y| y, |x, _| x)
    }

    /// Elementwise quotient with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shapes cannot be broadcast together.
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        binary_op("div", self, rhs, |x, y| x / y, |_, y| 1.0 / y, |x, y| -x / (y * y))
    }

    /// Multiplies every element by a constant.
    pub fn scale(&self, factor: f32) -> Tensor {
        let out: Vec<f32> = self.data().iter().map(|&v| v * factor).collect();
        make_op(self.shape().to_vec(), Arc::new(out), &[self], move |g, _| {
            vec![Some(g.iter().map(|&gi| gi * factor).collect())]
        })
    }

    /// Adds a constant to every element.
    pub fn add_scalar(&self, value: f32) -> Tensor {
        let out: Vec<f32> = self.data().iter().map(|&v| v + value).collect();
        make_op(self.shape().to_vec(), Arc::new(out), &[self], |g, _| vec![Some(g.to_vec())])
    }

    /// Elementwise negation.
    pub fn neg(&self) -> Tensor {
        unary_op(self, |x| -x, |_, _| -1.0)
    }

    /// Elementwise `e^x`.
    pub fn exp(&self) -> Tensor {
        unary_op(self, f32::exp, |_, y| y)
    }

    /// Elementwise natural logarithm. Non-positive inputs give NaN or infinity.
    pub fn ln(&self) -> Tensor {
        unary_op(self, f32::ln, |x, _| 1.0 / x)
    }

    /// Elementwise square root.
    pub fn sqrt(&self) -> Tensor {
        unary_op(self, f32::sqrt, |_, y| 0.5 / y)
    }

    /// Elementwise power `x^p` for a constant exponent.
    pub fn powf(&self, p: f32) -> Tensor {
        unary_op(self, move |x| x.powf(p), move |x, _| p * x.powf(p - 1.0))
    }

    /// Elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        unary_op(self, f32::tanh, |_, y| 1.0 - y * y)
    }

    /// Elementwise logistic sigmoid.
    pub fn sigmoid(&self) -> Tensor {
        unary_op(self, sigmoid_value, |_, y| y * (1.0 - y))
    }

    /// Elementwise rectified linear unit, `max(0, x)`.
    pub fn relu(&self) -> Tensor {
        unary_op(self, |x| x.max(0.0), |x, _| if x > 0.0 { 1.0 } else { 0.0 })
    }

    /// Elementwise SiLU (swish), `x * sigmoid(x)`. The activation of the gated MLP.
    pub fn silu(&self) -> Tensor {
        unary_op(
            self,
            |x| x * sigmoid_value(x),
            |x, _| {
                let s = sigmoid_value(x);
                s * (1.0 + x * (1.0 - s))
            },
        )
    }

    /// Elementwise GELU using the tanh approximation.
    pub fn gelu(&self) -> Tensor {
        unary_op(
            self,
            |x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)).tanh()),
            |x, _| {
                let t = (SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)).tanh();
                0.5 * (1.0 + t)
                    + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x)
            },
        )
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
    fn gelu_constant_matches_its_definition() {
        assert!((SQRT_2_OVER_PI - (2.0 / std::f32::consts::PI).sqrt()).abs() < 1e-7);
    }

    #[test]
    fn arithmetic_on_equal_shapes() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        assert_eq!(a.add(&b).unwrap().to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(b.sub(&a).unwrap().to_vec(), vec![9.0, 18.0, 27.0, 36.0]);
        assert_eq!(a.mul(&b).unwrap().to_vec(), vec![10.0, 40.0, 90.0, 160.0]);
        assert_eq!(b.div(&a).unwrap().to_vec(), vec![10.0; 4]);
    }

    #[test]
    fn broadcasting_a_row_a_column_and_a_scalar() {
        let m = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row = t(&[10.0, 20.0, 30.0], &[3]);
        let col = t(&[100.0, 200.0], &[2, 1]);
        let s = Tensor::scalar(2.0);
        assert_eq!(m.add(&row).unwrap().to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert_eq!(m.add(&col).unwrap().to_vec(), vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]);
        assert_eq!(m.mul(&s).unwrap().to_vec(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(s.mul(&m).unwrap().shape(), &[2, 3]);
        // both operands stretch: [2,1] with [3] gives [2,3]
        let outer = col.mul(&row).unwrap();
        assert_eq!(outer.shape(), &[2, 3]);
        assert_eq!(outer.to_vec(), vec![1000.0, 2000.0, 3000.0, 2000.0, 4000.0, 6000.0]);
    }

    #[test]
    fn incompatible_shapes_are_an_error() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[1.0, 2.0], &[2]);
        assert!(matches!(a.add(&b), Err(Error::Shape { .. })));
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn broadcast_add_gradient_sums_over_the_stretched_axis() {
        let m = t(&[1.0; 6], &[2, 3]).leaf_requiring_grad();
        let bias = t(&[0.0; 3], &[3]).leaf_requiring_grad();
        let grads = m.add(&bias).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.get(&m).unwrap(), &[1.0; 6]);
        assert_eq!(grads.get(&bias).unwrap(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn mul_gradients_use_the_other_operand() {
        let a = t(&[2.0, 3.0], &[2]).leaf_requiring_grad();
        let b = t(&[5.0, 7.0], &[2]).leaf_requiring_grad();
        let grads = a.mul(&b).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[5.0, 7.0]);
        assert_eq!(grads.get(&b).unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn a_constant_operand_gets_no_gradient() {
        let a = t(&[2.0, 3.0], &[2]).leaf_requiring_grad();
        let c = t(&[5.0, 7.0], &[2]);
        let grads = a.div(&c).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.len(), 1);
        close(grads.get(&a).unwrap(), &[0.2, 1.0 / 7.0]);
    }

    #[test]
    fn scalar_helpers() {
        let a = t(&[1.0, -2.0], &[2]);
        assert_eq!(a.scale(3.0).to_vec(), vec![3.0, -6.0]);
        assert_eq!(a.add_scalar(0.5).to_vec(), vec![1.5, -1.5]);
        assert_eq!(a.neg().to_vec(), vec![-1.0, 2.0]);
    }

    #[test]
    fn activation_values() {
        let x = t(&[-1.0, 0.0, 2.0], &[3]);
        close(&x.exp().to_vec(), &[(-1.0f32).exp(), 1.0, 2.0f32.exp()]);
        close(&x.relu().to_vec(), &[0.0, 0.0, 2.0]);
        close(&x.sigmoid().to_vec(), &[0.268_941_4, 0.5, 0.880_797_1]);
        close(&x.tanh().to_vec(), &[-0.761_594_2, 0.0, 0.964_027_6]);
        close(&x.silu().to_vec(), &[-0.268_941_4, 0.0, 1.761_594_2]);
        close(&x.gelu().to_vec(), &[-0.158_808, 0.0, 1.954_597_7]);
        close(&t(&[4.0, 9.0], &[2]).sqrt().to_vec(), &[2.0, 3.0]);
        close(&t(&[1.0, std::f32::consts::E], &[2]).ln().to_vec(), &[0.0, 1.0]);
        close(&t(&[4.0, 16.0], &[2]).powf(-0.5).to_vec(), &[0.5, 0.25]);
    }

    #[test]
    fn sigmoid_does_not_overflow_for_extreme_inputs() {
        let y = t(&[-200.0, 200.0], &[2]).sigmoid();
        assert!(y.is_finite());
        close(&y.to_vec(), &[0.0, 1.0]);
    }

    /// Values of mixed sign and magnitude, never near zero, so divisions are safe and the
    /// order of float additions shows in the bits.
    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                sign * (0.05 + unit * 4.0) * [1.0, 10.0, 0.1][(state >> 8) as usize % 3]
            })
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Offset into a tensor of `shape` of the element that output index `i` of `out` reads,
    /// worked out from the multi-index with no use of the strided walkers.
    fn offset_in(shape: &[usize], out: &[usize], i: usize) -> usize {
        let mut index = vec![0usize; out.len()];
        let mut rest = i;
        for d in (0..out.len()).rev() {
            index[d] = rest % out[d];
            rest /= out[d];
        }
        let lead = out.len() - shape.len();
        let (mut offset, mut stride) = (0, 1);
        for d in (0..shape.len()).rev() {
            let at = if shape[d] == 1 { 0 } else { index[lead + d] };
            offset += at * stride;
            stride *= shape[d];
        }
        offset
    }

    /// Pairs of operand shapes that broadcast, covering every mix of kept and stretched
    /// last axes (both orders are tried by the tests).
    const PAIRS: &[(&[usize], &[usize])] = &[
        (&[2, 3, 4], &[4]),
        (&[2, 3, 4], &[3, 1]),
        (&[2, 3, 4], &[2, 1, 1]),
        (&[2, 3, 4], &[1, 3, 4]),
        (&[2, 3, 4], &[2, 3, 1]),
        (&[2, 3, 4], &[]),
        (&[3, 1], &[1, 4]),
        (&[1], &[2, 3]),
        (&[5, 1], &[1]),
        (&[1, 5, 1], &[3, 1, 1]),
        (&[2, 1, 4], &[3, 1]),
        (&[70, 9], &[9]),
        (&[70, 9], &[70, 1]),
    ];

    type Op = fn(&Tensor, &Tensor) -> Result<Tensor>;

    /// name, operation, forward, partial with respect to the left, partial with respect to the right
    #[allow(clippy::type_complexity)]
    const OPS: &[(&str, Op, fn(f32, f32) -> f32, fn(f32, f32) -> f32, fn(f32, f32) -> f32)] = &[
        ("add", Tensor::add, |x, y| x + y, |_, _| 1.0, |_, _| 1.0),
        ("sub", Tensor::sub, |x, y| x - y, |_, _| 1.0, |_, _| -1.0),
        ("mul", Tensor::mul, |x, y| x * y, |_, y| y, |x, _| x),
        ("div", Tensor::div, |x, y| x / y, |_, y| 1.0 / y, |x, y| -x / (y * y)),
    ];

    #[test]
    fn broadcast_results_match_an_independent_reference_bit_for_bit() {
        for (n, &(sa, sb)) in PAIRS.iter().enumerate() {
            for (left, right) in [(sa, sb), (sb, sa)] {
                let (a, b) = (
                    t(&noise(numel(left), 3 + n as u64), left),
                    t(&noise(numel(right), 9 + n as u64), right),
                );
                let out_shape = broadcast_shapes(left, right).unwrap();
                for (name, op, forward, _, _) in OPS {
                    let got = op(&a, &b).unwrap();
                    assert_eq!(got.shape(), out_shape.as_slice());
                    let want: Vec<f32> = (0..numel(&out_shape))
                        .map(|i| {
                            forward(
                                a.data()[offset_in(left, &out_shape, i)],
                                b.data()[offset_in(right, &out_shape, i)],
                            )
                        })
                        .collect();
                    assert_eq!(bits(got.data()), bits(&want), "{name} of {left:?} and {right:?}");
                }
            }
        }
    }

    #[test]
    fn broadcast_gradients_match_an_independent_reference_bit_for_bit() {
        for (n, &(sa, sb)) in PAIRS.iter().enumerate() {
            for (left, right) in [(sa, sb), (sb, sa)] {
                let a = t(&noise(numel(left), 13 + n as u64), left).leaf_requiring_grad();
                let b = t(&noise(numel(right), 19 + n as u64), right).leaf_requiring_grad();
                let out_shape = broadcast_shapes(left, right).unwrap();
                // Weighting the output by `w` makes `w` the gradient that reaches the operation.
                let w = t(&noise(numel(&out_shape), 29 + n as u64), &out_shape);
                for (name, op, _, da, db) in OPS {
                    let grads = op(&a, &b).unwrap().mul(&w).unwrap().sum_all().backward().unwrap();
                    let mut want_a = vec![0.0f32; a.numel()];
                    let mut want_b = vec![0.0f32; b.numel()];
                    for i in 0..numel(&out_shape) {
                        let (oa, ob) =
                            (offset_in(left, &out_shape, i), offset_in(right, &out_shape, i));
                        let (x, y) = (a.data()[oa], b.data()[ob]);
                        want_a[oa] += w.data()[i] * da(x, y);
                        want_b[ob] += w.data()[i] * db(x, y);
                    }
                    assert_eq!(
                        bits(grads.get(&a).unwrap()),
                        bits(&want_a),
                        "{name} d/da of {left:?} and {right:?}"
                    );
                    assert_eq!(
                        bits(grads.get(&b).unwrap()),
                        bits(&want_b),
                        "{name} d/db of {left:?} and {right:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn only_the_operand_that_needs_a_gradient_gets_one_through_a_broadcast() {
        let a = t(&noise(12, 5), &[3, 4]).leaf_requiring_grad();
        let b = t(&noise(4, 6), &[4]);
        let grads = a.div(&b).unwrap().sum_all().backward().unwrap();
        assert!(grads.get(&a).is_some());
        assert!(grads.get(&b).is_none());
        let c = t(&noise(4, 7), &[4]).leaf_requiring_grad();
        let d = t(&noise(12, 8), &[3, 4]);
        let grads = c.mul(&d).unwrap().sum_all().backward().unwrap();
        assert!(grads.get(&c).is_some() && grads.get(&d).is_none());
    }
}
