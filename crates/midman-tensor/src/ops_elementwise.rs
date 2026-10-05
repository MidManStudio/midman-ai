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
use crate::shape::{broadcast_shapes, broadcast_strides, for_each_offset2, numel, sum_to_shape};
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

/// Applies `f` to every pair of broadcast elements, in row-major output order.
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
    for_each_offset2(out_shape, &sa, &sb, |i, oa, ob| out[i] = f(ad[oa], bd[ob]));
    out
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
    fa: fn(f32, f32) -> f32,
    fb: fn(f32, f32) -> f32,
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
    for_each_offset2(out_shape, &sa, &sb, |i, oa, ob| {
        let (x, y) = (ad[oa], bd[ob]);
        if let Some(da) = da.as_mut() {
            da[i] = g[i] * fa(x, y);
        }
        if let Some(db) = db.as_mut() {
            db[i] = g[i] * fb(x, y);
        }
    });
    vec![
        da.map(|v| sum_to_shape(&v, out_shape, a.shape())),
        db.map(|v| sum_to_shape(&v, out_shape, b.shape())),
    ]
}

fn binary_op(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: fn(f32, f32) -> f32,
    fa: fn(f32, f32) -> f32,
    fb: fn(f32, f32) -> f32,
) -> Result<Tensor> {
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
}
