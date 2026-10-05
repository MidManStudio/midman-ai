// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "check.rs"
// ============================================================================
//! Numerical gradient checking.
//!
//! [`grad_check`] compares the gradients autograd computes with central finite
//! differences. It is how every operation, and later every layer, is tested.

use midman_foundation::{Result, Rng};

use crate::tensor::Tensor;

/// Default finite-difference step. Large enough to beat `f32` rounding noise.
pub const DEFAULT_EPS: f32 = 1e-2;

/// Outcome of [`grad_check`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradCheckReport {
    /// Largest `|analytic - numeric|` over all checked elements.
    pub max_abs_error: f32,
    /// Largest `|analytic - numeric| / max(1, |analytic|, |numeric|)`.
    pub max_scaled_error: f32,
    /// Index of the input holding the worst element.
    pub worst_input: usize,
    /// Index of the worst element within that input.
    pub worst_element: usize,
    /// The analytic gradient at the worst element.
    pub analytic: f32,
    /// The numeric gradient at the worst element.
    pub numeric: f32,
}

impl GradCheckReport {
    /// Panics with a description of the worst element if the scaled error
    /// exceeds `tolerance`. Intended for use in tests.
    pub fn assert_within(&self, tolerance: f32) {
        assert!(
            self.max_scaled_error <= tolerance,
            "gradient mismatch: scaled error {} exceeds {} at input {} element {} (analytic {}, numeric {})",
            self.max_scaled_error,
            tolerance,
            self.worst_input,
            self.worst_element,
            self.analytic,
            self.numeric
        );
    }
}

/// Checks the gradients of `f` at `inputs` against central finite differences.
///
/// `f` may return a tensor of any shape. It is reduced to a scalar by a fixed
/// random weighting (so every output element matters), and the gradient of that
/// scalar with respect to each input element is compared with
/// `(f(x + eps) - f(x - eps)) / (2 eps)`.
///
/// `f` must be deterministic and smooth near the inputs: avoid points where an
/// activation has a kink, such as ReLU at zero.
///
/// # Errors
///
/// Returns any error `f` returns, and an error if the output of `f` does not
/// depend on the inputs.
pub fn grad_check<F>(f: F, inputs: &[Tensor], eps: f32) -> Result<GradCheckReport>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let mut rng = Rng::seed_from_u64(0xC0FFEE);
    let leaves: Vec<Tensor> = inputs.iter().map(Tensor::leaf_requiring_grad).collect();
    let probe = f(&leaves)?;
    let weights = Tensor::uniform(probe.shape(), 0.5, 1.5, &mut rng)?;
    let grads = probe.mul(&weights)?.sum_all().backward()?;

    let constants: Vec<Tensor> = inputs.iter().map(Tensor::detach).collect();
    let weighted = |args: &[Tensor]| -> Result<f64> {
        Ok(f64::from(f(args)?.mul(&weights)?.sum_all().item()?))
    };

    let mut report = GradCheckReport {
        max_abs_error: 0.0,
        max_scaled_error: 0.0,
        worst_input: 0,
        worst_element: 0,
        analytic: 0.0,
        numeric: 0.0,
    };
    for (i, input) in inputs.iter().enumerate() {
        let analytic =
            grads.get(&leaves[i]).map(<[f32]>::to_vec).unwrap_or_else(|| vec![0.0; input.numel()]);
        for (j, &a) in analytic.iter().enumerate() {
            let mut args = constants.clone();
            let mut values = input.to_vec();
            values[j] += eps;
            args[i] = Tensor::from_vec(values.clone(), input.shape())?;
            let plus = weighted(&args)?;
            values[j] -= 2.0 * eps;
            args[i] = Tensor::from_vec(values, input.shape())?;
            let minus = weighted(&args)?;
            let numeric = ((plus - minus) / (2.0 * f64::from(eps))) as f32;

            let abs = (a - numeric).abs();
            let scaled = abs / 1.0f32.max(a.abs()).max(numeric.abs());
            if scaled > report.max_scaled_error {
                report = GradCheckReport {
                    max_abs_error: report.max_abs_error.max(abs),
                    max_scaled_error: scaled,
                    worst_input: i,
                    worst_element: j,
                    analytic: a,
                    numeric,
                };
            } else {
                report.max_abs_error = report.max_abs_error.max(abs);
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_foundation::Error;

    fn t(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape).unwrap()
    }

    #[test]
    fn a_correct_gradient_passes() {
        let x = t(&[0.5, 1.0, 1.5, 2.0], &[2, 2]);
        let report = grad_check(
            |v| Ok(v[0].mul(&v[0])?.exp().ln().scale(0.5).add_scalar(1.0).sqrt()),
            &[x],
            DEFAULT_EPS,
        )
        .unwrap();
        report.assert_within(1e-2);
        assert!(report.max_abs_error < 1e-2, "{report:?}");
    }

    #[test]
    fn a_wrong_gradient_is_caught() {
        // Build an op whose backward is deliberately wrong (it returns twice the true gradient).
        use crate::autograd::make_op;
        use std::sync::Arc;
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let report = grad_check(
            |v| {
                let x2 = v[0].clone();
                let out: Vec<f32> = x2.data().iter().map(|a| a * a).collect();
                Ok(make_op(vec![3], Arc::new(out), &[&v[0]], move |g, _| {
                    let wrong: Vec<f32> =
                        g.iter().zip(x2.data()).map(|(gi, a)| gi * 4.0 * a).collect();
                    vec![Some(wrong)]
                }))
            },
            &[x],
            DEFAULT_EPS,
        )
        .unwrap();
        assert!(report.max_scaled_error > 0.3, "{report:?}");
        let caught = std::panic::catch_unwind(|| report.assert_within(1e-2));
        assert!(caught.is_err(), "assert_within must panic on a wrong gradient");
    }

    #[test]
    fn an_output_that_ignores_the_inputs_is_an_error() {
        let x = t(&[1.0], &[1]);
        let err = grad_check(|_| Ok(Tensor::scalar(3.0)), &[x], DEFAULT_EPS).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn errors_from_the_function_propagate() {
        let x = t(&[1.0, 2.0], &[2]);
        let r = grad_check(|v| v[0].matmul(&v[0]), &[x], DEFAULT_EPS);
        assert!(r.is_err());
    }
}
