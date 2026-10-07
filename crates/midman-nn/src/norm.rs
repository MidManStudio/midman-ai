// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "norm.rs"
// ============================================================================
//! Root-mean-square layer normalization.

use crate::Module;
use midman_foundation::{Error, Result};
use midman_tensor::{Parameter, Tensor};

/// `x * (mean(x^2) + eps)^(-1/2) * weight`, with the mean over the last axis.
///
/// `weight` must have shape `[x.shape().last()]`. The function is built from
/// `midman-tensor` primitives, so it is differentiable with respect to both
/// arguments.
///
/// # Errors
///
/// Returns an error if `eps` is not positive and finite, `x` has no axes, or
/// `weight` is not a vector as long as the last axis of `x`.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::invalid(format!(
            "rms_norm: eps must be positive and finite, got {eps}"
        )));
    }
    let Some(&dim) = x.shape().last() else {
        return Err(Error::shape("rms_norm", "the input has no axes"));
    };
    if weight.shape() != &[dim][..] {
        return Err(Error::shape(
            "rms_norm",
            format!("weight must have shape [{dim}], got {:?}", weight.shape()),
        ));
    }
    let mean_square = x.mul(x)?.mean_axis(-1, true)?;
    let inv_rms = mean_square.add_scalar(eps).powf(-0.5);
    x.mul(&inv_rms)?.mul(weight)
}

/// An RMSNorm layer with a learned per-channel weight, initialized to 1.
#[derive(Debug)]
pub struct RmsNorm {
    weight: Parameter,
    eps: f32,
}

impl RmsNorm {
    /// Creates a layer over `dim` channels whose weight is named `"{name}.weight"`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` is 0 or `eps` is not positive and finite.
    pub fn new(name: &str, dim: usize, eps: f32) -> Result<RmsNorm> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Error::invalid(format!(
                "{name}: eps must be positive and finite, got {eps}"
            )));
        }
        let weight = Parameter::new(format!("{name}.weight"), Tensor::ones(&[dim])?);
        Ok(RmsNorm { weight, eps })
    }

    /// Normalizes `x` over its last axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the last axis of `x` is not the layer's width.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_norm(x, &self.weight.tensor(), self.eps)
    }

    /// The weight parameter.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }
}

impl Module for RmsNorm {
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_foundation::Rng;

    #[test]
    fn rows_come_out_with_unit_root_mean_square() {
        let x = Tensor::uniform(&[4, 8], -3.0, 3.0, &mut Rng::seed_from_u64(1)).unwrap();
        let y = RmsNorm::new("n", 8, 1e-6).unwrap().forward(&x).unwrap();
        for row in y.data().chunks_exact(8) {
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / 8.0;
            assert!((ms - 1.0).abs() < 1e-3, "mean square {ms}");
        }
    }

    #[test]
    fn a_known_row() {
        // x = [3, 4]: mean square 12.5, so the output is x / sqrt(12.5).
        let x = Tensor::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();
        let w = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let y = rms_norm(&x, &w, 1e-12).unwrap();
        let r = 12.5f32.sqrt();
        assert!((y.data()[0] - 3.0 / r).abs() < 1e-6);
        assert!((y.data()[1] - 2.0 * 4.0 / r).abs() < 1e-6);
    }

    #[test]
    fn epsilon_keeps_tiny_inputs_from_being_blown_up() {
        // x = [1e-3, 1e-3]: mean square 1e-6. With eps = 1e-6 the divisor is
        // sqrt(2e-6), giving 1e-3 / sqrt(2e-6) = 1 / sqrt(2). Without eps it would be 1.
        let x = Tensor::from_vec(vec![1e-3, 1e-3], &[1, 2]).unwrap();
        let y = rms_norm(&x, &Tensor::ones(&[2]).unwrap(), 1e-6).unwrap();
        let want = 1.0 / 2f32.sqrt();
        assert!(y.data().iter().all(|v| (v - want).abs() < 1e-4), "{:?}", y.data());
    }

    #[test]
    fn the_weight_scales_each_channel() {
        let x = Tensor::ones(&[2, 3]).unwrap();
        let w = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = rms_norm(&x, &w, 1e-9).unwrap();
        for row in y.data().chunks_exact(3) {
            for (got, want) in row.iter().zip([1.0, 2.0, 3.0]) {
                assert!((got - want).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn the_layer_starts_at_weight_one_and_is_named() {
        let layer = RmsNorm::new("blocks.0.attn_norm", 5, 1e-5).unwrap();
        assert_eq!(layer.weight().name(), "blocks.0.attn_norm.weight");
        assert_eq!(layer.weight().data(), &[1.0; 5]);
    }

    #[test]
    fn bad_arguments_are_errors() {
        assert!(RmsNorm::new("n", 0, 1e-5).is_err());
        assert!(RmsNorm::new("n", 4, 0.0).is_err());
        assert!(RmsNorm::new("n", 4, f32::NAN).is_err());
        let x = Tensor::ones(&[2, 4]).unwrap();
        assert!(rms_norm(&x, &Tensor::ones(&[3]).unwrap(), 1e-5).is_err());
        assert!(rms_norm(&x, &Tensor::ones(&[1]).unwrap(), 1e-5).is_err());
        assert!(rms_norm(&x, &Tensor::ones(&[2, 4]).unwrap(), 1e-5).is_err());
        assert!(rms_norm(&Tensor::scalar(1.0), &Tensor::ones(&[1]).unwrap(), 1e-5).is_err());
        assert!(RmsNorm::new("n", 3, 1e-5).unwrap().forward(&x).is_err());
    }
}
