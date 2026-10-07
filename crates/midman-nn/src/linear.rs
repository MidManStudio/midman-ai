// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "linear.rs"
// ============================================================================
//! A bias-free linear layer.

use crate::init::check_std;
use crate::Module;
use midman_foundation::{Result, Rng};
use midman_tensor::{Parameter, Tensor};

/// `y = x * W` with `W` of shape `[in_features, out_features]` and no bias.
#[derive(Debug)]
pub struct Linear {
    weight: Parameter,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Creates a layer whose weight is named `"{name}.weight"` and drawn from a
    /// normal distribution with standard deviation `std`.
    ///
    /// # Errors
    ///
    /// Returns an error if either size is 0 or `std` is not positive and finite.
    pub fn new(
        name: &str,
        in_features: usize,
        out_features: usize,
        std: f32,
        rng: &mut Rng,
    ) -> Result<Linear> {
        check_std(name, std)?;
        let init = Tensor::randn(&[in_features, out_features], std, rng)?;
        Ok(Linear {
            weight: Parameter::new(format!("{name}.weight"), init),
            in_features,
            out_features,
        })
    }

    /// Applies the layer to `x` of shape `[.., in_features]`, giving `[.., out_features]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` has fewer than two axes or its last axis is not
    /// `in_features`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.tensor())
    }

    /// The weight parameter.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// The size of the last axis the layer accepts.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The size of the last axis the layer produces.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
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

    #[test]
    fn forward_applies_the_matrix_at_every_position() {
        let mut rng = Rng::seed_from_u64(1);
        let mut layer = Linear::new("l", 2, 3, 0.1, &mut rng).unwrap();
        layer.parameters_mut()[0].set_data(&[1.0, 0.0, 2.0, 0.0, 1.0, 3.0]).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data(), &[1.0, 2.0, 8.0, 3.0, 4.0, 18.0]);
    }

    #[test]
    fn leading_axes_are_preserved() {
        let mut rng = Rng::seed_from_u64(2);
        let layer = Linear::new("l", 4, 5, 0.1, &mut rng).unwrap();
        let y = layer.forward(&Tensor::ones(&[2, 3, 4]).unwrap()).unwrap();
        assert_eq!(y.shape(), &[2, 3, 5]);
    }

    #[test]
    fn the_weight_is_named_and_sized_from_the_arguments() {
        let mut rng = Rng::seed_from_u64(3);
        let layer = Linear::new("blocks.0.attn.q_proj", 6, 4, 0.02, &mut rng).unwrap();
        assert_eq!(layer.weight().name(), "blocks.0.attn.q_proj.weight");
        assert_eq!(layer.weight().shape(), &[6, 4]);
        assert_eq!((layer.in_features(), layer.out_features()), (6, 4));
    }

    #[test]
    fn the_same_seed_gives_the_same_weight() {
        let a = Linear::new("l", 3, 3, 0.5, &mut Rng::seed_from_u64(9)).unwrap();
        let b = Linear::new("l", 3, 3, 0.5, &mut Rng::seed_from_u64(9)).unwrap();
        let c = Linear::new("l", 3, 3, 0.5, &mut Rng::seed_from_u64(10)).unwrap();
        assert_eq!(a.weight().data(), b.weight().data());
        assert_ne!(a.weight().data(), c.weight().data());
    }

    #[test]
    fn the_weight_has_the_requested_spread() {
        let layer = Linear::new("l", 100, 100, 0.25, &mut Rng::seed_from_u64(4)).unwrap();
        let data = layer.weight().data();
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let var = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var.sqrt() - 0.25).abs() < 0.02, "std {}", var.sqrt());
    }

    #[test]
    fn bad_arguments_are_errors() {
        let mut rng = Rng::seed_from_u64(5);
        assert!(Linear::new("l", 0, 3, 0.1, &mut rng).is_err());
        assert!(Linear::new("l", 3, 0, 0.1, &mut rng).is_err());
        assert!(Linear::new("l", 3, 3, 0.0, &mut rng).is_err());
        let layer = Linear::new("l", 3, 3, 0.1, &mut rng).unwrap();
        assert!(layer.forward(&Tensor::ones(&[2, 4]).unwrap()).is_err());
        assert!(layer.forward(&Tensor::ones(&[3]).unwrap()).is_err());
    }
}
