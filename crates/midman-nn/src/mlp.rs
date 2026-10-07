// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "mlp.rs"
// ============================================================================
//! The gated (SwiGLU) feed-forward block.

use crate::init::Init;
use crate::{Linear, Module};
use midman_foundation::{Result, Rng};
use midman_tensor::{Parameter, Tensor};

/// `(silu(x * w_gate) * (x * w_up)) * w_down`.
///
/// `x` is `[.., d_model]`, `w_gate` and `w_up` are `[d_model, d_ff]`, and
/// `w_down` is `[d_ff, d_model]`.
///
/// # Errors
///
/// Returns an error if the shapes do not fit together.
pub fn swiglu(x: &Tensor, w_gate: &Tensor, w_up: &Tensor, w_down: &Tensor) -> Result<Tensor> {
    let gate = x.matmul(w_gate)?.silu();
    let up = x.matmul(w_up)?;
    gate.mul(&up)?.matmul(w_down)
}

/// The SwiGLU MLP of a transformer block, with no biases.
#[derive(Debug)]
pub struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluMlp {
    /// Creates the layer. Its weights are named `"{name}.gate_proj.weight"`,
    /// `up_proj` and `down_proj`; the down projection uses `init.out_std`.
    ///
    /// # Errors
    ///
    /// Returns an error if a size is 0 or an init value is not positive and finite.
    pub fn new(
        name: &str,
        d_model: usize,
        d_ff: usize,
        init: Init,
        rng: &mut Rng,
    ) -> Result<SwiGluMlp> {
        Ok(SwiGluMlp {
            gate_proj: Linear::new(&format!("{name}.gate_proj"), d_model, d_ff, init.std, rng)?,
            up_proj: Linear::new(&format!("{name}.up_proj"), d_model, d_ff, init.std, rng)?,
            down_proj: Linear::new(&format!("{name}.down_proj"), d_ff, d_model, init.out_std, rng)?,
        })
    }

    /// Applies the block to `x` of shape `[.., d_model]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the last axis of `x` is not `d_model`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        swiglu(
            x,
            &self.gate_proj.weight().tensor(),
            &self.up_proj.weight().tensor(),
            &self.down_proj.weight().tensor(),
        )
    }
}

impl Module for SwiGluMlp {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut all = self.gate_proj.parameters();
        all.extend(self.up_proj.parameters());
        all.extend(self.down_proj.parameters());
        all
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut all = self.gate_proj.parameters_mut();
        all.extend(self.up_proj.parameters_mut());
        all.extend(self.down_proj.parameters_mut());
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_value() {
        // One channel, one hidden unit, input 2: silu(2) * (2 * 2) * 3 with
        // silu(2) = 2 / (1 + e^-2). (Input 1 would not do: silu(1) equals sigmoid(1).)
        let one = |v: f32| Tensor::from_vec(vec![v], &[1, 1]).unwrap();
        let out = swiglu(&one(2.0), &one(1.0), &one(2.0), &one(3.0)).unwrap();
        let silu_two = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((out.item().unwrap() - silu_two * 12.0).abs() < 1e-5);
    }

    #[test]
    fn a_zero_gate_input_gives_zero_output() {
        let x = Tensor::zeros(&[2, 3]).unwrap();
        let w = |r, c| Tensor::ones(&[r, c]).unwrap();
        let out = swiglu(&x, &w(3, 5), &w(3, 5), &w(5, 3)).unwrap();
        assert_eq!(out.data(), &[0.0; 6]);
    }

    #[test]
    fn the_layer_keeps_leading_axes_and_the_model_width() {
        let mlp =
            SwiGluMlp::new("mlp", 4, 6, Init::uniform(0.3), &mut Rng::seed_from_u64(1)).unwrap();
        let out = mlp.forward(&Tensor::ones(&[2, 3, 4]).unwrap()).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
        assert!(mlp.forward(&Tensor::ones(&[2, 5]).unwrap()).is_err());
    }

    #[test]
    fn weights_are_named_and_sized() {
        let mlp =
            SwiGluMlp::new("blocks.1.mlp", 4, 6, Init::uniform(0.3), &mut Rng::seed_from_u64(2))
                .unwrap();
        let params = mlp.parameters();
        let summary: Vec<(&str, &[usize])> = params.iter().map(|p| (p.name(), p.shape())).collect();
        assert_eq!(
            summary,
            [
                ("blocks.1.mlp.gate_proj.weight", &[4, 6][..]),
                ("blocks.1.mlp.up_proj.weight", &[4, 6][..]),
                ("blocks.1.mlp.down_proj.weight", &[6, 4][..]),
            ]
        );
    }

    #[test]
    fn the_down_projection_uses_the_residual_std() {
        let init = Init { std: 1.0, out_std: 0.01 };
        let mlp = SwiGluMlp::new("m", 64, 64, init, &mut Rng::seed_from_u64(3)).unwrap();
        let spread = |p: &Parameter| {
            let d = p.data();
            (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt()
        };
        let params = mlp.parameters();
        assert!((spread(params[0]) - 1.0).abs() < 0.1);
        assert!((spread(params[2]) - 0.01).abs() < 0.002);
    }
}
