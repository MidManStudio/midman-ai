// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "block.rs"
// ============================================================================
//! One pre-norm decoder block.

use crate::init::Init;
use crate::{Attention, Module, RmsNorm, RotaryTables, SwiGluMlp};
use midman_foundation::{Result, Rng};
use midman_tensor::{Parameter, Tensor};

/// The sizes and init values of a decoder block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockSpec {
    /// Width of the residual stream.
    pub d_model: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads; `n_heads` must be a multiple of it.
    pub n_kv_heads: usize,
    /// Hidden width of the MLP.
    pub d_ff: usize,
    /// Epsilon inside both RMSNorm layers.
    pub norm_eps: f32,
    /// Weight initialization.
    pub init: Init,
}

/// `h = x + attention(norm(x))`, then `h + mlp(norm(h))`.
#[derive(Debug)]
pub struct DecoderBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: SwiGluMlp,
}

impl DecoderBlock {
    /// Creates a block whose parameters are named under `name`:
    /// `"{name}.attn_norm"`, `"{name}.attn.*"`, `"{name}.mlp_norm"` and
    /// `"{name}.mlp.*"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sizes in `spec` do not fit together.
    pub fn new(name: &str, spec: &BlockSpec, rng: &mut Rng) -> Result<DecoderBlock> {
        Ok(DecoderBlock {
            attn_norm: RmsNorm::new(&format!("{name}.attn_norm"), spec.d_model, spec.norm_eps)?,
            attn: Attention::new(
                &format!("{name}.attn"),
                spec.d_model,
                spec.n_heads,
                spec.n_kv_heads,
                spec.init,
                rng,
            )?,
            mlp_norm: RmsNorm::new(&format!("{name}.mlp_norm"), spec.d_model, spec.norm_eps)?,
            mlp: SwiGluMlp::new(&format!("{name}.mlp"), spec.d_model, spec.d_ff, spec.init, rng)?,
        })
    }

    /// Runs the block on `x` of shape `[batch, time, d_model]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` or `rope` do not fit the block.
    pub fn forward(&self, x: &Tensor, rope: &RotaryTables) -> Result<Tensor> {
        let h = x.add(&self.attn.forward(&self.attn_norm.forward(x)?, rope)?)?;
        h.add(&self.mlp.forward(&self.mlp_norm.forward(&h)?)?)
    }
}

impl Module for DecoderBlock {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut all = self.attn_norm.parameters();
        all.extend(self.attn.parameters());
        all.extend(self.mlp_norm.parameters());
        all.extend(self.mlp.parameters());
        all
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut all = self.attn_norm.parameters_mut();
        all.extend(self.attn.parameters_mut());
        all.extend(self.mlp_norm.parameters_mut());
        all.extend(self.mlp.parameters_mut());
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::count_parameters;

    fn spec() -> BlockSpec {
        BlockSpec {
            d_model: 8,
            n_heads: 2,
            n_kv_heads: 1,
            d_ff: 12,
            norm_eps: 1e-5,
            init: Init::uniform(0.4),
        }
    }

    fn block() -> DecoderBlock {
        DecoderBlock::new("blocks.3", &spec(), &mut Rng::seed_from_u64(1)).unwrap()
    }

    #[test]
    fn parameters_are_listed_in_the_order_they_run() {
        let block = block();
        let params = block.parameters();
        let names: Vec<&str> = params.iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            [
                "blocks.3.attn_norm.weight",
                "blocks.3.attn.q_proj.weight",
                "blocks.3.attn.k_proj.weight",
                "blocks.3.attn.v_proj.weight",
                "blocks.3.attn.o_proj.weight",
                "blocks.3.mlp_norm.weight",
                "blocks.3.mlp.gate_proj.weight",
                "blocks.3.mlp.up_proj.weight",
                "blocks.3.mlp.down_proj.weight",
            ]
        );
    }

    #[test]
    fn the_parameter_count_matches_the_hand_calculation() {
        // norms 2 * 8, attention 8*8 + 2 * (8*4) + 8*8, mlp 3 * 8 * 12.
        assert_eq!(count_parameters(&block()), 16 + (64 + 64 + 64) + 288);
    }

    #[test]
    fn the_output_keeps_the_input_shape() {
        let rope = RotaryTables::new(4, 16, 10_000.0).unwrap();
        let x = Tensor::uniform(&[2, 5, 8], -1.0, 1.0, &mut Rng::seed_from_u64(2)).unwrap();
        let y = block().forward(&x, &rope).unwrap();
        assert_eq!(y.shape(), &[2, 5, 8]);
        assert!(y.is_finite());
    }

    #[test]
    fn zeroed_output_projections_make_the_block_an_identity() {
        // Both branches write into the residual stream through these two weights.
        let rope = RotaryTables::new(4, 16, 10_000.0).unwrap();
        let mut block = block();
        for p in block.parameters_mut() {
            if p.name().ends_with("o_proj.weight") || p.name().ends_with("down_proj.weight") {
                let zeros = vec![0.0; p.numel()];
                p.set_data(&zeros).unwrap();
            }
        }
        let x = Tensor::uniform(&[1, 4, 8], -1.0, 1.0, &mut Rng::seed_from_u64(3)).unwrap();
        assert_eq!(block.forward(&x, &rope).unwrap().data(), x.data());
    }

    #[test]
    fn both_branches_add_to_the_stream_they_see() {
        // The attention branch reads x; the MLP branch reads h = x + attention(x).
        let rope = RotaryTables::new(4, 16, 10_000.0).unwrap();
        let block = block();
        let x = Tensor::uniform(&[1, 4, 8], -1.0, 1.0, &mut Rng::seed_from_u64(5)).unwrap();
        let h = x
            .add(&block.attn.forward(&block.attn_norm.forward(&x).unwrap(), &rope).unwrap())
            .unwrap();
        let expected =
            h.add(&block.mlp.forward(&block.mlp_norm.forward(&h).unwrap()).unwrap()).unwrap();
        assert_eq!(block.forward(&x, &rope).unwrap().data(), expected.data());
        // And both branches really contribute, or the comparison proves nothing.
        assert_ne!(h.data(), x.data());
        assert_ne!(expected.data(), h.data());
    }

    #[test]
    fn a_misfit_spec_is_an_error() {
        let mut bad = spec();
        bad.n_heads = 3;
        assert!(DecoderBlock::new("b", &bad, &mut Rng::seed_from_u64(4)).is_err());
        let mut bad = spec();
        bad.norm_eps = 0.0;
        assert!(DecoderBlock::new("b", &bad, &mut Rng::seed_from_u64(4)).is_err());
    }
}
