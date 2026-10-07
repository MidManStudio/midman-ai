// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model.md, section "model.rs"
// ============================================================================
//! The decoder-only transformer.

use midman_foundation::{Error, Result, Rng};
use midman_model_config::ModelConfig;
use midman_nn::{
    count_parameters, BlockSpec, DecoderBlock, Embedding, Init, Linear, Module, RmsNorm,
    RotaryTables,
};
use midman_tensor::{Parameter, Tensor};

/// The standard deviation of ordinary weights unless a caller picks another.
pub const DEFAULT_INIT_STD: f32 = 0.02;

/// A decoder-only transformer: token embedding, `n_layers` pre-norm decoder
/// blocks, a final RMSNorm, and an output projection that is either the
/// embedding table transposed (tied) or its own matrix.
#[derive(Debug)]
pub struct MidManModel {
    config: ModelConfig,
    embed: Embedding,
    blocks: Vec<DecoderBlock>,
    final_norm: RmsNorm,
    lm_head: Option<Linear>,
    rope: RotaryTables,
}

impl MidManModel {
    /// Builds a model with [`DEFAULT_INIT_STD`].
    ///
    /// # Errors
    ///
    /// See [`MidManModel::with_init_std`].
    pub fn new(config: &ModelConfig, rng: &mut Rng) -> Result<MidManModel> {
        MidManModel::with_init_std(config, DEFAULT_INIT_STD, rng)
    }

    /// Builds a model, drawing every weight from `rng` in a fixed order so one
    /// seed fixes the whole model. Ordinary weights use `std`; the attention
    /// output and MLP down projections use `std / sqrt(2 * n_layers)`; norm
    /// weights start at 1.
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid, `std` is not positive and
    /// finite, or the built parameters do not add up to
    /// [`ModelConfig::param_count`] (which would be a bug in this crate).
    pub fn with_init_std(config: &ModelConfig, std: f32, rng: &mut Rng) -> Result<MidManModel> {
        config.validate()?;
        let init = Init::scaled(std, config.n_layers)?;
        let embed = Embedding::new("embed_tokens", config.vocab_size, config.d_model, std, rng)?;
        let spec = BlockSpec {
            d_model: config.d_model,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            d_ff: config.d_ff,
            norm_eps: config.norm_eps,
            init,
        };
        let blocks = (0..config.n_layers)
            .map(|i| DecoderBlock::new(&format!("blocks.{i}"), &spec, rng))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = RmsNorm::new("final_norm", config.d_model, config.norm_eps)?;
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(Linear::new("lm_head", config.d_model, config.vocab_size, std, rng)?)
        };
        let rope = RotaryTables::new(config.head_dim(), config.max_seq_len, config.rope_theta)?;

        let model =
            MidManModel { config: config.clone(), embed, blocks, final_norm, lm_head, rope };
        let built = model.num_parameters() as u64;
        let counted = config.param_count()?;
        if built != counted {
            return Err(Error::invalid(format!(
                "the model has {built} parameters but the config counts {counted}"
            )));
        }
        Ok(model)
    }

    /// The configuration the model was built from.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The total number of scalar parameters.
    pub fn num_parameters(&self) -> usize {
        count_parameters(self)
    }

    /// Computes next-token logits of shape `[batch, seq, vocab_size]` for `ids`,
    /// which holds `batch` sequences of `seq` tokens, row-major.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch` or `seq` is 0, `seq` exceeds `max_seq_len`,
    /// `ids.len()` is not `batch * seq`, or an id is not below `vocab_size`.
    pub fn forward(&self, ids: &[usize], batch: usize, seq: usize) -> Result<Tensor> {
        if batch == 0 || seq == 0 {
            return Err(Error::invalid("forward: batch and seq must be at least 1"));
        }
        if seq > self.config.max_seq_len {
            return Err(Error::invalid(format!(
                "forward: sequence length {seq} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }
        if batch.checked_mul(seq) != Some(ids.len()) {
            return Err(Error::invalid(format!(
                "forward: {} ids do not fill {batch} sequences of {seq} tokens",
                ids.len()
            )));
        }
        let mut x = self.embed.forward(ids, &[batch, seq])?;
        for block in &self.blocks {
            x = block.forward(&x, &self.rope)?;
        }
        let x = self.final_norm.forward(&x)?;
        match &self.lm_head {
            Some(head) => head.forward(&x),
            None => x.matmul(&self.embed.weight().tensor().transpose(0, 1)?),
        }
    }

    /// The mean next-token cross-entropy: position `i` of `targets` is what
    /// should follow position `i` of `inputs`. Both are `batch * seq` ids.
    ///
    /// # Errors
    ///
    /// Returns an error under the conditions of [`MidManModel::forward`], or if
    /// `targets` has the wrong length or an id out of range.
    pub fn loss(
        &self,
        inputs: &[usize],
        targets: &[usize],
        batch: usize,
        seq: usize,
    ) -> Result<Tensor> {
        let logits = self.forward(inputs, batch, seq)?;
        logits.reshape(&[batch * seq, self.config.vocab_size])?.cross_entropy(targets)
    }
}

impl Module for MidManModel {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut all = self.embed.parameters();
        for block in &self.blocks {
            all.extend(block.parameters());
        }
        all.extend(self.final_norm.parameters());
        if let Some(head) = &self.lm_head {
            all.extend(head.parameters());
        }
        all
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut all = self.embed.parameters_mut();
        for block in &mut self.blocks {
            all.extend(block.parameters_mut());
        }
        all.extend(self.final_norm.parameters_mut());
        if let Some(head) = &mut self.lm_head {
            all.extend(head.parameters_mut());
        }
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(tie: bool) -> ModelConfig {
        ModelConfig {
            vocab_size: 11,
            d_model: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            d_ff: 12,
            max_seq_len: 6,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            tie_embeddings: tie,
        }
    }

    fn model(tie: bool) -> MidManModel {
        MidManModel::new(&config(tie), &mut Rng::seed_from_u64(1)).unwrap()
    }

    #[test]
    fn logits_have_one_row_per_token_and_one_column_per_word() {
        let logits = model(true).forward(&[0, 1, 2, 3, 4, 5], 2, 3).unwrap();
        assert_eq!(logits.shape(), &[2, 3, 11]);
        assert!(logits.is_finite());
    }

    #[test]
    fn an_untied_model_has_one_extra_matrix_at_the_end() {
        let tied = model(true);
        let untied = model(false);
        let names = |m: &MidManModel| -> Vec<String> {
            m.parameters().iter().map(|p| p.name().to_string()).collect()
        };
        assert_eq!(names(&tied).len() + 1, names(&untied).len());
        assert_eq!(names(&untied).last().unwrap(), "lm_head.weight");
        assert_eq!(untied.num_parameters() - tied.num_parameters(), 11 * 8);
    }

    #[test]
    fn untied_logits_come_from_the_head_not_the_embedding() {
        let mut m = model(false);
        for p in m.parameters_mut() {
            if p.name() == "lm_head.weight" {
                let zeros = vec![0.0; p.numel()];
                p.set_data(&zeros).unwrap();
            }
        }
        let logits = m.forward(&[1, 2, 3], 1, 3).unwrap();
        assert!(logits.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_loss_of_a_fresh_model_is_near_the_uniform_guess() {
        let m = model(true);
        let ids = [0usize, 3, 5, 7, 2, 9];
        let targets = [3usize, 5, 7, 2, 9, 1];
        let loss = m.loss(&ids, &targets, 2, 3).unwrap().item().unwrap();
        let uniform = 11f32.ln();
        assert!((loss - uniform).abs() < 0.15, "loss {loss} vs ln(V) {uniform}");
    }

    #[test]
    fn a_wrong_call_is_an_error() {
        let m = model(true);
        assert!(m.forward(&[0, 1, 2], 0, 3).is_err());
        assert!(m.forward(&[0, 1, 2], 1, 0).is_err());
        assert!(m.forward(&[0, 1, 2, 3], 1, 3).is_err());
        assert!(m.forward(&[0; 7], 1, 7).is_err());
        assert!(m.forward(&[0, 1, 11], 1, 3).is_err());
        assert!(m.loss(&[0, 1, 2], &[1, 2], 1, 3).is_err());
        assert!(m.loss(&[0, 1, 2], &[1, 2, 11], 1, 3).is_err());
    }

    #[test]
    fn an_invalid_config_or_std_is_an_error() {
        let mut bad = config(true);
        bad.n_heads = 3;
        assert!(MidManModel::new(&bad, &mut Rng::seed_from_u64(1)).is_err());
        assert!(MidManModel::with_init_std(&config(true), 0.0, &mut Rng::seed_from_u64(1)).is_err());
    }
}
