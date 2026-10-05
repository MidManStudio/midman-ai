// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model-config.md, section "config.rs"
// ============================================================================
//! The model architecture configuration and its parameter arithmetic.

use midman_foundation::{Error, Result};

/// Version of the on-disk config format written by [`ModelConfig::to_toml_string`].
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

/// Architecture of a decoder-only transformer.
///
/// The v0 architecture is a Llama-style decoder: pre-normalization with RMSNorm,
/// rotary position embeddings, grouped-query attention, a gated (SwiGLU) MLP, and
/// no bias terms. This struct is the single source of truth for the model's shape;
/// the model crate must build exactly the parameters [`ModelConfig::param_breakdown`]
/// counts.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Width of the residual stream.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of attention (query) heads. `d_model` must be divisible by it.
    pub n_heads: usize,
    /// Number of key/value heads. Equal to `n_heads` for standard multi-head
    /// attention; smaller for grouped-query attention. `n_heads` must be
    /// divisible by it.
    pub n_kv_heads: usize,
    /// Hidden width of the gated MLP.
    pub d_ff: usize,
    /// Longest sequence the model is built to handle.
    pub max_seq_len: usize,
    /// Base frequency of the rotary position embeddings.
    pub rope_theta: f32,
    /// Epsilon added inside RMSNorm for numerical stability.
    pub norm_eps: f32,
    /// Whether the output projection shares the token embedding matrix.
    pub tie_embeddings: bool,
}

/// Parameter counts by component, as produced by [`ModelConfig::param_breakdown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamBreakdown {
    /// Token embedding matrix: `vocab_size * d_model`.
    pub embedding: u64,
    /// Attention projections in all layers (query, key, value, output).
    pub attention: u64,
    /// Gated MLP matrices in all layers (gate, up, down).
    pub mlp: u64,
    /// RMSNorm scales: two per layer plus the final norm.
    pub norms: u64,
    /// Separate output projection, or 0 when embeddings are tied.
    pub lm_head: u64,
}

impl ParamBreakdown {
    /// Total number of parameters.
    pub fn total(&self) -> u64 {
        self.embedding + self.attention + self.mlp + self.norms + self.lm_head
    }
}

/// True when `b` divides `a` exactly. Written without `%` on purpose.
fn divides(a: usize, b: usize) -> bool {
    b != 0 && (a / b) * b == a
}

fn positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(Error::config(field, "must be greater than zero"))
    } else {
        Ok(())
    }
}

fn finite_positive(field: &str, value: f32) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::config(field, format!("must be a finite number greater than zero, got {value}")))
    }
}

fn overflow() -> Error {
    Error::config("model", "parameter count overflows a 64-bit integer")
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}

impl ModelConfig {
    /// Checks every field and the relations between them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the first field that is invalid.
    pub fn validate(&self) -> Result<()> {
        positive("vocab_size", self.vocab_size)?;
        positive("d_model", self.d_model)?;
        positive("n_layers", self.n_layers)?;
        positive("n_heads", self.n_heads)?;
        positive("n_kv_heads", self.n_kv_heads)?;
        positive("d_ff", self.d_ff)?;
        positive("max_seq_len", self.max_seq_len)?;
        finite_positive("rope_theta", self.rope_theta)?;
        finite_positive("norm_eps", self.norm_eps)?;
        if !divides(self.d_model, self.n_heads) {
            return Err(Error::config(
                "n_heads",
                format!("d_model {} is not divisible by n_heads {}", self.d_model, self.n_heads),
            ));
        }
        if !divides(self.n_heads, self.n_kv_heads) {
            return Err(Error::config(
                "n_kv_heads",
                format!(
                    "n_heads {} is not divisible by n_kv_heads {}",
                    self.n_heads, self.n_kv_heads
                ),
            ));
        }
        if self.head_dim() & 1 == 1 {
            return Err(Error::config(
                "d_model",
                format!("head dimension {} must be even for rotary embeddings", self.head_dim()),
            ));
        }
        Ok(())
    }

    /// Width of one attention head: `d_model / n_heads`.
    ///
    /// Only meaningful for a config that passes [`ModelConfig::validate`].
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Total width of the key (or value) projection: `n_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }

    /// Counts parameters by component.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if the config is invalid or the count would
    /// overflow a `u64`.
    pub fn param_breakdown(&self) -> Result<ParamBreakdown> {
        self.validate()?;
        let vocab = self.vocab_size as u64;
        let d = self.d_model as u64;
        let layers = self.n_layers as u64;
        let ff = self.d_ff as u64;
        let kv = self.kv_dim() as u64;

        let embedding = mul(vocab, d)?;
        // Query and output projections are d x d; key and value are d x kv_dim.
        let attention_per_layer = add(mul(2, mul(d, d)?)?, mul(2, mul(d, kv)?)?)?;
        let attention = mul(layers, attention_per_layer)?;
        let mlp = mul(layers, mul(3, mul(d, ff)?)?)?;
        let norms = add(mul(layers, mul(2, d)?)?, d)?;
        let lm_head = if self.tie_embeddings { 0 } else { embedding };
        Ok(ParamBreakdown { embedding, attention, mlp, norms, lm_head })
    }

    /// Total parameter count.
    ///
    /// # Errors
    ///
    /// Same as [`ModelConfig::param_breakdown`].
    pub fn param_count(&self) -> Result<u64> {
        Ok(self.param_breakdown()?.total())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smoke() -> ModelConfig {
        ModelConfig {
            vocab_size: 256,
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            d_ff: 176,
            max_seq_len: 128,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    #[test]
    fn param_count_matches_a_hand_calculation() {
        // embedding 256*64 = 16384
        // per layer: attention 4*64*64 = 16384, mlp 3*64*176 = 33792, norms 2*64 = 128
        // two layers: 2 * (16384 + 33792 + 128) = 100608, plus final norm 64
        let cfg = smoke();
        let b = cfg.param_breakdown().unwrap();
        assert_eq!(b.embedding, 16_384);
        assert_eq!(b.attention, 32_768);
        assert_eq!(b.mlp, 67_584);
        assert_eq!(b.norms, 2 * 2 * 64 + 64);
        assert_eq!(b.lm_head, 0);
        assert_eq!(b.total(), 117_056);
        assert_eq!(cfg.param_count().unwrap(), 117_056);
    }

    #[test]
    fn untied_embeddings_add_an_output_matrix() {
        let mut cfg = smoke();
        cfg.tie_embeddings = false;
        assert_eq!(cfg.param_breakdown().unwrap().lm_head, 16_384);
        assert_eq!(cfg.param_count().unwrap(), 117_056 + 16_384);
    }

    #[test]
    fn grouped_query_attention_shrinks_key_and_value_projections() {
        let mut cfg = smoke();
        cfg.n_kv_heads = 1;
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.kv_dim(), 16);
        // per layer attention: 2*64*64 + 2*64*16 = 10240; two layers = 20480
        assert_eq!(cfg.param_breakdown().unwrap().attention, 20_480);
    }

    #[test]
    fn validation_names_the_offending_field() {
        let field_of = |cfg: ModelConfig| match cfg.validate().unwrap_err() {
            Error::Config { field, .. } => field,
            other => panic!("unexpected error {other:?}"),
        };
        let mut c = smoke();
        c.vocab_size = 0;
        assert_eq!(field_of(c), "vocab_size");
        let mut c = smoke();
        c.d_model = 63;
        assert_eq!(field_of(c), "n_heads");
        let mut c = smoke();
        c.n_kv_heads = 3;
        assert_eq!(field_of(c), "n_kv_heads");
        let mut c = smoke();
        c.d_model = 60; // head_dim 15 is odd
        assert_eq!(field_of(c), "d_model");
        let mut c = smoke();
        c.rope_theta = f32::NAN;
        assert_eq!(field_of(c), "rope_theta");
        let mut c = smoke();
        c.norm_eps = 0.0;
        assert_eq!(field_of(c), "norm_eps");
        let mut c = smoke();
        c.max_seq_len = 0;
        assert_eq!(field_of(c), "max_seq_len");
    }

    #[test]
    fn a_valid_config_validates() {
        assert!(smoke().validate().is_ok());
    }

    #[test]
    fn overflow_is_an_error_not_a_wraparound() {
        let mut cfg = smoke();
        cfg.vocab_size = usize::MAX / 2;
        cfg.d_model = 1 << 40;
        cfg.n_heads = 1 << 20;
        cfg.n_kv_heads = 1 << 20;
        assert!(cfg.param_breakdown().is_err());
    }

    #[test]
    fn divides_handles_zero_and_exact_multiples() {
        assert!(divides(12, 4));
        assert!(!divides(12, 5));
        assert!(!divides(12, 0));
        assert!(divides(0, 3));
    }
}
