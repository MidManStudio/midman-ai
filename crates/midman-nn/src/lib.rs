// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "lib.rs"
// ============================================================================
//! midman-nn: the layers a decoder-only transformer is built from.
//!
//! Every layer owns its weights as [`midman_tensor::Parameter`]s and builds its
//! forward pass out of `midman-tensor` operations, so [`Tensor::backward`]
//! differentiates it with no extra code. The math of each layer is also exposed
//! as a plain function over tensors ([`rms_norm`], [`apply_rope`],
//! [`causal_attention`], [`attention_forward`], [`swiglu`]) so it can be
//! gradient-checked with respect to its weights.
//!
//! ```
//! use midman_foundation::Rng;
//! use midman_nn::{Linear, Module};
//! use midman_tensor::Tensor;
//!
//! # fn main() -> midman_foundation::Result<()> {
//! let mut rng = Rng::seed_from_u64(7);
//! let layer = Linear::new("proj", 4, 3, 0.02, &mut rng)?;
//! let x = Tensor::ones(&[2, 5, 4])?;
//! let y = layer.forward(&x)?;
//! assert_eq!(y.shape(), &[2, 5, 3]);
//! assert_eq!(layer.parameters().len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! This crate does not know about `ModelConfig`: every layer takes its sizes
//! explicitly, and `midman-model` assembles them. See `docs/midman-nn.md`.
//!
//! [`Tensor::backward`]: midman_tensor::Tensor::backward

mod attention;
mod block;
mod embedding;
mod init;
mod linear;
mod mlp;
mod module;
mod norm;
mod rope;

pub use attention::{
    attention_forward, causal_attention, causal_mask, Attention, AttentionWeights,
};
pub use block::{BlockSpec, DecoderBlock};
pub use embedding::Embedding;
pub use init::Init;
pub use linear::Linear;
pub use mlp::{swiglu, SwiGluMlp};
pub use module::{count_parameters, Module};
pub use norm::{rms_norm, RmsNorm};
pub use rope::{apply_rope, RotaryTables};
