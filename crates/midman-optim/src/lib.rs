// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-optim.md, section "lib.rs"
// ============================================================================
//! midman-optim: AdamW, a warmup-and-cosine learning-rate schedule, and global
//! gradient-norm clipping.
//!
//! One training step with these pieces, minimizing `sum((w - 3)^2)`:
//!
//! ```
//! use midman_optim::{clip_global_norm, gather_gradients, AdamW};
//! use midman_tensor::{Parameter, Tensor};
//!
//! # fn main() -> midman_foundation::Result<()> {
//! let mut w = Parameter::new("w", Tensor::zeros(&[2])?);
//! let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.0)?;
//! for _ in 0..200 {
//!     let target = Tensor::full(&[2], 3.0)?;
//!     let diff = w.tensor().sub(&target)?;
//!     let loss = diff.mul(&diff)?.sum_all();
//!     let mut grads = loss.backward()?;
//!
//!     let mut params = [&mut w];
//!     let mut g = gather_gradients(&params, &mut grads)?;
//!     clip_global_norm(&mut g, 1.0)?;
//!     opt.step(&mut params, &g, 0.1)?;
//! }
//! assert!((w.data()[0] - 3.0).abs() < 0.05);
//! # Ok(())
//! # }
//! ```
//!
//! Gradients travel between the steps as plain `Vec<Vec<f32>>`, one vector per
//! parameter in the same order as the parameter list, so clipping needs nothing
//! from the tensor crate. See `docs/midman-optim.md`.

mod adamw;
mod clip;
mod schedule;

pub use adamw::AdamW;
pub use clip::{clip_global_norm, gather_gradients, global_norm};
pub use schedule::WarmupCosine;
