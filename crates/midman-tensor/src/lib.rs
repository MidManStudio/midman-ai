// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "lib.rs"
// ============================================================================
//! midman-tensor: a from-scratch CPU tensor library with reverse-mode autodiff.
//!
//! Tensors are dense, row-major `f32` arrays. Operations return new tensors, and
//! when an input tracks gradients the result records how to differentiate itself,
//! so [`Tensor::backward`] can compute the gradient of a scalar loss with respect
//! to every [`Parameter`] it depends on.
//!
//! ```
//! use midman_tensor::Tensor;
//!
//! # fn main() -> midman_foundation::Result<()> {
//! // loss = sum(x * x), so d(loss)/dx = 2x
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?.leaf_requiring_grad();
//! let loss = x.mul(&x)?.sum_all();
//! assert_eq!(loss.item()?, 14.0);
//!
//! let grads = loss.backward()?;
//! assert_eq!(grads.get(&x).unwrap(), &[2.0, 4.0, 6.0]);
//! # Ok(())
//! # }
//! ```
//!
//! The operations live in `impl Tensor` blocks spread over several modules:
//! elementwise arithmetic and activations, reductions with softmax and
//! cross-entropy, matrix multiplication, and shape operations. The [`check`]
//! module compares any of them against finite differences. This crate is CPU
//! only; see `docs/midman-tensor.md` for the design and its limits.
//!
//! Matrix products run on plain scalar loops by default. The optional `avx`
//! feature adds AVX kernels for x86_64 that are chosen at run time when the CPU
//! has AVX and give the same results bit for bit; see [`kernels`].

// The `avx` feature compiles intrinsics, which are unsafe to call. This is the one
// place the crate opts out of the workspace's `unsafe_code = "deny"`, and only when
// that feature is on, so a default build still denies it. The unsafe code is all in
// `kernels_avx.rs`; every block there carries a SAFETY comment.
#![cfg_attr(all(feature = "avx", target_arch = "x86_64"), allow(unsafe_code))]

mod autograd;
pub mod check;
pub mod kernels;
#[cfg(all(feature = "avx", target_arch = "x86_64"))]
mod kernels_avx;
mod ops_elementwise;
mod ops_matmul;
mod ops_reduce;
mod ops_shape;
mod param;
pub mod shape;
mod tensor;

pub use param::Parameter;
pub use tensor::{Grads, Tensor};
